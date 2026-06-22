import dash
from dash import dcc, html, Input, Output, dash_table, ALL
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from app import app
from page_supporting_files.analysis_utils import parse_contents
from dash import callback_context as ctx
from io import StringIO
import traceback
from page_supporting_files.analysis_utils import make_overview_figures, normalize, low_irra_power_filter, aggregate_daily, compute_yoy, get_full_code
from page_supporting_files.analysis_utils import compute_lr, compute_hw, compute_arima, compute_csd, compute_pvpro
from page_supporting_files.pvcopilot_filter_functions import identify_outliers_iqr, clear_sky_filter, basic_value_filter
import base64
import os
import time
import threading
import uuid
import copy
import collections

# =============================================================================
# PVPRO BACKGROUND-JOB INFRASTRUCTURE
#
# PVPRO can take 1–3 minutes; running it inside a Dash callback would block the
# entire request and the user would see nothing until it finished.  Instead we
# launch the fit in a background thread and let a dcc.Interval poll a
# job-state store for progress updates.
#
# DEPLOYMENT NOTES
# ----------------
# The job-state store needs to be VISIBLE to the polling callback regardless
# of which Gunicorn worker handles each poll request.  We support two backends:
#
#   1. **In-memory dict** (default) -- works for single-worker setups
#      (`flask run`, `gunicorn --workers 1`, the Dash dev server).
#
#   2. **diskcache** -- a tiny disk-backed key/value store that all workers
#      on the same host share.  Used automatically if the `diskcache`
#      package is importable AND the env var `PVPRO_DISKCACHE_DIR` is set
#      to a writable directory (e.g. `/tmp/pvpro-jobs`).
#
#      Recommended for any multi-worker Gunicorn deployment.  In your
#      requirements.txt add:    diskcache>=5.6
#      In your start command:   gunicorn -w 4 ... env PVPRO_DISKCACHE_DIR=/tmp/pvpro-jobs
#
# Without either fix, the symptom is exactly what you saw on the deployed
# server: the user clicks Run, the progress bar shows 0% forever, and no
# result ever appears -- because the worker handling the poll requests has
# never seen the job_id created by the (different) worker that started the
# thread.
# =============================================================================

_PVPRO_DISKCACHE_DIR = os.environ.get("PVPRO_DISKCACHE_DIR")
_PVPRO_USE_DISKCACHE = False
_PVPRO_JOB_CACHE = None
_PVPRO_DISKCACHE_STATUS = ""      # human-readable reason for what happened

if not _PVPRO_DISKCACHE_DIR:
    _PVPRO_DISKCACHE_STATUS = (
        "PVPRO_DISKCACHE_DIR env var not set -- using in-memory dict "
        "(works only with a single worker process)."
    )
else:
    try:
        import diskcache as _diskcache
    except ImportError as _e:
        _PVPRO_DISKCACHE_STATUS = (
            f"PVPRO_DISKCACHE_DIR is set ({_PVPRO_DISKCACHE_DIR!r}) but "
            f"the `diskcache` package is NOT installed. "
            f"Add `diskcache>=5.6` to requirements.txt and redeploy.  "
            f"(ImportError: {_e})"
        )
    else:
        # Let diskcache manage the directory itself.  It creates the dir
        # (and any missing parents) when first written to.  We do NOT
        # pre-probe with a raw open() -- on container filesystems the
        # parent-of-parent might not exist yet, and diskcache's own
        # initialisation handles that more reliably than os.makedirs +
        # tempfile.  Verify the cache works with a real round-trip
        # through the diskcache API itself.
        try:
            os.makedirs(_PVPRO_DISKCACHE_DIR, exist_ok=True)
            _PVPRO_JOB_CACHE = _diskcache.Cache(_PVPRO_DISKCACHE_DIR)
            # Round-trip a tiny test key to confirm everything actually
            # works (catches permission issues, corrupt SQLite from a
            # previous crash, etc).
            _PVPRO_JOB_CACHE.set("__pvpro_init_probe__", 1, expire=60)
            assert _PVPRO_JOB_CACHE.get("__pvpro_init_probe__") == 1
            _PVPRO_JOB_CACHE.delete("__pvpro_init_probe__")
            _PVPRO_USE_DISKCACHE = True
            _PVPRO_DISKCACHE_STATUS = (
                f"diskcache ENABLED at {_PVPRO_DISKCACHE_DIR}"
            )
        except OSError as _e:
            _PVPRO_DISKCACHE_STATUS = (
                f"PVPRO_DISKCACHE_DIR={_PVPRO_DISKCACHE_DIR!r} is not "
                f"usable from this worker (PID {os.getpid()}).  "
                f"On Heroku containers, /tmp is per-dyno and not shared "
                f"across dynos; if you're running multiple dynos you need "
                f"Redis instead.  "
                f"(OSError: {_e})"
            )
        except Exception as _e:
            _PVPRO_DISKCACHE_STATUS = (
                f"diskcache failed to initialise at "
                f"{_PVPRO_DISKCACHE_DIR!r}: {type(_e).__name__}: {_e}.  "
                f"Try `rm -rf {_PVPRO_DISKCACHE_DIR}` to clear stale "
                f"state, then restart the dyno."
            )

_PVPRO_JOBS = {}              # in-memory fallback
_PVPRO_JOBS_LOCK = threading.Lock()

# -----------------------------------------------------------------------------
# Per-job render lock.  The polling callback can fire concurrently from
# multiple gthread worker threads (each browser-side dcc.Interval tick is a
# separate HTTP request).  When the PVPRO worker has just transitioned to
# phase="done", two polls can race into the done-branch simultaneously:
#   - Thread A starts building the (slow, 200-500ms) final_layout.
#   - Thread B reads phase="done" too, also enters the done-branch, and
#     finishes its rendering FIRST (maybe with a slightly different figure
#     state if the worker is still writing).
# Dash sees overlapping responses for the same Output, and the late arrival
# either clobbers the good UI or gets dropped depending on the property's
# allow_duplicate semantics.  Symptom: progress bar stuck at 99%.
#
# Solution: a single Lock per job_id.  The first thread into the done-branch
# acquires the lock, builds the final layout, marks phase="rendered", and
# returns.  All subsequent threads that try to acquire the lock either:
#   - Block briefly, find phase="rendered", and return no_update (cheap).
#   - Or skip with non-blocking acquire when we're sure the UI is already
#     written (even cheaper).
# -----------------------------------------------------------------------------
_PVPRO_RENDER_LOCKS = {}            # job_id -> threading.Lock
_PVPRO_RENDER_LOCKS_LOCK = threading.Lock()


def _pvpro_get_render_lock(job_id):
    """Return the per-job render lock, creating it on first use."""
    with _PVPRO_RENDER_LOCKS_LOCK:
        lock = _PVPRO_RENDER_LOCKS.get(job_id)
        if lock is None:
            lock = threading.Lock()
            _PVPRO_RENDER_LOCKS[job_id] = lock
            # Tidy up: cap the dict so a long-running server doesn't leak
            # one Lock per ever-created job_id.  Keep only the 32 newest.
            if len(_PVPRO_RENDER_LOCKS) > 32:
                # Drop the oldest entries (dict insertion order in Py3.7+).
                oldest = list(_PVPRO_RENDER_LOCKS.keys())[:-32]
                for k in oldest:
                    _PVPRO_RENDER_LOCKS.pop(k, None)
        return lock


# -----------------------------------------------------------------------------
# Per-worker debug ring buffer.  Captures the most recent N events that
# touched the PVPRO job store.  Surfaced in a collapsible panel next to the
# PVPRO progress UI so users can see -- without ssh'ing into the server --
# which worker received their poll request, whether it found the job_id
# they expected, and the lifecycle of the job.
# -----------------------------------------------------------------------------
_PVPRO_DEBUG_LOG = collections.deque(maxlen=80)
_PVPRO_DEBUG_LOCK = threading.Lock()


def _pvpro_debug(event, **fields):
    """Append a single line to the per-worker debug log."""
    entry = {
        "t":     time.strftime("%H:%M:%S"),
        "pid":   os.getpid(),
        "event": event,
        **fields,
    }
    with _PVPRO_DEBUG_LOCK:
        _PVPRO_DEBUG_LOG.append(entry)


def _pvpro_debug_snapshot():
    """Return a list copy of the current debug log (oldest -> newest)."""
    with _PVPRO_DEBUG_LOCK:
        return list(_PVPRO_DEBUG_LOG)


def _pvpro_make_job():
    job_id = str(uuid.uuid4())
    job = {
        "phase": "starting", "current": 0, "total": 1,
        "message": "Queued…",
        "result": None, "error": None,
        "started_at": time.time(),
    }
    if _PVPRO_USE_DISKCACHE:
        _PVPRO_JOB_CACHE.set(job_id, job, expire=3600)   # auto-expire after 1 h
    else:
        with _PVPRO_JOBS_LOCK:
            _PVPRO_JOBS[job_id] = job
    _pvpro_debug("make_job", job_id=job_id[:8])
    return job_id


def _pvpro_update_job(job_id, **kwargs):
    if _PVPRO_USE_DISKCACHE:
        # diskcache get/set is atomic enough for our purposes; the worker
        # thread is the sole writer for any given job_id.
        cur = _PVPRO_JOB_CACHE.get(job_id)
        if cur is not None:
            cur.update(kwargs)
            _PVPRO_JOB_CACHE.set(job_id, cur, expire=3600)
    else:
        with _PVPRO_JOBS_LOCK:
            if job_id in _PVPRO_JOBS:
                _PVPRO_JOBS[job_id].update(kwargs)
    _pvpro_debug("update_job", job_id=job_id[:8],
                 phase=kwargs.get("phase"),
                 current=kwargs.get("current"),
                 total=kwargs.get("total"))


def _pvpro_read_job(job_id):
    """Return a shallow snapshot — caller mutates without affecting state."""
    if _PVPRO_USE_DISKCACHE:
        job = _PVPRO_JOB_CACHE.get(job_id)
    else:
        with _PVPRO_JOBS_LOCK:
            job = _PVPRO_JOBS.get(job_id)
    found = job is not None
    _pvpro_debug("read_job", job_id=job_id[:8],
                 found=found, phase=(job.get("phase") if found else None))
    return None if not found else dict(job)


def _pvpro_drop_job(job_id):
    """Free memory once a finished job has been rendered into the UI.

    Note: callers should generally PREFER marking the job phase as
    'rendered' and letting the 1-hour diskcache TTL clean it up.  Hard
    delete creates a tiny race window where a late poll from the browser
    (queued before the polling Interval was disabled) sees `found=False`
    and triggers the 'PVPRO progress lost' UI, clobbering the just-rendered
    result.  See _pvpro_poll_callback for the full reasoning.
    """
    if _PVPRO_USE_DISKCACHE:
        try:
            _PVPRO_JOB_CACHE.delete(job_id)
        except Exception:
            pass
    else:
        with _PVPRO_JOBS_LOCK:
            _PVPRO_JOBS.pop(job_id, None)
    _pvpro_debug("drop_job", job_id=job_id[:8])

# =============================================================================
# DESIGN TOKENS — editorial / agent-chat aesthetic
# =============================================================================
INK            = "#0f172a"        # body text (slate-900)
INK_SOFT       = "#475569"        # secondary text (slate-600)
PAPER          = "#f8fafc"        # app background (slate-50, light gray)
PAPER_RAISED   = "#ffffff"        # cards / message bubbles
BORDER         = "#e2e8f0"        # subtle dividers (slate-200)
BORDER_STRONG  = "#cbd5e1"        # slate-300
NAVY           = "#0064AB"        # unified primary color — LBNL-style blue
NAVY_DEEP      = "#004d80"        # darker for hover
NAVY_SOFT      = "#dbeafe"        # tint for subtle highlights
ACCENT         = NAVY             # accent everywhere is navy
ACCENT_SOFT    = NAVY_SOFT
SIDEBAR_BG     = "#ffffff"        # white
# All agent accents collapse to one navy
TEAL           = NAVY
INDIGO         = NAVY
ROSE           = NAVY
SLATE          = NAVY
SUCCESS        = "#65BCF0"
MUTED          = "#94a3b8"        # disabled / pending step

# Color reserved for the "filtered out" data in Step 2 -- pie slice, scatter
# dots, and the "Filtered: N" counter all use this so the user can match the
# number visually to its pie slice at a glance.  Matches the pale blue tone
# we use elsewhere for de-emphasised information.
FILTERED_COLOR = "#A3CEED"

# -----------------------------------------------------------------------------
# Numeric-value tokens.
#
# Throughout the UI we display two flavours of number side-by-side:
#   - "major"  : the headline / featured number the user came to see
#                (e.g. the retained-count, the degradation rate).  Coloured.
#   - "detail" : supporting numbers (totals, durations, elapsed seconds).
#                Plain dark text.
#
# Defining them once at the top makes it trivial to retune the palette
# without hunting through every component.
# -----------------------------------------------------------------------------
VALUE_MAJOR    = NAVY       # blue for the featured number
VALUE_DETAIL   = INK        # near-black for supporting numbers

# Legacy aliases (used inside existing callback bodies — kept identical so
# nothing downstream breaks)
MAJOR_CARD_BACKGROUND = PAPER_RAISED
MAJOR_CARD_FONT_COLOR = INK
BODY_CARD_BACKGROUND  = PAPER_RAISED
CODE_BLOCK_BACKGROUND = "#f1f5f9"

AGENTS = {
    "data":   {"name": "Data Prescreening Agent", "color": NAVY, "glyph": "1", "step": 1},
    "filter": {"name": "Filter Agent",            "color": NAVY, "glyph": "2", "step": 2},
    "calc":   {"name": "Degradation Agent",       "color": NAVY, "glyph": "3", "step": 3},
    "code":   {"name": "Code Agent",              "color": NAVY, "glyph": "4", "step": 4},
}


# =============================================================================
# HELPERS (unchanged behavior)
# =============================================================================
def _df_from_store(value):
    """Robustly reconstruct a DataFrame from a dcc.Store payload."""
    if value is None or value == {} or value == "":
        raise ValueError("No dataframe in store")
    if isinstance(value, dict):
        if {"columns", "index", "data"} <= value.keys():
            return pd.DataFrame(**value)
        return pd.DataFrame(value)
    if isinstance(value, str):
        return pd.read_json(StringIO(value), orient="split")
    return pd.DataFrame(value)


def _no_data_alert(message):
    return html.Div(
        [
            html.Span("⚠", style={"marginRight": "8px", "color": ACCENT, "fontSize": "18px"}),
            html.Span(message, style={"color": INK_SOFT, "fontSize": "15px"}),
        ],
        style={
            "padding": "12px 14px",
            "background": ACCENT_SOFT,
            "border": f"1px solid #bae6fd",
            "borderRadius": "10px",
            "fontFamily": "Arial, sans-serif",
        }
    )


def get_layout():
    return layout


def _pvpro_progress_ui(phase, current, total, message, elapsed_s):
    """Render the PVPRO progress display: tqdm-style bar + status text.

    Renders into `pvpro-progress-output`, which sits OUTSIDE the
    dcc.Loading wrapper around `degradation-output`.  Keeping the two
    separate prevents the Loading spinner from overlaying this UI on
    every Interval tick.

    Bottom-row stats
    ----------------
    Once at least one window has been fit (`current >= 1` and phase is
    "fitting"), we compute two derived numbers from the wall-clock
    elapsed time and the (current, total) counter:

        time_per_window = elapsed_s / current
        eta_seconds     = time_per_window * (total - current)

    These get appended to the existing "Elapsed: Ns" line as
    "  ·  ~Xs/window  ·  ETA: Ms".  They're observed-rate-based, so they
    self-correct if the worker speeds up or slows down (warm-start, for
    example, makes later windows faster -- the ETA shrinks accordingly).
    We don't show them during pre-fitting phases (prepare, p0) because
    the counter is meaningless there.
    """
    pct = 0 if not total else int(round(100 * current / max(total, 1)))
    pct = max(0, min(100, pct))

    PHASE_LABELS = {
        "starting":   "Starting PVPRO",
        "prepare":    "Preparing data",
        "p0":         "Estimating initial parameters",
        "fitting":    "Fitting time-windows",
        "trend":      "Computing degradation trends",
        "finalising": "Packing results",
        "done":       "Finalising",
        "rendered":   "Complete",
    }
    sub_label = PHASE_LABELS.get(phase, phase or "Working")

    bar_inner = html.Div(
        style={
            "height": "100%",
            "width": f"{pct}%",
            "background": ACCENT,
            "borderRadius": "999px",
            "transition": "width 0.25s ease",
        }
    )
    bar = html.Div(
        bar_inner,
        style={
            "height": "8px",
            "width": "100%",
            "background": "#e2e8f0",
            "borderRadius": "999px",
            "overflow": "hidden",
            "margin": "10px 0 6px 0",
        }
    )

    # Build the bottom stats line.  Always shows elapsed; when we have
    # enough info, also shows the observed per-window time and ETA.
    def _fmt_secs(s):
        """Render seconds as either 'Xs' (under a minute) or 'Mm Ss'."""
        s = max(0, int(round(s)))
        if s < 60:
            return f"{s}s"
        m, s_rem = divmod(s, 60)
        return f"{m}m {s_rem:02d}s"

    stats_text = f"Elapsed: {_fmt_secs(elapsed_s)}"
    if phase == "fitting" and current and current >= 1 and total and total > current:
        per_window = elapsed_s / max(current, 1)
        eta_secs   = per_window * (total - current)
        stats_text += (
            f"  ·  ~{per_window:.1f}s/window"
            f"  ·  ETA: {_fmt_secs(eta_secs)}"
        )

    return html.Div([
        # Title line: constant "Running PVPRO" + percent only.
        html.Div([
            html.Span(
                "Running PVPRO",
                style={
                    "fontWeight": "600", "fontSize": "15px",
                    "color": INK, "fontFamily": "Arial, sans-serif",
                }
            ),
            html.Span(
                f"  ·  {pct}%",
                style={
                    "marginLeft": "8px", "fontSize": "13px",
                    "color": ACCENT, "fontWeight": "600",
                    "fontFamily": "Arial, sans-serif",
                }
            ),
        ]),
        bar,
        # Subtitle: the current phase label, with (current/total) appended
        # only during the "fitting" phase where a counter is meaningful.
        html.Div(
            sub_label + (
                f" ({current}/{total})"
                if phase == "fitting" and total and total > 1
                else ""
            ),
            style={
                "fontSize": "13px", "color": INK_SOFT,
                "fontFamily": "Arial, sans-serif",
            }
        ),
        # Bottom stats: elapsed + (during fitting) per-window time + ETA.
        html.Div(
            stats_text,
            style={
                "fontSize": "11px", "color": MUTED, "marginTop": "4px",
                "fontFamily": "Arial, sans-serif",
            }
        ),
    ], style={
        "padding": "16px 18px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "marginTop": "16px",
    })


# =============================================================================
# DEBUG PANEL -- collapsible inline diagnostic next to the PVPRO progress UI.
#
# Shows worker PID, storage backend, diskcache status, active job_id, and the
# most recent N events from this worker's job store.  Default-folded so it
# doesn't clutter the main UI; expand it when something's wrong on a deployed
# server and the user can copy/paste the contents into a support email.
#
# The panel is shown DURING progress AND on the final 'done' UI, so the user
# can always inspect the trace of their last PVPRO run.
# =============================================================================
def _pvpro_debug_panel(current_job_id=None):
    log_lines = []
    for entry in _pvpro_debug_snapshot()[-40:]:
        bits = [f"{entry['t']}", f"pid={entry['pid']}", entry["event"]]
        for k, v in entry.items():
            if k in ("t", "pid", "event") or v is None:
                continue
            bits.append(f"{k}={v}")
        log_lines.append("  ".join(bits))
    log_text = "\n".join(log_lines) if log_lines else "(no events yet)"

    storage_summary = (
        f"diskcache @ {_PVPRO_DISKCACHE_DIR}"
        if _PVPRO_USE_DISKCACHE
        else "in-memory dict (single-worker only)"
    )

    rows = [
        ("Worker PID",       str(os.getpid())),
        ("Storage backend",  storage_summary),
        ("Diskcache status", _PVPRO_DISKCACHE_STATUS or "(unset)"),
        ("Active job_id",    (current_job_id or "—")[:8]),
        ("Events in log",    str(len(_PVPRO_DEBUG_LOG))),
    ]
    info_table = html.Table(
        [html.Tr([
            html.Td(k, style={"color": INK_SOFT, "paddingRight": "12px",
                              "verticalAlign": "top",
                              "fontFamily": "Arial, sans-serif"}),
            html.Td(v, style={"fontFamily": "monospace",
                              "wordBreak": "break-all"}),
         ]) for k, v in rows],
        style={"fontSize": "12px", "lineHeight": "1.5",
               "borderCollapse": "collapse", "marginBottom": "8px"},
    )

    return html.Details(
        [
            html.Summary(
                "Debug · expand for worker state and recent events",
                style={
                    "cursor": "pointer",
                    "fontSize": "11px",
                    "color": MUTED,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "fontWeight": "600",
                    "fontFamily": "Arial, sans-serif",
                    # 20px left padding leaves room for the CSS-generated
                    # disclosure triangle (at left:2 from .pvcopilot-root
                    # summary::before) -- otherwise the text overlaps it.
                    "padding": "8px 0 8px 20px",
                }
            ),
            info_table,
            html.Pre(
                log_text,
                style={
                    "fontSize": "11px",
                    "color": "#334155",
                    "background": "#f1f5f9",
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "6px",
                    "padding": "8px 10px",
                    "maxHeight": "180px",
                    "overflowY": "auto",
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace",
                    "margin": "0",
                },
            ),
        ],
        open=False,
        style={
            "marginTop": "12px",
            "borderTop": f"1px dashed {BORDER}",
            "paddingTop": "6px",
        },
    )


# =============================================================================
# REUSABLE UI PRIMITIVES — chat shell
# =============================================================================
def agent_avatar(agent_key, size=32):
    a = AGENTS[agent_key]
    return html.Div(
        a["glyph"],
        style={
            "width": f"{size}px",
            "height": f"{size}px",
            "borderRadius": "50%",
            "background": a["color"],
            "color": "white",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontSize": f"{size * 0.55}px",
            "fontWeight": "700",
            "fontFamily": "Arial, sans-serif",
            "flexShrink": "0",
            "boxShadow": f"0 2px 8px {a['color']}25",
        }
    )


def agent_message(agent_key, body, intro=None):
    """A chat bubble from one of the agents — contains the step's UI."""
    a = AGENTS[agent_key]
    header = html.Div(
        [
            agent_avatar(agent_key, size=34),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(a["name"], style={
                                "fontWeight": "600",
                                "color": INK,
                                "fontSize": "16px",
                                "fontFamily": "Arial, sans-serif",
                            }),
                            html.Span(
                                f"Step {a['step']} of 4",
                                style={
                                    "marginLeft": "10px",
                                    "fontSize": "13px",
                                    "color": INK_SOFT,
                                    "padding": "2px 8px",
                                    "background": "#e2e8f0",
                                    "borderRadius": "10px",
                                    "fontFamily": "Arial, sans-serif",
                                    "letterSpacing": "0.02em",
                                }
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"}
                    ),
                    html.Div(
                        intro or "",
                        style={
                            "fontSize": "15px",
                            "color": INK_SOFT,
                            "fontFamily": "Arial, sans-serif",
                            "fontStyle": "italic",
                            "marginTop": "2px",
                        }
                    ),
                ],
                style={"marginLeft": "12px"}
            ),
        ],
        style={"display": "flex", "alignItems": "flex-start", "marginBottom": "16px"}
    )

    return html.Div(
        [
            header,
            html.Div(
                body,
                style={
                    "marginLeft": "46px",
                    "padding": "20px 22px",
                    "background": PAPER_RAISED,
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "12px",
                    "boxShadow": "0 1px 2px rgba(0,0,0,0.02)",
                }
            ),
        ],
        className="agent-msg slide-in-up",
        style={"marginBottom": "32px"}
    )


def locked_placeholder(agent_key, name, step_num):
    """A muted preview card shown until the previous step completes."""
    a = AGENTS[agent_key]
    return html.Div(
        [
            # Compact header — small bullet + agent name
            html.Div(
                [
                    html.Div(
                        # Show the step number rather than a lock icon --
                        # the number alone communicates "step X is next"
                        # and is consistent with the active/done states.
                        str(step_num),
                        style={
                            "width": "26px",
                            "height": "26px",
                            "borderRadius": "50%",
                            "background": "#e2e8f0",
                            "color": MUTED,
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "13px",
                            "fontWeight": "600",
                            "fontFamily": "Arial, sans-serif",
                            "flexShrink": "0",
                        }
                    ),
                    html.Div(
                        [
                            html.Span(name, style={
                                "fontWeight": "600",
                                "color": MUTED,
                                "fontSize": "15px",
                                "fontFamily": "Arial, sans-serif",
                            }),
                            html.Span(
                                f"Step {step_num} of 4",
                                style={
                                    "marginLeft": "10px",
                                    "fontSize": "12px",
                                    "color": MUTED,
                                    "padding": "2px 8px",
                                    "background": "#e2e8f0",
                                    "borderRadius": "10px",
                                    "fontFamily": "Arial, sans-serif",
                                    "letterSpacing": "0.02em",
                                }
                            ),
                        ],
                        style={"marginLeft": "12px"}
                    ),
                ],
                style={"display": "flex", "alignItems": "center"}
            ),
            html.Div(
                "Complete the previous step to unlock this agent.",
                style={
                    "marginLeft": "38px",
                    "marginTop": "8px",
                    "fontSize": "14px",
                    "color": MUTED,
                    "fontStyle": "italic",
                    "fontFamily": "Arial, sans-serif",
                }
            ),
        ],
        style={
            "padding": "16px 18px",
            "marginBottom": "16px",
            "background": "rgba(241, 245, 249, 0.5)",
            "border": f"1px dashed {BORDER_STRONG}",
            "borderRadius": "10px",
        }
    )



def section_label(text):
    return html.Div(
        text,
        style={
            "fontSize": "13px",
            "fontWeight": "600",
            "color": INK_SOFT,
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "10px",
        }
    )


def soft_blue_callout(children, icon=None, margin_bottom="14px", margin_top="0"):
    """A pale-blue rounded rectangle for informational notes.

    Visually matches the parent app's "Note: This tool is currently under
    active development" banner.  Use it for short warnings or requirement
    summaries that should be noticed but aren't full alerts.  Pass `icon`
    (a one-character emoji or symbol) to prepend a leading glyph.
    """
    body = []
    if icon:
        body.append(html.Span(icon, style={"marginRight": "8px"}))
    if isinstance(children, list):
        body.extend(children)
    else:
        body.append(children)
    return html.Div(
        body,
        style={
            "fontSize": "14px",
            "color": "#0c4a6e",       # dark blue, readable on pale BG
            "background": "#eff6ff",  # blue-50
            "border": "1px solid #bfdbfe",   # blue-200
            "borderRadius": "10px",
            "padding": "10px 14px",
            "lineHeight": "1.55",
            "fontFamily": "Arial, sans-serif",
            "marginTop": margin_top,
            "marginBottom": margin_bottom,
        }
    )


def _ref_style():
    return {"fontSize": "11px", "color": "#4a6fa5", "marginTop": "6px", "lineHeight": "1.5"}

def _eq_style():
    return {"color": "#475569", "margin": "8px 0", "overflowX": "auto", "fontFamily": "Times New Roman, serif"}

def _exp_link_style():
    return {"color": NAVY, "fontSize": "11px", "textDecoration": "none"}

def _exp_inner_style():
    return {"marginTop": "4px", "paddingLeft": "12px", "fontSize": "13px", "color": INK_SOFT, "lineHeight": "1.55"}

def _exp_summary_style():
    return {"cursor": "pointer", "marginBottom": "2px", "color": INK, "fontSize": "13px"}

def _exp_outer_style():
    return {
        "marginTop": "12px",
        "padding": "12px 14px",
        "border": "1px solid #bfdbfe",       # sky-200
        "borderRadius": "10px",
        "backgroundColor": "#eff6ff",        # sky-50
        "fontSize": "13px",
        "lineHeight": "1.6",
    }


def filter_explanations_block():
    """Collapsible 'Filter detail' panel with descriptions, equations, refs.
    Ported from the original PV-Copilot reference content."""
    return html.Details([
        html.Summary("Filter details (equations & references)", style={
            "cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "fontWeight": "600",
            "fontFamily": "Arial, sans-serif",
        }),
        html.Div([

            html.Details([
                html.Summary(html.B("Basic value filter (always applied)"), style=_exp_summary_style()),
                html.Div(
                    "Applied automatically before all other filters. Removes physically implausible "
                    "sensor readings: irradiance outside [0, 1500] W/m², module temperature outside "
                    "[−40, 100] °C, and DC power below −1 W. Catches sensor faults (e.g. irradiance = "
                    "34,000 W/m²) that would corrupt normalization and clear-sky scoring.",
                    style=_exp_inner_style()
                ),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("Time zone & DST correction"), style=_exp_summary_style()),
                html.Div(
                    "Corrects timestamps for local time-zone offsets and Daylight Saving Time (DST) "
                    "transitions. Ensures the datetime index is monotonic and properly localized before "
                    "any temporal analysis.",
                    style=_exp_inner_style()
                ),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("Low irradiance / power filter"), style=_exp_summary_style()),
                html.Div([
                    "Removes non-representative operating points using three simultaneous conditions: ",
                    "① irradiance above a minimum threshold; ",
                    "② power exceeding a minimum fraction of irradiance; ",
                    "③ temperature-corrected normalized power ",
                    html.Span("(norm = P / [G · (1 + γ(T − 25))] × 1000)", style={"fontFamily": "Times New Roman, serif"}),
                    html.Sup("[1]"),
                    " within a valid range. Points failing any condition are excluded."
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("Outlier removal (IQR)"), style=_exp_summary_style()),
                html.Div([
                    "Detects statistical outliers on the temperature-corrected normalized power signal "
                    "using the IQR method",
                    html.Sup("[2]"),
                    ". Points outside ",
                    html.Span("[Q1 − k·IQR, Q3 + k·IQR]", style={"fontFamily": "Times New Roman, serif"}),
                    " (default k = 1.5, Tukey's fence) are flagged and excluded from downstream "
                    "degradation analysis."
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("Clear-sky filter"), style=_exp_summary_style()),
                html.Div([
                    "Applied to the raw irradiance signal before power normalization, preserving the "
                    "full intraday bell-shaped profile needed for smoothness scoring. Follows the "
                    "approach of Meyers et al.",
                    html.Sup("[3]"),
                    " The algorithm is resolution-aware:",
                    html.Br(), html.Br(),
                    html.B("Sub-daily data (≥4 readings/day): "),
                    "① a smoothness score derived from the L1-norm of the 2nd-order temporal difference "
                    "of the intraday irradiance signal (smooth bell-shaped profiles score high); ",
                    "② a seasonally-normalized daily energy score (ratio of daily irradiance sum to a "
                    "rolling 90th-percentile baseline, ±30-day window). A day is classified as clear "
                    "only if both scores exceed their respective thresholds (AND rule).",
                    html.Br(), html.Br(),
                    html.B("Coarse / downsampled data (<4 readings/day): "),
                    "smoothness cannot be reliably estimated from sparse samples, so the filter falls "
                    "back to energy-only mode — retaining days whose seasonally-normalized irradiance "
                    "exceeds the energy threshold."
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            # References
            html.Hr(style={"borderColor": BORDER, "margin": "10px 0 6px"}),
            html.Details([
                html.Summary("References", style={
                    "cursor": "pointer", "fontSize": "11px", "fontWeight": "700", "color": NAVY,
                }),
                html.Ol([
                    html.Li([
                        "IEC 60891:2021 — Photovoltaic devices: Procedures for temperature and "
                        "irradiance corrections to measured I-V characteristics. ",
                        html.A("webstore.iec.ch", href="https://webstore.iec.ch/en/publication/61766",
                               target="_blank", style=_exp_link_style()), "."
                    ], style={"marginBottom": "4px"}),
                    html.Li([
                        "Kim, G. G., Hyun, J. H., Choi, J. H., Bhang, B. G., & Ahn, H. K. (2023). "
                        "Quality analysis of photovoltaic system using descriptive statistics of "
                        "power performance index. ", html.Em("IEEE Access"), ", 11, 28427–28438. ",
                        html.A("10.1109/ACCESS.2023.3257373",
                               href="https://doi.org/10.1109/ACCESS.2023.3257373",
                               target="_blank", style=_exp_link_style()), "."
                    ], style={"marginBottom": "4px"}),
                    html.Li([
                        "B. E. Meyers, E. Apostolaki-Iosifidou, and L. Schelhas, \"Solar Data Tools: "
                        "Automatic Solar Data Processing Pipeline,\" ",
                        html.Em("2020 47th IEEE Photovoltaic Specialists Conference (PVSC)"),
                        ", Calgary, AB, Canada, 2020, pp. 0655–0656. doi: ",
                        html.A("10.1109/PVSC45281.2020.9300847",
                               href="https://doi.org/10.1109/PVSC45281.2020.9300847",
                               target="_blank", style=_exp_link_style()), "."
                    ]),
                ], style={"paddingLeft": "16px", "marginTop": "6px", "marginBottom": "0",
                          "fontSize": "11px", "color": "#4a6fa5", "lineHeight": "1.5"})
            ]),

        ], style=_exp_outer_style())
    ], style={"marginTop": "12px"})


def metric_explanations_block():
    """Collapsible 'Metric detail' panel for the 5 degradation methods.
    Each entry has description, LaTeX-style equation, and a reference."""
    return html.Details([
        html.Summary("Metric details (equations & references)", style={
            "cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "fontWeight": "600",
            "fontFamily": "Arial, sans-serif",
        }),
        html.Div([

            html.Details([
                html.Summary(html.B("YoY (Year-over-Year)"), style=_exp_summary_style()),
                html.Div([
                    "Compares daily irradiance-weighted power to the same calendar day one year prior. "
                    "The degradation rate is the median of all year-over-year ratios after IQR-based "
                    "outlier removal.",
                    dcc.Markdown(
                        r"$$R_i = \frac{P(t)}{P(t-1\,\text{yr})} - 1, \quad R_d = \text{median}(R_i) \times \frac{100\%}{\text{yr}}$$",
                        mathjax=True, style=_eq_style()
                    ),
                    html.Div([
                        html.Sup("[1] "),
                        "Jordan, D. et al., IEEE J. Photovoltaics 8(2), 525–531, 2018. ",
                        html.A("10.1109/JPHOTOV.2017.2779779",
                               href="https://doi.org/10.1109/JPHOTOV.2017.2779779",
                               target="_blank", style=_exp_link_style())
                    ], style=_ref_style()),
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("LR (Linear Regression)"), style=_exp_summary_style()),
                html.Div([
                    "Fits an ordinary least-squares line to the daily power time series. "
                    "The degradation rate is the slope normalized by mean power.",
                    dcc.Markdown(
                        r"$$P(t) = \beta_0 + \beta_1 t, \quad R_d = \frac{\beta_1}{\bar{P}} \times \frac{100\%}{\text{yr}}$$",
                        mathjax=True, style=_eq_style()
                    ),
                    html.Div("No tunable parameters.",
                             style={"fontSize": "11px", "color": MUTED, "fontStyle": "italic"}),
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("HW (Holt-Winters)"), style=_exp_summary_style()),
                html.Div([
                    "Additive Holt-Winters exponential smoothing decomposes the signal into level, "
                    "trend, and seasonal components. A linear regression on the fitted values yields "
                    "the degradation rate.",
                    dcc.Markdown(
                        r"$$\hat{y}(t) = L(t) + T(t) + S(t), \quad R_d = \frac{\text{slope}(\hat{y})}{\bar{\hat{y}}} \times \frac{100\%}{\text{yr}}$$",
                        mathjax=True, style=_eq_style()
                    ),
                    html.Div([
                        html.Sup("[2] "),
                        "Phinikarides, A. et al., Renew. Sustain. Energy Rev. 40, 143–152, 2014. ",
                        html.A("10.1016/j.rser.2014.07.155",
                               href="https://doi.org/10.1016/j.rser.2014.07.155",
                               target="_blank", style=_exp_link_style())
                    ], style=_ref_style()),
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("ARIMA / SARIMA"), style=_exp_summary_style()),
                html.Div([
                    "Fits a SARIMA(p,d,q)(0,1,1,s) model. A linear regression on the fitted values "
                    "extracts the degradation rate.",
                    dcc.Markdown(
                        r"$$\text{SARIMA}(p,d,q)(0,1,1,s), \quad R_d = \frac{\text{slope}(\hat{y})}{\bar{\hat{y}}} \times \frac{100\%}{\text{yr}}$$",
                        mathjax=True, style=_eq_style()
                    ),
                    html.Div([
                        html.Sup("[2] "),
                        "Phinikarides, A. et al., Renew. Sustain. Energy Rev. 40, 143–152, 2014. ",
                        html.A("10.1016/j.rser.2014.07.155",
                               href="https://doi.org/10.1016/j.rser.2014.07.155",
                               target="_blank", style=_exp_link_style())
                    ], style=_ref_style()),
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("CSD (Classical Seasonal Decomposition)"), style=_exp_summary_style()),
                html.Div([
                    "Decomposes the daily power series additively into trend, seasonal, and residual "
                    "components. A linear regression on the extracted trend gives the degradation rate.",
                    dcc.Markdown(
                        r"$$P(t) = T(t) + S(t) + R(t), \quad R_d = \frac{\text{slope}(T)}{\bar{T}} \times \frac{100\%}{\text{yr}}$$",
                        mathjax=True, style=_eq_style()
                    ),
                    html.Div([
                        html.Sup("[2] "),
                        "Phinikarides, A. et al., Renew. Sustain. Energy Rev. 40, 143–152, 2014. ",
                        html.A("10.1016/j.rser.2014.07.155",
                               href="https://doi.org/10.1016/j.rser.2014.07.155",
                               target="_blank", style=_exp_link_style())
                    ], style=_ref_style()),
                ], style=_exp_inner_style()),
            ], style={"marginBottom": "6px"}),

            html.Details([
                html.Summary(html.B("PVPRO (Single-diode-model fitting)"), style=_exp_summary_style()),
                html.Div([
                    "Fits the five single-diode-model parameters (photocurrent, saturation current, "
                    "series resistance, shunt resistance, diode factor) to short time-windows of DC "
                    "voltage and current measurements. The reference maximum-power point P_mp,ref is "
                    "reconstructed from the fitted parameters in each window, and its long-term trend "
                    "(via rdtools year-over-year) is reported as the degradation rate. Unlike the "
                    "power-based methods, PVPRO can attribute degradation to specific physical mechanisms.",
                    dcc.Markdown(
                        r"$$\hat{P}_{mp,ref}(t) = f_{SDM}\bigl(I_L, I_0, R_s, R_{sh}, n\bigr)(t), \quad "
                        r"R_d = \text{YoY}\bigl(\hat{P}_{mp,ref}\bigr)$$",
                        mathjax=True, style=_eq_style()
                    ),
                    html.Div([
                        html.Sup("[3] "),
                        "Li, B., Karin, T., Meyers, B. E., Chen, X., Jordan, D. C., "
                        "Hansen, C. W., ... & Jain, A. (2023). Determining circuit model "
                        "parameters from operation data for PV system degradation analysis: "
                        "PVPRO. Solar Energy 254, 168–181. ",
                        html.A("10.1016/j.solener.2023.03.011",
                               href="https://doi.org/10.1016/j.solener.2023.03.011",
                               target="_blank", style=_exp_link_style())
                    ], style=_ref_style()),
                ], style=_exp_inner_style()),
            ]),

        ], style=_exp_outer_style())
    ], style={"marginTop": "12px"})


def _example_chip_style():
    return {
        "padding": "8px 14px",
        "background": "white",
        "color": NAVY,
        "border": f"1px solid {BORDER_STRONG}",
        "borderRadius": "999px",
        "fontSize": "13px",
        "fontWeight": "500",
        "cursor": "pointer",
        "fontFamily": "Arial, sans-serif",
        "whiteSpace": "nowrap",
        "transition": "all 0.15s ease",
    }


def _chat_bubble(role, text, fresh=False):
    """Render one chat message bubble.

    For assistants: ALWAYS uses the typing-bubble DOM structure (3 spans).
    Keeping the structure identical across renders prevents React from tearing
    down old typing-bubbles when a new message is appended (which would cause
    a `removeChild` reconciliation error).

    - fresh=True  → empty visible span, visible caret. JS animates the typing.
    - fresh=False → pre-filled visible span, hidden caret, className includes
                    `chat-bubble-done` so the JS leaves it alone.
    """
    is_user = role == "user"

    bubble_style = {
        "padding": "12px 16px",
        "background": "#dbeafe" if is_user else "white",
        "color": NAVY_DEEP if is_user else INK,
        "border": f"1px solid #93c5fd" if is_user else f"1px solid {BORDER}",
        "boxShadow": "0 1px 2px rgba(0, 100, 171, 0.08)" if is_user else "0 1px 2px rgba(15, 23, 42, 0.03)",
        "borderRadius": "14px",
        "borderBottomRightRadius": "4px" if is_user else "14px",
        "borderBottomLeftRadius": "14px" if is_user else "4px",
        "maxWidth": "88%",
        "fontSize": "14px",
        "fontWeight": "600" if is_user else "400",
        "lineHeight": "1.6",
        "fontFamily": "Arial, sans-serif",
        "whiteSpace": "pre-wrap",
    }

    if is_user:
        inner = html.Div(text, style=bubble_style)
    else:
        # Always render the typing-bubble structure for assistant messages
        # For non-fresh (already-typed-out) bubbles, pre-fill the visible span
        # with the text minus markdown bold markers. The clientside JS will
        # then convert it to proper HTML <strong> tags on its next pass.
        visible_initial = "" if fresh else text.replace("**", "")
        caret_style = {} if fresh else {"opacity": "0"}
        wrapper_class = "chat-bubble-typing" if fresh else "chat-bubble-typing chat-bubble-done"
        inner = html.Div(
            [
                html.Span(visible_initial, className="chat-typed"),
                html.Span(text, className="chat-typed-source", style={"display": "none"}),
                html.Span("▍", className="chat-typing-caret", style=caret_style),
            ],
            className=wrapper_class,
            style=bubble_style,
        )

    return html.Div(
        inner,
        style={
            "display": "flex",
            "justifyContent": "flex-end" if is_user else "flex-start",
            "marginBottom": "10px",
        }
    )


# =============================================================================
# SIDEBAR — workflow stepper + login
# =============================================================================
def stepper_item(num, title, sub, color, state="pending", step_key=None):
    """state: 'done' | 'active' | 'pending'"""
    is_done   = state == "done"
    is_active = state == "active"

    # Bullet (number / checkmark)
    if is_done:
        bullet_bg = SUCCESS
        bullet_fg = "white"
        bullet_border = "none"
    elif is_active:
        bullet_bg = color  # navy
        bullet_fg = "white"
        bullet_border = "none"
    else:
        bullet_bg = "transparent"
        bullet_fg = MUTED
        bullet_border = f"1.5px solid {BORDER_STRONG}"
    bullet_content = "✓" if is_done else str(num)

    title_color = INK if (is_done or is_active) else MUTED
    title_weight = "600" if is_active else ("500" if is_done else "500")
    row_bg = NAVY_SOFT if is_active else "transparent"
    row_border = f"1px solid #bfdbfe" if is_active else "1px solid transparent"

    return html.Div(
        [
            html.Div(
                bullet_content,
                style={
                    "width": "26px",
                    "height": "26px",
                    "borderRadius": "50%",
                    "background": bullet_bg,
                    "color": bullet_fg,
                    "border": bullet_border,
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "14px",
                    "fontWeight": "700",
                    "fontFamily": "Arial, sans-serif",
                    "flexShrink": "0",
                    "transition": "all 0.25s ease",
                }
            ),
            html.Div(
                [
                    html.Div(title, style={
                        "fontSize": "15px",
                        "fontWeight": title_weight,
                        "color": title_color,
                        "fontFamily": "Arial, sans-serif",
                        "whiteSpace": "nowrap",
                    }),
                    html.Div(sub, style={
                        "fontSize": "13px",
                        "color": MUTED if state == "pending" else INK_SOFT,
                        "marginTop": "1px",
                        "fontFamily": "Arial, sans-serif",
                        "whiteSpace": "nowrap",
                    }),
                ],
                style={"marginLeft": "12px", "flex": "1", "minWidth": "0"}
            ),
            # Status pill (right side) — fixed slot
            html.Div(
                "done" if is_done else ("active" if is_active else ""),
                style={
                    "fontSize": "10px",
                    "fontWeight": "700",
                    "color": SUCCESS if is_done else (color if is_active else "transparent"),
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "fontFamily": "Arial, sans-serif",
                    "flexShrink": "0",
                    "marginLeft": "8px",
                }
            ),
        ],
        id={"type": "step-row", "step": step_key} if step_key else None,
        n_clicks=0,
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "10px 12px",
            "borderRadius": "8px",
            "background": row_bg,
            "border": row_border,
            "marginBottom": "4px",
            "transition": "all 0.25s ease",
            "cursor": "pointer" if step_key else "default",
            "userSelect": "none",
        }
    )


def _step_state(progress, step_key, prior_done):
    """Return the visual state for a single step."""
    if progress.get(step_key):
        return "done"
    if prior_done:
        return "active"
    return "pending"


def _chat_sidebar_item():
    """A clickable sidebar row for the Ask-Assistant chat section.

    Visually distinct from the numbered steppers:
      * Light-blue tint (NAVY_SOFT) for the background, matching the
        "active" treatment of the numbered steps -- chat is *always*
        available, so always reading as available is on-brand.
      * Top-border + margin-top to create a clear gray divider between
        the linear workflow (steps 1-4) and the always-on chat helper.
        Without the divider the chat row read as a fifth step.

    Carries the same pattern-dict id (`{"type": "step-row", "step": "chat"}`)
    as the numbered steps, so the existing scroll-to-target clientside
    callback picks it up automatically; it scrolls to the element with
    id="agent-chat-wrap" on the right panel.
    """
    return html.Div(
        [
            # Bullet -- chat glyph instead of a digit.  Solid blue
            # background instead of an outline since this row is always
            # "active".
            html.Div(
                "💬",
                style={
                    "width": "26px",
                    "height": "26px",
                    "borderRadius": "50%",
                    "background": "white",
                    "color": NAVY,
                    "border": f"1.5px solid {BORDER_STRONG}",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "13px",
                    "flexShrink": "0",
                }
            ),
            html.Div(
                [
                    html.Div("Ask the Assistant", style={
                        "fontSize": "15px",
                        "fontWeight": "600",
                        "color": INK,
                        "fontFamily": "Arial, sans-serif",
                        "whiteSpace": "nowrap",
                    }),
                    html.Div("Chat about your data", style={
                        "fontSize": "13px",
                        "color": INK_SOFT,
                        "marginTop": "1px",
                        "fontFamily": "Arial, sans-serif",
                        "whiteSpace": "nowrap",
                    }),
                ],
                style={"marginLeft": "12px", "flex": "1", "minWidth": "0"}
            ),
            # Empty right slot to match the stepper_item layout (where
            # the "done"/"active" status pill lives).
            html.Div(
                "",
                style={
                    "fontSize": "10px",
                    "flexShrink": "0",
                    "marginLeft": "8px",
                }
            ),
        ],
        id={"type": "step-row", "step": "chat"},
        n_clicks=0,
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "10px 12px",
            "borderRadius": "8px",
            # Light gray (slate-100) instead of NAVY_SOFT -- the blue
            # was reading too close to the "active" stepper row and
            # competing for attention; a neutral gray makes the chat
            # entry feel like a calm, always-available helper rather
            # than a step the user needs to act on next.
            "background": "#f1f5f9",
            "border": f"1px solid {BORDER}",
            "marginTop": "0",
            "marginBottom": "4px",
            "transition": "all 0.25s ease",
            "cursor": "pointer",
            "userSelect": "none",
        }
    )


def build_sidebar(progress=None):
    progress = progress or {"data": False, "filter": False, "calc": False, "code": False}

    s_data   = _step_state(progress, "data",   prior_done=True)
    s_filter = _step_state(progress, "filter", prior_done=progress.get("data", False))
    s_calc   = _step_state(progress, "calc",   prior_done=progress.get("filter", False))
    s_code   = _step_state(progress, "code",   prior_done=progress.get("calc", False))

    return html.Div(
        [
            # Brand — logo on top, slogan beneath
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("pvcopilot_logo.png"),
                        style={
                            "height": "56px",
                            "width": "auto",
                            "objectFit": "contain",
                            "display": "block",
                            "marginBottom": "8px",
                        }
                    ),
                    html.Div("Data in. Results out.", style={
                        "fontSize": "14px",
                        "color": INK,
                        "fontFamily": "Arial, sans-serif",
                        "fontWeight": "700",
                        "textAlign": "left",
                    }),
                ],
                style={"padding": "20px 18px 24px"}
            ),

            # Workflow section.  Horizontal padding matches the brand
            # block above (18px) so the "WORKFLOW" label aligns flush
            # with the left edge of the "PV Copilot" logo and the
            # "Data in. Results out." slogan.  (The previous 12px left
            # padding offset it 6px to the left and read as misaligned.)
            html.Div(
                [
                    section_label("Workflow"),
                    stepper_item(1, "Data Prescreening", "Upload & inspect", TEAL,   state=s_data,   step_key="data"),
                    stepper_item(2, "Filter",             "Clean the signal", INDIGO, state=s_filter, step_key="filter"),
                    stepper_item(3, "Degradation",        "Compute the rate", ROSE,   state=s_calc,   step_key="calc"),
                    stepper_item(4, "Code",               "Export & reuse",   SLATE,  state=s_code,   step_key="code"),
                    # Divider between the linear workflow (steps 1-4)
                    # and the always-on Chat helper.  A 1px gray line
                    # sitting in 14px of vertical breathing room reads
                    # as a hard category break rather than a fifth step.
                    html.Div(style={
                        "borderTop": f"1px solid {BORDER}",
                        "margin": "14px 4px",
                    }),
                    # Chat is a "bonus" entry, NOT a numbered step -- not
                    # gated on prior progress, never marked done.  Clicking
                    # it scrolls the right panel to the Ask-Assistant chat
                    # section (id="agent-chat-wrap"), via the same
                    # scroll-into-view clientside callback that handles the
                    # numbered steps.  See _chat_sidebar_item() above.
                    _chat_sidebar_item(),

                    # Restart button — shown when at least one step is complete
                    html.Div(
                        html.Button(
                            ["Restart workflow"],
                            id="restart-btn",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px 14px",
                                "marginTop": "16px",
                                "background": "#f1f5f9",
                                "color": INK_SOFT,
                                "border": f"1px solid {BORDER_STRONG}",
                                "borderRadius": "8px",
                                "fontSize": "13px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                                "fontFamily": "Arial, sans-serif",
                            }
                        ),
                        style={
                            "display": "block" if any(progress.values()) else "none",
                            "marginBottom": "24px",
                        }
                    ),
                ],
                style={"padding": "0 18px"}
            ),

            # Spacer
            html.Div(style={"flex": "1"}),

            # About box.  Same 18px horizontal padding as the workflow
            # block above and the brand block at the top -- keeps the
            # whole left rail visually aligned.
            html.Div(
                [
                    section_label("About"),
                    html.Ul(
                        [
                            html.Li("LLM-powered PV analysis"),
                            html.Li("No coding required"),
                            html.Li("Downloadable Python at the end"),
                        ],
                        style={
                            "fontSize": "13px",
                            "color": INK_SOFT,
                            "lineHeight": "1.6",
                            "fontFamily": "Arial, sans-serif",
                            "marginBottom": "12px",
                            "paddingLeft": "18px",
                        }
                    ),
                    # Demo video link — text only
                    html.A(
                        [
                            html.Span("▶", style={
                                "color": NAVY,
                                "marginRight": "6px",
                                "fontSize": "11px",
                            }),
                            "Watch 30-second demo",
                        ],
                        href="https://www.youtube.com/watch?v=QuTOc8Fb4g4",
                        target="_blank",
                        style={
                            "fontSize": "13px",
                            "color": NAVY,
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "600",
                            "textDecoration": "none",
                            "display": "inline-flex",
                            "alignItems": "center",
                        }
                    ),
                ],
                style={"padding": "0 18px 20px"}
            ),

            # User / login block
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "G",
                                style={
                                    "width": "32px",
                                    "height": "32px",
                                    "borderRadius": "50%",
                                    # Light gray, not navy -- it's a disabled
                                    # placeholder until sign-in ships.
                                    "background": "#cbd5e1",
                                    "color": "#ffffff",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "fontSize": "15px",
                                    "fontWeight": "600",
                                    "fontFamily": "Arial, sans-serif",
                                    "opacity": "0.85",
                                }
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Sign in", style={
                                                "fontSize": "15px",
                                                "fontWeight": "600",
                                                "color": INK,
                                                "fontFamily": "Arial, sans-serif",
                                            }),
                                            html.Span("Coming soon", style={
                                                "fontSize": "10px",
                                                "fontWeight": "700",
                                                "color": NAVY,
                                                "background": NAVY_SOFT,
                                                "padding": "2px 8px",
                                                "borderRadius": "999px",
                                                "marginLeft": "8px",
                                                "letterSpacing": "0.04em",
                                                "textTransform": "uppercase",
                                                "fontFamily": "Arial, sans-serif",
                                                "verticalAlign": "middle",
                                            }),
                                        ],
                                        style={"display": "flex", "alignItems": "center"}
                                    ),
                                    html.Div("Save and reload past sessions", style={
                                        "fontSize": "13px",
                                        "color": INK_SOFT,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginTop": "2px",
                                    }),
                                ],
                                style={"marginLeft": "10px", "flex": "1"}
                            ),
                            html.Button(
                                "→",
                                id="login-btn",
                                n_clicks=0,
                                disabled=True,
                                title="Sign-in is coming soon",
                                style={
                                    "width": "28px",
                                    "height": "28px",
                                    "borderRadius": "6px",
                                    "border": f"1px solid {BORDER_STRONG}",
                                    "background": "transparent",
                                    "color": MUTED,
                                    "cursor": "not-allowed",
                                    "fontSize": "16px",
                                    "opacity": "0.5",
                                }
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"}
                    ),
                ],
                style={
                    "padding": "14px 18px",
                    "borderTop": f"1px solid {BORDER}",
                    "background": "transparent",
                }
            ),
        ],
        style={
            "width": "320px",
            "flexShrink": "0",
            "background": SIDEBAR_BG,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "display": "flex",
            "flexDirection": "column",
            "height": "calc(100vh - 110px)",
            "overflowY": "auto",
            "boxShadow": "0 1px 3px rgba(15, 23, 42, 0.04)",
            "fontFamily": "Arial, sans-serif",
        }
    )


# Initial sidebar render (no steps complete yet)
sidebar = build_sidebar()


# =============================================================================
# CHAT — AGENT 1 · DATA
# =============================================================================
data_agent_body = html.Div([
    html.Div(
        [
            "Welcome — let's analyze some PV data. Drop a ",
            html.B("CSV, Excel, or Parquet"),
            " file below, or try one of the example datasets. I'll detect your columns and preview the raw signal.",
        ],
        style={
            "fontSize": "16px",
            "color": INK,
            "lineHeight": "1.6",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "18px",
        }
    ),

    # Requirements -- the IMPORTANT badge sits inside the summary, just
    # after the CSS-generated triangle.  The whole summary line is bold.
    html.Details(
        [
            html.Summary(
                [
                    html.Span("IMPORTANT", className="important-badge", style={
                        "fontSize": "10px",
                        "fontWeight": "700",
                        "color": "white",
                        "background": NAVY,
                        "padding": "2px 8px",
                        "borderRadius": "999px",
                        "letterSpacing": "0.06em",
                        "marginRight": "10px",
                        "verticalAlign": "middle",
                    }),
                    html.Span("Click to see data requirements"),
                ],
                style={
                    "cursor": "pointer",
                    "fontSize": "14px",
                    "color": INK,
                    "fontFamily": "Arial, sans-serif",
                    "fontWeight": "700",
                    "marginBottom": "8px",
                }
            ),
            html.Ul(
                [
                    html.Li([html.Span("Columns: ", style={"color": INK_SOFT}), html.B("time, power, irradiance, temperature")]),
                    html.Li([html.Span("Duration: ", style={"color": INK_SOFT}), html.B("≥ 2 years"), " for reliable degradation"]),
                    html.Li([html.Span("Resolution: ", style={"color": INK_SOFT}), html.B("1–6 hours")]),
                ],
                style={
                    "fontSize": "14px",
                    "color": INK,
                    "paddingLeft": "18px",
                    "lineHeight": "1.7",
                    "marginBottom": "0",
                    "marginTop": "10px",
                    "padding": "12px 14px 12px 32px",
                    "background": "#eff6ff",
                    "border": "1px solid #bfdbfe",
                    "borderRadius": "10px",
                    "fontFamily": "Arial, sans-serif",
                }
            ),
        ],
        style={"marginBottom": "16px"}
    ),

    # Upload zone.  Padding intentionally compact -- the box was twice
    # this tall and users found it visually dominant relative to the
    # other Step-1 elements (example chips, Analyze button).  16px of
    # vertical padding fits the icon + two lines of text without
    # crowding, and matches the visual weight of the rest of the panel.
    dcc.Upload(
        id="upload-data",
        accept=".csv, text/csv, .xls, .xlsx, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, .parquet",
        children=html.Div(
            [
                html.Div(
                    "↑",
                    style={
                        "fontSize": "20px",
                        "color": INK_SOFT,
                        "marginBottom": "2px",
                        "fontWeight": "300",
                        "lineHeight": "1",
                    }
                ),
                html.Div(
                    [
                        "Drag a file here, or ",
                        html.Span("browse", style={"color": ACCENT, "textDecoration": "underline", "fontWeight": "500"}),
                    ],
                    style={"fontSize": "14px", "color": INK, "fontFamily": "Arial, sans-serif"}
                ),
                html.Div(".csv  ·  .xlsx  ·  .parquet", style={
                    "fontSize": "12px",
                    "color": INK_SOFT,
                    "marginTop": "2px",
                    "fontFamily": "Arial, sans-serif",
                }),
            ],
            style={"textAlign": "center"}
        ),
        style={
            "width": "100%",
            "padding": "14px 16px",
            "border": f"1.5px dashed {BORDER_STRONG}",
            "borderRadius": "10px",
            "backgroundColor": "#f8fafc",
            "cursor": "pointer",
            "transition": "all 0.15s ease",
        }
    ),

    html.Div(id="upload-status-output", style={"marginTop": "10px"}),

    # Example buttons
    html.Div(
        [
            html.Div("Or try an example:", style={
                "fontSize": "14px",
                "color": INK_SOFT,
                "marginBottom": "8px",
                "fontFamily": "Arial, sans-serif",
            }),
            html.Div(
                [
                    html.Button(
                        ["⬢  ", "Example 1"],
                        id="load-example-btn-1",
                        n_clicks=0,
                        style={
                            "padding": "8px 14px",
                            "background": "transparent",
                            "border": f"1px solid {BORDER_STRONG}",
                            "borderRadius": "8px",
                            "fontSize": "14px",
                            "color": INK,
                            "cursor": "pointer",
                            "marginRight": "8px",
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "500",
                        }
                    ),
                    html.Button(
                        ["⬢  ", "Example 2"],
                        id="load-example-btn-2",
                        n_clicks=0,
                        style={
                            "padding": "8px 14px",
                            "background": "transparent",
                            "border": f"1px solid {BORDER_STRONG}",
                            "borderRadius": "8px",
                            "fontSize": "14px",
                            "color": INK,
                            "cursor": "pointer",
                            "marginRight": "8px",
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "500",
                        }
                    ),
                    html.Button(
                        ["⬢  ", "Example 3"],
                        id="load-example-btn-3",
                        n_clicks=0,
                        style={
                            "padding": "8px 14px",
                            "background": "transparent",
                            "border": f"1px solid {BORDER_STRONG}",
                            "borderRadius": "8px",
                            "fontSize": "14px",
                            "color": INK,
                            "cursor": "pointer",
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "500",
                        }
                    ),
                ],
            ),
        ],
        style={"marginTop": "18px"}
    ),

    # Analyze button — the action that triggers analysis
    html.Button(
        "Analyze Data",
        id="analyze-btn",
        n_clicks=0,
        style={
            "width": "100%",
            "padding": "12px 16px",
            "marginTop": "18px",
            "background": INK,
            "color": PAPER,
            "border": "none",
            "borderRadius": "10px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Arial, sans-serif",
            "letterSpacing": "0.01em",
        }
    ),
    html.Div(
        "Analysis typically takes 2–10 seconds",
        style={
            "fontSize": "13px",
            "color": INK_SOFT,
            "marginTop": "6px",
            "textAlign": "center",
            "fontFamily": "Arial, sans-serif",
        }
    ),

    # Output area — identified variables + raw figures (filled by callback)
    dcc.Loading(
        id="loading-summary-and-figs",
        type="circle",
        color=TEAL,
        children=html.Div(
            id="data-summary-output",
            style={
                "marginTop": "22px",
                "minHeight": "0",
            }
        ),
    ),
], style={"fontFamily": "Arial, sans-serif"})


# =============================================================================
# CHAT — AGENT 2 · FILTER
# =============================================================================
def filter_row(checkbox_id, label, customize_body=None):
    parts = [
        html.Div(
            [
                dbc.Checkbox(
                    id=checkbox_id, value=True,
                    className="me-2 d-inline-block",
                    # Make the filled checkbox NAVY (matches the rest of
                    # the app's accents).  `accent-color` is the modern
                    # CSS property browsers use to theme form controls.
                    input_style={"accentColor": NAVY},
                ),
                html.Span(label, style={
                    "fontSize": "15px",
                    "color": INK,
                    "fontFamily": "Arial, sans-serif",
                    "fontWeight": "700",
                }),
            ],
            style={"display": "flex", "alignItems": "center"}
        )
    ]
    if customize_body is not None:
        parts.append(html.Details([
            html.Summary("Customize parameters", style={
                "cursor": "pointer",
                "color": INK_SOFT,
                "fontSize": "13px",
                "fontWeight": "500",
                "marginTop": "6px",
                "marginLeft": "26px",
                "fontFamily": "Arial, sans-serif",
            }),
            html.Div(customize_body, style={
                "marginTop": "8px",
                "marginLeft": "26px",
                "padding": "12px 14px",
                "background": "#f1f5f9",
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
                "fontSize": "14px",
            })
        ]))
    return html.Div(parts, style={"marginBottom": "12px"})


_param_input_style = {
    "width": "100%",
    "fontSize": "14px",
    "padding": "6px 8px",
    "borderRadius": "6px",
    "border": f"1px solid {BORDER_STRONG}",
    "color": INK,
    "fontFamily": "Arial, sans-serif",
    "background": "white",
}

_label_style = {"fontSize": "13px", "fontWeight": "600", "color": INK, "marginBottom": "3px", "fontFamily": "Arial, sans-serif"}
_help_style  = {"fontSize": "13px", "color": INK_SOFT, "marginBottom": "5px", "lineHeight": "1.4", "fontFamily": "Arial, sans-serif"}


low_irra_params = html.Div([
    html.Div([
        html.Label("γ — temperature coefficient of power (/°C)", style=_label_style),
        dcc.Input(id="param-gamma", type="number", value=-0.004, step=0.001, style=_param_input_style),
    ], style={"marginBottom": "10px"}),
    html.Div([
        html.Label("Min. irradiance threshold (W/m²)", style=_label_style),
        html.Div("Excludes data below this irradiance level.", style=_help_style),
        dcc.Input(id="param-irr-thresh", type="number", value=300, step=10, min=0, style=_param_input_style),
    ], style={"marginBottom": "10px"}),
    html.Div([
        html.Label("Min. power / irradiance ratio", style=_label_style),
        html.Div("Rejects points where P < ratio × G.", style=_help_style),
        dcc.Input(id="param-power-ratio", type="number", value=0.02, step=0.005, min=0, style=_param_input_style),
    ]),
    dcc.Input(id="param-norm-lower",     type="number", value=0.01, style={"display": "none"}),
    dcc.Input(id="param-norm-upper-pct", type="number", value=99,   style={"display": "none"}),
])

outlier_params = html.Div([
    html.Label("IQR multiplier (k)", style=_label_style),
    html.Div("Bounds = [Q1 − k·IQR, Q3 + k·IQR]. Tukey default k = 1.5.", style=_help_style),
    dcc.Input(id="param-iqr-multiplier", type="number", value=1.5, step=0.1, min=0.1, style=_param_input_style),
])

clearsky_params = html.Div([
    html.Div([
        html.Label("Smoothness threshold", style=_label_style),
        html.Div("Min per-day smoothness (0–1). Higher = stricter.", style=_help_style),
        dcc.Input(id="param-cs-smooth", type="number", value=0.3, step=0.05, min=0.0, max=1.0, style=_param_input_style),
    ], style={"marginBottom": "10px"}),
    html.Div([
        html.Label("Energy threshold", style=_label_style),
        html.Div("Min seasonally-normalized daily irradiance (0–1).", style=_help_style),
        dcc.Input(id="param-cs-energy", type="number", value=0.5, step=0.05, min=0.0, max=1.0, style=_param_input_style),
    ]),
])


filter_agent_body = html.Div([
    html.Div(
        "I've prepared the recommended filters for your dataset. You can toggle individual filters "
        "off or expand them to customize parameters. When you're ready, hit Apply.",
        style={
            "fontSize": "16px",
            "color": INK,
            "lineHeight": "1.6",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "16px",
        }
    ),

    # Hidden checklist preserving the value contract for callbacks
    dbc.Checklist(
        id="filter-options",
        options=[
            {"label": "", "value": "timezone"},
            {"label": "", "value": "low-irra-power"},
            {"label": "", "value": "outlier"},
            {"label": "", "value": "clearsky"},
        ],
        value=["timezone", "low-irra-power", "outlier", "clearsky"],
        inline=False,
        style={"display": "none"}
    ),

    section_label("Recommended filters"),
    html.Div(
        [
            filter_row("cb-timezone",       "Time zone & DST correction"),
            filter_row("cb-low-irra-power", "Low irradiance / power filter", low_irra_params),
            filter_row("cb-outlier",        "Outlier removal (IQR)",          outlier_params),
            filter_row("cb-clearsky",       "Clear-sky filter",               clearsky_params),
        ],
        style={
            "padding": "16px 18px",
            "background": "#f8fafc",
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "marginBottom": "14px",
        }
    ),

    dcc.Store(id="_cb-sync-dummy"),

    html.Button(
        "Apply Filters",
        id="filter-btn",
        n_clicks=0,
        style={
            "width": "100%",
            "padding": "12px 16px",
            "background": INK,
            "color": "white",
            "border": "none",
            "borderRadius": "10px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Arial, sans-serif",
        }
    ),

    # Collapsible filter explanations (descriptions, equations, references)
    filter_explanations_block(),

    # Output area
    dcc.Loading(
        id="data-filter-result",
        type="circle",
        color=INDIGO,
        children=html.Div(
            id="data-filter-output",
            style={"marginTop": "22px"}
        ),
    ),
], style={"fontFamily": "Arial, sans-serif"})


# =============================================================================
# CHAT — AGENT 3 · DEGRADATION
# =============================================================================
metric_options = [
    {
        "label": html.Div([
            html.B("YoY", style={"fontFamily": "Arial, sans-serif", "fontSize": "16px"}),
            html.Span(" — Year-over-Year", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Details([
                html.Summary("Customize parameters", style={"cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "marginTop": "4px"}),
                html.Div([
                    html.Div("Rolling trend window (days)", style=_label_style),
                    dcc.Input(id="param-yoy-window", type="number", value=30, step=5, min=7, style={**_param_input_style, "marginBottom": "8px"}),
                    html.Div("IQR multiplier k", style=_label_style),
                    dcc.Input(id="param-yoy-iqr", type="number", value=1.5, step=0.1, min=0.5, style=_param_input_style),
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "8px", "border": f"1px solid {BORDER}"}),
            ]),
        ]),
        "value": "YOY",
    },
    {
        "label": html.Div([
            html.B("LR", style={"fontFamily": "Arial, sans-serif", "fontSize": "16px"}),
            html.Span(" — Linear regression", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Div("No tunable parameters.", style={"fontSize": "13px", "color": INK_SOFT, "fontStyle": "italic", "marginTop": "2px"}),
            dcc.Input(id="param-yoy-iqr-dummy", style={"display": "none"}),
        ]),
        "value": "LR",
    },
    {
        "label": html.Div([
            html.B("HW", style={"fontFamily": "Arial, sans-serif", "fontSize": "16px"}),
            html.Span(" — Holt-Winters", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Details([
                html.Summary("Customize parameters", style={"cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "marginTop": "4px"}),
                html.Div([
                    html.Div("Seasonal period (months)", style=_label_style),
                    dcc.Input(id="param-hw-period", type="number", value=12, step=1, min=2, style=_param_input_style),
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "8px", "border": f"1px solid {BORDER}"}),
            ]),
        ]),
        "value": "HW",
    },
    {
        "label": html.Div([
            html.B("ARIMA", style={"fontFamily": "Arial, sans-serif", "fontSize": "16px"}),
            html.Span(" — Auto Regressive Integrated Moving Average", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Details([
                html.Summary("Customize parameters", style={"cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "marginTop": "4px"}),
                html.Div([
                    html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "8px"}, children=[
                        html.Div([
                            html.Div("p", style=_label_style),
                            dcc.Input(id="param-arima-p", type="number", value=1, step=1, min=0, style=_param_input_style),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("d", style=_label_style),
                            dcc.Input(id="param-arima-d", type="number", value=1, step=1, min=0, style=_param_input_style),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("q", style=_label_style),
                            dcc.Input(id="param-arima-q", type="number", value=0, step=1, min=0, style=_param_input_style),
                        ], style={"flex": "1"}),
                    ]),
                    html.Div("Seasonal period s (months)", style=_label_style),
                    dcc.Input(id="param-arima-s", type="number", value=12, step=1, min=2, style=_param_input_style),
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "8px", "border": f"1px solid {BORDER}"}),
            ]),
        ]),
        "value": "ARIMA",
    },
    {
        "label": html.Div([
            html.B("CSD", style={"fontFamily": "Arial, sans-serif", "fontSize": "16px"}),
            html.Span(" — Classical Seasonal Decomposition", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Details([
                html.Summary("Customize parameters", style={"cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "marginTop": "4px"}),
                html.Div([
                    html.Div("Seasonal period (months)", style=_label_style),
                    dcc.Input(id="param-csd-period", type="number", value=12, step=1, min=2, style=_param_input_style),
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "8px", "border": f"1px solid {BORDER}"}),
            ]),
        ]),
        "value": "CSD",
    },
    {
        "label": html.Div([
            # Title row: PVPRO name + description on the left, PVPRO logo
            # (link to the upstream repo) on the right.  `justifyContent:
            # space-between` keeps the logo pinned to the right edge even
            # when the parent <label> is content-sized; `marginLeft: auto`
            # on the anchor is a belt-and-braces fallback.
            # Title row: PVPRO name + a short "light version" subtitle on
            # the left, PVPRO logo (link to upstream repo) on the right.
            html.Div([
                html.Div([
                    html.B("PVPRO", style={"fontFamily": "Arial, sans-serif",
                                           "fontSize": "16px"}),
                    # No data-requirement here -- it moves down to the note
                    # below, where it can be underlined for emphasis.
                    html.Span(
                        " — a lightweight in-app implementation",
                        style={"color": INK_SOFT, "fontSize": "14px"},
                    ),
                ], style={"flex": "1", "minWidth": "0"}),
                # Logo link — opens the upstream PVPRO repo in a new tab.
                html.A(
                    html.Img(
                        src=app.get_asset_url("pvpro_logo.png"),
                        alt="PVPRO",
                        style={"height": "40px", "width": "auto",
                               "display": "block"},
                    ),
                    href="https://github.com/DuraMAT/pvpro",
                    target="_blank",
                    title="PVPRO on GitHub",
                    style={"marginLeft": "auto", "paddingLeft": "12px",
                           "flexShrink": "0",
                           "display": "inline-block",
                           "textDecoration": "none"},
                ),
            ], style={"display": "flex", "alignItems": "center",
                      "justifyContent": "space-between",
                      "width": "100%"}),
            # Runtime + data-requirement note, presented as a soft-blue
            # callout (matches the parent app's "active development" banner).
            # Per the latest spec: "Need" (not "needs"), no bolding -- the
            # data-requirement gets a simple underline so it's visually
            # distinct without competing with the IMPORTANT row below.
            soft_blue_callout(
                [
                    "⏱ ~1–3 minutes runtime · Need ",
                    html.Span("DC Voltage and DC Current",
                              style={"textDecoration": "underline"}),
                    " columns identified in Step 1",
                ],
                margin_top="6px",
                # Breathing room between the warning callout and the
                # IMPORTANT disclosure that follows.
                margin_bottom="12px",
            ),
            html.Details(
                [
                html.Summary(
                    [
                        html.Span("IMPORTANT",
                                  className="important-badge",
                                  style={
                                      "fontSize": "10px",
                                      "fontWeight": "700",
                                      "color": "white",
                                      "background": NAVY,
                                      "padding": "2px 8px",
                                      "borderRadius": "999px",
                                      "letterSpacing": "0.06em",
                                      "marginRight": "10px",
                                      "verticalAlign": "middle",
                                  }),
                        html.Span("Provide module & array parameters for PVPRO"),
                    ],
                    style={"cursor": "pointer",
                           "fontSize": "13px",
                           "color": INK,
                           "fontWeight": "700",
                           "fontFamily": "Arial, sans-serif",
                           "marginTop": "4px"},
                ),
                html.Div([
                    html.Div(style={"display": "flex", "gap": "8px",
                                    "marginBottom": "8px"}, children=[
                        html.Div([
                            html.Div("Cells in series (per module)",
                                     style=_label_style),
                            dcc.Input(id="param-pvpro-cells", type="number",
                                      value=60, step=1, min=1,
                                      style=_param_input_style),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("Modules per string",
                                     style=_label_style),
                            dcc.Input(id="param-pvpro-mps", type="number",
                                      value=1, step=1, min=1,
                                      style=_param_input_style),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("Parallel strings",
                                     style=_label_style),
                            dcc.Input(id="param-pvpro-ps", type="number",
                                      value=1, step=1, min=1,
                                      style=_param_input_style),
                        ], style={"flex": "1"}),
                    ]),
                    html.Div(style={"display": "flex", "gap": "8px",
                                    "marginBottom": "8px"}, children=[
                        html.Div([
                            html.Div("alpha_isc (A/°C)",
                                     style=_label_style),
                            dcc.Input(id="param-pvpro-alphaisc",
                                      type="number", value=0.0046,
                                      step=0.0001, min=0,
                                      style=_param_input_style),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("Technology", style=_label_style),
                            dcc.Dropdown(
                                id="param-pvpro-tech",
                                options=[
                                    {"label": "mono-c-Si", "value": "mono-c-Si"},
                                    {"label": "multi-c-Si", "value": "multi-c-Si"},
                                    {"label": "GaAs", "value": "GaAs"},
                                    {"label": "CIGS", "value": "CIGS"},
                                    {"label": "CdTe", "value": "CdTe"},
                                ],
                                value="mono-c-Si",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ], style={"flex": "1"}),
                    ]),
                    html.Div(style={"display": "flex", "gap": "8px"}, children=[
                        html.Div([
                            html.Div("Days per run", style=_label_style),
                            dcc.Input(id="param-pvpro-days",
                                      type="number", value=14,
                                      step=1, min=2,
                                      style=_param_input_style),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("Iterations per year",
                                     style=_label_style),
                            dcc.Input(id="param-pvpro-iters",
                                      type="number", value=12,
                                      step=1, min=2,
                                      style=_param_input_style),
                        ], style={"flex": "1"}),
                    ]),
                ], style={"marginTop": "6px", "padding": "12px 14px",
                          "background": "#f1f5f9", "borderRadius": "8px",
                          "border": f"1px solid {BORDER}",
                          "width": "100%", "minWidth": "640px",
                          "boxSizing": "border-box"}),
                ],
                id="pvpro-params-details",
                # open state is driven by the metric-selected callback below:
                # auto-open when PVPRO is the active metric, collapse otherwise
                # and on any new-data event.  Start closed (the default radio
                # value is "YOY", not PVPRO).
                open=False,
            ),  # closes html.Details
        ], style={"width": "100%"}),
        "value": "PVPRO",
    },
]


# Split the radio options into two categories so we can render them as two
# logically-separate groups with a divider and category headings between
# them.  The first group is the fast statistical / trend methods that all
# operate on aggregated daily power; the second is PVPRO, the only
# physics-based single-diode-model fit (different inputs, much slower).
stat_metric_options  = [o for o in metric_options if o["value"] != "PVPRO"]
pvpro_metric_options = [o for o in metric_options if o["value"] == "PVPRO"]


def _metric_category_heading(text):
    """Small uppercase, letter-spaced heading used to introduce each category
    of degradation method in the radio group."""
    return html.Div(text, style={
        "fontSize": "11px",
        "color": INK_SOFT,
        "textTransform": "uppercase",
        "letterSpacing": "0.1em",
        "fontWeight": "600",
        "fontFamily": "Arial, sans-serif",
        "marginBottom": "8px",
    })


calc_agent_body = html.Div([
    html.Div(
        "Time to estimate the degradation rate. Choose a method — YoY is the most robust default — "
        "then run the calculation.",
        style={
            "fontSize": "16px",
            "color": INK,
            "lineHeight": "1.6",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "16px",
        }
    ),

    section_label("Choose a metric"),
    html.Div([
        # Category 1 — statistical / trend methods (YoY, LR, HW, ARIMA, CSD).
        _metric_category_heading("statistical / trend methods"),
        dcc.RadioItems(
            id="metric-stat-radio",
            value="YOY",
            options=stat_metric_options,
            labelStyle={"display": "block", "marginBottom": "10px",
                        "cursor": "pointer", "color": "inherit"},
            labelClassName="metric-radio-label",
            inputStyle={"marginRight": "10px", "marginTop": "3px",
                        "accentColor": NAVY},
            style={"marginBottom": "0"},
        ),

        # Visual separator between the two categories.  Stepped up from
        # the BORDER token (#e2e8f0) to slate-400 because lighter shades
        # disappear on the #f8fafc card background; the user explicitly
        # wanted it more visible.
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid #94a3b8",
            "margin": "18px 0 16px 0",
        }),

        # Category 2 — physics-based SDM fit (PVPRO only, for now).
        _metric_category_heading("single-diode-model fitting"),
        dcc.RadioItems(
            id="metric-pvpro-radio",
            value=None,
            options=pvpro_metric_options,
            labelStyle={"display": "block", "marginBottom": "10px",
                        "cursor": "pointer", "color": "inherit"},
            labelClassName="metric-radio-label",
            inputStyle={"marginRight": "10px", "marginTop": "3px",
                        "accentColor": NAVY},
            style={"marginBottom": "0"},
        ),

        # Hidden "master" radio that downstream callbacks read for the
        # currently-selected method.  The two visible RadioItems above
        # mirror their selection into this one via clientside callbacks.
        # Keeping it as a RadioItems (rather than a dcc.Store) means we
        # don't have to touch the rest of the codebase.
        #
        # IMPORTANT: we pass *plain-text* options here -- NOT `metric_options`
        # -- because `metric_options` contains dcc.Input components inside
        # each option's label (the per-method "Customize parameters" panel).
        # Including those again in the hidden master would create duplicate
        # component IDs in the layout, in which case Dash reads from the
        # WRONG (hidden, always-default) copy and the user's input values
        # silently never reach the callback.
        html.Div(
            dcc.RadioItems(
                id="metric-selected-visible",
                value="YOY",
                options=[{"label": opt["value"], "value": opt["value"]}
                         for opt in metric_options],
            ),
            style={"display": "none"},
        ),
    ], style={
        "padding": "16px 18px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "marginBottom": "14px",
    }),

    dcc.Store(id="_rb-sync-dummy"),

    html.Button(
        "Calculate Degradation",
        id="run-btn",
        n_clicks=0,
        style={
            "width": "100%",
            "padding": "12px 16px",
            "background": INK,
            "color": "white",
            "border": "none",
            "borderRadius": "10px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Arial, sans-serif",
        }
    ),

    # Collapsible metric explanations (descriptions, equations, references)
    metric_explanations_block(),

    # FAST methods (YoY/LR/HW/ARIMA/CSD) render under the dcc.Loading spinner.
    dcc.Loading(
        type="circle",
        color=ROSE,
        children=html.Div(
            id="degradation-output",
            style={"marginTop": "22px"}
        ),
    ),

    # PVPRO renders here, OUTSIDE the dcc.Loading boundary. The polling
    # callback writes to this element every ~400 ms while the fit runs,
    # and the dcc.Loading overlay must not flicker on top of it on every
    # tick — that's why it's a sibling, not a child, of the Loading.
    html.Div(id="pvpro-progress-output", style={"marginTop": "22px"}),
], style={"fontFamily": "Arial, sans-serif"})


# =============================================================================
# CHAT — AGENT 4 · CODE
# =============================================================================
code_agent_body = html.Div([
    html.Div(
        "Want to reproduce this analysis on your own machine? I'll bundle every step into a "
        "single runnable Python script — your data path, mapped variables, chosen filters, "
        "and selected metric — ready to download.",
        style={
            "fontSize": "16px",
            "color": INK,
            "lineHeight": "1.6",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "16px",
        }
    ),

    # Generate-code button.  The leading ⬇ glyph was removed per
    # request -- the button text alone reads cleanly enough, and an
    # arrow inside a primary-action button risks being confused with
    # navigation ("download" vs. "generate then download").
    html.Button(
        "Generate Full Python Code",
        id="generate-code-btn",
        n_clicks=0,
        style={
            "width": "100%",
            "padding": "12px 16px",
            "background": INK,
            "color": "white",
            "border": "none",
            "borderRadius": "10px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Arial, sans-serif",
        }
    ),
    html.Div(
        "(typically takes 2–10 seconds)",
        style={
            "fontSize": "13px",
            "color": INK_SOFT,
            "marginTop": "6px",
            "textAlign": "center",
            "fontFamily": "Arial, sans-serif",
        }
    ),

    dcc.Loading(
        id="code-loading",
        type="circle",
        color=SLATE,
        children=html.Div(
            id="code-preview",
            style={"marginTop": "16px"}
        ),
    ),

    html.A(
        ["⬇  Download code (.py)"],
        id="download-link",
        href="",
        download="generated_code.py",
        style={
            "display": "none",
            "marginTop": "12px",
            "color": SLATE,
            "textDecoration": "none",
            "fontSize": "15px",
            "fontWeight": "500",
            "padding": "10px 14px",
            "border": f"1px solid {BORDER_STRONG}",
            "borderRadius": "8px",
            "background": "white",
            "fontFamily": "Arial, sans-serif",
        }
    ),
], style={"fontFamily": "Arial, sans-serif"})


# =============================================================================
# MAIN CHAT STREAM
# =============================================================================
chat_stream = html.Div(
    [
        # Top header bar
        html.Div(
            [
                html.Div(
                    [
                        html.Span("●", style={"color": SUCCESS, "fontSize": "11px", "marginRight": "6px"}),
                        html.Span("Session live", style={"fontSize": "13px", "color": INK_SOFT, "fontFamily": "Arial, sans-serif"}),
                    ],
                    style={"display": "flex", "alignItems": "center"}
                ),
                html.Div(
                    [
                        html.Span("pv-copilot ", style={"color": INK_SOFT, "fontFamily": "Arial, sans-serif", "fontSize": "13px"}),
                        html.Span("v1.0", style={"color": INK, "fontFamily": "Arial, sans-serif", "fontSize": "13px", "fontWeight": "600"}),
                    ],
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "padding": "14px 64px",
                "borderBottom": f"1px solid {BORDER}",
                "background": "rgba(248, 250, 252, 0.95)",
            }
        ),

        # Conversation
        html.Div(
            [
                # Big editorial intro
                html.Div(
                    [
                        html.Div("A conversation with your data", style={
                            "fontSize": "15px",
                            "color": ACCENT,
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "600",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.12em",
                            "marginBottom": "12px",
                        }),
                        html.H1("Estimate PV degradation,", style={
                            "fontSize": "40px",
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "700",
                            "color": INK,
                            
                            "lineHeight": "1.05",
                            "margin": "0",
                        }),
                        html.H1([
                            "one ",
                            html.Em("agent", style={"color": ACCENT}),
                            " at a time.",
                        ], style={
                            "fontSize": "40px",
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "700",
                            "color": INK,
                            
                            "lineHeight": "1.05",
                            "margin": "0 0 16px",
                        }),
                        html.Div(
                            html.Ul(
                                [
                                    html.Li([
                                        html.B("Four specialized agents"),
                                        " — Data Prescreening, Filter, Degradation, and Code — work through your dataset step by step."
                                    ]),
                                    html.Li([
                                        html.B("You stay in control"),
                                        " — review at each stage and tweak the parameters."
                                    ]),
                                    html.Li([
                                        html.B("Executable Python"),
                                        " — walk away with a downloadable script you can run locally."
                                    ]),
                                ],
                                style={
                                    "fontSize": "16px",
                                    "color": INK_SOFT,
                                    "lineHeight": "1.7",
                                    "fontFamily": "Arial, sans-serif",
                                    "maxWidth": "680px",
                                    "paddingLeft": "20px",
                                    "marginBottom": "0",
                                }
                            ),
                        ),
                        # Active-development banner -- placed inside the
                        # hero block so it sits ABOVE the hero's bottom
                        # divider line.  Visually: hero text + banner read
                        # as one introductory unit, separated from the
                        # agents below by the divider.
                        soft_blue_callout(
                            [
                                html.B("Note: "),
                                "This tool is currently under active "
                                "development. If you encounter issues, "
                                "have suggestions, or would like to "
                                "collaborate, please ",
                                html.A(
                                    "contact us",
                                    href="mailto:baojieli@lbl.gov",
                                    style={"color": "#0c4a6e",
                                           "textDecoration": "underline",
                                           "fontWeight": "600"},
                                ),
                                ".",
                            ],
                            margin_top="20px",
                            margin_bottom="0",
                        ),
                    ],
                    style={"padding": "32px 0 28px", "borderBottom": f"1px solid {BORDER}", "marginBottom": "32px"}
                ),

                # Agent 1 — Data (always visible)
                html.Div(
                    agent_message(
                        "data",
                        data_agent_body,
                        intro="I prescreen your dataset — load, inspect, and identify variables."
                    ),
                    id="agent-data-wrap",
                ),

                # Agent 2 — Filter (hidden until step 1 done)
                html.Div(
                    [
                        html.Div(id="agent-filter-locked", children=locked_placeholder("filter", "Filter Agent", 2)),
                        html.Div(
                            agent_message(
                                "filter",
                                filter_agent_body,
                                intro="I clean the signal — outliers, low irradiance, clear-sky filtering."
                            ),
                            id="agent-filter-content",
                            style={"display": "none"},
                        ),
                    ],
                    id="agent-filter-wrap",
                ),

                # Agent 3 — Degradation (hidden until step 2 done)
                html.Div(
                    [
                        html.Div(id="agent-calc-locked", children=locked_placeholder("calc", "Degradation Agent", 3)),
                        html.Div(
                            agent_message(
                                "calc",
                                calc_agent_body,
                                intro="I estimate the annual degradation rate."
                            ),
                            id="agent-calc-content",
                            style={"display": "none"},
                        ),
                    ],
                    id="agent-calc-wrap",
                ),

                # Agent 4 — Code (hidden until step 3 done)
                html.Div(
                    [
                        html.Div(id="agent-code-locked", children=locked_placeholder("code", "Code Agent", 4)),
                        html.Div(
                            agent_message(
                                "code",
                                code_agent_body,
                                intro="I bundle everything into runnable Python."
                            ),
                            id="agent-code-content",
                            style={"display": "none"},
                        ),
                    ],
                    id="agent-code-wrap",
                ),

            ],
            style={
                "maxWidth": "none",
                "padding": "0 64px",
            }
        ),

        # ── Conversational chat (LLM-powered Q&A) ──────────────────────────
        # `agent-chat-wrap` is the scroll target for the "Ask the Assistant"
        # sidebar item; the same scroll-into-view callback that drives the
        # numbered steppers picks this id up via its pattern-dict input.
        html.Div(
            [
                # Section heading — outside the chat panel
                html.Div(
                    [
                        html.Div("Ask the Assistant", style={
                            "fontSize": "20px",
                            "fontWeight": "700",
                            "color": INK,
                            "fontFamily": "Arial, sans-serif",
                            "marginBottom": "6px",
                        }),
                        html.Div(
                            "Questions about the workflow, methods, or your results?",
                            style={
                                "fontSize": "14px",
                                "color": INK_SOFT,
                                "fontFamily": "Arial, sans-serif",
                                "marginBottom": "18px",
                            }
                        ),
                    ]
                ),

                # The chat panel itself — soft blue-tinted surface
                html.Div(
                    [
                        # Message history — scrollable
                        html.Div(
                            id="chat-history",
                            children=[],
                            style={
                                "minHeight": "120px",
                                "maxHeight": "440px",
                                "overflowY": "auto",
                                "padding": "20px 22px",
                                "background": "transparent",
                            }
                        ),

                        # Divider
                        html.Div(style={
                            "height": "1px",
                            "background": "#cbd5e1",
                            "margin": "0",
                        }),

                        # Composer (input + send button) — distinct gray tone, not white
                        html.Div(
                            [
                                # Input wrapper — flex container ensures input fills available width
                                html.Div(
                                    dcc.Input(
                                        id="chat-composer",
                                        placeholder="Ask a question about PV-Copilot…",
                                        type="text",
                                        value="",
                                        debounce=False,
                                        n_submit=0,
                                        style={
                                            "width": "100%",
                                            "boxSizing": "border-box",
                                            "border": "none",
                                            "outline": "none",
                                            "background": "transparent",
                                            # Bumped from 15px -> 17px so
                                            # the composer reads as the
                                            # main input on the page.
                                            "fontSize": "17px",
                                            "fontFamily": "Arial, sans-serif",
                                            "fontWeight": "700",
                                            "color": "white",
                                            "padding": "0",
                                            "margin": "0",
                                            "lineHeight": "1.5",
                                            "height": "auto",
                                        }
                                    ),
                                    style={"flex": "1", "minWidth": "0"}
                                ),
                                html.Button(
                                    "Send",
                                    id="chat-send",
                                    n_clicks=0,
                                    style={
                                        "padding": "10px 24px",
                                        "borderRadius": "999px",
                                        "background": INK,          # black
                                        "color": "white",
                                        "border": "none",
                                        "fontSize": "14px",
                                        "fontWeight": "700",
                                        "cursor": "pointer",
                                        "flexShrink": "0",
                                        "fontFamily": "Arial, sans-serif",
                                        "letterSpacing": "0.02em",
                                    }
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "12px",
                                "padding": "14px 18px",
                                "background": "#0070C0",  # solid blue composer area
                            }
                        ),
                    ],
                    style={
                        "background": "#eff6ff",          # sky-50, soft blue tint
                        "border": f"1px solid #bfdbfe",   # sky-200 — clearer panel edge
                        "borderRadius": "14px",
                        "overflow": "hidden",
                        "boxShadow": "0 1px 3px rgba(15, 23, 42, 0.04)",
                    }
                ),

                # Example chips — below the panel
                html.Div(
                    [
                        html.Div("Try asking:", style={
                            "fontSize": "12px",
                            "color": INK_SOFT,
                            "marginRight": "8px",
                            "fontFamily": "Arial, sans-serif",
                            "alignSelf": "center",
                        }),
                        html.Button(
                            "What's a normal degradation rate?",
                            id={"type": "chat-example", "idx": 0},
                            n_clicks=0,
                            style=_example_chip_style(),
                        ),
                        html.Button(
                            "Is my degradation rate normal?",
                            id={"type": "chat-example", "idx": 1},
                            n_clicks=0,
                            style=_example_chip_style(),
                        ),
                        html.Button(
                            "What does the clear-sky filter do?",
                            id={"type": "chat-example", "idx": 2},
                            n_clicks=0,
                            style=_example_chip_style(),
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "8px",
                        "marginTop": "14px",
                    }
                ),

                # Hidden — chat message store for multi-turn context
                dcc.Store(id="chat-history-store", data=[]),
                # Pending assistant reply being typed out (animated reveal)
                dcc.Store(id="chat-pending-store", data={"text": "", "shown": 0}),
                # Trigger store: signals when a new user question has been posted
                # and the LLM call should fire. Decoupling this from the submit
                # callback lets the browser repaint the user's question instantly.
                dcc.Store(id="chat-trigger-store", data={"question": "", "seq": 0}),
                # Captured key facts from each completed step — injected into LLM
                # system prompt so it can answer questions about the user's data.
                dcc.Store(id="chat-data-context", data={}),
                # Drives the typing animation
                dcc.Interval(id="chat-typer-interval", interval=20, disabled=True),
            ],
            id="agent-chat-wrap",
            style={
                "padding": "32px 64px 40px",
                "background": "transparent",
                "borderTop": f"1px solid {BORDER}",
                "marginTop": "32px",
                # scroll-margin-top compensates for any fixed header above
                # the scroll container so `scrollIntoView` doesn't tuck
                # this section's heading under it.  20px = visual breathing
                # room above the "Ask the Assistant" title.
                "scrollMarginTop": "20px",
            }
        ),
    ],
    style={
        "flex": "1",
        "minWidth": "0",
        "background": PAPER_RAISED,
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "height": "calc(100vh - 110px)",
        "overflowY": "auto",
    }
)


# =============================================================================
# FULL LAYOUT
# =============================================================================
layout = html.Div([

    # Hidden stores (unchanged)
    dcc.Store(id="mapped-vars-store",     data={}),
    dcc.Store(id="dataframe-store",       data={}),
    dcc.Store(id="dataframe-filtered",    data={}),
    dcc.Store(id="code-read-store",       data={}),
    dcc.Store(id="data-source-store",     data=None),
    dcc.Store(id="stored-data-file-name", data=None),
    # Tracks which example chip is currently "active" (the source of the
    # loaded dataset).  Values: "load-example-btn-1" | "load-example-btn-2"
    # | "load-example-btn-3" | None (cleared when the user uploads a file
    # or hasn't picked an example yet).  Drives the blue ring around the
    # active chip; the styling itself happens in a clientside callback.
    dcc.Store(id="selected-example-store", data=None),
    # NEW: holds the computed degradation rate & method so the chat can reference it
    dcc.Store(id="degradation-result-store", data={}),

    # NEW: track which steps are complete
    dcc.Store(id="step-progress", data={"data": False, "filter": False, "calc": False, "code": False}),

    # PVPRO long-running job tracker: holds {"job_id": "..."} when a fit is
    # running, {} when idle.
    dcc.Store(id="pvpro-job", data={}),

    # Disabled by default; the PVPRO branch of the degradation callback flips
    # it on, and the polling callback flips it off again when done.
    #
    # interval=1000ms (was 400ms): on Heroku, a faster poll starves the
    # PVPRO worker thread of the GIL.  scipy.least_squares releases the
    # GIL inside its C extension, but the Python-level fitting loop in
    # compute_pvpro has to reacquire it between scipy calls -- and if a
    # poll callback is sitting on the GIL too often each window's
    # wall-clock time balloons.  1s polls give the worker enough breathing
    # room while keeping the elapsed counter and progress numbers
    # advancing visibly every second.  (We tried 2s -- the worker is a
    # bit faster but the UI feels jumpy.)
    dcc.Interval(id="pvpro-poll-interval", interval=1000,
                 n_intervals=0, disabled=True),

    # Main container — sidebar + chat side-by-side, BOTH inside dbc.Container so
    # their edges line up with the LBNL/DuraMAT logos in the global header.
    dbc.Container(
        [
            html.Div(
                [
                    # Sidebar — sticky inside the flex shell
                    html.Div(id="sidebar-render", children=sidebar),
                    # Chat panel
                    chat_stream,
                ],
                className="pvcopilot-shell",
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "20px",
                    "background": "transparent",
                    "fontFamily": "Arial, sans-serif",
                    "color": INK,
                }
            ),

            # Page footer — below the shell, full width of the container
            html.Div(
                [
                    html.Hr(style={"border": "none", "borderTop": f"1px solid {BORDER}", "margin": "0 0 10px"}),
                    html.Div(
                        [
                            "Built at LBNL · Questions or feedback? ",
                            html.A("baojieli@lbl.gov",
                                href="mailto:baojieli@lbl.gov",
                                style={"color": MUTED,
                                       "textDecoration": "none",
                                       "fontWeight": "400"}
                            ),
                        ],
                        style={
                            "fontSize": "12px",
                            "color": MUTED,
                            "textAlign": "center",
                            "padding": "0 0 8px",
                            "fontFamily": "Arial, sans-serif",
                        }
                    ),
                ],
                style={"marginTop": "10px"}
            ),
        ],
        fluid=False,
        style={"paddingTop": "8px", "paddingBottom": "8px"}
    ),
],
className="pvcopilot-root",
)


# =============================================================================
# CLIENTSIDE SYNC — checkboxes -> hidden checklist (UNCHANGED)
# =============================================================================
app.clientside_callback(
    """
    function(tz, lip, out, cs) {
        var vals = [];
        if (tz)  vals.push("timezone");
        if (lip) vals.push("low-irra-power");
        if (out) vals.push("outlier");
        if (cs)  vals.push("clearsky");
        return vals;
    }
    """,
    Output("filter-options", "value"),
    Input("cb-timezone", "value"),
    Input("cb-low-irra-power", "value"),
    Input("cb-outlier", "value"),
    Input("cb-clearsky", "value"),
)


# =============================================================================
# CLIENTSIDE SYNC — two visible RadioItems -> one hidden master radio.
#
# The "Choose a metric" panel splits its options into two visible groups
# (statistical methods vs PVPRO).  We mirror whichever one the user
# touched most recently into the hidden `metric-selected-visible` radio,
# which is the source of truth read by every downstream callback.  We
# also clear the OTHER group so only one ever shows a selected dot,
# preserving single-select semantics.
# =============================================================================
app.clientside_callback(
    """
    function(statVal, pvproVal) {
        var triggered = dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) {
            // Initial firing -- whichever has a value wins; default to stat.
            return [statVal || pvproVal || "YOY", statVal, pvproVal];
        }
        var prop = triggered[0].prop_id;  // "metric-stat-radio.value" etc.
        if (prop.indexOf("metric-stat-radio") === 0 && statVal) {
            // Stat group picked -> clear PVPRO group, mirror stat.
            return [statVal, statVal, null];
        }
        if (prop.indexOf("metric-pvpro-radio") === 0 && pvproVal) {
            // PVPRO picked -> clear stat group, mirror PVPRO.
            return [pvproVal, null, pvproVal];
        }
        // Neither has a value: fall back to YoY.
        return ["YOY", "YOY", null];
    }
    """,
    Output("metric-selected-visible", "value"),
    Output("metric-stat-radio",       "value"),
    Output("metric-pvpro-radio",      "value"),
    Input("metric-stat-radio",  "value"),
    Input("metric-pvpro-radio", "value"),
)


# =============================================================================
# CALLBACK — UPLOAD STATUS  (UNCHANGED LOGIC, restyled output)
# =============================================================================
@app.callback(
    Output("upload-status-output", "children"),
    Output("data-source-store",    "data"),
    Output("data-summary-output",  "children"),
    Output("stored-data-file-name","data"),
    Input("upload-data", "filename"),
    prevent_initial_call=False
)
def update_upload_status(filename):
    if filename:
        msg = html.Div(
            [
                html.Span("✓", style={"color": SUCCESS, "marginRight": "8px", "fontWeight": "600"}),
                html.Span(f"File selected: ", style={"color": INK_SOFT, "fontSize": "14px"}),
                html.Span(filename, style={"color": INK, "fontSize": "14px", "fontWeight": "600", "fontFamily": "Arial, sans-serif"}),
            ],
            style={
                "padding": "10px 14px",
                "background": "#f0fdf4",
                "border": "1px solid #bbf7d0",
                "borderRadius": "8px",
                "fontSize": "14px",
            },
            className="slide-in-top",
        )
        return [msg, "upload", "", filename]
    return [
        html.Div("Awaiting file…",
                 style={"color": INK_SOFT, "fontSize": "13px",
                        "fontFamily": "Arial, sans-serif"}),
        None, "", None
    ]


# =============================================================================
# CALLBACK — FILTER  (UNCHANGED, only restyling output)
# =============================================================================
@app.callback(
    Output("data-filter-output",  "children"),
    Output("dataframe-filtered",  "data"),

    Input("filter-btn",          "n_clicks"),
    Input("upload-data",         "filename"),
    Input("load-example-btn-1",  "n_clicks"),
    Input("load-example-btn-2",  "n_clicks"),
    Input("load-example-btn-3",  "n_clicks"),

    State("filter-options",      "value"),
    State("mapped-vars-store",   "data"),
    State("dataframe-store",     "data"),
    State("param-gamma",         "value"),
    State("param-irr-thresh",    "value"),
    State("param-power-ratio",   "value"),
    State("param-norm-lower",    "value"),
    State("param-norm-upper-pct","value"),
    State("param-iqr-multiplier","value"),
    State("param-cs-smooth",     "value"),
    State("param-cs-energy",     "value"),

    prevent_initial_call=True
)
def run_filter(filter_clicks, upload_clicks,
        example1_clicks, example2_clicks, example3_clicks, selected_filters, mapped_variables_dict, df_json,
        gamma, irr_thresh, power_ratio, norm_lower, norm_upper_pct, iqr_multiplier,
        cs_smooth, cs_energy):

    trigger = ctx.triggered_id

    if not df_json:
        if trigger == "filter-btn":
            return [_no_data_alert("Please click 'Analyze Data' first to load your dataset before filtering."), None]
        return ["", None]

    if trigger == "upload-data" or (trigger and trigger.startswith("load-example-btn")):
        return ["", None]

    df = _df_from_store(df_json)
    irra_key = mapped_variables_dict["Irradiance"] if mapped_variables_dict else None
    if irra_key is None or irra_key not in df.columns:
        return ["❌ Irradiance column not found.", None]

    gamma          = gamma if gamma is not None else -0.004
    irr_thresh     = irr_thresh if irr_thresh is not None else 300
    power_ratio    = power_ratio if power_ratio is not None else 0.02
    norm_lower     = norm_lower if norm_lower is not None else 0.01
    norm_upper_pct = norm_upper_pct if norm_upper_pct is not None else 99

    # Basic value filter
    bv_normal, bv_outlier = basic_value_filter(df, mapped_variables_dict)
    df = df.loc[bv_normal].copy()

    clearsky_mask = pd.Series(True, index=df.index)
    if "clearsky" in selected_filters:
        cs_smooth = cs_smooth if cs_smooth is not None else 0.3
        cs_energy = cs_energy if cs_energy is not None else 0.5
        normal_idx, outlier_idx = clear_sky_filter(df, irra_key,
                                                    smoothness_threshold=cs_smooth,
                                                    energy_threshold=cs_energy)
        clearsky_mask = df.index.isin(normal_idx)

    df_filtered = normalize(df, mapped_variables_dict, gamma=gamma)
    current_mask = pd.Series(clearsky_mask, index=df_filtered.index)
    filter_stats = []

    if "timezone" in selected_filters:
        try:
            df_filtered.index = pd.to_datetime(df_filtered.index)
            df_filtered.index = df_filtered.index.tz_localize("UTC").tz_convert("US/Pacific")
            filter_stats.append("Timezone corrected (UTC → US/Pacific)")
        except Exception:
            filter_stats.append("⚠️ Timezone correction failed")

    if "clearsky" in selected_filters:
        removed = (~clearsky_mask).sum()
        filter_stats.append(f"Clear-sky filter removed {removed} points")

    if "low-irra-power" in selected_filters:
        normal_idx, outlier_idx = low_irra_power_filter(
            df_filtered, mapped_variables_dict,
            irr_thresh=irr_thresh, power_ratio=power_ratio,
            norm_lower=norm_lower, norm_upper_pct=norm_upper_pct
        )
        mask = df_filtered.index.isin(normal_idx)
        removed = (~mask & current_mask).sum()
        current_mask &= mask
        filter_stats.append(f"Low irra-power filter removed {removed} points")

    if "outlier" in selected_filters:
        iqr_multiplier = iqr_multiplier if iqr_multiplier is not None else 1.5
        normal_idx, outlier_idx = identify_outliers_iqr(df_filtered, "norm", iqr_multiplier=iqr_multiplier)
        mask = df_filtered.index.isin(normal_idx)
        removed = (~mask & current_mask).sum()
        current_mask &= mask
        filter_stats.append(f"IQR outlier filter removed {removed} points")

    normal_indices  = df_filtered.index[current_mask]
    outlier_indices = df_filtered.index[~current_mask]

    n_total = len(df_filtered)
    n_good  = len(normal_indices)
    n_bad   = len(outlier_indices)

    # Pie chart
    pie_fig = go.Figure(data=[go.Pie(
        labels=["High-quality", "Filtered"],
        values=[n_good, n_bad],
        hole=0.62,
        marker=dict(colors=[INDIGO, FILTERED_COLOR]),
        textinfo="percent",
        hoverinfo="label+percent",
    )])
    pie_fig.update_layout(
        height=180,
        margin=dict(t=20, b=20, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, font=dict(size=13, family="Arial")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=INK),
    )

    # Scatter plot
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scattergl(
        x=df_filtered.loc[outlier_indices].index,
        y=df_filtered.loc[outlier_indices]["norm"],
        mode="markers",
        marker=dict(size=4, opacity=0.35, color=FILTERED_COLOR),
        name="Filtered"
    ))
    scatter_fig.add_trace(go.Scattergl(
        x=df_filtered.loc[normal_indices].index,
        y=df_filtered.loc[normal_indices]["norm"],
        mode="markers",
        marker=dict(size=4, opacity=0.55, color=INDIGO),
        name="High-quality"
    ))
    scatter_fig.update_layout(
        title=dict(text="Normalized Power Over Time", font=dict(family="Arial", size=18, color=INK), x=0, xanchor="left"),
        xaxis_title="Time",
        yaxis_title="Normalized Power",
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=60),
        height=320,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=13, family="Arial")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=INK),
    )
    scatter_fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    scatter_fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)

    # Editorial-styled summary
    pct_good = n_good / n_total if n_total else 0
    summary_block = html.Div([
        html.Div("filtering result", style={
            "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
            "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "8px",
            "fontFamily": "Arial, sans-serif",
        }),
        # Headline percentage -- this is the "major" featured number for
        # this step, so it gets the major-value blue.
        html.Div(f"{pct_good:.1%}", style={
            "fontSize": "44px",
            "fontFamily": "Arial, sans-serif",
            "fontWeight": "700",
            "color": VALUE_MAJOR,
            "lineHeight": "1",
            "marginBottom": "4px",
        }),
        html.Div("high-quality points retained", style={
            "fontSize": "15px", "color": INK_SOFT,
            "fontFamily": "Arial, sans-serif",
            "fontStyle": "italic", "marginBottom": "12px",
        }),
        html.Div([
            # All three supporting counts are detail values -- plain dark
            # text.  The headline percentage above is the sole blue number.
            html.Div(
                [html.Span("Total: ", style={"color": INK_SOFT}),
                 html.B(f"{n_total:,}", style={"color": VALUE_DETAIL})],
                style={"fontSize": "14px", "marginBottom": "3px"},
            ),
            html.Div(
                [html.Span("Retained: ", style={"color": INK_SOFT}),
                 html.B(f"{n_good:,}", style={"color": VALUE_DETAIL})],
                style={"fontSize": "14px", "marginBottom": "3px"},
            ),
            html.Div(
                [html.Span("Filtered: ", style={"color": INK_SOFT}),
                 html.B(f"{n_bad:,}", style={"color": VALUE_DETAIL})],
                style={"fontSize": "14px"},
            ),
        ], style={"fontFamily": "Arial, sans-serif"}),
        html.Details([
            html.Summary("Show details", style={"color": INK_SOFT, "cursor": "pointer", "fontSize": "13px", "marginTop": "10px", "fontFamily": "Arial, sans-serif"}),
            html.Ul(
                [html.Li(s, style={"fontSize": "13px", "color": INK_SOFT, "marginBottom": "2px"}) for s in filter_stats],
                style={"marginTop": "6px", "paddingLeft": "16px"}
            )
        ]),
    ])

    filter_layout = html.Div([
        html.Div([
            html.Div(summary_block, style={"flex": "1", "minWidth": "180px"}),
            html.Div(dcc.Graph(figure=pie_fig, config={"displayModeBar": False}), style={"flex": "1", "minWidth": "240px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "16px"}),
        # Trend figure sits inside its own white rounded card, matching the
        # per-panel cards on the PVPRO results page.
        html.Div(
            dcc.Graph(figure=scatter_fig, config={"displayModeBar": False}),
            style={
                "background": "#ffffff",
                "border": f"1px solid {BORDER}",
                "borderRadius": "10px",
                "padding": "10px 12px",
                "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
            },
        ),
    ], className="slide-in-up", style={
        "padding": "20px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "marginTop": "16px",
    })

    df_filtered_store = df_filtered.loc[normal_indices]
    return [filter_layout, df_filtered_store.to_json(date_format="iso", orient="split")]


# =============================================================================
# CALLBACK — DEGRADATION (UNCHANGED logic, restyled output)
# =============================================================================
@app.callback(
    Output("degradation-output", "children", allow_duplicate=True),
    Output("pvpro-progress-output", "children", allow_duplicate=True),
    Output("run-btn", "disabled",  allow_duplicate=True),
    Output("run-btn", "children",  allow_duplicate=True),
    Output("degradation-result-store", "data"),
    Output("pvpro-job", "data", allow_duplicate=True),
    Output("pvpro-poll-interval", "disabled", allow_duplicate=True),

    Input("run-btn",              "n_clicks"),
    Input("upload-data",          "filename"),
    Input("load-example-btn-1",   "n_clicks"),
    Input("load-example-btn-2",   "n_clicks"),
    Input("load-example-btn-3",   "n_clicks"),

    State("dataframe-filtered",      "data"),
    State("mapped-vars-store",       "data"),
    State("metric-selected-visible", "value"),
    State("param-yoy-window",        "value"),
    State("param-yoy-iqr",           "value"),
    State("param-hw-period",         "value"),
    State("param-arima-p",           "value"),
    State("param-arima-d",           "value"),
    State("param-arima-q",           "value"),
    State("param-arima-s",           "value"),
    State("param-csd-period",        "value"),
    State("param-pvpro-cells",       "value"),
    State("param-pvpro-mps",         "value"),
    State("param-pvpro-ps",          "value"),
    State("param-pvpro-alphaisc",    "value"),
    State("param-pvpro-tech",        "value"),
    State("param-pvpro-days",        "value"),
    State("param-pvpro-iters",       "value"),

    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        degradation_clicks, upload_clicks,
        example1_clicks, example2_clicks, example3_clicks,
        df_filtered_json, mapped_variables_dict, selected_metric,
        yoy_window, yoy_iqr, hw_period,
        arima_p, arima_d, arima_q, arima_s, csd_period,
        pvpro_cells, pvpro_mps, pvpro_ps, pvpro_alphaisc,
        pvpro_tech, pvpro_days, pvpro_iters):

    trigger = ctx.triggered_id

    if trigger in ["load-example-btn-1", "load-example-btn-2", "load-example-btn-3", "upload-data"]:
        return ["", "", False, "Calculate Degradation", {}, {}, True]

    if not df_filtered_json:
        if trigger == "run-btn":
            return [_no_data_alert("Please apply filters first before running degradation analysis."),
                    "", False, "Calculate Degradation", {}, {}, True]
        return ["", "", False, "Calculate Degradation", {}, {}, True]

    df_filtered = _df_from_store(df_filtered_json)
    irra_key = mapped_variables_dict["Irradiance"] if mapped_variables_dict else None
    if irra_key is None or irra_key not in df_filtered.columns:
        return ["❌ Irradiance column not found.", "", False, "Calculate Degradation", {}, {}, True]

    # ---------- PVPRO: long-running, so launch in a thread and let a polling
    # ---------- callback render the result when it's ready.
    if selected_metric == "PVPRO":
        # Snapshot all the user-controlled params into kwargs.
        pvpro_kwargs = dict(
            cells_in_series   = pvpro_cells     if pvpro_cells     else 60,
            modules_per_string= pvpro_mps       if pvpro_mps       else 1,
            parallel_strings  = pvpro_ps        if pvpro_ps        else 1,
            alpha_isc         = pvpro_alphaisc  if pvpro_alphaisc is not None else 0.0046,
            technology        = pvpro_tech      if pvpro_tech      else "mono-c-Si",
            days_per_run      = pvpro_days      if pvpro_days      else 14,
            iterations_per_year = pvpro_iters   if pvpro_iters     else 12,
        )

        job_id = _pvpro_make_job()

        def _progress_cb(stage, current, total, message, _jid=job_id):
            _pvpro_update_job(_jid, phase=stage, current=current,
                              total=total, message=message)

        def _worker(_df=df_filtered, _mapping=mapped_variables_dict,
                    _kwargs=pvpro_kwargs, _jid=job_id, _cb=_progress_cb):
            try:
                rd, figs, rates = compute_pvpro(_df, _mapping,
                                                progress_callback=_cb, **_kwargs)
                # Log AFTER compute_pvpro returns successfully but BEFORE
                # we serialize the (potentially heavy) figs dict into the
                # job store.  If the worker dies between the fitting loop
                # and the "done" update -- e.g. OOM-killed by Heroku
                # because the figs accumulator pushed us over the dyno's
                # memory limit -- this event will be the last thing in
                # the debug log, making the cause obvious.
                _pvpro_update_job(_jid, phase="finalising",
                                  message="Packing results…")
                _pvpro_update_job(
                    _jid, phase="done",
                    result={"rd": float(rd), "figs": figs, "rates": rates},
                    message="Done",
                )
            except Exception as exc:
                _pvpro_update_job(_jid, phase="error",
                                  error=f"{type(exc).__name__}: {exc}",
                                  message=str(exc))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Initial progress UI: status block + debug panel. The polling
        # Interval lives in the page layout (initially disabled); we
        # enable it here.
        initial_ui = html.Div([
            _pvpro_progress_ui(
                phase="starting", current=0, total=1,
                message="Spinning up PVPRO worker…", elapsed_s=0,
            ),
            _pvpro_debug_panel(current_job_id=job_id),
        ])
        # Clear any previous fast-method output, send the progress UI to its
        # own (Loading-free) container, disable the run button, enable poll.
        return ["", initial_ui, True, "Running PVPRO…",
                {}, {"job_id": job_id}, False]

    else:
        daily_data = aggregate_daily(df_filtered, irra_key)

        if selected_metric == "YOY":
            rd, fig = compute_yoy(daily_data,
                                  rolling_window=yoy_window if yoy_window else 30,
                                  iqr_multiplier=yoy_iqr if yoy_iqr else 1.5)
        elif selected_metric == "LR":
            rd, fig = compute_lr(daily_data)
        elif selected_metric == "HW":
            rd, fig = compute_hw(daily_data, period=hw_period if hw_period else 12)
        elif selected_metric == "ARIMA":
            rd, fig = compute_arima(daily_data,
                                    p=arima_p if arima_p is not None else 1,
                                    d=arima_d if arima_d is not None else 1,
                                    q=arima_q if arima_q is not None else 0,
                                    seasonal_period=arima_s if arima_s else 12)
        elif selected_metric == "CSD":
            rd, fig = compute_csd(daily_data, period=csd_period if csd_period else 12)
        else:
            raise ValueError(f"Unknown metric: {selected_metric}")

    # Restyle the figure
    if fig is not None:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", color=INK),
            margin=dict(l=50, r=20, t=50, b=60),
            title=dict(font=dict(family="Arial", size=18, color=INK), x=0, xanchor="left"),
            height=340,
        )
        fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)

    start_date = df_filtered.index.min()
    end_date   = df_filtered.index.max()
    duration_years = (end_date - start_date).days / 365.25

    # Editorial summary with HUGE rate display
    rate_pct = rd / 100
    # Single unified rate color across methods (major-value blue).
    rate_color = VALUE_MAJOR

    summary_block = html.Div([
        html.Div("annual degradation rate", style={
            "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
            "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
            "fontFamily": "Arial, sans-serif",
        }),
        html.Div([
            html.Span(f"{rate_pct:.2%}", style={
                "fontSize": "56px",
                "fontFamily": "Arial, sans-serif",
                "fontWeight": "700",
                "color": rate_color,
                "lineHeight": "1",
            }),
            html.Span("/year", style={
                "fontSize": "20px",
                "color": INK_SOFT,
                "marginLeft": "8px",
                "fontFamily": "Arial, sans-serif",
                "fontStyle": "italic",
            }),
        ], style={"marginBottom": "16px"}),
        html.Div([
            # All three are supporting/detail values -- plain dark text.
            html.Div([html.Span("Method: ",   style={"color": INK_SOFT}),
                      html.B(selected_metric, style={"color": VALUE_DETAIL})],
                     style={"fontSize": "14px", "marginBottom": "3px"}),
            html.Div([html.Span("Duration: ", style={"color": INK_SOFT}),
                      html.B(f"{duration_years:.1f} years",
                             style={"color": VALUE_DETAIL})],
                     style={"fontSize": "14px", "marginBottom": "3px"}),
            html.Div([html.Span("Window: ",   style={"color": INK_SOFT}),
                      html.B(f"{start_date.strftime('%Y-%m-%d') if hasattr(start_date,'strftime') else start_date}  →  {end_date.strftime('%Y-%m-%d') if hasattr(end_date,'strftime') else end_date}",
                            style={"fontFamily": "Arial, sans-serif",
                                   "fontSize": "13px", "color": VALUE_DETAIL})],
                     style={"fontSize": "14px"}),
        ], style={"fontFamily": "Arial, sans-serif"}),
    ])

    degradation_layout = html.Div([
        html.Div(summary_block, style={"marginBottom": "20px"}),
        html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False})),
    ], className="slide-in-up", style={
        "padding": "20px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "marginTop": "16px",
    })

    result_dict = {
        "rate_pct_per_year": round(float(rate_pct) * 100, 4),
        "method": selected_metric,
        "duration_years": round(float(duration_years), 2),
        "start": start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
        "end":   end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
    }
    # Write to degradation-output (under Loading), clear pvpro-progress-output.
    return [degradation_layout, "", False, "Calculate Degradation",
            result_dict, {}, True]


# =============================================================================
# CALLBACK — PVPRO progress polling
#
# While `compute_pvpro` runs in a background thread, this callback fires every
# ~400ms, reads the latest progress from _PVPRO_JOBS, and re-renders the
# progress bar in `pvpro-progress-output`.  This output lives OUTSIDE the
# dcc.Loading wrapper so the spinner does not overlay the progress bar on
# every tick.  When the worker finishes (`phase == "done"`) it pulls the
# result off the job and replaces the progress UI with the final summary +
# multi-panel figure.  At that point the Interval is disabled, so polling
# naturally stops.
# =============================================================================
@app.callback(
    Output("pvpro-progress-output", "children", allow_duplicate=True),
    Output("run-btn", "disabled",  allow_duplicate=True),
    Output("run-btn", "children",  allow_duplicate=True),
    Output("degradation-result-store", "data", allow_duplicate=True),
    Output("pvpro-job", "data", allow_duplicate=True),
    Output("pvpro-poll-interval", "disabled", allow_duplicate=True),
    Input("pvpro-poll-interval", "n_intervals"),
    State("pvpro-job", "data"),
    State("dataframe-filtered", "data"),
    prevent_initial_call=True,
)
def _pvpro_poll_callback(_n, job_store, df_filtered_json):
    from dash import no_update

    job_id = (job_store or {}).get("job_id")
    if not job_id:
        # No active job_id in the store, which means a previous poll's
        # response already cleared it -- the run is done and the UI is
        # already on screen.  Return all-no_update so this late tick (the
        # browser had queued it before disabled=True propagated) cannot
        # clobber the freshly-rendered final UI.  If we returned concrete
        # values like "Calculate Degradation" for the button, those would
        # race against the winner's response and potentially overwrite
        # only some of the six outputs (leaving pvpro-progress-output
        # stuck on the progress bar -- the bug we saw in production).
        return [no_update, no_update, no_update,
                no_update, no_update, no_update]

    job = _pvpro_read_job(job_id)
    if job is None:
        # Could be either:
        #   (1) genuine multi-worker bug -- the worker that created the job
        #       is different from the one handling this poll, and diskcache
        #       isn't bridging them.  Symptom: the user never saw progress
        #       advance, so the orange "lost" alert is helpful.
        #   (2) a benign race after a successful render -- the browser had
        #       already queued the next poll tick by the time the Interval's
        #       disable=True propagated back, so a late poll arrives AFTER
        #       _pvpro_drop_job() has run.  This is harmless and used to
        #       clobber the just-rendered result with the orange alert.
        #
        # We disambiguate using the debug log: if THIS worker has ever seen
        # this job_id (any event mentioning it), the None means "job was
        # just dropped" -- race case, all-no_update.  Otherwise it's the
        # smoking-gun multi-worker bug, show the orange banner.
        prefix = job_id[:8]
        seen_here = any(e.get("job_id") == prefix
                        for e in _pvpro_debug_snapshot())
        if seen_here:
            # Race after success/error: silently idle, do NOT touch any
            # output.  See the comment on the not-job_id early-out above
            # for why all-no_update is essential here.
            return [no_update, no_update, no_update,
                    no_update, no_update, no_update]

        # Genuine multi-worker bug -- show the diagnostic banner alongside
        # the debug panel so the user can inspect what happened.
        lost_ui = html.Div([
            html.Div(
                "PVPRO progress lost — this worker has never seen the job "
                "that was started. If you're on a multi-worker deployment, "
                "enable diskcache (set PVPRO_DISKCACHE_DIR + install "
                "diskcache). Expand the Debug panel below for details.",
                style={
                    "padding": "12px 14px",
                    "background": "#fff7ed",
                    "border": "1px solid #fed7aa",
                    "borderRadius": "10px",
                    "color": "#7c2d12",
                    "fontSize": "13px",
                    "fontFamily": "Arial, sans-serif",
                },
            ),
            _pvpro_debug_panel(current_job_id=job_id),
        ])
        return [lost_ui, False, "Calculate Degradation",
                no_update, {}, True]

    phase = job.get("phase", "")
    elapsed = max(0.0, time.time() - job.get("started_at", time.time()))

    # --- Still working ---
    # Phases that mean "PVPRO worker is still going": anything other than
    # the terminal states (done, error, rendered).  "rendered" means a
    # previous poll already rendered the final UI; we still re-render it
    # below (cheap belt-and-suspenders) so that a race between the done
    # branch and a late poll never leaves the user looking at an old
    # progress bar.
    if phase not in ("done", "error", "rendered"):
        ui = html.Div([
            _pvpro_progress_ui(
                phase=phase,
                current=job.get("current", 0),
                total=job.get("total", 1),
                message=job.get("message", ""),
                elapsed_s=elapsed,
            ),
            _pvpro_debug_panel(current_job_id=job_id),
        ])
        return [ui, True, "Running PVPRO…", no_update, job_store, False]

    # --- Failed ---
    if phase == "error":
        # Mark rendered instead of hard-dropping so a late poll doesn't
        # clobber the error UI we're about to render.
        _pvpro_update_job(job_id, phase="rendered")
        return [
            html.Div([
                _no_data_alert(f"PVPRO failed: {job.get('error', 'unknown error')}"),
                _pvpro_debug_panel(current_job_id=job_id),
            ]),
            False, "Calculate Degradation", {}, {}, True,
        ]

    # --- Done OR already-rendered: render the final layout under a per-job
    # lock to prevent concurrent polls from racing into the same render
    # path.  The first thread in builds the UI and marks phase="rendered";
    # all subsequent threads either find phase="rendered" inside the lock
    # (and return no_update — the UI is already on screen) or skip with a
    # non-blocking try-acquire that fails (also returns no_update).
    # Either way only ONE thread does the expensive final_layout build,
    # which eliminates the "stuck at 99%" timing race.
    render_lock = _pvpro_get_render_lock(job_id)
    acquired = render_lock.acquire(blocking=False)
    if not acquired:
        # Another thread is currently rendering this job.  We can safely
        # idle this poll out -- the other thread WILL write the final UI
        # and set disabled=True.  CRITICALLY, we return no_update for
        # ALL SIX outputs, not just the children.  If we returned any
        # concrete value (e.g. button text = "Calculate Degradation",
        # interval-disabled = True), this late response could land at the
        # browser AFTER the winner's response and Dash's "last response
        # wins" semantics would apply our no_update children alongside
        # those concrete values -- the user would see button reset and
        # polling stopped BUT pvpro-progress-output still stuck on the
        # old progress bar (no_update means "keep previous value").
        # All-no_update means Dash applies nothing from this response,
        # so the winner's full UI sticks regardless of arrival order.
        _pvpro_debug("done_branch_skip",
                     job_id=job_id[:8], reason="render-locked")
        return [no_update, no_update, no_update,
                no_update, no_update, no_update]

    try:
        # Re-read inside the lock.  If a different thread already marked
        # this job "rendered", the UI is already on screen -- this is a
        # late poll, return all-no_update (same reasoning as above).
        job = _pvpro_read_job(job_id)
        if job is None or job.get("phase") == "rendered":
            _pvpro_debug("done_branch_skip",
                         job_id=job_id[:8],
                         reason=("job-gone" if job is None
                                 else "already-rendered"))
            return [no_update, no_update, no_update,
                    no_update, no_update, no_update]

        result = job.get("result") or {}
        rd = result.get("rd", float("nan"))
        figs = result.get("figs") or {}
        rates = result.get("rates", {}) or {}

        # Recover the window dates from the filtered dataframe (same logic as
        # the synchronous path).
        try:
            df_f = _df_from_store(df_filtered_json)
            start_date = df_f.index.min()
            end_date   = df_f.index.max()
            duration_years = (end_date - start_date).days / 365.25
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str   = end_date.strftime('%Y-%m-%d')   if hasattr(end_date,   'strftime') else str(end_date)
        except Exception:
            start_str = end_str = "—"
            duration_years = 0.0

        rate_pct = rd / 100 if np.isfinite(rd) else 0.0
        # Major-value blue for the headline rate (consistent across methods).
        rate_color = VALUE_MAJOR

        # Per-quantity rates panel (the "Pmp, Vmp, Imp, Voc, Isc" table).
        QUANT_LABELS = [
            ("p_mp_ref", "Pmp", "max power point power"),
            ("v_mp_ref", "Vmp", "max power point voltage"),
            ("i_mp_ref", "Imp", "max power point current"),
            ("v_oc_ref", "Voc", "open-circuit voltage"),
            ("i_sc_ref", "Isc", "short-circuit current"),
        ]
        rates_rows = []
        for key, short, descr in QUANT_LABELS:
            r = rates.get(key, float("nan"))
            if not np.isfinite(r):
                r_str = "n/a"
            else:
                r_str = f"{r:+.2f}%/yr"
            # Label cell:  <b>Pmp</b> (ref)
            label_cell = html.Td(
                [html.B(short, style={"fontFamily": "Arial, sans-serif"}),
                 html.Span(" (ref)", style={
                     "color": INK_SOFT, "fontSize": "13px",
                     "fontFamily": "Arial, sans-serif",
                 })],
                style={"padding": "4px 12px 4px 0", "whiteSpace": "nowrap"},
            )
            rates_rows.append(html.Tr([
                label_cell,
                html.Td(descr, style={"padding": "4px 12px 4px 0",
                                      "color": INK_SOFT, "fontSize": "13px",
                                      "fontFamily": "Arial, sans-serif"}),
                # Detail-value (per-row supporting number, not the headline).
                html.Td(r_str, style={"padding": "4px 0", "color": VALUE_DETAIL,
                                      "fontWeight": "700", "textAlign": "right",
                                      "fontFamily": "Arial, sans-serif"}),
            ]))
        rates_table = html.Div([
            html.Div("parameter degradation rates", style={
                "fontSize": "12px", "color": INK_SOFT,
                "textTransform": "uppercase", "letterSpacing": "0.1em",
                "fontWeight": "600", "marginBottom": "8px",
                "fontFamily": "Arial, sans-serif",
            }),
            html.Table(html.Tbody(rates_rows), style={
                "width": "100%", "fontSize": "14px",
                "borderCollapse": "collapse",
            }),
        ], style={"marginTop": "14px", "marginBottom": "16px"})

        summary_block = html.Div([
            html.Div("annual degradation rate (Pmp, ref)", style={
                "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
                "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
                "fontFamily": "Arial, sans-serif",
            }),
            html.Div([
                html.Span(f"{rate_pct:.2%}", style={
                    "fontSize": "56px", "fontFamily": "Arial, sans-serif",
                    "fontWeight": "700", "color": rate_color, "lineHeight": "1",
                }),
                html.Span("/year", style={
                    "fontSize": "20px", "color": INK_SOFT, "marginLeft": "8px",
                    "fontFamily": "Arial, sans-serif", "fontStyle": "italic",
                }),
            ], style={"marginBottom": "10px"}),
            html.Div([
                # Detail (supporting) numbers -- plain dark text.
                html.Div([html.Span("Method: ",   style={"color": INK_SOFT}),
                          html.B("PVPRO", style={"color": VALUE_DETAIL})],
                         style={"fontSize": "14px", "marginBottom": "3px"}),
                html.Div([html.Span("Duration: ", style={"color": INK_SOFT}),
                          html.B(f"{duration_years:.1f} years",
                                 style={"color": VALUE_DETAIL})],
                         style={"fontSize": "14px", "marginBottom": "3px"}),
                html.Div([html.Span("Window: ",   style={"color": INK_SOFT}),
                          html.B(f"{start_str}  →  {end_str}",
                                 style={"fontFamily": "Arial, sans-serif",
                                        "fontSize": "13px",
                                        "color": VALUE_DETAIL})],
                         style={"fontSize": "14px"}),
                html.Div([html.Span("Elapsed: ",  style={"color": INK_SOFT}),
                          html.B(f"{int(elapsed)} s",
                                 style={"color": VALUE_DETAIL})],
                         style={"fontSize": "13px", "marginTop": "4px"}),
            ], style={"fontFamily": "Arial, sans-serif"}),
            rates_table,
        ])

        # ---------------------------------------------------------------
        # Build the figure grid.  Each Plotly figure is wrapped in its OWN
        # rounded-corner card div (white background, subtle border).  Using
        # HTML cards instead of Plotly shapes guarantees axis tick labels
        # don't overlap the background, and lets us tune row heights with CSS.
        #
        # Layout (CSS Grid):
        #   row 1:  [          Pmp (spans 2 cols)          ]
        #   row 2:  [   Vmp    ]   [   Imp    ]
        #   row 3:  [   Voc    ]   [   Isc    ]
        # ---------------------------------------------------------------
        card_style = {
            "background": "#ffffff",
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "padding": "8px 10px",
            "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
        }

        def _fig_card(key, span_cols=False):
            f = figs.get(key)
            if f is None:
                return html.Div()
            style = dict(card_style)
            if span_cols:
                style["gridColumn"] = "1 / -1"   # span both columns
            return html.Div(
                dcc.Graph(figure=f, config={"displayModeBar": False}),
                style=style,
            )

        fig_grid = html.Div(
            [
                _fig_card("p_mp_ref", span_cols=True),
                _fig_card("v_mp_ref"),
                _fig_card("i_mp_ref"),
                _fig_card("v_oc_ref"),
                _fig_card("i_sc_ref"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "10px",
                "marginTop": "8px",
            },
        )

        # Section heading above the figure grid — matches the rates-table
        # heading style for consistency.
        fig_grid_heading = html.Div("pvpro-lite degradation trends", style={
            "fontSize": "12px", "color": INK_SOFT,
            "textTransform": "uppercase", "letterSpacing": "0.1em",
            "fontWeight": "600", "marginBottom": "6px", "marginTop": "8px",
            "fontFamily": "Arial, sans-serif",
        })

        final_layout = html.Div([
            html.Div(summary_block, style={"marginBottom": "20px"}),
            fig_grid_heading,
            fig_grid,
        ], className="slide-in-up", style={
            "padding": "20px",
            "background": "#f8fafc",
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "marginTop": "16px",
        })

        result_dict = {
            "rate_pct_per_year": round(float(rate_pct) * 100, 4)
                if np.isfinite(rate_pct) else None,
            "method": "PVPRO",
            "duration_years": round(float(duration_years), 2),
            "start": start_str,
            "end":   end_str,
            "rates_per_quantity": {k: round(float(v), 4)
                                   for k, v in rates.items()
                                   if v is not None and np.isfinite(v)},
        }
        # Mark 'rendered' AS LATE AS POSSIBLE -- only once the full final_layout
        # has been built.  If we marked earlier, a concurrent poll could see
        # phase="rendered" mid-construction and take an early code path that
        # competes with this branch (the race that caused the 96%-stuck bug).
        _pvpro_update_job(job_id, phase="rendered")
        return [html.Div([final_layout, _pvpro_debug_panel(current_job_id=job_id)]),
                False, "Calculate Degradation",
                result_dict, {}, True]
    finally:
        # Always release the lock, even if rendering raised.  If we
        # don't release, every future poll for this job_id will
        # silently no_update and the user will never see the result.
        render_lock.release()


# =============================================================================
# CALLBACKS — Example chip "selected" highlight
#
# When the user clicks one of the three example chips, we want the
# clicked chip to gain a blue ring and keep it until the user either
# (a) clicks a DIFFERENT example chip (ring moves to the new one) or
# (b) uploads a file of their own (rings clear from all chips).
#
# Two callbacks:
#   1. selected-example-store gets written whenever an example chip is
#      clicked (set to that chip's id) or an upload arrives (set to
#      None).
#   2. A single clientside callback reads selected-example-store and
#      rewrites all three chips' style objects, applying the active
#      style to the matching chip and the resting style to the others.
# =============================================================================

@app.callback(
    Output("selected-example-store", "data", allow_duplicate=True),
    Input("load-example-btn-1", "n_clicks"),
    Input("load-example-btn-2", "n_clicks"),
    Input("load-example-btn-3", "n_clicks"),
    Input("upload-data", "contents"),
    prevent_initial_call=True,
)
def track_selected_example(*_):
    trigger = ctx.triggered_id
    if trigger in ("load-example-btn-1", "load-example-btn-2",
                   "load-example-btn-3"):
        return trigger
    # An upload happened; the user is no longer using an example dataset.
    return None


# Style-update clientside callback.  We build the two style dicts in JS
# (rather than relying on a CSS class) so the look is self-contained --
# no external stylesheet to keep in sync.  The "active" look is a 2px
# blue ring + faint blue tint; the "resting" look is the original
# transparent + neutral grey border.
app.clientside_callback(
    """
    function(selected) {
        var resting = {
            "padding": "8px 14px",
            "background": "transparent",
            "border": "1px solid #cbd5e1",
            "borderRadius": "8px",
            "fontSize": "14px",
            "color": "#0f172a",
            "cursor": "pointer",
            "marginRight": "8px",
            "fontFamily": "Arial, sans-serif",
            "fontWeight": "500",
            "transition": "border-color 0.15s ease, background 0.15s ease"
        };
        var active = Object.assign({}, resting, {
            "border": "2px solid #0064AB",
            "background": "#eff6ff",
            "padding": "7px 13px"   // -1px to keep total width with 2px border
        });
        // Third chip has no right-margin in the layout -- mirror that.
        var resting3 = Object.assign({}, resting); delete resting3.marginRight;
        var active3  = Object.assign({}, active);  delete active3.marginRight;

        return [
            selected === "load-example-btn-1" ? active  : resting,
            selected === "load-example-btn-2" ? active  : resting,
            selected === "load-example-btn-3" ? active3 : resting3
        ];
    }
    """,
    [Output("load-example-btn-1", "style"),
     Output("load-example-btn-2", "style"),
     Output("load-example-btn-3", "style")],
    Input("selected-example-store", "data"),
)


# =============================================================================
# CALLBACKS — clientside button state machines
#
# Each step's primary action button needs to flip into a "working" state
# the instant the user clicks, then back to the idle state once the
# corresponding Python callback finishes.  We use clientside callbacks
# for the "working" half (so the UI feels instant -- no round-trip to
# the server) and let the Python callbacks supply the "done" half via
# allow_duplicate Outputs.
#
# Three buttons:
#   * analyze-btn  (Step 1): "Analyze Data" -> "Uploading data..." when a
#     new file/example is loading, OR -> "Analyzing..." when the user
#     hits Analyze on an already-loaded dataset.  Both states disable
#     the button.
#   * filter-btn   (Step 2): "Apply Filters" -> "Applying filters..."
#   * run-btn      (Step 4): "Calculate Degradation" -> "Calculating…"
# =============================================================================

# Step 4 -- Calculate / run-btn.  Same as before; no upload-state to
# worry about because by Step 4 the dataset is already loaded.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks === 0) {
            return [false, "Calculate Degradation"];
        }
        return [true, "Calculating…"];
    }
    """,
    [Output("run-btn", "disabled"), Output("run-btn", "children")],
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True
)

# Step 2 -- filter-btn.  Goes "Applying filters..." disabled the moment
# the user clicks; the existing run_filter Python callback returns
# results which (via allow_duplicate=True on the duplicate-Output
# callback below) flips it back to "Apply Filters" enabled.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks === 0) {
            return [false, "Apply Filters"];
        }
        return [true, "Applying filters..."];
    }
    """,
    [Output("filter-btn", "disabled"), Output("filter-btn", "children")],
    Input("filter-btn", "n_clicks"),
    prevent_initial_call=True
)

# Step 4 -- generate-code-btn ("Generate Full Python Code").  Same idea:
# the moment the user clicks, show "Generating code..." and disable so
# the user can't double-trigger the (intentionally slow, 2s+) code build.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks === 0) {
            return [false, "Generate Full Python Code"];
        }
        return [true, "Generating code..."];
    }
    """,
    [Output("generate-code-btn", "disabled"),
     Output("generate-code-btn", "children")],
    Input("generate-code-btn", "n_clicks"),
    prevent_initial_call=True
)

# Reset generate-code-btn back to the idle state once the code preview
# has landed in the DOM (i.e. the Python `generate_code` callback has
# finished writing `code-preview.children`).
app.clientside_callback(
    """
    function(_preview_children) {
        return [false, "Generate Full Python Code"];
    }
    """,
    [Output("generate-code-btn", "disabled",  allow_duplicate=True),
     Output("generate-code-btn", "children",  allow_duplicate=True)],
    Input("code-preview", "children"),
    prevent_initial_call=True
)

# When filtering finishes -- detectable as a non-null `dataframe-filtered`
# write OR an error rendered into `data-filter-output` -- restore the
# Apply Filters button to its idle state.  Triggering off the OUTPUT of
# run_filter sidesteps having to modify run_filter's signature.
app.clientside_callback(
    """
    function(_filtered_data, _output_children) {
        // Either output landing means the filter callback has finished.
        return [false, "Apply Filters"];
    }
    """,
    [Output("filter-btn", "disabled",  allow_duplicate=True),
     Output("filter-btn", "children",  allow_duplicate=True)],
    Input("dataframe-filtered",  "data"),
    Input("data-filter-output",  "children"),
    prevent_initial_call=True
)

# Step 1 -- analyze-btn.  Three trigger paths:
#   1. User clicks one of the example chips -> "Uploading data..."
#   2. A file lands in upload-data.contents  -> "Uploading data..."
#   3. User clicks Analyze on a loaded set   -> "Analyzing..."
# The Python `analyze_uploaded_data_callback` resets the button at the
# end of (1) and (3); for (2) we additionally reset on a non-null
# data-source-store write below.
app.clientside_callback(
    """
    function(analyze_n, ex1_n, ex2_n, ex3_n, upload_contents) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) {
            return [false, "Analyze Data"];
        }
        var trigger = ctx.triggered[0].prop_id.split('.')[0];
        if (trigger === "analyze-btn") {
            if (!analyze_n || analyze_n === 0) return [false, "Analyze Data"];
            return [true, "Analyzing..."];
        }
        if (trigger === "upload-data") {
            // Only treat a *non-empty* contents arrival as an upload.
            if (!upload_contents) return [false, "Analyze Data"];
            return [true, "Uploading data..."];
        }
        // Any of the three example chips.
        return [true, "Uploading data..."];
    }
    """,
    [Output("analyze-btn", "disabled"), Output("analyze-btn", "children")],
    Input("analyze-btn", "n_clicks"),
    Input("load-example-btn-1", "n_clicks"),
    Input("load-example-btn-2", "n_clicks"),
    Input("load-example-btn-3", "n_clicks"),
    Input("upload-data", "contents"),
    prevent_initial_call=True
)

# When an upload completes (browser has the file bytes -> data-source-store
# gets set by update_upload_status), reset the analyze button to enabled.
# Example loads go through analyze_uploaded_data_callback which already
# resets the button; this callback covers ONLY the upload-finished case.
app.clientside_callback(
    """
    function(data_source) {
        if (data_source === "upload") {
            return [false, "Analyze Data"];
        }
        return window.dash_clientside.no_update;
    }
    """,
    [Output("analyze-btn", "disabled",  allow_duplicate=True),
     Output("analyze-btn", "children",  allow_duplicate=True)],
    Input("data-source-store", "data"),
    prevent_initial_call=True
)


# =============================================================================
# CALLBACK — DATA UPLOAD & PARSE  (UNCHANGED logic, restyled output)
# =============================================================================
@app.callback(
    Output("data-summary-output",  "children", allow_duplicate=True),
    Output("mapped-vars-store",    "data"),
    Output("dataframe-store",      "data"),
    Output("code-read-store",      "data"),
    Output("analyze-btn",          "disabled",  allow_duplicate=True),
    Output("analyze-btn",          "children",  allow_duplicate=True),
    Output("data-source-store",    "data",      allow_duplicate=True),
    Output("upload-status-output", "children",  allow_duplicate=True),
    Output("stored-data-file-name","data",      allow_duplicate=True),
    Input("analyze-btn",          "n_clicks"),
    Input("load-example-btn-1",   "n_clicks"),
    Input("load-example-btn-2",   "n_clicks"),
    Input("load-example-btn-3",   "n_clicks"),
    State("upload-data",          "contents"),
    State("upload-data",          "filename"),
    State("dataframe-store",      "data"),
    State("data-source-store",    "data"),
    State("stored-data-file-name","data"),
    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        analyze_clicks, example_clicks_1, example_clicks_2, example_clicks_3,
        contents, filename, stored_df_json, data_source, stored_file_name):

    trigger = ctx.triggered_id

    # Example dataset
    if trigger in ["load-example-btn-1", "load-example-btn-2", "load-example-btn-3"]:
        file_map = {
            "load-example-btn-1": "sys_1278_downsampled_with_VI.parquet",
            "load-example-btn-2": "sys_1403_part1_downsampled_with_VI.parquet",
            "load-example-btn-3": "sys_1422_downsampled.parquet",
        }
        example_filename = file_map.get(trigger)
        try:
            df = pd.read_parquet(f"data/{example_filename}")
            df_json = df.to_json(date_format="iso", orient="split")
            output_msg = html.Div(
                [
                    html.Span("✓", style={"color": SUCCESS, "marginRight": "8px", "fontWeight": "600"}),
                    html.Span(f"{example_filename}", style={"fontFamily": "Arial, sans-serif", "fontSize": "14px", "color": INK, "fontWeight": "600"}),
                    html.Span(" loaded", style={"color": INK_SOFT, "fontSize": "14px", "marginLeft": "4px"}),
                ],
                style={"padding": "10px 14px", "background": "#f0fdf4", "border": "1px solid #bbf7d0", "borderRadius": "8px", "fontSize": "14px"},
                className="slide-in-top",
            )
        except Exception as e:
            return (html.Div(f"Error loading example: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Analyze Data", None, "", example_filename)
        return (html.Div("", className="text-muted"),
                {}, df_json, "", False, "Analyze Data", "example", output_msg, example_filename)

    # Analyze clicked
    if trigger == "analyze-btn":
        if data_source == "upload" and contents is not None:
            df, summary_table, mapped_variables_dict, code_read = parse_contents(contents, filename)
            if df is None:
                return summary_table, {}, None, "", False, "Analyze Data", None, "", stored_file_name
        elif data_source == "example" and stored_df_json is not None:
            try:
                df = _df_from_store(stored_df_json)
                df, summary_table, mapped_variables_dict, code_read = parse_contents(df=df)
            except Exception as e:
                return (html.Div(f"Error processing stored dataset: {e}", className="alert alert-danger"),
                        {}, None, "", False, "Analyze Data", None, "", stored_file_name)
        else:
            return (_no_data_alert("Please upload a file or click an example button, then click 'Analyze Data'."),
                    {}, None, "", False, "Analyze Data", None, "", filename)

        try:
            df_json = df.to_json(date_format="iso", orient="split")
        except Exception as e:
            return (html.Div(f"Error converting DataFrame: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Analyze Data", None, "", stored_file_name)

        # Figures
        figures_output = html.Div()
        try:
            if df is not None and mapped_variables_dict:
                figures_output, err = make_overview_figures(df, mapped_variables_dict)
                figures_output = html.Div(figures_output)
        except Exception:
            figures_output = html.Div("Figure generation failed.", style={"color": ACCENT})

        combined_output = html.Div([
            html.Div([
                html.Div("identified variables", style={
                    "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
                    "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
                    "fontFamily": "Arial, sans-serif",
                }),
                html.Div(summary_table, style={"fontSize": "14px"}),
            ], style={
                "padding": "18px 20px",
                "background": "#f8fafc",
                "border": f"1px solid {BORDER}",
                "borderRadius": "10px",
                "marginBottom": "16px",
            }),
            html.Div([
                html.Div("raw data preview", style={
                    "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
                    "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
                    "fontFamily": "Arial, sans-serif",
                }),
                figures_output,
            ], style={
                "padding": "18px 20px",
                "background": "#f8fafc",
                "border": f"1px solid {BORDER}",
                "borderRadius": "10px",
            }),
        ], className="slide-in-up")

        return (combined_output, mapped_variables_dict, df_json, code_read, False,
                "Analyze Data", None, "", stored_file_name)

    return ("", {}, None, "", False, "Analyze Data", None, "", stored_file_name)


# =============================================================================
# CALLBACK — CLEAR CODE PANEL ON NEW DATA (UNCHANGED)
# =============================================================================
@app.callback(
    Output("code-preview",   "children", allow_duplicate=True),
    Output("download-link",  "href",     allow_duplicate=True),
    Output("download-link",  "style",    allow_duplicate=True),
    Input("upload-data",         "filename"),
    Input("analyze-btn",         "n_clicks"),
    Input("load-example-btn-1",  "n_clicks"),
    Input("load-example-btn-2",  "n_clicks"),
    Input("load-example-btn-3",  "n_clicks"),
    prevent_initial_call=True,
)
def clear_code_panel_on_new_data(*_):
    hidden_style = {"display": "none"}
    return None, "", hidden_style


# =============================================================================
# CALLBACK — AUTO-OPEN PVPRO PARAMS WHEN PVPRO IS THE SELECTED METRIC
#
# The PVPRO param panel is an html.Details whose `open` attribute we
# drive from the master metric selector.  Two reasons to do this rather
# than leaving it as a manually-toggled disclosure:
#
#   1. PVPRO is the *only* metric with required dataset-specific
#      parameters (cells in series, modules per string, etc.).  If the
#      user picks PVPRO but doesn't notice the collapsed panel, they'll
#      run with defaults that almost certainly don't match their array
#      and get garbage degradation rates.  Auto-unfolding makes the
#      requirement impossible to miss.
#
#   2. When the user switches AWAY from PVPRO (back to YOY, LR, etc.),
#      the panel becomes irrelevant clutter.  Collapsing it preserves
#      vertical space for the controls that ARE relevant.
#
# The callback uses allow_duplicate=True on the open output so the
# "reset on new data" callback below can also drive it.
# =============================================================================
@app.callback(
    Output("pvpro-params-details", "open", allow_duplicate=True),
    Input("metric-selected-visible", "value"),
    prevent_initial_call=True,
)
def autoopen_pvpro_params_panel(metric):
    return metric == "PVPRO"


# =============================================================================
# CALLBACK — RESET PVPRO PARAMS AND COLLAPSE PANEL ON NEW DATA
#
# Triggered whenever the user loads a different dataset -- either by
# uploading a file (`upload-data.filename`) or clicking one of the
# three example chips (`load-example-btn-{1,2,3}`).
#
# What this does
# --------------
#   * Wipes every PVPRO input back to the SAME defaults declared in the
#     layout above (cells=60, mps=1, ps=1, alphaisc=0.0046,
#     tech="mono-c-Si", days=14, iters=12).  This protects users from
#     the footgun of running PVPRO on a new dataset with the previous
#     dataset's array geometry -- which earlier produced rd ≈ 0 and
#     made the tool look broken.
#
#   * Collapses the param panel (`open=False`).  Even if PVPRO is
#     currently selected, when a new dataset lands we want the user to
#     consciously expand and review the parameters again -- not just
#     hit Calculate on auto-pilot.  The auto-open callback above will
#     re-open it automatically the next time the user reselects PVPRO,
#     which is the correct "make me look at these defaults again"
#     behaviour.
#
# allow_duplicate=True on the Output is required because
# `autoopen_pvpro_params_panel` above already targets the same prop;
# Dash forbids two callbacks writing to the same Output unless every
# binding marks itself as duplicate-aware.
# =============================================================================
@app.callback(
    Output("param-pvpro-cells",    "value", allow_duplicate=True),
    Output("param-pvpro-mps",      "value", allow_duplicate=True),
    Output("param-pvpro-ps",       "value", allow_duplicate=True),
    Output("param-pvpro-alphaisc", "value", allow_duplicate=True),
    Output("param-pvpro-tech",     "value", allow_duplicate=True),
    Output("param-pvpro-days",     "value", allow_duplicate=True),
    Output("param-pvpro-iters",    "value", allow_duplicate=True),
    Output("pvpro-params-details", "open",  allow_duplicate=True),
    Input("upload-data",         "filename"),
    Input("load-example-btn-1",  "n_clicks"),
    Input("load-example-btn-2",  "n_clicks"),
    Input("load-example-btn-3",  "n_clicks"),
    prevent_initial_call=True,
)
def reset_pvpro_params_on_new_data(*_):
    # The defaults below MUST stay in sync with the layout's
    # dcc.Input(value=...) declarations.  If you change one, change
    # both, or "reset" stops actually returning to fresh-page state.
    return 60, 1, 1, 0.0046, "mono-c-Si", 14, 12, False


# =============================================================================
# CALLBACK — GENERATE CODE  (UNCHANGED logic, restyled output)
# =============================================================================
@app.callback(
    Output("code-preview",  "children", allow_duplicate=True),
    Output("download-link", "href",     allow_duplicate=True),
    Output("download-link", "style",    allow_duplicate=True),
    Input("generate-code-btn",  "n_clicks"),
    State("stored-data-file-name",  "data"),
    State("mapped-vars-store",      "data"),
    State("filter-options",         "value"),
    State("metric-selected-visible","value"),
    prevent_initial_call=True
)
def generate_code(n, filename, mapped_variables_dict, selected_filters, selected_metric):
    clean_code = get_full_code(filename, mapped_variables_dict, selected_filters, selected_metric)
    time.sleep(2)

    preview_lines = "\n".join(clean_code.splitlines()[:24]) + "\n…"

    preview = html.Div([
        html.Div("generated python", style={
            "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
            "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
            "fontFamily": "Arial, sans-serif",
        }),
        html.Pre(
            preview_lines,
            style={
                "whiteSpace": "pre-wrap",
                "fontSize": "13px",
                "background": INK,
                "color": "#e8e4dc",
                "padding": "16px",
                "borderRadius": "10px",
                "maxHeight": "260px",
                "overflowY": "auto",
                "fontFamily": "Arial, sans-serif",
                "lineHeight": "1.55",
            },
            className="slide-in-up",
        ),
    ])

    b64 = base64.b64encode(clean_code.encode()).decode()
    href = f"data:text/plain;base64,{b64}"

    download_style = {
        "display": "inline-block",
        "marginTop": "12px",
        "color": SLATE,
        "textDecoration": "none",
        "fontSize": "15px",
        "fontWeight": "500",
        "padding": "10px 14px",
        "border": f"1px solid {BORDER_STRONG}",
        "borderRadius": "8px",
        "background": "white",
        "fontFamily": "Arial, sans-serif",
    }
    return preview, href, download_style


# =============================================================================
# CALLBACK — CONVERSATIONAL CHAT (LLM-powered Q&A about the tool)
# =============================================================================

# Load the static system context once at import time
_CHAT_CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "pvcopilot_chat_context.md") \
    if "__file__" in globals() else "pvcopilot_chat_context.md"

try:
    with open(_CHAT_CONTEXT_PATH, "r", encoding="utf-8") as _f:
        CHAT_SYSTEM_PROMPT = _f.read()
except Exception:
    CHAT_SYSTEM_PROMPT = (
        "You are the PV-Copilot Assistant, embedded in an LBNL web tool for analyzing PV "
        "degradation. Answer the user's questions about the tool's workflow (Data "
        "Prescreening, Filter, Degradation, Code), available methods (YoY, LR, HW, ARIMA, "
        "CSD), and PV concepts. Be concise (3–6 sentences), plain text, no markdown headers."
    )


# Try to import the same LLM client used by Step 1. Fall back gracefully if unavailable.
try:
    from page_supporting_files.analysis_utils import client as _llm_client
except Exception:
    _llm_client = None

_EXAMPLE_QUESTIONS = [
    "What's a normal degradation rate?",
    "Is my degradation rate normal?",
    "What does the clear-sky filter do?",
]


# ----------------------------------------------------------------------------
# CALLBACK A — Example chip click → fill composer (do NOT submit)
# ----------------------------------------------------------------------------
@app.callback(
    Output("chat-composer", "value", allow_duplicate=True),
    Input({"type": "chat-example", "idx": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def fill_composer_from_chip(_clicks):
    trigger = ctx.triggered_id
    if isinstance(trigger, dict) and trigger.get("type") == "chat-example":
        idx = trigger.get("idx", 0)
        if 0 <= idx < len(_EXAMPLE_QUESTIONS):
            return _EXAMPLE_QUESTIONS[idx]
    from dash import no_update
    return no_update


# ----------------------------------------------------------------------------
# Off-topic classifier — a quick, single-purpose LLM call that returns YES/NO
# ----------------------------------------------------------------------------
_TOPIC_CLASSIFIER_PROMPT = """You are a strict topic classifier for the PV-Copilot tool.

PV-Copilot is a web app for analyzing photovoltaic (PV) field data to estimate
module / system degradation rates. In-scope topics include:
- The PV-Copilot tool itself (its 4 steps: Data Prescreening, Filter, Degradation, Code)
- PV / solar panel degradation analysis, methods (YoY, LR, ARIMA, Holt-Winters, CSD,
  and PVPRO single-diode-model fitting -- including its reference parameters
  IL_ref, I0_ref, Rs, Rsh, n, and the reconstructed STC quantities Pmp, Vmp,
  Imp, Voc, Isc)
- Filtering of PV time-series data (irradiance, clear-sky, outliers, temperature)
- PV physics and engineering concepts directly relevant to degradation analysis
  (e.g., normalized power, IV curves, temperature coefficients, soiling, encapsulant)
- File formats / data requirements for the tool (CSV, Excel, Parquet, timestamps)
- Questions about the USER'S CURRENTLY-UPLOADED DATA or session results — e.g.
  "what's my degradation rate?", "how many rows did the filter remove?",
  "what columns are in my file?", "what time range does my data cover?",
  "how long is the analysis window?", "which method was used?", "my dataset",
  "my results", "my chart". These are always in-scope.

Out-of-scope topics include (but are not limited to):
- People, public figures, biographies, history, politics, current events
- General programming help unrelated to PV analysis
- Weather, geography, recipes, sports, entertainment, philosophy, advice
- Math / homework problems that aren't about PV
- Greetings or small talk WITHOUT a related question
- Anything that doesn't directly connect to PV degradation analysis or this tool

Classify the user's question. Respond with EXACTLY one word:
- "YES" if the question is in-scope (related to PV-Copilot, PV degradation,
  directly-relevant solar/PV concepts, OR the user's own session data/results).
- "NO" if the question is out-of-scope.

Do not explain. Do not add punctuation. One word only."""


_OFF_TOPIC_REPLY = (
    "That's outside what I can help with here. "
    "Try asking about the PV-Copilot workflow, the filters, the degradation methods, "
    "or general PV degradation concepts."
)


def _is_on_topic(question: str) -> bool:
    """Quick gate: classifier call returns True if the question is in-scope."""
    if _llm_client is None:
        return True  # no client → don't gate; fall through to the main handler
    try:
        resp = _llm_client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": _TOPIC_CLASSIFIER_PROMPT},
                {"role": "user",   "content": question},
            ],
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        # Accept anything starting with YES as on-topic; everything else (NO, or any
        # other unexpected output) treated as off-topic.
        return verdict.startswith("YES")
    except Exception:
        # If the classifier errors out, fail OPEN (let the main handler run).
        return True


# ----------------------------------------------------------------------------
# CALLBACK B1 — Send/Enter → INSTANTLY post user bubble + fire trigger
# (No LLM call here, so this returns immediately and the browser repaints.)
# ----------------------------------------------------------------------------
@app.callback(
    Output("chat-history-store",   "data",     allow_duplicate=True),
    Output("chat-composer",        "value",    allow_duplicate=True),
    Output("chat-trigger-store",   "data"),
    Output("chat-pending-store",   "data",     allow_duplicate=True),
    Input("chat-send",     "n_clicks"),
    Input("chat-composer", "n_submit"),
    State("chat-composer",      "value"),
    State("chat-history-store", "data"),
    State("chat-trigger-store", "data"),
    prevent_initial_call=True,
)
def post_user_question(send_clicks, n_submit, composer_text, history, trigger):
    from dash import no_update
    question = (composer_text or "").strip()
    if not question:
        return no_update, no_update, no_update, no_update

    history = (history or []) + [{"role": "user", "content": question}]
    trigger = trigger or {"question": "", "seq": 0}
    new_trigger = {"question": question, "seq": trigger.get("seq", 0) + 1}
    # Mark assistant area as "thinking" so render_chat shows dots immediately
    thinking_pending = {"text": "", "shown": 0, "thinking": True}
    return history, "", new_trigger, thinking_pending


# ----------------------------------------------------------------------------
# CALLBACK — Build chat data context from the per-step stores.
# Whenever any of the data stores changes, refresh the summary that gets
# injected into the LLM's system prompt so the assistant can answer questions
# about the user's actual data.
# ----------------------------------------------------------------------------
@app.callback(
    Output("chat-data-context", "data"),
    Input("mapped-vars-store",        "data"),
    Input("dataframe-store",          "data"),
    Input("dataframe-filtered",       "data"),
    Input("degradation-result-store", "data"),
    Input("metric-selected-visible",  "value"),
    Input("download-link",            "style"),  # visible when code generated
    State("stored-data-file-name",    "data"),
    State("cb-timezone",              "value"),
    State("cb-low-irra-power",        "value"),
    State("cb-outlier",               "value"),
    State("cb-clearsky",              "value"),
    prevent_initial_call=False,
)
def build_chat_context(mapped_vars, df_data, df_filtered, deg_result,
                       selected_metric, dl_style, filename,
                       cb_tz, cb_irra, cb_out, cb_cs):
    """Returns a structured dict the LLM uses to ground its answers."""
    ctx = {
        "data_loaded": False,
        "filter_applied": False,
        "degradation_computed": False,
        "code_generated": False,
    }

    # ----- Step 1: Data prescreening -----
    if mapped_vars and df_data:
        try:
            df = _df_from_store(df_data)
            time_col = mapped_vars.get("Time") or mapped_vars.get("time")
            start, end, n_rows = None, None, len(df)
            if hasattr(df.index, "min"):
                try:
                    start = df.index.min()
                    end   = df.index.max()
                    start = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)
                    end   = end.strftime("%Y-%m-%d")   if hasattr(end,   "strftime") else str(end)
                except Exception:
                    pass
            ctx["data_loaded"] = True
            ctx["data"] = {
                "filename": filename or "(uploaded file)",
                "n_rows": int(n_rows),
                "n_columns": int(len(df.columns)),
                "time_range_start": start,
                "time_range_end": end,
                "identified_variables": {k: v for k, v in (mapped_vars or {}).items() if v},
            }
        except Exception as e:
            ctx["data"] = {"error": f"Could not summarize raw data: {e}"}

    # ----- Step 2: Filtering -----
    if df_filtered:
        try:
            df_f = _df_from_store(df_filtered)
            filters_applied = []
            if cb_tz:   filters_applied.append("timezone correction")
            if cb_irra: filters_applied.append("low irradiance / power")
            if cb_out:  filters_applied.append("IQR outlier removal")
            if cb_cs:   filters_applied.append("clear-sky")
            n_kept = len(df_f)
            n_raw  = (ctx.get("data", {}) or {}).get("n_rows")
            ctx["filter_applied"] = True
            ctx["filter"] = {
                "filters_applied": filters_applied,
                "n_rows_after_filter": int(n_kept),
                "n_rows_before_filter": int(n_raw) if n_raw else None,
                "fraction_kept_pct": round(100.0 * n_kept / n_raw, 2) if n_raw else None,
            }
        except Exception as e:
            ctx["filter"] = {"error": f"Could not summarize filter result: {e}"}

    # ----- Step 3: Degradation -----
    if deg_result and deg_result.get("rate_pct_per_year") is not None:
        ctx["degradation_computed"] = True
        ctx["degradation"] = {
            "rate_percent_per_year": deg_result.get("rate_pct_per_year"),
            "method": deg_result.get("method"),
            "duration_years": deg_result.get("duration_years"),
            "window_start": deg_result.get("start"),
            "window_end":   deg_result.get("end"),
            # Carry through the per-quantity PVPRO rates (Pmp / Vmp / Imp /
            # Voc / Isc) if they exist, so the LLM can answer questions like
            # "did my current degrade more than my voltage?".
            "rates_per_quantity": deg_result.get("rates_per_quantity"),
        }

    # ----- Step 4: Code generation -----
    # The download link's style switches from display:none → display:block when ready
    if dl_style and isinstance(dl_style, dict) and dl_style.get("display") not in (None, "none"):
        ctx["code_generated"] = True

    return ctx


def _format_context_for_prompt(ctx: dict) -> str:
    """Convert the chat-data-context dict into a human-readable block for the
    LLM system prompt. Lists which steps the user has completed (with results)
    and which they haven't, so the LLM can either answer from the data or tell
    the user to run the missing step."""
    if not ctx:
        return (
            "CURRENT SESSION STATE: The user has not yet uploaded any data. "
            "If they ask about specific values (their degradation rate, how much "
            "data was filtered, what columns are in their file, etc.), tell them "
            "to upload a file and run the relevant step first."
        )

    lines = ["CURRENT SESSION STATE — what the user has done so far:"]

    # Step 1
    if ctx.get("data_loaded") and ctx.get("data"):
        d = ctx["data"]
        lines.append("")
        lines.append("✓ STEP 1 (Data Prescreening) — COMPLETED")
        lines.append(f"  • File: {d.get('filename')}")
        lines.append(f"  • Rows: {d.get('n_rows')}, Columns: {d.get('n_columns')}")
        if d.get("time_range_start"):
            lines.append(f"  • Time range: {d.get('time_range_start')} to {d.get('time_range_end')}")
        idv = d.get("identified_variables", {})
        if idv:
            iv_str = ", ".join(f"{k}={v}" for k, v in idv.items())
            lines.append(f"  • Identified variables: {iv_str}")
    else:
        lines.append("")
        lines.append("✗ STEP 1 (Data Prescreening) — NOT YET RUN")
        lines.append("  If the user asks about their data (variables, time range, file size), "
                     "tell them to upload a file and click 'Analyze Data' first.")

    # Step 2
    if ctx.get("filter_applied") and ctx.get("filter"):
        f = ctx["filter"]
        lines.append("")
        lines.append("✓ STEP 2 (Filter) — COMPLETED")
        lines.append(f"  • Filters applied: {', '.join(f.get('filters_applied') or []) or 'none'}")
        if f.get("n_rows_before_filter"):
            lines.append(f"  • Rows kept: {f.get('n_rows_after_filter')} / {f.get('n_rows_before_filter')} ({f.get('fraction_kept_pct')}%)")
        else:
            lines.append(f"  • Rows kept: {f.get('n_rows_after_filter')}")
    else:
        lines.append("")
        lines.append("✗ STEP 2 (Filter) — NOT YET RUN")
        lines.append("  If the user asks about filter results (how much data was removed, "
                     "what filters did, etc.), tell them to click 'Apply Filters' first.")

    # Step 3
    if ctx.get("degradation_computed") and ctx.get("degradation"):
        g = ctx["degradation"]
        lines.append("")
        lines.append("✓ STEP 3 (Degradation) — COMPLETED")
        # Always quote the rate to TWO decimal places in chat output ("0.46%").
        # The result-store keeps higher precision for the figure summary.
        rate_raw = g.get('rate_percent_per_year')
        rate_fmt = (f"{float(rate_raw):.2f}%/year"
                    if rate_raw is not None and rate_raw == rate_raw  # not NaN
                    else "unavailable")
        lines.append(f"  • Annual degradation rate: {rate_fmt}")
        lines.append(f"    (when discussing this number in chat, always format to "
                     f"TWO decimal places like '{rate_fmt}', NOT four decimals)")
        lines.append(f"  • Method used: {g.get('method')}")
        lines.append(f"  • Window: {g.get('window_start')} to {g.get('window_end')} ({g.get('duration_years')} years)")
        # Per-quantity PVPRO rates (if present) -- each gets the same
        # two-decimal formatting rule.
        rpq = g.get("rates_per_quantity") or {}
        if rpq:
            lines.append("  • Per-quantity rates (only when method = PVPRO):")
            for key, val in rpq.items():
                try:
                    lines.append(f"      - {key}: {float(val):.2f}%/year")
                except Exception:
                    pass
    else:
        lines.append("")
        lines.append("✗ STEP 3 (Degradation) — NOT YET RUN")
        lines.append("  If the user asks 'what is my degradation rate' or about method results, "
                     "tell them to click 'Calculate Degradation' first.")

    # Step 4
    if ctx.get("code_generated"):
        lines.append("")
        lines.append("✓ STEP 4 (Code) — COMPLETED — downloadable Python script is ready.")
    else:
        lines.append("")
        lines.append("✗ STEP 4 (Code) — NOT YET RUN")
        lines.append("  If the user asks about the generated code, tell them to click "
                     "'Generate Full Python Code' first.")

    lines.append("")
    lines.append("RULE: If a user asks about a specific value or result that comes from a step "
                 "they haven't run, politely tell them to run that step first. Do NOT make up numbers.")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CALLBACK B2 — Triggered by the trigger-store: classify + call LLM + stage reply
# Now also injects the data-context summary so the LLM can answer questions
# about the user's uploaded data.
# ----------------------------------------------------------------------------
@app.callback(
    Output("chat-pending-store",  "data",     allow_duplicate=True),
    Output("chat-typer-interval", "disabled", allow_duplicate=True),
    Input("chat-trigger-store",  "data"),
    State("chat-history-store",  "data"),
    State("chat-data-context",   "data"),
    prevent_initial_call=True,
)
def fetch_assistant_reply(trigger, history, data_ctx):
    from dash import no_update
    if not trigger or not trigger.get("question"):
        return no_update, no_update

    question = trigger["question"]

    # STEP 1: Off-topic gate
    if not _is_on_topic(question):
        pending = {"text": _OFF_TOPIC_REPLY, "shown": 0}
        return pending, False

    # STEP 2: Main answer call — inject data context into system prompt
    if _llm_client is None:
        reply = (
            "The chat backend isn't configured in this environment. "
            "Once an OpenAI client is wired up (same one used by Step 1), "
            "your question will be answered here."
        )
    else:
        try:
            full_system_prompt = (
                CHAT_SYSTEM_PROMPT
                + "\n\n---\n\n"
                + _format_context_for_prompt(data_ctx)
            )
            messages = [{"role": "system", "content": full_system_prompt}]
            for m in (history or []):
                messages.append({"role": m["role"], "content": m["content"]})
            response = _llm_client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=messages,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = f"(Sorry — the assistant ran into an error: {e})"

    pending = {"text": reply, "shown": 0}
    return pending, False


# ----------------------------------------------------------------------------
# CALLBACK C — Commit completed assistant reply to history
# (No more incremental typing; the reply is shown immediately with CSS fade-in.)
# ----------------------------------------------------------------------------
@app.callback(
    Output("chat-history-store",  "data",     allow_duplicate=True),
    Output("chat-pending-store",  "data",     allow_duplicate=True),
    Output("chat-typer-interval", "disabled"),
    Input("chat-pending-store",   "data"),
    State("chat-history-store",   "data"),
    prevent_initial_call=True,
)
def commit_pending_to_history(pending, history):
    """When a real (non-thinking) pending reply arrives, append it to history."""
    from dash import no_update
    pending = pending or {}
    text = pending.get("text", "")
    thinking = pending.get("thinking", False)

    # Skip if it's just a thinking indicator or empty
    if thinking or not text:
        return no_update, no_update, True

    # Commit and clear pending
    history = (history or []) + [{"role": "assistant", "content": text}]
    return history, {"text": "", "shown": 0}, True


# ----------------------------------------------------------------------------
# CALLBACK D — Render: build chat bubbles from history + thinking indicator
# ----------------------------------------------------------------------------
@app.callback(
    Output("chat-history", "children"),
    Input("chat-history-store", "data"),
    Input("chat-pending-store", "data"),
)
def render_chat(history, pending):
    history = history or []
    pending = pending or {}

    # Empty state
    if not history and not pending.get("text") and not pending.get("thinking"):
        return html.Div(
            html.Div(
                "Ask a question or pick an example below.",
                style={
                    "fontSize": "14px",
                    "color": MUTED,
                    "fontFamily": "Arial, sans-serif",
                    "textAlign": "center",
                    "fontStyle": "italic",
                }
            ),
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "120px",
                "color": MUTED,
            }
        )

    # Render bubbles. Mark only the LAST assistant message as "fresh" so the
    # clientside JS types it out. Older messages render fully.
    bubbles = []
    last_idx = len(history) - 1
    for i, m in enumerate(history):
        is_last_assistant = (i == last_idx and m["role"] == "assistant")
        bubbles.append(_chat_bubble(m["role"], m["content"], fresh=is_last_assistant))

    thinking = pending.get("thinking", False)

    if thinking:
        # Thinking indicator — shows after user submits, until reply arrives
        bubbles.append(
            html.Div(
                html.Div(
                    html.Span("● ● ●", className="chat-thinking-dots"),
                    style={
                        "padding": "12px 16px",
                        "background": "white",
                        "color": MUTED,
                        "borderRadius": "14px",
                        "borderBottomLeftRadius": "4px",
                        "fontSize": "14px",
                        "letterSpacing": "0.15em",
                        "border": f"1px solid {BORDER}",
                    }
                ),
                style={"display": "flex", "justifyContent": "flex-start", "marginBottom": "10px"}
            )
        )
    return bubbles


# ----------------------------------------------------------------------------
# CLIENTSIDE — auto-scroll + JS typewriter animation
# Runs entirely in the browser, so it's smooth even on a deployed server with
# network latency (no per-character round-trips).
# ----------------------------------------------------------------------------
app.clientside_callback(
    """
    function(children) {
        const el = document.getElementById('chat-history');
        if (!el) return window.dash_clientside.no_update;

        // HTML-escape so source text can't inject markup
        function escapeHtml(text) {
            return text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        }

        // Convert all closed **bold** markers in text into <strong> tags.
        // Used when we know we have the FULL final text.
        function renderBold(text) {
            return escapeHtml(text)
                .replace(/\\*\\*([^*]+?)\\*\\*/g, '<strong>$1</strong>');
        }

        // Convert PARTIAL text (mid-typing) into safe HTML. If there's a half-
        // opened `**` without a matching close yet, show its content as plain
        // text until the closing `**` arrives. This avoids broken markup and
        // avoids showing the literal `**` characters.
        function renderBoldPartial(partial) {
            // Find the last `**` and check if it's "open" (no closing pair after)
            const lastOpen = partial.lastIndexOf('**');
            if (lastOpen === -1) return escapeHtml(partial);

            // Count `**` occurrences — if even, all bolds are closed.
            const numMarkers = (partial.match(/\\*\\*/g) || []).length;
            if (numMarkers % 2 === 0) {
                // All bolds closed → safe to render fully
                return renderBold(partial);
            }
            // Odd number → the last `**` is opening but not yet closed.
            // Render the part before it normally (with closed bolds), and the
            // part after it as plain text (no `**`, no <strong>).
            const before = partial.substring(0, lastOpen);
            const after  = partial.substring(lastOpen + 2);    // skip the open `**`
            return renderBold(before) + escapeHtml(after);
        }

        // For any bubble already marked done (e.g. older messages on re-render),
        // make sure its visible HTML has the bold tags rendered.
        const done = el.querySelectorAll('.chat-bubble-typing.chat-bubble-done');
        done.forEach(function(bubble) {
            const visible = bubble.querySelector('.chat-typed');
            const source  = bubble.querySelector('.chat-typed-source');
            if (visible && source && !visible.querySelector('strong')) {
                const raw = source.textContent || '';
                visible.innerHTML = renderBold(raw);
            }
        });

        // Find "fresh" assistant bubbles that haven't been typed yet.
        const fresh = el.querySelectorAll('.chat-bubble-typing:not(.chat-bubble-done)');
        fresh.forEach(function(bubble) {
            const visible = bubble.querySelector('.chat-typed');
            const source  = bubble.querySelector('.chat-typed-source');
            const caret   = bubble.querySelector('.chat-typing-caret');
            if (!visible || !source) return;

            if (bubble.getAttribute('data-typing-started') === '1') return;
            bubble.setAttribute('data-typing-started', '1');

            const rawText = source.textContent || '';
            // We advance an index through the RAW text (including `**` markers)
            // but skip over `**` markers in the counter so they don't slow the
            // visible character pace.
            let i = 0;
            const CHARS_PER_STEP = 2;
            const STEP_MS = 18;

            const interval = setInterval(function() {
                if (!bubble.isConnected) { clearInterval(interval); return; }

                // Advance i by CHARS_PER_STEP visible characters, skipping `**`
                let advanced = 0;
                while (advanced < CHARS_PER_STEP && i < rawText.length) {
                    if (rawText.substr(i, 2) === '**') {
                        i += 2;     // skip the marker, doesn't count as visible
                    } else {
                        i += 1;
                        advanced += 1;
                    }
                }

                const partial = rawText.slice(0, i);
                visible.innerHTML = renderBoldPartial(partial);
                el.scrollTop = el.scrollHeight;

                if (i >= rawText.length) {
                    clearInterval(interval);
                    bubble.classList.add('chat-bubble-done');
                    if (caret) caret.style.opacity = '0';
                    // Final render — ensure full bolded HTML is in place
                    visible.innerHTML = renderBold(rawText);
                }
            }, STEP_MS);
        });

        el.scrollTop = el.scrollHeight;
        return window.dash_clientside.no_update;
    }
    """,
    Output("chat-history", "style"),
    Input("chat-history", "children"),
    prevent_initial_call=True,
)


# =============================================================================
# CALLBACK — STEP PROGRESS TRACKER
# Watches the existing data stores; flips boolean flags as steps complete.
# Uses `degradation-result-store` (rather than the output children) so the
# `calc` flag flips only when a successful rate has been computed — both
# fast methods and PVPRO write to this store on completion.
# =============================================================================
@app.callback(
    Output("step-progress", "data", allow_duplicate=True),
    Input("mapped-vars-store",  "data"),
    Input("dataframe-filtered", "data"),
    Input("degradation-result-store", "data"),
    Input("download-link",      "style"),
    prevent_initial_call="initial_duplicate",
)
def update_progress(mapped_vars, df_filtered, deg_result, dl_style):
    return {
        "data":   bool(mapped_vars),                               # data parsed
        "filter": bool(df_filtered),                               # filters applied
        "calc":   bool(deg_result) and deg_result.get("rate_pct_per_year") is not None,
        "code":   bool(dl_style) and dl_style.get("display") not in (None, "none"),
    }


# =============================================================================
# CALLBACK — RE-RENDER SIDEBAR ON PROGRESS CHANGE
# =============================================================================
@app.callback(
    Output("sidebar-render", "children"),
    Input("step-progress", "data"),
)
def render_sidebar(progress):
    return build_sidebar(progress)


# =============================================================================
# CALLBACK — SHOW / HIDE AGENT MESSAGES BASED ON PROGRESS
# Each subsequent agent becomes visible only when the previous step is done.
# =============================================================================
def _show_hide(visible):
    """style dict for show/hide blocks."""
    return {} if visible else {"display": "none"}


@app.callback(
    Output("agent-filter-locked",  "style"),
    Output("agent-filter-content", "style"),
    Output("agent-calc-locked",    "style"),
    Output("agent-calc-content",   "style"),
    Output("agent-code-locked",    "style"),
    Output("agent-code-content",   "style"),
    Input("step-progress", "data"),
)
def gate_agents(progress):
    data_done   = progress.get("data",   False)
    filter_done = progress.get("filter", False)
    calc_done   = progress.get("calc",   False)

    return (
        _show_hide(not data_done),    # filter locked  shown if data NOT done
        _show_hide(data_done),        # filter content shown if data done
        _show_hide(not filter_done),  # calc   locked
        _show_hide(filter_done),      # calc   content
        _show_hide(not calc_done),    # code   locked
        _show_hide(calc_done),        # code   content
    )


# =============================================================================
# CALLBACK — RESTART WORKFLOW
# Reload the page to clear all state.
# =============================================================================
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks && n_clicks > 0) {
            window.location.reload();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("step-progress", "data", allow_duplicate=True),
    Input("restart-btn", "n_clicks"),
    prevent_initial_call=True,
)


# =============================================================================
# CALLBACK — STEPPER CLICK SCROLLS RIGHT PANEL TO MATCHING AGENT
# =============================================================================
app.clientside_callback(
    """
    function(n_clicks_list) {
        const ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) {
            return window.dash_clientside.no_update;
        }
        const trig = ctx.triggered[0];
        if (!trig.value || trig.value === 0) {
            return window.dash_clientside.no_update;
        }
        // trig.prop_id looks like: {"step":"data","type":"step-row"}.n_clicks
        let stepKey;
        try {
            const idObj = JSON.parse(trig.prop_id.split('.n_clicks')[0]);
            stepKey = idObj.step;
        } catch (e) {
            return window.dash_clientside.no_update;
        }
        const target = document.getElementById('agent-' + stepKey + '-wrap');
        if (target) {
            target.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("step-progress", "data", allow_duplicate=True),
    Input({"type": "step-row", "step": ALL}, "n_clicks"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)