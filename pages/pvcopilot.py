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
from page_supporting_files.analysis_utils import make_overview_figures, normalize, low_irra_power_filter, low_power_filter, aggregate_daily, compute_yoy, get_full_code
from page_supporting_files.analysis_utils import compute_lr, compute_hw, compute_arima, compute_csd, compute_pvpro
from page_supporting_files.analysis_utils import rate_is_plausible
from page_supporting_files.pvcopilot_filter_functions import identify_outliers_iqr, clear_sky_filter, basic_value_filter
import base64
import os
import time
import threading
import uuid
import copy
import collections
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

# =============================================================================
# TIMEOUT GUARD FOR SYNCHRONOUS STEPS
#
# Analyze, Apply Filters, and the fast degradation methods (YoY/LR/HW/ARIMA/CSD)
# all run synchronously inside their Dash callbacks. A malformed dataset can
# make them run effectively forever and hang the UI. We cap them at 10 s by
# running the pure computation in a worker and waiting on its result; if it
# overruns, the callback aborts and surfaces a "something wrong with your data"
# error instead of leaving the user staring at a spinner.
#
# Note: PVPRO is deliberately NOT guarded here — it has its own background-job
# infrastructure below and legitimately takes 1–3 minutes.
#
# Caveat: a timed-out worker keeps running to completion in the background
# (Python can't force-kill a thread). That's acceptable — the only requirement
# is that the *UI* stops waiting and reports the problem.
# =============================================================================
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=4)
STEP_TIMEOUT_S = 10


def _run_with_timeout(fn, *args, timeout=STEP_TIMEOUT_S, **kwargs):
    """Run fn(*args, **kwargs) but raise FutureTimeout if it exceeds `timeout` s."""
    fut = _TIMEOUT_POOL.submit(fn, *args, **kwargs)
    return fut.result(timeout=timeout)

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


def _duration_years(df):
    """Time span of a datetime-indexed frame, in years. Returns None if it
    can't be determined (non-datetime index, empty frame, etc.)."""
    try:
        idx = pd.to_datetime(df.index)
        return (idx.max() - idx.min()).days / 365.25
    except Exception:
        return None


def _data_quality_notes(df, mapping, extra_notes=None):
    """Build a single collapsed disclosure listing missing-data items and what
    each implies for the degradation result.

    `extra_notes` is an optional list of strings (e.g. the AC-fallback /
    ambiguous-column warnings parse_contents emits) folded into the same toggle.

    Returns (n_notes, component). Component is None when there's nothing to
    flag, so the caller can skip rendering entirely.
    """
    mapping = mapping or {}
    notes = list(extra_notes or [])

    # (The "power computed as V x I" note already comes from parse_contents via
    # extra_notes, so we don't add a duplicate here.)

    irra_key = mapping.get("Irradiance")
    has_irr = bool(irra_key) and (df is None or irra_key in getattr(df, "columns", []))
    if not has_irr:
        notes.append("No irradiance column — power is NOT irradiance-normalized, so the "
                     "degradation rate is not weather-corrected and is less reliable.")

    temp_key = mapping.get("Module temperature")
    has_temp = bool(temp_key) and (df is None or temp_key in getattr(df, "columns", []))
    if has_irr and not has_temp:
        notes.append("No module-temperature column — normalization uses irradiance only, "
                     "with no temperature correction.")

    if not notes:
        return 0, None

    n = len(notes)
    component = html.Details(
        [
            html.Summary(
                f"⚠️ {n} data-quality note{'s' if n != 1 else ''} — click to expand",
                style={"cursor": "pointer", "color": "#92400e", "fontSize": "14px",
                       "fontWeight": "600", "fontFamily": "Arial, sans-serif"},
            ),
            html.Ul(
                [html.Li(t, style={"fontSize": "13px", "color": "#92400e",
                                   "marginBottom": "4px", "lineHeight": "1.5"})
                 for t in notes],
                style={"marginTop": "8px", "marginBottom": "0", "paddingLeft": "18px"},
            ),
        ],
        style={
            "padding": "12px 14px",
            "background": "#fffbeb",
            "border": "1px solid #fde68a",
            "borderRadius": "10px",
            "marginBottom": "16px",
        },
    )
    return n, component


def _duration_block_banner(duration_years):
    """Prominent (non-collapsed) red banner shown in the Analyze output when the
    dataset spans less than a year, warning that Calculate Degradation is blocked."""
    months = int(round((duration_years or 0) * 12))
    return html.Div(
        [
            html.B("Not enough data for degradation analysis. "),
            f"This dataset spans only about {months} month{'s' if months != 1 else ''} "
            "(less than 1 year). Degradation analysis needs at least 1 year of data, so "
            "the Calculate Degradation step is disabled for this dataset.",
        ],
        style={
            "padding": "12px 14px",
            "background": "#fef2f2",
            "border": "1px solid #fecaca",
            "borderRadius": "10px",
            "color": "#991b1b",
            "fontSize": "14px",
            "lineHeight": "1.5",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "16px",
        },
    )


def _implausible_rate_banner(rd):
    """Red banner shown above a degradation result whose rate is outside the
    plausible band (positive or very large). Returns None when the rate is fine,
    so callers can skip it. Real degradation is a small negative number; an
    implausible value almost always means un-normalizable (no irradiance) or
    too-sparse data."""
    if rate_is_plausible(rd):
        return None
    return html.Div(
        [
            html.B(f"⚠️ Unreliable result ({rd:.1f}%/year). "),
            "This is outside the physically plausible range for panel degradation "
            "(which is a small negative number, roughly 0 to −3%/year). It usually "
            "means the data has no irradiance to normalize against, or too few / too "
            "sparse points — so this number reflects weather and sampling noise, not "
            "real aging. Treat it as not trustworthy.",
        ],
        style={
            "padding": "12px 14px",
            "background": "#fef2f2",
            "border": "1px solid #fecaca",
            "borderRadius": "10px",
            "color": "#991b1b",
            "fontSize": "13px",
            "lineHeight": "1.5",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "16px",
        },
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
    # The upload box, example chips and data-requirements now live in the
    # shared upload header above the mode tabs (so both modes share one
    # data source).  This agent runs DETECTION on that already-loaded data.
    html.Div(
        [
            "Using the data you loaded above. Click ",
            html.B("Analyze Data"),
            " and I'll detect your columns and preview the raw signal.",
        ],
        style={
            "fontSize": "16px",
            "color": INK,
            "lineHeight": "1.6",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "18px",
        }
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


def build_stat_metric_options(disable_yoy=False):
    """Return the statistical-method radio options, optionally greying out YoY.

    YoY compares each day against the same day one year earlier, so it needs at
    least two years of data to produce any comparison. For shorter datasets we
    disable the option (and the gating callback falls the selection back to LR).
    """
    opts = []
    for o in stat_metric_options:
        if disable_yoy and o["value"] == "YOY":
            opts.append({**o, "disabled": True})
        else:
            opts.append(o)
    return opts


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
            options=build_stat_metric_options(disable_yoy=False),
            labelStyle={"display": "block", "marginBottom": "10px",
                        "cursor": "pointer", "color": "inherit"},
            labelClassName="metric-radio-label",
            inputStyle={"marginRight": "10px", "marginTop": "3px",
                        "accentColor": NAVY},
            style={"marginBottom": "0"},
        ),
        # Shown by gate_yoy_by_duration() when the dataset is under 2 years.
        html.Div(id="yoy-disabled-note", style={"display": "none"}),

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

# =============================================================================
# ADVANCED/SIMPLE MODE SYSTEM  (merged in from pvcopilotMaster)
#   Shared upload header, common header, mode-switcher tabs, and the
#   entire Simple-mode panel + helpers.  None of the original Advanced
#   workflow above was removed; this only ADDS the second mode.
# =============================================================================

def _success_banner(message, prefix="Note:"):
    """Light-green success banner used by both modes.  `message` is the text
    after the bold prefix (e.g. 'example data 2 uploaded')."""
    return html.Div(
        [
            html.B(f"{prefix} ", style={"color": "#15803d"}),
            html.Span(message, style={"color": "#15803d"}),
        ],
        style={
            "padding": "12px 16px",
            "background": "#ecfdf5",          # light green
            "border": "1px solid #86efac",    # green border
            "borderRadius": "10px",
            "fontSize": "15px",
            "fontFamily": "Arial, sans-serif",
            "marginBottom": "8px",
        },
        className="slide-in-top",
    )


def _working_banner(message):
    """Blue 'in progress' banner used during the Simple-mode staged reveal."""
    return html.Div(
        [
            html.Span("⏳", style={"marginRight": "8px"}),
            html.Span(message,
                      style={"fontFamily": "Arial, sans-serif", "fontSize": "14px",
                             "color": INK, "fontWeight": "600"}),
        ],
        style={"padding": "12px 16px", "background": "#eff6ff",
               "border": "1px solid #bfdbfe", "borderRadius": "10px",
               "fontSize": "14px", "marginBottom": "8px"},
    )


def _shared_example_btn(btn_id, label):
    return html.Button(
        ["⬢  ", label],
        id=btn_id,
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
    )


shared_upload_header = html.Div(
    [
        html.Div("LOAD YOUR DATA", style={
            "fontSize": "15px", "color": ACCENT, "fontWeight": "800",
            "fontFamily": "Arial, sans-serif", "textTransform": "uppercase",
            "letterSpacing": "0.12em", "marginBottom": "10px",
        }),
        html.Div(
            [
                "Drop a ",
                html.B("CSV, Excel, or Parquet"),
                " file, or try an example dataset. Then pick a mode below.",
            ],
            style={
                "fontSize": "16px", "color": INK, "lineHeight": "1.6",
                "fontFamily": "Arial, sans-serif", "marginBottom": "16px",
            }
        ),

        # Data requirements disclosure
        html.Details(
            [
                html.Summary(
                    [
                        html.Span("IMPORTANT", className="important-badge", style={
                            "fontSize": "10px", "fontWeight": "700", "color": "white",
                            "background": NAVY, "padding": "2px 8px",
                            "borderRadius": "999px", "letterSpacing": "0.06em",
                            "marginRight": "10px", "verticalAlign": "middle",
                        }),
                        html.Span("Click to see data requirements"),
                    ],
                    style={
                        "cursor": "pointer", "fontSize": "14px", "color": INK,
                        "fontFamily": "Arial, sans-serif", "fontWeight": "700",
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
                        "fontSize": "14px", "color": INK, "paddingLeft": "18px",
                        "lineHeight": "1.7", "marginBottom": "0", "marginTop": "10px",
                        "padding": "12px 14px 12px 32px", "background": "#eff6ff",
                        "border": "1px solid #bfdbfe", "borderRadius": "10px",
                        "fontFamily": "Arial, sans-serif",
                    }
                ),
            ],
            style={"marginBottom": "16px"}
        ),

        # Upload box
        dcc.Upload(
            id="upload-data",
            accept=".csv, text/csv, .xls, .xlsx, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, .parquet",
            children=html.Div(
                [
                    "Drag a file here, or ",
                    html.Span("browse", style={"color": ACCENT, "textDecoration": "underline", "fontWeight": "500"}),
                    html.Span("  ·  .csv / .xlsx / .parquet", style={
                        "fontSize": "12px", "color": INK_SOFT, "marginLeft": "6px",
                    }),
                ],
                style={"textAlign": "center", "fontSize": "14px", "color": INK,
                       "fontFamily": "Arial, sans-serif"}
            ),
            style={
                "width": "100%", "padding": "14px 16px",
                "border": f"1.5px dashed {BORDER_STRONG}", "borderRadius": "10px",
                "backgroundColor": "#f8fafc", "cursor": "pointer",
                "transition": "all 0.15s ease",
            }
        ),

        html.Div(id="upload-status-output", style={"marginTop": "10px"}),

        # Example buttons
        html.Div(
            [
                html.Div("Or try an example:", style={
                    "fontSize": "14px", "color": INK_SOFT, "marginBottom": "8px",
                    "fontFamily": "Arial, sans-serif",
                }),
                html.Div(
                    [
                        _shared_example_btn("load-example-btn-1", "Example 1"),
                        _shared_example_btn("load-example-btn-2", "Example 2"),
                        _shared_example_btn("load-example-btn-3", "Example 3"),
                    ],
                ),
            ],
            style={"marginTop": "14px"}
        ),
    ],
    style={
        "padding": "32px 64px 36px",
        "background": PAPER_RAISED,
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "marginBottom": "16px",
    }
)


# =============================================================================
# CHAT — AGENT 2 · FILTER
# =============================================================================
_common_hero_bullets = html.Ul(
    [
        html.Li([html.B("Agentic analysis"), " — drop raw data, the agent runs the full pipeline for you."]),
        html.Li([html.B("Two modes"), " — a fast degradation result, or deep diagnostic insight."]),
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
)

# The common header (eyebrow + big title + bullets + dev note), shown ONCE
# above the shared data-upload area, boxed in the same card style as the
# mode panels below.
common_header = html.Div(
    html.Div(
        [
            html.Div("AGENTIC PV DEGRADATION ANALYSIS", style={
                "fontSize": "15px",
                "color": ACCENT,
                "fontFamily": "Arial, sans-serif",
                "fontWeight": "600",
                "textTransform": "uppercase",
                "letterSpacing": "0.12em",
                "marginBottom": "12px",
            }),
            html.H1("Drop your data, get the degradation rate.", style={
                "fontSize": "40px",
                "fontFamily": "Arial, sans-serif",
                "fontWeight": "700",
                "color": INK,
                "lineHeight": "1.05",
                "margin": "0 0 16px",
            }),
            html.Div(_common_hero_bullets),
            soft_blue_callout(
                [
                    html.B("Note: "),
                    "This tool is currently under active development. ",
                    html.B("We don't store your data"),
                    ", and online usage doesn't require "
                    "a user API key. If you encounter issues, please ",
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
        style={"padding": "32px 64px 36px"},
    ),
    style={
        "background": PAPER_RAISED,
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "marginBottom": "16px",
    }
)


# Advanced-mode hero middle content — the four-agent bullet list.
def _mode_tab(label, sub, mode_key, active):
    """One pill in the mode switcher — capsule-shaped, single line."""
    return html.Button(
        [
            html.Span(label, style={
                "fontSize": "14px",
                "fontWeight": "700",
                "fontFamily": "Arial, sans-serif",
                "color": "#ffffff" if active else INK,
                "marginRight": "9px",
            }),
            html.Span(sub, style={
                "fontSize": "12px",
                "fontWeight": "500",
                "fontFamily": "Arial, sans-serif",
                "color": "rgba(255,255,255,0.85)" if active else "#94a3b8",
            }),
        ],
        id={"type": "mode-tab", "mode": mode_key},
        className="mode-tab-active" if active else "mode-tab-idle",
        n_clicks=0,
        style={
            "display": "flex",
            "alignItems": "center",
            "flexWrap": "wrap",
            "flex": "0 1 auto",
            "minWidth": "0",
            "justifyContent": "center",
            "padding": "11px 22px",
            "border": "none",
            "borderRadius": "999px",   # full capsule
            "cursor": "pointer",
            "background": NAVY if active else "transparent",
            "boxShadow": "0 1px 3px rgba(0,100,171,0.25)" if active else "none",
            "transition": "all 0.15s ease",
        },
    )


def build_mode_tabs(mode="simple"):
    return html.Div(
        [
            _mode_tab("Simple mode", "Drop data, get the rate", "simple",
                      active=(mode == "simple")),
            _mode_tab("Advanced mode", "Control every step", "advanced",
                      active=(mode == "advanced")),
        ],
        id="mode-tabs",
        style={
            # Flex that wraps so both pills stay visible (and the sub-captions
            # aren't cut off) on narrow screens.
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "8px",
            "padding": "6px",
            "background": "#f1f5f9",
            "border": f"1px solid {BORDER}",
            "borderRadius": "18px",   # softer corners when wrapped to 2 rows
            "marginBottom": "4px",
            "maxWidth": "100%",
            "boxSizing": "border-box",
        },
    )


# -----------------------------------------------------------------------------
# SIMPLE-MODE PANEL
# -----------------------------------------------------------------------------
# Uses the SHARED upload area above; the panel itself just has a single
# "Analyze" button that runs the whole default pipeline and drops the result
# into `simple-result`.
# -----------------------------------------------------------------------------




def _simple_analyze_style(disabled):
    base = {
        "padding": "13px 32px",
        "border": "none",
        "borderRadius": "999px",
        "fontSize": "14px",          # matches the mode-tab label size
        "fontWeight": "700",
        "fontFamily": "Arial, sans-serif",
        "letterSpacing": "0.01em",
        "whiteSpace": "nowrap",
    }
    if disabled:
        base.update({"background": "#cbd5e1", "color": "#ffffff",
                     "cursor": "not-allowed"})
    else:
        base.update({"background": INK, "color": PAPER, "cursor": "pointer"})
    return base


def _explainer_bullets(items):
    """Bulleted list with a bold lead-in per item (image-2 style).
    `items` is a list of (bold, rest) tuples."""
    return html.Ul(
        [html.Li([html.B(bold), rest], style={"marginBottom": "8px"})
         for bold, rest in items],
        style={
            "margin": "0", "paddingLeft": "22px",
            "fontSize": "15px", "color": INK_SOFT, "lineHeight": "1.6",
            "fontFamily": "Arial, sans-serif",
        },
    )


_SIMPLE_EXPLAINER_BULLETS = [
    ("Fully automatic", " — runs the full default pipeline (prescreen → filter → degradation)."),
    ("Preset parameters & metric", " — defaults chosen for you; use Advanced mode to tune them."),
]

def _simple_pvpro_btn_style(disabled):
    """Pill button for the Simple-mode PVPRO box (matches Analyze button)."""
    base = {
        "padding": "13px 32px",
        "border": "none",
        "borderRadius": "999px",
        "fontSize": "14px",
        "fontWeight": "700",
        "fontFamily": "Arial, sans-serif",
        "letterSpacing": "0.01em",
        "whiteSpace": "nowrap",
    }
    if disabled:
        base.update({"background": "#cbd5e1", "color": "#ffffff",
                     "cursor": "not-allowed"})
    else:
        base.update({"background": INK, "color": PAPER, "cursor": "pointer"})
    return base


def _simple_pvpro_params_block():
    """Collapsible module/array parameters for the Simple-mode PVPRO box.
    Identical fields to the Advanced-mode PVPRO metric, but with distinct
    `simple-param-pvpro-*` ids so the two never collide."""
    return html.Details(
        [
            html.Summary(
                [
                    html.Span("IMPORTANT", className="important-badge", style={
                        "fontSize": "10px", "fontWeight": "700", "color": "white",
                        "background": NAVY, "padding": "2px 8px",
                        "borderRadius": "999px", "letterSpacing": "0.06em",
                        "marginRight": "10px", "verticalAlign": "middle",
                    }),
                    html.Span("Provide module & array parameters for PVPRO"),
                ],
                style={"cursor": "pointer", "fontSize": "13px", "color": INK,
                       "fontWeight": "700", "fontFamily": "Arial, sans-serif",
                       "marginTop": "4px"},
            ),
            html.Div([
                html.Div(style={"display": "flex", "gap": "8px",
                                "flexWrap": "wrap", "marginBottom": "8px"},
                         children=[
                    html.Div([
                        html.Div("Cells in series (per module)", style=_label_style),
                        dcc.Input(id="simple-param-pvpro-cells", type="number",
                                  value=60, step=1, min=1, style=_param_input_style),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Div("Modules per string", style=_label_style),
                        dcc.Input(id="simple-param-pvpro-mps", type="number",
                                  value=1, step=1, min=1, style=_param_input_style),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Div("Parallel strings", style=_label_style),
                        dcc.Input(id="simple-param-pvpro-ps", type="number",
                                  value=1, step=1, min=1, style=_param_input_style),
                    ], style={"flex": "1", "minWidth": "140px"}),
                ]),
                html.Div(style={"display": "flex", "gap": "8px",
                                "flexWrap": "wrap", "marginBottom": "8px"},
                         children=[
                    html.Div([
                        html.Div("alpha_isc (A/°C)", style=_label_style),
                        dcc.Input(id="simple-param-pvpro-alphaisc", type="number",
                                  value=0.0046, step=0.0001, min=0,
                                  style=_param_input_style),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Div("Technology", style=_label_style),
                        dcc.Dropdown(
                            id="simple-param-pvpro-tech",
                            options=[
                                {"label": "mono-c-Si", "value": "mono-c-Si"},
                                {"label": "multi-c-Si", "value": "multi-c-Si"},
                                {"label": "GaAs", "value": "GaAs"},
                                {"label": "CIGS", "value": "CIGS"},
                                {"label": "CdTe", "value": "CdTe"},
                            ],
                            value="mono-c-Si", clearable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], style={"flex": "1", "minWidth": "140px"}),
                ]),
                html.Div(style={"display": "flex", "gap": "8px",
                                "flexWrap": "wrap"}, children=[
                    html.Div([
                        html.Div("Days per run", style=_label_style),
                        dcc.Input(id="simple-param-pvpro-days", type="number",
                                  value=14, step=1, min=2, style=_param_input_style),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Div("Iterations per year", style=_label_style),
                        dcc.Input(id="simple-param-pvpro-iters", type="number",
                                  value=12, step=1, min=2, style=_param_input_style),
                    ], style={"flex": "1", "minWidth": "140px"}),
                ]),
            ], style={"marginTop": "6px", "padding": "12px 14px",
                      "background": "#f1f5f9", "borderRadius": "8px",
                      "border": f"1px solid {BORDER}",
                      "boxSizing": "border-box"}),
        ],
        open=False,
        style={"marginTop": "12px"},
    )


def _simple_method_radio():
    """Method chooser for Simple mode: YoY vs PVPRO.  Both are always
    selectable; if PVPRO is chosen but the data has no DC voltage/current,
    Stage 1 surfaces an error and points the user back to YoY."""
    pvpro_label = html.Span(
        [
            html.Span([
                html.B("PVPRO", style={"fontFamily": "Arial, sans-serif"}),
                html.Span(" — reveals single-diode-model parameter trends "
                          "(Pmp, Voc, Isc, etc.) (~1–3 min)",
                          style={"color": INK_SOFT, "fontSize": "13px"}),
            ], style={"flex": "1", "minWidth": "0"}),
            # PVPRO logo (links to the upstream repo), mirroring Advanced mode.
            html.A(
                html.Img(
                    src=app.get_asset_url("pvpro_logo.png"), alt="PVPRO",
                    style={"height": "28px", "width": "auto", "display": "block"},
                ),
                href="https://github.com/DuraMAT/pvpro", target="_blank",
                title="PVPRO on GitHub",
                style={"marginLeft": "10px", "flexShrink": "0",
                       "display": "inline-block", "textDecoration": "none"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "flex": "1",
               "minWidth": "0"},
    )
    return dcc.RadioItems(
        id="simple-method-radio",
        options=[
            {"label": html.Span([
                html.B("YoY", style={"fontFamily": "Arial, sans-serif"}),
                html.Span(" — year-on-year statistical degradation (fast)",
                          style={"color": INK_SOFT, "fontSize": "13px"}),
            ]), "value": "YOY"},
            {"label": pvpro_label, "value": "PVPRO"},
        ],
        value="YOY",
        labelStyle={"display": "flex", "alignItems": "center", "gap": "8px",
                    "marginBottom": "8px", "fontSize": "15px",
                    "fontFamily": "Arial, sans-serif", "color": INK},
        inputStyle={"marginRight": "6px"},
        style={"marginTop": "4px"},
    )


def _simple_pvpro_about():
    """Collapsible 'learn more' detail for the PVPRO method, folded by default.
    Explains what the single-diode-model fit produces and its requirements."""
    return html.Details(
        [
            html.Summary(
                "About PVPRO",
                style={"cursor": "pointer", "fontSize": "13px",
                       "fontWeight": "600", "color": ACCENT,
                       "fontFamily": "Arial, sans-serif"},
            ),
            html.Div(
                [
                    html.P(
                        "PVPRO fits the single-diode model (SDM) to your "
                        "measured operating points in successive time windows, "
                        "then tracks how the reference-condition (STC) "
                        "parameters drift over time.",
                        style={"margin": "0 0 8px"},
                    ),
                    html.P("It reports an annual degradation rate for each of:",
                           style={"margin": "0 0 4px"}),
                    html.Ul(
                        [
                            html.Li([html.B("Pmp"), " — maximum-power-point power"]),
                            html.Li([html.B("Vmp"), " — maximum-power-point voltage"]),
                            html.Li([html.B("Imp"), " — maximum-power-point current"]),
                            html.Li([html.B("Voc"), " — open-circuit voltage"]),
                            html.Li([html.B("Isc"), " — short-circuit current"]),
                        ],
                        style={"margin": "0 0 8px", "paddingLeft": "20px"},
                    ),
                    html.P(
                        "Separating voltage- and current-side trends helps "
                        "attribute loss to specific physical mechanisms — "
                        "something power-only methods like YoY can't do.",
                        style={"margin": "0 0 8px"},
                    ),
                    html.P(
                        [
                            html.B("Requires "),
                            "DC voltage and DC current columns. ",
                            html.B("Runtime "),
                            "is typically 1–3 minutes.",
                        ],
                        style={"margin": "0"},
                    ),
                ],
                style={"fontSize": "13px", "color": INK_SOFT,
                       "lineHeight": "1.6", "fontFamily": "Arial, sans-serif",
                       "marginTop": "8px", "padding": "10px 12px",
                       "background": "rgba(241, 245, 249, 0.6)",
                       "border": f"1px solid {BORDER}", "borderRadius": "8px"},
            ),
        ],
        open=False,
        style={"marginTop": "8px"},
    )


simple_mode_panel = html.Div(
    [
        # Single box: bullets + method radio (+ PVPRO params) + Analyze button.
        html.Div(
            [
                _explainer_bullets(_SIMPLE_EXPLAINER_BULLETS),

                # Method chooser (YoY vs PVPRO).  Rebuilt on data load so the
                # PVPRO option enables only when DC V & I are present.
                html.Div("method", style={
                    "fontSize": "12px", "color": INK_SOFT,
                    "textTransform": "uppercase", "letterSpacing": "0.1em",
                    "fontWeight": "600", "marginTop": "16px",
                    "marginBottom": "8px", "fontFamily": "Arial, sans-serif",
                }),
                html.Div(id="simple-method-wrap",
                         children=_simple_method_radio()),

                # Folded-by-default 'learn more' for PVPRO — only shown when
                # PVPRO is the selected method.
                html.Div(id="simple-pvpro-about-wrap",
                         children=_simple_pvpro_about(),
                         style={"display": "none"}),

                # PVPRO module/array params — only relevant when PVPRO is the
                # chosen method.  Hidden by default; revealed by a callback.
                html.Div(id="simple-pvpro-params-wrap",
                         children=_simple_pvpro_params_block(),
                         style={"display": "none"}),

                html.Div(
                    html.Button(
                        "Click to run the analysis",
                        id="simple-analyze-btn",
                        n_clicks=0,
                        disabled=True,
                        style=_simple_analyze_style(disabled=True),
                    ),
                    style={"marginTop": "16px"},
                ),
            ],
            style={
                "padding": "18px 20px",
                "background": "rgba(241, 245, 249, 0.5)",
                "border": f"1px dashed {BORDER_STRONG}",
                "borderRadius": "10px",
            },
        ),

        # Status + result (shared by both methods).
        html.Div(id="simple-status", style={"marginTop": "16px"}),
        html.Div(id="simple-result", style={"marginTop": "8px"}),
        # PVPRO long-running progress renders here while a fit is in flight.
        html.Div(id="simple-pvpro-progress-output", style={"marginTop": "16px"}),
    ],
    id="simple-mode-panel",
    style={}
)


# =============================================================================
# FULL LAYOUT
# =============================================================================
def _render_pvpro_layout(rd, figs, rates, start_str, end_str,
                         duration_years, elapsed):
    rates = rates or {}
    figs = figs or {}
    rate_pct = rd / 100 if (rd is not None and np.isfinite(rd)) else 0.0
    rate_color = VALUE_MAJOR

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
        r_str = "n/a" if not np.isfinite(r) else f"{r:+.2f}%/yr"
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
            html.Td(r_str, style={"padding": "4px 0", "color": VALUE_DETAIL,
                                  "fontWeight": "700", "textAlign": "right",
                                  "fontFamily": "Arial, sans-serif"}),
        ]))
    rates_table = html.Details(
        [
            html.Summary(
                "parameter degradation rates",
                style={
                    "fontSize": "12px", "color": INK_SOFT,
                    "textTransform": "uppercase", "letterSpacing": "0.1em",
                    "fontWeight": "600", "fontFamily": "Arial, sans-serif",
                    "cursor": "pointer",
                },
            ),
            html.Table(html.Tbody(rates_rows), style={
                "width": "100%", "fontSize": "14px", "borderCollapse": "collapse",
                "marginTop": "10px",
            }),
        ],
        open=False,
        style={"marginTop": "14px", "marginBottom": "16px"},
    )

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
            html.Div([html.Span("Method: ", style={"color": INK_SOFT}),
                      html.B("PVPRO", style={"color": VALUE_DETAIL})],
                     style={"fontSize": "14px", "marginBottom": "3px"}),
            # Window + duration on a single line: window first, then duration.
            html.Div([html.Span("Window: ", style={"color": INK_SOFT}),
                      html.B(f"{start_str}  →  {end_str}",
                             style={"fontFamily": "Arial, sans-serif",
                                    "fontSize": "13px", "color": VALUE_DETAIL}),
                      html.Span(f"  ({duration_years:.1f} years)",
                                style={"color": INK_SOFT, "fontSize": "13px"})],
                     style={"fontSize": "14px"}),
        ], style={"fontFamily": "Arial, sans-serif"}),
        rates_table,
    ])

    card_style = {
        "background": "#ffffff",
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "padding": "8px 10px",
        "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
        "minWidth": "0",          # allow the card to shrink inside the grid
        "overflow": "hidden",     # keep the plot from spilling past the card
    }

    def _fig_card(key, span_cols=False):
        f = figs.get(key)
        if f is None:
            return html.Div()
        style = dict(card_style)
        if span_cols:
            # Pmp spans the full row on wider screens; harmless at 1 column.
            style["gridColumn"] = "1 / -1"
        return html.Div(
            dcc.Graph(
                figure=f,
                config={"displayModeBar": False, "responsive": True},
                style={"width": "100%"},
            ),
            style=style,
        )

    # Responsive grid: as many ~360px columns as fit, collapsing to a single
    # full-width column (one figure per row) on narrow screens.
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
            "gridTemplateColumns": "repeat(auto-fit, minmax(min(100%, 360px), 1fr))",
            "gap": "10px",
            "marginTop": "8px",
        },
    )

    fig_grid_heading = html.Div("pvpro-lite degradation trends", style={
        "fontSize": "12px", "color": INK_SOFT,
        "textTransform": "uppercase", "letterSpacing": "0.1em",
        "fontWeight": "600", "marginBottom": "6px", "marginTop": "8px",
        "fontFamily": "Arial, sans-serif",
    })

    return html.Div([
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


# =============================================================================
# SIMPLE-MODE METHOD CHOOSER + PVPRO HELPERS
#
# The Simple-mode box offers a YoY / PVPRO radio.  Both are always selectable.
# If PVPRO is chosen but parse_contents (Stage 1) finds no DC Voltage / DC
# Current, Stage 1 surfaces an error pointing the user back to YoY.
# =============================================================================
def _simple_has_dc_vi(mapped):
    """True iff the mapping has usable DC Voltage AND DC Current columns."""
    if not mapped:
        return False
    v = mapped.get("DC Voltage")
    i = mapped.get("DC Current")
    def _ok(x):
        return bool(x) and str(x).strip().upper() not in ("", "N/A", "NA", "NONE")
    return _ok(v) and _ok(i)


# ---- Show PVPRO about + params only when PVPRO is the chosen method ----------
_METRIC_LABELS = {
    "YOY":   "YoY (Year-over-Year)",
    "LR":    "LR (Linear Regression)",
    "HW":    "HW (Holt-Winters)",
    "ARIMA": "ARIMA",
    "CSD":   "CSD (Classical Seasonal Decomposition)",
    "PVPRO": "PVPRO (physics-based)",
}


def _metric_label(metric):
    """Human-readable metric name for the diagnostic, e.g. 'YOY' -> 'YoY
    (Year-over-Year)'.  Falls back to the raw code if unrecognized."""
    if not metric:
        return None
    return _METRIC_LABELS.get(str(metric).upper(), str(metric))


def _summarize_daily_series(series, metric_label=None):
    """Build a compact, LLM-readable description of the power trend.

    The summary gives the model:
      * the underlying degradation rate's direction/slope,
      * the monthly mean power as a short dated series,
      * whether the POWER DATA shows a clear repeating seasonal cycle (computed
        from the data itself, by calendar month across years), and
      * any genuinely anomalous period -- detected AFTER removing the seasonal
        cycle, so the normal winter trough of a seasonal site is NOT mistaken
        for a fault.

    Input is the DAILY-AGGREGATED power series (the dots in the plot), NOT the
    smoothed trend line. Built from the real date-indexed pandas Series."""
    try:
        import numpy as np
        import pandas as pd

        s = series.dropna()
        if len(s) < 3:
            return "Trend series too short to summarize."

        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index)
            except Exception:
                pass

        y_all = s.values.astype("float64")
        n = y_all.size
        mean_all = float(np.mean(y_all))

        # Overall linear trend (just for direction/magnitude context).
        x = np.arange(n)
        slope_day, intercept = np.polyfit(x, y_all, 1)
        slope_year = slope_day * 365.25
        fit_start = float(intercept)
        fit_pct = (slope_day * (n - 1) / fit_start * 100) if abs(fit_start) > 1e-9 else 0.0

        metric_txt = f" Metric: {metric_label}." if metric_label else ""
        lines = [
            f"Input = daily-aggregated power, {n} days, mean ~{mean_all:.0f} W, "
            f"overall linear trend ~{slope_year:+.0f} W/year "
            f"({fit_pct:+.1f}% across the window).{metric_txt}"
        ]

        # --- Monthly means -------------------------------------------------
        monthly = None
        if isinstance(s.index, pd.DatetimeIndex):
            monthly = s.resample("MS").mean().dropna()
            monthly_full = monthly.copy()
            if len(monthly) > 30:
                step = int(np.ceil(len(monthly) / 30))
                monthly = monthly.iloc[::step]

        if monthly is None or len(monthly) < 4:
            lines.append(f"Approximate linear slope ~{slope_day:.3f} W/day.")
            return " ".join(lines)

        pairs = ", ".join(
            f"{idx.strftime('%Y-%m')}:{val:.0f}" for idx, val in monthly.items()
        )
        lines.append(
            "Monthly mean power (W) — these are monthly samples of the SAME "
            "daily/30-day-rolling trend the user sees on the chart, given here "
            "compactly for analysis: " + pairs + "."
        )

        # --- Seasonality detection (from the data, by calendar month) ------
        # A clear seasonal cycle = a repeating annual pattern whose amplitude
        # is large relative to the residual scatter. We measure it via the
        # month-of-year profile and how much variance it explains.
        seasonal_strength = 0.0
        season_amp_pct = 0.0
        hi_month = lo_month = None
        deseasonalized = None
        try:
            if isinstance(s.index, pd.DatetimeIndex) and (s.index.max() - s.index.min()).days > 400:
                m_idx = s.index.month
                month_profile = s.groupby(m_idx).mean()
                if len(month_profile) >= 8:
                    season_amp = float(month_profile.max() - month_profile.min())
                    season_amp_pct = season_amp / mean_all * 100 if mean_all else 0.0
                    # Strength: amplitude vs. within-month spread.
                    centered = s - s.index.map(lambda d: month_profile.get(d.month, mean_all))
                    resid_std = float(np.std(centered.values))
                    seasonal_strength = season_amp / (resid_std + 1e-9)
                    hi_month = int(month_profile.idxmax())
                    lo_month = int(month_profile.idxmin())
                    # Deseasonalize the daily series for honest anomaly checks.
                    deseasonalized = centered + mean_all
        except Exception:
            deseasonalized = None

        # "Clear" seasonality: amplitude at least ~8% of the mean AND large
        # relative to scatter.
        clear_seasonal = (season_amp_pct >= 8.0 and seasonal_strength >= 1.5)
        if clear_seasonal:
            _mn = ("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())
            lines.append(
                f"A CLEAR repeating seasonal cycle is present in the power data "
                f"(~{season_amp_pct:.0f}% swing, peaks ~{_mn[hi_month-1]}, "
                f"troughs ~{_mn[lo_month-1]}); the regular dips are seasonal, "
                f"NOT degradation."
            )
        else:
            lines.append("No clear repeating seasonal cycle in the power data.")

        # --- Recent-status check on the DESEASONALIZED series --------------
        # We care about the CURRENT health of the array, not historical blips.
        # So instead of hunting the worst window anywhere in the record, we ask:
        # is the MOST RECENT stretch sitting materially below where the trend
        # says it should be, AND has it failed to recover by the end? Only then
        # do we flag a period worth inspecting. Old, already-recovered dips are
        # intentionally ignored.
        try:
            base = deseasonalized if (clear_seasonal and deseasonalized is not None) else s
            base_monthly = base.resample("MS").mean().dropna()
            m = len(base_monthly)
            if m >= 6:
                bvals = base_monthly.values.astype("float64")
                bx = np.arange(m)
                # Fit the trend on the EARLIER portion only, then see whether
                # the recent months fall below that expectation (a drop the
                # long-run trend doesn't explain).
                hist_n = max(4, m - 3)
                bslope, bint = np.polyfit(bx[:hist_n], bvals[:hist_n], 1)
                expected = bslope * bx + bint
                resid = bvals - expected
                hist_std = float(np.std(resid[:hist_n])) + 1e-9

                recent_k = 3 if m >= 9 else max(2, m // 3)
                recent_resid = float(np.mean(resid[-recent_k:]))
                end_resid = float(resid[-1])

                # Conditions for flagging a CURRENT problem:
                #  (a) recent stretch is materially low vs. expectation
                #      (> ~2.5% of mean below, and > 2 sigma of historical scatter)
                #  (b) it has NOT recovered: the very last point is still low.
                recent_low = (recent_resid < -0.025 * mean_all
                              and recent_resid < -2.0 * hist_std)
                not_recovered = end_resid < -0.015 * mean_all
                if recent_low and not_recovered:
                    w_from = base_monthly.index[m - recent_k].strftime("%Y-%m")
                    w_to = base_monthly.index[-1].strftime("%Y-%m")
                    qualifier = "season-adjusted " if clear_seasonal else ""
                    lines.append(
                        f"RECENT STATUS: the most recent period ({w_from} to "
                        f"{w_to}) sits materially below the {qualifier}trend and "
                        f"has not recovered — worth inspecting."
                    )
                else:
                    lines.append(
                        "RECENT STATUS: the latest period is in line with the "
                        "overall trend; no current drop or unrecovered dip. Do "
                        "NOT call out any specific period."
                    )
            else:
                lines.append(
                    "RECENT STATUS: window too short to assess a recent-only "
                    "anomaly; do not call out a specific period."
                )
        except Exception:
            pass

        return " ".join(lines)
    except Exception as e:
        return f"(trend summary unavailable: {e})"


def _summarize_raw_data(df, mapping):
    """Scan the RAW input channels (power, irradiance, temperature, DC voltage,
    DC current) for data-quality issues the AI diagnostic should surface in
    Advanced mode:
      * coverage gaps -- long stretches with no data, and the overall span,
      * abrupt level shifts -- e.g. a temperature channel that jumps by a
        constant offset/scale (a classic Fahrenheit<->Celsius unit switch),
      * downsampled monthly values per channel so the model can see each
        channel's shape and judge whether a normalized-power trend might be
        driven by irradiance/temperature data rather than the array itself.

    Returns a compact text block. Built from the real (pre-normalization)
    dataframe and the variable mapping."""
    try:
        import numpy as np
        import pandas as pd

        if df is None or mapping is None or len(df) == 0:
            return "Raw-data summary unavailable."

        # Resolve column names from the canonical mapping keys.
        chans = [
            ("Power", mapping.get("DC Power"), "W"),
            ("Irradiance", mapping.get("Irradiance"), "W/m^2"),
            ("Temperature", mapping.get("Module temperature"), "deg"),
            ("DC Voltage", mapping.get("DC Voltage"), "V"),
            ("DC Current", mapping.get("DC Current"), "A"),
        ]

        # Ensure a usable datetime index for gap analysis.
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            time_col = mapping.get("Time")
            if time_col and time_col in df.columns:
                try:
                    idx = pd.to_datetime(df[time_col])
                except Exception:
                    idx = None
            else:
                idx = None

        lines = []

        # --- Overall coverage + gaps (using whichever index we have) --------
        if idx is not None and len(idx) > 2:
            try:
                tt = pd.Series(pd.to_datetime(idx)).sort_values().reset_index(drop=True)
                span_from = tt.iloc[0].strftime("%Y-%m")
                span_to = tt.iloc[-1].strftime("%Y-%m")
                gaps = tt.diff().dt.days.dropna()
                # A "gap" = a break much larger than the typical cadence.
                typical = float(gaps.median()) if len(gaps) else 1.0
                big = gaps[gaps > max(30, typical * 20)]
                gap_txt = ""
                if len(big) > 0:
                    # Report up to 2 largest gaps with their dates.
                    order = big.sort_values(ascending=False).index[:2]
                    parts = []
                    for j in order:
                        start = tt.iloc[j - 1].strftime("%Y-%m")
                        end = tt.iloc[j].strftime("%Y-%m")
                        parts.append(f"{start}->{end} (~{int(big[j])} days)")
                    gap_txt = " Notable coverage gaps: " + "; ".join(parts) + "."
                lines.append(
                    f"Coverage: {span_from} to {span_to}.{gap_txt}"
                )
            except Exception:
                pass

        # --- Per-channel: presence, monthly shape, abrupt shifts ------------
        for label, col, unit in chans:
            if not col or col not in df.columns:
                lines.append(f"{label}: MISSING (no column mapped).")
                continue
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(ser) < 3:
                lines.append(f"{label}: present but almost no valid values.")
                continue

            # Monthly means (downsampled) if we can index by time.
            monthly_txt = ""
            shift_txt = ""
            try:
                if idx is not None:
                    tser = pd.Series(ser.values, index=pd.to_datetime(idx)[:len(df)][ser.index]
                                     if len(idx) == len(df) else pd.to_datetime(idx))
                    tser = tser.dropna()
                    mser = tser.resample("MS").mean().dropna()
                    if len(mser) > 24:
                        step = int(np.ceil(len(mser) / 24))
                        mser = mser.iloc[::step]
                    if len(mser) >= 3:
                        monthly_txt = " monthly: " + ", ".join(
                            f"{i.strftime('%Y-%m')}:{v:.0f}" for i, v in mser.items()
                        )
                        # Abrupt level shift: largest month-to-month jump vs.
                        # the channel's own spread (unit change / step).
                        mv = mser.values.astype("float64")
                        d = np.abs(np.diff(mv))
                        spread = float(np.std(mv)) + 1e-9
                        if len(d) and d.max() > 4 * spread and d.max() > 0.4 * (abs(float(np.mean(mv))) + 1e-9):
                            k = int(np.argmax(d))
                            when = mser.index[k + 1].strftime("%Y-%m")
                            before = float(mv[k]); after = float(mv[k + 1])
                            hint = ""
                            if label == "Temperature":
                                ratio = (after / before) if abs(before) > 1e-6 else 0
                                if 1.5 < ratio < 2.2 or 0.45 < ratio < 0.7:
                                    hint = " (possible F<->C unit change)"
                            shift_txt = (f" ABRUPT SHIFT around {when}: "
                                         f"~{before:.0f}->{after:.0f} {unit}{hint}.")
            except Exception:
                pass

            rng = f"min {float(ser.min()):.0f}, max {float(ser.max()):.0f}, mean {float(ser.mean()):.0f}"

            # Extra unit-change check for temperature: a regime shift between
            # the early and late halves whose ratio looks like F<->C, or a
            # physically implausible spread spanning typical C and F values.
            if label == "Temperature" and not shift_txt:
                try:
                    vals = ser.values.astype("float64")
                    full_range = float(np.nanmax(vals)) - float(np.nanmin(vals))
                    half = len(vals) // 2
                    looks_fc = False
                    early_m = late_m = float(np.mean(vals))
                    if half > 30:
                        early_m = float(np.mean(vals[:half]))
                        late_m = float(np.mean(vals[half:]))
                        ratio = (late_m / early_m) if abs(early_m) > 1e-6 else 1.0
                        looks_fc = (1.5 < ratio < 2.3) or (0.40 < ratio < 0.67)
                    # A module-temp channel spanning >80 deg almost always
                    # means C and F values are mixed in one column.
                    wide_range = full_range > 80
                    if looks_fc or wide_range:
                        shift_txt = (
                            f" POSSIBLE UNIT CHANGE: temperature spans an "
                            f"implausibly wide range (~{float(np.nanmin(vals)):.0f} "
                            f"to ~{float(np.nanmax(vals)):.0f}); the early vs late "
                            f"halves differ (~{early_m:.0f} vs ~{late_m:.0f}) — "
                            f"check for a Fahrenheit/Celsius mix."
                        )
                except Exception:
                    pass

            lines.append(f"{label} ({unit}): {rng}.{monthly_txt}{shift_txt}")

        return " ".join(lines)
    except Exception as e:
        return f"(raw-data summary unavailable: {e})"


# -----------------------------------------------------------------------------
# Enable the Simple-mode Analyze button only once data has been loaded in the
# shared upload area (an upload or an example).
# -----------------------------------------------------------------------------
_EXAMPLE_FRIENDLY = {
    "sys_1278_downsampled_with_VI.parquet": "Example data 1",
    "sys_1403_part1_downsampled_with_VI.parquet": "Example data 2",
    "sys_1422_downsampled.parquet": "Example data 3",
}


def _simple_fail(alert):
    """Common failure return for the data stage (6 outputs)."""
    _none = {"started": True, "data": False, "filter": False,
             "calc": False, "code": False}
    # simple-pipe-data, simple-status, simple-result, simple-stash, step-progress
    return {}, alert, "", {}, _none


# ---- STAGE 1 : load the dataframe + identify variables ----------------------
def _simple_result_layout(stash):
    import plotly.io as pio
    fig = pio.from_json(stash["fig"]) if stash.get("fig") else None
    rate_pct = stash["rate_pct"]
    duration_years = stash["duration_years"]
    start = stash["start"]
    end = stash["end"]
    n_raw = stash.get("n_raw", 0)
    n_kept = stash.get("n_kept", 0)
    pct_kept = stash.get("pct_kept", 0.0)

    def _detail_row(label, value_node):
        return html.Div(
            [html.Span(f"{label}: ", style={"color": INK_SOFT}), value_node],
            style={"fontSize": "16px", "marginBottom": "5px",
                   "fontFamily": "Arial, sans-serif"},
        )

    summary_block = html.Div([
        html.Div("annual degradation rate", style={
            "fontSize": "15px", "color": INK_SOFT, "textTransform": "uppercase",
            "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
            "fontFamily": "Arial, sans-serif",
        }),
        html.Div([
            html.Span(f"{rate_pct:.2%}", style={
                "fontSize": "62px", "fontFamily": "Arial, sans-serif",
                "fontWeight": "700", "color": INK, "lineHeight": "1",
            }),
            html.Span("/year", style={
                "fontSize": "22px", "color": INK_SOFT, "marginLeft": "8px",
                "fontFamily": "Arial, sans-serif", "fontStyle": "italic",
            }),
        ], style={"marginBottom": "18px"}),
        html.Div([
            _detail_row("Method", html.B(stash.get("method", "YoY"), style={"color": VALUE_DETAIL})),
            _detail_row("Duration", html.B(f"{duration_years:.1f} years", style={"color": VALUE_DETAIL})),
            _detail_row("Window", html.B(f"{start}  →  {end}",
                        style={"fontFamily": "Arial, sans-serif", "color": VALUE_DETAIL})),
            _detail_row("Data used", html.Span([
                html.B(f"{n_kept:,}", style={"color": VALUE_DETAIL}),
                html.Span(f" of {n_raw:,} points kept after filtering "
                          f"({pct_kept:.0f}%)", style={"color": INK}),
            ])),
        ], style={"fontFamily": "Arial, sans-serif"}),
        html.Div(
            [
                html.B("Note: "),
                "Default pipeline (YoY). Switch to ",
                html.B("Advanced mode"),
                " to tune any step.",
            ],
            style={"fontSize": "13px", "color": INK_SOFT, "marginTop": "16px",
                   "fontStyle": "italic", "fontFamily": "Arial, sans-serif"},
        ),
    ])

    # Same consolidated data-quality notes toggle as Advanced mode (no
    # irradiance / no temperature / computed power). Built from the detected
    # mapping carried in the stash; collapsed by default so it stays neat.
    _n_notes, notes_component = _data_quality_notes(
        None, stash.get("mapped"), extra_notes=stash.get("mapping_notes"))
    pre_blocks = []
    _rate_banner = _implausible_rate_banner(rate_pct * 100)  # stash holds rate_pct = rd/100
    if _rate_banner is not None:
        pre_blocks.append(_rate_banner)
    if notes_component is not None:
        pre_blocks.append(notes_component)

    return html.Div(pre_blocks + [
        html.Div(summary_block, style={"marginBottom": "20px"}),
        html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False})
                 if fig is not None else html.Div()),
    ], className="slide-in-up", style={
        "padding": "26px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "marginTop": "8px",
    })


# -----------------------------------------------------------------------------
# Status labels for each Simple-mode pipeline stage (shown while it runs).
# -----------------------------------------------------------------------------
_SIMPLE_STEP_LABELS = {
    1: "Inspecting data & identifying variables…",
    2: "Applying default filters…",
    3: "Estimating degradation…",
}


# =============================================================================
# CALLBACK — AI QUICK DIAGNOSTIC (Simple mode)
#
# Feeds the degradation result + a numeric summary of the power-trend figure to
# the LLM and asks for a few short bullet points.  We summarize the figure's
# trend series numerically (start/end level, slope, noise, min/max) rather than
# shipping the image, because the shared client is a text chat client — the
# numbers are what carry the diagnostic signal anyway.
# =============================================================================


# =============================================================================
# VARIABLE SELECTION  (editable variable-mapping table, ported verbatim from
# pvcopilotMaster).  Lets the user override the LLM-detected column mapping in
# Advanced Step 1; the Apply callback (appended at end) rebuilds the mapping
# and redraws the overview figures.
# =============================================================================

def _build_overview_figures_div(df, mapped_variables_dict):
    try:
        if df is not None and mapped_variables_dict:
            figs, _err = make_overview_figures(df, mapped_variables_dict)
            return html.Div(figs)
    except Exception:
        return html.Div("Figure generation failed.", style={"color": ACCENT})
    return html.Div(
        "No variables selected to plot.",
        style={"color": INK_SOFT, "fontSize": "13px",
               "fontFamily": "Arial, sans-serif"},
    )



# Renders the "Identified Variables" panel as an editable table: each metric
# row carries a dropdown so the user can override what the LLM detected (or
# fill it in when the LLM detected nothing). Defaults to the LLM result.
#
# The dropdowns use pattern-matching IDs {"type": "var-map-dd", "metric": m}
# so a single callback (apply_variable_mapping_callback) can read them all.
# =============================================================================

# Metrics shown in the editable mapping table, in display order. "Time" is
# handled specially because it is usually the DataFrame index ("__index__")
# rather than a real column.
_MAP_METRICS = [
    "DC Power", "DC Voltage", "DC Current",
    "Irradiance", "Module temperature", "Time",
]

# Metrics required for degradation analysis (used to flag missing selections).
_REQUIRED_FOR_DEGRADATION = {"DC Power", "Time"}


def _alt_hint(others):
    """A subtle one-line hint listing other valid columns for a role, shown
    directly under that row's dropdown. Returns "" when there's nothing to add."""
    if not others:
        return ""
    return html.Div([
        html.Span("Also valid: ", style={"fontWeight": "600"}),
        html.Span(", ".join(others)),
        html.Span(" — switch above if this isn't the right one.",
                  style={"color": INK_SOFT}),
    ], style={
        "marginTop": "4px", "fontSize": "11.5px", "color": "#0369a1",
        "fontFamily": "Arial, sans-serif", "lineHeight": "1.35",
    })


def build_variable_mapping_table(mapped_variables_dict, columns,
                                 time_in_index=False, status_children=None,
                                 alternatives=None):
    """Build an editable variable-mapping table.

    Args:
        mapped_variables_dict: {metric: column_name} currently mapped (N/A omitted).
        columns: list of available DataFrame column names.
        time_in_index: whether the Time variable is the DataFrame index.
        status_children: optional element rendered in the status slot (used by
            the Apply callback to show the confirmation/warning after re-render).
        alternatives: optional {metric: [other valid column names]} — when a role
            had more than one valid match, the others are shown as a subtle hint
            directly under that row's dropdown so the user can switch if needed.

    Returns:
        A Dash component (the editable table + apply button + status line).
    """
    mapped_variables_dict = mapped_variables_dict or {}
    columns = list(columns or [])
    alternatives = alternatives or {}

    header = html.Div([
        html.Div("Metric", style={
            "flex": "0 0 42%", "fontWeight": "700", "color": INK,
            "fontFamily": "Arial, sans-serif", "fontSize": "14px",
        }),
        html.Div("Variable Name", style={
            "flex": "1", "fontWeight": "700", "color": INK,
            "fontFamily": "Arial, sans-serif", "fontSize": "14px",
        }),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "14px",
        "padding": "8px 4px 12px 4px",
        "borderBottom": f"2px solid {BORDER_STRONG}",
    })

    rows = []
    for i, metric in enumerate(_MAP_METRICS):
        current = mapped_variables_dict.get(metric)

        # Time options: offer the index sentinel plus any real columns.
        if metric == "Time":
            opts = []
            if time_in_index or current == "__index__":
                opts.append({"label": "(use index / __index__)",
                             "value": "__index__"})
            opts += [{"label": c, "value": c} for c in columns]
            # Make sure the current value is selectable even if odd.
            if current and current not in [o["value"] for o in opts]:
                opts.insert(0, {"label": current, "value": current})
        else:
            opts = [{"label": c, "value": c} for c in columns]
            if current and current not in columns:
                # LLM picked something not in the column list — keep it visible.
                opts.insert(0, {"label": current, "value": current})

        detected = bool(current)
        dot_color = "#16a34a" if detected else "#d97706"   # green / amber
        dot_title = "Detected by AI" if detected else "Not detected — please select"

        row = html.Div([
            html.Div([
                html.Span(style={
                    "display": "inline-block", "width": "8px", "height": "8px",
                    "borderRadius": "50%", "background": dot_color,
                    "marginRight": "8px", "flex": "0 0 auto",
                }, title=dot_title),
                html.Span(metric, style={
                    "color": INK, "fontFamily": "Arial, sans-serif",
                    "fontSize": "14px",
                }),
            ], style={"flex": "0 0 42%", "display": "flex",
                      "alignItems": "center"}),
            html.Div([
                dcc.Dropdown(
                    id={"type": "var-map-dd", "metric": metric},
                    options=opts,
                    value=current if current else None,
                    placeholder="— select column —",
                    clearable=True,
                    style={"fontSize": "13px"},
                ),
                # Inline hint: other valid columns the user could switch to.
                _alt_hint(alternatives.get(metric)),
            ], style={"flex": "1"}),
        ], style={
            "display": "flex", "alignItems": "flex-start", "gap": "14px",
            "padding": "8px 4px",
            "background": "#ffffff" if i % 2 else "#f1f5f9",
            "borderRadius": "6px",
        })
        rows.append(row)

    return html.Div([
        header,
        html.Div(rows, style={"marginTop": "4px"}),
        html.Div([
            html.Button(
                "Apply mapping",
                id="var-map-apply-btn",
                n_clicks=0,
                style={
                    "background": ACCENT, "color": "#ffffff", "border": "none",
                    "padding": "8px 18px", "borderRadius": "8px",
                    "fontSize": "13px", "fontWeight": "600", "cursor": "pointer",
                    "fontFamily": "Arial, sans-serif",
                },
            ),
            html.Span(
                "Defaults to AI detection. Adjust any row and click Apply.",
                style={"marginLeft": "12px", "fontSize": "12px",
                       "color": INK_SOFT, "fontFamily": "Arial, sans-serif"},
            ),
        ], style={"marginTop": "14px", "display": "flex",
                  "alignItems": "center"}),
        html.Div(status_children if status_children is not None else "",
                 id="var-map-status", style={"marginTop": "8px"}),
    ])


# =============================================================================
# CALLBACK — APPLY USER-EDITED VARIABLE MAPPING
#
# Reads every var-map dropdown and rebuilds mapped-vars-store using ONLY the
# variables the user actually selected. Unselected variables are dropped from
# the mapping, so they are neither plotted (make_overview_figures skips absent
# keys) nor used in any subsequent analysis (every downstream step reads
# mapped-vars-store). The mapping panel is re-rendered so the detected/
# undetected dots reflect the applied state, and the figures are redrawn to
# remove any de-selected variable.
# =============================================================================


layout = html.Div([

    # Hidden stores (unchanged)
    dcc.Store(id="mapped-vars-store",     data={}),
    # Available DataFrame columns, used to populate the editable
    # variable-mapping dropdowns in Advanced Step 1.
    dcc.Store(id="data-columns-store",    data=[]),
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

    # Time span (in years) of the most recently analyzed dataset. Computed once
    # at Analyze time and reused to (a) hard-block degradation for <1yr data and
    # (b) disable the YoY method for <2yr data. None until a dataset is analyzed.
    dcc.Store(id="data-duration-store", data=None),

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

    # ── NEW: stores/intervals for the mode system + Simple-mode pipeline ──
    # Active analysis mode: "simple" (default) or "advanced".
    dcc.Store(id="ui-mode", data="simple"),
    # Simple-mode staged reveal + chained-pipeline intermediates.
    dcc.Store(id="simple-stash", data={}),
    dcc.Store(id="simple-pipe-data", data={}),
    dcc.Store(id="simple-pipe-filtered", data={}),
    dcc.Store(id="simple-run-trigger", data={}),
    # Simple-mode PVPRO: its own job tracker + poll interval, separate from
    # the Advanced-mode ones above so the two can never collide.
    dcc.Store(id="simple-pvpro-job", data={}),
    dcc.Interval(id="simple-pvpro-poll-interval", interval=1000,
                 n_intervals=0, disabled=True),

    # Main container — sidebar (left) always visible; the right column holds
    # the shared upload header, the Simple/Advanced tab switcher, and whichever
    # mode's content is active.
    dbc.Container(
        [
            html.Div(
                [
                    # LEFT — progress panel / sidebar (always visible in both modes)
                    html.Div(id="sidebar-render", children=sidebar),

                    # RIGHT — fixed-height scroll column.
                    html.Div(
                        [
                            html.Div(
                                [
                                    # Shared editorial header (above the tabs).
                                    common_header,

                                    # Shared data-upload area — common to both modes.
                                    shared_upload_header,

                                    # ── ANALYZE card: title + mode tabs + active panel ──
                                    html.Div(
                                        [
                                            html.Div("ANALYZE", style={
                                                "fontSize": "15px", "color": ACCENT,
                                                "fontWeight": "800", "fontFamily": "Arial, sans-serif",
                                                "textTransform": "uppercase", "letterSpacing": "0.12em",
                                                "marginBottom": "14px",
                                            }),
                                            html.Div(
                                                "Pick a mode, then run the analysis on "
                                                "the data you loaded above.",
                                                style={"fontSize": "15px", "color": INK_SOFT,
                                                       "fontFamily": "Arial, sans-serif",
                                                       "marginBottom": "16px"},
                                            ),

                                            # Mode switcher — sticky inside this card.
                                            html.Div(
                                                html.Div(id="mode-tabs-render",
                                                         children=build_mode_tabs("simple")),
                                                style={
                                                    "position": "sticky",
                                                    "top": "0",
                                                    "zIndex": "20",
                                                    "background": PAPER_RAISED,
                                                    "padding": "6px 0",
                                                    "marginBottom": "4px",
                                                },
                                            ),

                                            # SIMPLE-MODE PANEL — visible by default.
                                            html.Div(
                                                simple_mode_panel,
                                                id="simple-mode-wrap",
                                                style={},
                                            ),

                                            # ADVANCED-MODE content — the original
                                            # four-agent chat_stream, hidden until switch.
                                            html.Div(
                                                chat_stream,
                                                id="advanced-mode-wrap",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        style={
                                            "padding": "32px 64px 36px",
                                            "background": PAPER_RAISED,
                                            "border": f"1px solid {BORDER}",
                                            "borderRadius": "10px",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                ],
                                style={
                                    "flex": "1",
                                    "minHeight": "0",
                                    "overflowY": "auto",
                                    "paddingRight": "4px",
                                    "paddingBottom": "4px",
                                },
                            ),
                        ],
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "display": "flex",
                            "flexDirection": "column",
                            "height": "calc(100vh - 110px)",
                        },
                    ),
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
# CALLBACK — DISABLE YoY FOR DATASETS UNDER 2 YEARS
#
# YoY pairs each day with the same calendar day one year earlier, so it needs
# at least two years of data to yield a single comparison. When the analyzed
# dataset is shorter, grey out the YoY option and, if it was selected, fall the
# selection back to LR (the clientside sync above then mirrors it into the
# hidden master radio). Fires whenever a new dataset is analyzed (the duration
# store changes), which also resets the embedded per-method param inputs to
# their defaults — consistent with the existing reset-on-new-data behavior.
# =============================================================================
@app.callback(
    Output("metric-stat-radio", "options", allow_duplicate=True),
    Output("metric-stat-radio", "value",   allow_duplicate=True),
    Output("yoy-disabled-note", "children"),
    Output("yoy-disabled-note", "style"),
    Input("data-duration-store", "data"),
    State("metric-stat-radio",   "value"),
    prevent_initial_call=True,
)
def gate_yoy_by_duration(duration_years, current_value):
    disable_yoy = duration_years is not None and duration_years < 2.0
    options = build_stat_metric_options(disable_yoy=disable_yoy)
    if disable_yoy:
        new_value = "LR" if current_value == "YOY" else dash.no_update
        note = "YoY needs at least 2 years of data; it's disabled for this dataset."
        note_style = {"fontSize": "12px", "color": "#92400e", "fontStyle": "italic",
                      "marginTop": "8px", "fontFamily": "Arial, sans-serif"}
    else:
        new_value = dash.no_update
        note = ""
        note_style = {"display": "none"}
    return options, new_value, note, note_style


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

    # Power is the minimum requirement. parse_contents already computed
    # power = V * I when only voltage+current were present, so by here a
    # missing DC Power means we genuinely cannot analyze.
    power_key = mapped_variables_dict.get("DC Power") if mapped_variables_dict else None
    if not power_key or power_key not in df.columns:
        return [_no_data_alert("Cannot analyze: no DC Power column found, and no Voltage + Current to compute it from. Provide power, or voltage and current."), None]

    irra_key = mapped_variables_dict.get("Irradiance") if mapped_variables_dict else None
    has_irr  = bool(irra_key) and irra_key in df.columns
    temp_key = mapped_variables_dict.get("Module temperature") if mapped_variables_dict else None
    has_temp = bool(temp_key) and temp_key in df.columns

    gamma          = gamma if gamma is not None else -0.004
    irr_thresh     = irr_thresh if irr_thresh is not None else 300
    power_ratio    = power_ratio if power_ratio is not None else 0.02
    norm_lower     = norm_lower if norm_lower is not None else 0.01
    norm_upper_pct = norm_upper_pct if norm_upper_pct is not None else 99

    # The filtering pipeline (clear-sky, low-irradiance, IQR, normalization) is
    # the heavy part of this step; cap it so a pathological dataset can't hang
    # the UI. Figure assembly below is cheap and stays outside the timed region.
    def _do_filter():
        # Basic value filter
        bv_normal, bv_outlier = basic_value_filter(df, mapped_variables_dict)
        _df = df.loc[bv_normal].copy()

        clearsky_mask = pd.Series(True, index=_df.index)
        if "clearsky" in selected_filters and has_irr:
            _cs_smooth = cs_smooth if cs_smooth is not None else 0.3
            _cs_energy = cs_energy if cs_energy is not None else 0.5
            normal_idx, outlier_idx = clear_sky_filter(_df, irra_key,
                                                        smoothness_threshold=_cs_smooth,
                                                        energy_threshold=_cs_energy)
            clearsky_mask = _df.index.isin(normal_idx)

        _df_filtered = normalize(_df, mapped_variables_dict, gamma=gamma)
        _current_mask = pd.Series(clearsky_mask, index=_df_filtered.index)
        _filter_stats = []

        # Data-availability notices (missing power/irradiance/temperature) are now
        # consolidated into the collapsible "data-quality notes" toggle shown right
        # after Analyze — see _data_quality_notes(). Only filter-action stats
        # (what each selected filter actually did) are reported here.

        if "timezone" in selected_filters:
            try:
                _df_filtered.index = pd.to_datetime(_df_filtered.index)
                _df_filtered.index = _df_filtered.index.tz_localize("UTC").tz_convert("US/Pacific")
                _filter_stats.append("Timezone corrected (UTC → US/Pacific)")
            except Exception:
                _filter_stats.append("⚠️ Timezone correction failed")

        if "clearsky" in selected_filters:
            if has_irr:
                removed = (~clearsky_mask).sum()
                _filter_stats.append(f"Clear-sky filter removed {removed} points")
            else:
                _filter_stats.append("⚠️ Clear-sky filter skipped — requires irradiance.")

        if "low-irra-power" in selected_filters and has_irr:
            normal_idx, outlier_idx = low_irra_power_filter(
                _df_filtered, mapped_variables_dict,
                irr_thresh=irr_thresh, power_ratio=power_ratio,
                norm_lower=norm_lower, norm_upper_pct=norm_upper_pct
            )
            mask = _df_filtered.index.isin(normal_idx)
            removed = (~mask & _current_mask).sum()
            _current_mask &= mask
            _filter_stats.append(f"Low irra-power filter removed {removed} points")
        elif "low-irra-power" in selected_filters and not has_irr:
            # No irradiance: drop night / low-output points by power alone so
            # the trend isn't contaminated by uncleaned low-light readings.
            normal_idx, _ = low_power_filter(_df_filtered, mapped_variables_dict)
            mask = _df_filtered.index.isin(normal_idx)
            removed = (~mask & _current_mask).sum()
            _current_mask &= mask
            _filter_stats.append(f"Low-power filter (no irradiance) removed {removed} points")

        if "outlier" in selected_filters:
            _iqr = iqr_multiplier if iqr_multiplier is not None else 1.5
            # Compute IQR fences on the points that survived prior filters (not
            # the full frame) so night/low-output values don't dominate the
            # distribution and flag real daytime production as outliers.
            _kept = _df_filtered.loc[_df_filtered.index[_current_mask]]
            normal_idx, outlier_idx = identify_outliers_iqr(_kept, "norm", iqr_multiplier=_iqr)
            mask = _df_filtered.index.isin(normal_idx)
            removed = (~mask & _current_mask).sum()
            _current_mask &= mask
            _filter_stats.append(f"IQR outlier filter removed {removed} points")

        return _df_filtered, _current_mask, _filter_stats

    try:
        df_filtered, current_mask, filter_stats = _run_with_timeout(_do_filter)
    except FutureTimeout:
        return [_no_data_alert("Filtering is taking longer than expected — something may be wrong "
                               "with your data. Check the file's formatting and columns, then try again."),
                None]

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

    # Power is required; irradiance is optional (un-normalized if absent).
    power_key = mapped_variables_dict.get("DC Power") if mapped_variables_dict else None
    if not power_key or power_key not in df_filtered.columns:
        return [_no_data_alert("Cannot run degradation: no DC Power (and no Voltage + Current to compute it)."),
                "", False, "Calculate Degradation", {}, {}, True]

    # Hard block: degradation analysis is meaningless on under a year of data.
    # The user is warned earlier in the Analyze output; this enforces it for
    # every method (statistical and PVPRO alike).
    _dur = _duration_years(df_filtered)
    if _dur is not None and _dur < 1.0:
        _months = int(round(_dur * 12))
        return [_no_data_alert(f"This dataset spans only about {_months} month"
                               f"{'s' if _months != 1 else ''} (less than 1 year). "
                               "Degradation analysis needs at least 1 year of data."),
                "", False, "Calculate Degradation", {}, {}, True]

    irra_key = mapped_variables_dict.get("Irradiance") if mapped_variables_dict else None
    if not irra_key or irra_key not in df_filtered.columns:
        irra_key = None   # aggregate_daily falls back to a simple daily mean

    # ---------- PVPRO: long-running, so launch in a thread and let a polling
    # ---------- callback render the result when it's ready.
    if selected_metric == "PVPRO":
        # PVPRO needs the full physical input set; warn (don't crash) if short.
        _missing = []
        for _label in ("DC Voltage", "DC Current", "Irradiance", "Module temperature"):
            _col = mapped_variables_dict.get(_label) if mapped_variables_dict else None
            if not _col or _col not in df_filtered.columns:
                _missing.append(_label)
        if _missing:
            return [_no_data_alert("PVPRO needs " + ", ".join(_missing)
                                   + ". Use a statistical method (YoY / LR / HW / ARIMA / CSD) instead."),
                    "", False, "Calculate Degradation", {}, {}, True]
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
        # Fast statistical methods run synchronously; cap them so a bad dataset
        # can't hang the UI. (PVPRO, above, has its own background-job path and
        # is deliberately not capped.)
        def _compute_fast():
            daily_data = aggregate_daily(df_filtered, irra_key)

            def _run(m):
                if m == "YOY":
                    return compute_yoy(daily_data,
                                       rolling_window=yoy_window if yoy_window else 30,
                                       iqr_multiplier=yoy_iqr if yoy_iqr else 1.5)
                elif m == "LR":
                    return compute_lr(daily_data)
                elif m == "HW":
                    return compute_hw(daily_data, period=hw_period if hw_period else 12)
                elif m == "ARIMA":
                    return compute_arima(daily_data,
                                         p=arima_p if arima_p is not None else 1,
                                         d=arima_d if arima_d is not None else 1,
                                         q=arima_q if arima_q is not None else 0,
                                         seasonal_period=arima_s if arima_s else 12)
                elif m == "CSD":
                    return compute_csd(daily_data, period=csd_period if csd_period else 12)
                else:
                    raise ValueError(f"Unknown metric: {m}")

            rd, fig = _run(selected_metric)
            used = selected_metric
            # If the chosen method can't produce a rate (e.g. YoY on sparse,
            # irregular data), fall back to Linear Regression rather than NaN.
            if (rd is None or not np.isfinite(rd)) and selected_metric != "LR":
                rd_lr, fig_lr = _run("LR")
                if rd_lr is not None and np.isfinite(rd_lr):
                    rd, fig, used = rd_lr, fig_lr, "LR"
            return rd, fig, used

        try:
            rd, fig, _used_metric = _run_with_timeout(_compute_fast)
        except FutureTimeout:
            return [_no_data_alert("Calculating degradation is taking longer than expected — "
                                   "something may be wrong with your data. Check the file's "
                                   "formatting and columns, then try again."),
                    "", False, "Calculate Degradation", {}, {}, True]

        # Reflect the method actually used (in case of fallback) in the result.
        if _used_metric != selected_metric:
            selected_metric = _used_metric

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

    # The "no irradiance → un-normalized, less reliable" caveat is now shown in
    # the consolidated data-quality notes toggle right after Analyze (see
    # _data_quality_notes()). Here we additionally flag a rate that came out
    # physically implausible (positive / very large) so the user doesn't trust it.
    _rate_banner = _implausible_rate_banner(rd)

    degradation_layout = html.Div(
        ([_rate_banner] if _rate_banner is not None else []) + [
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
    Output("data-duration-store",  "data",      allow_duplicate=True),
    Output("data-columns-store",   "data",      allow_duplicate=True),
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
                    {}, None, "", False, "Analyze Data", None, "", example_filename, None, [])
        # Duration is (re)computed when the user clicks Analyze; reset it here so a
        # freshly loaded example can't be gated by the previous dataset's span.
        return (html.Div("", className="text-muted"),
                {}, df_json, "", False, "Analyze Data", "example", output_msg, example_filename, None, [])

    # Analyze clicked
    if trigger == "analyze-btn":
        # Parsing (which includes an LLM column-identification call) is capped at
        # STEP_TIMEOUT_S so a malformed file can't hang the UI indefinitely.
        try:
            if data_source == "upload" and contents is not None:
                df, summary_table, mapped_variables_dict, code_read, mapping_notes = _run_with_timeout(
                    parse_contents, contents, filename)
                if df is None:
                    return summary_table, {}, None, "", False, "Analyze Data", dash.no_update, dash.no_update, stored_file_name, None, []
            elif data_source == "example" and stored_df_json is not None:
                df = _df_from_store(stored_df_json)
                df, summary_table, mapped_variables_dict, code_read, mapping_notes = _run_with_timeout(
                    parse_contents, df=df)
            else:
                return (_no_data_alert("Please upload a file or click an example button, then click 'Analyze Data'."),
                        {}, None, "", False, "Analyze Data", None, "", filename, None, [])
        except FutureTimeout:
            return (_no_data_alert("This is taking longer than expected — something may be wrong "
                                   "with your data. Check the file's formatting and columns, then try again."),
                    {}, None, "", False, "Analyze Data", dash.no_update, dash.no_update, stored_file_name, None, [])
        except Exception as e:
            return (html.Div(f"Error processing dataset: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Analyze Data", dash.no_update, dash.no_update, stored_file_name, None, [])

        try:
            df_json = df.to_json(date_format="iso", orient="split")
        except Exception as e:
            return (html.Div(f"Error converting DataFrame: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Analyze Data", dash.no_update, dash.no_update, stored_file_name, None, [])

        # Dataset time span — drives the <1yr block (below) and the <2yr YoY disable.
        duration_years = _duration_years(df)
        duration_store = round(duration_years, 3) if duration_years is not None else None

        # Figures (also capped — large frames can make plotting slow).
        figures_output = html.Div()
        try:
            if df is not None and mapped_variables_dict:
                figures_output, err = _run_with_timeout(
                    make_overview_figures, df, mapped_variables_dict)
                figures_output = html.Div(figures_output)
        except FutureTimeout:
            figures_output = html.Div("Preview figures took too long to render and were skipped.",
                                      style={"color": ACCENT})
        except Exception:
            figures_output = html.Div("Figure generation failed.", style={"color": ACCENT})

        # Consolidated, collapsible data-quality notes + (if <1yr) a hard-block banner,
        # surfaced right after analysis so the user sees implications before calculating.
        # Column-ambiguity ("also valid: …") hints live ONLY inline under each
        # dropdown (see build_variable_mapping_table); they're intentionally not
        # in mapping_notes, so the consolidated panel keeps the other caveats only.
        _n_notes, notes_component = _data_quality_notes(
            df, mapped_variables_dict, extra_notes=mapping_notes)
        pre_blocks = []
        if duration_years is not None and duration_years < 1.0:
            pre_blocks.append(_duration_block_banner(duration_years))
        if notes_component is not None:
            pre_blocks.append(notes_component)

        # Available columns + Time-in-index flag for the editable mapping table.
        data_columns = [str(c) for c in df.columns.tolist()]
        time_in_index = (
            isinstance(df.index, pd.DatetimeIndex)
            or (mapped_variables_dict or {}).get("Time") == "__index__"
        )
        # Editable variable-mapping table (defaults to the LLM detection;
        # the user can override any row or fill in ones the LLM missed). When a
        # role had several valid matches, the others are offered inline under
        # that row's dropdown (parse_contents stashed them on df.attrs).
        alternatives = df.attrs.get("mapping_alternatives", {}) if df is not None else {}
        editable_mapping = build_variable_mapping_table(
            mapped_variables_dict, data_columns, time_in_index=time_in_index,
            alternatives=alternatives,
        )

        combined_output = html.Div(pre_blocks + [
            html.Div([
                html.Div("identified variables", style={
                    "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
                    "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
                    "fontFamily": "Arial, sans-serif",
                }),
                # Editable mapping table — replaces the old read-only list.
                html.Div(editable_mapping, id="var-map-panel",
                         style={"fontSize": "14px"}),
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
                html.Div(figures_output, id="var-map-figures"),
            ], style={
                "padding": "18px 20px",
                "background": "#f8fafc",
                "border": f"1px solid {BORDER}",
                "borderRadius": "10px",
            }),
        ], className="slide-in-up")

        # Preserve data-source-store and the upload-status banner (dash.no_update)
        # so the loaded file's name stays visible after analyzing AND the source
        # stays "upload"/"example" — otherwise the same file can't be re-analyzed.
        return (combined_output, mapped_variables_dict, df_json, code_read, False,
                "Analyze Data", dash.no_update, dash.no_update, stored_file_name,
                duration_store, data_columns)

    return ("", {}, None, "", False, "Analyze Data", dash.no_update, dash.no_update, stored_file_name, None, [])


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


# =============================================================================
# CALLBACKS FOR THE MODE SYSTEM + SIMPLE-MODE PIPELINE  (merged from Master)
# =============================================================================

@app.callback(
    Output("simple-pvpro-params-wrap", "style"),
    Output("simple-pvpro-about-wrap",  "style"),
    Input("simple-method-radio", "value"),
    prevent_initial_call=True,
)
def simple_toggle_pvpro_params(method):
    if method == "PVPRO":
        return ({"display": "block", "marginTop": "8px"},
                {"display": "block"})
    return ({"display": "none"}, {"display": "none"})


# =============================================================================
# CALLBACK — SIMPLE-MODE PVPRO: poll the background job and render the result
# into `simple-result` (the shared result area).  The job is launched from the
# shared Stage-3 pipeline callback when the chosen method is PVPRO.
# =============================================================================
@app.callback(
    Output("simple-pvpro-progress-output", "children", allow_duplicate=True),
    Output("simple-result",                "children", allow_duplicate=True),
    Output("simple-status",                "children", allow_duplicate=True),
    Output("simple-pvpro-job",             "data",     allow_duplicate=True),
    Output("simple-pvpro-poll-interval",   "disabled", allow_duplicate=True),
    Output("step-progress",                "data",     allow_duplicate=True),
    Output("simple-stash",                 "data",     allow_duplicate=True),
    Input("simple-pvpro-poll-interval", "n_intervals"),
    State("simple-pvpro-job", "data"),
    State("simple-pipe-filtered", "data"),
    prevent_initial_call=True,
)
def simple_pvpro_poll(_n, job_store, pfiltered):
    from dash import no_update

    job_id = (job_store or {}).get("job_id")
    if not job_id:
        return [no_update] * 7

    job = _pvpro_read_job(job_id)
    if job is None:
        return [no_update] * 7

    phase = job.get("phase", "")
    elapsed = max(0.0, time.time() - job.get("started_at", time.time()))

    # Still working.
    if phase not in ("done", "error", "rendered"):
        ui = _pvpro_progress_ui(
            phase=phase, current=job.get("current", 0),
            total=job.get("total", 1), message=job.get("message", ""),
            elapsed_s=elapsed,
        )
        return [ui, no_update, no_update, job_store, False, no_update, no_update]

    # Failed.
    if phase == "error":
        _pvpro_update_job(job_id, phase="rendered")
        none = {"started": True, "data": True, "filter": True,
                "calc": False, "code": False}
        return ["", _no_data_alert(
            f"PVPRO failed: {job.get('error', 'unknown error')}"),
            "", {}, True, none, {}]

    # Done / already-rendered.
    render_lock = _pvpro_get_render_lock(job_id)
    if not render_lock.acquire(blocking=False):
        return [no_update] * 7
    try:
        job = _pvpro_read_job(job_id)
        if job is None or job.get("phase") == "rendered":
            return [no_update] * 7

        result = job.get("result") or {}
        rd = result.get("rd", float("nan"))
        figs = result.get("figs") or {}
        rates = result.get("rates", {}) or {}

        src_name = (pfiltered or {}).get("source_name", "Your data")
        try:
            df_f = _df_from_store((pfiltered or {}).get("df_good"))
            start_date = df_f.index.min()
            end_date = df_f.index.max()
            duration_years = (end_date - start_date).days / 365.25
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        except Exception:
            start_str = end_str = "\u2014"
            duration_years = 0.0

        layout_div = _render_pvpro_layout(
            rd, figs, rates, start_str, end_str, duration_years, elapsed)

        # Build a diagnostic-friendly stash so the AI assistant works for the
        # PVPRO result too (mirrors the Advanced-mode PVPRO summary).
        rate_pct = (rd / 100) if (rd is not None and np.isfinite(rd)) else 0.0
        _QUANT = [("p_mp_ref", "Pmp"), ("v_mp_ref", "Vmp"), ("i_mp_ref", "Imp"),
                  ("v_oc_ref", "Voc"), ("i_sc_ref", "Isc")]
        if np.isfinite(rate_pct):
            _parts = [f"PVPRO-fitted reference-condition degradation. "
                      f"Pmp(ref): {rate_pct*100:+.2f}%/yr over {duration_years:.1f} years."]
            _bits = [f"{s}(ref) {rates[k]:+.2f}%/yr" for k, s in _QUANT
                     if np.isfinite(rates.get(k, float('nan')))]
            if _bits:
                _parts.append("Per-parameter rates: " + ", ".join(_bits) + ".")
            trend_summary = " ".join(_parts)
        else:
            trend_summary = "PVPRO fit did not return a finite Pmp rate."

        # Per-parameter reference rates the PVPRO diagnostic branch consumes.
        rates_per_quantity = {k: round(float(v), 4)
                              for k, v in (rates or {}).items()
                              if v is not None and np.isfinite(v)}

        # Raw-channel data-quality scan (same as Advanced mode) so the
        # diagnostic's data-findings bullet has something to work with.
        try:
            raw_summary = _summarize_raw_data(
                _df_from_store((pfiltered or {}).get("df_good")),
                (pfiltered or {}).get("mapped") or {},
            )
        except Exception as _e:
            raw_summary = f"(raw-data summary unavailable: {_e})"

        stash = {
            "rate_pct": float(rate_pct) if np.isfinite(rate_pct) else 0.0,
            "method": "PVPRO",
            "duration_years": float(duration_years),
            "start": start_str,
            "end": end_str,
            "source_name": src_name,
            "n_raw": 0, "n_kept": 0, "n_removed": 0, "pct_kept": 100.0,
            "rates_per_quantity": rates_per_quantity,
            "raw_summary": raw_summary,
            "trend_summary": trend_summary,
            "fig": None,
        }

        done_status = _success_banner(f"{src_name} analysis complete (PVPRO)")
        progress = {"started": True, "data": True, "filter": True,
                    "calc": True, "code": False}
        _pvpro_update_job(job_id, phase="rendered")
        return ["", layout_div, done_status, {}, True, progress, stash]
    finally:
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
    Output("step-progress", "data", allow_duplicate=True),
    Input("analyze-btn", "n_clicks"),
    State("step-progress", "data"),
    prevent_initial_call=True,
)
def mark_started_advanced(n_clicks, prev):
    if not n_clicks:
        return dash.no_update
    prog = dict(prev or {})
    prog["started"] = True
    return prog


# =============================================================================
# CALLBACK — SHOW / HIDE AGENT MESSAGES BASED ON PROGRESS
# Each subsequent agent becomes visible only when the previous step is done.
# =============================================================================
@app.callback(
    Output("ui-mode", "data"),
    Input({"type": "mode-tab", "mode": ALL}, "n_clicks"),
    State("ui-mode", "data"),
    prevent_initial_call=True,
)
def set_ui_mode(_clicks, current):
    trigger = ctx.triggered_id
    if not trigger or not isinstance(trigger, dict):
        return current or "simple"
    # Ignore the spurious initial 0-click fire.
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return current or "simple"
    return trigger.get("mode", current or "simple")


@app.callback(
    Output("simple-mode-wrap",   "style"),
    Output("advanced-mode-wrap", "style"),
    Output("mode-tabs-render",   "children"),
    Input("ui-mode", "data"),
)
def render_mode(mode):
    mode = mode or "simple"
    simple_style   = {} if mode == "simple" else {"display": "none"}
    advanced_style = {} if mode == "advanced" else {"display": "none"}
    return simple_style, advanced_style, build_mode_tabs(mode)


# =============================================================================
# CALLBACK — SIMPLE-MODE END-TO-END PIPELINE
#
# When a file is dropped (or an example chip clicked) in Simple mode, run the
# entire pipeline with default settings and show ONLY the degradation rate +
# figure.  Reuses the exact same compute functions as Advanced mode, so the
# numbers match Advanced-with-defaults.
#
#   parse_contents -> basic_value_filter -> normalize -> default filters
#                  -> aggregate_daily -> compute_yoy
# =============================================================================


@app.callback(
    Output("simple-analyze-btn", "disabled"),
    Output("simple-analyze-btn", "style"),
    Input("data-source-store", "data"),
    Input("dataframe-store",   "data"),
    Input("upload-data",       "contents"),
    Input("mapped-vars-store", "data"),
    Input("ui-mode",           "data"),
)
def toggle_simple_analyze(data_source, df_store, upload_contents, mapped_vars, mode):
    # Ready whenever data is loaded in the shared area — an uploaded file
    # (contents present), an example (df in store), or anything Advanced mode
    # has already parsed (mapped vars present).  Re-checked on mode switch so
    # returning to Simple after using Advanced leaves the button enabled.
    ready = (
        bool(upload_contents) or
        (data_source == "example" and bool(df_store)) or
        bool(df_store) or
        bool(mapped_vars)
    )
    return (not ready), _simple_analyze_style(disabled=not ready)


# -----------------------------------------------------------------------------
# Simple mode, STAGE A (instant): on Analyze click, immediately show a status
# banner and write the run trigger.  Returns right away so the banner paints
# with no perceptible delay; the heavy compute happens in Stage B below.
# -----------------------------------------------------------------------------
@app.callback(
    Output("simple-status",       "children", allow_duplicate=True),
    Output("simple-result",       "children", allow_duplicate=True),
    Output("simple-run-trigger",  "data"),
    Output("step-progress",       "data", allow_duplicate=True),
    Input("simple-analyze-btn",   "n_clicks"),
    State("simple-run-trigger",   "data"),
    State("simple-method-radio",  "value"),
    prevent_initial_call=True,
)
def simple_start(n_clicks, prev_trigger, method):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    seq = ((prev_trigger or {}).get("seq", 0)) + 1
    method = method or "YOY"
    label = "Reading data…" if method == "YOY" else "Reading data for PVPRO…"
    return (
        _working_banner(label),
        "",   # clear any prior result immediately
        {"source": "simple-analyze-btn", "seq": seq, "method": method},
        {"started": True, "data": False, "filter": False, "calc": False, "code": False},
    )


# =============================================================================
# Simple-mode pipeline — THREE CHAINED STAGES.
# Each stage runs one real part of the pipeline, then writes the store that
# triggers the next stage.  After each stage finishes it marks its sidebar
# step "done" and the next step "active", so the sidebar advances exactly as
# each real stage completes.
# =============================================================================
@app.callback(
    Output("simple-pipe-data", "data"),
    Output("simple-status",    "children", allow_duplicate=True),
    Output("simple-result",    "children", allow_duplicate=True),
    Output("simple-stash",     "data", allow_duplicate=True),
    Output("step-progress",    "data", allow_duplicate=True),
    Input("simple-run-trigger",   "data"),
    State("upload-data",          "contents"),
    State("upload-data",          "filename"),
    State("dataframe-store",      "data"),
    State("data-source-store",    "data"),
    State("stored-data-file-name","data"),
    prevent_initial_call=True,
)
def simple_stage_data(run_trigger, contents, filename, stored_df_json,
                      data_source, stored_file_name):
    if not (run_trigger or {}).get("source"):
        return (dash.no_update,) * 5

    # Load + identify variables.
    try:
        if data_source == "example" and stored_df_json:
            df_raw = _df_from_store(stored_df_json)
            df, summary_table, mapped, code_read, mapping_notes = parse_contents(df=df_raw)
            source_name = _EXAMPLE_FRIENDLY.get(stored_file_name, "Example data")
        elif data_source == "upload" and contents is not None:
            df, summary_table, mapped, code_read, mapping_notes = parse_contents(contents, filename)
            source_name = filename or "your file"
        else:
            return _simple_fail(_no_data_alert(
                "Please upload a file or pick an example above first."))
    except Exception as e:
        return _simple_fail(_no_data_alert(f"Could not read the data: {e}"))

    if df is None or not mapped:
        return _simple_fail(_no_data_alert(
            "Couldn't identify the required columns automatically. "
            "Try Advanced mode to map variables manually."
        ))

    # Block under-a-year data up front, exactly like Advanced mode.
    _dur = _duration_years(df)
    if _dur is not None and _dur < 1.0:
        return _simple_fail(_duration_block_banner(_dur))

    # Power is REQUIRED; irradiance and temperature are OPTIONAL. Like Advanced
    # mode, we proceed without them and surface a data-quality note (shown in the
    # result) rather than refusing to run.
    power_key = mapped.get("DC Power")
    if not power_key or power_key not in df.columns:
        return _simple_fail(_no_data_alert(
            "No DC Power column found, and no Voltage + Current to compute it from. "
            "Try Advanced mode to map variables manually."
        ))

    irra_key = mapped.get("Irradiance")
    if not irra_key or irra_key not in df.columns:
        irra_key = None   # un-normalized; flagged in the data-quality notes

    # If PVPRO was chosen, it needs DC Voltage + DC Current.  Check right after
    # identification (Step 1) and fail fast with a clear pointer back to YoY.
    method = (run_trigger or {}).get("method", "YOY")
    if method == "PVPRO" and not _simple_has_dc_vi(mapped):
        return _simple_fail(_no_data_alert(
            "PVPRO needs DC voltage and DC current columns, but they weren't "
            "found in this dataset. Switch the method to YoY above, or try "
            "Advanced mode to map the columns manually."
        ))

    # Success: Step 1 DONE, Step 2 ACTIVE.  Pass the loaded df + meta forward.
    payload = {
        "df": df.to_json(date_format="iso", orient="split"),
        "mapped": mapped,
        "mapping_notes": mapping_notes,
        "irra_key": irra_key,
        "source_name": source_name,
        "n_raw": int(len(df)),
        "seq": run_trigger.get("seq", 0),
        "method": method,
    }
    progress = {"started": True, "data": True, "filter": False,
                "calc": False, "code": False}
    status = _working_banner(_SIMPLE_STEP_LABELS.get(2, "Applying default filters…"))
    return payload, status, "", {}, progress


# ---- STAGE 2 : default filtering -------------------------------------------
@app.callback(
    Output("simple-pipe-filtered", "data"),
    Output("simple-status",        "children", allow_duplicate=True),
    Output("step-progress",        "data", allow_duplicate=True),
    Input("simple-pipe-data", "data"),
    prevent_initial_call=True,
)
def simple_stage_filter(pdata):
    if not pdata or "df" not in pdata:
        return dash.no_update, dash.no_update, dash.no_update

    # Brief pause so the "Applying default filters…" step is visible.
    time.sleep(2)

    def _fail(alert):
        none = {"started": True, "data": True, "filter": False,
                "calc": False, "code": False}
        return {}, alert, none

    try:
        df = _df_from_store(pdata["df"])
        mapped = pdata["mapped"]
        irra_key = pdata["irra_key"]

        bv_normal, _bv_outlier = basic_value_filter(df, mapped)
        df = df.loc[bv_normal].copy()

        # Clear-sky and low-irradiance/power filters need irradiance; skip them
        # when it's absent (matches Advanced mode), still doing value/IQR filters.
        clearsky_mask = pd.Series(True, index=df.index)
        if irra_key:
            try:
                cs_normal_idx, _ = clear_sky_filter(
                    df, irra_key, smoothness_threshold=0.3, energy_threshold=0.5)
                clearsky_mask = df.index.isin(cs_normal_idx)
            except Exception:
                clearsky_mask = pd.Series(True, index=df.index)

        df_filtered = normalize(df, mapped, gamma=-0.004)
        current_mask = pd.Series(clearsky_mask, index=df_filtered.index)

        try:
            df_filtered.index = pd.to_datetime(df_filtered.index)
            df_filtered.index = (df_filtered.index
                                 .tz_localize("UTC").tz_convert("US/Pacific"))
        except Exception:
            pass

        if irra_key:
            normal_idx, _ = low_irra_power_filter(
                df_filtered, mapped,
                irr_thresh=300, power_ratio=0.02,
                norm_lower=0.01, norm_upper_pct=99,
            )
            current_mask &= df_filtered.index.isin(normal_idx)
        else:
            # No irradiance: drop night / low-output points by power alone.
            normal_idx, _ = low_power_filter(df_filtered, mapped)
            current_mask &= df_filtered.index.isin(normal_idx)

        # IQR on the points that survived prior filters (not the full frame), so
        # night/low values don't pull the fences down and flag daytime as outliers.
        _kept = df_filtered.loc[df_filtered.index[current_mask]]
        normal_idx, _ = identify_outliers_iqr(_kept, "norm", iqr_multiplier=1.5)
        current_mask &= df_filtered.index.isin(normal_idx)

        df_good = df_filtered.loc[df_filtered.index[current_mask]]
    except Exception as e:
        return _fail(_no_data_alert(f"Filtering step failed: {e}"))

    if df_good.empty:
        return _fail(_no_data_alert(
            "No data points survived default filtering. "
            "Try Advanced mode to loosen the filters."
        ))

    # Success: Step 2 DONE, Step 3 ACTIVE.  Pass the filtered df forward.
    payload = {
        "df_good": df_good.to_json(date_format="iso", orient="split"),
        "irra_key": irra_key,
        "source_name": pdata["source_name"],
        "n_raw": pdata["n_raw"],
        "n_kept": int(len(df_good)),
        "method": pdata.get("method", "YOY"),
        "mapped": mapped,
        "mapping_notes": pdata.get("mapping_notes", []),
    }
    method = pdata.get("method", "YOY")
    progress = {"started": True, "data": True, "filter": True,
                "calc": False, "code": False}
    step3_label = ("Fitting PVPRO (single-diode model)…" if method == "PVPRO"
                   else _SIMPLE_STEP_LABELS.get(3, "Estimating degradation…"))
    status = _working_banner(step3_label)
    return payload, status, progress

# ---- STAGE 3 : degradation — YoY (fast) OR PVPRO (background fit) -----------
@app.callback(
    Output("simple-stash",   "data", allow_duplicate=True),
    Output("simple-status",  "children", allow_duplicate=True),
    Output("simple-result",  "children", allow_duplicate=True),
    Output("step-progress",  "data", allow_duplicate=True),
    Output("simple-pvpro-job",           "data",     allow_duplicate=True),
    Output("simple-pvpro-poll-interval", "disabled", allow_duplicate=True),
    Output("simple-pvpro-progress-output", "children", allow_duplicate=True),
    Input("simple-pipe-filtered", "data"),
    State("simple-param-pvpro-cells",    "value"),
    State("simple-param-pvpro-mps",      "value"),
    State("simple-param-pvpro-ps",       "value"),
    State("simple-param-pvpro-alphaisc", "value"),
    State("simple-param-pvpro-tech",     "value"),
    State("simple-param-pvpro-days",     "value"),
    State("simple-param-pvpro-iters",    "value"),
    prevent_initial_call=True,
)
def simple_stage_calc(pfiltered, cells, mps, ps, alphaisc, tech, days, iters):
    if not pfiltered or "df_good" not in pfiltered:
        return (dash.no_update,) * 7

    method = pfiltered.get("method", "YOY")

    # -------------------------------------------------------------------
    # PVPRO branch: launch the long-running fit in a background thread and
    # let `simple_pvpro_poll` render the result into `simple-result`.
    # -------------------------------------------------------------------
    if method == "PVPRO":
        mapped = pfiltered.get("mapped") or {}
        if not _simple_has_dc_vi(mapped):
            none = {"started": True, "data": True, "filter": True,
                    "calc": False, "code": False}
            return ({}, _no_data_alert(
                "PVPRO needs DC Voltage and DC Current columns, which weren't "
                "identified in this dataset. Try Advanced mode to map them, or "
                "use the YoY method."), "", none, {}, True, "")
        try:
            df_filtered = _df_from_store(pfiltered["df_good"])
        except Exception as e:
            none = {"started": True, "data": True, "filter": True,
                    "calc": False, "code": False}
            return ({}, _no_data_alert(f"Could not load data for PVPRO: {e}"),
                    "", none, {}, True, "")

        pvpro_kwargs = dict(
            cells_in_series     = cells     if cells     else 60,
            modules_per_string  = mps       if mps       else 1,
            parallel_strings    = ps        if ps        else 1,
            alpha_isc           = alphaisc  if alphaisc is not None else 0.0046,
            technology          = tech      if tech      else "mono-c-Si",
            days_per_run        = days      if days      else 14,
            iterations_per_year = iters     if iters     else 12,
        )
        job_id = _pvpro_make_job()

        def _progress_cb(stage, current, total, message, _jid=job_id):
            _pvpro_update_job(_jid, phase=stage, current=current,
                              total=total, message=message)

        def _worker(_df=df_filtered, _mapping=mapped,
                    _kwargs=pvpro_kwargs, _jid=job_id, _cb=_progress_cb):
            try:
                rd, figs, rates = compute_pvpro(_df, _mapping,
                                                progress_callback=_cb, **_kwargs)
                _pvpro_update_job(_jid, phase="finalising", message="Packing results…")
                _pvpro_update_job(_jid, phase="done",
                                  result={"rd": float(rd), "figs": figs,
                                          "rates": rates},
                                  message="Done")
            except Exception as exc:
                _pvpro_update_job(_jid, phase="error",
                                  error=f"{type(exc).__name__}: {exc}",
                                  message=str(exc))

        threading.Thread(target=_worker, daemon=True).start()

        initial_ui = _pvpro_progress_ui(
            phase="starting", current=0, total=1,
            message="Spinning up PVPRO worker…", elapsed_s=0,
        )
        # Keep Step 3 ACTIVE while the fit runs; the poll flips it DONE.
        progress = {"started": True, "data": True, "filter": True,
                    "calc": False, "code": False}
        running_status = _working_banner("Fitting PVPRO (single-diode model)…")
        return ({}, running_status, "", progress,
                {"job_id": job_id}, False, initial_ui)

    # -------------------------------------------------------------------
    # YoY branch (default): fast synchronous estimate.
    # -------------------------------------------------------------------
    # Brief pause so the "Estimating degradation…" step is visible.
    time.sleep(2)

    def _fail(alert):
        none = {"started": True, "data": True, "filter": True,
                "calc": False, "code": False}
        return (
            {}, alert, "", none,
            dash.no_update, dash.no_update, dash.no_update,
        )

    try:
        df_good = _df_from_store(pfiltered["df_good"])
        irra_key = pfiltered["irra_key"]
        daily_data = aggregate_daily(df_good, irra_key)
        rd, fig = compute_yoy(daily_data, rolling_window=30, iqr_multiplier=1.5)
        _simple_method = "YoY"
        # Fall back to Linear Regression if YoY can't produce a rate (sparse /
        # irregular data) instead of returning NaN on an otherwise-fine dataset.
        if rd is None or not np.isfinite(rd):
            rd_lr, fig_lr = compute_lr(daily_data)
            if rd_lr is not None and np.isfinite(rd_lr):
                rd, fig, _simple_method = rd_lr, fig_lr, "LR"
    except Exception as e:
        return _fail(_no_data_alert(f"Degradation calculation failed: {e}"))

    if fig is not None:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", color=INK),
            margin=dict(l=50, r=20, t=50, b=60),
            title=dict(font=dict(family="Arial", size=18, color=INK),
                       x=0, xanchor="left"),
            height=340,
        )
        fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)

    start_date = df_good.index.min()
    end_date   = df_good.index.max()
    duration_years = (end_date - start_date).days / 365.25
    rate_pct = rd / 100

    n_raw = pfiltered["n_raw"]
    n_kept = pfiltered["n_kept"]
    n_removed = max(n_raw - n_kept, 0)
    pct_kept = (n_kept / n_raw * 100) if n_raw else 0.0
    trend_summary = _summarize_daily_series(daily_data, _metric_label("YOY"))
    source_name = pfiltered["source_name"]

    stash = {
        "rate_pct": float(rate_pct),
        "method": _simple_method,
        "duration_years": float(duration_years),
        "start": start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
        "end":   end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
        "source_name": source_name,
        "n_raw": n_raw,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "pct_kept": float(pct_kept),
        "trend_summary": trend_summary,
        "fig": fig.to_json() if fig is not None else None,
        # Carry the detected mapping + parse warnings so the result can show the
        # same data-quality notes toggle as Advanced mode.
        "mapped": pfiltered.get("mapped") or {},
        "mapping_notes": pfiltered.get("mapping_notes", []),
    }

    # Success: Step 3 DONE.  Reveal the result.
    progress = {"started": True, "data": True, "filter": True,
                "calc": True, "code": False}
    done_status = _success_banner(f"{source_name} analysis complete")
    return (stash, done_status, _simple_result_layout(stash), progress,
            dash.no_update, dash.no_update, dash.no_update)


# -----------------------------------------------------------------------------
# Rebuild the Simple-mode result layout from the stashed primitives.
# -----------------------------------------------------------------------------


# =============================================================================
# CALLBACK — APPLY USER-EDITED VARIABLE MAPPING  (ported from pvcopilotMaster)
# =============================================================================
@app.callback(
    Output("mapped-vars-store", "data",     allow_duplicate=True),
    Output("var-map-panel",     "children"),
    Output("var-map-figures",   "children"),
    Input("var-map-apply-btn",  "n_clicks"),
    State({"type": "var-map-dd", "metric": ALL}, "value"),
    State({"type": "var-map-dd", "metric": ALL}, "id"),
    State("dataframe-store",    "data"),
    State("data-columns-store", "data"),
    prevent_initial_call=True,
)
def apply_variable_mapping_callback(n_clicks, values, ids, df_json, data_columns):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # Rebuild the mapping dict, keeping ONLY non-empty selections.
    new_mapping = {}
    for val, id_obj in zip(values, ids):
        metric = id_obj.get("metric")
        if val:
            new_mapping[metric] = val

    # Flag any required variables that are still unset.
    missing = [m for m in _MAP_METRICS
               if m in _REQUIRED_FOR_DEGRADATION and not new_mapping.get(m)]

    if missing:
        status = html.Div(
            "Mapping applied. Still missing for degradation analysis: "
            + ", ".join(missing) + ".",
            className="alert alert-warning",
            style={"marginBottom": "0", "fontSize": "13px"},
        )
    else:
        applied = ", ".join(f"{m} → {new_mapping[m]}" for m in _MAP_METRICS
                            if new_mapping.get(m))
        status = html.Div(
            "✓ Mapping applied. " + applied,
            className="alert alert-success",
            style={"marginBottom": "0", "fontSize": "13px"},
        )

    # Rebuild the mapping panel so the dots match the applied state.
    time_in_index = new_mapping.get("Time") == "__index__"
    panel = build_variable_mapping_table(
        new_mapping, data_columns or [],
        time_in_index=time_in_index, status_children=status,
    )

    # Redraw figures from the new mapping — unselected variables are not drawn.
    try:
        df = _df_from_store(df_json) if df_json else None
    except Exception:
        df = None
    figures = _build_overview_figures_div(df, new_mapping)

    return new_mapping, panel, figures


# =============================================================================
# CALLBACK — DATA UPLOAD & PARSE  (UNCHANGED logic, restyled output)
# =============================================================================

