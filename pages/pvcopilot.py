import dash
from dash import dcc, html, Input, Output, dash_table, ALL
import dash_mantine_components as dmc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from app import app
from page_supporting_files.analysis_utils_2606 import parse_contents
from dash import callback_context as ctx
from io import StringIO
import traceback
from page_supporting_files.analysis_utils_2606 import make_overview_figures, normalize, low_irra_power_filter, aggregate_daily, compute_yoy, get_full_code
from page_supporting_files.analysis_utils_2606 import compute_lr, compute_hw, compute_arima, compute_csd, compute_pvpro
from page_supporting_files.analysis_utils_2606 import estimate_pvpro_params
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
# Analyze and the fast degradation methods run synchronously inside their Dash
# callbacks. A malformed dataset can make them run effectively forever and hang
# the UI. We cap them by running the pure computation in a worker and waiting on
# its result; if it overruns, the callback aborts and surfaces a clear error
# instead of leaving the user staring at a spinner.
#
# PVPRO is deliberately NOT guarded here — it has its own background-job
# infrastructure below and legitimately takes 1–3 minutes.
#
# Caveat: a timed-out worker keeps running to completion in the background
# (Python can't force-kill a thread). That's acceptable — the only requirement
# is that the *UI* stops waiting and reports the problem.
# =============================================================================
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=8)
STEP_TIMEOUT_S = 10
# Analyze makes an LLM column-identification call, which legitimately takes a
# few seconds and can retry once on a busy gateway — give it more headroom than
# the pure-pandas steps so a slow-but-fine call isn't reported as a timeout.
ANALYZE_TIMEOUT_S = 25


def _run_with_timeout(fn, *args, timeout=STEP_TIMEOUT_S, **kwargs):
    """Run fn(*args, **kwargs), raising FutureTimeout if it exceeds `timeout` s."""
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
# Solution: a single Lock per job_id.  One poll builds the final layout at a
# time; concurrent polls return all-no_update.  The backend job deliberately
# remains terminal ("done"/"error") until its TTL expires.  The browser-side
# job store and Interval are cleared only by a successfully applied terminal
# response, so a lost response can be retried idempotently on the next tick.
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


# -----------------------------------------------------------------------------
# LIVE ANALYZE STATUS  (reuses the PVPRO job cache as a tiny message bus)
#
# The Analyze callback publishes a stage message under a key derived from a
# per-tab token + n_clicks; a dcc.Interval poll (served by any worker thread)
# reads the same key and mirrors it into the UI while the callback is still
# running. Nothing depends on these records for correctness — a lost message
# just means the status line skips a beat.
# -----------------------------------------------------------------------------
_ANALYZE_STATUS_PREFIX = "analyze-status:"
_ANALYZE_STATUS_TTL_S = 600


def _analyze_status_key(token, n_clicks):
    return f"{_ANALYZE_STATUS_PREFIX}{token}:{n_clicks}"


def _analyze_status_set(key, message):
    """Publish the current Analyze stage message. started_at persists from the
    first write so the poll can show a running elapsed counter."""
    now = time.time()
    if _PVPRO_USE_DISKCACHE:
        rec = _PVPRO_JOB_CACHE.get(key) or {"started_at": now}
        rec["message"] = message
        _PVPRO_JOB_CACHE.set(key, rec, expire=_ANALYZE_STATUS_TTL_S)
    else:
        with _PVPRO_JOBS_LOCK:
            rec = _PVPRO_JOBS.get(key) or {"started_at": now}
            rec["message"] = message
            _PVPRO_JOBS[key] = rec
            # In-memory dict has no TTL — prune stale status entries here.
            stale = [k for k, v in _PVPRO_JOBS.items()
                     if isinstance(k, str) and k.startswith(_ANALYZE_STATUS_PREFIX)
                     and now - v.get("started_at", now) > _ANALYZE_STATUS_TTL_S]
            for k in stale:
                _PVPRO_JOBS.pop(k, None)


def _analyze_status_get(key):
    """Snapshot of {'message', 'started_at'} for this run, or None."""
    if _PVPRO_USE_DISKCACHE:
        rec = _PVPRO_JOB_CACHE.get(key)
    else:
        with _PVPRO_JOBS_LOCK:
            rec = _PVPRO_JOBS.get(key)
    return None if rec is None else dict(rec)


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
# DESIGN TOKENS — liquid-glass PV Copilot aesthetic
# =============================================================================
INK            = "#1c2540"
INK_SOFT       = "#5a6784"
PAPER          = "rgba(238, 244, 253, 0.54)"
PAPER_RAISED   = "rgba(255, 255, 255, 0.58)"
BORDER         = "rgba(255, 255, 255, 0.72)"
BORDER_STRONG  = "rgba(120, 140, 180, 0.30)"
NAVY           = "#2f6bff"
NAVY_DEEP      = "#1f52d6"
NAVY_SOFT      = "rgba(79, 139, 255, 0.12)"
ACCENT         = NAVY             # accent everywhere is navy
ACCENT_SOFT    = NAVY_SOFT
SIDEBAR_BG     = "linear-gradient(145deg, rgba(255,255,255,0.72), rgba(255,255,255,0.46))"
# All agent accents collapse to one navy
TEAL           = NAVY
INDIGO         = NAVY
ROSE           = NAVY
SLATE          = NAVY
SUCCESS        = "#1aa06e"
MUTED          = "#8090ad"

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
    # Code is no longer a numbered pipeline step -- it's an optional add-on
    # (glyph is a code symbol, step=None so agent_message shows an "Add-on"
    # badge instead of "Step N of 3").
    "code":   {"name": "Code Agent",              "color": NAVY, "glyph": "⟨⟩", "step": None},
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
            "borderRadius": "16px",
            "fontFamily": "Archivo, system-ui, sans-serif",
        }
    )


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
            "borderRadius": "16px",
            "fontSize": "15px",
            "fontFamily": "Archivo, system-ui, sans-serif",
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
                      style={"fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "14px",
                             "color": INK, "fontWeight": "600"}),
        ],
        style={"padding": "12px 16px", "background": "#eff6ff",
               "border": "1px solid #bfdbfe", "borderRadius": "16px",
               "fontSize": "14px", "marginBottom": "8px"},
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

    # Elapsed is rendered in PLAIN SECONDS ("Elapsed: Ns") so it matches what
    # the 1 Hz clientside ticker (assets/pvpro_elapsed_ticker.js) writes; if the
    # server used the m/s form here, the 2 s re-render and the 1 s tick would
    # flip-flop between "2m 15s" and "135s". The ticker finds this element by id
    # and reads data-started-at (ms), so we expose both below.
    elapsed_int = max(0, int(round(elapsed_s)))
    started_at_ms = int((time.time() - elapsed_s) * 1000)
    # The per-window / ETA text is kept SEPARATE from the elapsed text because
    # the ticker overwrites the elapsed element's whole textContent every second
    # — anything sharing that element would be wiped.
    extra_text = ""
    if phase == "fitting" and current and current >= 1 and total and total > current:
        per_window = elapsed_s / max(current, 1)
        eta_secs   = per_window * (total - current)
        extra_text = (
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
                    "color": INK, "fontFamily": "Archivo, system-ui, sans-serif",
                }
            ),
            html.Span(
                f"  ·  {pct}%",
                style={
                    "marginLeft": "8px", "fontSize": "13px",
                    "color": ACCENT, "fontWeight": "600",
                    "fontFamily": "Archivo, system-ui, sans-serif",
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
                "fontFamily": "Archivo, system-ui, sans-serif",
            }
        ),
        # Bottom stats: the "Elapsed: Ns" span is updated every second by the
        # clientside ticker, which locates it by id and reads data-started-at
        # (wall-clock ms). The per-window/ETA text sits in a SEPARATE span so
        # the ticker's textContent write can't clobber it.
        html.Div(
            [
                html.Span(
                    f"Elapsed: {elapsed_int}s",
                    id="pvpro-elapsed-display",
                    **{"data-started-at": str(started_at_ms)},
                ),
                html.Span(extra_text),
            ],
            style={
                "fontSize": "11px", "color": MUTED, "marginTop": "4px",
                "fontFamily": "Archivo, system-ui, sans-serif",
            }
        ),
    ], style={
        "padding": "16px 18px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
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
                              "fontFamily": "Archivo, system-ui, sans-serif"}),
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
                    "fontFamily": "Archivo, system-ui, sans-serif",
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
            "fontFamily": "Archivo, system-ui, sans-serif",
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
                                "fontFamily": "Archivo, system-ui, sans-serif",
                            }),
                            html.Span(
                                (f"Step {a['step']} of 3" if a.get("step")
                                 else "Add-on"),
                                style={
                                    "marginLeft": "10px",
                                    "fontSize": "13px",
                                    "color": INK_SOFT,
                                    "padding": "2px 8px",
                                    "background": "#e2e8f0",
                                    "borderRadius": "16px",
                                    "fontFamily": "Archivo, system-ui, sans-serif",
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
                            "fontFamily": "Archivo, system-ui, sans-serif",
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
                    # Left-aligned with the header row (no avatar indent) so the
                    # box's left edge lines up with the "①" badge / agent title.
                    "padding": "24px 40px",
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


def locked_placeholder(agent_key, name, step_num, addon=False):
    """A muted preview card shown until the previous step completes.

    addon=True renders the 'optional add-on' variant (code glyph + 'Add-on'
    badge + a message pointing at the prerequisite) instead of a numbered step.
    """
    a = AGENTS[agent_key]
    badge_text = "Add-on" if addon else f"Step {step_num} of 3"
    glyph_text = "⟨⟩" if addon else str(step_num)
    unlock_msg = ("Complete the Degradation step to unlock this."
                  if addon else "Complete the previous step to unlock this agent.")
    return html.Div(
        [
            # Compact header — small bullet + agent name
            html.Div(
                [
                    html.Div(
                        # Show the step number (or code glyph for add-ons).
                        glyph_text,
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
                            "fontFamily": "Archivo, system-ui, sans-serif",
                            "flexShrink": "0",
                        }
                    ),
                    html.Div(
                        [
                            html.Span(name, style={
                                "fontWeight": "600",
                                "color": MUTED,
                                "fontSize": "15px",
                                "fontFamily": "Archivo, system-ui, sans-serif",
                            }),
                            html.Span(
                                badge_text,
                                style={
                                    "marginLeft": "10px",
                                    "fontSize": "12px",
                                    "color": MUTED,
                                    "padding": "2px 8px",
                                    "background": "#e2e8f0",
                                    "borderRadius": "16px",
                                    "fontFamily": "Archivo, system-ui, sans-serif",
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
                unlock_msg,
                style={
                    "marginLeft": "38px",
                    "marginTop": "8px",
                    "fontSize": "14px",
                    "color": MUTED,
                    "fontStyle": "italic",
                    "fontFamily": "Archivo, system-ui, sans-serif",
                }
            ),
        ],
        style={
            "padding": "16px 18px",
            "marginBottom": "16px",
            "background": "rgba(241, 245, 249, 0.5)",
            "border": f"1px dashed {BORDER_STRONG}",
            "borderRadius": "16px",
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
            "fontFamily": "Archivo, system-ui, sans-serif",
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
            "borderRadius": "16px",
            "padding": "10px 14px",
            "lineHeight": "1.55",
            "fontFamily": "Archivo, system-ui, sans-serif",
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
        "borderRadius": "16px",
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
            "fontFamily": "Archivo, system-ui, sans-serif",
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
            "fontFamily": "Archivo, system-ui, sans-serif",
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
        "fontFamily": "Archivo, system-ui, sans-serif",
        "whiteSpace": "nowrap",
        "transition": "all 0.15s ease",
    }


def _chat_bubble(role, text, fresh=False):
    """Render one chat message bubble; assistant Markdown appears at once."""
    is_user = role == "user"

    bubble_style = {
        "padding": "12px 16px",
        "background": "#e2e8f0" if is_user else "white",   # slate-200 user bubble
        "color": INK,
        "border": f"1px solid #cbd5e1" if is_user else f"1px solid {BORDER}",
        "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)" if is_user else "0 1px 2px rgba(15, 23, 42, 0.03)",
        "borderRadius": "14px",
        "borderBottomRightRadius": "4px" if is_user else "14px",
        "borderBottomLeftRadius": "14px" if is_user else "4px",
        "maxWidth": "88%",
        "fontSize": "14px",
        "fontWeight": "600" if is_user else "400",
        "lineHeight": "1.6",
        "fontFamily": "Archivo, system-ui, sans-serif",
        "whiteSpace": "pre-wrap" if is_user else "normal",
    }

    if is_user:
        inner = html.Div(text, style=bubble_style)
    else:
        inner = dcc.Markdown(
            text,
            className="pvc-chat-answer-markdown",
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
    row_bg = "rgba(79,139,255,0.13)" if is_active else "rgba(255,255,255,0.26)"
    row_border = "1px solid rgba(79,139,255,0.24)" if is_active else "1px solid rgba(255,255,255,0.36)"

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
                    "fontFamily": "Archivo, system-ui, sans-serif",
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
                        "fontFamily": "Archivo, system-ui, sans-serif",
                        "whiteSpace": "nowrap",
                    }),
                    html.Div(sub, style={
                        "fontSize": "13px",
                        "color": MUTED if state == "pending" else INK_SOFT,
                        "marginTop": "1px",
                        "fontFamily": "Archivo, system-ui, sans-serif",
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
                    "fontFamily": "Archivo, system-ui, sans-serif",
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
            "borderRadius": "14px",
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
            # Larger black star icon (no circle background).
            html.Div(
                "✦",
                style={
                    "color": INK,
                    "fontSize": "22px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "flexShrink": "0",
                    "width": "28px",
                }
            ),
            html.Div(
                [
                    html.Div("Ask the AI Assistance", style={
                        "fontSize": "15px",
                        "fontWeight": "700",
                        "color": INK,
                        "fontFamily": "Archivo, system-ui, sans-serif",
                        "whiteSpace": "nowrap",
                    }),
                    html.Div("Chat about your results", style={
                        "fontSize": "13px",
                        "color": INK_SOFT,
                        "marginTop": "1px",
                        "fontFamily": "Archivo, system-ui, sans-serif",
                        "whiteSpace": "nowrap",
                    }),
                ],
                style={"marginLeft": "12px", "flex": "1", "minWidth": "0"}
            ),
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
        className="ai-chat-row",
        n_clicks=0,
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "12px 14px",
            "borderRadius": "16px",
            # Soft pink -> blue gradient, no border.
            "background": "linear-gradient(135deg, rgba(255,238,197,0.74), rgba(219,232,255,0.78))",
            "border": "1px solid rgba(255,255,255,0.55)",
            "marginTop": "0",
            "marginBottom": "4px",
            "transition": "all 0.25s ease",
            "cursor": "pointer",
            "userSelect": "none",
        }
    )


def _build_legacy_sidebar(progress=None):
    progress = progress or {"data": False, "filter": False, "calc": False, "code": False}

    # Step 1 only becomes "active" once the user has actually started an
    # analysis (clicked Analyze in Simple mode, or Run prescreening in
    # Advanced mode).  Before that it reads as "pending".
    started  = progress.get("started", False)
    s_data   = _step_state(progress, "data",   prior_done=started)
    s_filter = _step_state(progress, "filter", prior_done=progress.get("data", False))
    s_calc   = _step_state(progress, "calc",   prior_done=progress.get("filter", False))

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
                ],
                style={"padding": "20px 24px 24px"}
            ),

            # Workflow section.  Horizontal padding matches the brand
            # block above (18px) so the "WORKFLOW" label aligns flush
            # with the left edge of the "PV Copilot" logo.  (The previous
            # 12px left padding offset it 6px to the left and read as
            # misaligned.)
            html.Div(
                [
                    section_label("Workflow"),
                    stepper_item(1, "Data Prescreening", "Upload & inspect", TEAL,   state=s_data,   step_key="data"),
                    stepper_item(2, "Filter",             "Clean the signal", INDIGO, state=s_filter, step_key="filter"),
                    stepper_item(3, "Degradation",        "Compute the rate", ROSE,   state=s_calc,   step_key="calc"),
                    # Divider between the linear 3-step workflow and the
                    # always-on "bonus" entries (Code export + Chat helper).
                    # A 1px gray line in 14px of vertical breathing room reads
                    # as a hard category break rather than another step.
                    html.Div(style={
                        "borderTop": f"1px solid {BORDER}",
                        "margin": "14px 4px",
                    }),
                    # Chat is a "bonus" entry, NOT a numbered step -- not gated
                    # on prior progress, never marked done.  Clicking scrolls
                    # the right panel to the AI helper section.
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
                                "borderRadius": "12px",
                                "fontSize": "13px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                                "fontFamily": "Archivo, system-ui, sans-serif",
                            }
                        ),
                        style={
                            "display": "block" if any(progress.values()) else "none",
                            "marginBottom": "24px",
                        }
                    ),
                ],
                style={"padding": "0 24px"}
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
                            "fontFamily": "Archivo, system-ui, sans-serif",
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
                            "fontFamily": "Archivo, system-ui, sans-serif",
                            "fontWeight": "600",
                            "textDecoration": "none",
                            "display": "inline-flex",
                            "alignItems": "center",
                        }
                    ),
                ],
                style={"padding": "0 24px 20px"}
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
                                    "fontFamily": "Archivo, system-ui, sans-serif",
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
                                                "fontFamily": "Archivo, system-ui, sans-serif",
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
                                                "fontFamily": "Archivo, system-ui, sans-serif",
                                                "verticalAlign": "middle",
                                            }),
                                        ],
                                        style={"display": "flex", "alignItems": "center"}
                                    ),
                                    html.Div("Save and reload past sessions", style={
                                        "fontSize": "13px",
                                        "color": INK_SOFT,
                                        "fontFamily": "Archivo, system-ui, sans-serif",
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
                    "padding": "14px 24px",
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
            "borderRadius": "26px",
            "display": "flex",
            "flexDirection": "column",
            "height": "calc(100vh - 110px)",
            "overflowY": "auto",
            "boxShadow": "0 14px 44px rgba(30,58,120,0.11), inset 0 1px 0 rgba(255,255,255,0.62)",
            "backdropFilter": "blur(30px) saturate(1.5)",
            "WebkitBackdropFilter": "blur(30px) saturate(1.5)",
            "fontFamily": "Archivo, system-ui, sans-serif",
            # Stay pinned while the right column scrolls past it.
            "position": "sticky",
            "top": "8px",
        }
    )


# New-layout workflow rail.  Keep the same function name and callback-facing
# component IDs, but remove the old app-wide brand/login sidebar: in the demo
# layout the rail belongs inside the analysis card.
def build_sidebar(progress=None):
    progress = progress or {"data": False, "filter": False, "calc": False, "code": False}
    started = progress.get("started", False)
    s_data = _step_state(progress, "data", prior_done=started)
    s_filter = _step_state(progress, "filter", prior_done=progress.get("data", False))
    s_calc = _step_state(progress, "calc", prior_done=progress.get("filter", False))

    return html.Div(
        [
            html.Div("ADVANCED WORKFLOW", style={
                "fontSize": "12px", "fontWeight": "800", "letterSpacing": "0.12em",
                "color": MUTED, "marginBottom": "14px",
            }),
            stepper_item(1, "Data prescreening", "Raw signals & quality", TEAL,
                         state=s_data, step_key="data"),
            stepper_item(2, "Intelligent filtering", "Configure & review", INDIGO,
                         state=s_filter, step_key="filter"),
            stepper_item(3, "Degradation model", "Metric & calculation", ROSE,
                         state=s_calc, step_key="calc"),
            html.Div(style={"borderTop": f"1px solid {BORDER_STRONG}", "margin": "14px 4px"}),
            _chat_sidebar_item(),
            html.Div(
                html.Button(
                    "↻  Restart workflow", id="restart-btn", n_clicks=0,
                    style={
                        "width": "100%", "padding": "10px 14px", "marginTop": "16px",
                        "background": "rgba(255,255,255,0.48)", "color": INK_SOFT,
                        "border": f"1px solid {BORDER_STRONG}", "borderRadius": "12px",
                        "fontSize": "13px", "fontWeight": "700", "cursor": "pointer",
                    },
                ),
                style={"display": "block" if any(progress.values()) else "none"},
            ),
        ],
        className="glass-soft",
        style={
            "width": "286px", "flexShrink": "0", "padding": "20px",
            "background": "rgba(255,255,255,0.38)",
            "border": "1px solid rgba(255,255,255,0.68)", "borderRadius": "22px",
            "position": "sticky", "top": "18px",
        },
    )


# =============================================================================
# CHAT — AGENT 1 · DATA
# =============================================================================
data_agent_body = html.Div([
    # Analyze button — runs detection on the shared, already-loaded data.
    html.Button(
        "Run prescreening",
        id="analyze-btn",
        n_clicks=0,
        className="pvc-step1-intro-item pvc-primary-action",
        style={
            "width": "100%",
            "padding": "12px 16px",
            "marginTop": "0",
            "background": "linear-gradient(135deg, #4b8bff, #2f6bff)",
            "color": PAPER,
            "border": "none",
            "borderRadius": "16px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Archivo, system-ui, sans-serif",
            "letterSpacing": "0.01em",
        }
    ),
    html.Div(
        "",
        id="analyze-caption",
        className="pvc-step1-intro-item",
        style={"display": "none"},
    ),
    # Live status line — hidden until Analyze is clicked, then driven by the
    # analyze-status-interval poll while the callback runs.
    html.Div(
        [
            html.Span("Starting analysis…", id="analyze-status-text",
                      style={"color": INK, "fontWeight": "500"}),
            html.Span("", id="analyze-status-elapsed",
                      style={"color": INK_SOFT, "marginLeft": "2px"}),
        ],
        id="analyze-status-line",
        className="pvc-step1-intro-item",
        style={"display": "none"},
    ),
    dcc.Store(id="analyze-status-token"),
    dcc.Interval(id="analyze-status-interval", interval=450,
                 n_intervals=0, disabled=True),

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
    html.Button(
        ["Next step ", html.Span("→")],
        id="advanced-next-step-1",
        n_clicks=0,
        className="pvc-next-step-button",
    ),
], id="advanced-data-body", className="pvc-advanced-step-body",
   style={"fontFamily": "Archivo, system-ui, sans-serif"})


# =============================================================================
# SHARED DATA-UPLOAD HEADER  (common to both modes, sits above the mode tabs)
#   Upload box + 3 example buttons + data requirements.
#   Keeps the original component IDs (upload-data, load-example-btn-*,
#   upload-status-output) so the existing load/parse callbacks keep working.
# =============================================================================
def _shared_example_btn(btn_id, label, description):
    return html.Button(
        [
            html.Span("▤", className="pvc-example-icon"),
            html.Span(
                [
                    html.Span(label, className="pvc-example-title"),
                    html.Span(description, className="pvc-example-description"),
                ],
                className="pvc-example-copy",
            ),
        ],
        id=btn_id,
        n_clicks=0,
        className="pvc-example-card",
    )


shared_upload_header = html.Div(
    [
        html.Div(
            [
                # Upload box
                html.Div(
                    [
                        html.Button(
                            "Data requirements",
                            id={"type": "pvc-main-nav", "index": "datareq"},
                            n_clicks=0,
                            className="pvc-datareq-button",
                        ),
                        dcc.Upload(
                            id="upload-data",
                            className="pvc-upload",
                            accept=".csv, text/csv, .xls, .xlsx, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, .parquet",
                            children=html.Div(
                                [
                                    html.Div("⬆", className="pvc-up-icon", style={
                                        "width": "56px", "height": "56px", "borderRadius": "18px",
                                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                                        "margin": "0 auto 14px", "fontSize": "24px", "color": NAVY,
                                        "background": "rgba(79,139,255,0.12)",
                                    }),
                                    html.Div(["Drag & drop your data, or ",
                                              html.Span("click to browse", style={"color": ACCENT})],
                                             style={"fontSize": "17px", "fontWeight": "700", "marginBottom": "5px"}),
                                    html.Div("CSV · Excel · Parquet · timestamp + power / energy / irradiance columns",
                                             style={"fontSize": "13px", "color": INK_SOFT}),
                                ],
                                className="upload-inner",
                                style={"textAlign": "center", "color": INK,
                                       "fontFamily": "Archivo, system-ui, sans-serif"}
                            ),
                            style={
                                "width": "100%", "padding": "28px 24px",
                                "border": f"1.6px dashed {BORDER_STRONG}", "borderRadius": "22px",
                                "backgroundColor": "rgba(255,255,255,0.34)", "cursor": "pointer",
                                "transition": "all 0.15s ease",
                            }
                        ),
                    ],
                    className="pvc-upload-column",
                ),

                # Example datasets sit beside the upload box on wide screens.
                html.Div(
                    [
                        html.Div("OR START WITH EXAMPLE DATA", className="pvc-example-heading"),
                        html.Div(
                            [
                                _shared_example_btn(
                                    "load-example-btn-1", "System 1278",
                                    "c-Si · DC V + I · PVPRO-ready",
                                ),
                                _shared_example_btn(
                                    "load-example-btn-2", "System 1403",
                                    "c-Si · DC V + I · PVPRO-ready",
                                ),
                                _shared_example_btn(
                                    "load-example-btn-3", "System 1422",
                                    "c-Si · power + irradiance",
                                ),
                            ],
                            id="example-row",
                            className="pvc-example-list",
                        ),
                    ],
                    className="pvc-example-column",
                ),
            ],
            className="pvc-upload-grid",
        ),
        # Empty until a file/example is loaded; success and error messages span
        # the full upload + examples block instead of changing one column's height.
        html.Div(id="upload-status-output", className="pvc-upload-status"),
    ],
    className="pvc-upload-section",
    style={
        "padding": "0 44px 40px",
    }
)


# =============================================================================
# CHAT — AGENT 2 · FILTER
# =============================================================================
def filter_row(checkbox_id, label, description, customize_body=None):
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
                    "fontFamily": "Archivo, system-ui, sans-serif",
                    "fontWeight": "700",
                }),
            ],
            style={"display": "flex", "alignItems": "center"}
        )
    ]
    parts.append(
        html.P(description, className="pvc-advanced-filter-description")
    )
    if customize_body is not None:
        parts.append(html.Details([
            html.Summary("Customize parameters", style={
                "cursor": "pointer",
                "color": INK_SOFT,
                "fontSize": "13px",
                "fontWeight": "500",
                "marginTop": "6px",
                "marginLeft": "26px",
                "fontFamily": "Archivo, system-ui, sans-serif",
            }),
            html.Div(customize_body, style={
                "marginTop": "8px",
                "marginLeft": "26px",
                "padding": "12px 14px",
                "background": "#f1f5f9",
                "border": f"1px solid {BORDER}",
                "borderRadius": "12px",
                "fontSize": "14px",
            })
        ]))
    return html.Div(parts, className="pvc-advanced-filter-content")


_param_input_style = {
    "width": "100%",
    "fontSize": "14px",
    "padding": "6px 8px",
    "borderRadius": "6px",
    "border": f"1px solid {BORDER_STRONG}",
    "color": INK,
    "fontFamily": "Archivo, system-ui, sans-serif",
    "background": "white",
}

_label_style = {"fontSize": "13px", "fontWeight": "600", "color": INK, "marginBottom": "3px", "fontFamily": "Archivo, system-ui, sans-serif"}
_help_style  = {"fontSize": "13px", "color": INK_SOFT, "marginBottom": "5px", "lineHeight": "1.4", "fontFamily": "Archivo, system-ui, sans-serif"}

# --- PVPRO numeric stepper -------------------------------------------------
# Each PVPRO number field is drawn with our own - / + buttons flanking the
# input, because this Dash version's native number spinners blank the value
# on click. The native spinner is hidden via the .pvpro-num CSS class (see
# assets/pvcopilot_styles.css); the _pvpro_step callback below handles clicks.
_step_btn_style = {
    "border": f"1px solid {BORDER_STRONG}", "background": "#fff", "cursor": "pointer",
    "fontSize": "18px", "fontWeight": "700", "color": INK, "lineHeight": "1",
    "padding": "0 12px", "minWidth": "34px", "flex": "0 0 auto",
    "fontFamily": "Archivo, system-ui, sans-serif", "boxSizing": "border-box",
}
# Per-field (step, min, decimals), keyed by the id suffix.
_PVPRO_STEP_CFG = {
    "cells":    (1, 1, 0),
    "mps":      (1, 1, 0),
    "ps":       (1, 1, 0),
    "alphaisc": (0.0001, 0, 4),
    "days":     (1, 2, 0),
    "iters":    (1, 2, 0),
}
# Middle-input style; module-level so callbacks can restore it or swap in the
# "auto-filled from your data" highlight variant.
# The appearance:* keys suppress the browser's native number-input spinner
# INLINE — this matters because on some engines (notably Firefox/GTK) that
# spinner renders as − / + buttons flanking the value, which then sits next to
# our own − / + and looks doubled. Doing it inline (not only via the .pvpro-num
# CSS in assets/) makes it immune to a stale/cached stylesheet.
_PVPRO_MID_BASE = {**_param_input_style, "borderRadius": "0", "textAlign": "center",
                   "minWidth": "0", "flex": "1 1 auto", "boxSizing": "border-box",
                   "MozAppearance": "textfield", "WebkitAppearance": "textfield",
                   "appearance": "textfield"}
_PVPRO_MID_AUTOFILL = {**_PVPRO_MID_BASE, "background": "#eff6ff",
                       "border": "1px solid #60a5fa", "fontWeight": "600"}
# Small blue dot shown on a field's LABEL when that field was auto-filled from
# the data — same 7px #3b82f6 dot as the "Estimated from your data" note, so
# the marked fields and the note read as the same thing.
_PVPRO_DOT_ON = {"display": "inline-block", "width": "7px", "height": "7px",
                 "borderRadius": "50%", "background": "#3b82f6",
                 "marginRight": "6px", "verticalAlign": "middle"}
_PVPRO_DOT_OFF = {"display": "none"}


def _beta_badge():
    """Small shared marker for features that are still in beta."""
    return html.Span("BETA", className="pvc-beta-badge", title="Beta feature")


def _pvnum(x, default, cast=float):
    """Coerce a stepper field's value to a number. The stepper inputs are
    type='text' (so no browser draws a native spinner beside our own - / +
    buttons), which means their value arrives as a string — this turns it back
    into a number, falling back to `default` when blank or non-numeric."""
    try:
        return cast(float(x))
    except (TypeError, ValueError):
        return default


def _pvpro_num_field(label, id_suffix, value, prefix="", prefillable=False):
    """A labelled numeric field with its own - / + buttons flanking the input.
    `prefix` is "" for Advanced ids (param-pvpro-*) or "simple-" for Simple.
    `prefillable=True` adds a hidden blue dot to the label that the autofill
    callback reveals when this field is estimated from the data.

    The input is type='text' ON PURPOSE: a number input makes the browser draw
    its OWN spinner (on some engines as - / + buttons) right next to ours, which
    looks doubled and can't be reliably killed from a (cache-prone) stylesheet.
    A text input never has a spinner, so only our buttons show. Values are
    coerced back to numbers via _pvnum wherever they're read. (No inputMode prop
    — this dcc.Input version rejects it.)"""
    input_id = f"{prefix}param-pvpro-{id_suffix}"
    step, minv, _decimals = _PVPRO_STEP_CFG[id_suffix]
    left = {**_step_btn_style, "borderRadius": "6px 0 0 6px", "borderRight": "none"}
    right = {**_step_btn_style, "borderRadius": "0 6px 6px 0", "borderLeft": "none"}
    label_children = [label]
    if prefillable:
        label_children = [
            html.Span(id=f"{input_id}-dot", style=dict(_PVPRO_DOT_OFF)),
            label,
        ]
    return html.Div([
        html.Div(label_children, style=_label_style),
        html.Div([
            html.Button("\u2212",  # minus sign
                        id={"type": "pvpro-step", "target": input_id, "dir": "down"},
                        n_clicks=0, style=left),
            dcc.Input(id=input_id, type="text", value=value,
                      className="pvpro-num", style=dict(_PVPRO_MID_BASE)),
            html.Button("+",
                        id={"type": "pvpro-step", "target": input_id, "dir": "up"},
                        n_clicks=0, style=right),
        ], style={"display": "flex", "alignItems": "stretch"}),
    ])


# Every stepper input the _pvpro_step callback manages (both modes), in a fixed
# order that the callback's Output/State lists mirror.
_PVPRO_STEP_TARGETS = [
    "param-pvpro-cells", "param-pvpro-mps", "param-pvpro-ps",
    "param-pvpro-alphaisc", "param-pvpro-days", "param-pvpro-iters",
    "simple-param-pvpro-cells", "simple-param-pvpro-mps", "simple-param-pvpro-ps",
    "simple-param-pvpro-alphaisc", "simple-param-pvpro-days", "simple-param-pvpro-iters",
]



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

    html.Button(
        [
            html.Span("✓", className="pvc-step-fold-check"),
            html.Span("Filters applied", className="pvc-step-fold-title"),
            html.Span("Click to show or hide filter settings", className="pvc-step-fold-help"),
            html.Span("⌄", className="pvc-step-fold-arrow"),
        ],
        id="toggle-filter-settings",
        n_clicks=0,
        className="pvc-step-fold-summary",
    ),

    html.Div(section_label("Recommended filters"), className="pvc-step-config-item"),
    html.Div(
        [
            html.Div(
                filter_row(
                    "cb-timezone", "Time zone & DST correction",
                    "Aligns timestamps to local solar time and corrects daylight-saving jumps.",
                ),
                className="pvc-advanced-filter-card",
            ),
            html.Div(
                filter_row(
                    "cb-clearsky", "Clear-sky filter",
                    "Keeps smooth, cloud-free irradiance profiles for comparable operating conditions.",
                    clearsky_params,
                ),
                className="pvc-advanced-filter-card",
            ),
            html.Div(
                filter_row(
                    "cb-low-irra-power", "Low irradiance / power filter",
                    "Drops low-light points where the power-to-irradiance relationship is noisy.",
                    low_irra_params,
                ),
                className="pvc-advanced-filter-card",
            ),
            html.Div(
                filter_row(
                    "cb-outlier", "Outlier removal (IQR)",
                    "Removes statistical outliers in the normalized series using an IQR rule.",
                    outlier_params,
                ),
                className="pvc-advanced-filter-card",
            ),
        ],
        className="pvc-advanced-filter-grid pvc-step-config-item",
        style={
            "padding": "16px 18px",
            "background": "#f8fafc",
            "border": f"1px solid {BORDER}",
            "borderRadius": "16px",
            "marginBottom": "14px",
        }
    ),

    dcc.Store(id="_cb-sync-dummy"),

    html.Button(
        ["Apply filters ", html.Span("→")],
        id="filter-btn",
        n_clicks=0,
        className="pvc-step-config-item pvc-primary-action",
        style={
            "width": "100%",
            "padding": "12px 16px",
            "background": "linear-gradient(135deg, #4b8bff, #2f6bff)",
            "color": "white",
            "border": "none",
            "borderRadius": "16px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Archivo, system-ui, sans-serif",
        }
    ),

    # Collapsible filter explanations (descriptions, equations, references)
    html.Div(filter_explanations_block(), className="pvc-step-config-item"),

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
    html.Button(
        ["Next step ", html.Span("→")],
        id="advanced-next-step-2",
        n_clicks=0,
        className="pvc-next-step-button",
    ),
], id="advanced-filter-body", className="pvc-advanced-step-body",
   style={"fontFamily": "Archivo, system-ui, sans-serif"})


# =============================================================================
# CHAT — AGENT 3 · DEGRADATION
# =============================================================================
metric_options = [
    {
        "label": html.Div([
            html.B("YoY", style={"fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "16px"}),
            html.Span(" — Year-over-Year", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Details([
                html.Summary("Customize parameters", style={"cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "marginTop": "4px"}),
                html.Div([
                    html.Div([
                        html.Div("Rolling trend window (days)", style=_label_style),
                        dcc.Input(id="param-yoy-window", type="number", value=30, step=5, min=7, style=_param_input_style),
                    ], style={"flex": "1 1 0", "minWidth": "0"}),
                    html.Div([
                        html.Div("IQR multiplier k", style=_label_style),
                        dcc.Input(id="param-yoy-iqr", type="number", value=1.5, step=0.1, min=0.5, style=_param_input_style),
                    ], style={"flex": "1 1 0", "minWidth": "0"}),
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "12px", "border": f"1px solid {BORDER}", "display": "flex", "gap": "12px"}),
            ]),
        ]),
        "value": "YOY",
    },
    {
        "label": html.Div([
            html.B("LR", style={"fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "16px"}),
            html.Span(" — Linear regression", style={"color": INK_SOFT, "fontSize": "14px"}),
            dcc.Input(id="param-yoy-iqr-dummy", style={"display": "none"}),
        ]),
        "value": "LR",
    },
    {
        "label": html.Div([
            html.B("HW", style={"fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "16px"}),
            html.Span(" — Holt-Winters", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Details([
                html.Summary("Customize parameters", style={"cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "marginTop": "4px"}),
                html.Div([
                    html.Div("Seasonal period (months)", style=_label_style),
                    dcc.Input(id="param-hw-period", type="number", value=12, step=1, min=2, style=_param_input_style),
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "12px", "border": f"1px solid {BORDER}"}),
            ]),
        ]),
        "value": "HW",
    },
    {
        "label": html.Div([
            html.B("ARIMA", style={"fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "16px"}),
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
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "12px", "border": f"1px solid {BORDER}"}),
            ]),
        ]),
        "value": "ARIMA",
    },
    {
        "label": html.Div([
            html.B("CSD", style={"fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "16px"}),
            html.Span(" — Classical Seasonal Decomposition", style={"color": INK_SOFT, "fontSize": "14px"}),
            html.Details([
                html.Summary("Customize parameters", style={"cursor": "pointer", "color": INK_SOFT, "fontSize": "13px", "marginTop": "4px"}),
                html.Div([
                    html.Div("Seasonal period (months)", style=_label_style),
                    dcc.Input(id="param-csd-period", type="number", value=12, step=1, min=2, style=_param_input_style),
                ], style={"marginTop": "6px", "padding": "10px", "background": "#f1f5f9", "borderRadius": "12px", "border": f"1px solid {BORDER}"}),
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
                    html.B("PVPRO", style={"fontFamily": "Archivo, system-ui, sans-serif",
                                           "fontSize": "16px"}),
                    _beta_badge(),
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
            html.Div([
                # "Estimate from data" — sibling of <details> (not inside
                # <summary>) so a click never toggles the panel. Advanced mode
                # already identified the columns in Step 1, so this reads the
                # existing mapping (no re-parse) and estimates mps/ps.
                html.Button(
                    [html.Span("\u2726", style={"marginRight": "6px"}),
                     "Estimate from data"],
                    id="adv-pvpro-estimate-btn", n_clicks=0,
                    title=("Estimate Modules per string and Parallel strings "
                           "from the DC voltage / current identified in Step 1."),
                    style={
                        "position": "absolute", "top": "2px", "right": "0",
                        "zIndex": "2", "fontSize": "12px", "fontWeight": "700",
                        "fontFamily": "Archivo, system-ui, sans-serif", "color": NAVY,
                        "background": "#fff", "border": f"1px solid {NAVY}",
                        "borderRadius": "999px", "padding": "5px 14px",
                        "cursor": "pointer", "whiteSpace": "nowrap",
                    },
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
                           "fontFamily": "Archivo, system-ui, sans-serif",
                           "marginTop": "4px",
                           "paddingRight": "170px"},
                ),
                html.Div([
                    html.Div(style={"display": "flex", "gap": "8px",
                                    "flexWrap": "wrap",
                                    "marginBottom": "8px"}, children=[
                        html.Div(
                            _pvpro_num_field("Cells in series (per module)",
                                             "cells", 60),
                            style={"flex": "1", "minWidth": "140px"}),
                        html.Div(
                            _pvpro_num_field("Modules per string", "mps", 1,
                                             prefillable=True),
                            style={"flex": "1", "minWidth": "140px"}),
                        html.Div(
                            _pvpro_num_field("Parallel strings", "ps", 1,
                                             prefillable=True),
                            style={"flex": "1", "minWidth": "140px"}),
                    ]),
                    html.Div(style={"display": "flex", "gap": "8px",
                                    "flexWrap": "wrap",
                                    "marginBottom": "8px"}, children=[
                        html.Div(
                            _pvpro_num_field("alpha_isc (A/\u00b0C)",
                                             "alphaisc", 0.0046),
                            style={"flex": "1", "minWidth": "140px"}),
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
                        ], style={"flex": "1", "minWidth": "140px"}),
                    ]),
                    html.Div(style={"display": "flex", "gap": "8px",
                                    "flexWrap": "wrap"}, children=[
                        html.Div(
                            _pvpro_num_field("Days per run", "days", 14),
                            style={"flex": "1", "minWidth": "140px"}),
                        html.Div(
                            _pvpro_num_field("Iterations per year", "iters", 12),
                            style={"flex": "1", "minWidth": "140px"}),
                    ]),
                    # Filled only when the user clicks "Estimate from data"
                    # (estimate_pvpro_advanced): says exactly what was estimated
                    # from the data and from what. Empty otherwise — Advanced
                    # never auto-estimates.
                    html.Div(id="pvpro-autofill-note", style={"marginTop": "8px"}),
                ], style={"marginTop": "18px", "padding": "12px 14px",
                          "background": "#f1f5f9", "borderRadius": "12px",
                          "border": f"1px solid {BORDER}",
                          "width": "100%",
                          "boxSizing": "border-box"}),
                ],
                id="pvpro-params-details",
                # open state is driven by the metric-selected callback below:
                # auto-open when PVPRO is the active metric, collapse otherwise
                # and on any new-data event.  Start closed (the default radio
                # value is "YOY", not PVPRO).
                open=False,
                ),  # closes html.Details
            ], style={"position": "relative"}),
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


# Minimum dataset span (years) before YoY is offered. YoY compares each day to
# the same day one year earlier, so it strictly needs data spanning more than a
# year to yield any comparison. Set to 1.0 per request; raise toward 2.0 for
# more robust YoY (a ~1-year span yields very few comparison points).
_MIN_YEARS_FOR_YOY = 1.0


def _duration_years(df):
    """Time span of a datetime-indexed frame, in years. Returns None if it
    can't be determined (non-datetime index, empty frame, etc.)."""
    try:
        idx = pd.to_datetime(df.index)
        return (idx.max() - idx.min()).days / 365.25
    except Exception:
        return None


def build_stat_metric_options(disable_yoy=False):
    """Return the statistical-method radio options, optionally greying out YoY.

    For datasets shorter than _MIN_YEARS_FOR_YOY, YoY is disabled (and the
    gating callback falls the selection back to LR)."""
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
        "fontFamily": "Archivo, system-ui, sans-serif",
        "marginBottom": "8px",
    })


def _ai_diagnostic_panel(prefix):
    """Shared result-level AI diagnosis panel with mode-specific component IDs."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.Span("AI diagnosis"), _beta_badge()],
                        className="pvc-ai-diagnostic-title",
                    ),
                    html.Button(
                        "Restart",
                        id=f"{prefix}-ai-diagnostic-restart",
                        n_clicks=0,
                        className="pvc-ai-diagnostic-restart",
                        style={"display": "none"},
                    ),
                ],
                className="pvc-ai-diagnostic-header",
            ),
            html.Button(
                [html.Span("✦", className="pvc-ai-diagnostic-icon"), "Diagnose with AI"],
                id=f"{prefix}-ai-diagnostic-btn",
                n_clicks=0,
                hidden=False,
                className="pvc-ai-diagnostic-button pvc-primary-action",
            ),
            dcc.Loading(
                type="circle",
                color=ACCENT,
                children=html.Div(
                    id=f"{prefix}-ai-diagnostic-output",
                    className="pvc-ai-diagnostic-output",
                ),
            ),
        ],
        id=f"{prefix}-ai-diagnostic-card",
        className="pvc-ai-diagnostic-card",
        style={"display": "none"},
    )


calc_agent_body = html.Div([
    html.Button(
        [
            html.Span("✓", className="pvc-step-fold-check"),
            html.Span("Degradation calculated", className="pvc-step-fold-title"),
            html.Span("Click to show or hide metric settings", className="pvc-step-fold-help"),
            html.Span("⌄", className="pvc-step-fold-arrow"),
        ],
        id="toggle-metric-settings",
        n_clicks=0,
        className="pvc-step-fold-summary",
    ),
    html.Div(section_label("Choose a metric"), className="pvc-step-config-item"),
    html.Div([
        # Category 1 — statistical / trend methods (YoY, LR, HW, ARIMA, CSD).
        # Heading on the left, "Select all / Clear all" toggle on the right.
        # The button selects every enabled method or clears them; the clientside
        # sync keeps this group mutually exclusive with the PVPRO option below.
        html.Div([
            _metric_category_heading("statistical / trend methods"),
            html.Button(
                "Select all",
                id="metric-stat-selectall-btn",
                n_clicks=0,
                style={
                    "fontSize": "12px", "fontWeight": "600",
                    "fontFamily": "Archivo, system-ui, sans-serif",
                    "color": NAVY, "background": "white",
                    "border": f"1px solid {BORDER_STRONG}",
                    "borderRadius": "8px", "padding": "4px 12px",
                    "cursor": "pointer",
                },
            ),
        ], style={"display": "flex", "alignItems": "center",
                  "justifyContent": "space-between", "marginBottom": "12px"}),
        dcc.Checklist(
            id="metric-stat-radio",
            value=["YOY"],
            options=build_stat_metric_options(disable_yoy=False),
            labelStyle={"display": "block", "marginBottom": "10px",
                        "cursor": "pointer", "color": "inherit"},
            labelClassName="metric-radio-label",
            inputStyle={"marginRight": "10px", "marginTop": "3px",
                        "accentColor": NAVY},
            style={"marginBottom": "0"},
        ),
        # Populated by gate_yoy_by_duration() when the dataset is too short.
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
    ], className="pvc-step-config-item pvc-advanced-metric-panel", style={
        "padding": "16px 18px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
        "marginBottom": "14px",
    }),

    dcc.Store(id="_rb-sync-dummy"),

    html.Button(
        ["Calculate degradation ", html.Span("→")],
        id="run-btn",
        n_clicks=0,
        className="pvc-step-config-item pvc-primary-action",
        style={
            "width": "100%",
            "padding": "12px 16px",
            "background": "linear-gradient(135deg, #4b8bff, #2f6bff)",
            "color": "white",
            "border": "none",
            "borderRadius": "16px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Archivo, system-ui, sans-serif",
        }
    ),

    # Collapsible metric explanations (descriptions, equations, references)
    html.Div(metric_explanations_block(), className="pvc-step-config-item"),

    # FAST methods (YoY/LR/HW/ARIMA/CSD) render under the dcc.Loading spinner.
    # `target_components` scopes the spinner to ONLY the degradation-output's
    # own children — so updating the nested AI-diagnostic output (a different
    # component id) does NOT flash the spinner over the whole result.
    dcc.Loading(
        type="circle",
        color=ROSE,
        target_components={"degradation-output": "children"},
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
    _ai_diagnostic_panel("advanced"),
    html.Button(
        ["Next step ", html.Span("→")],
        id="advanced-next-step-3",
        n_clicks=0,
        className="pvc-next-step-button",
    ),
], id="advanced-calc-body", className="pvc-advanced-step-body",
   style={"fontFamily": "Archivo, system-ui, sans-serif"})


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
            "fontFamily": "Archivo, system-ui, sans-serif",
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
        className="pvc-primary-action",
        style={
            "width": "100%",
            "padding": "12px 16px",
            "background": "linear-gradient(135deg, #4b8bff, #2f6bff)",
            "color": "white",
            "border": "none",
            "borderRadius": "16px",
            "fontSize": "16px",
            "fontWeight": "600",
            "cursor": "pointer",
            "fontFamily": "Archivo, system-ui, sans-serif",
        }
    ),
    html.Div(
        "(typically takes 2–10 seconds)",
        style={
            "fontSize": "13px",
            "color": INK_SOFT,
            "marginTop": "6px",
            "textAlign": "center",
            "fontFamily": "Archivo, system-ui, sans-serif",
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
            "borderRadius": "12px",
            "background": "white",
            "fontFamily": "Archivo, system-ui, sans-serif",
        }
    ),
], style={"fontFamily": "Archivo, system-ui, sans-serif"})


# =============================================================================
# MAIN CHAT STREAM
# =============================================================================
def build_hero(eyebrow, sub_children):
    """Shared editorial hero used by BOTH Simple and Advanced modes.

    `eyebrow`      : the small uppercase label above the headline
                     ("Simple mode" / "Advanced mode").
    `sub_children` : the mode-specific block between the headline and the
                     active-development warning bar (the bullet list).
    The headline and warning bar are identical across modes.
    """
    return html.Div(
        [
            html.Div(eyebrow, style={
                "fontSize": "15px",
                "color": ACCENT,
                "fontFamily": "Archivo, system-ui, sans-serif",
                "fontWeight": "800",
                "textTransform": "uppercase",
                "letterSpacing": "0.12em",
                "marginBottom": "12px",
            }),
            html.H1("Agentic PV Degradation Analysis", style={
                "fontSize": "40px",
                "fontFamily": "Archivo, system-ui, sans-serif",
                "fontWeight": "850",
                "letterSpacing": "-0.02em",
                "color": INK,
                "lineHeight": "1.05",
                "margin": "0 0 12px",
            }),
            html.P(
                "Upload a PV time-series — Copilot screens, filters and computes "
                "the degradation rate end-to-end.",
                style={
                    "margin": "0 0 16px", "fontSize": "16px",
                    "lineHeight": "1.55", "color": INK_SOFT,
                    "fontFamily": "Archivo, system-ui, sans-serif",
                },
            ),

            # Mode-specific middle content (bullets).
            html.Div(sub_children),

            # Active-development banner — kept in BOTH modes.
            soft_blue_callout(
                [
                    html.B("Note: "),
                    "This tool is currently under active "
                    "development. If you encounter issues, please ",
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
        style={"padding": "32px 0 28px",
               "borderBottom": f"1px solid {BORDER}",
               "marginBottom": "32px"}
    )


# The common header (eyebrow + big title + subtitle + dev note), shown ONCE
# above the shared data-upload area, boxed in the same card style as the
# mode panels below.
common_header = html.Div(
    html.Div(
        [
            html.Div("✦  LLM-EMPOWERED PV DEGRADATION PIPELINE", style={
                "fontSize": "15px",
                "color": ACCENT,
                "fontFamily": "Archivo, system-ui, sans-serif",
                "fontWeight": "800",
                "textTransform": "uppercase",
                "letterSpacing": "0.12em",
                "marginBottom": "12px",
            }),
            html.H1("Agentic PV Degradation Analysis", style={
                "fontSize": "44px",
                "fontFamily": "Archivo, system-ui, sans-serif",
                "fontWeight": "850",
                "letterSpacing": "-0.02em",
                "color": INK,
                "lineHeight": "1.08",
                "margin": "0 0 12px",
            }),
            html.P(
                "Upload a PV time-series — Copilot screens, filters and computes "
                "the degradation rate end-to-end.",
                style={
                    "margin": 0, "fontSize": "17px", "lineHeight": "1.55",
                    "color": INK_SOFT, "fontFamily": "Archivo, system-ui, sans-serif",
                },
            ),
            soft_blue_callout(
                [
                    html.B("Note: "),
                    "This tool is currently under active development. ",
                    "Online usage doesn't require "
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
        style={"padding": "40px 44px 24px"},
    ),
    className="pvc-landing-header",
)


# One landing block: overview first, then upload and examples in a two-column
# row. It spans the former header/upload grid rows so downstream placement and
# all callback component IDs remain unchanged.
landing_upload_block = html.Div(
    [common_header, shared_upload_header],
    className="glass rise pvc-landing",
    style={
        "background": "linear-gradient(135deg, rgba(255,255,255,0.68), rgba(255,255,255,0.44))",
        "border": f"1px solid {BORDER}",
        "borderRadius": "28px",
        "boxShadow": "0 14px 44px rgba(30,58,120,0.11), inset 0 1px 0 rgba(255,255,255,0.62)",
        "backdropFilter": "blur(30px) saturate(1.5)",
        "WebkitBackdropFilter": "blur(30px) saturate(1.5)",
        "marginBottom": "22px",
        "gridColumn": "1 / -1",
        "gridRow": "2 / 4",
        "overflow": "hidden",
    },
)


# Simple-mode hero middle content — bullets describing the auto pipeline.
_simple_hero_bullets = html.Ul(
    [
        html.Li([html.B("Fully automatic"), " — prescreen, filter, and estimate for you."]),
        html.Li([html.B("Preset parameters & metric"), " — defaults chosen for you; use Advanced mode to tune them."]),
        html.Li([html.B("Want control?"), " — switch to Advanced mode above."]),
    ],
    style={
        "fontSize": "16px",
        "color": INK_SOFT,
        "lineHeight": "1.7",
        "fontFamily": "Archivo, system-ui, sans-serif",
        "maxWidth": "680px",
        "paddingLeft": "20px",
        "marginBottom": "0",
    }
)


chat_stream = html.Div(
    [
        html.Div(
            [
                # Persistent four-step rail. Completion unlocks later steps;
                # the viewed step is controlled independently by a UI store.
                html.Div(
                    [
                        html.Button([
                            html.Div([
                                html.Span("01", className="pvc-advanced-step-number"),
                                html.Span("✓", className="pvc-advanced-step-check"),
                            ], className="pvc-advanced-step-badge"),
                            html.Div([
                                html.Div("Data prescreening", className="pvc-advanced-step-name"),
                                html.Div("Raw signals & quality", className="pvc-advanced-step-caption"),
                            ], className="pvc-advanced-step-copy"),
                            html.Span("›", className="pvc-advanced-step-chevron"),
                        ], id={"type": "advanced-step-tab", "step": 1}, n_clicks=0,
                           disabled=False,
                           className="pvc-advanced-rail-card pvc-advanced-step-1 is-active"),
                        html.Button([
                            html.Div([
                                html.Span("02", className="pvc-advanced-step-number"),
                                html.Span("✓", className="pvc-advanced-step-check"),
                            ], className="pvc-advanced-step-badge"),
                            html.Div([
                                html.Div("Intelligent filtering", className="pvc-advanced-step-name"),
                                html.Div("Configure & review", className="pvc-advanced-step-caption"),
                            ], className="pvc-advanced-step-copy"),
                            html.Span("›", className="pvc-advanced-step-chevron"),
                        ], id={"type": "advanced-step-tab", "step": 2}, n_clicks=0,
                           disabled=True,
                           className="pvc-advanced-rail-card pvc-advanced-step-2 is-locked"),
                        html.Button([
                            html.Div([
                                html.Span("03", className="pvc-advanced-step-number"),
                                html.Span("✓", className="pvc-advanced-step-check"),
                            ], className="pvc-advanced-step-badge"),
                            html.Div([
                                html.Div("Degradation model", className="pvc-advanced-step-name"),
                                html.Div("Metrics & calculation", className="pvc-advanced-step-caption"),
                            ], className="pvc-advanced-step-copy"),
                            html.Span("›", className="pvc-advanced-step-chevron"),
                        ], id={"type": "advanced-step-tab", "step": 3}, n_clicks=0,
                           disabled=True,
                           className="pvc-advanced-rail-card pvc-advanced-step-3 is-locked"),
                        html.Button([
                            html.Div([
                                html.Span("04", className="pvc-advanced-step-number"),
                                html.Span("✓", className="pvc-advanced-step-check"),
                            ], className="pvc-advanced-step-badge"),
                            html.Div([
                                html.Div("Code generation", className="pvc-advanced-step-name"),
                                html.Div("Optional export", className="pvc-advanced-step-caption"),
                            ], className="pvc-advanced-step-copy"),
                            html.Span("›", className="pvc-advanced-step-chevron"),
                        ], id={"type": "advanced-step-tab", "step": 4}, n_clicks=0,
                           disabled=True,
                           className="pvc-advanced-rail-card pvc-advanced-step-4 is-locked"),
                    ],
                    className="pvc-advanced-rail",
                ),

                # Existing functional bodies are preserved verbatim and only
                # mounted in a new single-panel host.
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("STEP 01", className="pvc-advanced-panel-kicker"),
                                html.H2("Data prescreening", className="pvc-advanced-panel-title"),
                                html.P(
                                    "Validate the raw power, irradiance and temperature signals before analysis; "
                                    "detect variables and preview the raw signals (2–10 seconds).",
                                    className="pvc-advanced-panel-description",
                                ),
                                data_agent_body,
                            ],
                            id="agent-data-wrap",
                            className="pvc-advanced-stage pvc-advanced-stage-data",
                        ),
                        html.Div(
                            [
                                html.Div(id="agent-filter-locked", children=locked_placeholder("filter", "Filter Agent", 2)),
                                html.Div([
                                    html.Div("STEP 02", className="pvc-advanced-panel-kicker"),
                                    html.H2("Intelligent filtering", className="pvc-advanced-panel-title"),
                                    html.P(
                                        "Tune the filters applied before fitting — all are on by default with best-practice thresholds.",
                                        className="pvc-advanced-panel-description",
                                    ),
                                    filter_agent_body,
                                ], id="agent-filter-content", style={"display": "none"}),
                            ],
                            id="agent-filter-wrap",
                            className="pvc-advanced-stage pvc-advanced-stage-filter",
                        ),
                        html.Div(
                            [
                                html.Div(id="agent-calc-locked", children=locked_placeholder("calc", "Degradation Agent", 3)),
                                html.Div([
                                    html.Div("STEP 03", className="pvc-advanced-panel-kicker"),
                                    html.H2("Degradation model", className="pvc-advanced-panel-title"),
                                    html.P(
                                        "Pick one or more metrics (and tune their parameters), then run to compare degradation rates.",
                                        className="pvc-advanced-panel-description",
                                    ),
                                    calc_agent_body,
                                ], id="agent-calc-content", style={"display": "none"}),
                            ],
                            id="agent-calc-wrap",
                            className="pvc-advanced-stage pvc-advanced-stage-calc",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    id="agent-code-locked",
                                    children=locked_placeholder("code", "Code Agent", 3, addon=True),
                                ),
                                html.Div([
                                    html.Div("STEP 04", className="pvc-advanced-panel-kicker"),
                                    html.H2("Code generation", className="pvc-advanced-panel-title"),
                                    html.P(
                                        "Export a runnable Python script that reproduces the workflow you configured.",
                                        className="pvc-advanced-panel-description",
                                    ),
                                    code_agent_body,
                                ], id="agent-code-content", style={"display": "none"}),
                            ],
                            id="agent-code-wrap",
                            className="pvc-advanced-stage pvc-advanced-stage-code",
                        ),
                    ],
                    className="pvc-advanced-panel",
                ),
            ],
            className="pvc-advanced-layout",
        ),
    ],
    className="pvc-advanced-workspace",
)


# =============================================================================
# SHARED CHAT Q&A BLOCK  (used by BOTH Simple and Advanced modes)
#
# This was previously nested inside `chat_stream`.  Pulled out so a single
# instance can sit below whichever mode panel is active — the chat's component
# IDs (chat-composer, chat-history, the stores, etc.) must appear exactly once
# in the layout, so it cannot be duplicated per-mode.
# =============================================================================
chat_qa_block = html.Div(
            [
                # Section heading — Part 2 of the AI Assistant
                html.Div(
                    [
                        html.Div([
                            html.Span("✦ ", style={"color": ACCENT}),
                            "Chat assistance",
                        ], style={
                            "fontSize": "20px",
                            "fontWeight": "700",
                            "color": INK,
                            "fontFamily": "Archivo, system-ui, sans-serif",
                            "marginBottom": "6px",
                        }),
                        html.Div(
                            "Questions about the workflow, methods, or your results?",
                            style={
                                "fontSize": "14px",
                                "color": INK_SOFT,
                                "fontFamily": "Archivo, system-ui, sans-serif",
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
                            className="pvc-chat-messages",
                            children=[],
                            style={
                                "minHeight": "60px",
                                "maxHeight": "220px",
                                "overflowY": "auto",
                                "padding": "14px 16px",
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
                                        className="pvc-chat-input",
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
                                            "fontSize": "14px",
                                            "fontFamily": "Archivo, system-ui, sans-serif",
                                            "fontWeight": "600",
                                            "color": INK,
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
                                    className="pvc-chat-send",
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
                                        "fontFamily": "Archivo, system-ui, sans-serif",
                                        "letterSpacing": "0.02em",
                                    }
                                ),
                            ],
                            className="pvc-chat-input-row",
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "12px",
                                "padding": "14px 18px",
                                "background": "#e2e8f0",  # slate-200 composer area
                            }
                        ),
                        html.Div(
                            "PVCopilot is AI and can make mistakes.",
                            className="pvc-chat-disclaimer",
                        ),
                    ],
                    style={
                        "background": "#f8fafc",          # slate-50, light gray
                        "border": f"1px solid #e2e8f0",   # slate-200 panel edge
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
                            "fontFamily": "Archivo, system-ui, sans-serif",
                            "alignSelf": "center",
                        }),
                        html.Button(
                            "What's my degradation rate?",
                            id={"type": "chat-example", "idx": 0},
                            n_clicks=0,
                            className="pvc-chat-prompt",
                        ),
                        html.Button(
                            "Which method should I try?",
                            id={"type": "chat-example", "idx": 1},
                            n_clicks=0,
                            className="pvc-chat-prompt",
                        ),
                        html.Button(
                            "How were points filtered?",
                            id={"type": "chat-example", "idx": 2},
                            n_clicks=0,
                            className="pvc-chat-prompt",
                        ),
                        html.Button(
                            "What does PVPRO add?",
                            id={"type": "chat-example", "idx": 3},
                            n_clicks=0,
                            className="pvc-chat-prompt",
                        ),
                    ],
                    className="pvc-chat-prompts",
                    style={
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
                "padding": "20px 0 8px",
                "background": "transparent",
                "marginTop": "0",
                "scrollMarginTop": "20px",
            }
        )


# =============================================================================
# SHARED CHAT ASSISTANT  (used by both modes)
# =============================================================================
ai_assistant_block = html.Div(
    [chat_qa_block],
    id="ai-assistant-block",
)


floating_chat_widget = html.Div(
    [
        html.Button(
            [
                html.Span("✦", className="pvc-chat-pill-icon"),
                html.Span("Ask PVCopilot"),
            ],
            id="chat-drawer-open",
            n_clicks=0,
            className="pvc-chat-open",
            **{"aria-label": "Open PV Copilot chat"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("✦", className="pvc-chat-head-icon"),
                        html.Div(
                            [
                                html.Div(
                                    [html.Span("PVCopilot"), _beta_badge()],
                                    className="pvc-chat-title",
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "7px",
                                    },
                                ),
                                html.Div(
                                    [html.Span(className="pvc-chat-ready-dot"), "Ready to help"],
                                    className="pvc-chat-ready",
                                ),
                            ],
                            className="pvc-chat-head-copy",
                        ),
                        html.Button(
                            "×",
                            id="chat-drawer-close",
                            n_clicks=0,
                            className="pvc-chat-close",
                            **{"aria-label": "Close PV Copilot chat"},
                        ),
                    ],
                    className="pvc-chat-head",
                ),
                ai_assistant_block,
            ],
            id="chat-drawer-panel",
            className="pvc-chat-drawer",
            style={"display": "none"},
        ),
    ],
    id="floating-chat-widget",
)


# =============================================================================
# MODE TABS  (Simple vs Advanced)
#
# Simple mode: user drops data and immediately sees the degradation rate +
# figure.  All intermediate steps (variable table, raw-data plot, filter
# results) run with default settings under the hood but are NOT shown.
#
# Advanced mode: the full four-agent, step-by-step workflow (the original UI).
# =============================================================================
def _mode_tab(label, sub, mode_key, active):
    """One pill in the mode switcher — capsule-shaped, single line."""
    glyph = (
        html.Span(
            className="pvc-bolt-icon",
            style={"color": "#ffffff" if active else INK_SOFT},
            **{"aria-hidden": "true"},
        )
        if mode_key == "simple"
        else "⚙"
    )
    return html.Button(
        [
            html.Span(glyph, className="pvc-mode-glyph", style={
                "fontSize": "15px",
                "color": "#ffffff" if active else INK_SOFT,
            }),
            html.Span(label, style={
                "fontSize": "14px",
                "fontWeight": "700",
                "fontFamily": "Archivo, system-ui, sans-serif",
                "color": "#ffffff" if active else INK,
            }),
        ],
        id={"type": "mode-tab", "mode": mode_key},
        className="mode-tab-active" if active else "mode-tab-idle",
        n_clicks=0,
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "7px",
            "flex": "0 0 auto",
            "minWidth": "0",
            "justifyContent": "center",
            "padding": "10px 18px",
            "border": "none",
            "borderRadius": "13px",
            "cursor": "pointer",
            "background": NAVY if active else "transparent",
            "boxShadow": "0 8px 22px rgba(47,107,255,0.28)" if active else "none",
            "transition": "all 0.15s ease",
        },
    )


def build_mode_tabs(mode="simple"):
    return html.Div(
        [
            _mode_tab("Simple", "Drop data, get the rate", "simple",
                      active=(mode == "simple")),
            _mode_tab("Advanced", "Control every step", "advanced",
                      active=(mode == "advanced")),
        ],
        id="mode-tabs",
        style={
            "display": "flex",
            "flexWrap": "nowrap",
            "gap": "4px",
            "padding": "6px",
            "background": "rgba(255,255,255,0.46)",
            "border": "1px solid rgba(255,255,255,0.72)",
            "borderRadius": "17px",
            "boxShadow": "0 6px 20px rgba(30,58,120,0.08)",
            "backdropFilter": "blur(20px) saturate(1.4)",
            "marginBottom": "0",
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
        "fontWeight": "850",
        "fontFamily": "Archivo, system-ui, sans-serif",
        "letterSpacing": "-0.02em",
        "whiteSpace": "nowrap",
    }
    if disabled:
        base.update({"background": "#cbd5e1", "color": "#ffffff",
                     "cursor": "not-allowed"})
    else:
        base.update({"background": "linear-gradient(135deg, #4b8bff, #2f6bff)",
                     "color": "#ffffff", "cursor": "pointer",
                     "boxShadow": "0 10px 26px rgba(47,107,255,0.30)"})
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
            "fontFamily": "Archivo, system-ui, sans-serif",
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
        "fontFamily": "Archivo, system-ui, sans-serif",
        "letterSpacing": "0.01em",
        "whiteSpace": "nowrap",
    }
    if disabled:
        base.update({"background": "#cbd5e1", "color": "#ffffff",
                     "cursor": "not-allowed"})
    else:
        base.update({"background": "linear-gradient(135deg, #4b8bff, #2f6bff)",
                     "color": "#ffffff", "cursor": "pointer",
                     "boxShadow": "0 10px 26px rgba(47,107,255,0.30)"})
    return base


def _simple_pvpro_params_block():
    """Collapsible module/array parameters for the Simple-mode PVPRO box.
    Identical fields to the Advanced-mode PVPRO metric, but with distinct
    `simple-param-pvpro-*` ids so the two never collide."""
    return html.Div([
        # "Estimate from data" — a SIBLING of <details> (NOT inside <summary>),
        # so clicking it never toggles the panel. Absolutely positioned onto the
        # header line at the right. Runs identify-variables -> estimate mps/ps.
        html.Button(
            [html.Span("\u2726", style={"marginRight": "6px"}),
             "Estimate from data"],
            id="simple-pvpro-estimate-btn", n_clicks=0,
            title=("Identify the DC voltage / current / power columns in your "
                   "data, then estimate Modules per string and Parallel strings."),
            style={
                "position": "absolute", "top": "2px", "right": "0", "zIndex": "2",
                "fontSize": "12px", "fontWeight": "700",
                "fontFamily": "Archivo, system-ui, sans-serif", "color": NAVY,
                "background": "#fff", "border": f"1px solid {NAVY}",
                "borderRadius": "999px", "padding": "5px 14px",
                "cursor": "pointer", "whiteSpace": "nowrap",
            },
        ),
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
                    html.Span("Provide module & array parameters for PVPRO"),
                ],
                # paddingRight reserves room so the header text never slides
                # under the absolutely-positioned Estimate button.
                style={"cursor": "pointer", "fontSize": "13px", "color": INK,
                       "fontWeight": "700", "fontFamily": "Archivo, system-ui, sans-serif",
                       "marginTop": "4px", "paddingRight": "170px"},
            ),
            html.Div([
                html.Div(style={"display": "flex", "gap": "8px",
                                "flexWrap": "wrap", "marginBottom": "8px"},
                         children=[
                    html.Div(
                        _pvpro_num_field("Cells in series (per module)",
                                         "cells", 60, prefix="simple-"),
                        style={"flex": "1", "minWidth": "140px"}),
                    html.Div(
                        _pvpro_num_field("Modules per string", "mps", 1,
                                         prefix="simple-", prefillable=True),
                        style={"flex": "1", "minWidth": "140px"}),
                    html.Div(
                        _pvpro_num_field("Parallel strings", "ps", 1,
                                         prefix="simple-", prefillable=True),
                        style={"flex": "1", "minWidth": "140px"}),
                ]),
                html.Div(style={"display": "flex", "gap": "8px",
                                "flexWrap": "wrap", "marginBottom": "8px"},
                         children=[
                    html.Div(
                        _pvpro_num_field("alpha_isc (A/\u00b0C)", "alphaisc",
                                         0.0046, prefix="simple-"),
                        style={"flex": "1", "minWidth": "140px"}),
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
                    html.Div(
                        _pvpro_num_field("Days per run", "days", 14,
                                         prefix="simple-"),
                        style={"flex": "1", "minWidth": "140px"}),
                    html.Div(
                        _pvpro_num_field("Iterations per year", "iters", 12,
                                         prefix="simple-"),
                        style={"flex": "1", "minWidth": "140px"}),
                ]),
                # Spinner shows while "Estimate from data" is identifying
                # columns + estimating (parse_contents can take a few seconds).
                dcc.Loading(
                    html.Div(id="simple-pvpro-autofill-note",
                             style={"marginTop": "8px"}),
                    type="circle", color=NAVY,
                ),
            ], style={"marginTop": "18px", "padding": "12px 14px",
                      "background": "#f1f5f9", "borderRadius": "12px",
                      "border": f"1px solid {BORDER}",
                      "boxSizing": "border-box"}),
        ],
        id="simple-pvpro-params-details",
        open=False,
        style={"marginTop": "0"},
        ),
    ], style={"position": "relative", "marginTop": "12px"})


def _simple_method_radio():
    """Method chooser for Simple mode: YoY vs PVPRO.  Both are always
    selectable; if PVPRO is chosen but the data has no DC voltage/current,
    Stage 1 surfaces an error and points the user back to YoY."""
    def method_card(icon, title, description, show_logo=False, beta=False):
        title_children = [
            html.Span(title, className="pvc-simple-method-title"),
        ]
        if beta:
            title_children.append(_beta_badge())
        children = [
            html.Span(icon, className="pvc-simple-method-icon"),
            html.Span(
                [
                    html.Span(
                        title_children,
                        className="pvc-simple-method-title-row",
                    ),
                    html.Span(description, className="pvc-simple-method-description"),
                ],
                className="pvc-simple-method-copy",
            ),
        ]
        if show_logo:
            children.append(
                html.Img(
                    src=app.get_asset_url("pvpro_logo.png"),
                    alt="PVPRO",
                    className="pvc-simple-method-logo",
                )
            )
        return html.Span(children, className="pvc-simple-method-card-content")

    return dcc.RadioItems(
        id="simple-method-radio",
        options=[
            {
                "label": method_card(
                    "✓", "Year-on-year", "Fast best-practice trend fit",
                ),
                "value": "YOY",
            },
            {
                "label": method_card(
                    "✦", "PVPRO", "Physics single-diode model",
                    show_logo=True, beta=True,
                ),
                "value": "PVPRO",
            },
        ],
        value="YOY",
        className="pvc-simple-method-cards",
        labelStyle={"display": "block", "cursor": "pointer"},
        inputStyle={"position": "absolute", "opacity": "0", "pointerEvents": "none"},
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
                       "fontFamily": "Archivo, system-ui, sans-serif"},
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
                       "lineHeight": "1.6", "fontFamily": "Archivo, system-ui, sans-serif",
                       "marginTop": "8px", "padding": "10px 12px",
                       "background": "rgba(241, 245, 249, 0.6)",
                       "border": f"1px solid {BORDER}", "borderRadius": "12px"},
            ),
        ],
        open=False,
        style={"marginTop": "8px"},
    )


simple_mode_panel = html.Div(
    [
        html.Div(
            [
                html.Div("One click, full pipeline.", className="pvc-simple-heading"),
                html.Div(
                    "PV Copilot runs pre-screening and filtering with best-practice "
                    "defaults, then fits the degradation rate with your chosen method.",
                    className="pvc-simple-description",
                ),
                html.Div(
                    id="simple-method-wrap",
                    children=_simple_method_radio(),
                    className="pvc-simple-method-wrap",
                ),

                html.Div(
                    [
                        html.Div(id="simple-pvpro-about-wrap",
                                 children=_simple_pvpro_about(),
                                 style={"display": "none"}),
                        html.Div(id="simple-pvpro-params-wrap",
                                 children=_simple_pvpro_params_block(),
                                 style={"display": "none"}),
                    ],
                    className="pvc-simple-pvpro-extra",
                ),

                html.Div(
                    html.Button(
                        [
                            html.Span(
                                className="pvc-bolt-icon",
                                style={"color": "#ffffff"},
                                **{"aria-hidden": "true"},
                            ),
                            html.Span("Run analysis"),
                        ],
                        id="simple-analyze-btn",
                        n_clicks=0,
                        disabled=False,
                        className="pvc-simple-run-button pvc-primary-action",
                        style=_simple_analyze_style(disabled=False),
                    ),
                    className="pvc-simple-run-row",
                ),
                html.Div(id="simple-status", className="pvc-simple-progress"),
            ],
            id="simple-start-view",
            className="pvc-simple-start",
        ),

        # The start panel stays visible while work runs. Once either method
        # succeeds, simple-stash swaps it for this result view in-place.
        html.Div(
            [
                html.Div(id="simple-result"),
                _ai_diagnostic_panel("simple"),
                html.Div(
                    [
                        html.Button(
                            [html.Span("←", className="pvc-simple-result-action-icon"), "Return"],
                            id="simple-result-return",
                            n_clicks=0,
                            className="pvc-simple-result-return",
                        ),
                        html.Button(
                            ["Open Advanced mode to tune every step ", html.Span("→")],
                            id="simple-result-advanced",
                            n_clicks=0,
                            className="pvc-simple-result-advanced",
                        ),
                    ],
                    className="pvc-simple-result-actions",
                ),
            ],
            id="simple-result-view",
            style={"display": "none"},
        ),
        # PVPRO long-running progress renders here while a fit is in flight.
        html.Div(id="simple-pvpro-progress-output", style={"marginTop": "16px"}),
    ],
    id="simple-mode-panel",
    style={}
)


# =============================================================================
# TOP NAVIGATION + POP-UP WINDOWS
# =============================================================================
def nav_pills():
    """Compact top navigation; every item opens an in-page modal."""
    def pill(label, key):
        return html.Button(
            label,
            id={"type": "pvc-main-nav", "index": key},
            n_clicks=0,
            className="nav-pill",
        )

    return html.Div(
        [
            pill("What's new", "whatsnew"),
            pill("Team", "team"),
            pill("How to cite", "cite"),
            pill("Methods", "methods"),
        ],
        className="nav-pills",
    )


def _modal_shell(kicker, title, subtitle, body_children, modal_class=""):
    heading = [
        html.Div(
            kicker,
            style={
                "display": "inline-flex", "padding": "6px 14px",
                "borderRadius": "20px", "background": "rgba(79,139,255,0.12)",
                "border": "1px solid rgba(79,139,255,0.22)",
                "fontSize": "12.5px", "fontWeight": 700, "color": NAVY,
                "marginBottom": "14px",
            },
        ),
        html.H2(
            title,
            style={
                "margin": "0 0 8px", "fontSize": "26px", "lineHeight": "1.1",
                "fontWeight": 800, "letterSpacing": "-0.02em", "color": INK,
            },
        ),
    ]
    if subtitle:
        heading.append(html.P(
            subtitle,
            style={
                "margin": "0 0 16px", "fontSize": "14.5px",
                "lineHeight": "1.5", "color": INK_SOFT,
            },
        ))

    return html.Div(
        className="pvc-modal-overlay",
        children=html.Div(
            className="pvc-modal" + (" " + modal_class if modal_class else ""),
            children=[
                html.Button(
                    "✕", id={"type": "pvc-main-modalclose", "index": 0}, n_clicks=0,
                    className="pvc-modal-close", **{"aria-label": "Close window"},
                ),
                *heading,
                *body_children,
            ],
        ),
    )


def methods_modal():
    def row(number, title, description):
        return html.Div(
            style={
                "padding": "12px 15px", "display": "flex", "gap": "14px",
                "borderRadius": "14px", "marginBottom": "8px",
                "background": "rgba(255,255,255,0.7)",
                "border": "1px solid rgba(255,255,255,0.8)",
            },
            children=[
                html.Div(
                    str(number),
                    style={
                        "width": "34px", "height": "34px", "flexShrink": 0,
                        "borderRadius": "11px", "background": "rgba(79,139,255,0.14)",
                        "color": NAVY, "display": "flex", "alignItems": "center",
                        "justifyContent": "center", "fontWeight": 800,
                    },
                ),
                html.Div([
                    html.Div(title, style={"fontSize": "14.5px", "fontWeight": 800, "marginBottom": "2px"}),
                    html.P(description, style={"margin": 0, "fontSize": "12.5px", "lineHeight": "1.45", "color": INK_SOFT}),
                ]),
            ],
        )

    return _modal_shell(
        "Methods & documentation",
        "How PV Copilot works.",
        "Every rate comes from the same four-stage pipeline; Advanced mode exposes every knob.",
        [
            row(1, "Pre-screening & QA", "Completeness and gap checks, timezone alignment, automatic column identification, and outlier flagging on raw signals."),
            row(2, "Filtering & normalization", "Basic range checks, clear-sky detection, an irradiance threshold, temperature-corrected normalization, and night removal — all tunable."),
            row(3, "Degradation modelling", "Fit year-on-year, linear regression, Holt-Winters, ARIMA, seasonal decomposition, or the PVPRO single-diode model."),
            row(4, "Code generation", "Export a runnable Python script that reproduces your exact pipeline."),
        ],
    )


def cite_modal():
    return _modal_shell(
        "Citation",
        "How to cite this work.",
        "If PV Copilot supports your research, please cite it.",
        [
            html.Div(
                style={
                    "padding": "20px 22px", "marginBottom": "14px", "borderRadius": "16px",
                    "background": "rgba(255,255,255,0.7)",
                    "border": "1px solid rgba(255,255,255,0.8)",
                },
                children=[
                    html.Div("REFERENCE", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": NAVY, "marginBottom": "10px"}),
                    html.P([
                        html.Span("Li, B., Karin, T., Chen, X., & Jain, A. (2026). "),
                        html.Em("PV Copilot: An LLM-empowered end-to-end tool for photovoltaic degradation analysis. "),
                        html.Span("Lawrence Berkeley National Laboratory."),
                    ], style={"margin": 0, "fontSize": "15px", "lineHeight": "1.7", "color": INK}),
                ],
            ),
            html.Pre(
                "@misc{pvcopilot2026,\n"
                "  title   = {PV Copilot: An LLM-empowered end-to-end tool\n"
                "             for PV degradation analysis},\n"
                "  author  = {Li, Baojie and Karin, Todd and Chen, Xin and Jain, Anubhav},\n"
                "  year    = {2026},\n"
                "  institution = {Lawrence Berkeley National Laboratory}\n}",
                style={
                    "margin": 0, "padding": "16px 18px", "borderRadius": "14px",
                    "background": "rgba(15,23,42,0.96)", "color": "#dbe7ff",
                    "fontFamily": "'JetBrains Mono',monospace", "fontSize": "12px",
                    "lineHeight": "1.7", "overflowX": "auto",
                },
            ),
        ],
    )


def team_modal():
    members = [
        ("Baojie Li", "Lead developer & primary contributor", "team_baojieL.jpg"),
        ("Nishanth Koushik", "Algorithm development", "team_nishanthK.jpg"),
        ("Anubhav Jain", "Principal investigator", "team_anubhavJ.jpg"),
    ]

    def card(name, role, photo):
        return html.Div(className="glass pvc-team-card", children=[
            html.Img(
                src=app.get_asset_url(f"pvcopilot_team/{photo}"),
                alt=name,
                className="pvc-team-photo",
            ),
            html.Div(className="pvc-team-copy", children=[
                html.Div(name, className="pvc-team-name"),
                html.Div(role, className="pvc-team-role"),
                html.Div("Lawrence Berkeley National Laboratory", className="pvc-team-org"),
            ]),
        ])

    return _modal_shell(
        "Team",
        "Meet the PV Copilot team.",
        "Research, software, and photovoltaic degradation expertise at Berkeley Lab.",
        [html.Div([card(*member) for member in members], className="pvc-team-grid")],
        modal_class="pvc-team-modal",
    )


_CHANGELOG = [
    ("v1.3", "2026-07", [
        "Simple mode offers a YOY / PVPRO choice with automatic best-practice defaults",
        "Upload, example data, and analysis now use a streamlined full-width workflow",
        "PVCopilot chat is available from a floating expandable assistant",
    ]),
    ("v1.2", "2026-06", [
        "Advanced mode supports multi-method comparison",
        "PVPRO single-diode fitting includes live progress",
        "In-app PVCopilot chat assistant",
    ]),
    ("v1.1", "2026-05", [
        "Liquid-glass redesign integrated into the pvtools site",
        "Intelligent filtering with tunable thresholds",
    ]),
    ("v1.0", "2026-04", [
        "First release: upload → pre-screen → filter → degradation rate",
    ]),
]


def whatsnew_modal():
    rows = []
    for version, date, changes in _CHANGELOG:
        rows.append(html.Div(
            style={
                "padding": "14px 16px", "marginBottom": "10px", "borderRadius": "14px",
                "background": "rgba(255,255,255,0.7)",
                "border": "1px solid rgba(255,255,255,0.8)",
            },
            children=[
                html.Div(
                    [
                        html.Span(version, style={"fontSize": "15px", "fontWeight": 800, "color": INK}),
                        html.Span(date, style={"fontSize": "12px", "color": MUTED, "fontFamily": "'JetBrains Mono',monospace"}),
                    ],
                    style={"display": "flex", "alignItems": "baseline", "gap": "10px", "marginBottom": "6px"},
                ),
                html.Ul(
                    [html.Li(change, style={"fontSize": "12.5px", "color": INK_SOFT, "lineHeight": "1.5", "marginBottom": "2px"}) for change in changes],
                    style={"margin": 0, "paddingLeft": "20px"},
                ),
            ],
        ))
    return _modal_shell("What's new", "Version history.", "Recent releases and major changes.", rows)


def datareq_modal():
    """Data-requirements window adapted from the 260701 interface."""
    def section_heading(title, description):
        return html.Div(className="pvc-datareq-section-head", children=[
            html.Div(title, className="pvc-datareq-section-title"),
            html.Div(description, className="pvc-datareq-section-description"),
        ])

    def signal_card(level, title, fields, description, accent, tint):
        return html.Div(
            className="glass pvc-datareq-card",
            style={"--req-accent": accent, "--req-tint": tint},
            children=[
                html.Div(html.Span(level, className="pvc-datareq-level"),
                         className="pvc-datareq-card-head"),
                html.Div(
                    [html.Span(field, className="glass-soft pvc-datareq-field") for field in fields],
                    className="pvc-datareq-fields",
                ),
                html.Div(title, className="pvc-datareq-title"),
                html.P(description, className="pvc-datareq-description"),
            ],
        )

    signal_grid = html.Div(className="pvc-datareq-signal-surface", children=[
        html.Div(className="pvc-datareq-grid", children=[
            signal_card(
                "REQUIRED", "Core analysis", ["Time", "Power"],
                "The minimum signals needed to calculate a degradation trend.",
                "#c43d4b", "rgba(196,61,75,0.11)",
            ),
            html.Div("+", className="glass-soft pvc-datareq-plus", **{"aria-hidden": "true"}),
            signal_card(
                "RECOMMENDED", "Cleaner normalization", ["Irradiance", "Module temperature"],
                "Adds irradiance normalization and temperature correction.",
                "#159468", "rgba(21,148,104,0.10)",
            ),
            html.Div("+", className="glass-soft pvc-datareq-plus", **{"aria-hidden": "true"}),
            signal_card(
                "PVPRO ONLY", "Physics diagnostics", ["DC voltage", "DC current"],
                "Unlocks Pmp, Voc, Isc and other single-diode parameter trends.",
                "#667085", "rgba(102,112,133,0.12)",
            ),
        ]),
    ])

    def checklist_item(label, value, note):
        return html.Div(className="glass-soft pvc-datareq-check", children=[
            html.Div(label, className="pvc-datareq-check-label"),
            html.Div(value, className="pvc-datareq-check-value"),
            html.Div(note, className="pvc-datareq-check-note"),
        ])

    checklist = html.Div(className="pvc-datareq-file", children=[
        html.Div(className="pvc-datareq-check-grid", children=[
            checklist_item("FORMAT", "CSV, Excel, or Parquet", "One file per upload"),
            checklist_item("HISTORY", "2+ years", "Longer records are better"),
            checklist_item("SAMPLING", "1–6 hours", "Consistent intervals preferred"),
        ]),
    ])

    note = html.Div(className="pvc-datareq-note", children=[
        html.Span("✦", className="pvc-datareq-note-icon"),
        html.Span([
            html.B("Column names can vary. "),
            "PV Copilot identifies likely signals automatically, and you can review the mapping before analysis.",
        ]),
    ])

    return _modal_shell(
        "Data requirements", "Prepare your dataset.", "",
        [
            html.Div(className="pvc-datareq-section", children=[
                section_heading(
                    "Signals to include",
                    "Begin with the required pair, then add signals when your analysis needs them.",
                ),
                signal_grid,
            ]),
            html.Div(className="pvc-datareq-section", children=[
                section_heading(
                    "File format & coverage",
                    "Upload one time-series file with enough history and a consistent sampling interval.",
                ),
                checklist,
            ]),
            note,
        ],
        modal_class="pvc-datareq-modal",
    )


def render_modal(view):
    return {
        "whatsnew": whatsnew_modal,
        "team": team_modal,
        "cite": cite_modal,
        "methods": methods_modal,
        "datareq": datareq_modal,
    }.get(view, lambda: None)()


# =============================================================================
# FULL LAYOUT
# =============================================================================
_page_body = html.Div([

    # Demo-style atmospheric PV background.  It is purely decorative and sits
    # behind the production component tree, so no callback IDs are affected.
    html.Div(className="bg-layer", children=[
        html.Div(className="bg-photo"),
        html.Div(className="bg-veil"),
        html.Div(className="bg-glow"),
    ]),
    # Hidden stores (unchanged)
    dcc.Store(id="mapped-vars-store",     data={}),
    # Available column names of the currently-loaded dataset, used to
    # populate the editable variable-mapping dropdowns in Advanced Step 1.
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

    # Advanced-mode completion state. Simple mode must never write this store;
    # otherwise a one-click Simple run lights up the Advanced workflow rail.
    dcc.Store(id="step-progress", data={"data": False, "filter": False, "calc": False, "code": False}),
    # Advanced navigation is intentionally separate from completion: finishing
    # a step unlocks the next tab without moving the user away from the result.
    dcc.Store(id="advanced-active-step", data=1),
    # Completed Advanced steps default to a compact result-first view. These
    # stores remember when the user explicitly reopens their settings.
    dcc.Store(id="advanced-filter-expanded", data=False),
    dcc.Store(id="advanced-metric-expanded", data=False),

    # NEW: which analysis mode is active — "simple" (default) or "advanced".
    dcc.Store(id="ui-mode", data="simple"),

    # Simple-mode staged reveal: the pipeline computes everything at once, then
    # we animate the sidebar steps lighting up ~1.2s apart for a sense of
    # progress.  `simple-stash` holds the finished result + status until the
    # final reveal shows it.
    dcc.Store(id="simple-stash", data={}),
    dcc.Store(id="simple-step-progress", data={
        "started": False, "data": False, "filter": False,
        "calc": False, "code": False,
    }),
    # Chained Simple-mode pipeline stages.  Each stage writes the next store,
    # which triggers the next stage — so the sidebar advances exactly as each
    # real stage finishes.  These carry the intermediate dataframes (JSON).
    dcc.Store(id="simple-pipe-data",     data={}),   # after load+identify
    dcc.Store(id="simple-pipe-filtered", data={}),   # after filtering
    # Two-stage Simple-mode run: a fast callback shows a status banner instantly
    # and writes this trigger; the pipeline stages then run in sequence.
    # Payload: {"source": <btn-id|"upload">, "seq": n}.
    dcc.Store(id="simple-run-trigger", data={}),

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

    # Simple-mode PVPRO: its own job tracker + poll interval, separate from
    # the Advanced-mode ones above so the two can never collide.
    dcc.Store(id="simple-pvpro-job", data={}),
    dcc.Interval(id="simple-pvpro-poll-interval", interval=1000,
                 n_intervals=0, disabled=True),

    # The top navigation controls a single shared pop-up surface.
    dcc.Store(id="pvc-main-modal-view", data=None),
    html.Div(id="pvc-main-modal-root", style={"position": "relative", "zIndex": 6000}),

    # Floating assistant: the pill remains fixed at the lower-right and opens
    # the chat drawer without taking space in the page flow.
    floating_chat_widget,

    # Demo layout: one scrolling page with a top navigation, full-width hero
    # and upload cards, then a workflow grid (rail + analysis panel).
    dbc.Container(
        [
            html.Div(
                [
                    # Top navigation spans the complete page width.
                    html.Div(
                        [
                            html.Div("PV Copilot", className="pvc-brand-wordmark"),
                            nav_pills(),
                        ],
                        className="nav",
                        style={"gridColumn": "1 / -1", "gridRow": "1"},
                    ),

                    # Main workflow content.
                    html.Div(
                        [
                            # display:contents lets the hero/upload/analyze cards
                            # participate directly in the outer page grid.
                            html.Div(
                                [
                                    # Overview + upload + examples share one card.
                                    landing_upload_block,

                                    # ── 2 · Analyze card ─────────────────────
                                    # Title + mode tabs + the active mode panel,
                                    # all inside one card matching the others.
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div([
                                                        html.Span("ANALYZE", style={
                                                            "fontSize": "15px", "color": ACCENT,
                                                            "fontWeight": "800", "fontFamily": "Archivo, system-ui, sans-serif",
                                                            "textTransform": "uppercase", "letterSpacing": "0.12em",
                                                        }),
                                                        # Filled with the uploaded file name (see callback).
                                                        html.Span(id="analyze-title-file", style={
                                                            "fontSize": "14px", "color": INK_SOFT,
                                                            "fontWeight": "700", "fontFamily": "Archivo, system-ui, sans-serif",
                                                            "marginLeft": "10px",
                                                        }),
                                                    ], className="pvc-analyze-title"),
                                                    html.Div(
                                                        id="mode-tabs-render",
                                                        children=build_mode_tabs("simple"),
                                                        className="pvc-mode-tabs-top",
                                                    ),
                                                ],
                                                className="pvc-analyze-header",
                                            ),
                                            # SIMPLE-MODE PANEL — visible by default.
                                            html.Div(
                                                simple_mode_panel,
                                                id="simple-mode-wrap",
                                                style={},
                                            ),

                                            # ADVANCED-MODE content — hidden until switch.
                                            html.Div(
                                                chat_stream,
                                                id="advanced-mode-wrap",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        id="analyze-section-card",
                                        className="glass rise pvc-analyze-card is-hidden",
                                        style={
                                            "padding": "32px 40px 36px",
                                            "background": "linear-gradient(135deg, rgba(255,255,255,0.66), rgba(255,255,255,0.44))",
                                            "border": f"1px solid {BORDER}",
                                            "borderRadius": "28px",
                                            "boxShadow": "0 14px 44px rgba(30,58,120,0.10)",
                                            "backdropFilter": "blur(30px) saturate(1.5)",
                                            "WebkitBackdropFilter": "blur(30px) saturate(1.5)",
                                            "marginBottom": "0",
                                            "gridColumn": "1 / -1",
                                            "gridRow": "4",
                                        },
                                    ),

                                ],
                                style={
                                    "display": "contents",
                                },
                            ),
                        ],
                        style={
                            "display": "contents",
                        },
                    ),
                ],
                className="pvcopilot-shell",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1fr)",
                    "gridTemplateRows": "auto auto auto auto auto",
                    "alignItems": "flex-start",
                    "columnGap": "0",
                    "rowGap": "12px",
                    "background": "transparent",
                    "fontFamily": "Archivo, system-ui, sans-serif",
                    "color": INK,
                    "position": "relative",
                    "zIndex": "1",
                }
            ),

        ],
        fluid=False,
        style={
            "paddingTop": "26px", "paddingBottom": "8px",
            "maxWidth": "1320px", "position": "relative", "zIndex": "1",
        }
    ),
],
className="pvcopilot-root",
)

# dmc components (the variable-mapping Selects) need Mantine context. Wrapping
# this page's body in a MantineProvider supplies it to every descendant,
# including the Selects the analyze/apply callbacks insert dynamically. (If the
# app root in app.py already provides one, this nests harmlessly.)
layout = dmc.MantineProvider(_page_body)


@app.callback(
    Output("pvc-main-modal-view", "data"),
    Input({"type": "pvc-main-nav", "index": ALL}, "n_clicks"),
    Input({"type": "pvc-main-modalclose", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_nav_modal(nav_clicks, close_clicks):
    """Open the requested navigation window or close the current one."""
    trigger = ctx.triggered_id
    value = ctx.triggered[0]["value"] if ctx.triggered else None
    if not isinstance(trigger, dict) or not value:
        return dash.no_update
    if trigger.get("type") == "pvc-main-nav":
        return trigger["index"]
    if trigger.get("type") == "pvc-main-modalclose":
        return None
    return dash.no_update


@app.callback(
    Output("pvc-main-modal-root", "children"),
    Input("pvc-main-modal-view", "data"),
)
def show_nav_modal(view):
    return render_modal(view) if view else None


# Open/close the floating chat entirely in the browser for immediate feedback.
app.clientside_callback(
    """
    function(openClicks, closeClicks) {
        const ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) {
            return {display: "none"};
        }
        const trigger = ctx.triggered[0].prop_id.split('.')[0];
        if (trigger === "chat-drawer-close" && closeClicks) {
            return {display: "none"};
        }
        if (trigger === "chat-drawer-open" && openClicks) {
            return {display: "flex"};
        }
        return {display: "none"};
    }
    """,
    Output("chat-drawer-panel", "style"),
    Input("chat-drawer-open", "n_clicks"),
    Input("chat-drawer-close", "n_clicks"),
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
# (statistical methods vs PVPRO).  The statistical group is a multi-select
# checklist; PVPRO is a single radio.  We mirror the FIRST checked stat
# method (or "PVPRO") into the hidden `metric-selected-visible` radio, which
# downstream callbacks read to detect the PVPRO branch.  Picking PVPRO clears
# the stat checklist and vice-versa, so the two groups stay mutually
# exclusive (you run stat methods OR PVPRO, never both at once).
# =============================================================================
app.clientside_callback(
    """
    function(statVals, pvproVal) {
        var nu = dash_clientside.no_update;
        statVals = statVals || [];
        // Master value: "PVPRO" when PVPRO is picked, otherwise the FIRST
        // checked statistical method (a non-PVPRO code). Downstream callbacks
        // only ever test master === "PVPRO"; the run callback reads the full
        // checked list from metric-stat-radio directly for the stat methods.
        var triggered = dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) {
            // Initial firing.
            if (pvproVal) { return ["PVPRO", [], pvproVal]; }
            if (statVals.length) { return [statVals[0], statVals, null]; }
            return ["YOY", ["YOY"], null];
        }
        var prop = triggered[0].prop_id;  // "metric-stat-radio.value" etc.
        if (prop.indexOf("metric-pvpro-radio") === 0 && pvproVal) {
            // PVPRO picked -> clear the stat group, mirror PVPRO.
            return ["PVPRO", [], pvproVal];
        }
        if (prop.indexOf("metric-stat-radio") === 0) {
            if (statVals.length) {
                // One or more stat methods checked -> clear PVPRO, mirror the
                // first checked method into the master. Pass statVals back
                // unchanged (same reference) so no echo re-fire is needed.
                return [statVals[0], statVals, null];
            }
            // Everything unchecked: leave the master as-is (avoids clobbering
            // a PVPRO selection when the stat group merely emptied).
            return [nu, statVals, pvproVal];
        }
        // Fallback.
        if (pvproVal) { return ["PVPRO", [], pvproVal]; }
        if (statVals.length) { return [statVals[0], statVals, null]; }
        return ["YOY", ["YOY"], null];
    }
    """,
    Output("metric-selected-visible", "value"),
    Output("metric-stat-radio",       "value"),
    Output("metric-pvpro-radio",      "value"),
    Input("metric-stat-radio",  "value"),
    Input("metric-pvpro-radio", "value"),
)


# =============================================================================
# CALLBACK — DISABLE YoY FOR SHORT DATASETS (< _MIN_YEARS_FOR_YOY)
#
# YoY pairs each day with the same calendar day one year earlier, so it needs a
# span longer than a year to yield any comparison. When the analyzed dataset is
# shorter, grey out the YoY option and, if it was selected, fall the selection
# back to LR (the clientside sync above then mirrors it into the hidden master
# radio). Fires whenever a new dataset is analyzed (dataframe-store changes).
# =============================================================================
@app.callback(
    Output("metric-stat-radio", "options", allow_duplicate=True),
    Output("metric-stat-radio", "value",   allow_duplicate=True),
    Output("yoy-disabled-note", "children"),
    Output("yoy-disabled-note", "style"),
    Input("dataframe-store",     "data"),
    State("metric-stat-radio",   "value"),
    prevent_initial_call=True,
)
def gate_yoy_by_duration(df_json, current_value):
    try:
        df = _df_from_store(df_json) if df_json else None
    except Exception:
        df = None
    duration_years = _duration_years(df) if df is not None else None

    disable_yoy = duration_years is not None and duration_years < _MIN_YEARS_FOR_YOY
    options = build_stat_metric_options(disable_yoy=disable_yoy)
    if disable_yoy:
        # current_value is now a list of checked methods. Drop YoY if present;
        # fall back to LR only if that would otherwise leave nothing checked.
        current = list(current_value) if isinstance(current_value, (list, tuple)) else \
            ([current_value] if current_value else [])
        if "YOY" in current:
            current = [m for m in current if m != "YOY"]
            new_value = current if current else ["LR"]
        else:
            new_value = dash.no_update
        note = (f"YoY needs at least {_MIN_YEARS_FOR_YOY:g} year"
                f"{'s' if _MIN_YEARS_FOR_YOY != 1 else ''} of data; "
                "it's disabled for this dataset.")
        note_style = {"fontSize": "12px", "color": "#92400e", "fontStyle": "italic",
                      "marginTop": "8px", "fontFamily": "Archivo, system-ui, sans-serif"}
    else:
        new_value = dash.no_update
        note = ""
        note_style = {"display": "none"}
    return options, new_value, note, note_style


# =============================================================================
# CALLBACK — "Select all / Clear all" toggle for the statistical-method
# checklist. One button: if every currently-enabled method is already checked
# it clears the selection, otherwise it checks them all (skipping any option
# greyed out by the YoY duration gate). A second callback keeps the button
# label in sync with the current selection.
# =============================================================================
def _enabled_stat_values(options):
    """Return the values of the non-disabled options in the stat checklist."""
    vals = []
    for o in (options or []):
        if isinstance(o, dict) and not o.get("disabled"):
            v = o.get("value")
            if v is not None:
                vals.append(v)
    if not vals:  # fallback if options didn't round-trip as expected
        vals = ["YOY", "LR", "HW", "ARIMA", "CSD"]
    return vals


@app.callback(
    Output("metric-stat-radio", "value", allow_duplicate=True),
    Input("metric-stat-selectall-btn", "n_clicks"),
    State("metric-stat-radio", "value"),
    State("metric-stat-radio", "options"),
    prevent_initial_call=True,
)
def toggle_select_all_metrics(n_clicks, current_value, options):
    enabled = _enabled_stat_values(options)
    current = set(current_value or [])
    # Every enabled method already checked -> clear; otherwise select them all.
    if current.issuperset(set(enabled)):
        return []
    return enabled


@app.callback(
    Output("metric-stat-selectall-btn", "children"),
    Input("metric-stat-radio", "value"),
    State("metric-stat-radio", "options"),
)
def label_select_all_btn(current_value, options):
    enabled = _enabled_stat_values(options)
    current = set(current_value or [])
    return "Clear all" if current.issuperset(set(enabled)) else "Select all"


# =============================================================================
# CALLBACK — UPLOAD STATUS  (UNCHANGED LOGIC, restyled output)
# =============================================================================
@app.callback(
    Output("upload-status-output", "children"),
    Output("data-source-store",    "data"),
    Output("data-summary-output",  "children"),
    Output("stored-data-file-name","data"),
    Input("upload-data", "filename"),
    prevent_initial_call=True
)
def update_upload_status(filename):
    if filename:
        msg = html.Div(
            [
                html.B("File selected: ", style={"color": "#15803d"}),
                html.Span(filename, style={"color": "#15803d"}),
            ],
            style={
                "padding": "12px 16px",
                "background": "#ecfdf5",
                "border": "1px solid #86efac",
                "borderRadius": "16px",
                "fontSize": "15px",
                "fontFamily": "Archivo, system-ui, sans-serif",
            },
            className="slide-in-top",
        )
        return [msg, "upload", "", filename]
    return ["", None, "", None]


@app.callback(
    Output("analyze-title-file", "children"),
    Input("stored-data-file-name", "data"),
)
def show_analyze_filename(filename):
    """Show the loaded file's name as a pill next to the ANALYZE title.
    Returns "" (nothing rendered) when no file is loaded, so no empty pill."""
    if not filename:
        return ""
    return html.Span(filename, style={
        "display": "inline-block",
        "padding": "3px 12px",
        "background": "#e6f2fb",           # light blue
        "border": "1px solid #a6cded",     # light blue border
        "borderRadius": "999px",
        "fontSize": "13px",
        "fontWeight": "600",
        "color": "#0064AB",                # dark blue (NAVY)
        "fontFamily": "Archivo, system-ui, sans-serif",
        "letterSpacing": "0",
        "textTransform": "none",
        "lineHeight": "1.4",
    })


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

    # Brief pause so Step 2 (Filter) reads as actively working.
    if trigger == "filter-btn":
        time.sleep(1)

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
            "fontFamily": "Archivo, system-ui, sans-serif",
        }),
        # Headline percentage -- this is the "major" featured number for
        # this step, so it gets the major-value blue.
        html.Div(f"{pct_good:.1%}", style={
            "fontSize": "44px",
            "fontFamily": "Archivo, system-ui, sans-serif",
            "fontWeight": "700",
            "color": VALUE_MAJOR,
            "lineHeight": "1",
            "marginBottom": "4px",
        }),
        html.Div("high-quality points retained", style={
            "fontSize": "15px", "color": INK_SOFT,
            "fontFamily": "Archivo, system-ui, sans-serif",
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
        ], style={"fontFamily": "Archivo, system-ui, sans-serif"}),
        html.Details([
            html.Summary("Show details", style={"color": INK_SOFT, "cursor": "pointer", "fontSize": "13px", "marginTop": "10px", "fontFamily": "Archivo, system-ui, sans-serif"}),
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
                "borderRadius": "16px",
                "padding": "10px 12px",
                "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
            },
        ),
    ], className="slide-in-up", style={
        "padding": "20px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
        "marginTop": "16px",
    })

    df_filtered_store = df_filtered.loc[normal_indices]
    return [filter_layout, df_filtered_store.to_json(date_format="iso", orient="split")]


# =============================================================================
# MULTI-METHOD SUPPORT (statistical / trend methods)
#
# The Step-3 "statistical / trend methods" chooser is a multi-select checklist.
# When more than one method is checked, we run them all on the same daily
# series and present a comparison: a bar chart of each method's rate, and one
# combined power-trend figure that overlays every method's fitted trend on a
# single shared scatter of the daily power.
# =============================================================================

# Distinct trend-line colors for the overlay / bars, one per method. Kept in a
# fixed order so a given method always gets the same color across runs. Palette
# is the requested blue/green swatch set (deep blue -> sky blue -> turquoise ->
# dark green -> lime green).
_STAT_METHOD_COLORS = {
    "YOY":   "#0070C0",   # deep blue
    "LR":    "#83CBEB",   # sky blue
    "HW":    "#68DBCE",   # turquoise
    "ARIMA": "#048E2F",   # dark green
    "CSD":   "#92D050",   # lime green
}


def _dispatch_stat_method(method, daily_data, params):
    """Run a single statistical / trend method on the daily series and return
    (rate_percent_per_year, plotly_figure).  `params` carries the per-method
    tunables read from the Step-3 "Customize parameters" panels."""
    if method == "YOY":
        return compute_yoy(daily_data,
                           rolling_window=params.get("yoy_window") or 30,
                           iqr_multiplier=params.get("yoy_iqr") or 1.5)
    if method == "LR":
        return compute_lr(daily_data)
    if method == "HW":
        return compute_hw(daily_data, period=params.get("hw_period") or 12)
    if method == "ARIMA":
        return compute_arima(daily_data,
                             p=params.get("arima_p") if params.get("arima_p") is not None else 1,
                             d=params.get("arima_d") if params.get("arima_d") is not None else 1,
                             q=params.get("arima_q") if params.get("arima_q") is not None else 0,
                             seasonal_period=params.get("arima_s") or 12)
    if method == "CSD":
        return compute_csd(daily_data, period=params.get("csd_period") or 12)
    raise ValueError(f"Unknown metric: {method}")


def _build_multi_method_layout(results, daily_data, start_date, end_date,
                               duration_years):
    """Build the Step-3 result block for a MULTI-method run.

    `results` is a list of (method_code, rate_pct_per_year, figure) tuples in
    the order the methods were checked.  Renders (1) a bar chart comparing the
    methods' rates and (2) a single combined power-trend figure overlaying each
    method's fitted trend on one shared daily-power scatter.
    """
    # ---- Bar chart: rate by method -----------------------------------------
    bar_labels = [m for (m, _rd, _f) in results]
    bar_rates  = [(float(rd) if rd is not None and np.isfinite(rd) else None)
                  for (_m, rd, _f) in results]
    bar_colors = [_STAT_METHOD_COLORS.get(m, NAVY) for m in bar_labels]
    # Drop the "%/yr" suffix on the in-chart labels to save space (the axis
    # title already says "Rate (%/yr)"); a bigger font keeps them readable.
    bar_text   = [f"{r:+.2f}" if r is not None else "n/a" for r in bar_rates]

    bar_fig = go.Figure(go.Bar(
        x=bar_labels,
        y=[r if r is not None else 0 for r in bar_rates],
        marker_color=bar_colors,
        text=bar_text,
        textposition="outside",
        textfont=dict(family="Arial", size=13, color=INK),
        cliponaxis=False,
        # Fixed bar width in category units (each category is 1 unit apart), so
        # bars stay a sensible width regardless of how many methods are chosen
        # -- in particular they don't balloon when only 2 are selected.
        width=0.45,
        hovertemplate="%{x}: %{y:+.2f}%/yr<extra></extra>",
    ))
    bar_fig.update_layout(
        yaxis_title="Rate (%/yr)",
        xaxis_title="Method",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=INK),
        margin=dict(l=62, r=24, t=22, b=52),
        height=310,
        showlegend=False,
    )
    bar_fig.update_xaxes(showgrid=False, zeroline=False)
    bar_fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=True,
                         zerolinecolor=BORDER_STRONG)

    # ---- Combined trend overlay: one scatter + every method's trend line ----
    combined = go.Figure()
    combined.add_trace(go.Scatter(
        x=daily_data.index,
        y=daily_data.values,
        mode="markers",
        marker=dict(size=6, opacity=0.40, color="#C7D9EC"),
        name="Daily power",
    ))
    for (m, rd, fig) in results:
        # Each per-method figure has the trend/fit as its 2nd trace
        # (trace 0 is the daily scatter). Pull it out and re-color it.
        if fig is None or len(fig.data) < 2:
            continue
        trend_trace = fig.data[1]
        rate_txt = f"{rd:+.2f}%/yr" if (rd is not None and np.isfinite(rd)) else "n/a"
        combined.add_trace(go.Scatter(
            x=trend_trace.x,
            y=trend_trace.y,
            mode="lines",
            line=dict(color=_STAT_METHOD_COLORS.get(m, NAVY), width=2.5),
            name=f"{m} ({rate_txt})",
        ))
    # Some fits (notably ARIMA) can throw a large transient spike in their
    # first few fitted points, which otherwise squashes the whole plot. Set a
    # readable DEFAULT y-range from the robust spread of the *daily power*
    # (the ground-truth series), padded a little. The user can still zoom out
    # / autoscale from the figure's toolbar to see the full excursion.
    try:
        _dvals = pd.Series(daily_data).replace([np.inf, -np.inf], np.nan).dropna().values
        _ylo = float(np.nanpercentile(_dvals, 1))
        _yhi = float(np.nanpercentile(_dvals, 99))
        _pad = 0.08 * (_yhi - _ylo) if _yhi > _ylo else max(abs(_yhi), 1.0) * 0.1
        _y_range = [_ylo - _pad, _yhi + _pad]
    except Exception:
        _y_range = None

    # X-range pinned to the data span so the plot edges sit flush with the
    # first/last observation and never extend past it.
    try:
        _xidx = pd.Series(daily_data).dropna().index
        _x_range = [_xidx.min(), _xidx.max()]
    except Exception:
        _x_range = None

    combined.update_layout(
        title=None,
        xaxis_title="Time",
        yaxis_title="Power (W)",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=INK),
        margin=dict(l=64, r=190, t=22, b=54),
        height=390,
        # Legend: vertical, placed OUTSIDE the plot on the right so it never
        # overlaps the trend lines. Transparent background.
        legend=dict(orientation="v", yanchor="top", y=1.0,
                    xanchor="left", x=1.02, bgcolor="rgba(0,0,0,0)",
                    borderwidth=0, font=dict(size=11)),
    )
    combined.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False,
                          range=_x_range, autorange=(_x_range is None))
    combined.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False,
                          range=_y_range, autorange=(_y_range is None))

    # ---- Header summary -----------------------------------------------------
    finite = [(m, rd) for (m, rd, _f) in results
              if rd is not None and np.isfinite(rd)]

    if finite:
        rates_only = np.asarray([rd for (_m, rd) in finite], dtype=float)
        mean_rate = float(np.mean(rates_only))
        spread_rate = float(np.std(rates_only))
        headline = f"{mean_rate:+.2f} ± {spread_rate:.2f}"
    else:
        headline = "n/a"

    start_text = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
    end_text = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
    summary_block = html.Div(className="pvc-advanced-multi-summary", children=[
        html.Div("Degradation summary", className="pvc-advanced-result-kicker"),
        html.Div(headline, className="pvc-advanced-multi-rate"),
        html.Div("%/year", className="pvc-advanced-multi-unit"),
        html.Div(className="pvc-advanced-result-details", children=[
            html.Div([html.Span("Methods: "), html.Strong(", ".join(bar_labels))]),
            html.Div([html.Span("Duration: "), html.Strong(f"{duration_years:.1f} years")]),
            html.Div([html.Span("Window: "), html.Strong(f"{start_text} → {end_text}")]),
        ]),
    ])

    return html.Div(className="pvc-advanced-multi-result slide-in-up", children=[
        html.Div(className="pvc-advanced-multi-top", children=[
            summary_block,
            html.Div(className="pvc-advanced-result-chart-card", children=[
                html.Div("Annual degradation rate — method comparison",
                         className="pvc-advanced-result-kicker"),
                dcc.Graph(figure=bar_fig, config={"displayModeBar": False, "responsive": True}),
            ]),
        ]),
        html.Div(className="pvc-advanced-result-chart-card pvc-advanced-multi-trend", children=[
            html.Div("Power trend — all selected methods",
                     className="pvc-advanced-result-kicker"),
            dcc.Graph(
                figure=combined,
                config={"displaylogo": False, "scrollZoom": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "responsive": True},
            ),
        ]),
    ])


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
    State("metric-stat-radio",       "value"),
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
        selected_stat_methods,
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

    # Brief pause so Step 3 reads as actively working. PVPRO has live progress.
    if trigger == "run-btn" and selected_metric != "PVPRO":
        time.sleep(1)

    # ---------- PVPRO: long-running, so launch in a thread and let a polling
    # ---------- callback render the result when it's ready.
    if selected_metric == "PVPRO":
        # Snapshot all the user-controlled params into kwargs.
        pvpro_kwargs = dict(
            cells_in_series     = _pvnum(pvpro_cells, 60, int),
            modules_per_string  = _pvnum(pvpro_mps, 1, int),
            parallel_strings    = _pvnum(pvpro_ps, 1, int),
            alpha_isc           = _pvnum(pvpro_alphaisc, 0.0046, float),
            technology          = pvpro_tech      if pvpro_tech      else "mono-c-Si",
            days_per_run        = _pvnum(pvpro_days, 14, int),
            iterations_per_year = _pvnum(pvpro_iters, 12, int),
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
        initial_ui = _pvpro_progress_ui(
            phase="starting", current=0, total=1,
            message="Spinning up PVPRO worker…", elapsed_s=0,
        )
        # Clear any previous fast-method output, send the progress UI to its
        # own (Loading-free) container, disable the run button, enable poll.
        return ["", initial_ui, True, "Running PVPRO…",
                {}, {"job_id": job_id}, False]

    else:
        daily_data = aggregate_daily(df_filtered, irra_key)

        # The statistical / trend chooser is a multi-select checklist. Read the
        # full checked list; fall back to the master value / YoY if somehow
        # empty. PVPRO is handled entirely in the branch above, so anything
        # here is one or more of YOY/LR/HW/ARIMA/CSD.
        methods = [m for m in (selected_stat_methods or []) if m and m != "PVPRO"]
        if not methods:
            methods = [selected_metric] if (selected_metric and
                                            selected_metric != "PVPRO") else ["YOY"]

        stat_params = dict(
            yoy_window=yoy_window, yoy_iqr=yoy_iqr, hw_period=hw_period,
            arima_p=arima_p, arima_d=arima_d, arima_q=arima_q, arima_s=arima_s,
            csd_period=csd_period,
        )

        # ---- MULTI-METHOD: run them all, render a comparison ----------------
        if len(methods) > 1:
            results = []            # list of (method, rd, fig)
            for m in methods:
                try:
                    rd_m, fig_m = _dispatch_stat_method(m, daily_data, stat_params)
                except Exception as exc:
                    rd_m, fig_m = np.nan, None
                    print(f"[degradation] {m} failed: {exc}")
                results.append((m, rd_m, fig_m))

            start_date = df_filtered.index.min()
            end_date   = df_filtered.index.max()
            duration_years = (end_date - start_date).days / 365.25

            multi_layout = _build_multi_method_layout(
                results, daily_data, start_date, end_date, duration_years)

            methods_rates = {
                m: (round(float(rd), 4) if rd is not None and np.isfinite(rd) else None)
                for (m, rd, _f) in results
            }
            # Primary rate for backward-compatible consumers: first finite rate.
            primary_rate = next(
                (rd for (_m, rd, _f) in results
                 if rd is not None and np.isfinite(rd)), np.nan)
            primary_pct = (float(primary_rate) if np.isfinite(primary_rate) else 0.0) / 100

            rates_line = "; ".join(
                f"{m}: {r:+.2f}%/yr" if r is not None else f"{m}: n/a"
                for m, r in methods_rates.items()
            )
            trend_summary = _summarize_daily_series(
                daily_data, "multiple methods (" + ", ".join(methods) + ")")
            trend_summary = f"{trend_summary}\nPer-method rates — {rates_line}."

            result_dict = {
                "rate_pct_per_year": round(float(primary_rate) * 100, 4)
                    if np.isfinite(primary_rate) else None,
                "method": ", ".join(methods),
                "duration_years": round(float(duration_years), 2),
                "start": start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
                "end":   end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
                "rate_pct": float(primary_pct),
                "n_raw": int(len(df_filtered)),
                "n_kept": int(len(df_filtered)),
                "pct_kept": 100.0,
                "trend_summary": trend_summary,
                # Extra field: every checked method's rate (%/yr).
                "methods_rates": methods_rates,
            }
            return [multi_layout, "", False, "Calculate Degradation",
                    result_dict, {}, True]

        # ---- SINGLE METHOD: keep the featured hero-number display -----------
        selected_metric = methods[0]
        rd, fig = _dispatch_stat_method(selected_metric, daily_data, stat_params)

    # Restyle the figure
    if fig is not None:
        if len(fig.data) > 0:
            fig.data[0].name = "Daily-aggregated Power"
            if hasattr(fig.data[0], "marker"):
                fig.data[0].marker.update(size=8, opacity=0.58, color="#9fcaf1")
        if len(fig.data) > 1:
            fig.data[1].name = f"{selected_metric} trend"
            if hasattr(fig.data[1], "line"):
                fig.data[1].line.update(color="#0878c9", width=3)
        fig.update_layout(
            title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Archivo, Arial, sans-serif", color=INK_SOFT, size=13),
            margin=dict(l=64, r=28, t=20, b=86),
            height=390,
            legend=dict(
                orientation="h", x=.5, xanchor="center", y=-.22, yanchor="top",
                bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=INK_SOFT),
            ),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)

    start_date = df_filtered.index.min()
    end_date   = df_filtered.index.max()
    duration_years = (end_date - start_date).days / 365.25

    # Editorial summary with featured rate display.
    rate_pct = rd / 100
    # Single unified rate color across methods (major-value blue).
    rate_color = VALUE_MAJOR

    start_text = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
    end_text = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
    summary_block = html.Div(className="pvc-advanced-single-summary", children=[
        html.Div("Annual degradation rate", className="pvc-advanced-result-kicker"),
        html.Div(className="pvc-advanced-single-rate", children=[
            html.Span(f"{rate_pct:.2%}", className="pvc-advanced-single-rate-value"),
            html.Span("/year", className="pvc-advanced-single-rate-unit"),
        ]),
        html.Div(className="pvc-advanced-result-details", children=[
            html.Div([html.Span("Method: "), html.Strong(selected_metric)]),
            html.Div([html.Span("Duration: "), html.Strong(f"{duration_years:.1f} years")]),
            html.Div([html.Span("Window: "), html.Strong(f"{start_text} → {end_text}")]),
        ]),
    ])

    degradation_layout = html.Div(className="pvc-advanced-single-result slide-in-up", children=[
        summary_block,
        html.Div(className="pvc-advanced-result-chart-card", children=[
            html.Div("Power trend", className="pvc-advanced-result-kicker"),
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False, "displaylogo": False, "responsive": True},
            ),
        ]),
    ])

    result_dict = {
        "rate_pct_per_year": round(float(rate_pct) * 100, 4),
        "method": selected_metric,
        "duration_years": round(float(duration_years), 2),
        "start": start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
        "end":   end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
        # Extra fields the AI diagnostic uses (don't affect existing consumers).
        "rate_pct": float(rate_pct),
        "n_raw": int(len(df_filtered)),
        "n_kept": int(len(df_filtered)),
        "pct_kept": 100.0,
        "trend_summary": _summarize_daily_series(daily_data, _metric_label(selected_metric)),
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
    State("mapped-vars-store", "data"),
    prevent_initial_call=True,
)
def _pvpro_poll_callback(_n, job_store, df_filtered_json, mapped_variables_dict):
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
        lost_ui = html.Div(
                "PVPRO progress lost — this worker has never seen the job "
                "that was started. If you're on a multi-worker deployment, "
                "enable diskcache (set PVPRO_DISKCACHE_DIR + install "
                "diskcache). Expand the Debug panel below for details.",
                style={
                    "padding": "12px 14px",
                    "background": "#fff7ed",
                    "border": "1px solid #fed7aa",
                    "borderRadius": "16px",
                    "color": "#7c2d12",
                    "fontSize": "13px",
                    "fontFamily": "Archivo, system-ui, sans-serif",
                },
            )
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
        ui = _pvpro_progress_ui(
                phase=phase,
                current=job.get("current", 0),
                total=job.get("total", 1),
                message=job.get("message", ""),
                elapsed_s=elapsed,
            )
        return [ui, True, "Running PVPRO…", no_update, job_store, False]

    # --- Failed ---
    if phase == "error":
        # Keep the terminal error in the backend until the browser confirms
        # receipt by clearing its job store in this same response.  If this
        # response is lost, the next poll can deliver the error again instead
        # of seeing "rendered" and returning no_update forever.
        return [
            _no_data_alert(f"PVPRO failed: {job.get('error', 'unknown error')}"),
            False, "Calculate Degradation", {}, {}, True,
        ]

    # --- Done OR legacy-rendered: render the final layout under a per-job
    # lock.  Crucially, do NOT mark the backend job as rendered before the
    # browser has received the response.  HTTP gives us no acknowledgement
    # that Dash applied the payload; on a slow/mobile connection that one
    # final response may be lost or overtaken.  Leaving the job in "done"
    # makes delivery idempotent: while the browser still has job_id and its
    # Interval enabled, a later poll can return the same final UI again.
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
        # Re-read inside the lock. A missing job cannot be rendered, but both
        # "done" and the legacy "rendered" state remain deliverable.
        job = _pvpro_read_job(job_id)
        if job is None:
            _pvpro_debug("done_branch_skip",
                         job_id=job_id[:8],
                         reason="job-gone")
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

        # QUANT_LABELS retained here for the AI-diagnostic summary below.
        QUANT_LABELS = [
            ("p_mp_ref", "Pmp", "max power point power"),
            ("v_mp_ref", "Vmp", "max power point voltage"),
            ("i_mp_ref", "Imp", "max power point current"),
            ("v_oc_ref", "Voc", "open-circuit voltage"),
            ("i_sc_ref", "Isc", "short-circuit current"),
        ]

        # Shared renderer: headline rate, merged window+duration line, and
        # the per-parameter rates folded into a collapsible "detail".
        final_layout = _render_pvpro_layout(
            rd, figs, rates, start_str, end_str, duration_years, elapsed)

        rates_per_quantity = {k: round(float(v), 4)
                              for k, v in rates.items()
                              if v is not None and np.isfinite(v)}

        # Build a compact trend summary for the AI diagnostic.  PVPRO doesn't
        # produce a daily power series the way the synchronous path does, so we
        # synthesize the summary from the headline Pmp rate plus the per-
        # quantity reference rates that PVPRO fits.
        if np.isfinite(rate_pct):
            _summary_parts = [
                f"PVPRO-fitted reference-condition degradation. "
                f"Pmp(ref): {rate_pct*100:+.2f}%/yr over {duration_years:.1f} years."
            ]
            _quant_bits = []
            for _key, _short, _ in QUANT_LABELS:
                _r = rates.get(_key, float("nan"))
                if np.isfinite(_r):
                    _quant_bits.append(f"{_short}(ref) {_r:+.2f}%/yr")
            if _quant_bits:
                _summary_parts.append("Per-parameter rates: "
                                      + ", ".join(_quant_bits) + ".")
            trend_summary = " ".join(_summary_parts)
        else:
            trend_summary = "PVPRO fit did not return a finite Pmp rate."

        # Raw-channel data-quality scan (Advanced mode only) -- coverage gaps,
        # abrupt unit shifts, and per-channel shapes so the diagnostic can flag
        # data issues and note whether a normalized trend may be driven by
        # irradiance/temperature data rather than the array.
        try:
            raw_summary = _summarize_raw_data(
                df_f if 'df_f' in dir() else _df_from_store(df_filtered_json),
                mapped_variables_dict,
            )
        except Exception as _e:
            raw_summary = f"(raw-data summary unavailable: {_e})"

        result_dict = {
            "rate_pct_per_year": round(float(rate_pct) * 100, 4)
                if np.isfinite(rate_pct) else None,
            "method": "PVPRO",
            "duration_years": round(float(duration_years), 2),
            "start": start_str,
            "end":   end_str,
            "rates_per_quantity": rates_per_quantity,
            # Fields the AI diagnostic consumes.  PVPRO has no point-level
            # keep/drop filtering exposed here, so n_raw/n_kept/pct_kept are
            # reported as-fitted (not applicable -> 100% kept).
            "rate_pct": float(rate_pct) if np.isfinite(rate_pct) else 0.0,
            "n_raw": 0,
            "n_kept": 0,
            "pct_kept": 100.0,
            "trend_summary": trend_summary,
            "raw_summary": raw_summary,
        }
        # The same response paints the result, clears the browser-side job_id,
        # and disables polling.  Keep the backend job terminal/result intact
        # for its TTL so a lost response can be retried by the next poll.
        return [final_layout,
                False, "Calculate Degradation",
                result_dict, {}, True]
    finally:
        # Always release the lock, even if rendering raised.  If we
        # don't release, every future poll for this job_id will
        # silently no_update and the user will never see the result.
        render_lock.release()


# =============================================================================
# SHARED PVPRO RESULT RENDERER
#
# Builds the headline-rate block, the per-parameter rates table, and the
# figure grid from a finished PVPRO job's (rd, figs, rates).  Used by the
# Simple-mode PVPRO box below.  Returns the final layout Div only (no debug
# panel, no diagnostic dict) -- callers that need the diagnostic context can
# read `rates`/`rd` directly.
# =============================================================================
def _render_pvpro_layout_legacy(rd, figs, rates, start_str, end_str,
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
            [html.B(short, style={"fontFamily": "Archivo, system-ui, sans-serif"}),
             html.Span(" (ref)", style={
                 "color": INK_SOFT, "fontSize": "13px",
                 "fontFamily": "Archivo, system-ui, sans-serif",
             })],
            style={"padding": "4px 12px 4px 0", "whiteSpace": "nowrap"},
        )
        rates_rows.append(html.Tr([
            label_cell,
            html.Td(descr, style={"padding": "4px 12px 4px 0",
                                  "color": INK_SOFT, "fontSize": "13px",
                                  "fontFamily": "Archivo, system-ui, sans-serif"}),
            html.Td(r_str, style={"padding": "4px 0", "color": VALUE_DETAIL,
                                  "fontWeight": "700", "textAlign": "right",
                                  "fontFamily": "Archivo, system-ui, sans-serif"}),
        ]))
    rates_table = html.Details(
        [
            html.Summary(
                "parameter degradation rates",
                style={
                    "fontSize": "12px", "color": INK_SOFT,
                    "textTransform": "uppercase", "letterSpacing": "0.1em",
                    "fontWeight": "600", "fontFamily": "Archivo, system-ui, sans-serif",
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
            "fontFamily": "Archivo, system-ui, sans-serif",
        }),
        html.Div([
            html.Span(f"{rate_pct:.2%}", style={
                "fontSize": "56px", "fontFamily": "Archivo, system-ui, sans-serif",
                "fontWeight": "700", "color": rate_color, "lineHeight": "1",
            }),
            html.Span("/year", style={
                "fontSize": "20px", "color": INK_SOFT, "marginLeft": "8px",
                "fontFamily": "Archivo, system-ui, sans-serif", "fontStyle": "italic",
            }),
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Div([html.Span("Method: ", style={"color": INK_SOFT}),
                      html.B("PVPRO", style={"color": VALUE_DETAIL})],
                     style={"fontSize": "14px", "marginBottom": "3px"}),
            # Window + duration on a single line: window first, then duration.
            html.Div([html.Span("Window: ", style={"color": INK_SOFT}),
                      html.B(f"{start_str}  →  {end_str}",
                             style={"fontFamily": "Archivo, system-ui, sans-serif",
                                    "fontSize": "13px", "color": VALUE_DETAIL}),
                      html.Span(f"  ({duration_years:.1f} years)",
                                style={"color": INK_SOFT, "fontSize": "13px"})],
                     style={"fontSize": "14px"}),
        ], style={"fontFamily": "Archivo, system-ui, sans-serif"}),
        rates_table,
    ])

    card_style = {
        "background": "#ffffff",
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
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
        "fontFamily": "Archivo, system-ui, sans-serif",
    })

    return html.Div([
        html.Div(summary_block, style={"marginBottom": "20px"}),
        fig_grid_heading,
        fig_grid,
    ], className="slide-in-up", style={
        "padding": "20px",
        "background": "#f8fafc",
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
        "marginTop": "16px",
    })


def _render_pvpro_layout(rd, figs, rates, start_str, end_str,
                         duration_years, elapsed):
    """Render the Advanced PVPRO dashboard as a summary plus tabbed trend."""
    del elapsed  # retained in the shared call signature; not shown in results.
    figs = figs or {}
    rates = rates or {}
    rate_pct = rd / 100 if (rd is not None and np.isfinite(rd)) else 0.0
    quantities = [
        ("p_mp_ref", "Pmp"),
        ("v_mp_ref", "Vmp"),
        ("i_mp_ref", "Imp"),
        ("v_oc_ref", "Voc"),
        ("i_sc_ref", "Isc"),
    ]

    summary = html.Div(className="pvc-advanced-pvpro-summary", children=[
        html.Div("Annual degradation rate", className="pvc-advanced-result-kicker"),
        html.Div(className="pvc-advanced-pvpro-rate", children=[
            html.Span(f"{rate_pct:.2%}", className="pvc-advanced-pvpro-rate-value"),
            html.Span("/year", className="pvc-advanced-pvpro-rate-unit"),
        ]),
        html.Div(className="pvc-advanced-result-details", children=[
            html.Div([html.Span("Method: "), html.Strong("PVPRO"), _beta_badge()]),
            html.Div([html.Span("Duration: "), html.Strong(f"{duration_years:.1f} years")]),
            html.Div([html.Span("Window: "), html.Strong(f"{start_str} → {end_str}")]),
        ]),
    ])

    selectors = []
    charts = []
    for index, (key, short) in enumerate(quantities):
        active = index == 0
        selectors.append(html.Button(
            short,
            id={"type": "advanced-pvpro-result-param", "key": key},
            n_clicks=0,
            className="pvc-advanced-pvpro-param is-active" if active
                      else "pvc-advanced-pvpro-param",
        ))

        raw_fig = figs.get(key)
        if raw_fig is None:
            chart = html.Div(
                "Trend unavailable for this parameter.",
                className="pvc-advanced-pvpro-chart-empty",
            )
        else:
            figure = go.Figure(raw_fig)
            parameter_rate = rates.get(key, float("nan"))
            finite_rate = parameter_rate is not None and np.isfinite(parameter_rate)
            rate_label = "n/a" if not finite_rate else f"{parameter_rate:+.2f} %/yr"
            figure.update_layout(
                title=dict(
                    text=f"<b>{short}</b> (ref) &nbsp; ({rate_label})",
                    x=.5, xanchor="center", font=dict(size=16, color=INK),
                ),
                height=390,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Archivo, Arial, sans-serif", color=INK_SOFT, size=13),
                margin=dict(l=66, r=24, t=62, b=50),
                showlegend=False,
                hovermode="x unified",
            )
            figure.update_xaxes(
                showgrid=True, gridcolor="rgba(105,135,180,0.18)",
                zeroline=False, linecolor="rgba(105,135,180,0.16)",
            )
            figure.update_yaxes(
                showgrid=True, gridcolor="rgba(105,135,180,0.18)",
                zeroline=False, linecolor="rgba(105,135,180,0.16)",
            )
            chart = dcc.Graph(
                figure=figure,
                config={"displayModeBar": False, "displaylogo": False, "responsive": True},
                className="pvc-advanced-pvpro-graph",
            )

        charts.append(html.Div(
            chart,
            id={"type": "advanced-pvpro-result-chart", "key": key},
            className="pvc-advanced-pvpro-chart",
            style={} if active else {"display": "none"},
        ))

    trends = html.Div(className="pvc-advanced-pvpro-trends", children=[
        html.Div(selectors, className="pvc-advanced-pvpro-params"),
        html.Div(charts, className="pvc-advanced-pvpro-chart-stage"),
    ])
    return html.Div(
        [summary, trends],
        className="pvc-advanced-pvpro-result slide-in-up",
    )


@app.callback(
    Output({"type": "advanced-pvpro-result-param", "key": ALL}, "className"),
    Output({"type": "advanced-pvpro-result-chart", "key": ALL}, "style"),
    Input({"type": "advanced-pvpro-result-param", "key": ALL}, "n_clicks"),
    State({"type": "advanced-pvpro-result-param", "key": ALL}, "id"),
    State({"type": "advanced-pvpro-result-chart", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def switch_advanced_pvpro_result(_clicks, button_ids, chart_ids):
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict) or not ctx.triggered or not ctx.triggered[0].get("value"):
        return dash.no_update, dash.no_update
    selected = trigger.get("key")
    button_classes = [
        "pvc-advanced-pvpro-param is-active" if item.get("key") == selected
        else "pvc-advanced-pvpro-param"
        for item in (button_ids or [])
    ]
    chart_styles = [
        {} if item.get("key") == selected else {"display": "none"}
        for item in (chart_ids or [])
    ]
    return button_classes, chart_styles


def _render_simple_pvpro_layout(rd, figs, rates, start_str, end_str,
                                duration_years):
    """Render the compact Simple-mode-only PVPRO result dashboard."""
    figs = figs or {}
    rates = rates or {}
    rate_pct = rd / 100 if (rd is not None and np.isfinite(rd)) else 0.0
    quantities = [
        ("p_mp_ref", "Pmp", "Power at MPP"),
        ("v_mp_ref", "Vmp", "Voltage at MPP"),
        ("i_mp_ref", "Imp", "Current at MPP"),
        ("v_oc_ref", "Voc", "Open-circuit voltage"),
        ("i_sc_ref", "Isc", "Short-circuit current"),
    ]

    summary = html.Div(className="pvc-simple-pvpro-summary", children=[
        html.Div("Power degradation rate", className="pvc-simple-pvpro-kicker"),
        html.Div(className="pvc-simple-pvpro-rate", children=[
            html.Span(f"{rate_pct * 100:.2f}%", className="pvc-simple-pvpro-rate-value"),
            html.Span("/year", className="pvc-simple-pvpro-rate-unit"),
        ]),
        html.Div(className="pvc-simple-pvpro-details", children=[
            html.Div([html.Span("Method: "), html.Strong("PVPRO"), _beta_badge()]),
            html.Div([html.Span("Duration: "), html.Strong(f"{duration_years:.1f} years")]),
            html.Div([
                html.Span("Window: "),
                html.Strong(f"{start_str} → {end_str}", className="pvc-simple-pvpro-window"),
            ]),
        ]),
    ])

    selectors = []
    charts = []
    for index, (key, short, description) in enumerate(quantities):
        active = index == 0
        selectors.append(html.Button(
            [
                html.Span(short, className="pvc-simple-pvpro-param-name"),
                html.Span(f"({description})", className="pvc-simple-pvpro-param-description"),
            ],
            id={"type": "simple-pvpro-result-param", "key": key},
            n_clicks=0,
            className="pvc-simple-pvpro-param is-active" if active else "pvc-simple-pvpro-param",
        ))

        raw_fig = figs.get(key)
        if raw_fig is None:
            chart = html.Div("Trend unavailable for this parameter.", className="pvc-simple-pvpro-chart-empty")
        else:
            figure = go.Figure(raw_fig)
            parameter_rate = rates.get(key, float("nan"))
            rate_is_finite = parameter_rate is not None and np.isfinite(parameter_rate)
            rate_label = "n/a" if not rate_is_finite else f"{parameter_rate:+.2f} %/yr"
            figure.update_layout(
                title=dict(
                    text=f"<b>{short}</b> (ref) &nbsp; ({rate_label})",
                    x=0.5, xanchor="center", font=dict(size=16, color=INK),
                ),
                height=350,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Archivo, Arial, sans-serif", color=INK_SOFT, size=13),
                margin=dict(l=66, r=24, t=58, b=50),
                showlegend=False,
                hovermode="x unified",
            )
            figure.update_xaxes(
                showgrid=True, gridcolor="rgba(105,135,180,0.18)",
                zeroline=False, linecolor="rgba(105,135,180,0.16)",
            )
            figure.update_yaxes(
                showgrid=True, gridcolor="rgba(105,135,180,0.18)",
                zeroline=False, linecolor="rgba(105,135,180,0.16)",
            )
            chart = dcc.Graph(
                figure=figure,
                config={"displayModeBar": False, "displaylogo": False, "responsive": True},
                className="pvc-simple-pvpro-graph",
            )

        charts.append(html.Div(
            chart,
            id={"type": "simple-pvpro-result-chart", "key": key},
            className="pvc-simple-pvpro-chart",
            style={} if active else {"display": "none"},
        ))

    trends = html.Div(className="pvc-simple-pvpro-trends", children=[
        html.Div(selectors, className="pvc-simple-pvpro-params"),
        html.Div(charts, className="pvc-simple-pvpro-chart-stage"),
    ])

    return html.Div(
        [summary, trends],
        className="pvc-simple-pvpro-result slide-in-up",
    )


@app.callback(
    Output({"type": "simple-pvpro-result-param", "key": ALL}, "className"),
    Output({"type": "simple-pvpro-result-chart", "key": ALL}, "style"),
    Input({"type": "simple-pvpro-result-param", "key": ALL}, "n_clicks"),
    State({"type": "simple-pvpro-result-param", "key": ALL}, "id"),
    State({"type": "simple-pvpro-result-chart", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def switch_simple_pvpro_result(_clicks, button_ids, chart_ids):
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict) or not ctx.triggered or not ctx.triggered[0].get("value"):
        return dash.no_update, dash.no_update
    selected = trigger.get("key")
    button_classes = [
        "pvc-simple-pvpro-param is-active" if item.get("key") == selected
        else "pvc-simple-pvpro-param"
        for item in (button_ids or [])
    ]
    chart_styles = [
        {} if item.get("key") == selected else {"display": "none"}
        for item in (chart_ids or [])
    ]
    return button_classes, chart_styles


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
@app.callback(
    Output("simple-pvpro-params-wrap", "style"),
    Output("simple-pvpro-about-wrap",  "style"),
    Output("simple-pvpro-params-details", "open", allow_duplicate=True),
    Input("simple-method-radio", "value"),
    prevent_initial_call=True,
)
def simple_toggle_pvpro_params(method):
    if method == "PVPRO":
        # Keep the optional controls available but collapsed. Simple mode
        # estimates the array geometry automatically when Run is clicked.
        return ({"display": "block", "marginTop": "8px"},
                {"display": "block"}, False)
    return ({"display": "none"}, {"display": "none"}, False)


@app.callback(
    Output("simple-result", "children", allow_duplicate=True),
    Output("simple-status", "children", allow_duplicate=True),
    Output("simple-pvpro-progress-output", "children", allow_duplicate=True),
    Output("simple-stash", "data", allow_duplicate=True),
    Output("simple-step-progress", "data", allow_duplicate=True),
    Input("simple-method-radio", "value"),
    prevent_initial_call=True,
)
def clear_simple_result_for_method_switch(_method):
    """Never leave a result from the previously selected method on screen."""
    return "", "", "", {}, {
        "started": False, "data": False, "filter": False,
        "calc": False, "code": False,
    }


@app.callback(
    Output("simple-start-view", "style"),
    Output("simple-result-view", "style"),
    Input("simple-stash", "data"),
)
def toggle_simple_start_and_result(stash):
    result_ready = (stash or {}).get("method") in ("YOY", "PVPRO")
    if result_ready:
        return {"display": "none"}, {"display": "block"}
    return {}, {"display": "none"}


@app.callback(
    Output("ui-mode", "data", allow_duplicate=True),
    Output("simple-stash", "data", allow_duplicate=True),
    Output("simple-result", "children", allow_duplicate=True),
    Output("simple-status", "children", allow_duplicate=True),
    Output("simple-pvpro-progress-output", "children", allow_duplicate=True),
    Output("simple-step-progress", "data", allow_duplicate=True),
    Input("simple-result-return", "n_clicks"),
    Input("simple-result-advanced", "n_clicks"),
    prevent_initial_call=True,
)
def simple_result_actions(return_clicks, advanced_clicks):
    trigger = ctx.triggered_id
    if trigger not in ("simple-result-return", "simple-result-advanced"):
        return (dash.no_update,) * 6
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return (dash.no_update,) * 6

    next_mode = "advanced" if trigger == "simple-result-advanced" else dash.no_update
    empty_progress = {
        "started": False, "data": False, "filter": False,
        "calc": False, "code": False,
    }
    return next_mode, {}, "", "", "", empty_progress


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
    Output("simple-step-progress",         "data",     allow_duplicate=True),
    Output("simple-stash",                 "data",     allow_duplicate=True),
    Input("simple-pvpro-poll-interval", "n_intervals"),
    State("simple-pvpro-job", "data"),
    State("simple-pipe-filtered", "data"),
    State("simple-method-radio", "value"),
    prevent_initial_call=True,
)
def simple_pvpro_poll(_n, job_store, pfiltered, selected_method):
    from dash import no_update

    job_id = (job_store or {}).get("job_id")
    if not job_id:
        return [no_update] * 7

    job = _pvpro_read_job(job_id)
    if job is None:
        return [no_update] * 7

    phase = job.get("phase", "")
    elapsed = max(0.0, time.time() - job.get("started_at", time.time()))

    # The user may switch back to YOY while a PVPRO worker is still running.
    # Let it finish in the background, but never paint its progress/result into
    # a result area that currently belongs to YOY.
    if selected_method != "PVPRO":
        if phase in ("done", "error", "rendered"):
            if phase != "rendered":
                _pvpro_update_job(job_id, phase="rendered")
            return ["", no_update, no_update, {}, True, no_update, no_update]
        return ["", no_update, no_update, job_store, False, no_update, no_update]

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
        if job is None:
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

        layout_div = _render_simple_pvpro_layout(
            rd, figs, rates, start_str, end_str, duration_years)

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

        done_status = ""   # no success banner on completion (per request)
        progress = {"started": True, "data": True, "filter": True,
                    "calc": True, "code": False}
        # Do not acknowledge delivery in backend state.  Clearing job_id and
        # disabling the Interval in this response is the browser-side ack. If
        # the response is lost, the unchanged terminal job is safely rendered
        # again by the next poll instead of leaving Simple mode at 98%.
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


# Style-update clientside callback. Layout and typography live in the shared
# CSS class; this callback changes only the selected card's visual state.
app.clientside_callback(
    """
    function(selected) {
        var resting = {
            "border": "1px solid rgba(255,255,255,0.82)",
            "background": "rgba(255,255,255,0.46)",
            "boxShadow": "inset 0 1px 0 rgba(255,255,255,0.65)"
        };
        var active = Object.assign({}, resting, {
            "border": "2px solid #2f6bff",
            "background": "rgba(239,246,255,0.92)",
            "boxShadow": "0 0 0 3px rgba(47,107,255,0.12)"
        });

        return [
            selected === "load-example-btn-1" ? active  : resting,
            selected === "load-example-btn-2" ? active  : resting,
            selected === "load-example-btn-3" ? active  : resting
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
#   * analyze-btn  (Step 1): "Run prescreening" -> "Uploading data..." when a
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
            return [false, "Run prescreening"];
        }
        var trigger = ctx.triggered[0].prop_id.split('.')[0];
        if (trigger === "analyze-btn") {
            if (!analyze_n || analyze_n === 0) return [false, "Run prescreening"];
            return [true, "Analyzing..."];
        }
        if (trigger === "upload-data") {
            // Only treat a *non-empty* contents arrival as an upload.
            if (!upload_contents) return [false, "Run prescreening"];
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
            return [false, "Run prescreening"];
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
# OVERVIEW FIGURES HELPER
#
# Builds the "raw data preview" figure stack from the CURRENT mapping. Used by
# both the analyze callback and the Apply-mapping callback so that unselected
# variables are never plotted. make_overview_figures already skips any metric
# whose key is missing or not a real column, so passing a mapping without a
# given key guarantees that variable is not drawn.
# =============================================================================
def _build_overview_figures_div(df, mapped_variables_dict):
    try:
        if df is not None and mapped_variables_dict:
            figs, _err = make_overview_figures(df, mapped_variables_dict)
            # Step 1 presents raw signals as a roomy two-column gallery. The
            # underlying figure data is unchanged; only display sizing differs.
            for graph in figs:
                figure = getattr(graph, "figure", None)
                if hasattr(figure, "update_layout"):
                    figure.update_layout(height=225, margin=dict(l=50, r=18, t=38, b=32))
            return html.Div(figs, className="pvc-step1-fig-grid")
    except Exception:
        return html.Div("Figure generation failed.", style={"color": ACCENT})
    return html.Div(
        "No variables selected to plot.",
        style={"color": INK_SOFT, "fontSize": "13px",
               "fontFamily": "Archivo, system-ui, sans-serif"},
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
    "DC Power", "Time",                    # required — degradation can't run without these
    "Irradiance", "Module temperature",    # optional refinements (normalization / temp correction)
    "DC Voltage", "DC Current",            # optional — only used by the PVPRO physics method
]

# Metrics required for degradation analysis (used to flag missing selections).
_REQUIRED_FOR_DEGRADATION = {"DC Power", "Time"}


# -----------------------------------------------------------------------------
# Inline notices shown UNDER each mapping row: quiet grays, hairline borders;
# color is reserved for a single small status dot on the missing-variable
# notices (red = blocks degradation, amber = caveat only).
# -----------------------------------------------------------------------------
_HINT_FONT = ("-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', "
              "Roboto, Helvetica, Archivo, system-ui, sans-serif")
_HINT_INK = "#1d1d1f"          # primary text
_HINT_INK_SOFT = "#86868b"     # secondary text
_HINT_HAIRLINE = "#d2d2d7"     # pill border
_HINT_FILL = "#ffffff"         # pill fill


def _col_chip(name, tag=None):
    """A column name as a quiet, hairline pill, with an optional quality tag."""
    txt = f"{name} ({tag})" if tag else name
    return html.Span(txt, style={
        "fontFamily": "SFMono-Regular, ui-monospace, Menlo, monospace",
        "fontSize": "11px", "color": _HINT_INK,
        "background": _HINT_FILL,
        "border": f"1px solid {_HINT_HAIRLINE}",
        "borderRadius": "980px", "padding": "2.5px 10px",
        "whiteSpace": "nowrap", "display": "inline-block",
    })


def _alt_hint(others, exclude=None, tags=None):
    """A quiet one-line notice listing the OTHER candidate columns for a role
    (excluding whatever is currently selected). Plain column names only — the
    quality tags live on the dropdown options (as right-aligned pills), so we
    don't repeat them here."""
    items = [c for c in (others or []) if c and c != exclude]
    if not items:
        return ""
    return html.Div(
        [html.Span("Also detected", style={
            "fontSize": "11px", "color": _HINT_INK_SOFT,
            "fontWeight": "500", "flex": "0 0 auto",
        })] + [_col_chip(c) for c in items],
        title="Other columns that could match this variable — "
              "switch above if the selected one isn't right.",
        className="var-alt-hint",
        style={
            "marginTop": "7px", "display": "flex", "flexWrap": "wrap",
            "alignItems": "center", "gap": "6px",
            "fontFamily": _HINT_FONT, "lineHeight": "1.4",
        },
    )


# What a MISSING variable means, shown inline under its row. (message, required):
# required variables (Power, Time) genuinely block the degradation analysis;
# the rest are OPTIONAL — skipping them only disables an extra method or a
# refinement, so the copy stays neutral and says the main analysis still runs.
# The "Required ·" / "Optional ·" lead is added in _missing_hint, so messages
# begin lowercase.
_MISSING_HINT = {
    "DC Power": ("select a power column, or voltage + current (derived as V × I).",
                 True),
    "Time": ("select the timestamp column.", True),
    "Irradiance": ("enables weather normalization; the rate is less reliable "
                   "without it.", False),
    "Module temperature": ("enables temperature correction.", False),
    "DC Voltage": ("used only by the PVPRO physics method.", False),
    "DC Current": ("used only by the PVPRO physics method.", False),
}


def _missing_hint(metric):
    """Inline note under a row with no column selected. Two tiers:
    - required (Power, Time): red dot, prominent — it blocks the analysis;
    - optional (the rest): gray dot, smaller and lighter — it only disables an
      extra method/refinement, so it shouldn't read like an error."""
    entry = _MISSING_HINT.get(metric)
    if not entry:
        return ""
    msg, required = entry
    if required:
        dot = "#dc2626"                              # red — genuine blocker
        lead, lead_style = "Required", {"color": _HINT_INK, "fontWeight": "600"}
        msg_style = {"color": _HINT_INK_SOFT}
        text_style = {"fontSize": "12px", "lineHeight": "1.45"}
        row_style = {"marginTop": "7px", "display": "flex", "gap": "8px",
                     "alignItems": "flex-start", "fontFamily": _HINT_FONT}
        dot_mt = "6px"
    else:
        dot = "#c7c7cc"                              # light gray — recedes
        lead, lead_style = "Optional", {"color": _HINT_INK_SOFT, "fontWeight": "600"}
        msg_style = {"color": "#a1a1a6"}
        text_style = {"fontSize": "11px", "lineHeight": "1.4"}
        row_style = {"marginTop": "6px", "display": "flex", "gap": "7px",
                     "alignItems": "flex-start", "fontFamily": _HINT_FONT}
        dot_mt = "5px"
    return html.Div([
        html.Span(style={
            "width": "6px", "height": "6px", "borderRadius": "50%",
            "background": dot, "flex": "0 0 auto", "marginTop": dot_mt,
        }),
        html.Span([
            html.Span(f"{lead} · ", style=lead_style),
            html.Span(msg, style=msg_style),
        ], style=text_style),
    ],
        className=f"var-missing-hint {'blocking' if required else 'optional'}",
        style=row_style)


def build_variable_mapping_table(mapped_variables_dict, columns,
                                 time_in_index=False, alternatives=None,
                                 status_children=None, detected_map=None,
                                 quality_tags=None):
    """Build an editable variable-mapping table.

    Args:
        mapped_variables_dict: {metric: column_name} currently mapped (N/A omitted).
        columns: list of available DataFrame column names.
        time_in_index: whether the Time variable is the DataFrame index.
        status_children: optional element rendered in the status slot at the
            bottom (used by the Apply callback to show its confirmation/warning).
        alternatives: optional {metric: [other valid column names]} — when a role
            had more than one valid match, the others are pinned at the top of
            that row's dropdown and listed as a subtle hint under it. Populated
            from df.attrs["mapping_alternatives"] when parse_contents provides
            it; safely defaults to none.
        detected_map: optional {metric: column} of the LLM's ORIGINAL detection.
            Kept separate from the current mapping so the "LLM-detected" group
            stays populated even after the user clears a row (current would then
            be empty, but the LLM's pick still belongs in that group).

    Returns:
        A Dash component (the editable table + apply button + status line).
    """
    mapped_variables_dict = mapped_variables_dict or {}
    columns = list(columns or [])
    alternatives = alternatives or {}
    detected_map = detected_map or {}
    quality_tags = quality_tags or {}

    header = html.Div([
        html.Div("Metric", style={
            "flex": "0 0 30%", "fontWeight": "700", "color": INK,
            "fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "14px",
        }),
        html.Div("Variable Name", style={
            "flex": "1", "fontWeight": "700", "color": INK,
            "fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "14px",
        }),
    ], className="pvc-var-map-header", style={
        "display": "flex", "alignItems": "center", "gap": "14px",
        "padding": "8px 4px 12px 4px",
        "borderBottom": f"2px solid {BORDER_STRONG}",
    })

    rows = []
    # Short role noun for the "LLM-detected {role}" group label.
    _ROLE_GROUP = {
        "DC Power": "power", "DC Voltage": "voltage", "DC Current": "current",
        "Irradiance": "irradiance", "Module temperature": "temperature",
        "Time": "time",
    }

    for i, metric in enumerate(_MAP_METRICS):
        current = mapped_variables_dict.get(metric)
        detected_col = detected_map.get(metric)

        # The "LLM-detected {role}" group = the LLM's original pick + any valid
        # alternatives + the current selection (if the user changed it). Built
        # from detected_map (not just current) so clearing a row doesn't empty
        # this group — the LLM's detection still belongs here.
        valid = []
        if metric == "Time" and (
                time_in_index or current == "__index__" or detected_col == "__index__"):
            valid.append("__index__")
        for c in (detected_col, current):
            if c and c != "__index__" and c not in valid:
                valid.append(c)
        for c in alternatives.get(metric, []):
            if c and c not in valid:
                valid.append(c)

        rest = [c for c in columns if c not in valid]

        def _item(c):
            if c == "__index__":
                return {"value": c, "label": "(use index / __index__)",
                        "quality": ""}
            return {"value": c, "label": c, "quality": quality_tags.get(c) or ""}

        data = []
        if valid:
            data.append({"group": "LLM-detected " + _ROLE_GROUP.get(metric, metric),
                         "items": [_item(c) for c in valid]})
        if rest:
            data.append({"group": "Other columns",
                         "items": [_item(c) for c in rest]})

        detected = bool(current)
        _required = metric in _REQUIRED_FOR_DEGRADATION
        if detected:
            dot_color, dot_title = "#16a34a", "Selected"        # green
        elif _required:
            dot_color, dot_title = "#dc2626", "Required — please select"   # red
        else:
            dot_color, dot_title = "#a1a1aa", "Not selected"    # gray

        row = html.Div([
            html.Div([
                html.Span(
                    id={"type": "var-map-dot", "metric": metric},
                    style={
                        "display": "inline-block", "width": "8px", "height": "8px",
                        "borderRadius": "50%", "background": dot_color,
                        "marginRight": "8px", "flex": "0 0 auto",
                    }, title=dot_title),
                html.Span(metric, style={
                    "color": INK, "fontFamily": "Archivo, system-ui, sans-serif",
                    "fontSize": "14px",
                }),
            ], className="pvc-var-map-label", style={"display": "flex",
                      "alignItems": "center"}),
            html.Div([
                dmc.Select(
                    id={"type": "var-map-dd", "metric": metric},
                    data=data,
                    value=current if current else None,
                    placeholder="— select column —",
                    clearable=True,
                    searchable=True,
                    nothingFoundMessage="No matching column",
                    w="100%",
                    size="sm",
                    # Bold the selected column name in the input (only when a
                    # value is chosen, so the placeholder stays regular weight).
                    styles={"input": {"fontWeight": 700 if current else 400}},
                    # Custom option renderer: column name on the left, a small
                    # quality pill pushed to the right (JS fn in assets/, reads
                    # each option's "quality" field). Falls back gracefully to a
                    # plain name if the assets file isn't present.
                    renderOption={"function": "renderVarMapOption"},
                    # Scoped hook for the group ("category") titles. The dropdown
                    # renders in a portal at <body>, so .pvcopilot-root CSS can't
                    # reach it and a bare .mantine-Select-groupLabel rule would
                    # leak into every other dmc.Select in pvtools. This unique
                    # class keeps the blue title change local to these dropdowns.
                    classNames={"groupLabel": "pvcopilot-var-group"},
                    comboboxProps={
                        "withinPortal": True,
                        "zIndex": 3000,
                    },
                ),
                # Missing-variable note — always in the DOM but shown only while
                # this row has no column selected. A clientside callback toggles
                # it live, so clearing a selection surfaces the warning at once
                # (no need to click Apply first).
                html.Div(_missing_hint(metric),
                         id={"type": "var-map-miss", "metric": metric},
                         style={"display": "block" if not detected else "none"}),
                # "Also detected" alternatives — static, relevant once selected.
                _alt_hint(alternatives.get(metric), exclude=current,
                          tags=quality_tags) if detected else "",
            ], className="pvc-var-map-control"),
        ], className="pvc-var-map-card", style={
            "padding": "8px 4px",
        })
        rows.append(row)

    return html.Div([
        header,
        html.Div(rows, className="pvc-var-map-grid", style={"marginTop": "4px"}),
        html.Div([
            html.Button(
                "Apply mapping",
                id="var-map-apply-btn",
                n_clicks=0,
                style={
                    "background": ACCENT, "color": "#ffffff", "border": "none",
                    "padding": "8px 18px", "borderRadius": "12px",
                    "fontSize": "13px", "fontWeight": "600", "cursor": "pointer",
                    "fontFamily": "Archivo, system-ui, sans-serif",
                },
            ),
            html.Span(
                "Click Apply to confirm your changes.",
                style={"marginLeft": "12px", "fontSize": "12px",
                       "color": INK_SOFT, "fontFamily": "Archivo, system-ui, sans-serif"},
            ),
        ], id="var-map-apply-row",
           # Hidden until a dropdown differs from the applied/detected state; a
           # clientside callback (below) toggles this on any selection change.
           style={"marginTop": "14px", "display": "none", "alignItems": "center"}),
        # Baseline the "show Apply on change" comparison reads against. Rebuilt
        # with the table, so after Apply the baseline resets and the button
        # hides again until the next change.
        dcc.Store(id="var-map-initial",
                  data={m: (mapped_variables_dict.get(m) or None)
                        for m in _MAP_METRICS}),
        # Apply confirmation/warning renders here, under the button.
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
def apply_variable_mapping_callback(n_clicks, values, ids, df_json, columns_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # data-columns-store now carries both the full column list AND the LLM's
    # alternatives, because df.attrs (where parse_contents stashed them) does
    # NOT survive the DataFrame's round-trip through dcc.Store as JSON. Reading
    # them here keeps the "LLM-detected" grouping intact after Apply. (Legacy
    # list payloads — e.g. from an error path — degrade gracefully.)
    if isinstance(columns_store, dict):
        data_columns = columns_store.get("columns", []) or []
        alternatives = columns_store.get("alternatives", {}) or {}
        detected_map = columns_store.get("detected", {}) or {}
        quality_tags = columns_store.get("quality_tags", {}) or {}
    else:
        data_columns = columns_store or []
        alternatives = {}
        detected_map = {}
        quality_tags = {}

    # Rebuild the mapping dict, keeping ONLY non-empty real selections
    # (never a '__hdr__' section-label pseudo-value).
    new_mapping = {}
    for val, id_obj in zip(values, ids):
        metric = id_obj.get("metric")
        if val and not str(val).startswith("__hdr__"):
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

    # Redraw figures from the new mapping — unselected variables are not drawn.
    try:
        df = _df_from_store(df_json) if df_json else None
    except Exception:
        df = None

    # Rebuild the mapping panel so the dots match the applied state (and the
    # baseline store resets, re-hiding the Apply button until the next change).
    time_in_index = new_mapping.get("Time") == "__index__"
    panel = build_variable_mapping_table(
        new_mapping, data_columns,
        time_in_index=time_in_index, alternatives=alternatives,
        status_children=status, detected_map=detected_map,
        quality_tags=quality_tags,
    )

    figures = _build_overview_figures_div(df, new_mapping)

    return new_mapping, panel, figures


# Show the "Apply mapping" row only when at least one dropdown differs from the
# baseline (the detected/last-applied mapping). Pure clientside — compares the
# current pattern-matched dropdown values against the var-map-initial store.
app.clientside_callback(
    """
    function(values, ids, initial) {
        initial = initial || {};
        var changed = false;
        for (var i = 0; i < (ids || []).length; i++) {
            var m = ids[i].metric;
            var cur = (values[i] === undefined) ? null : values[i];
            var init = (initial[m] === undefined) ? null : initial[m];
            if (cur !== init) { changed = true; break; }
        }
        var base = {"marginTop": "14px", "alignItems": "center"};
        base["display"] = changed ? "flex" : "none";
        return base;
    }
    """,
    Output("var-map-apply-row", "style"),
    Input({"type": "var-map-dd", "metric": ALL}, "value"),
    State({"type": "var-map-dd", "metric": ALL}, "id"),
    State("var-map-initial", "data"),
)


# Live-recolor each row's status dot as the user edits, before Apply: green when
# a column is selected, red for the required-but-empty ones (DC Power, Time),
# gray for the optional-but-empty ones. Input and output are the same metric
# set, so Dash lists them in the same order — values[i] pairs with dot[i].
app.clientside_callback(
    """
    function(values, ids) {
        var required = {"DC Power": 1, "Time": 1};
        function real(v){ return (v && String(v).indexOf("__hdr__") !== 0) ? v : null; }
        return (values || []).map(function(v, i) {
            var m = ids[i].metric;
            var color = real(v) ? "#16a34a" : (required[m] ? "#dc2626" : "#a1a1aa");
            return {"display": "inline-block", "width": "8px", "height": "8px",
                    "borderRadius": "50%", "background": color,
                    "marginRight": "8px", "flex": "0 0 auto"};
        });
    }
    """,
    Output({"type": "var-map-dot", "metric": ALL}, "style"),
    Input({"type": "var-map-dd", "metric": ALL}, "value"),
    State({"type": "var-map-dd", "metric": ALL}, "id"),
)


# Show/hide each row's missing-variable note as the user edits, before Apply:
# visible whenever the row has no column selected, hidden once one is. Makes the
# warning appear immediately when a selection is cleared.
app.clientside_callback(
    """
    function(values) {
        return (values || []).map(function(v) {
            var real = (v && String(v).indexOf("__hdr__") !== 0);
            return real ? {"display": "none"} : {"display": "block"};
        });
    }
    """,
    Output({"type": "var-map-miss", "metric": ALL}, "style"),
    Input({"type": "var-map-dd", "metric": ALL}, "value"),
)


# =============================================================================
# LIVE ANALYZE STATUS LINE
#
# (1) mint a per-tab token on load; (2) on Analyze click, clientside swaps the
# caption for the status line and starts the poll — no server round-trip;
# (3) every 450 ms the poll mirrors the stage message the analyze callback
# published (plus an elapsed counter) into stable spans; (4) when the analyze
# callback writes data-summary-output (any outcome), a clientside callback hides
# the line, restores the caption, and stops the poll.
# =============================================================================

# (1) Mint the per-tab token on page load (n_clicks=0 as the layout mounts).
app.clientside_callback(
    """
    function(_n, existing) {
        if (existing) { return window.dash_clientside.no_update; }
        return "t" + Date.now().toString(36) +
               Math.floor(Math.random() * 1e9).toString(36);
    }
    """,
    Output("analyze-status-token", "data"),
    Input("analyze-btn", "n_clicks"),
    State("analyze-status-token", "data"),
)

# (2) Analyze clicked -> show the status line, start polling.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            var nu = window.dash_clientside.no_update;
            return [nu, nu, nu];
        }
        return [
            {"display": "flex", "alignItems": "center",
             "justifyContent": "flex-start", "gap": "7px", "marginTop": "8px",
             "fontFamily": "Archivo, system-ui, sans-serif", "fontSize": "13px"},
            {"display": "none"},
            false
        ];
    }
    """,
    Output("analyze-status-line",     "style"),
    Output("analyze-caption",         "style"),
    Output("analyze-status-interval", "disabled"),
    Input("analyze-btn", "n_clicks"),
    prevent_initial_call=True,
)

# (4) Analysis finished (any outcome writes data-summary-output) -> hide the
# status line, restore the caption, stop polling, reset for the next run.
app.clientside_callback(
    """
    function(_children) {
        return [
            {"display": "none"},
            {"display": "none"},
            true,
            "Starting analysis…",
            ""
        ];
    }
    """,
    Output("analyze-status-line",     "style",    allow_duplicate=True),
    Output("analyze-caption",         "style",    allow_duplicate=True),
    Output("analyze-status-interval", "disabled", allow_duplicate=True),
    Output("analyze-status-text",     "children", allow_duplicate=True),
    Output("analyze-status-elapsed",  "children", allow_duplicate=True),
    Input("data-summary-output", "children"),
    prevent_initial_call=True,
)

# (3) Mirror the published stage message into the status line.
@app.callback(
    Output("analyze-status-text",    "children"),
    Output("analyze-status-elapsed", "children"),
    Input("analyze-status-interval", "n_intervals"),
    State("analyze-status-token",    "data"),
    State("analyze-btn",             "n_clicks"),
    prevent_initial_call=True,
)
def poll_analyze_status(_n, token, n_clicks):
    rec = _analyze_status_get(_analyze_status_key(token, n_clicks))
    if not rec or not rec.get("message"):
        # First tick can beat the callback's first publish — keep the placeholder.
        return dash.no_update, dash.no_update
    elapsed = int(time.time() - rec.get("started_at", time.time()))
    return rec["message"], (f"{elapsed}s" if elapsed >= 1 else "")


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
    State("analyze-status-token", "data"),
    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        analyze_clicks, example_clicks_1, example_clicks_2, example_clicks_3,
        contents, filename, stored_df_json, data_source, stored_file_name,
        status_token):

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
            output_msg = _success_banner(f"{example_filename} loaded")
        except Exception as e:
            return (html.Div(f"Error loading example: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Run prescreening", None, "", example_filename, [])
        return (html.Div("", className="text-muted"),
                {}, df_json, "", False, "Run prescreening", "example", output_msg, example_filename, [])

    # Analyze clicked
    if trigger == "analyze-btn":
        # Live status: publish stage messages under this run's key; the poll
        # callback mirrors them into the UI while this callback is still running.
        _status_key = _analyze_status_key(status_token, analyze_clicks)

        def _status(msg):
            _analyze_status_set(_status_key, msg)

        _status("Reading your data…")
        # Parsing (which includes an LLM column-identification call) is capped
        # so a malformed file can't hang the UI indefinitely. parse_contents
        # also calls progress() at its own sub-steps.
        try:
            if data_source == "upload" and contents is not None:
                df, summary_table, mapped_variables_dict, code_read, mapping_notes = _run_with_timeout(
                    parse_contents, contents, filename, progress=_status, timeout=ANALYZE_TIMEOUT_S)
                if df is None:
                    return summary_table, {}, None, "", False, "Run prescreening", None, "", stored_file_name, []
            elif data_source == "example" and stored_df_json is not None:
                df = _df_from_store(stored_df_json)
                df, summary_table, mapped_variables_dict, code_read, mapping_notes = _run_with_timeout(
                    parse_contents, df=df, progress=_status, timeout=ANALYZE_TIMEOUT_S)
            else:
                return (_no_data_alert("Please upload a file or click an example button, then click 'Analyze Data'."),
                        {}, None, "", False, "Run prescreening", None, "", filename, [])
        except FutureTimeout:
            return (_no_data_alert("This is taking longer than expected — something may be wrong with your "
                                   "data. Check the file's formatting and columns, then try again."),
                    {}, None, "", False, "Run prescreening", None, "", stored_file_name, [])
        except Exception as e:
            return (html.Div(f"Error processing dataset: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Run prescreening", None, "", stored_file_name, [])

        _status("Packaging your dataset…")
        try:
            df_json = df.to_json(date_format="iso", orient="split")
        except Exception as e:
            return (html.Div(f"Error converting DataFrame: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Run prescreening", None, "", stored_file_name, [])

        # Available columns for the editable mapping dropdowns, and whether
        # the Time variable lives in the index.
        data_columns = [str(c) for c in df.columns.tolist()]
        time_in_index = (
            isinstance(df.index, pd.DatetimeIndex)
            or mapped_variables_dict.get("Time") == "__index__"
        )

        # Only plot variables that are actually mapped to a real column.
        _status("Rendering data preview…")
        figures_output = _build_overview_figures_div(df, mapped_variables_dict)
        _status("Finishing up…")

        # Editable variable-mapping table (defaults to LLM detection; user can
        # override any row, or fill in rows the LLM missed). When a role had
        # several valid matches, the others are pinned in that row's dropdown
        # and shown inline — parse_contents stashes them on df.attrs if it
        # supports it; otherwise this is simply empty.
        alternatives = df.attrs.get("mapping_alternatives", {}) if df is not None else {}
        quality_tags = df.attrs.get("mapping_quality_tags", {}) if df is not None else {}
        editable_mapping = build_variable_mapping_table(
            mapped_variables_dict, data_columns, time_in_index=time_in_index,
            alternatives=alternatives, detected_map=mapped_variables_dict,
            quality_tags=quality_tags,
        )

        # Transformation caveats from parse_contents (AC fallback, DC Power
        # computed as V×I, gappy columns, time-from-values). Shown once, above
        # the mapping table; empty list -> nothing rendered.
        notes_block = None
        if mapping_notes:
            notes_block = html.Div(
                [html.Div("data notes", style={
                    "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
                    "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "8px",
                    "fontFamily": "Archivo, system-ui, sans-serif",
                })] + [
                    html.Div("• " + n, style={
                        "fontSize": "13px", "color": INK, "lineHeight": "1.5",
                        "fontFamily": "Archivo, system-ui, sans-serif", "marginBottom": "3px",
                    }) for n in mapping_notes
                ],
                style={
                    "padding": "12px 16px", "marginTop": "14px",
                    "background": "#fffbeb", "border": "1px solid #fcd34d",
                    "borderRadius": "16px",
                },
            )

        combined_output = html.Div([
            html.Div([
                html.Div([
                    html.Span("identified variables", style={
                        "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
                        "letterSpacing": "0.1em", "fontWeight": "600",
                        "fontFamily": "Archivo, system-ui, sans-serif",
                    }),
                    html.Span("(LLM-detected)", style={
                        "fontSize": "12px", "color": INK_SOFT, "fontStyle": "italic",
                        "marginLeft": "8px", "fontWeight": "400",
                        "fontFamily": "Archivo, system-ui, sans-serif",
                    }),
                ], style={"marginBottom": "10px"}),
                # Stable container so the Apply callback can re-render the
                # mapping table (refreshing the detected/undetected dots).
                html.Div(editable_mapping, id="var-map-panel",
                         style={"fontSize": "14px"}),
            ], className="pvc-step1-mapping-surface", style={
                "padding": "18px 20px",
                "background": "#f8fafc",
                "border": f"1px solid {BORDER}",
                "borderRadius": "16px",
                "marginBottom": "16px",
            }),
            html.Div([
                html.Div("raw data preview", style={
                    "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
                    "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
                    "fontFamily": "Archivo, system-ui, sans-serif",
                }),
                # Stable container so the Apply callback can redraw figures,
                # dropping any variable the user de-selected.
                html.Div(figures_output, id="var-map-figures"),
                # Data-transformation caveats sit under the figure, inside the
                # preview block (empty list -> nothing rendered).
                notes_block if notes_block is not None else "",
            ], style={
                "padding": "18px 20px",
                "background": "#f8fafc",
                "border": f"1px solid {BORDER}",
                "borderRadius": "16px",
            }),
        ], className="slide-in-up")

        return (combined_output, mapped_variables_dict, df_json, code_read, False,
                "Run prescreening", None, "", stored_file_name,
                {"columns": data_columns, "alternatives": alternatives,
                 "detected": mapped_variables_dict, "quality_tags": quality_tags})

    return ("", {}, None, "", False, "Run prescreening", None, "", stored_file_name, [])


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
    # --- Advanced param values (existing) ---
    Output("param-pvpro-cells",    "value", allow_duplicate=True),
    Output("param-pvpro-mps",      "value", allow_duplicate=True),
    Output("param-pvpro-ps",       "value", allow_duplicate=True),
    Output("param-pvpro-alphaisc", "value", allow_duplicate=True),
    Output("param-pvpro-tech",     "value", allow_duplicate=True),
    Output("param-pvpro-days",     "value", allow_duplicate=True),
    Output("param-pvpro-iters",    "value", allow_duplicate=True),
    Output("pvpro-params-details", "open",  allow_duplicate=True),
    # --- Advanced: clear the auto-fill highlight / dots / note ---
    Output("param-pvpro-mps",      "style", allow_duplicate=True),
    Output("param-pvpro-ps",       "style", allow_duplicate=True),
    Output("param-pvpro-mps-dot",  "style", allow_duplicate=True),
    Output("param-pvpro-ps-dot",   "style", allow_duplicate=True),
    Output("pvpro-autofill-note",  "children", allow_duplicate=True),
    # --- Simple: method reverts to the YoY default, params to defaults ---
    Output("simple-method-radio",         "value", allow_duplicate=True),
    Output("simple-param-pvpro-cells",    "value", allow_duplicate=True),
    Output("simple-param-pvpro-mps",      "value", allow_duplicate=True),
    Output("simple-param-pvpro-ps",       "value", allow_duplicate=True),
    Output("simple-param-pvpro-alphaisc", "value", allow_duplicate=True),
    Output("simple-param-pvpro-tech",     "value", allow_duplicate=True),
    Output("simple-param-pvpro-days",     "value", allow_duplicate=True),
    Output("simple-param-pvpro-iters",    "value", allow_duplicate=True),
    # --- Simple: clear the auto-fill highlight / dots / note ---
    Output("simple-param-pvpro-mps",      "style", allow_duplicate=True),
    Output("simple-param-pvpro-ps",       "style", allow_duplicate=True),
    Output("simple-param-pvpro-mps-dot",  "style", allow_duplicate=True),
    Output("simple-param-pvpro-ps-dot",   "style", allow_duplicate=True),
    Output("simple-pvpro-autofill-note",  "children", allow_duplicate=True),
    # Advanced Step-3 metric selection reverts to the YoY default; the
    # clientside mirror then clears the PVPRO radio and updates the hidden
    # master. (The short-data gate may afterwards move it off YoY if needed.)
    Output("metric-stat-radio", "value", allow_duplicate=True),
    Input("upload-data",         "filename"),
    Input("load-example-btn-1",  "n_clicks"),
    Input("load-example-btn-2",  "n_clicks"),
    Input("load-example-btn-3",  "n_clicks"),
    prevent_initial_call=True,
)
def reset_pvpro_params_on_new_data(*_):
    # A fresh dataset invalidates any prior estimate. Reset BOTH modes to the
    # fresh-page state: Simple mode's method reverts to the YoY default, all
    # PVPRO fields go back to their defaults, and every auto-fill highlight /
    # blue dot / "pre-filled" note is cleared. Defaults MUST stay in sync with
    # the layout's dcc.Input(value=...) / RadioItems(value=...) declarations.
    base = dict(_PVPRO_MID_BASE)
    dot_off = dict(_PVPRO_DOT_OFF)
    return (
        # advanced values + close advanced disclosure
        60, 1, 1, 0.0046, "mono-c-Si", 14, 12, False,
        # advanced highlight/dot/note cleared
        base, base, dot_off, dot_off, "",
        # simple method (YoY default) + values
        "YOY", 60, 1, 1, 0.0046, "mono-c-Si", 14, 12,
        # simple highlight/dot/note cleared
        base, base, dot_off, dot_off, "",
        # advanced Step-3 metric selection back to the YoY default
        ["YOY"],
    )


# =============================================================================
# CALLBACK — AUTO-FILL PVPRO ARRAY PARAMS FROM THE DATA
#
# After Analyze (mapped-vars-store updates), estimate what the data implies
# about the array layout -- modules per string from the median DC operating
# voltage, parallel strings from the median DC current (or P/V) -- and pre-fill
# those fields in BOTH Simple and Advanced. Auto-filled inputs get a blue
# highlight and a note says exactly what was filled and from what, so the user
# knows to review rather than assume they typed it. Fields that can't be
# estimated honestly are left untouched at their defaults.
# =============================================================================
def _pvpro_autofill_note(filled_bits):
    """Small blue-dotted 'pre-filled from your data' line under the param grid."""
    return html.Div([
        html.Span(style={
            "display": "inline-block", "width": "7px", "height": "7px",
            "borderRadius": "50%", "background": "#3b82f6",
            "marginRight": "8px", "flex": "0 0 auto", "marginTop": "5px"}),
        html.Span([
            html.Span("Estimated from your data \u00b7 ", style={
                "fontWeight": "600", "color": INK}),
            html.Span("; ".join(filled_bits) + ". Estimates assume a typical "
                      "crystalline module \u2014 adjust if you know the real layout.",
                      style={"color": INK_SOFT}),
        ], style={"fontSize": "12px", "lineHeight": "1.5"}),
    ], className="pvcopilot-note-float-in",
       style={"display": "flex", "alignItems": "flex-start", "marginTop": "4px",
              "fontFamily": _HINT_FONT})


# NOTE: the automatic auto-fill callback (formerly `autofill_pvpro_params`,
# triggered on every mapped-vars-store change) has been REMOVED. PVPRO
# parameter estimation is now MANUAL only, via the "Estimate from data"
# buttons in each mode (estimate_pvpro_simple / estimate_pvpro_advanced).


# =============================================================================
# CALLBACK — Simple-mode "Estimate from data" button.
#
# On click: IDENTIFY the DC voltage/current/power columns (reuse the mapping
# already in the store if present, else run the same parse_contents the full
# Analyze uses), THEN ESTIMATE Modules per string + Parallel strings via
# estimate_pvpro_params and fill the Simple-mode fields — the same result as
# the Advanced-mode auto-fill, but on demand and before running the analysis.
# =============================================================================
@app.callback(
    Output("simple-param-pvpro-mps", "value", allow_duplicate=True),
    Output("simple-param-pvpro-ps",  "value", allow_duplicate=True),
    Output("simple-param-pvpro-mps", "style", allow_duplicate=True),
    Output("simple-param-pvpro-ps",  "style", allow_duplicate=True),
    Output("simple-param-pvpro-mps-dot", "style", allow_duplicate=True),
    Output("simple-param-pvpro-ps-dot",  "style", allow_duplicate=True),
    Output("simple-pvpro-autofill-note", "children", allow_duplicate=True),
    Input("simple-pvpro-estimate-btn", "n_clicks"),
    State("dataframe-store",          "data"),
    State("mapped-vars-store",        "data"),
    State("simple-param-pvpro-cells", "value"),
    prevent_initial_call=True,
)
def estimate_pvpro_simple(n, df_json, mapping, cells):
    nu = dash.no_update

    def _msg(text):
        return html.Div(text, className="pvcopilot-note-float-in",
                        style={"fontSize": "12px", "color": INK_SOFT,
                               "marginTop": "4px", "lineHeight": "1.5",
                               "fontFamily": _HINT_FONT})

    if not n or not df_json:
        return nu, nu, nu, nu, nu, nu, _msg(
            "Load a dataset first, then click \u201cEstimate from data.\u201d")
    try:
        df = _df_from_store(df_json)
        # Identify variables: reuse an existing mapping if we already have one,
        # otherwise run the same parser Analyze uses.
        if not mapping:
            res = _run_with_timeout(parse_contents, df=df, timeout=ANALYZE_TIMEOUT_S)
            mapping = res[2] if res else {}
        est = estimate_pvpro_params(df, mapping or {},
                                    cells_in_series=_pvnum(cells, 60, int))
    except Exception:
        est = {}
    if not est:
        return nu, nu, nu, nu, nu, nu, _msg(
            "Couldn't estimate from this data \u2014 PVPRO estimation needs DC "
            "voltage + current (or DC power) columns that agree with each other.")

    base = dict(_PVPRO_MID_BASE)
    dot_off = dict(_PVPRO_DOT_OFF)
    dot_on = dict(_PVPRO_DOT_ON)
    mps = est.get("modules_per_string")
    ps = est.get("parallel_strings")
    filled_bits = []
    if mps:
        filled_bits.append(f"Modules per string = {mps['value']} ({mps['basis']})")
    if ps:
        filled_bits.append(f"Parallel strings = {ps['value']} ({ps['basis']})")
    return (mps["value"] if mps else nu,
            ps["value"] if ps else nu,
            dict(_PVPRO_MID_AUTOFILL) if mps else base,
            dict(_PVPRO_MID_AUTOFILL) if ps else base,
            dot_on if mps else dot_off,
            dot_on if ps else dot_off,
            _pvpro_autofill_note(filled_bits))


# =============================================================================
# CALLBACK — Advanced-mode "Estimate from data" button.
#
# Same idea as the Simple-mode button, but Advanced mode has ALREADY identified
# the columns in Step 1 (prescreening), so this does NOT re-parse: it reads the
# existing mapped-vars-store directly and estimates Modules per string +
# Parallel strings into the Advanced fields. If Step 1 hasn't run yet (no
# mapping), it prompts the user to run prescreening first.
# =============================================================================
@app.callback(
    Output("param-pvpro-mps", "value", allow_duplicate=True),
    Output("param-pvpro-ps",  "value", allow_duplicate=True),
    Output("param-pvpro-mps", "style", allow_duplicate=True),
    Output("param-pvpro-ps",  "style", allow_duplicate=True),
    Output("param-pvpro-mps-dot", "style", allow_duplicate=True),
    Output("param-pvpro-ps-dot",  "style", allow_duplicate=True),
    Output("pvpro-autofill-note", "children", allow_duplicate=True),
    Input("adv-pvpro-estimate-btn", "n_clicks"),
    State("dataframe-store",   "data"),
    State("mapped-vars-store", "data"),
    State("param-pvpro-cells", "value"),
    prevent_initial_call=True,
)
def estimate_pvpro_advanced(n, df_json, mapping, cells):
    nu = dash.no_update

    def _msg(text):
        return html.Div(text, className="pvcopilot-note-float-in",
                        style={"fontSize": "12px", "color": INK_SOFT,
                               "marginTop": "4px", "lineHeight": "1.5",
                               "fontFamily": _HINT_FONT})

    if not n:
        return nu, nu, nu, nu, nu, nu, nu
    if not df_json or not mapping:
        # Advanced mode expects Step 1 (prescreening) to have identified columns.
        return nu, nu, nu, nu, nu, nu, _msg(
            "Run prescreening (Step 1) first so the DC voltage / current columns "
            "are identified, then click Estimate from data.")
    try:
        est = estimate_pvpro_params(_df_from_store(df_json), mapping,
                                    cells_in_series=_pvnum(cells, 60, int))
    except Exception:
        est = {}
    if not est:
        return nu, nu, nu, nu, nu, nu, _msg(
            "Couldn't estimate from this data \u2014 PVPRO estimation needs DC "
            "voltage + current (or DC power) columns that agree with each other.")

    base = dict(_PVPRO_MID_BASE)
    dot_off = dict(_PVPRO_DOT_OFF)
    dot_on = dict(_PVPRO_DOT_ON)
    mps = est.get("modules_per_string")
    ps = est.get("parallel_strings")
    filled_bits = []
    if mps:
        filled_bits.append(f"Modules per string = {mps['value']} ({mps['basis']})")
    if ps:
        filled_bits.append(f"Parallel strings = {ps['value']} ({ps['basis']})")
    return (mps["value"] if mps else nu,
            ps["value"] if ps else nu,
            dict(_PVPRO_MID_AUTOFILL) if mps else base,
            dict(_PVPRO_MID_AUTOFILL) if ps else base,
            dot_on if mps else dot_off,
            dot_on if ps else dot_off,
            _pvpro_autofill_note(filled_bits))


# =============================================================================
# CALLBACK — PVPRO numeric steppers (our own - / + buttons)
#
# The native number-input spinners blank the value in this Dash version, so
# each PVPRO number field has explicit - / + buttons (see _pvpro_num_field).
# This one callback handles all of them (Advanced + Simple): the clicked
# button's id carries which input to change and the direction; we read that
# input's current value, step it, clamp to its minimum, and write it back.
# =============================================================================
@app.callback(
    [Output(t, "value", allow_duplicate=True) for t in _PVPRO_STEP_TARGETS],
    Input({"type": "pvpro-step", "target": ALL, "dir": ALL}, "n_clicks"),
    [State(t, "value") for t in _PVPRO_STEP_TARGETS],
    prevent_initial_call=True,
)
def _pvpro_step(_clicks, *vals):
    out = [dash.no_update] * len(_PVPRO_STEP_TARGETS)
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return out
    target = trig.get("target")
    direction = trig.get("dir")
    if target not in _PVPRO_STEP_TARGETS:
        return out
    idx = _PVPRO_STEP_TARGETS.index(target)
    suffix = target.split("param-pvpro-")[-1]           # e.g. "cells"
    step, minv, decimals = _PVPRO_STEP_CFG.get(suffix, (1, None, 0))
    cur = _pvnum(vals[idx], minv if minv is not None else 0, float)
    newv = cur + (step if direction == "up" else -step)
    if minv is not None and newv < minv:
        newv = minv
    newv = round(newv, decimals) if decimals else int(round(newv))
    out[idx] = newv
    return out


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
    time.sleep(1)

    preview_lines = "\n".join(clean_code.splitlines()[:24]) + "\n…"

    preview = html.Div([
        html.Div("generated python", style={
            "fontSize": "13px", "color": INK_SOFT, "textTransform": "uppercase",
            "letterSpacing": "0.1em", "fontWeight": "600", "marginBottom": "10px",
            "fontFamily": "Archivo, system-ui, sans-serif",
        }),
        html.Pre(
            preview_lines,
            style={
                "whiteSpace": "pre-wrap",
                "fontSize": "13px",
                "background": "linear-gradient(135deg, #303640, #20242b)",
                "color": "#e8e4dc",
                "padding": "16px",
                "borderRadius": "16px",
                "maxHeight": "260px",
                "overflowY": "auto",
                "fontFamily": "Archivo, system-ui, sans-serif",
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
        "borderRadius": "12px",
        "background": "white",
        "fontFamily": "Archivo, system-ui, sans-serif",
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
    from page_supporting_files.analysis_utils import (
        client as _llm_client,
        LLM_MODEL as _diagnostic_model,
    )
except Exception:
    _llm_client = None
    _diagnostic_model = None

try:
    from page_supporting_files.diagnostic_prompts import (
        DIAGNOSTIC_SYSTEM_PROMPT as _diagnostic_system_prompt,
        DIAGNOSTIC_SYSTEM_PROMPT_PVPRO as _diagnostic_system_prompt_pvpro,
    )
except Exception:
    _diagnostic_system_prompt = (
        "You are PV Copilot, an expert in photovoltaic degradation analysis. "
        "Give a concise, practical interpretation of the supplied result, its "
        "data-quality caveats, and the most useful next validation step."
    )
    _diagnostic_system_prompt_pvpro = _diagnostic_system_prompt


@app.callback(
    Output("simple-ai-diagnostic-card", "style"),
    Output("advanced-ai-diagnostic-card", "style"),
    Input("simple-stash", "data"),
    Input("degradation-result-store", "data"),
)
def show_ai_diagnostic_cards(simple_result, advanced_result):
    simple_visible = bool(simple_result and simple_result.get("method"))
    advanced_visible = bool(advanced_result and advanced_result.get("method"))
    return (
        {} if simple_visible else {"display": "none"},
        {} if advanced_visible else {"display": "none"},
    )


def _format_diagnostic_result_context(result, session_context):
    result = result or {}
    rate_fraction = result.get("rate_pct")
    try:
        headline_rate = f"{float(rate_fraction) * 100:+.2f}%/year"
    except (TypeError, ValueError):
        headline_rate = "unavailable"
    lines = [
        f"Method: {result.get('method') or 'unknown'}",
        f"Headline degradation rate: {headline_rate}",
        f"Analysis window: {result.get('start')} to {result.get('end')} "
        f"({result.get('duration_years')} years)",
        f"Points retained: {result.get('n_kept')} of {result.get('n_raw')} "
        f"({result.get('pct_kept')}%)",
    ]
    methods_rates = result.get("methods_rates") or {}
    if methods_rates:
        lines.append("Per-method rates (%/year): " + ", ".join(
            f"{key}={value:+.2f}" if value is not None else f"{key}=n/a"
            for key, value in methods_rates.items()
        ))
    quantity_rates = result.get("rates_per_quantity") or {}
    if quantity_rates:
        lines.append("PVPRO reference-parameter rates (%/year): " + ", ".join(
            f"{key}={float(value):+.2f}" for key, value in quantity_rates.items()
            if value is not None
        ))
    if result.get("trend_summary"):
        lines.append("Trend evidence: " + str(result["trend_summary"]))
    if result.get("raw_summary"):
        lines.append("Raw-data quality evidence: " + str(result["raw_summary"]))
    if session_context:
        lines.append("Completed workflow context: " + str(session_context))
    return "\n".join(lines)


@app.callback(
    Output("simple-ai-diagnostic-output", "children"),
    Output("advanced-ai-diagnostic-output", "children"),
    Output("simple-ai-diagnostic-btn", "hidden"),
    Output("simple-ai-diagnostic-restart", "style"),
    Output("advanced-ai-diagnostic-btn", "hidden"),
    Output("advanced-ai-diagnostic-restart", "style"),
    Input("simple-ai-diagnostic-btn", "n_clicks"),
    Input("advanced-ai-diagnostic-btn", "n_clicks"),
    Input("simple-ai-diagnostic-restart", "n_clicks"),
    Input("advanced-ai-diagnostic-restart", "n_clicks"),
    Input("simple-stash", "data"),
    Input("degradation-result-store", "data"),
    State("chat-data-context", "data"),
    prevent_initial_call=True,
)
def run_result_ai_diagnostic(_simple_clicks, _advanced_clicks,
                             _simple_restarts, _advanced_restarts,
                             simple_result, advanced_result, session_context):
    trigger = ctx.triggered_id
    if trigger == "simple-stash":
        return "", dash.no_update, False, {"display": "none"}, dash.no_update, dash.no_update
    if trigger == "degradation-result-store":
        return dash.no_update, "", dash.no_update, dash.no_update, False, {"display": "none"}
    if trigger == "simple-ai-diagnostic-restart":
        return ("", dash.no_update, False, {"display": "none"},
                dash.no_update, dash.no_update)
    if trigger == "advanced-ai-diagnostic-restart":
        return (dash.no_update, "", dash.no_update, dash.no_update,
                False, {"display": "none"})
    if trigger not in ("simple-ai-diagnostic-btn", "advanced-ai-diagnostic-btn"):
        return (dash.no_update,) * 6
    result = (simple_result if trigger == "simple-ai-diagnostic-btn"
              else advanced_result)
    if not result or not result.get("method"):
        node = html.Div("Run an analysis first.", className="pvc-ai-diagnostic-error")
    else:
        # Simple and Advanced are independent analyses. The shared chat context
        # describes the Advanced workflow, so it must never be mixed into a
        # Simple-mode diagnosis.
        diagnostic_session = (None if trigger == "simple-ai-diagnostic-btn"
                              else session_context)
        context_text = _format_diagnostic_result_context(result, diagnostic_session)
        uses_pvpro = str(result.get("method", "")).upper() == "PVPRO"
        if _llm_client is None:
            node = dcc.Markdown(
                "**Diagnosis unavailable:** the AI client is not configured in this environment.",
                className="pvc-ai-diagnostic-markdown",
            )
        else:
            try:
                system_prompt = (_diagnostic_system_prompt_pvpro if uses_pvpro
                                 else _diagnostic_system_prompt)
                response = _llm_client.chat.completions.create(
                    model=_diagnostic_model or "gpt-5.4-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": (
                            "Analysis context:\n" + context_text
                            + "\n\nGive a concise diagnosis with a headline, key evidence, "
                              "important caveats, and one recommended next step."
                        )},
                    ],
                    timeout=60,
                )
                text = (response.choices[0].message.content or "").strip()
                node = dcc.Markdown(text, className="pvc-ai-diagnostic-markdown")
            except Exception as exc:
                node = html.Div(
                    f"AI diagnosis unavailable: {exc}",
                    className="pvc-ai-diagnostic-error",
                )
    if trigger == "simple-ai-diagnostic-btn":
        return (node, dash.no_update, True, {},
                dash.no_update, dash.no_update)
    return (dash.no_update, node, dash.no_update, dash.no_update,
            True, {})

_EXAMPLE_QUESTIONS = [
    "What's my degradation rate?",
    "Which degradation method should I try?",
    "How were points filtered?",
    "What does PVPRO add?",
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

_CHAT_RESPONSE_FORMAT = """
RESPONSE FORMAT:
- Use Markdown and begin with one short, descriptive level-3 heading (`### Title`).
- Never use level-1 or level-2 headings (`#` or `##`).
- Organize the answer into 2–4 short paragraphs when explanation is needed.
- Use a compact bullet list for rates, comparisons, evidence, or next steps.
- Bold only the most important values or conclusions.
- Keep the answer concise and grounded in CURRENT SESSION STATE.
"""


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
    Input("simple-stash",             "data"),
    Input("ui-mode",                  "data"),
    State("stored-data-file-name",    "data"),
    State("cb-timezone",              "value"),
    State("cb-low-irra-power",        "value"),
    State("cb-outlier",               "value"),
    State("cb-clearsky",              "value"),
    prevent_initial_call=False,
)
def build_chat_context(mapped_vars, df_data, df_filtered, deg_result,
                       selected_metric, dl_style, simple_result, ui_mode, filename,
                       cb_tz, cb_irra, cb_out, cb_cs):
    """Returns a structured dict the LLM uses to ground its answers."""
    active_mode = "simple" if ui_mode == "simple" else "advanced"
    ctx = {
        "analysis_mode": active_mode,
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

    # Simple and Advanced are independent analyses. In Simple mode, ground the
    # assistant only in simple-stash; never leak an Advanced result into chat.
    if active_mode == "simple":
        simple_result = simple_result or {}
        if simple_result.get("method"):
            n_raw = simple_result.get("n_raw")
            n_kept = simple_result.get("n_kept")
            ctx["filter_applied"] = True
            ctx["filter"] = {
                "filters_applied": ["Simple-mode best-practice defaults"],
                "n_rows_after_filter": int(n_kept) if n_kept else None,
                "n_rows_before_filter": int(n_raw) if n_raw else None,
                "fraction_kept_pct": simple_result.get("pct_kept"),
            }
            try:
                rate_percent = float(simple_result.get("rate_pct")) * 100.0
            except (TypeError, ValueError):
                rate_percent = None
            ctx["degradation_computed"] = rate_percent is not None
            ctx["degradation"] = {
                "rate_percent_per_year": rate_percent,
                "method": simple_result.get("method"),
                "duration_years": simple_result.get("duration_years"),
                "window_start": simple_result.get("start"),
                "window_end": simple_result.get("end"),
                "rates_per_quantity": simple_result.get("rates_per_quantity"),
                "trend_summary": simple_result.get("trend_summary"),
                "raw_summary": simple_result.get("raw_summary"),
            }
        return ctx

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
            "trend_summary": deg_result.get("trend_summary"),
            "raw_summary": deg_result.get("raw_summary"),
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

    mode = "simple" if ctx.get("analysis_mode") == "simple" else "advanced"
    lines = [
        f"CURRENT SESSION STATE — active analysis mode: {mode.upper()}.",
        "Simple and Advanced are independent analyses. Use ONLY the result for "
        "the active mode shown below. Never reuse a result mentioned earlier in "
        "the conversation if it came from the other mode.",
    ]

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
        lines.append("✗ FILTERING — NOT YET RUN")
        if mode == "simple":
            lines.append("  Tell the user to click 'Run analysis' in Simple mode first.")
        else:
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
        if g.get("trend_summary"):
            lines.append(f"  • Calculated trend evidence: {g.get('trend_summary')}")
        if g.get("raw_summary"):
            lines.append(f"  • Data-quality evidence: {g.get('raw_summary')}")
    else:
        lines.append("")
        lines.append("✗ DEGRADATION — NOT YET RUN")
        if mode == "simple":
            lines.append("  If the user asks for their rate or result, tell them to click "
                         "'Run analysis' in Simple mode first.")
        else:
            lines.append("  If the user asks 'what is my degradation rate' or about method results, "
                         "tell them to click 'Calculate Degradation' first.")

    # Step 4
    if mode == "advanced":
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
                + "\n\n---\n\n"
                + _CHAT_RESPONSE_FORMAT
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
            "Hi — I'm your PV Copilot. Upload a dataset or pick an example, "
            "then ask me about the methods, filters, or results.",
            className="pvc-chat-welcome",
            style={
                "alignSelf": "flex-start",
            }
        )

    # Render completed replies at once; no typewriter/line-by-line animation.
    bubbles = []
    for m in history:
        bubbles.append(_chat_bubble(m["role"], m["content"]))

    thinking = pending.get("thinking", False)

    if thinking:
        # Thinking indicator — shows after user submits, until reply arrives
        bubbles.append(
            html.Div(
                html.Div(
                    html.Div(
                        [
                            html.Span(className="chat-thinking-dot"),
                            html.Span(className="chat-thinking-dot"),
                            html.Span(className="chat-thinking-dot"),
                        ],
                        className="chat-thinking-dots",
                        **{"aria-label": "PVCopilot is thinking"},
                    ),
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

        // Render completed answers as compact, safe Markdown-like HTML.
        // Every Markdown heading level uses the same modest visual treatment.
        function renderRichMarkdown(text) {
            const lines = (text || '').replace(/\\r\\n/g, '\\n').split('\\n');
            const blocks = [];
            let paragraph = [];
            let bullets = [];

            function flushParagraph() {
                if (!paragraph.length) return;
                blocks.push('<p>' + renderBold(paragraph.join(' ')) + '</p>');
                paragraph = [];
            }
            function flushBullets() {
                if (!bullets.length) return;
                blocks.push('<ul>' + bullets.map(function(item) {
                    return '<li>' + renderBold(item) + '</li>';
                }).join('') + '</ul>');
                bullets = [];
            }

            lines.forEach(function(rawLine) {
                const line = rawLine.trim();
                const heading = line.match(/^#{1,6}\\s+(.+)$/);
                const bullet = line.match(/^[-*]\\s+(.+)$/);
                if (!line) {
                    flushParagraph();
                    flushBullets();
                } else if (heading) {
                    flushParagraph();
                    flushBullets();
                    blocks.push('<h4 class="chat-md-heading">' + renderBold(heading[1]) + '</h4>');
                } else if (bullet) {
                    flushParagraph();
                    bullets.push(bullet[1]);
                } else {
                    flushBullets();
                    paragraph.push(line);
                }
            });
            flushParagraph();
            flushBullets();
            return blocks.join('');
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
            if (visible && source && bubble.getAttribute('data-rich-rendered') !== '1') {
                const raw = source.textContent || '';
                visible.innerHTML = renderRichMarkdown(raw);
                bubble.setAttribute('data-rich-rendered', '1');
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
                    // Final render — compact heading, paragraphs, bullets + bold.
                    visible.innerHTML = renderRichMarkdown(rawText);
                    bubble.setAttribute('data-rich-rendered', '1');
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
    State("step-progress",      "data"),
    State("ui-mode",            "data"),
    prevent_initial_call="initial_duplicate",
)
def update_progress(mapped_vars, df_filtered, deg_result, dl_style, prev, mode):
    # Simple mode drives the sidebar via its own staged-reveal callback, so
    # this store-watcher must not clobber it.  Only track real-store progress
    # in Advanced mode.
    if mode != "advanced":
        return dash.no_update
    data_done = bool(mapped_vars)
    # "started" latches on: once any work has begun (or a click already set
    # it), it stays true so Step 1 keeps reading as active/done.
    started = bool((prev or {}).get("started")) or data_done
    return {
        "started": started,
        "data":   data_done,                                       # data parsed
        "filter": bool(df_filtered),                               # filters applied
        "calc":   bool(deg_result) and deg_result.get("rate_pct_per_year") is not None,
        "code":   bool(dl_style) and dl_style.get("display") not in (None, "none"),
    }


# Mark the workflow as "started" the instant the user clicks Run prescreening
# (Advanced) so Step 1 in the sidebar lights up as active before parsing
# finishes.  Simple mode sets this from its own Stage-A callback.
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
        _show_hide(not calc_done),    # code   locked  (add-on) until calc done
        _show_hide(calc_done),        # code   content
    )


# Advanced navigation is presentation-only. Pipeline completion unlocks the
# next tab, but never changes the panel the user is currently reviewing.
@app.callback(
    Output("advanced-active-step", "data"),
    Input({"type": "advanced-step-tab", "step": ALL}, "n_clicks"),
    Input("advanced-next-step-1", "n_clicks"),
    Input("advanced-next-step-2", "n_clicks"),
    Input("advanced-next-step-3", "n_clicks"),
    State("step-progress", "data"),
    State("advanced-active-step", "data"),
    prevent_initial_call=True,
)
def select_advanced_step(_clicks, _next_1, _next_2, _next_3,
                         progress, current_step):
    trigger = ctx.triggered_id
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return current_step or 1

    progress = progress or {}
    enabled = {
        1: True,
        2: bool(progress.get("data")),
        3: bool(progress.get("filter")),
        4: bool(progress.get("calc")),
    }
    next_steps = {
        "advanced-next-step-1": 2,
        "advanced-next-step-2": 3,
        "advanced-next-step-3": 4,
    }
    if isinstance(trigger, dict):
        requested_step = int(trigger.get("step", 1))
    else:
        requested_step = next_steps.get(trigger, current_step or 1)
    return requested_step if enabled.get(requested_step, False) else (current_step or 1)


@app.callback(
    Output("advanced-filter-expanded", "data"),
    Output("advanced-metric-expanded", "data"),
    Input("toggle-filter-settings", "n_clicks"),
    Input("toggle-metric-settings", "n_clicks"),
    Input("filter-btn", "n_clicks"),
    Input("run-btn", "n_clicks"),
    Input("step-progress", "data"),
    State("advanced-filter-expanded", "data"),
    State("advanced-metric-expanded", "data"),
    prevent_initial_call=True,
)
def toggle_completed_step_settings(_filter_clicks, _metric_clicks,
                                   _filter_runs, _metric_runs, progress,
                                   filter_expanded, metric_expanded):
    """Remember only the user's UI expansion choices; no pipeline state changes."""
    trigger = ctx.triggered_id
    if trigger == "toggle-filter-settings":
        return not bool(filter_expanded), bool(metric_expanded)
    if trigger == "toggle-metric-settings":
        return bool(filter_expanded), not bool(metric_expanded)
    if trigger == "filter-btn":
        return False, bool(metric_expanded)
    if trigger == "run-btn":
        return bool(filter_expanded), False
    if trigger == "step-progress":
        progress = progress or {}
        # Reset a fold preference only when its underlying result is reset.
        return (
            bool(filter_expanded) if progress.get("filter") else False,
            bool(metric_expanded) if progress.get("calc") else False,
        )
    return bool(filter_expanded), bool(metric_expanded)


@app.callback(
    Output({"type": "advanced-step-tab", "step": ALL}, "className"),
    Output({"type": "advanced-step-tab", "step": ALL}, "disabled"),
    Output("agent-data-wrap", "style"),
    Output("agent-filter-wrap", "style"),
    Output("agent-calc-wrap", "style"),
    Output("agent-code-wrap", "style"),
    Output("advanced-data-body", "className"),
    Output("advanced-filter-body", "className"),
    Output("advanced-calc-body", "className"),
    Input("advanced-active-step", "data"),
    Input("step-progress", "data"),
    Input("advanced-filter-expanded", "data"),
    Input("advanced-metric-expanded", "data"),
)
def render_advanced_navigation(active_step, progress, filter_expanded, metric_expanded):
    progress = progress or {}
    completed = {
        1: bool(progress.get("data")),
        2: bool(progress.get("filter")),
        3: bool(progress.get("calc")),
        4: bool(progress.get("code")),
    }
    enabled = {
        1: True,
        2: completed[1],
        3: completed[2],
        4: completed[3],
    }

    active_step = int(active_step or 1)
    # A workflow reset can invalidate a previously selected later step.
    # Display Step 1 in that case without coupling completion to navigation.
    visible_step = active_step if enabled.get(active_step, False) else 1

    classes = []
    disabled = []
    for step in range(1, 5):
        states = []
        if step == visible_step:
            states.append("is-active")
        if completed[step]:
            states.append("is-done")
        elif enabled[step]:
            states.append("is-enabled")
        else:
            states.append("is-locked")
        classes.append(f"pvc-advanced-rail-card pvc-advanced-step-{step} {' '.join(states)}")
        disabled.append(not enabled[step])

    panel_styles = [
        {"display": "block"} if visible_step == step else {"display": "none"}
        for step in range(1, 5)
    ]
    data_body_class = "pvc-advanced-step-body" + (" is-complete" if completed[1] else "")
    filter_body_class = "pvc-advanced-step-body"
    metric_body_class = "pvc-advanced-step-body"
    if completed[2]:
        filter_body_class += " is-complete " + ("is-expanded" if filter_expanded else "is-collapsed")
    if completed[3]:
        metric_body_class += " is-complete " + ("is-expanded" if metric_expanded else "is-collapsed")

    return (
        classes, disabled, *panel_styles,
        data_body_class, filter_body_class, metric_body_class,
    )


# =============================================================================
# CALLBACK — MODE SWITCH (Simple <-> Advanced)
#
# Clicking either tab sets `ui-mode`.  A second callback toggles which panel
# is visible and re-renders the tab bar so the active tab is highlighted.
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
# Reveal the Analyze card only once data has been selected in the shared
# upload area. Its Run button remains enabled whenever the card is visible.
# -----------------------------------------------------------------------------
@app.callback(
    Output("simple-analyze-btn", "disabled"),
    Output("simple-analyze-btn", "style"),
    Output("analyze-section-card", "className"),
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
    card_class = "glass rise pvc-analyze-card" + ("" if ready else " is-hidden")
    # The card itself is the readiness gate, so its Run button never needs a
    # disabled visual state once the card becomes visible.
    return False, _simple_analyze_style(disabled=False), card_class


# -----------------------------------------------------------------------------
# Simple mode, STAGE A (instant): on Analyze click, immediately show a status
# banner and write the run trigger.  Returns right away so the banner paints
# with no perceptible delay; the heavy compute happens in Stage B below.
# -----------------------------------------------------------------------------
@app.callback(
    Output("simple-status",       "children", allow_duplicate=True),
    Output("simple-result",       "children", allow_duplicate=True),
    Output("simple-run-trigger",  "data"),
    Output("simple-step-progress", "data", allow_duplicate=True),
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
_EXAMPLE_FRIENDLY = {
    "sys_1278_downsampled_with_VI.parquet": "Example data 1",
    "sys_1403_part1_downsampled_with_VI.parquet": "Example data 2",
    "sys_1422_downsampled.parquet": "Example data 3",
}


def _simple_fail(alert):
    """Common failure return for the data stage (6 outputs)."""
    _none = {"started": True, "data": False, "filter": False,
             "calc": False, "code": False}
    # simple-pipe-data, simple-status, simple-result, simple-stash,
    # simple-step-progress
    return {}, alert, "", {}, _none


# ---- STAGE 1 : load the dataframe + identify variables ----------------------
@app.callback(
    Output("simple-pipe-data", "data"),
    Output("simple-status",    "children", allow_duplicate=True),
    Output("simple-result",    "children", allow_duplicate=True),
    Output("simple-stash",     "data", allow_duplicate=True),
    Output("simple-step-progress", "data", allow_duplicate=True),
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

    irra_key = mapped.get("Irradiance")
    if irra_key is None or irra_key not in df.columns:
        return _simple_fail(_no_data_alert(
            "Irradiance column not found. Try Advanced mode to map it manually."
        ))

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
    Output("simple-step-progress", "data", allow_duplicate=True),
    Input("simple-pipe-data", "data"),
    prevent_initial_call=True,
)
def simple_stage_filter(pdata):
    if not pdata or "df" not in pdata:
        return dash.no_update, dash.no_update, dash.no_update

    # Brief pause so the filtering progress state remains visible.
    time.sleep(1)

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

        clearsky_mask = pd.Series(True, index=df.index)
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

        normal_idx, _ = low_irra_power_filter(
            df_filtered, mapped,
            irr_thresh=300, power_ratio=0.02,
            norm_lower=0.01, norm_upper_pct=99,
        )
        current_mask &= df_filtered.index.isin(normal_idx)

        normal_idx, _ = identify_outliers_iqr(df_filtered, "norm", iqr_multiplier=1.5)
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
    }
    method = pdata.get("method", "YOY")
    progress = {"started": True, "data": True, "filter": True,
                "calc": False, "code": False}
    step3_label = ("Estimating PVPRO parameters and fitting the model…" if method == "PVPRO"
                   else _SIMPLE_STEP_LABELS.get(3, "Estimating degradation…"))
    status = _working_banner(step3_label)
    return payload, status, progress

# ---- STAGE 3 : degradation — YoY (fast) OR PVPRO (background fit) -----------
@app.callback(
    Output("simple-stash",   "data", allow_duplicate=True),
    Output("simple-status",  "children", allow_duplicate=True),
    Output("simple-result",  "children", allow_duplicate=True),
    Output("simple-step-progress", "data", allow_duplicate=True),
    Output("simple-pvpro-job",           "data",     allow_duplicate=True),
    Output("simple-pvpro-poll-interval", "disabled", allow_duplicate=True),
    Output("simple-pvpro-progress-output", "children", allow_duplicate=True),
    Output("simple-param-pvpro-mps", "value", allow_duplicate=True),
    Output("simple-param-pvpro-ps",  "value", allow_duplicate=True),
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
        return (dash.no_update,) * 9

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
                "use the YoY method."), "", none, {}, True, "",
                    dash.no_update, dash.no_update)
        try:
            df_filtered = _df_from_store(pfiltered["df_good"])
        except Exception as e:
            none = {"started": True, "data": True, "filter": True,
                    "calc": False, "code": False}
            return ({}, _no_data_alert(f"Could not load data for PVPRO: {e}"),
                    "", none, {}, True, "", dash.no_update, dash.no_update)

        # Simple mode is intentionally one-click: infer array geometry from
        # the filtered DC voltage/current/power signals before fitting. A
        # quantity that cannot be estimated safely falls back to the current
        # (normally default) value rather than aborting the run.
        estimated = {}
        cells_value = _pvnum(cells, 60, int)
        try:
            estimated = _run_with_timeout(
                estimate_pvpro_params,
                df_filtered,
                mapped,
                cells_in_series=cells_value,
                timeout=ANALYZE_TIMEOUT_S,
            ) or {}
        except Exception:
            estimated = {}

        estimated_mps = estimated.get("modules_per_string") or {}
        estimated_ps = estimated.get("parallel_strings") or {}

        modules_per_string = _pvnum(estimated_mps.get("value", mps), 1, int)
        parallel_strings = _pvnum(estimated_ps.get("value", ps), 1, int)
        pvpro_kwargs = dict(
            cells_in_series     = cells_value,
            modules_per_string  = modules_per_string,
            parallel_strings    = parallel_strings,
            alpha_isc           = _pvnum(alphaisc, 0.0046, float),
            technology          = tech      if tech      else "mono-c-Si",
            days_per_run        = _pvnum(days, 14, int),
            iterations_per_year = _pvnum(iters, 12, int),
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
                {"job_id": job_id}, False, initial_ui,
                modules_per_string, parallel_strings)

    # -------------------------------------------------------------------
    # YoY branch (default): fast synchronous estimate.
    # -------------------------------------------------------------------
    # Brief pause so the degradation progress state remains visible.
    time.sleep(1)

    def _fail(alert):
        none = {"started": True, "data": True, "filter": True,
                "calc": False, "code": False}
        return (
            {}, alert, "", none,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,
        )

    try:
        df_good = _df_from_store(pfiltered["df_good"])
        irra_key = pfiltered["irra_key"]
        daily_data = aggregate_daily(df_good, irra_key)
        rd, fig = compute_yoy(daily_data, rolling_window=30, iqr_multiplier=1.5)
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
        "method": "YOY",
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
    }

    # Success: Step 3 DONE.  Reveal the result.
    progress = {"started": True, "data": True, "filter": True,
                "calc": True, "code": False}
    done_status = ""   # no success banner on completion (per request)
    return (stash, done_status, _simple_result_layout(stash), progress,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update)


# -----------------------------------------------------------------------------
# Rebuild the Simple-mode result layout from the stashed primitives.
# -----------------------------------------------------------------------------
def _simple_result_layout(stash):
    import plotly.io as pio
    fig = pio.from_json(stash["fig"]) if stash.get("fig") else None
    rate_pct = stash["rate_pct"]
    duration_years = stash["duration_years"]
    n_kept = stash.get("n_kept", 0)
    pct_kept = stash.get("pct_kept", 0.0)

    if fig is not None:
        if len(fig.data) > 0:
            fig.data[0].name = "Daily-aggregated Power"
            fig.data[0].marker.update(size=8, opacity=0.62, color="#9fcaf1")
        if len(fig.data) > 1:
            fig.data[1].name = "Trend (30-day rolling)"
            fig.data[1].line.update(color="#0878c9", width=3)
        fig.update_layout(
            title=None,
            height=390,
            autosize=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Archivo, Arial, sans-serif", color=INK_SOFT, size=13),
            margin=dict(l=66, r=30, t=22, b=92),
            legend=dict(
                orientation="h", x=0.5, xanchor="center", y=-0.24, yanchor="top",
                bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=INK_SOFT),
            ),
            hovermode="x unified",
        )
        fig.update_xaxes(
            title="Time", showgrid=True, gridcolor="rgba(105,135,180,0.18)",
            zeroline=False, linecolor="rgba(105,135,180,0.16)",
        )
        fig.update_yaxes(
            title="Power (W)", showgrid=True, gridcolor="rgba(105,135,180,0.18)",
            zeroline=False, linecolor="rgba(105,135,180,0.16)",
        )

    def status_row(label, value):
        return html.Div(className="pvc-yoy-status-row", children=[
            html.Span("✓", className="pvc-yoy-status-check"),
            html.Span(label, className="pvc-yoy-status-label"),
            html.Span("·", className="pvc-yoy-status-separator"),
            html.Strong(value, className="pvc-yoy-status-value"),
        ])

    summary_card = html.Div(className="pvc-yoy-summary-card", children=[
        html.Div("Performance loss rate", className="pvc-yoy-kicker"),
        html.Div(className="pvc-yoy-rate", children=[
            html.Span(f"{rate_pct * 100:.2f}", className="pvc-yoy-rate-value"),
            html.Span("%/yr", className="pvc-yoy-rate-unit"),
        ]),
        html.Div(
            ["Year-on-year", html.Span("·"), f"{duration_years:.1f}-year record"],
            className="pvc-yoy-meta",
        ),
        html.Div(className="pvc-yoy-status-list", children=[
            status_row("Pre-screening", "complete"),
            status_row("Filtering", f"{pct_kept:.0f}% points retained"),
            status_row("Model", f"{n_kept:,} points fitted"),
        ]),
    ])

    chart_card = html.Div(className="pvc-yoy-chart-card", children=[
        html.H3("Normalized power trend", className="pvc-yoy-chart-title"),
        dcc.Graph(
            figure=fig,
            className="pvc-yoy-graph",
            style={"width": "100%", "height": "390px", "maxHeight": "390px"},
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            },
        ) if fig is not None else html.Div("Trend chart unavailable.", className="pvc-yoy-chart-empty"),
    ])

    return html.Div(
        [summary_card, chart_card],
        className="pvc-simple-yoy-result slide-in-up",
    )


# -----------------------------------------------------------------------------
# Status labels for each Simple-mode pipeline stage (shown while it runs).
# -----------------------------------------------------------------------------
_SIMPLE_STEP_LABELS = {
    1: "Inspecting data & identifying variables…",
    2: "Applying default filters…",
    3: "Estimating degradation…",
}


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
