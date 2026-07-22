"""
PV Copilot — pvtools page, new liquid-glass UI (Plan B).

The whole page is rebuilt in the new Apple "liquid glass" design and wired to the
REAL analysis pipeline, with the full Advanced-mode feature set carried over:

  * data prescreening (make_overview_figures) + LLM variable mapping
  * intelligent filtering with tunable thresholds
  * degradation modelling with MULTI-METHOD comparison and per-method parameters
    (year-on-year, linear regression, Holt-Winters, ARIMA, seasonal decomposition)
  * PVPRO single-diode physics fit (synchronous)
  * AI diagnostics (LLM) + Ask-Copilot chat
  * reproducible code generation

Integration: it is a pvtools page — `from app import app`, exposes get_layout(),
uses no dcc.Location (the site owns the URL); the top nav switches an internal
view via a Store. All dynamic buttons use pattern-matching ids so they are
tolerated when absent.
"""

import os
import base64
import io
import json
import pickle
import socket
import threading
import time
import traceback
import uuid
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH, no_update
import dash_mantine_components as dmc

# ---- shared Dash instance (pvtools app.py) ----
from app import app

# ---- real pipeline ----
from page_supporting_files.analysis_utils_260701 import (
    parse_contents, make_overview_figures, normalize, low_irra_power_filter,
    aggregate_daily, compute_yoy, compute_lr, compute_hw, compute_arima,
    compute_csd, get_full_code, estimate_pvpro_params, compute_pvpro,
)
from page_supporting_files.pvcopilot_filter_functions import (
    identify_outliers_iqr, clear_sky_filter, basic_value_filter,
)
try:
    # Single source of truth for the 5-parameter blue->green scheme, shared
    # with the PVPRO trend plots so pill accents match the trend colors.
    from page_supporting_files.analysis_utils_260701 import PVPRO_VAR_COLORS
except Exception:  # pragma: no cover
    PVPRO_VAR_COLORS = {"p_mp_ref": "#1558b0", "v_mp_ref": "#1a86c7",
                        "i_mp_ref": "#12998e", "v_oc_ref": "#2fa855", "i_sc_ref": "#7cc242"}
try:
    from page_supporting_files.analysis_utils_260701 import _quality_tag
except Exception:  # pragma: no cover
    def _quality_tag(df, col, role=None):
        return ""
try:
    from page_supporting_files.analysis_utils_260701 import client as _LLM_CLIENT, LLM_MODEL as _LLM_MODEL
except Exception:  # pragma: no cover
    _LLM_CLIENT, _LLM_MODEL = None, None
try:
    from page_supporting_files.diagnostic_prompts import (
        DIAGNOSTIC_SYSTEM_PROMPT as _DIAG_SYS,
        DIAGNOSTIC_SYSTEM_PROMPT_PVPRO as _DIAG_SYS_PVPRO,
    )
except Exception:  # pragma: no cover
    _DIAG_SYS = ("You are PV Copilot, an expert assistant for photovoltaic degradation "
                 "analysis. Given the analysis context, give a concise, practical read "
                 "of the degradation result and any data-quality caveats.")
    _DIAG_SYS_PVPRO = _DIAG_SYS
_CHAT_SYS = ("You are PV Copilot, an assistant for photovoltaic degradation analysis. "
             "Answer concisely and only about the user's PV dataset, the degradation "
             "methods, the filtering steps, or how to read the results.")

# --------------------------------------------------------------------------- palette
BLUE = "#2f6bff"
INK = "#1c2540"
SUB = "#5a6784"
MUTE = "#8090ad"
BORDER = "rgba(120,140,180,0.2)"

# Graph config: keep zoom / pan / autoscale + "reset axes" (home) so users can
# zoom in/out and return to the default scale; drop lasso / select / snapshot.
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"],
                "scrollZoom": False}
_METHOD_COLORS = {"YOY": "#0070C0", "LR": "#83CBEB", "HW": "#68DBCE",
                  "ARIMA": "#048E2F", "CSD": "#92D050", "PVPRO": "#e08a2b"}

# ---- Shared background-job registry ---------------------------------------
# Diskcache is already part of this project.  Use it for every async stage so
# a request handled by another Gunicorn worker can still see the job and its
# complete history.  The in-memory fallback keeps local development working.
_JOB_CACHE = None
_JOB_INSTANCE = (os.environ.get("DYNO") or os.environ.get("RENDER_INSTANCE_ID") or
                 os.environ.get("HOSTNAME") or socket.gethostname())
_JOB_STORE_STATUS = "process-memory fallback"
_job_store_errors = []


class _RedisJobCache:
    """Small diskcache-compatible adapter for truly cross-container jobs."""
    def __init__(self, url):
        import redis
        self.client = redis.Redis.from_url(
            url, socket_connect_timeout=3, socket_timeout=5,
            health_check_interval=30, retry_on_timeout=True)
        self.client.ping()

    def set(self, key, value, expire=None):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return bool(self.client.set(key, payload, ex=int(expire) if expire else None))

    def get(self, key, default=None):
        payload = self.client.get(key)
        return default if payload is None else pickle.loads(payload)

    def delete(self, key):
        self.client.delete(key)


# A local /tmp directory is shared by Gunicorn workers in one container, but
# never by separate containers/dynos. Prefer Redis whenever deployment supplies
# REDIS_URL; otherwise retain diskcache plus an in-process mirror.
_redis_url = os.environ.get("PVPRO_REDIS_URL") or os.environ.get("REDIS_URL")
if _redis_url:
    try:
        _JOB_CACHE = _RedisJobCache(_redis_url)
        _JOB_STORE_STATUS = "shared Redis"
    except Exception as exc:
        _job_store_errors.append(f"Redis {type(exc).__name__}: {exc}")

if _JOB_CACHE is None:
    try:
        import diskcache as _diskcache
        _job_cache_dir = os.environ.get("PVPRO_DISKCACHE_DIR", "/tmp/pvcopilot-jobs")
        os.makedirs(_job_cache_dir, exist_ok=True)
        _JOB_CACHE = _diskcache.Cache(_job_cache_dir)
        if not _JOB_CACHE.set("__pvc_probe__", {"pid": os.getpid()}, expire=30):
            raise RuntimeError("diskcache write failed")
        if not _JOB_CACHE.get("__pvc_probe__"):
            raise RuntimeError("diskcache round-trip failed")
        _JOB_STORE_STATUS = f"container-local diskcache: {_job_cache_dir}"
    except Exception as exc:
        _JOB_CACHE = None
        _job_store_errors.append(f"diskcache {type(exc).__name__}: {exc}")
        _JOB_STORE_STATUS = "process-memory fallback" + (
            f" ({'; '.join(_job_store_errors)})" if _job_store_errors else "")


class _JobRegistry:
    def __init__(self, namespace):
        self.namespace = namespace
        self.memory = {}

    def _key(self, jid):
        return f"pvc:{self.namespace}:{jid}"

    def __setitem__(self, jid, value):
        # Always keep a process-local mirror. It prevents a transient sqlite
        # miss from killing a job launched and polled by this same process.
        self.memory[jid] = value
        if _JOB_CACHE is not None:
            try:
                _JOB_CACHE.set(self._key(jid), value, expire=60 * 60 * 6)
            except Exception:
                traceback.print_exc()

    def get(self, jid, default=None):
        if not jid:
            return default
        if _JOB_CACHE is not None:
            try:
                value = _JOB_CACHE.get(self._key(jid), default=None)
                if value is not None:
                    self.memory[jid] = value
                    return value
            except Exception:
                traceback.print_exc()
        return self.memory.get(jid, default)

    def pop(self, jid, default=None):
        value = self.get(jid, default)
        self.memory.pop(jid, None)
        if _JOB_CACHE is not None:
            try:
                _JOB_CACHE.delete(self._key(jid))
            except Exception:
                traceback.print_exc()
        return value


_PVPRO_JOBS = _JobRegistry("pvpro")
_DIAG_JOBS = _JobRegistry("diagnosis")
_INGEST_JOBS = _JobRegistry("upload")
_ANALYSIS_JOBS = _JobRegistry("analysis")
_DATASETS = _JobRegistry("dataset")


def _data_for_browser(data):
    """Return metadata only; the dataframe remains in the shared job store."""
    if not data:
        return EMPTY_DATA
    return {k: v for k, v in dict(data).items() if k != "df"}


def _store_dataset(data, dataset_id=None):
    """Persist the complete dataset and return its lightweight browser handle."""
    full = dict(data or {})
    dataset_id = dataset_id or full.get("dataset_id") or uuid.uuid4().hex[:16]
    full["dataset_id"] = dataset_id
    _DATASETS[dataset_id] = full
    return _data_for_browser(full)


def _materialize_dataset(data):
    """Resolve a browser handle to its complete server-side dataset."""
    ref = dict(data or {})
    if not ref.get("loaded") or ref.get("df"):
        return ref
    dataset_id = ref.get("dataset_id")
    full = _DATASETS.get(dataset_id) if dataset_id else None
    if not full:
        raise RuntimeError(
            f"Dataset {dataset_id or '?'} is no longer available in {_JOB_STORE_STATUS}. "
            "The server may have restarted; reload the file.")
    merged = dict(full)
    # Mapping/UI metadata may have been updated since the original upload.
    merged.update(ref)
    return merged


def _pvpro_new_job(dataset=None):
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    _PVPRO_JOBS[jid] = {"phase": "starting", "current": 0, "total": 1,
                        "message": "Starting PVPRO\u2026", "result": None, "error": None,
                        "started_at": now, "updated_at": now, "worker_pid": os.getpid(),
                        "dataset": dataset or "dataset",
                        "history": [{"at": now, "phase": "starting", "message": "Starting PVPRO\u2026"}]}
    return jid


def _pvpro_set(jid, **kw):
    job = _PVPRO_JOBS.get(jid)
    if job:
        _update_job_record(job, **kw)
        _PVPRO_JOBS[jid] = job


def _update_job_record(job, **kw):
    """Update a job and retain a compact, user-visible phase history."""
    now = time.time()
    old_message = job.get("message")
    old_phase = job.get("phase")
    job.update(kw)
    job["updated_at"] = now
    new_message = job.get("message")
    new_phase = job.get("phase")
    if new_message and (new_message != old_message or new_phase != old_phase):
        history = list(job.get("history") or [])
        history.append({"at": now, "phase": new_phase, "message": new_message,
                        "current": job.get("current"), "total": job.get("total"),
                        "error": job.get("error"), "detail": job.get("detail")})
        job["history"] = history[-1000:]


def _append_pipeline_log(state, stage, message, level="info", job_id=None,
                         worker_pid=None, at=None, detail=None, dataset=None):
    """Persist a compact diagnostic event in the browser-backed app state."""
    logs = list(state.get("pipeline_log") or [])
    event = {"at": float(at or time.time()), "stage": str(stage),
             "message": str(message), "level": str(level),
             "job_id": job_id, "worker_pid": worker_pid or os.getpid(),
             "dataset": dataset or state.get("current_dataset") or "dataset"}
    if detail:
        event["detail"] = str(detail)
    signature = (event["stage"], event["message"], event.get("job_id"), event["level"])
    if not logs or signature != (logs[-1].get("stage"), logs[-1].get("message"),
                                 logs[-1].get("job_id"), logs[-1].get("level")):
        logs.append(event)
    state["pipeline_log"] = logs[-2000:]


def _capture_job_history(state, stage, job_id, job):
    """Copy a job's server-side history into state before its active id is cleared."""
    if not job:
        return
    existing = {(round(float(e.get("at") or 0), 4), e.get("stage"), e.get("message"), e.get("job_id"))
                for e in (state.get("pipeline_log") or [])}
    for event in job.get("history") or []:
        key = (round(float(event.get("at") or 0), 4), stage, event.get("message"), job_id)
        if key in existing:
            continue
        _append_pipeline_log(
            state, stage, event.get("message") or event.get("phase") or "Update",
            level="error" if event.get("error") or event.get("phase") == "error" else event.get("phase") or "info",
            job_id=job_id, worker_pid=job.get("worker_pid"), at=event.get("at"),
            detail=event.get("detail") or event.get("error"), dataset=job.get("dataset"))


def _new_async_job(registry, message, dataset=None):
    """Create a cancellable in-process job record.

    A cancelled worker may still finish a library call that cannot be interrupted,
    but its result is discarded and can never be committed to the Dash stores.
    """
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    registry[jid] = {"phase": "running", "message": message, "result": None,
                     "error": None, "cancelled": False, "current": 0, "total": 1,
                     "started_at": now, "updated_at": now, "worker_pid": os.getpid(),
                     "dataset": dataset or "dataset",
                     "history": [{"at": now, "phase": "running", "message": message}]}
    return jid


def _cancel_async_job(registry, jid):
    job = registry.get(jid)
    if job:
        _update_job_record(job, cancelled=True, phase="cancelled", message="Cancelled by user.")
        registry[jid] = job


def _async_set(registry, jid, **kw):
    job = registry.get(jid)
    if job and not job.get("cancelled"):
        _update_job_record(job, **kw)
        registry[jid] = job


def _missing_job_message(kind, jid, launched_pid):
    if _JOB_CACHE is not None:
        return (f"{kind} job {jid} is absent from the job store on instance {_JOB_INSTANCE}, worker {os.getpid()} "
                f"(launched on worker {launched_pid or '?'}). The job may have expired, the server "
                "may have restarted, or this request reached another container/dyno. Configure REDIS_URL "
                "for cross-container storage, or deploy exactly one container/replica.")
    return (f"{kind} job {jid} is not visible on polling worker {os.getpid()} "
            f"(launched on worker {launched_pid or '?'}). Shared diskcache could not be enabled: "
            f"{_JOB_STORE_STATUS}. Run one web worker or provide shared job storage.")


def _defer_missing_job(state, kind, jid, max_misses=6):
    """Allow brief backend/read races before declaring an async job lost."""
    misses = dict(state.get("job_cache_misses") or {})
    key = f"{kind}:{jid}"
    misses[key] = int(misses.get(key, 0)) + 1
    # Keep the dictionary compact when users process many datasets.
    state["job_cache_misses"] = {key: misses[key]}
    return misses[key] <= max_misses


def _clear_missing_job(state, kind, jid):
    misses = dict(state.get("job_cache_misses") or {})
    misses.pop(f"{kind}:{jid}", None)
    state["job_cache_misses"] = misses

# --------------------------------------------------------------------------- config
EXAMPLES = [
    {"id": "load-example-btn-1", "file": "sys_1278_downsampled_with_VI.parquet",
     "label": "System 1278", "meta": "c-Si · DC V + I · PVPRO-ready"},
    {"id": "load-example-btn-2", "file": "sys_1403_part1_downsampled_with_VI.parquet",
     "label": "System 1403", "meta": "c-Si · DC V + I · PVPRO-ready"},
    {"id": "load-example-btn-3", "file": "sys_1422_downsampled.parquet",
     "label": "System 1422", "meta": "c-Si · power + irradiance"},
]
METHODS = [
    {"id": "YOY", "label": "Year-on-year", "full": "Year-over-Year", "tag": "Recommended"},
    {"id": "LR", "label": "Linear regression", "full": "Linear regression", "tag": "Fast"},
    {"id": "HW", "label": "Holt-Winters", "full": "Holt-Winters", "tag": "Seasonal"},
    {"id": "ARIMA", "label": "ARIMA", "full": "Auto Regressive Integrated Moving Average", "tag": "Seasonal"},
    {"id": "CSD", "label": "Seasonal decomposition", "full": "Classical Seasonal Decomposition", "tag": "Seasonal"},
    {"id": "PVPRO", "label": "PVPRO (single-diode)", "full": "a lightweight in-app implementation", "tag": "Physics · slow"},
]
METHOD_LABEL = {m["id"]: m["label"] for m in METHODS}
STAT_METHODS = ["YOY", "LR", "HW", "ARIMA", "CSD"]

STEP_META = [
    {"num": "01", "title": "Data prescreening", "sub": "Raw signals & quality",
     "desc": "Validate the raw power, irradiance and temperature signals before analysis."},
    {"num": "02", "title": "Intelligent filtering", "sub": "Configure & review",
     "desc": "Tune the filters applied before fitting — all are on by default with best-practice thresholds."},
    {"num": "03", "title": "Degradation model", "sub": "Metrics & calculation",
     "desc": "Pick one or more metrics (and tune their parameters), then run to compare degradation rates."},
    {"num": "04", "title": "Code generation", "sub": "Optional export",
     "desc": "Export a runnable Python script that reproduces your exact pipeline."},
]
FILTER_LABELS = {
    "tz": "Time zone & DST correction",
    "clearsky": "Clear-sky filter",
    "irr": "Low irradiance / power filter",
    "iqr": "Outlier removal (IQR)",
}
FILTER_EXPLAIN = {
    "tz": "Aligns timestamps to local solar time and corrects daylight-saving jumps.",
    "clearsky": "Keeps timestamps whose irradiance profile is smooth and cloud-free, so the fit sees comparable conditions.",
    "irr": "Drops low-light points (irradiance below the threshold) where the power ratio is noisy.",
    "iqr": "Removes statistical outliers in the normalized series using an inter-quartile-range rule.",
}
FILTER_PARAM_DEFAULTS = {"irr_thresh": 300, "clearsky_smooth": 0.3, "clearsky_energy": 0.5, "iqr": 1.5}
# which tunable params each filter exposes (label shown in its "Customize parameters" panel)
FILTER_PARAMS = {
    "tz": [],
    "clearsky": [("clearsky_smooth", "Smoothness threshold"), ("clearsky_energy", "Energy threshold")],
    "irr": [("irr_thresh", "Irradiance threshold (W/m²)")],
    "iqr": [("iqr", "IQR multiplier")],
}
METHOD_PARAM_DEFAULTS = {"yoy_window": 30, "yoy_iqr": 1.5, "hw_period": 12, "csd_period": 12,
                         "arima_p": 1, "arima_d": 1, "arima_q": 0, "arima_s": 12}
PVPRO_PARAM_DEFAULTS = {"cells": 60, "mps": 1, "ps": 1, "alphaisc": 0.0046,
                        "days": 14, "iters": 12, "tech": "mono-c-Si"}
PVPRO_TECHS = ["mono-c-Si", "multi-c-Si", "CdTe", "CIGS", "a-Si"]
# which params each method exposes
METHOD_PARAMS = {
    "YOY": [("yoy_window", "Rolling window (days)"), ("yoy_iqr", "IQR multiplier")],
    "LR": [],
    "HW": [("hw_period", "Seasonal period")],
    "ARIMA": [("arima_p", "p"), ("arima_d", "d"), ("arima_q", "q"), ("arima_s", "Seasonal period")],
    "CSD": [("csd_period", "Seasonal period")],
    "PVPRO": [("cells", "Cells in series")],
}

DEFAULT_STATE = {
    "selected": None, "selected_label": "", "mode": "simple", "simple_done": False,
    "adv": {"1": "idle", "2": "locked", "3": "locked", "4": "locked"}, "adv_tab": 1,
    "filters": {"tz": True, "clearsky": True, "irr": True, "iqr": True},
    "fparams": dict(FILTER_PARAM_DEFAULTS),
    "methods": ["YOY"],
    "mparams": {**METHOD_PARAM_DEFAULTS, **PVPRO_PARAM_DEFAULTS},
    "show_params": False, "pvpro_job": None, "pvpro_prog": None, "pvpro_dur": 0.0, "pvpro_nkept": 0, "diag_job": None,
    "pvpro_window": "", "pvpro_est_keys": [], "simple_method": "YOY", "pvpro_mode": "advanced", "pvpro_fig_sel": "p_mp_ref",
    "filt_open": False, "metric_open": False,
    "ingest_job": None, "ingest_worker_pid": None,
    "analysis_job": None, "analysis_worker_pid": None, "analysis_scope": None,
    "pvpro_worker_pid": None,
    "pipeline_log": [],
    "job_cache_misses": {},
    "current_dataset": "",
}
EMPTY_DATA = {"loaded": False}


def _enforce_adv_sequence(state):
    """Repair the Advanced-mode state into a strict sequential state machine.

    Step N+1 is available only when Step N is done.  This also protects the UI
    from stale browser-store data left by an older callback or interrupted run.
    The supplied state is mutated and returned for convenient use at callback
    and render boundaries.
    """
    adv = state.setdefault("adv", {})
    for key, default in DEFAULT_STATE["adv"].items():
        adv.setdefault(key, default)

    if adv.get("1") != "done":
        adv["1"] = "idle"
        adv["2"] = adv["3"] = adv["4"] = "locked"
        highest = 1
    elif adv.get("2") != "done":
        adv["2"] = "idle"
        adv["3"] = adv["4"] = "locked"
        highest = 2
    elif adv.get("3") not in ("done", "running", "running_async"):
        adv["3"] = "idle"
        adv["4"] = "locked"
        highest = 3
    elif adv.get("3") in ("running", "running_async"):
        adv["4"] = "locked"
        highest = 3
    else:
        if adv.get("4") not in ("idle", "done"):
            adv["4"] = "idle"
        highest = 4

    tab = int(state.get("adv_tab", 1) or 1)
    state["adv_tab"] = tab if 1 <= tab <= highest else highest
    return state


def _reset_advanced_from(state, step):
    """Invalidate `step` and every downstream step after its settings change."""
    adv = state["adv"]
    adv[str(step)] = "idle"
    for n in range(step + 1, 5):
        adv[str(n)] = "locked"
    return _enforce_adv_sequence(state)


# --------------------------------------------------------------------------- helpers
def _rgba(hexstr, a):
    h = hexstr.lstrip("#"); r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _df_to_json(df):
    """Serialize a dataframe compactly for Dash Store/network transport.

    Pandas split JSON is highly repetitive. Sending it verbatim made deployed
    uploads appear stuck for several seconds after the worker had already
    logged "File loaded". A fast level-3 zlib pass normally reduces this by
    80-95%, while the prefix keeps old uncompressed sessions compatible.
    """
    raw = df.to_json(date_format="iso", orient="split").encode("utf-8")
    packed = zlib.compress(raw, level=3)
    if len(packed) + 32 < len(raw):
        return "zlib64:" + base64.b64encode(packed).decode("ascii")
    return raw.decode("utf-8")


def _df_from_json(s):
    if isinstance(s, str) and s.startswith("zlib64:"):
        s = zlib.decompress(base64.b64decode(s[7:])).decode("utf-8")
    df = pd.read_json(s, orient="split")
    df.index = pd.to_datetime(df.index, errors="ignore")
    return df


def _num(v, default):
    try:
        if v is None or v == "":
            return default
        f = float(v)
        return int(f) if float(f).is_integer() else f
    except Exception:
        return default


def deg_str(r):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "\u2014"
    return ("\u2212" if r < 0 else "+") + f"{abs(r):.2f}"


def _svg(markup, size=18):
    import urllib.parse
    return html.Img(src="data:image/svg+xml;utf8," + urllib.parse.quote(markup),
                    style={"width": f"{size}px", "height": f"{size}px", "display": "block"})


def icon_bolt(size=17, color="#ffffff"):
    return _svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">'
               f'<path d="M13 2 4.6 13.4a.6.6 0 0 0 .5 1H10l-1 7.1a.5.5 0 0 0 .9.4l8.6-11.5a.6.6 0 0 0-.5-1H13.9L15 2.6A.5.5 0 0 0 14.1 2z"/></svg>', size)


def icon_cloud(size=26, color="#2f6bff"):
    return _svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
               f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
               f'<path d="M16 16l-4-4-4 4"/><path d="M12 12v8.5"/>'
               f'<path d="M20.4 15A4.2 4.2 0 0 0 18 7.4h-1.3A6.5 6.5 0 1 0 5 15.5"/></svg>', size)


def icon_spark(size=16, color="#ffffff"):
    return _svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">'
               f'<path d="M12 2c.3 3.9 1.9 5.5 5.8 5.8-3.9.3-5.5 1.9-5.8 5.8-.3-3.9-1.9-5.5-5.8-5.8C10.1 7.5 11.7 5.9 12 2z"/>'
               f'<path d="M18.5 13.5c.15 1.9.95 2.7 2.85 2.85-1.9.15-2.7.95-2.85 2.85-.15-1.9-.95-2.7-2.85-2.85 1.9-.15 2.7-.95 2.85-2.85z"/></svg>', size)


def glassify(fig, height=250, title=None, top=None):
    if fig is None:
        return fig
    t = top if top is not None else (30 if title else 10)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=SUB, size=11),
                      margin=dict(l=48, r=16, t=t, b=28), height=height, autosize=True)
    if title is not None:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left",
                                     font=dict(family="Inter", size=14, color=INK)))
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(family="Inter", color=MUTE))
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(family="Inter", color=MUTE))
    return fig


# --------------------------------------------------------------------------- REAL analysis
def load_example_df(example_file):
    return pd.read_parquet(os.path.join("data", example_file))


def load_raw_data(contents=None, filename=None, example_file=None, progress_callback=None):
    """Read bytes into a dataframe without inspecting, mapping, or scoring it."""
    if example_file:
        _report(progress_callback, 1, 2, "Reading example file…")
        df = load_example_df(example_file)
        filename = example_file
    else:
        if not contents or not filename:
            raise ValueError("No file was provided.")
        _report(progress_callback, 1, 2, "Decoding uploaded file…")
        _content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        lower = filename.lower()
        if lower.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(decoded), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(decoded), encoding="latin-1")
        elif lower.endswith((".parquet", ".pq")):
            df = pd.read_parquet(io.BytesIO(decoded))
        elif lower.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise ValueError("Unsupported file type. Upload CSV or Parquet.")
    _report(progress_callback, 2, 3, "Compressing the dataset for browser transfer…")
    packed_df = _df_to_json(df)
    _report(progress_callback, 3, 3,
            f"Dataset transfer payload is {len(packed_df.encode('utf-8')) / 1024:.0f} KB.")
    return {"loaded": True, "identified": False, "df": packed_df,
            "filename": filename or "your_data", "columns": [str(c) for c in df.columns],
            "rows": int(len(df)), "transfer_kb": round(len(packed_df.encode("utf-8")) / 1024, 1)}


def _report(progress_callback, current, total, message):
    if progress_callback:
        progress_callback(current, total, message)


def _build_mapping(df, mapped):
    roles = ["Time", "Irradiance", "DC Power", "Module temperature", "DC Voltage", "DC Current"]
    mapping = []
    for role in roles:
        col = mapped.get(role)
        if col and col in df.columns:
            try:
                tag = _quality_tag(df, col, role) or ""
            except Exception:
                tag = ""
            mapping.append({"role": role, "col": str(col), "tag": tag})
        elif col:
            mapping.append({"role": role, "col": str(col), "tag": ""})
    return mapping


def run_parse(contents=None, filename=None, df=None, progress_callback=None):
    _report(progress_callback, 1, 4, "Identifying signal columns…")
    parser_progress = (lambda message: _report(progress_callback, 1, 4, message)) if progress_callback else None
    df, _summary, mapped, code_read, notes = parse_contents(
        contents=contents, filename=filename, df=df, progress=parser_progress)
    _report(progress_callback, 2, 4, "Validating required signals and identified columns…")
    if df is None or not mapped:
        parse_error = getattr(df, "attrs", {}).get("_parse_error") if df is not None else None
        raise ValueError(parse_error or "Couldn't identify the required columns automatically.")
    irra_key = mapped.get("Irradiance")
    if not irra_key or irra_key not in df.columns:
        raise ValueError("Irradiance column not found — filtering needs an irradiance channel.")
    n = int(len(df))
    completeness = float(df.notna().mean().mean()) * 100.0
    _report(progress_callback, 3, 4, "Computing completeness and column quality…")
    qtags = {}
    for c in df.columns:
        try:
            qtags[str(c)] = _quality_tag(df, c) or ""
        except Exception:
            qtags[str(c)] = ""
    n_cols_total = len(df.columns) + (1 if isinstance(df.index, pd.DatetimeIndex) else 0)
    _report(progress_callback, 4, 4, "Serializing the validated dataset…")
    return {"loaded": True, "identified": True, "df": _df_to_json(df), "mapped": mapped, "irra_key": irra_key,
            "columns": [str(c) for c in df.columns], "n_columns": n_cols_total,
            "detected": dict(mapped), "quality_tags": qtags,
            "filename": filename or "your_data", "n_raw": n, "completeness": round(completeness, 1),
            "mapping": _build_mapping(df, mapped),
            "has_vi": bool(mapped.get("DC Voltage") and mapped.get("DC Current"))}


def apply_filter_chain(df, mapped, irra_key, selected, fparams, progress_callback=None):
    n_raw = int(len(df))
    _report(progress_callback, 1, 7, "Applying basic value and sensor-range checks…")
    bv_normal, _ = basic_value_filter(df, mapped)
    df = df.loc[bv_normal].copy()
    clearsky_mask = pd.Series(True, index=df.index)
    if selected.get("clearsky", True):
        _report(progress_callback, 2, 7, "Detecting clear-sky periods…")
        try:
            cs_idx, _ = clear_sky_filter(df, irra_key,
                                         smoothness_threshold=_num(fparams.get("clearsky_smooth"), 0.3),
                                         energy_threshold=_num(fparams.get("clearsky_energy"), 0.5))
            clearsky_mask = pd.Series(df.index.isin(cs_idx), index=df.index)
        except Exception:
            clearsky_mask = pd.Series(True, index=df.index)
    _report(progress_callback, 3, 7, "Normalizing power and correcting temperature…")
    df_f = normalize(df, mapped, gamma=-0.004)   # temperature correction (always applied)
    mask = pd.Series(clearsky_mask.values, index=df_f.index)
    if selected.get("tz", True):
        _report(progress_callback, 4, 7, "Aligning timezone and daylight-saving timestamps…")
        try:
            df_f.index = pd.to_datetime(df_f.index).tz_localize("UTC").tz_convert("US/Pacific")
            mask.index = df_f.index
        except Exception:
            pass
    if selected.get("irr", True):
        _report(progress_callback, 5, 7, "Removing low-irradiance and low-power points…")
        try:
            keep_idx, _ = low_irra_power_filter(df_f, mapped,
                                                irr_thresh=_num(fparams.get("irr_thresh"), 300),
                                                power_ratio=0.02, norm_lower=0.01, norm_upper_pct=99)
            mask &= df_f.index.isin(keep_idx)
        except Exception:
            pass
    if selected.get("iqr", True):
        _report(progress_callback, 6, 7, "Removing statistical outliers…")
        try:
            ni, _ = identify_outliers_iqr(df_f, "norm", iqr_multiplier=_num(fparams.get("iqr"), 1.5))
            mask &= df_f.index.isin(ni)
        except Exception:
            pass
    kept = mask.values.astype(bool)
    df_good = df_f.loc[df_f.index[kept]]
    n_kept = int(len(df_good))
    _report(progress_callback, 7, 7, "Building filtering summary figures…")
    pie_json, power_json = _filter_figures(df_f, kept, n_raw, n_kept)
    return df_good, n_raw, n_kept, pie_json, power_json


def _filter_figures(df_f, kept, n_raw, n_kept):
    """Donut (retained vs filtered) + normalized-power-over-time scatter coloured by kept/removed."""
    try:
        pie = go.Figure(go.Pie(values=[n_kept, max(n_raw - n_kept, 0)], labels=["High-quality", "Filtered"],
                               hole=0.58, sort=False, direction="clockwise",
                               marker=dict(colors=["#1b5fbf", "#a9c9ef"]),
                               textinfo="percent", textfont=dict(family="Inter", size=12)))
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=True, height=170,
                          margin=dict(l=6, r=6, t=6, b=6), font=dict(family="Inter", color=SUB, size=10),
                          legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)))
    except Exception:
        pie = None
    try:
        idx = pd.to_datetime(df_f.index)
        y = pd.to_numeric(df_f["norm"], errors="coerce").values
        n = len(df_f)
        step = max(1, n // 9000)
        sl = slice(None, None, step)
        xs = idx[sl]; ys = y[sl]; ks = kept[sl]
        power = go.Figure()
        power.add_trace(go.Scattergl(x=xs[~ks], y=ys[~ks], mode="markers", name="Filtered",
                                     marker=dict(color="#a9c9ef", size=3, opacity=0.5)))
        power.add_trace(go.Scattergl(x=xs[ks], y=ys[ks], mode="markers", name="High-quality",
                                     marker=dict(color="#1b5fbf", size=3)))
        power.update_layout(title=dict(text="Normalized power over time", x=0.01, xanchor="left",
                                       font=dict(family="Inter", size=14, color=INK)),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300,
                            margin=dict(l=52, r=16, t=34, b=40), font=dict(family="Inter", color=SUB, size=11),
                            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
        power.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(family="Inter", color=MUTE))
        power.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, title="Normalized power",
                           tickfont=dict(family="Inter", color=MUTE))
        power_json = power.to_json()
    except Exception:
        power_json = None
    return (pie.to_json() if pie is not None else None), power_json


def run_one_metric(daily, method, mparams, df_good=None, mapped=None):
    """Return (rate_pct_per_year, fig) for a single method."""
    if method == "YOY":
        return compute_yoy(daily, rolling_window=_num(mparams.get("yoy_window"), 30),
                           iqr_multiplier=_num(mparams.get("yoy_iqr"), 1.5))
    if method == "LR":
        return compute_lr(daily)
    if method == "HW":
        return compute_hw(daily, period=_num(mparams.get("hw_period"), 12))
    if method == "ARIMA":
        return compute_arima(daily, p=_num(mparams.get("arima_p"), 1), d=_num(mparams.get("arima_d"), 1),
                             q=_num(mparams.get("arima_q"), 0), seasonal_period=_num(mparams.get("arima_s"), 12))
    if method == "CSD":
        return compute_csd(daily, period=_num(mparams.get("csd_period"), 12))
    if method == "PVPRO":
        kwargs = dict(
            cells_in_series=_num(mparams.get("cells"), 60),
            modules_per_string=_num(mparams.get("mps"), 1),
            parallel_strings=_num(mparams.get("ps"), 1),
            alpha_isc=_num(mparams.get("alphaisc"), 0.0046),
            technology=mparams.get("tech") or "mono-c-Si",
            days_per_run=_num(mparams.get("days"), 14),
            iterations_per_year=_num(mparams.get("iters"), 12),
        )
        rd, figs, rates = compute_pvpro(df_good, mapped or {}, **kwargs)
        fig = None
        if isinstance(figs, dict) and figs:
            fig = figs.get("p_mp_ref") or figs.get("p_mp") or next(iter(figs.values()))
        elif isinstance(figs, (list, tuple)) and figs:
            fig = figs[0]
        elif figs is not None:
            fig = figs
        return float(rd), fig
    raise ValueError(f"Unknown metric: {method}")


def run_methods(df_good, irra_key, methods, mparams, mapped=None, progress_callback=None):
    """Run every selected statistical method; return {method: {rate, fig}}."""
    total = max(1, len(methods) + 1)
    _report(progress_callback, 1, total, "Aggregating the filtered time series…")
    daily = aggregate_daily(df_good, irra_key)
    out = {}
    for pos, m in enumerate(methods, start=2):
        _report(progress_callback, pos, total, f"Running {METHOD_LABEL.get(m, m)}…")
        try:
            rate, fig = run_one_metric(daily, m, mparams, df_good=df_good, mapped=mapped)
            out[m] = {"rate": float(rate) if rate == rate else None,
                      "fig": fig.to_json() if fig is not None else None}
        except Exception as e:
            traceback.print_exc()
            out[m] = {"rate": None, "fig": None, "error": str(e)}
    return out


def _launch_pvpro(dg, mapped, kwargs, dataset=None):
    """Start PVPRO in a background thread; progress is written to _PVPRO_JOBS[jid]."""
    jid = _pvpro_new_job(dataset=dataset)

    def worker():
        try:
            mapped_summary = ", ".join(
                f"{role}={mapped.get(role)}" for role in
                ("DC Voltage", "DC Current", "Irradiance", "Module temperature"))
            try:
                date_span = f"{pd.Timestamp(dg.index.min()):%Y-%m-%d} to {pd.Timestamp(dg.index.max()):%Y-%m-%d}"
            except Exception:
                date_span = "unknown"
            _pvpro_set(
                jid, phase="prepare",
                message=(f"PVPRO input: {len(dg):,} rows; {date_span}; {mapped_summary}; "
                         f"cells={kwargs.get('cells_in_series')}, modules/string={kwargs.get('modules_per_string')}, "
                         f"parallel strings={kwargs.get('parallel_strings')}, days/window={kwargs.get('days_per_run')}, "
                         f"windows/year={kwargs.get('iterations_per_year')}, technology={kwargs.get('technology')}"))
            def cb(stage, current, total, message):
                # compute_pvpro reports its own "done" immediately before it
                # returns. At that point this wrapper still has to serialize
                # rates and figures into job["result"]. Never expose that
                # intermediate signal as the job's terminal phase, otherwise a
                # poll can commit an empty result in this small race window.
                public_stage = "finalizing" if stage == "done" else stage
                public_message = ("Finalizing PVPRO rates and figures…"
                                  if stage == "done" else message)
                _pvpro_set(jid, phase=public_stage, current=current, total=total,
                           message=public_message)
            started = time.time()
            rd, figs, rates = compute_pvpro(dg, mapped or {}, progress_callback=cb, **kwargs)
            figs_json = {}
            if isinstance(figs, dict):
                for k, v in figs.items():
                    try:
                        figs_json[str(k)] = go.Figure(v).to_json()
                    except Exception:
                        pass
            rates_json = {}
            if isinstance(rates, dict):
                for k, v in rates.items():
                    try:
                        rates_json[str(k)] = float(v)
                    except Exception:
                        pass
            _pvpro_set(jid, phase="done",
                       message=(f"PVPRO completed in {time.time() - started:.1f}s; "
                                f"headline rate={float(rd):+.4f}%/yr; figures={len(figs_json)}; rates={len(rates_json)}."),
                       current=1, total=1,
                       result={"rate": float(rd), "figs": figs_json, "rates": rates_json})
        except Exception as e:
            traceback.print_exc()
            _pvpro_set(jid, phase="error",
                       message=f"PVPRO failed: {type(e).__name__}: {e}", error=str(e),
                       detail=traceback.format_exc(limit=20))

    threading.Thread(target=worker, daemon=True).start()
    return jid


def _launch_ingest(contents=None, filename=None, example_file=None, dataset=None):
    jid = _new_async_job(_INGEST_JOBS, "Reading the data file…", dataset=dataset)

    def worker():
        try:
            progress = lambda current, total, message: _async_set(
                _INGEST_JOBS, jid, phase="running", current=current, total=total, message=message)
            raw = load_raw_data(contents=contents, filename=filename, example_file=example_file,
                                progress_callback=progress)
            browser_data = _store_dataset(raw)
            _async_set(
                _INGEST_JOBS, jid, phase="done", current=3, total=3,
                message=(f"File loaded: {int(raw.get('rows', 0)):,} rows; compact browser payload "
                         "contains metadata only. No analysis has been run."),
                result=browser_data)
        except Exception as exc:
            traceback.print_exc()
            _async_set(_INGEST_JOBS, jid, phase="error", message="Data loading failed.", error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return jid


def _launch_analysis(kind, work, dataset=None):
    labels = {
        "simple_yoy": "Running the simple analysis…",
        "simple_prepare_pvpro": "Preparing the dataset for PVPRO…",
        "advanced_1": "Screening raw signals…",
        "advanced_2": "Applying filters…",
        "advanced_3": "Calculating degradation…",
    }
    jid = _new_async_job(_ANALYSIS_JOBS, labels.get(kind, "Analyzing…"), dataset=dataset)
    job = _ANALYSIS_JOBS.get(jid)
    job["kind"] = kind
    _ANALYSIS_JOBS[jid] = job

    def worker():
        try:
            progress = lambda current, total, message: _async_set(
                _ANALYSIS_JOBS, jid, phase="running", current=current, total=total, message=message)
            progress(0, 1, labels.get(kind, "Preparing analysis…"))
            payload = work(progress)
            _async_set(_ANALYSIS_JOBS, jid, phase="done", current=1, total=1,
                       message="Step completed successfully.", result=payload)
        except Exception as exc:
            traceback.print_exc()
            _async_set(_ANALYSIS_JOBS, jid, phase="error",
                       message=f"Step failed: {type(exc).__name__}: {exc}", error=str(exc),
                       detail=traceback.format_exc(limit=20))

    threading.Thread(target=worker, daemon=True).start()
    return jid


def _close_button(action, label="Stop this process"):
    return html.Button("×", id={"type": "act", "index": action}, n_clicks=0,
                       title=label, **{"aria-label": label}, className="pvc-busy-close")


def _busy_overlay(text, action, element_id=None, upload=False):
    return html.Div(id=element_id,
                    className="pvc-busy" + (" pvc-upload-busy" if upload else " show"), children=[
        _close_button(action),
        html.Div(className="pvc-spinner"),
        html.Div(text, id=("upload-text" if upload else "busy-text"), className="pvc-busy-text")])


# --------------------------------------------------------------------------- shared UI
def logo_mark():
    return html.Div("\u259a", style={"width": "42px", "height": "42px", "borderRadius": "13px",
        "background": "linear-gradient(135deg,#4b8bff,#2f6bff)", "color": "#fff", "display": "flex",
        "alignItems": "center", "justifyContent": "center", "fontSize": "20px",
        "boxShadow": "0 8px 20px rgba(47,107,255,0.35)"})


def nav_pills():
    def pill(label, key):
        return html.Button(label, id={"type": "nav", "index": key}, n_clicks=0, className="nav-pill")
    return html.Div(className="nav-pills", children=[
        pill("What's new", "whatsnew"), pill("Team", "team"),
        pill("How to cite", "cite"), pill("Methods", "methods")])


def example_cards(state):
    cards = []
    for ex in EXAMPLES:
        sel = state["selected"] == ex["id"]
        cards.append(html.Button(id={"type": "example", "index": ex["id"]}, n_clicks=0, children=[
            html.Div("\u2713" if sel else "\u25a4", style={"width": "36px", "height": "36px", "flexShrink": 0,
                     "borderRadius": "11px", "display": "flex", "alignItems": "center", "justifyContent": "center",
                     "background": "linear-gradient(135deg,#4b8bff,#2f6bff)" if sel else "rgba(79,139,255,0.12)",
                     "color": "#fff" if sel else BLUE, "fontWeight": 700}),
            html.Div([html.Div(ex["label"], style={"fontSize": "14px", "fontWeight": 700, "color": INK}),
                      html.Div(ex["meta"], style={"fontSize": "11.5px", "color": SUB})], style={"minWidth": 0})],
            style={"display": "flex", "alignItems": "center", "gap": "12px", "padding": "14px 15px",
                   "textAlign": "left", "borderRadius": "16px", "width": "100%", "cursor": "pointer", "flex": 1,
                   "border": ("1.5px solid #4f8bff" if sel else "1px solid rgba(255,255,255,0.7)"),
                   "background": ("linear-gradient(135deg,rgba(79,139,255,0.16),rgba(79,139,255,0.06))"
                                  if sel else "rgba(255,255,255,0.42)")}))
    return cards


_MAP_ROLES = [("Time", "time"), ("Irradiance", "irradiance"), ("DC Power", "power"),
              ("Module temperature", "temperature"), ("DC Voltage", "voltage"), ("DC Current", "current")]
_REQUIRED_ROLES = {"Time", "DC Power"}


def mapping_editor(data):
    cols = data.get("columns") or []
    mapped = data.get("mapped") or {}
    detected = data.get("detected") or mapped
    qtags = data.get("quality_tags") or {}

    def _item(c):
        if c == "__index__":
            return {"value": c, "label": "(use index / __index__)", "quality": ""}
        return {"value": c, "label": c, "quality": qtags.get(c) or ""}

    rows = []
    for role, noun in _MAP_ROLES:
        current = mapped.get(role)
        det = detected.get(role)
        valid = []
        if role == "Time" and (current == "__index__" or det == "__index__"):
            valid.append("__index__")
        for c in (det, current):
            if c and c != "__index__" and c not in valid:
                valid.append(c)
        rest = [c for c in cols if c not in valid]
        ddata = []
        if valid:
            ddata.append({"group": "LLM-detected " + noun, "items": [_item(c) for c in valid]})
        if rest:
            ddata.append({"group": "Other columns", "items": [_item(c) for c in rest]})

        dot = "#16a34a" if current else ("#dc2626" if role in _REQUIRED_ROLES else "#a1a1aa")
        selq = qtags.get(current) if current else ""
        neutral = bool(selq) and (selq.lower().startswith("one ") or "per-device" in selq.lower())
        pill_base = {"position": "absolute", "right": "58px", "top": "50%", "transform": "translateY(-50%)",
                     "pointerEvents": "none", "zIndex": 5, "fontSize": "10.5px", "fontWeight": 600, "lineHeight": "1",
                     "borderRadius": "980px", "padding": "3px 8px", "whiteSpace": "nowrap"}
        if selq:
            pill_style = {**pill_base, "display": "inline-block",
                          "color": "#57606a" if neutral else "#8a6d00",
                          "background": "#f1f3f5" if neutral else "#fff6e0",
                          "border": "1px solid " + ("#d7dce0" if neutral else "#f0dfa8")}
            pill_text = ("Note: " if neutral else "Warning: ") + selq
        else:
            pill_style = {"display": "none"}
            pill_text = ""
        overlay = html.Span(pill_text, id={"type": "mappill", "index": role}, style=pill_style)
        # required-but-unselected warning shown under the box (toggled live clientside)
        req = role in _REQUIRED_ROLES
        hint_shown = req and not current
        hint = html.Div("\u26a0 Required for the degradation fit — select a column.",
                        id={"type": "maphint", "index": role},
                        style={"display": "block" if hint_shown else "none", "color": "#b23", "fontSize": "11.5px",
                               "fontWeight": 600, "marginTop": "6px"})
        rows.append(html.Div(style={"padding": "6px 2px"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "8px"}, children=[
                html.Span(id={"type": "mapdot", "index": role},
                          style={"width": "9px", "height": "9px", "borderRadius": "50%", "background": dot, "flex": "0 0 auto"}),
                html.Span(role, style={"fontSize": "13px", "fontWeight": 700, "color": SUB})]),
            html.Div(style={"position": "relative"}, children=[
                dmc.Select(id={"type": "mapsel", "index": role}, data=ddata, value=current or None,
                           placeholder="\u2014 select \u2014", clearable=True, searchable=True, w="100%", size="sm",
                           styles={"input": {"fontWeight": 700 if current else 400}},
                           renderOption={"function": "renderVarMapOption"},
                           classNames={"groupLabel": "pvcopilot-var-group"},
                           comboboxProps={"withinPortal": True, "zIndex": 3000}),
                overlay]),
            hint]))
    return html.Div(html.Div(rows, className="pvc-two-col-grid", style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))",
                                           "gap": "14px 18px", "alignItems": "start"}),
                    style={"background": "rgba(255,255,255,0.62)", "border": "1px solid rgba(255,255,255,0.7)",
                           "borderRadius": "18px", "padding": "18px 20px"})


def mapping_chips(data):
    rows = data.get("mapping") or []
    if not rows:
        return html.Div()
    chips = []
    for r in rows:
        tag = r.get("tag") or ""
        pill = None
        if tag:
            neutral = tag.lower().startswith("one ") or "per-device" in tag.lower()
            pill = html.Span(("Note: " if neutral else "Warning: ") + tag,
                             className="pill " + ("pill-note" if neutral else "pill-warn"))
        chips.append(html.Div(className="varmap-chip", children=[
            html.Span(r["role"], style={"flex": "0 0 128px", "fontSize": "12px", "fontWeight": 700, "color": SUB}),
            html.Span(r["col"], style={"flex": 1, "minWidth": 0, "fontSize": "13px", "fontWeight": 700, "color": INK,
                      "fontFamily": "'JetBrains Mono',monospace", "overflow": "hidden",
                      "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
            pill if pill else html.Span()]))
    return html.Div(className="pvc-two-col-grid", style={"display": "grid", "gridTemplateColumns": "repeat(2,1fr)", "gap": "8px"}, children=chips)


def landing_card():
    return html.Div(className="glass pvc-landing", style={"padding": "30px 40px 32px", "marginBottom": "20px", "position": "relative"}, children=[
        html.Div(className="pvc-landing-nav", style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "gap": "16px",
                        "flexWrap": "wrap", "marginBottom": "20px"}, children=[
            html.Div("\u2726  LLM-empowered PV degradation pipeline", style={"display": "inline-flex",
                     "alignItems": "center", "gap": "8px", "padding": "6px 14px", "borderRadius": "20px",
                     "background": "rgba(79,139,255,0.12)", "border": "1px solid rgba(79,139,255,0.22)",
                     "fontSize": "12.5px", "fontWeight": 600, "color": BLUE}),
            nav_pills()]),
        html.H1("PV Copilot: Agentic PV Degradation Analysis", className="pvc-hero-title", style={"margin": "0 0 12px", "fontSize": "42px", "lineHeight": "1.05",
                "fontWeight": 800, "letterSpacing": "-0.03em", "color": INK}),
        html.P("Upload a PV time-series — Copilot screens, filters and computes the degradation rate end-to-end.",
               style={"margin": "0 0 24px", "fontSize": "16px", "lineHeight": "1.5", "color": SUB}),
        html.Div(className="pvc-upload-grid", style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "20px", "alignItems": "stretch"}, children=[
            html.Div(className="pvc-upload-host", style={"position": "relative", "display": "flex", "flexDirection": "column"}, children=[
                html.Button("Data requirements", id={"type": "nav", "index": "datareq"}, n_clicks=0, style={
                    "position": "absolute", "top": "12px", "left": "12px", "zIndex": 2, "padding": "6px 12px",
                    "borderRadius": "980px", "border": "1px solid rgba(79,139,255,0.3)", "background": "rgba(255,255,255,0.8)",
                    "color": BLUE, "fontWeight": 600, "fontSize": "12px", "cursor": "pointer"}),
                dcc.Upload(id="upload-data", className="pvc-upload", multiple=False, children=html.Div(style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "height": "100%"}, children=[
                    html.Div(icon_cloud(26), className="pvc-up-icon", style={"width": "58px", "height": "58px", "borderRadius": "18px", "margin": "0 auto 14px",
                             "background": "linear-gradient(135deg,rgba(120,170,255,0.35),rgba(255,224,150,0.35))",
                             "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                    html.Div("Drag & drop your data, or click to browse", style={"fontSize": "17px", "fontWeight": 700, "marginBottom": "5px"}),
                    html.Div("CSV \u00b7 Parquet \u00b7 one file \u00b7 timestamp + power / energy / irradiance columns", style={"fontSize": "13.5px", "color": "#6b7794"}),
                ]), style={"cursor": "pointer", "border": "2px dashed rgba(79,139,255,0.4)", "borderRadius": "22px",
                           "background": "rgba(255,255,255,0.4)", "padding": "34px 28px", "textAlign": "center", "flex": 1, "minHeight": "230px"}),
                _busy_overlay("Loading data\u2026", "stop-ingest", element_id="upload-overlay", upload=True)]),
            html.Div(style={"display": "flex", "flexDirection": "column"}, children=[
                html.Div("OR START WITH EXAMPLE DATA", style={"fontSize": "12px", "fontWeight": 600, "color": MUTE,
                         "letterSpacing": "0.04em", "marginBottom": "10px"}),
                html.Div(id="example-row", style={"display": "flex", "flexDirection": "column", "gap": "10px", "flex": 1})])]),
    ])


def home_body(state=None):
    return html.Div(className="pvcopilot-shell", children=[
        landing_card(),
        html.Div(id="ingest-status", style={"minHeight": "2px"}),
        html.Div(id="workflow-slot"),
    ])


# --------------------------------------------------------------------------- workflow shell
def mode_toggle(state):
    def tab(label, glyph, key):
        active = state["mode"] == key
        icon = icon_bolt(15, "#fff" if active else SUB) if key == "simple" else html.Span(glyph)
        return html.Button([icon, html.Span(label)], id={"type": "mode", "index": key}, n_clicks=0, style={
            "display": "inline-flex", "alignItems": "center", "gap": "7px", "padding": "9px 18px", "border": "none",
            "borderRadius": "11px", "fontWeight": 700 if active else 600, "fontSize": "14px", "cursor": "pointer",
            "color": "#fff" if active else SUB,
            "background": "linear-gradient(135deg,#4b8bff,#2f6bff)" if active else "transparent",
            "boxShadow": "0 6px 16px rgba(47,107,255,0.32)" if active else "none"})
    return html.Div(style={"display": "flex", "alignItems": "center", "gap": "5px", "padding": "5px",
                           "borderRadius": "15px", "background": "rgba(120,140,180,0.12)",
                           "border": "1px solid rgba(255,255,255,0.6)"},
                    children=[tab("Simple", "\u26a1", "simple"), tab("Advanced", "\u2699", "advanced")])


def workflow_layout(state, data, filtered, result):
    if not state["selected"] or not data.get("loaded"):
        return html.Div()
    header = html.Div(className="pvc-workflow-header", style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                             "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
            html.Div("\u2713", style={"width": "34px", "height": "34px", "borderRadius": "11px",
                     "background": "rgba(52,199,140,0.16)", "border": "1px solid rgba(52,199,140,0.3)",
                     "color": "#1a8f60", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": 800}),
            html.Div(state["selected_label"] or data.get("filename") or "Selected data file",
                     style={"fontSize": "16px", "fontWeight": 800})]),
        mode_toggle(state)])
    body = simple_body(state, data, result) if state["mode"] == "simple" else advanced_body(state, data, filtered, result)
    return html.Div(className="glass rise pvc-workflow", style={"padding": "26px 30px 30px", "position": "relative"}, children=[
        header, body])


def stat_chip(children):
    kids = children if isinstance(children, list) else [children]
    return html.Div([html.Span("\u2713", style={"marginRight": "8px", "color": "#1aa06e"}), *kids],
                    style={"display": "flex", "alignItems": "center", "fontSize": "13px", "color": "#374466"})


def _simple_method_toggle(state, data):
    def opt(label, key, sub):
        active = state.get("simple_method", "YOY") == key
        base = icon_bolt(15, "#fff" if active else BLUE) if key == "YOY" else icon_spark(15, "#fff" if active else BLUE)
        icon = html.Span("\u2713", style={"fontWeight": 800, "fontSize": "18px", "color": "#fff"}) if active else base
        return html.Button([
            html.Div(icon, style={"width": "36px", "height": "36px", "flexShrink": 0, "borderRadius": "11px",
                     "display": "flex", "alignItems": "center", "justifyContent": "center",
                     "background": "linear-gradient(135deg,#4b8bff,#2f6bff)" if active else "rgba(79,139,255,0.12)"}),
            html.Div([html.Div(label, style={"fontSize": "14px", "fontWeight": 700, "color": INK}),
                      html.Div(sub, style={"fontSize": "11.5px", "color": SUB})], style={"minWidth": 0, "flex": 1}),
            (html.Img(src=app.get_asset_url("pvpro_logo.png"), alt="PV-Pro",
                      style={"height": "20px", "flexShrink": 0, "opacity": 0.95}) if key == "PVPRO" else html.Div())],
            id={"type": "smpmethod", "index": key}, n_clicks=0, style={
            "display": "flex", "alignItems": "center", "gap": "12px", "padding": "14px 15px", "flex": 1,
            "textAlign": "left", "borderRadius": "16px", "cursor": "pointer",
            "border": ("1.5px solid #4f8bff" if active else "1px solid rgba(255,255,255,0.7)"),
            "background": ("linear-gradient(135deg,rgba(79,139,255,0.16),rgba(79,139,255,0.06))"
                          if active else "rgba(255,255,255,0.42)")})
    return html.Div([opt("Year-on-year", "YOY", "Fast best-practice trend fit"),
                     opt("PVPRO", "PVPRO", "Physics single-diode model")],
                    className="pvc-simple-methods",
                    style={"display": "flex", "gap": "12px", "maxWidth": "560px", "margin": "0 auto"})


def _simple_pvpro_settings(state, data):
    identified = bool(data.get("identified"))
    needs_vi = identified and not data.get("has_vi")
    bullet = lambda t: html.Li(t, style={"fontSize": "12.5px", "color": SUB, "lineHeight": "1.5", "marginBottom": "3px"})
    return html.Div([
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "8px"}, children=[
            html.Img(src=app.get_asset_url("pvpro_logo.png"), alt="PV-Pro", style={"height": "22px"}),
            html.Span("Single-diode model fitting", style={"fontSize": "13px", "fontWeight": 700, "color": INK})]),
        html.Ul([
            bullet(["Reveals ", html.B("single-diode-model parameter trends"), " (", html.B("Pmp, Voc, Isc"), ", etc.) \u00b7 ", html.B("~1\u20133 min")]),
            bullet(["Requires ", html.B("DC voltage"), " and ", html.B("DC current"), " columns"]),
            bullet(["Reference: ", html.A("PVPRO on GitHub", href="https://github.com/DuraMAT/pvpro",
                    target="_blank", style={"color": BLUE, "fontWeight": 600}), "  \u00b7  ",
                    html.A("paper (Solar Energy, 2023)", href="https://www.sciencedirect.com/science/article/pii/S0038092X23001573",
                    target="_blank", style={"color": BLUE, "fontWeight": 600})])],
            style={"margin": "0 0 8px", "paddingLeft": "20px"}),
        (html.Div("\u26a0 DC Voltage + DC Current weren't identified in this dataset \u2014 PVPRO can't run.",
                  style={"fontSize": "12.5px", "color": "#8a6d00", "fontWeight": 600, "marginBottom": "8px"}) if needs_vi else
         html.Div("Signal availability will be checked when analysis starts.",
                  style={"fontSize": "12.5px", "color": SUB, "marginBottom": "8px"}) if not identified else html.Div()),
        html.Details([
            html.Summary("Customize parameters", style={"cursor": "pointer", "fontSize": "12.5px", "fontWeight": 600, "color": BLUE, "padding": "4px 0"}),
            (html.Div(html.Button([icon_spark(13, BLUE), html.Span("Estimate from data")], id={"type": "act", "index": "estimate-pvpro"}, n_clicks=0, style={
                "display": "inline-flex", "alignItems": "center", "gap": "7px", "padding": "8px 14px", "marginTop": "8px",
                "border": "1px solid rgba(79,139,255,0.5)", "borderRadius": "980px", "background": "rgba(255,255,255,0.7)",
                "color": BLUE, "fontWeight": 600, "fontSize": "12px", "cursor": "pointer"})) if identified else html.Div()),
            html.Div([
                _num_input("mparam", "cells", state["mparams"].get("cells", 60), "Cells in series"),
                _num_input("mparam", "mps", state["mparams"].get("mps", 1), "Modules per string", dot=("mps" in state.get("pvpro_est_keys", []))),
                _num_input("mparam", "ps", state["mparams"].get("ps", 1), "Parallel strings", dot=("ps" in state.get("pvpro_est_keys", []))),
                _num_input("mparam", "alphaisc", state["mparams"].get("alphaisc", 0.0046), "\u03b1_Isc (/\u00b0C)"),
                _num_input("mparam", "days", state["mparams"].get("days", 14), "Days per window"),
                _num_input("mparam", "iters", state["mparams"].get("iters", 12), "Windows per year"),
                html.Div([html.Label("Technology", style={"fontSize": "11.5px", "fontWeight": 600, "color": SUB}),
                          dcc.Dropdown(id={"type": "mparam", "index": "tech"}, className="pvc-dd",
                                       options=[{"label": t, "value": t} for t in PVPRO_TECHS],
                                       value=state["mparams"].get("tech", "mono-c-Si"), clearable=False)],
                         style={"display": "flex", "flexDirection": "column", "gap": "4px"})],
                className="pvc-form-grid", style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "10px", "padding": "10px 0 0"})])],
        className="pvc-simple-settings", style={"marginTop": "16px", "maxWidth": "560px", "marginLeft": "auto", "marginRight": "auto",
               "background": "rgba(255,255,255,0.6)", "border": "1px solid rgba(255,255,255,0.7)",
               "borderRadius": "16px", "padding": "16px 18px", "textAlign": "left"})


def simple_body(state, data, result):
    method = state.get("simple_method", "YOY")
    if state.get("analysis_job") and state.get("analysis_scope") == "simple":
        resting = dict(state)
        resting["analysis_job"] = None
        resting["analysis_scope"] = None
        return html.Div(className="pvc-busy-host", children=[
            simple_body(resting, data, result),
            _busy_overlay("Analyzing…", "stop-analysis", element_id="busy-overlay")])

    # PVPRO fit in progress (simple mode)
    if state.get("pvpro_job") and state.get("pvpro_mode") == "simple":
        prog = state.get("pvpro_prog") or {}
        cur, tot = prog.get("current", 0) or 0, prog.get("total", 1) or 1
        pct = min(100, max(4, (cur / tot * 100) if tot else 5))
        return html.Div(className="glass-soft", style={"padding": "28px 26px", "background": "rgba(255,255,255,0.75)", "position": "relative"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "6px"}, children=[
                html.Span(style={"width": "10px", "height": "10px", "borderRadius": "50%", "background": "#e08a2b",
                          "animation": "pvc-pulseDot 1.1s ease-in-out infinite"}),
                html.Div("Fitting PVPRO (single-diode model)\u2026", style={"fontSize": "16px", "fontWeight": 800, "color": INK})]),
            html.Div(prog.get("message", "Working\u2026"), id="pvc-inline-pvpro-message",
                     style={"fontSize": "13px", "color": SUB, "marginBottom": "16px"}),
            html.Div(style={"height": "10px", "borderRadius": "980px", "background": "rgba(120,140,180,0.2)", "overflow": "hidden"}, children=[
                html.Div(id="pvc-inline-pvpro-fill", style={"height": "100%", "width": f"{pct:.0f}%", "borderRadius": "980px",
                         "background": "linear-gradient(90deg,#f0b45a,#e08a2b)", "transition": "width .4s ease"})]),
            html.Div(f"window {cur} / {tot} \u00b7 {prog.get('phase', '')}", id="pvc-inline-pvpro-meta",
                     style={"fontSize": "12px", "color": MUTE, "marginTop": "8px", "fontFamily": "'JetBrains Mono',monospace"}),
            _close_button("stop-pvpro", "Stop PVPRO")])

    # PVPRO result (simple mode) -> reuse the shared result view
    multi = (result or {}).get("multi") or {}
    if state.get("simple_done") and "PVPRO" in multi:
        return html.Div(className="rise", children=[
            result_cards(result, state.get("pvpro_fig_sel", "p_mp_ref")),
            diagnostics_block(result),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "marginTop": "18px"}, children=[
                html.Button([html.Span("\u2190", style={"fontSize": "16px", "fontWeight": 800}), html.Span("Return")], id={"type": "act", "index": "simple-reset"}, n_clicks=0, style={
                    "display": "inline-flex", "alignItems": "center", "gap": "7px", "padding": "11px 18px",
                    "border": "1px solid rgba(120,140,180,0.4)", "borderRadius": "13px", "background": "rgba(255,255,255,0.6)",
                    "color": "#374466", "fontWeight": 600, "fontSize": "13.5px", "cursor": "pointer"}),
                html.Button(["Open Advanced mode to tune every step  \u2192"], id={"type": "act", "index": "open-advanced"},
                            n_clicks=0, style={"padding": 0, "border": "none", "background": "none", "color": BLUE,
                                               "fontWeight": 600, "fontSize": "13.5px", "cursor": "pointer"})])])

    smp = (result or {}).get("simple")
    if not state["simple_done"] or not smp:
        return html.Div(className="pvc-simple-start", style={"borderRadius": "22px", "padding": "34px 40px", "textAlign": "center",
                               "background": "linear-gradient(135deg,rgba(79,139,255,0.08),rgba(255,224,150,0.12))",
                               "border": "1px solid rgba(255,255,255,0.7)"}, children=[
            html.Div("One click, full pipeline.", style={"fontSize": "20px", "fontWeight": 800, "marginBottom": "8px"}),
            html.P("PV Copilot runs pre-screening and filtering with best-practice defaults, then fits the "
                   "degradation rate with your chosen method.",
                   style={"margin": "0 auto 20px", "fontSize": "14.5px", "lineHeight": "1.55", "color": SUB, "maxWidth": "480px"}),
            _simple_method_toggle(state, data),
            (_simple_pvpro_settings(state, data) if method == "PVPRO" else html.Div()),
            html.Button([icon_bolt(16), html.Span("Run analysis")], id={"type": "act", "index": "run-simple"}, n_clicks=0, className="btn-run",
                        style={"padding": "15px 28px", "fontSize": "15.5px", "marginTop": "22px"})])
    if smp.get("error"):
        return html.Div([
            html.Div("Analysis failed", style={"fontSize": "16px", "fontWeight": 800, "marginBottom": "6px", "color": "#b23"}),
            html.Div(str(smp["error"]), style={"fontSize": "13px", "color": SUB, "marginBottom": "16px"}),
            html.Button([icon_bolt(15), html.Span("Try again")], id={"type": "act", "index": "run-simple"}, n_clicks=0, className="btn-run")])
    rate = smp.get("rate")
    fig = pio.from_json(smp["fig"]) if smp.get("fig") else None
    rate_card = html.Div(style={"borderRadius": "20px", "padding": "26px", "display": "flex", "flexDirection": "column",
                                "background": "linear-gradient(160deg,rgba(79,139,255,0.12),rgba(255,224,150,0.16))",
                                "border": "1px solid rgba(255,255,255,0.7)"}, children=[
        html.Div("PERFORMANCE LOSS RATE", style={"fontSize": "12px", "fontWeight": 600, "letterSpacing": "0.04em", "color": MUTE, "marginBottom": "8px"}),
        html.Div([html.Span(deg_str(rate), style={"fontSize": "52px", "fontWeight": 900, "letterSpacing": "-0.03em", "color": INK}),
                  html.Span(" %/yr", style={"fontSize": "20px", "fontWeight": 700, "color": SUB, "marginLeft": "8px"})],
                 style={"display": "flex", "alignItems": "baseline"}),
        html.Div(f"Year-on-year · {smp.get('duration_years', 0):.1f}-year record", style={"fontSize": "13px", "color": SUB, "marginTop": "8px"}),
        html.Div(style={"flex": 1}),
        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "20px"}, children=[
            stat_chip(["Pre-screening \u00b7 ", html.B(f"{data.get('completeness', 0)}%"), " complete"]),
            stat_chip(["Filtering \u00b7 ", html.B(f"{smp.get('pct_kept', 0):.0f}%"), " points retained"]),
            stat_chip(["Model \u00b7 ", html.B(f"{smp.get('n_kept', 0):,}"), " points fitted"])])])
    if fig is not None:
        fig.update_layout(title=None)
        fig = glassify(fig, height=260, top=10)
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", font=dict(size=11)),
                          margin=dict(l=52, r=150, t=10, b=30))
    trend_card = html.Div(className="glass-soft", style={"padding": "22px 24px", "background": "rgba(255,255,255,0.6)"}, children=[
        html.Div("Normalized power trend", style={"fontSize": "15px", "fontWeight": 700, "marginBottom": "8px"}),
        dcc.Graph(figure=fig, config=GRAPH_CONFIG) if fig is not None
        else html.Div("No trend figure for this metric.", style={"color": SUB, "fontSize": "13px"})])
    return html.Div(className="rise", children=[
        html.Div(className="pvc-result-grid", style={"display": "grid", "gridTemplateColumns": "300px 1fr", "gap": "20px", "alignItems": "stretch"},
                 children=[rate_card, trend_card]),
        diagnostics_block(result),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "marginTop": "18px"}, children=[
            html.Button([html.Span("\u2190", style={"fontSize": "16px", "fontWeight": 800}), html.Span("Return")], id={"type": "act", "index": "simple-reset"}, n_clicks=0, style={
                "display": "inline-flex", "alignItems": "center", "gap": "7px", "padding": "11px 18px",
                "border": "1px solid rgba(120,140,180,0.4)", "borderRadius": "13px", "background": "rgba(255,255,255,0.6)",
                "color": "#374466", "fontWeight": 600, "fontSize": "13.5px", "cursor": "pointer"}),
            html.Button(["Open Advanced mode to tune every step  \u2192"], id={"type": "act", "index": "open-advanced"},
                        n_clicks=0, style={"padding": 0, "border": "none", "background": "none", "color": BLUE,
                                           "fontWeight": 600, "fontSize": "13.5px", "cursor": "pointer"})])])


# --------------------------------------------------------------------------- advanced mode
def rail(state):
    state = dict(state)
    state["adv"] = dict(state.get("adv") or {})
    _enforce_adv_sequence(state)
    items = []
    for i, meta in enumerate(STEP_META, start=1):
        status = state["adv"][str(i)]
        active, locked, done = state["adv_tab"] == i, status == "locked", status == "done"
        if active:
            rs = {"background": "linear-gradient(135deg,rgba(255,255,255,0.9),rgba(255,255,255,0.64))",
                  "boxShadow": "0 14px 36px rgba(30,58,120,0.16)", "border": "1px solid rgba(255,255,255,0.9)", "cursor": "pointer"}
        elif locked:
            rs = {"background": "rgba(255,255,255,0.18)", "border": "1px solid rgba(255,255,255,0.4)", "opacity": 0.5, "cursor": "not-allowed"}
        else:
            rs = {"background": "rgba(255,255,255,0.32)", "border": "1px solid rgba(255,255,255,0.55)", "cursor": "pointer"}
        if done:
            bg, col = "linear-gradient(135deg,#34c78c,#1aa06e)", "#fff"
        elif active:
            bg, col = "linear-gradient(135deg,#4b8bff,#2f6bff)", "#fff"
        elif locked:
            bg, col = "rgba(120,140,180,0.18)", MUTE
        else:
            bg, col = "rgba(79,139,255,0.14)", BLUE
        items.append(html.Button(id={"type": "adv-tab", "index": i}, n_clicks=0, children=[
            html.Div("\u2713" if done else meta["num"], style={"width": "40px", "height": "40px", "flexShrink": 0,
                     "borderRadius": "13px", "display": "flex", "alignItems": "center", "justifyContent": "center",
                     "fontWeight": 800, "fontSize": "15px", "background": bg, "color": col}),
            html.Div([html.Div(meta["title"], style={"fontSize": "15px", "fontWeight": 800, "color": INK}),
                      html.Div(meta["sub"], style={"fontSize": "12px", "color": "#6b7794"})], style={"flex": 1, "minWidth": 0}),
            html.Span("\u203a", style={"color": "rgba(120,140,180,0.8)", "fontSize": "18px"})],
            style={**rs, "display": "flex", "alignItems": "center", "gap": "13px", "padding": "15px 16px",
                   "borderRadius": "18px", "textAlign": "left", "width": "100%"}))
    return html.Div(className="pvc-advanced-rail", style={"display": "flex", "flexDirection": "column", "gap": "10px"}, children=items)


def empty_box(icon, title, desc, button=None, muted=False):
    border = "rgba(120,140,180,0.4)" if muted else "rgba(79,139,255,0.42)"
    bgc = "rgba(255,255,255,0.24)" if muted else "rgba(255,255,255,0.3)"
    kids = [html.Div(icon, style={"width": "56px", "height": "56px", "borderRadius": "16px",
            "background": "rgba(120,140,180,0.16)" if muted else "rgba(79,139,255,0.12)",
            "color": MUTE if muted else BLUE, "display": "flex", "alignItems": "center",
            "justifyContent": "center", "fontSize": "24px"}),
            html.Div(title, style={"fontSize": "18px" if muted else "20px", "fontWeight": 800,
                     "margin": "18px 0 8px", "color": "#455172" if muted else INK}),
            html.P(desc, style={"maxWidth": "440px", "margin": "0 0 22px" if button else 0, "fontSize": "14px",
                     "lineHeight": "1.55", "color": MUTE if muted else SUB})]
    if button:
        kids.append(button)
    return html.Div(style={"border": f"1.6px dashed {border}", "borderRadius": "20px", "background": bgc,
                           "padding": "52px 40px", "display": "flex", "flexDirection": "column",
                           "alignItems": "center", "textAlign": "center"}, children=kids)


def _metric(label, value, color=INK):
    return html.Div(className="glass-soft", style={"borderRadius": "13px", "padding": "12px 14px", "background": "rgba(255,255,255,0.6)"}, children=[
        html.Div(label.upper(), style={"fontSize": "11px", "color": MUTE, "fontWeight": 600, "letterSpacing": ".03em"}),
        html.Div(value, style={"fontSize": "19px", "fontWeight": 800, "color": color, "marginTop": "3px"})])


def _fold_settings(summary_text, body, is_open, unfold_idx, fold_idx):
    """Controlled collapsible for a step's settings. Its open/closed state lives
    in app-state (not native <details>), so a re-run can force it back to folded
    even after the user had unfolded it to edit. Collapsed -> a single
    click-to-unfold bar; open -> a hide bar followed by the settings."""
    body = body if isinstance(body, list) else [body]
    if not is_open:
        return html.Button([html.Span("\u25B8", className="pvc-chev"), html.Span(summary_text)],
                           id={"type": "act", "index": unfold_idx}, n_clicks=0, className="pvc-foldbar")
    hide = html.Button([html.Span("\u25BE", className="pvc-chev"), html.Span("Hide")],
                       id={"type": "act", "index": fold_idx}, n_clicks=0, className="pvc-foldbar")
    return html.Div([hide, html.Div(body, style={"marginTop": "12px"})])


def _num_input(ptype, key, value, label, dot=False):
    lab = [html.Span(label)]
    if dot:
        lab = [html.Span(style={"display": "inline-block", "width": "7px", "height": "7px", "borderRadius": "50%",
               "background": "#1aa06e", "marginRight": "6px", "verticalAlign": "middle"}), html.Span(label + " · estimated")]
    return html.Div(style={"display": "flex", "flexDirection": "column", "gap": "4px"}, children=[
        html.Label(lab, style={"fontSize": "11.5px", "fontWeight": 600, "color": SUB}),
        dcc.Input(id={"type": ptype, "index": key}, type="number", value=value, className="pnum",
                  debounce=True, style={"width": "100%", "padding": "8px 10px", "borderRadius": "10px",
                  "border": "1px solid rgba(120,140,180,0.4)", "background": "rgba(255,255,255,0.7)",
                  "fontSize": "13px", "color": INK})])


def _filter_row(key, on, fparams):
    head = html.Button(id={"type": "filter", "index": key}, n_clicks=0, children=[
        html.Span("\u2713" if on else "", style={"width": "22px", "height": "22px", "borderRadius": "7px", "flexShrink": 0,
                  "display": "flex", "alignItems": "center", "justifyContent": "center", "color": "#fff", "fontSize": "13px",
                  "background": "linear-gradient(135deg,#4b8bff,#2f6bff)" if on else "transparent",
                  "border": "none" if on else "1.5px solid rgba(120,140,180,0.5)"}),
        html.Div([html.Div(FILTER_LABELS[key], style={"fontSize": "13.5px", "fontWeight": 700, "color": "#374466"}),
                  html.Div(FILTER_EXPLAIN[key], style={"fontSize": "11.5px", "color": SUB, "marginTop": "2px"})],
                 style={"textAlign": "left"})],
        style={"display": "flex", "alignItems": "flex-start", "gap": "11px", "padding": "13px 15px", "borderRadius": "13px",
               "border": "1px solid rgba(255,255,255,0.7)", "background": "rgba(255,255,255,0.62)", "cursor": "pointer", "width": "100%"})
    body = [head]
    specs = FILTER_PARAMS.get(key, [])
    if specs:
        inputs = html.Div([_num_input("fparam", k, fparams.get(k, FILTER_PARAM_DEFAULTS.get(k)), lbl) for k, lbl in specs], className="pvc-form-grid",
                          style={"display": "grid", "gridTemplateColumns": f"repeat({min(len(specs),2)},1fr)",
                                 "gap": "10px", "padding": "12px 15px 4px 48px"})
        body.append(html.Details([
            html.Summary("Customize parameters", style={"cursor": "pointer", "fontSize": "12.5px", "fontWeight": 600,
                         "color": BLUE, "padding": "8px 15px 4px 48px", "listStyle": "none"}),
            inputs]))
    return html.Div(body, className="pvc-sel-block", style={"background": "rgba(255,255,255,0.5)", "borderRadius": "14px", "paddingBottom": "4px"})


def _method_chip(m, on):
    return html.Button([html.Span("\u2713 " if on else "", style={"fontWeight": 800}), m["label"],
                        html.Span("  ·  " + m["tag"], style={"color": MUTE, "fontWeight": 500})],
                       id={"type": "method", "index": m["id"]}, n_clicks=0, style={
        "padding": "10px 15px", "borderRadius": "12px", "cursor": "pointer", "fontWeight": 600, "fontSize": "13px",
        "border": ("1.5px solid #2f6bff" if on else "1px solid rgba(120,140,180,0.35)"),
        "color": (BLUE if on else "#455172"),
        "background": ("linear-gradient(135deg,rgba(79,139,255,0.16),rgba(79,139,255,0.06))" if on else "rgba(255,255,255,0.55)")})


def _param_panel(state):
    blocks = []
    for mid in state["methods"]:
        specs = METHOD_PARAMS.get(mid, [])
        if not specs:
            continue
        blocks.append(html.Div(style={"marginBottom": "12px"}, children=[
            html.Div(METHOD_LABEL[mid], style={"fontSize": "12.5px", "fontWeight": 700, "color": "#374466", "marginBottom": "6px"}),
            html.Div([_num_input("mparam", k, state["mparams"].get(k), lbl) for k, lbl in specs], className="pvc-form-grid",
                     style={"display": "grid", "gridTemplateColumns": f"repeat({min(len(specs),4)},1fr)", "gap": "10px"})]))
    if not blocks:
        blocks = [html.Div("The selected metric(s) have no tunable parameters.", style={"fontSize": "12.5px", "color": MUTE})]
    return html.Div(className="glass-soft", style={"padding": "16px 18px", "background": "rgba(255,255,255,0.5)", "marginTop": "12px"}, children=blocks)


def _rate_card(title, big, unit, meta_rows, accent="#1b5fbf"):
    return html.Div(className="glass-soft", style={"padding": "26px 24px", "background": "rgba(255,255,255,0.72)",
                    "display": "flex", "flexDirection": "column", "minWidth": 0, "height": "100%"}, children=[
        html.Div(title, style={"fontSize": "12px", "fontWeight": 700, "letterSpacing": ".08em", "color": MUTE, "marginBottom": "10px"}),
        html.Div([html.Span(big, style={"fontSize": "44px", "fontWeight": 900, "letterSpacing": "-0.03em", "color": accent, "lineHeight": "1"}),
                  html.Span(" " + unit, style={"fontSize": "18px", "fontWeight": 700, "color": SUB, "marginLeft": "6px"})],
                 style={"display": "flex", "alignItems": "baseline", "flexWrap": "wrap"}),
        html.Div(style={"flex": 1, "minHeight": "18px"}),
        html.Div(meta_rows)])


def _fig_title(text):
    return html.Div(text.upper(), style={"fontSize": "12px", "fontWeight": 700, "letterSpacing": ".08em",
                    "color": MUTE, "margin": "2px 2px 8px"})


def _meta_row(label, value):
    return html.Div([html.Span(label + ": ", style={"color": SUB}), html.B(value, style={"color": INK})],
                    style={"fontSize": "13.5px", "margin": "3px 0"})


def result_cards(result, pvpro_fig="p_mp_ref", as_pills=False):
    # `as_pills` = Advanced-mode layout: the PVPRO parameter selector is a row
    # of pills across the TOP of the trend figure (short name only, full name
    # on hover). When False (Simple mode) the selector keeps the vertical
    # left-hand tabs. The blue->green color scheme applies in both cases.
    result = result or {}
    multi = result.get("multi") or {}
    items = list(multi.items())
    if not items:
        return html.Div()
    dur = result.get("duration_years", 0.0)
    window = result.get("window", "")

    # -------- single metric --------
    if len(items) == 1:
        mid, r = items[0]
        rate, err = r.get("rate"), r.get("error")
        meta = [_meta_row("Method", mid), _meta_row("Duration", f"{dur:.1f} years")]
        if window:
            meta.append(_meta_row("Window", window))
        card = _rate_card("ANNUAL DEGRADATION RATE", deg_str(rate) + "%", "/year", meta, accent="#1b5fbf")

        # PVPRO -> 2 blocks: summary | (param selector + trend figure)
        #   Advanced (as_pills=True): selector is a pill row ACROSS THE TOP of
        #     the figure; each pill shows the short name only, full name on hover.
        #   Simple  (as_pills=False): selector is the vertical left-hand tabs.
        # Colors come from the shared blue->green PVPRO_VAR_COLORS palette in
        # both cases, so a pill/tab accent matches its trend line.
        if not err and r.get("figs_all"):
            figs_all = r["figs_all"]
            _order = ["p_mp_ref", "v_mp_ref", "i_mp_ref", "v_oc_ref", "i_sc_ref"]
            _short = {"p_mp_ref": "Pmp", "v_mp_ref": "Vmp", "i_mp_ref": "Imp", "v_oc_ref": "Voc", "i_sc_ref": "Isc"}
            _full = {"p_mp_ref": "Power at MPP", "v_mp_ref": "Voltage at MPP", "i_mp_ref": "Current at MPP",
                     "v_oc_ref": "Open-circuit voltage", "i_sc_ref": "Short-circuit current"}
            _col = dict(PVPRO_VAR_COLORS)
            present = [k for k in _order if k in figs_all] + [k for k in figs_all if k not in _order]
            sel = pvpro_fig if pvpro_fig in figs_all else (present[0] if present else None)

            if as_pills:
                # ---- Advanced: pills across the top (short name + hover tooltip) ----
                selector = html.Div([html.Button(
                    _short.get(k, k),
                    id={"type": "pvprofig", "index": k}, n_clicks=0,
                    className="pvc-pvpro-pill", **{"data-full": _full.get(k, k)}, style={
                        "padding": "7px 15px", "borderRadius": "980px", "cursor": "pointer",
                        "fontSize": "13.5px", "fontWeight": 800, "letterSpacing": "0.01em",
                        "border": "1px solid " + ("transparent" if k == sel else "rgba(120,140,180,0.30)"),
                        "color": "#fff" if k == sel else "#455172",
                        "background": _col.get(k, BLUE) if k == sel else "rgba(255,255,255,0.62)"})
                    for k in present],
                    style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginBottom": "12px"})
            else:
                # ---- Simple: vertical left-hand tabs (short name + full name) ----
                selector = html.Div([html.Button([
                    html.Div(_short.get(k, k), style={"fontSize": "15px", "fontWeight": 800, "lineHeight": "1.1"}),
                    html.Div("(" + _full.get(k, k) + ")", style={"fontSize": "11px", "fontWeight": 500, "opacity": 0.85, "marginTop": "2px"})],
                    id={"type": "pvprofig", "index": k}, n_clicks=0, style={
                    "textAlign": "left", "padding": "9px 12px", "borderRadius": "11px", "cursor": "pointer",
                    "border": "1px solid " + ("transparent" if k == sel else "rgba(120,140,180,0.28)"),
                    "color": "#fff" if k == sel else "#455172",
                    "background": _col.get(k, BLUE) if k == sel else "rgba(255,255,255,0.6)"}) for k in present],
                    style={"display": "flex", "flexDirection": "column", "gap": "7px", "flex": "0 0 150px"})

            try:
                graph = html.Div(dcc.Graph(figure=glassify(pio.from_json(figs_all[sel]), height=300, top=30), config=GRAPH_CONFIG),
                                 style={"flex": 1, "minWidth": 0, "borderRadius": "14px", "padding": "8px 10px",
                                        "background": "rgba(255,255,255,0.55)", "border": "1px solid rgba(255,255,255,0.7)"}) if sel \
                    else html.Div("No figure.", style={"fontSize": "12px", "color": MUTE})
            except Exception:
                graph = html.Div("No figure.", style={"fontSize": "12px", "color": MUTE})

            if as_pills:
                inner = html.Div([selector, graph], style={"display": "flex", "flexDirection": "column", "height": "100%"})
            else:
                inner = html.Div([selector, graph], style={"display": "flex", "gap": "14px", "alignItems": "stretch", "height": "100%"})
            return html.Div(className="rise pvc-result-grid", style={"display": "grid", "gridTemplateColumns": "290px minmax(0,1fr)",
                            "gap": "18px", "alignItems": "stretch"}, children=[
                card,
                html.Div(inner, className="glass-soft", style={"padding": "16px 18px", "background": "rgba(255,255,255,0.7)"})])

        if err:
            body = html.Div(f"Error: {err}", style={"fontSize": "12.5px", "color": "#b23"})
        else:
            fig = pio.from_json(r["fig"]) if r.get("fig") else None
            if fig is not None:
                fig.update_layout(title=None)
                fig = glassify(fig, height=300, top=10)
            body = [_fig_title("Power trend"),
                    dcc.Graph(figure=fig, config=GRAPH_CONFIG) if fig is not None
                    else html.Div("No figure.", style={"fontSize": "12px", "color": MUTE})]
        return html.Div(className="rise pvc-result-grid", style={"display": "grid", "gridTemplateColumns": "290px minmax(0,1fr)",
                        "gap": "18px", "alignItems": "stretch"}, children=[
            card, html.Div(body if isinstance(body, list) else [body], className="glass-soft",
                           style={"padding": "16px 18px", "background": "rgba(255,255,255,0.7)"})])

    # -------- multiple metrics: summary + bar, then combined trend --------
    labels, rates, colors = [], [], []
    combined = go.Figure()
    daily_added = False
    daily_y = None
    for mid, r in items:
        rate = r.get("rate")
        labels.append(mid)
        rates.append(rate)
        colors.append(_METHOD_COLORS.get(mid, BLUE))
        if r.get("fig"):
            try:
                f = pio.from_json(r["fig"])
                if not daily_added and len(f.data) >= 1:
                    d = f.data[0]
                    d.update(mode="markers", name="Daily power", marker=dict(size=5, opacity=0.35, color="#C7D9EC"))
                    combined.add_trace(d)
                    daily_added = True
                    try:
                        yv = d.y
                        if isinstance(yv, dict) and "bdata" in yv:
                            import base64
                            daily_y = np.frombuffer(base64.b64decode(yv["bdata"]), dtype=np.dtype(yv["dtype"]))
                        else:
                            daily_y = np.asarray(yv, dtype="float64")
                    except Exception:
                        daily_y = None
                if len(f.data) >= 2:
                    tr = f.data[1]
                    tr.update(mode="lines", line=dict(color=_METHOD_COLORS.get(mid, BLUE), width=2.5),
                              name=f"{mid} ({deg_str(rate)}%/yr)")
                    combined.add_trace(tr)
            except Exception:
                pass
    valid = [x for x in rates if x is not None]
    mean = float(np.mean(valid)) if valid else float("nan")
    std = float(np.std(valid)) if valid else 0.0
    meta = [_meta_row("Methods", ", ".join(mid for mid, _ in items)),
            _meta_row("Duration", f"{dur:.1f} years")]
    if window:
        meta.append(_meta_row("Window", window))
    summary = _rate_card("DEGRADATION SUMMARY", f"{deg_str(mean)} \u00b1 {std:.2f}", "%/year", meta, accent="#1b5fbf")

    bar = go.Figure(go.Bar(x=labels, y=[r if r is not None else 0 for r in rates], marker_color=colors,
        text=[deg_str(r) for r in rates], textposition="outside", width=0.5,
        textfont=dict(family="Inter", size=13, color=INK), cliponaxis=False))
    bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280,
                      margin=dict(l=50, r=20, t=14, b=30), showlegend=False,
                      font=dict(family="Inter", color=SUB, size=11), yaxis_title="Rate (%/yr)")
    bar.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(family="Inter", color=MUTE, size=12))
    bar.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=True, tickfont=dict(family="Inter", color=MUTE))

    yr = None
    if daily_y is not None and np.isfinite(daily_y).any():
        lo, hi = np.nanpercentile(daily_y, 1), np.nanpercentile(daily_y, 99)
        pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
        yr = [lo - pad, hi + pad]
    combined.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320,
                           margin=dict(l=52, r=140, t=14, b=40), font=dict(family="Inter", color=SUB, size=11),
                           legend=dict(orientation="v", x=1.02, y=1.0, font=dict(size=11)))
    combined.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(family="Inter", color=MUTE))
    combined.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, range=yr,
                          autorange=(yr is None), tickfont=dict(family="Inter", color=MUTE))
    return html.Div(className="rise", children=[
        html.Div(className="pvc-result-grid", style={"display": "grid", "gridTemplateColumns": "300px minmax(0,1fr)", "gap": "18px", "alignItems": "stretch", "marginBottom": "14px"}, children=[
            summary,
            html.Div([_fig_title("Annual degradation rate \u2014 method comparison"),
                      dcc.Graph(figure=bar, config=GRAPH_CONFIG)],
                     className="glass-soft", style={"padding": "16px 18px", "background": "rgba(255,255,255,0.7)"})]),
        html.Div([_fig_title("Power trend \u2014 all selected methods"),
                  dcc.Graph(figure=combined, config=GRAPH_CONFIG)],
                 className="glass-soft", style={"padding": "16px 18px", "background": "rgba(255,255,255,0.7)"})])


def diagnostics_block(result):
    diag = (result or {}).get("diagnosis")
    thinking = (result or {}).get("diagnosing")
    if diag:
        inner = [dcc.Markdown(diag, className="pvc-md",
                              style={"fontSize": "13.5px", "lineHeight": "1.6", "color": "#26304d"})]
    elif thinking:
        inner = [html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "color": SUB, "fontSize": "13.5px"}, children=[
            html.Div(style={"display": "flex", "gap": "5px"}, children=[
                html.Span(style={"width": "8px", "height": "8px", "borderRadius": "50%", "background": BLUE,
                          "animation": f"pvc-pulseDot 1s ease-in-out {d}s infinite"}) for d in (0, 0.2, 0.4)]),
            html.Span("Analyzing your results\u2026")])]
    else:
        inner = [html.Button(["\u2726  Diagnose with AI"], id={"type": "act", "index": "diagnose"}, n_clicks=0,
                 className="btn-run", style={"padding": "11px 18px", "fontSize": "13.5px"})]
    return html.Div(className="glass-soft", style={"padding": "18px 20px", "background": "rgba(255,255,255,0.7)", "marginTop": "14px"}, children=[
        html.Div("AI DIAGNOSIS", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": BLUE, "marginBottom": "10px"}),
        *inner])


def advanced_body(state, data, filtered, result):
    state = dict(state)
    state["adv"] = dict(state.get("adv") or {})
    _enforce_adv_sequence(state)
    tab = state["adv_tab"]
    meta = STEP_META[tab - 1]
    status = state["adv"][str(tab)]

    if tab == 1:
        if status == "done":
            figs = data.get("prescreen_figs") or []
            fig_items = []
            for jf in figs[:6]:
                try:
                    fig_items.append(html.Div(style={"padding": "6px 6px"}, children=[
                        dcc.Graph(figure=glassify(pio.from_json(jf), height=170, top=34),
                                  config=GRAPH_CONFIG, style={"width": "100%"})]))
                except Exception:
                    pass
            figs_block = html.Div(className="pvc-two-col-grid", style={"background": "rgba(255,255,255,0.82)", "border": "1px solid rgba(255,255,255,0.85)",
                "borderRadius": "18px", "padding": "10px 14px", "boxShadow": "0 6px 18px rgba(30,58,120,0.06)",
                "display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "8px 18px"}, children=fig_items)
            content = html.Div(className="rise", children=[
                html.Div(className="pvc-three-col-grid", style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "10px"}, children=[
                    _metric("Rows", f"{data.get('n_raw', 0):,}"), _metric("Completeness", f"{data.get('completeness', 0)}%"),
                    _metric("Columns identified", str(data.get("n_columns") or len(data.get("columns") or [])))]),
                html.Div("Identified variables — edit any mapping, then re-plot", style={"fontSize": "13px", "fontWeight": 700, "color": "#374466", "margin": "18px 0 10px"}),
                mapping_editor(data),
                html.Div(id="apply-wrap", style={"display": "none"},
                         **{"data-orig": json.dumps({role: (data.get("mapped") or {}).get(role) or "" for role, _ in _MAP_ROLES})},
                         children=[
                    html.Button(["\u21bb  Apply mapping & re-plot"], id={"type": "act", "index": "apply-mapping"}, n_clicks=0, style={
                        "marginTop": "14px", "padding": "10px 16px", "border": "none", "borderRadius": "12px",
                        "background": "linear-gradient(135deg,#4b8bff,#2f6bff)", "color": "#fff", "fontWeight": 700,
                        "fontSize": "13px", "cursor": "pointer", "boxShadow": "0 8px 20px rgba(47,107,255,0.3)"})]),
                (html.Div("Raw signals", style={"fontSize": "13px", "fontWeight": 700, "color": "#374466", "margin": "20px 0 10px"}) if fig_items else html.Div()),
                (figs_block if fig_items else html.Div())])
        else:
            content = empty_box("\u25a5", "Inspect the raw signals first",
                "This checks time coverage, missing values, column types and sensor ranges before any data is removed.",
                html.Button(["Run data prescreening  \u2192"], id={"type": "run-step", "index": 1}, n_clicks=0, className="btn-run"))
    elif tab == 2:
        if status == "locked":
            content = empty_box("\U0001f512", "Run data prescreening first", "Filtering unlocks after screening.", muted=True)
        else:
            filter_settings = [
                html.Div([_filter_row(k, state["filters"][k], state["fparams"]) for k in FILTER_LABELS], className="pvc-two-col-grid",
                         style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "12px", "alignItems": "start"}),
                html.Div(html.Button(["Apply filters  \u2192"], id={"type": "run-step", "index": 2}, n_clicks=0, className="btn-run"),
                         style={"marginTop": "18px"})]
            done = status == "done" and filtered.get("n_raw")
            # Once the filtering result is on screen, fold the settings to save space.
            kids = ([_fold_settings("Filters applied \u2014 click to unfold or modify filters", filter_settings,
                                    state.get("filt_open", False), "unfold-filters", "fold-filters")]
                    if done else list(filter_settings))
            if done:
                n_raw, n_kept = filtered.get("n_raw", 0), filtered.get("n_kept", 0)
                pct = (n_kept / n_raw * 100) if n_raw else 0
                pie = pio.from_json(filtered["pie"]) if filtered.get("pie") else None
                power = pio.from_json(filtered["power"]) if filtered.get("power") else None
                result_block = html.Div(className="rise glass-soft", style={"padding": "20px 22px", "marginTop": "20px", "background": "rgba(255,255,255,0.82)"}, children=[
                    html.Div("FILTERING RESULT", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": BLUE, "marginBottom": "10px"}),
                    html.Div(className="pvc-two-col-grid", style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "alignItems": "center"}, children=[
                        html.Div([
                            html.Div([html.Span(f"{pct:.1f}%", style={"fontSize": "38px", "fontWeight": 900, "color": BLUE}),
                                      html.Div("high-quality points retained", style={"fontSize": "13px", "color": SUB, "fontStyle": "italic"})]),
                            html.Div(style={"marginTop": "12px", "fontSize": "13px", "color": "#374466", "lineHeight": "1.9"}, children=[
                                html.Div([html.Span("Total: "), html.B(f"{n_raw:,}")]),
                                html.Div([html.Span("Retained: "), html.B(f"{n_kept:,}")]),
                                html.Div([html.Span("Filtered: "), html.B(f"{n_raw - n_kept:,}")])])]),
                        (dcc.Graph(figure=pie, config=GRAPH_CONFIG) if pie else html.Div())]),
                    (dcc.Graph(figure=power, config=GRAPH_CONFIG, style={"marginTop": "10px"}) if power else html.Div()),
                ])
                kids.append(result_block)
            content = html.Div(kids)
    elif tab == 3:
        if status == "locked":
            content = empty_box("\U0001f512", "Run intelligent filtering first", "The model fits on the filtered series.", muted=True)
        elif status == "running":
            prog = state.get("pvpro_prog") or {}
            cur, tot = prog.get("current", 0) or 0, prog.get("total", 1) or 1
            pct = min(100, max(4, (cur / tot * 100) if tot else 5))
            content = html.Div(className="glass-soft", style={"padding": "28px 26px", "background": "rgba(255,255,255,0.75)", "position": "relative"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "6px"}, children=[
                    html.Span(style={"width": "10px", "height": "10px", "borderRadius": "50%", "background": "#e08a2b",
                              "animation": "pvc-pulseDot 1.1s ease-in-out infinite"}),
                    html.Div("Fitting PVPRO (single-diode model)\u2026", style={"fontSize": "16px", "fontWeight": 800, "color": INK})]),
                html.Div(prog.get("message", "Working\u2026"), id="pvc-inline-pvpro-message",
                         style={"fontSize": "13px", "color": SUB, "marginBottom": "16px"}),
                html.Div(style={"height": "10px", "borderRadius": "980px", "background": "rgba(120,140,180,0.2)", "overflow": "hidden"}, children=[
                    html.Div(id="pvc-inline-pvpro-fill", style={"height": "100%", "width": f"{pct:.0f}%", "borderRadius": "980px",
                             "background": "linear-gradient(90deg,#f0b45a,#e08a2b)", "transition": "width .4s ease"})]),
                html.Div(f"window {cur} / {tot} \u00b7 {prog.get('phase', '')}", id="pvc-inline-pvpro-meta",
                         style={"fontSize": "12px", "color": MUTE, "marginTop": "8px", "fontFamily": "'JetBrains Mono',monospace"}),
                html.Div("PVPRO walks the series in time windows; this can take 1\u20133 minutes.",
                         style={"fontSize": "12px", "color": MUTE, "marginTop": "10px"}),
                _close_button("stop-pvpro", "Stop PVPRO")])
        else:
            def _check(on):
                return html.Span("\u2713" if on else "", style={"width": "20px", "height": "20px", "borderRadius": "6px",
                    "flexShrink": 0, "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
                    "color": "#fff", "fontSize": "12px", "marginTop": "1px",
                    "background": "#1b5fbf" if on else "transparent",
                    "border": "none" if on else "1.5px solid rgba(120,140,180,0.6)"})

            def _stat_row(m):
                on = m["id"] in state["methods"]
                specs = METHOD_PARAMS.get(m["id"], [])
                head = html.Button([_check(on),
                    html.Span([html.B(m["id"]), html.Span("  \u2014  " + m["full"], style={"color": SUB, "fontWeight": 500})],
                              style={"fontSize": "14px", "color": INK})],
                    id={"type": "method", "index": m["id"]}, n_clicks=0,
                    style={"display": "flex", "alignItems": "flex-start", "gap": "11px", "border": "none",
                           "background": "none", "cursor": "pointer", "padding": "2px 0 6px", "width": "100%", "textAlign": "left"})
                if specs:
                    sub = html.Details([
                        html.Summary("Customize parameters", style={"cursor": "pointer", "fontSize": "12.5px",
                                     "fontWeight": 600, "color": BLUE, "padding": "2px 0 2px 31px", "listStyle": "none"}),
                        html.Div([_num_input("mparam", k, state["mparams"].get(k), lbl) for k, lbl in specs], className="pvc-form-grid",
                                 style={"display": "grid", "gridTemplateColumns": f"repeat({min(len(specs),4)},1fr)",
                                        "gap": "10px", "padding": "8px 0 8px 31px"})])
                else:
                    sub = html.Div("No tunable parameters.", style={"fontSize": "12.5px", "color": MUTE,
                                   "fontStyle": "italic", "padding": "2px 0 2px 31px"})
                return html.Div([head, sub], className="pvc-sel-block",
                                style={"background": "rgba(255,255,255,0.66)" if on else "rgba(255,255,255,0.5)",
                                       "borderRadius": "14px", "padding": "12px 15px",
                                       "border": ("1px solid rgba(79,139,255,0.4)" if on else "1px solid rgba(255,255,255,0.7)")})

            pvpro_on = "PVPRO" in state["methods"]
            needs_vi = pvpro_on and not data.get("has_vi")
            est = (result or {}).get("pvpro_estimated")
            pvpro_section = html.Div([
                html.Div("SINGLE-DIODE-MODEL FITTING", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": SUB, "margin": "0 0 12px 2px"}),
                html.Div([
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "11px", "marginBottom": "4px"}, children=[
                        html.Button([_check(pvpro_on),
                            html.Span([html.B("PVPRO"), html.Span("  \u2014  a lightweight in-app implementation", style={"color": SUB, "fontWeight": 500})],
                                      style={"fontSize": "14px", "color": INK})],
                            id={"type": "method", "index": "PVPRO"}, n_clicks=0,
                            style={"display": "flex", "alignItems": "center", "gap": "11px", "border": "none",
                                   "background": "none", "cursor": "pointer", "padding": "4px 0", "flex": 1, "textAlign": "left"}),
                        html.Img(src=app.get_asset_url("pvpro_logo.png"), alt="PV-Pro", style={"height": "24px", "flexShrink": 0})]),
                    html.Div("\u25f7  ~1\u20133 minutes runtime \u00b7 Need DC Voltage and DC Current columns identified in Step 1",
                             style={"margin": "10px 0 12px 31px", "padding": "10px 14px", "borderRadius": "12px",
                                    "background": "rgba(79,139,255,0.09)", "border": "1px solid rgba(79,139,255,0.2)",
                                    "fontSize": "12.5px", "color": ("#8a6d00" if needs_vi else "#374466")}),
                    html.Div(style={"paddingLeft": "31px"}, children=[
                        html.Details(open=bool(pvpro_on), children=[
                            html.Summary([html.Span("IMPORTANT", style={"background": "#1b5fbf", "color": "#fff", "fontSize": "10px",
                                          "fontWeight": 700, "padding": "2px 8px", "borderRadius": "980px", "marginRight": "8px"}),
                                          html.Span("Module & array parameters for PVPRO", style={"fontSize": "13px", "fontWeight": 700, "color": INK})],
                                         style={"cursor": "pointer", "padding": "4px 0 8px"}),
                            html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "gap": "12px", "marginBottom": "6px"}, children=[
                                html.Div("Provide your module & array layout, or estimate it from the data.",
                                         style={"fontSize": "12.5px", "color": SUB}),
                                html.Button([icon_spark(13, BLUE), html.Span("Estimate from data")], id={"type": "act", "index": "estimate-pvpro"}, n_clicks=0, style={
                                    "flexShrink": 0, "display": "inline-flex", "alignItems": "center", "gap": "7px",
                                    "padding": "9px 15px", "border": "1px solid rgba(79,139,255,0.5)", "borderRadius": "980px",
                                    "background": "rgba(255,255,255,0.7)", "color": BLUE, "fontWeight": 600, "fontSize": "12.5px", "cursor": "pointer"})]),
                            html.Div([
                                _num_input("mparam", "cells", state["mparams"].get("cells", 60), "Cells in series"),
                                _num_input("mparam", "mps", state["mparams"].get("mps", 1), "Modules per string", dot=("mps" in state.get("pvpro_est_keys", []))),
                                _num_input("mparam", "ps", state["mparams"].get("ps", 1), "Parallel strings", dot=("ps" in state.get("pvpro_est_keys", []))),
                                _num_input("mparam", "alphaisc", state["mparams"].get("alphaisc", 0.0046), "\u03b1_Isc (/\u00b0C)"),
                                _num_input("mparam", "days", state["mparams"].get("days", 14), "Days per window"),
                                _num_input("mparam", "iters", state["mparams"].get("iters", 12), "Windows per year"),
                                html.Div([html.Label("Technology", style={"fontSize": "11.5px", "fontWeight": 600, "color": SUB}),
                                          dcc.Dropdown(id={"type": "mparam", "index": "tech"}, className="pvc-dd",
                                                       options=[{"label": t, "value": t} for t in PVPRO_TECHS],
                                                       value=state["mparams"].get("tech", "mono-c-Si"), clearable=False)],
                                         style={"display": "flex", "flexDirection": "column", "gap": "4px"})],
                                     className="pvc-form-grid", style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "10px", "padding": "6px 0 4px"}),
                            (html.Div([html.Span("\u2713", style={"marginRight": "6px", "fontWeight": 800}), est.replace("\u2713 ", "")],
                                      style={"marginTop": "12px", "padding": "10px 14px", "borderRadius": "12px", "fontSize": "12.5px",
                                             "color": "#1a7f52", "background": "rgba(52,199,140,0.14)", "border": "1px solid rgba(52,199,140,0.35)",
                                             "fontWeight": 600}) if est else html.Div())])]),
                ], className="pvc-sel-block", style={"background": "rgba(255,255,255,0.62)", "border": "1px solid rgba(255,255,255,0.7)",
                          "borderRadius": "16px", "padding": "16px 18px"})
            ], style={"marginTop": "18px"})

            sel = html.Div([
                html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "12px"}, children=[
                    html.Div("STATISTICAL / TREND METHODS", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": SUB}),
                    html.Button("Unselect all" if all(m in state["methods"] for m in STAT_METHODS) else "Select all",
                                id={"type": "act", "index": "select-all-methods"}, n_clicks=0, style={
                        "padding": "6px 14px", "border": "1px solid rgba(79,139,255,0.5)", "borderRadius": "980px",
                        "background": "rgba(255,255,255,0.7)", "color": BLUE, "fontWeight": 600, "fontSize": "12.5px", "cursor": "pointer"})]),
                html.Div([_stat_row(m) for m in METHODS if m["id"] in STAT_METHODS], className="pvc-two-col-grid",
                         style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "12px", "alignItems": "start"}),
                pvpro_section,
                (html.Div("\u26a0 PVPRO needs DC Voltage + DC Current columns, which weren't identified in this dataset.",
                          style={"marginTop": "12px", "fontSize": "12.5px", "color": "#8a6d00", "fontWeight": 600}) if needs_vi else html.Div()),
                html.Div(html.Button(["Calculate degradation  \u2192"], id={"type": "run-step", "index": 3}, n_clicks=0, className="btn-run"),
                         style={"marginTop": "20px"})])
            done3 = status == "done" and (result or {}).get("multi")
            # Once results are on screen, fold the metric selection to save space.
            kids = ([_fold_settings("Metrics selected \u2014 click to unfold or modify", sel,
                                    state.get("metric_open", False), "unfold-metrics", "fold-metrics")]
                    if done3 else [sel])
            if done3:
                kids.append(html.Div(className="rise", style={"marginTop": "22px", "paddingTop": "20px",
                            "borderTop": "1px solid rgba(120,140,180,0.22)"}, children=[
                    result_cards(result, state.get("pvpro_fig_sel", "p_mp_ref"), as_pills=True),
                    diagnostics_block(result)]))
            content = html.Div(kids)
    else:
        if status == "locked":
            content = empty_box("\U0001f512", "Run the degradation model first", "Code generation exports the pipeline you ran.", muted=True)
        elif status == "done":
            content = html.Div(className="rise", style={"borderRadius": "15px", "overflow": "hidden", "border": "1px solid rgba(30,40,70,0.12)"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "padding": "10px 16px", "background": "rgba(20,30,55,0.9)"}, children=[
                    html.Span("pv_copilot_pipeline.py", style={"fontSize": "12px", "fontWeight": 600, "color": "#9db4e6", "fontFamily": "'JetBrains Mono',monospace"}),
                    html.Span("Python", style={"fontSize": "11px", "color": "#9db4e6"})]),
                html.Pre((result or {}).get("code", "# no code generated"), style={"margin": 0, "padding": "18px 20px",
                         "background": "rgba(15,23,42,0.96)", "color": "#dbe7ff", "fontFamily": "'JetBrains Mono',monospace",
                         "fontSize": "12.5px", "lineHeight": "1.7", "overflowX": "auto", "maxHeight": "440px"})])
        else:
            content = empty_box("\u27e8\u27e9", "Export a reproducible script",
                "Generates Python that reruns your exact screening, filters and metric end-to-end.",
                html.Button(["Generate code  \u2192"], id={"type": "run-step", "index": 4}, n_clicks=0, className="btn-run"))

    panel_kids = [
        html.Div(f"STEP {meta['num']}", style={"fontSize": "12.5px", "fontWeight": 800, "letterSpacing": "0.12em", "color": BLUE, "marginBottom": "9px"}),
        html.Div(meta["title"], style={"fontSize": "28px", "fontWeight": 800, "letterSpacing": "-0.02em", "marginBottom": "8px"}),
        html.P(meta["desc"], style={"margin": "0 0 22px", "fontSize": "14.5px", "lineHeight": "1.55", "color": SUB, "maxWidth": "none"})]
    if (result or {}).get("error"):
        panel_kids.append(html.Div("\u26a0  " + str(result["error"]), style={"margin": "0 0 16px", "padding": "12px 16px",
            "borderRadius": "14px", "background": "rgba(220,80,80,0.1)", "border": "1px solid rgba(220,80,80,0.3)",
            "color": "#8a1c1c", "fontSize": "13px", "fontWeight": 600}))
    panel_kids.append(content)
    # continue-to-next-step button (when the next step is available)
    if tab < 4 and state["adv"].get(str(tab + 1)) != "locked":
        nxt = STEP_META[tab]["title"]
        panel_kids.append(html.Div(style={"display": "flex", "justifyContent": "flex-end", "marginTop": "20px",
                          "paddingTop": "16px", "borderTop": "1px solid rgba(120,140,180,0.18)"}, children=[
            html.Button([f"Continue to {nxt}  \u2192"], id={"type": "act", "index": "goto-next"}, n_clicks=0, style={
                "display": "inline-flex", "alignItems": "center", "gap": "7px", "padding": "10px 18px",
                "border": "1px solid rgba(79,139,255,0.5)", "borderRadius": "13px", "background": "rgba(255,255,255,0.7)",
                "color": BLUE, "fontWeight": 700, "fontSize": "13.5px", "cursor": "pointer"})]))
    panel = html.Div(className="glass-soft pvc-advanced-panel", style={"padding": "28px 30px", "minHeight": "452px", "minWidth": 0,
                     "maxHeight": "var(--pvc-advanced-step-max-height)", "overflowY": "auto", "overflowX": "hidden",
                     "background": "rgba(255,255,255,0.42)"}, children=panel_kids)
    # Advanced jobs cover the complete right-hand panel (including the step
    # title and description).  The outer host clips the mask to the exact same
    # rounded boundary as the panel, so no corners or scroll-edge slivers leak.
    if state.get("analysis_job") and state.get("analysis_scope") == "advanced" and tab in (1, 2, 3):
        panel = html.Div(className="pvc-panel-busy-host", children=[
            panel,
            _busy_overlay("Analyzing…", "stop-analysis", element_id="busy-overlay")])
    return html.Div(className="pvc-advanced-layout", style={"display": "grid", "gridTemplateColumns": "296px minmax(0,1fr)", "gap": "22px", "alignItems": "start"},
                    children=[rail(state), panel])


# --------------------------------------------------------------------------- chat drawer
def chat_widget():
    return html.Div([
        html.Button([html.Span(icon_spark(15, "#fff"), style={"display": "inline-flex", "width": "28px", "height": "28px",
                     "borderRadius": "9px", "alignItems": "center", "justifyContent": "center",
                     "background": "linear-gradient(135deg,#7db4ff,#2f6bff)"}), html.Span("Ask PVCopilot")],
            id="chat-open", n_clicks=0, className="pvc-chat-open", style={
            "position": "fixed", "bottom": "26px", "right": "26px", "zIndex": 20, "display": "inline-flex",
            "alignItems": "center", "gap": "10px", "padding": "12px 18px 12px 12px", "borderRadius": "20px",
            "border": "1px solid rgba(255,255,255,0.7)", "cursor": "pointer", "fontWeight": 700, "fontSize": "14.5px",
            "color": INK, "background": "linear-gradient(135deg,rgba(255,255,255,0.8),rgba(255,255,255,0.6))",
            "backdropFilter": "blur(24px)", "boxShadow": "0 14px 40px rgba(30,58,120,0.2)"}),
        html.Div(id="chat-drawer")])


_CHAT_EXAMPLES = ["What's my degradation rate?", "Which method should I trust?",
                  "How were points filtered?", "What does PVPRO add?"]


def chat_drawer(cstate):
    if not cstate["open"]:
        return html.Div()
    bubbles = []
    for m in cstate["messages"]:
        user = m["role"] == "user"
        inner = m["text"] if user else dcc.Markdown(m["text"], className="pvc-md",
                                                    style={"fontSize": "13.5px", "lineHeight": "1.55", "color": "#26304d"})
        bubbles.append(html.Div(inner, style={"alignSelf": "flex-end" if user else "flex-start", "maxWidth": "85%",
            "padding": "12px 15px", "fontSize": "13.5px", "lineHeight": "1.5",
            "borderRadius": "16px 16px 4px 16px" if user else "16px 16px 16px 4px",
            "background": "linear-gradient(135deg,#4b8bff,#2f6bff)" if user else "rgba(255,255,255,0.72)",
            "color": "#fff" if user else "#26304d", "border": "none" if user else "1px solid rgba(255,255,255,0.8)"}))
    question_chips = html.Div(className="pvc-chat-prompts", children=[
        html.Button(q, id={"type": "chipq", "index": i}, n_clicks=0,
                    className="pvc-chat-prompt") for i, q in enumerate(_CHAT_EXAMPLES)])
    return html.Div(className="pvc-chat-drawer", style={"position": "fixed", "right": "26px", "bottom": "96px", "width": "380px",
        "height": "min(560px, 68vh)", "maxWidth": "calc(100vw - 40px)", "zIndex": 21, "display": "flex", "flexDirection": "column",
        "animation": "pvc-slideIn .3s ease both", "borderRadius": "26px", "overflow": "hidden",
        "background": "linear-gradient(160deg,rgba(255,255,255,0.86),rgba(255,255,255,0.66))",
        "backdropFilter": "blur(38px) saturate(1.6)", "border": "1px solid rgba(255,255,255,0.8)",
        "boxShadow": "0 24px 70px rgba(30,58,120,0.24)"}, children=[
        html.Div(className="pvc-chat-head", style={"display": "flex", "alignItems": "center", "gap": "12px", "padding": "18px 20px",
                        "borderBottom": "1px solid rgba(120,140,180,0.18)"}, children=[
            html.Div(icon_spark(18, "#fff"), style={"width": "40px", "height": "40px", "borderRadius": "13px",
                     "background": "linear-gradient(135deg,#7db4ff,#4f8bff)", "display": "flex", "alignItems": "center",
                     "justifyContent": "center"}),
            html.Div([html.Div("PVCopilot", style={"fontSize": "15.5px", "fontWeight": 800}),
                      html.Div("\u25cf Ready to help", style={"fontSize": "12px", "color": "#1aa06e", "fontWeight": 600})], style={"flex": 1}),
            html.Button("\u2715", id={"type": "chatbtn", "index": "close"}, n_clicks=0, style={"width": "34px", "height": "34px",
                        "border": "none", "borderRadius": "11px", "background": "rgba(120,140,180,0.14)", "color": "#455172", "cursor": "pointer"})]),
        html.Div(bubbles, className="pvc-chat-messages", style={"flex": 1, "overflowY": "auto", "padding": "20px", "display": "flex", "flexDirection": "column", "gap": "14px"}),
        html.Div(className="pvc-chat-composer", style={"padding": "14px 16px", "borderTop": "1px solid rgba(120,140,180,0.18)"}, children=[
            question_chips,
            html.Div(className="pvc-chat-input-row", style={"display": "flex", "gap": "9px", "padding": "6px 6px 6px 16px", "borderRadius": "16px",
                            "background": "rgba(255,255,255,0.72)", "border": "1px solid rgba(255,255,255,0.85)"}, children=[
                dcc.Input(id={"type": "chatbox", "index": 0}, type="text", placeholder="Ask about your analysis\u2026",
                          value=cstate.get("draft", ""), debounce=True, n_submit=0, style={"flex": 1, "border": "none", "background": "none",
                          "outline": "none", "fontSize": "14px", "color": INK, "minWidth": 0}),
                html.Button("\u27a4", id={"type": "chatbtn", "index": "send"}, n_clicks=0, style={"width": "38px", "height": "38px",
                            "border": "none", "borderRadius": "12px", "cursor": "pointer", "color": "#fff",
                            "background": "linear-gradient(135deg,#4b8bff,#2f6bff)"})])])])


def _context_str(data, result):
    lines = []
    if data.get("loaded"):
        lines.append(f"Dataset: {data.get('filename')} ({data.get('n_raw', 0)} rows, {data.get('completeness', 0)}% complete).")
        cols = ", ".join(f"{r['role']}={r['col']}" for r in (data.get('mapping') or []))
        if cols:
            lines.append("Identified columns: " + cols + ".")
    multi = (result or {}).get("multi") or {}
    for mid, r in multi.items():
        if r.get("rate") is not None:
            lines.append(f"{METHOD_LABEL.get(mid)} = {r['rate']:+.2f} %/yr.")
    smp = (result or {}).get("simple")
    if smp and smp.get("rate") is not None:
        lines.append(f"Simple YoY = {smp['rate']:+.2f} %/yr over {smp.get('duration_years',0):.1f} yr.")
    return "\n".join(lines) or "No dataset loaded yet."


def bot_reply(text, data, result):
    context = _context_str(data, result)
    if _LLM_CLIENT is not None and _LLM_MODEL:
        try:
            resp = _LLM_CLIENT.chat.completions.create(model=_LLM_MODEL, messages=[
                {"role": "system", "content": _CHAT_SYS + "\n\nContext:\n" + context},
                {"role": "user", "content": text}])
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    q = text.lower()
    if "rate" in q or "degrad" in q:
        return "Run a fit and I'll report the rate. Context so far:\n" + context
    if "method" in q or "metric" in q:
        return ("Metrics: year-on-year (robust, recommended), linear regression, Holt-Winters, ARIMA, seasonal "
                "decomposition, and PVPRO (physics single-diode, needs DC V+I). You can select several to compare.")
    if "filter" in q:
        return ("Filtering: basic checks, clear-sky detection, an irradiance threshold, temperature-corrected "
                "normalization, night removal, and IQR outlier rejection — each tunable in Step 2.")
    return "Ask about the degradation rate, the metrics, the filters, or how to read the trend figures."


# --------------------------------------------------------------------------- live progress monitor
def progress_monitor():
    return html.Div(className="pvc-monitor", children=[
        html.Button([
            html.Span(className="pvc-monitor-dot", id="progress-monitor-dot"),
            html.Span("Progress monitor")
        ], id="progress-monitor-toggle", n_clicks=0, className="pvc-monitor-toggle",
           title="Show background task progress"),
        html.Div(id="progress-monitor-panel", className="pvc-monitor-panel", children=[
            html.Div(className="pvc-monitor-head", children=[
                html.Div([
                    html.Div(className="pvc-monitor-titleline", children=[
                        html.Div("Pipeline activity", className="pvc-monitor-title"),
                        html.Span("↕ Move", className="pvc-monitor-move-hint")]),
                    html.Div("Live server-side progress and diagnostics", className="pvc-monitor-subtitle")
                ]),
                html.Div(className="pvc-monitor-actions", children=[
                    html.Button("Copy log", id="progress-monitor-copy", n_clicks=0,
                                className="pvc-monitor-copy", title="Copy the complete activity log"),
                    html.Button("−", id="progress-monitor-minimize", n_clicks=0,
                                className="pvc-monitor-minimize", title="Minimize", **{"aria-label": "Minimize progress monitor"}),
                    html.Button("×", id="progress-monitor-close", n_clicks=0,
                                className="pvc-monitor-close", **{"aria-label": "Close progress monitor"})])
            ]),
            html.Div(id="progress-monitor-body", className="pvc-monitor-body"),
            html.Div(className="pvc-monitor-resize-grip", title="Drag to resize",
                     **{"aria-label": "Resize progress monitor"})
        ])
    ])


def _monitor_row(label, status="idle", message="Waiting", job=None, job_id=None, launched_pid=None):
    status = status if status in ("idle", "running", "done", "error", "locked") else "idle"
    now = time.time()
    details = []
    if job:
        started = job.get("started_at") or now
        updated = job.get("updated_at") or started
        elapsed = max(0, int(now - started))
        since_update = max(0, int(now - updated))
        current = job.get("current") or 0
        total = job.get("total") or 0
        pct = min(100, max(3, current / total * 100)) if total and status == "running" else (100 if status == "done" else 0)
        if status == "running":
            details.append(html.Div(className="pvc-monitor-track", children=[
                html.Div(className="pvc-monitor-fill", style={"width": f"{pct:.1f}%"})]))
        meta = [f"{elapsed}s elapsed", f"updated {since_update}s ago"]
        if job_id:
            meta.append(f"job {job_id}")
        if launched_pid:
            meta.append(f"worker {launched_pid}")
        details.append(html.Div(" · ".join(meta), className="pvc-monitor-meta"))
        if job.get("error"):
            details.append(html.Div(str(job["error"]), className="pvc-monitor-error"))
    elif job_id:
        details.append(html.Div(
            f"Job {job_id} was started on worker {launched_pid or '?'} but is not visible on worker {os.getpid()}.",
            className="pvc-monitor-error"))
    return html.Div(className=f"pvc-monitor-row is-{status}", children=[
        html.Div(className="pvc-monitor-rowtop", children=[
            html.Span(className="pvc-monitor-statusdot"),
            html.Span(label, className="pvc-monitor-label"),
            html.Span(status.upper(), className="pvc-monitor-badge")]),
        html.Div(message, className="pvc-monitor-message"),
        *details])


def _monitor_log_console(state, active_jobs):
    """Render all retained and currently-live events as a scrollable log console."""
    events = list((state or {}).get("pipeline_log") or [])
    seen = {(round(float(e.get("at") or 0), 4), e.get("stage"), e.get("message"), e.get("job_id"))
            for e in events}
    for stage, job_id, job in active_jobs:
        if not job:
            continue
        for event in job.get("history") or []:
            item = {"at": event.get("at"), "stage": stage,
                    "message": event.get("message") or event.get("phase") or "Update",
                    "level": "error" if event.get("error") or event.get("phase") == "error" else event.get("phase") or "info",
                    "job_id": job_id, "worker_pid": job.get("worker_pid"),
                    "dataset": job.get("dataset") or "dataset",
                    "detail": event.get("detail") or event.get("error")}
            key = (round(float(item.get("at") or 0), 4), stage, item.get("message"), job_id)
            if key not in seen:
                events.append(item); seen.add(key)
    events.sort(key=lambda e: float(e.get("at") or 0))
    lines = []
    for event in events[-2000:]:
        stamp = time.strftime("%H:%M:%S", time.localtime(float(event.get("at") or time.time())))
        level = str(event.get("level") or "info")
        dataset = str(event.get("dataset") or "dataset")
        meta = f"{stamp}  {dataset}  {str(event.get('stage') or 'APP').upper()}  W{event.get('worker_pid') or '?'}"
        lines.append(html.Div(className=f"pvc-log-line is-{level}", children=[
            html.Span(meta, className="pvc-log-meta"),
            html.Span(str(event.get("message") or ""), className="pvc-log-message"),
            (html.Pre(str(event.get("detail")), className="pvc-log-detail") if event.get("detail") else None)]))
    if not lines:
        lines = [html.Div("No server-side activity yet.", className="pvc-log-empty")]
    return html.Div(className="pvc-log-section", children=[
        html.Div(className="pvc-log-heading", children=[
            html.Span("SERVER ACTIVITY LOG"),
            html.Div(className="pvc-log-heading-actions", children=[
                html.Span(f"{len(events)} events"),
                html.Button("↓ Latest", id="pvc-monitor-latest", n_clicks=0,
                            className="pvc-log-latest", title="Jump to the newest event")])]),
        html.Div(lines, id="pvc-monitor-log", className="pvc-log-console")])


# --------------------------------------------------------------------------- static pages
def _modal_shell(kicker, title, subtitle, body_children, modal_class=""):
    head = [html.Div(kicker, style={"display": "inline-flex", "padding": "6px 14px", "borderRadius": "20px",
                     "background": "rgba(79,139,255,0.12)", "border": "1px solid rgba(79,139,255,0.22)",
                     "fontSize": "12.5px", "fontWeight": 600, "color": BLUE, "marginBottom": "14px"}),
            html.H2(title, style={"margin": "0 0 8px", "fontSize": "26px", "lineHeight": "1.1", "fontWeight": 800,
                    "letterSpacing": "-0.02em", "color": INK})]
    if subtitle:
        head.append(html.P(subtitle, style={"margin": "0 0 16px", "fontSize": "14.5px", "lineHeight": "1.5", "color": SUB}))
    return html.Div(className="pvc-modal-overlay", children=html.Div(
        className="pvc-modal" + (" " + modal_class if modal_class else ""), children=[
        html.Button("\u2715", id={"type": "modalclose", "index": 0}, n_clicks=0, className="pvc-modal-close"),
        *head, *body_children]))


def methods_modal():
    row = lambda n, t, d: html.Div(style={"padding": "12px 15px", "display": "flex", "gap": "14px", "borderRadius": "14px",
        "marginBottom": "8px", "background": "rgba(255,255,255,0.7)", "border": "1px solid rgba(255,255,255,0.8)"}, children=[
        html.Div(str(n), style={"width": "34px", "height": "34px", "flexShrink": 0, "borderRadius": "11px",
                 "background": "rgba(79,139,255,0.14)", "color": BLUE, "display": "flex", "alignItems": "center",
                 "justifyContent": "center", "fontWeight": 800}),
        html.Div([html.Div(t, style={"fontSize": "14.5px", "fontWeight": 800, "marginBottom": "2px"}),
                  html.P(d, style={"margin": 0, "fontSize": "12.5px", "lineHeight": "1.45", "color": SUB})])])
    return _modal_shell("Methods & documentation", "How PV Copilot works.",
        "Every rate comes from the same four-stage pipeline; Advanced mode exposes every knob.", [
        row(1, "Pre-screening & QA", "Completeness/gap checks, timezone alignment, LLM column identification, and outlier flagging on raw signals."),
        row(2, "Filtering & normalization", "Basic range checks, clear-sky detection, an irradiance threshold, temperature-corrected normalization and night removal — all tunable."),
        row(3, "Degradation modelling", "Fit one or more of year-on-year, linear regression, Holt-Winters, ARIMA, seasonal decomposition, or the PVPRO single-diode model, and compare."),
        row(4, "Code generation", "Export a runnable Python script reproducing your exact pipeline.")])


def cite_modal():
    return _modal_shell("Citation", "How to cite this work.", "If PV Copilot supports your research, please cite it.", [
        html.Div(style={"padding": "20px 22px", "marginBottom": "14px", "borderRadius": "16px",
                        "background": "rgba(255,255,255,0.7)", "border": "1px solid rgba(255,255,255,0.8)"}, children=[
            html.Div("REFERENCE", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": BLUE, "marginBottom": "10px"}),
            html.P([html.Span("Li, B., Karin, T., Chen, X., & Jain, A. (2026). "),
                    html.Em("PV Copilot: An LLM-empowered end-to-end tool for photovoltaic degradation analysis. "),
                    html.Span("Lawrence Berkeley National Laboratory.")],
                   style={"margin": 0, "fontSize": "15px", "lineHeight": "1.7", "color": INK})]),
        html.Pre('@misc{pvcopilot2026,\n  title   = {PV Copilot: An LLM-empowered end-to-end tool\n             for PV degradation analysis},\n  author  = {Li, Baojie and Karin, Todd and Chen, Xin and Jain, Anubhav},\n  year    = {2026},\n  institution = {Lawrence Berkeley National Laboratory}\n}',
                 style={"margin": 0, "padding": "16px 18px", "borderRadius": "14px", "background": "rgba(15,23,42,0.96)",
                        "color": "#dbe7ff", "fontFamily": "'JetBrains Mono',monospace", "fontSize": "12px", "lineHeight": "1.7", "overflowX": "auto"})])


def team_modal():
    members = [
        ("Baojie Li", "Lead developer & primary contributor", "team_baojieL.jpg"),
        ("Nishanth Koushik", "Algorithm development", "team_nishanthK.jpg"),
        ("Anubhav Jain", "Principal investigator", "team_anubhavJ.jpg"),
    ]

    def card(name, role, photo):
        return html.Div(className="glass pvc-team-card", children=[
            html.Img(src=app.get_asset_url(f"pvcopilot_team/{photo}"), alt=name,
                     className="pvc-team-photo"),
            html.Div(className="pvc-team-copy", children=[
                html.Div(name, className="pvc-team-name"),
                html.Div(role, className="pvc-team-role"),
                html.Div("Lawrence Berkeley National Laboratory", className="pvc-team-org")])])

    return _modal_shell("Team", "Meet the PV Copilot team.",
                        "Research, software and photovoltaic degradation expertise at Berkeley Lab.", [
        html.Div([card(*member) for member in members], className="pvc-team-grid")
    ], modal_class="pvc-team-modal")


_CHANGELOG = [
    ("v1.3", "2026-07", ["Simple mode now offers a YOY / PVPRO choice with auto-estimated parameters",
                         "PVPRO results show switchable single-diode parameter trends (Pmp, Voc, Isc, …)",
                         "AI diagnosis added under every result"]),
    ("v1.2", "2026-06", ["Advanced mode: multi-method comparison (YOY, LR, Holt-Winters, ARIMA, seasonal)",
                         "PVPRO single-diode fitting with live progress",
                         "In-app PVCopilot chat assistant"]),
    ("v1.1", "2026-05", ["Liquid-glass redesign integrated into the pvtools site",
                         "Intelligent filtering step with tunable thresholds"]),
    ("v1.0", "2026-04", ["First release: upload \u2192 pre-screen \u2192 filter \u2192 degradation rate"]),
]


def whatsnew_modal():
    rows = []
    for ver, date, changes in _CHANGELOG:
        rows.append(html.Div(style={"padding": "14px 16px", "marginBottom": "10px", "borderRadius": "14px",
                    "background": "rgba(255,255,255,0.7)", "border": "1px solid rgba(255,255,255,0.8)"}, children=[
            html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "10px", "marginBottom": "6px"}, children=[
                html.Span(ver, style={"fontSize": "15px", "fontWeight": 800, "color": INK}),
                html.Span(date, style={"fontSize": "12px", "color": MUTE, "fontFamily": "'JetBrains Mono',monospace"})]),
            html.Ul([html.Li(c, style={"fontSize": "12.5px", "color": SUB, "lineHeight": "1.5", "marginBottom": "2px"}) for c in changes],
                    style={"margin": 0, "paddingLeft": "20px"})]))
    return _modal_shell("What's new", "Version history.", "Recent releases and major changes.", rows)


def datareq_modal():
    def section_heading(title, description):
        return html.Div(className="pvc-datareq-section-head", children=[
            html.Div(title, className="pvc-datareq-section-title"),
            html.Div(description, className="pvc-datareq-section-description")])

    def signal_card(level, title, fields, description, accent, tint):
        return html.Div(className="glass pvc-datareq-card", style={"--req-accent": accent, "--req-tint": tint}, children=[
            html.Div(className="pvc-datareq-card-head", children=[
                html.Span(level, className="pvc-datareq-level")]),
            html.Div([html.Span(field, className="glass-soft pvc-datareq-field") for field in fields],
                     className="pvc-datareq-fields"),
            html.Div(title, className="pvc-datareq-title"),
            html.P(description, className="pvc-datareq-description")])

    signal_grid = html.Div(className="pvc-datareq-signal-surface", children=[
        html.Div(className="pvc-datareq-grid", children=[
        signal_card("REQUIRED", "Core analysis", ["Time", "Power"],
                    "The minimum signals needed to calculate a degradation trend.",
                    "#c43d4b", "rgba(196,61,75,0.11)"),
        html.Div("+", className="glass-soft pvc-datareq-plus", **{"aria-hidden": "true"}),
        signal_card("RECOMMENDED", "Cleaner normalization", ["Irradiance", "Module temperature"],
                    "Adds irradiance normalization and temperature correction.",
                    "#159468", "rgba(21,148,104,0.10)"),
        html.Div("+", className="glass-soft pvc-datareq-plus", **{"aria-hidden": "true"}),
        signal_card("PVPRO ONLY", "Physics diagnostics", ["DC voltage", "DC current"],
                    "Unlocks Pmp, Voc, Isc and other single-diode parameter trends.",
                    "#667085", "rgba(102,112,133,0.12)")])])

    def checklist_item(label, value, note):
        return html.Div(className="glass-soft pvc-datareq-check", children=[
            html.Div(label, className="pvc-datareq-check-label"),
            html.Div(value, className="pvc-datareq-check-value"),
            html.Div(note, className="pvc-datareq-check-note")])

    checklist = html.Div(className="pvc-datareq-file", children=[
        html.Div(className="pvc-datareq-check-grid", children=[
            checklist_item("FORMAT", "CSV or Parquet", "One file per upload"),
            checklist_item("HISTORY", "2+ years", "Longer records are better"),
            checklist_item("SAMPLING", "1–6 hours", "Consistent intervals preferred")])])

    note = html.Div(className="pvc-datareq-note", children=[
        html.Span("✦", className="pvc-datareq-note-icon"),
        html.Span([html.B("Column names can vary."),
                   " PV Copilot identifies likely signals automatically, and you can review the mapping before analysis."])])

    signals_section = html.Div(className="pvc-datareq-section", children=[
        section_heading("Signals to include", "Begin with the required pair, then add signals when your analysis needs them."),
        signal_grid])
    file_section = html.Div(className="pvc-datareq-section", children=[
        section_heading("File format & coverage", "Upload one time-series file with enough history and a consistent sampling interval."),
        checklist])

    return _modal_shell("Data requirements", "Prepare your dataset.", "",
                        [signals_section, file_section, note], modal_class="pvc-datareq-modal")


def about_modal():
    members = [("Baojie Li", "#4f8bff", "#fff"), ("Todd Karin", "#ffb04d", "#7a4a00"),
               ("Xin Chen", "#7db4ff", "#fff"), ("Anubhav Jain", "#1aa06e", "#fff")]

    def card(name, c1, tc):
        initials = "".join(w[0] for w in name.split()[:2])
        return html.Div(style={"padding": "14px 14px 16px", "borderRadius": "14px", "background": "rgba(255,255,255,0.7)",
                        "border": "1px solid rgba(255,255,255,0.8)"}, children=[
            html.Div(initials, style={"width": "100%", "aspectRatio": "1 / 1", "borderRadius": "12px", "marginBottom": "10px",
                     "background": c1, "color": tc, "display": "flex", "alignItems": "center", "justifyContent": "center",
                     "fontSize": "26px", "fontWeight": 800}),
            html.Div(name, style={"fontSize": "14px", "fontWeight": 800}),
            html.Div("LBNL", style={"fontSize": "11.5px", "color": SUB, "marginTop": "2px"})])
    return _modal_shell("About", "The team & how to cite.", "Built at Lawrence Berkeley National Laboratory.", [
        html.Div("TEAM", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": BLUE, "margin": "2px 0 10px"}),
        html.Div([card(n, c, tc) for n, c, tc in members],
                 style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)", "gap": "10px", "marginBottom": "18px"}),
        html.Div("HOW TO CITE", style={"fontSize": "12px", "fontWeight": 800, "letterSpacing": ".08em", "color": BLUE, "margin": "2px 0 10px"}),
        html.P([html.Span("Li, B., Karin, T., Chen, X., & Jain, A. (2026). "),
                html.Em("PV Copilot: An LLM-empowered end-to-end tool for photovoltaic degradation analysis. "),
                html.Span("Lawrence Berkeley National Laboratory.")],
               style={"margin": "0 0 12px", "fontSize": "14px", "lineHeight": "1.6", "color": INK}),
        html.Pre('@misc{pvcopilot2026,\n  title  = {PV Copilot: An LLM-empowered end-to-end tool for PV degradation analysis},\n  author = {Li, Baojie and Karin, Todd and Chen, Xin and Jain, Anubhav},\n  year   = {2026},\n  institution = {Lawrence Berkeley National Laboratory}\n}',
                 style={"margin": 0, "padding": "14px 16px", "borderRadius": "12px", "background": "rgba(15,23,42,0.96)",
                        "color": "#dbe7ff", "fontFamily": "'JetBrains Mono',monospace", "fontSize": "11.5px", "lineHeight": "1.7", "overflowX": "auto"})])


def render_modal(view):
    return {"methods": methods_modal, "about": team_modal, "cite": cite_modal, "team": team_modal,
            "whatsnew": whatsnew_modal, "datareq": datareq_modal}.get(view, lambda: None)()


# --------------------------------------------------------------------------- page root (pvtools)
def _root_layout():
    return dmc.MantineProvider(forceColorScheme="light", children=html.Div(className="pvcopilot-root", children=[
        html.Div(id="pvc-page", children=home_body()),
        progress_monitor(),
        chat_widget(),
        dcc.Store(id="pvc-view", data="app"),
        dcc.Store(id="app-state", data=DEFAULT_STATE),
        dcc.Store(id="data-store", data=EMPTY_DATA),
        dcc.Store(id="filtered-store", data={}),
        dcc.Store(id="result-store", data={}),
        dcc.Store(id="pvpro-live", data={}),
        dcc.Store(id="pvpro-dom-sink"),
        dcc.Store(id="chat-state", data={"open": False, "messages": [
            {"role": "bot", "text": "Hi — I'm your PV Copilot. Upload a dataset or pick an example, then ask "
             "me about the methods, filters, or results."}]}),
        dcc.Interval(id="pvpro-poll", interval=1200, n_intervals=0, disabled=True),
        dcc.Interval(id="async-job-poll", interval=900, n_intervals=0, disabled=True),
        dcc.Store(id="modal-view", data=None),
        html.Div(id="modal-root", style={"position": "relative", "zIndex": 6000}),
        dcc.Store(id="busy-sink"),
        dcc.Store(id="busy-sink2"),
        dcc.Store(id="busy-sink3"),
    ]))


layout = _root_layout()


# Poll only while a relevant background task exists. Always-on intervals made
# the deployed tab flash "Updating…" and competed with the job-completion
# callback even when the application was idle.
app.clientside_callback(
    """
    function(state) {
        state = state || {};
        var asyncActive = !!(state.ingest_job || state.analysis_job);
        var longActive = !!(state.pvpro_job || state.diag_job);
        return [!asyncActive, !longActive];
    }
    """,
    Output("async-job-poll", "disabled"),
    Output("pvpro-poll", "disabled"),
    Input("app-state", "data"),
)


# Do not route high-frequency PVPRO progress through app-state. Updating that
# store rebuilds the complete workflow panel. This clientside bridge changes
# only the message, bar width and window counter in the already-mounted card.
app.clientside_callback(
    """
    function(progress) {
        progress = progress || {};
        var message = document.getElementById('pvc-inline-pvpro-message');
        var fill = document.getElementById('pvc-inline-pvpro-fill');
        var meta = document.getElementById('pvc-inline-pvpro-meta');
        var current = Number(progress.current || 0);
        var total = Number(progress.total || 1);
        var percent = Math.min(100, Math.max(4, total ? current / total * 100 : 5));
        if (message) message.textContent = progress.message || 'Working…';
        if (fill) fill.style.width = percent.toFixed(0) + '%';
        if (meta) meta.textContent = 'window ' + current + ' / ' + total + ' · ' + (progress.phase || '');
        return Date.now();
    }
    """,
    Output("pvpro-dom-sink", "data"),
    Input("pvpro-live", "data"),
)


app.clientside_callback(
    """
    function(openClicks, closeClicks, minimizeClicks, currentClass) {
        var triggered = (window.dash_clientside.callback_context.triggered || []);
        if (!triggered.length || !triggered[0].value) return window.dash_clientside.no_update;
        var id = triggered[0].prop_id || '';
        if (id.indexOf('progress-monitor-close') === 0) return 'pvc-monitor-panel';
        if (id.indexOf('progress-monitor-minimize') === 0) return 'pvc-monitor-panel';
        return (currentClass || '').indexOf(' is-open') >= 0
            ? 'pvc-monitor-panel' : 'pvc-monitor-panel is-open';
    }
    """,
    Output("progress-monitor-panel", "className"),
    Input("progress-monitor-toggle", "n_clicks"),
    Input("progress-monitor-close", "n_clicks"),
    Input("progress-monitor-minimize", "n_clicks"),
    State("progress-monitor-panel", "className"),
    prevent_initial_call=True,
)


@app.callback(
    Output("progress-monitor-body", "children"),
    Output("progress-monitor-dot", "className"),
    Input("async-job-poll", "n_intervals"),
    Input("pvpro-poll", "n_intervals"),
    Input("app-state", "data"), State("data-store", "data"),
    State("filtered-store", "data"), State("result-store", "data"),
)
def render_progress_monitor(_async_tick, _pvpro_tick, state, data, filtered, result):
    state = state or DEFAULT_STATE
    data = data or EMPTY_DATA
    filtered = filtered or {}
    result = result or {}
    adv = state.get("adv") or {}

    ingest_id = state.get("ingest_job")
    analysis_id = state.get("analysis_job")
    pvpro_id = state.get("pvpro_job")
    diag_id = state.get("diag_job")
    ingest_job = _INGEST_JOBS.get(ingest_id) if ingest_id else None
    analysis_job = _ANALYSIS_JOBS.get(analysis_id) if analysis_id else None
    pvpro_job = _PVPRO_JOBS.get(pvpro_id) if pvpro_id else None
    diag_job = _DIAG_JOBS.get(diag_id) if diag_id else None
    analysis_kind = (analysis_job or {}).get("kind")

    def running_status(job_id, job):
        if not job_id:
            return None
        if not job:
            return "error"
        return "error" if job.get("phase") == "error" else ("done" if job.get("phase") == "done" else "running")

    ingest_status = running_status(ingest_id, ingest_job)
    if not ingest_status:
        ingest_status = "done" if data.get("loaded") else "idle"
    rows = [_monitor_row(
        "Data loading", ingest_status,
        (ingest_job or {}).get("message") or ("Dataset is ready." if data.get("loaded") else "No dataset selected."),
        ingest_job, ingest_id, state.get("ingest_worker_pid"))]

    def advanced_row(step, label, kind, ready_message, done_message):
        if analysis_id and analysis_kind == kind:
            status = running_status(analysis_id, analysis_job)
            return _monitor_row(label, status, (analysis_job or {}).get("message") or ready_message,
                                analysis_job, analysis_id, state.get("analysis_worker_pid"))
        value = adv.get(str(step), "locked")
        status = "done" if value == "done" else ("locked" if value == "locked" else "idle")
        message = done_message if status == "done" else ("Complete the previous step first." if status == "locked" else ready_message)
        return _monitor_row(label, status, message)

    rows.append(advanced_row(1, "Data prescreening", "advanced_1", "Ready to inspect raw signals.", "Prescreening completed."))
    rows.append(advanced_row(2, "Intelligent filtering", "advanced_2", "Ready to apply filters.", "Filtering completed."))

    if analysis_id and analysis_kind in ("simple_yoy", "simple_prepare_pvpro", "advanced_3"):
        analysis_status = running_status(analysis_id, analysis_job)
        rows.append(_monitor_row("Degradation analysis", analysis_status,
                                 (analysis_job or {}).get("message") or "Calculating degradation…",
                                 analysis_job, analysis_id, state.get("analysis_worker_pid")))
    else:
        analysis_done = bool(state.get("simple_done") or adv.get("3") == "done")
        analysis_locked = not data.get("loaded") or (state.get("mode") == "advanced" and adv.get("3") == "locked")
        rows.append(_monitor_row("Degradation analysis",
                                 "done" if analysis_done else ("locked" if analysis_locked else "idle"),
                                 "Analysis completed." if analysis_done else ("Complete filtering first." if analysis_locked else "Ready to calculate degradation.")))

    pv_status = running_status(pvpro_id, pvpro_job)
    if not pv_status:
        pv_status = "done" if "PVPRO" in ((result.get("multi") or {})) else "idle"
    rows.append(_monitor_row("PVPRO fitting", pv_status,
                             (pvpro_job or {}).get("message") or ("PVPRO result is ready." if pv_status == "done" else "Not running."),
                             pvpro_job, pvpro_id, state.get("pvpro_worker_pid")))

    code_status = "done" if adv.get("4") == "done" else ("locked" if adv.get("4") == "locked" else "idle")
    rows.append(_monitor_row("Code generation", code_status,
                             "Reproducible code generated." if code_status == "done" else "Available after analysis."))

    if diag_id:
        diag_status = ("error" if not diag_job or diag_job.get("phase") == "error"
                       else ("done" if diag_job.get("done") else "running"))
        diag_message = (diag_job or {}).get("message") or ("Generating AI diagnosis…" if diag_status == "running" else "Diagnosis task is unavailable.")
        rows.append(_monitor_row("AI diagnosis", diag_status, diag_message, diag_job, diag_id,
                                 (diag_job or {}).get("worker_pid")))
    else:
        rows.append(_monitor_row("AI diagnosis", "done" if result.get("diagnosis") else "idle",
                                 "Diagnosis completed." if result.get("diagnosis") else "Not requested."))

    missing = []
    for label, jid, job, launched in (
        ("data loading", ingest_id, ingest_job, state.get("ingest_worker_pid")),
        ("analysis", analysis_id, analysis_job, state.get("analysis_worker_pid")),
        ("PVPRO", pvpro_id, pvpro_job, state.get("pvpro_worker_pid")),
    ):
        if jid and not job:
            missing.append(f"{label} job {jid}: launched on worker {launched or '?'}, polled on worker {os.getpid()}")
    banner = []
    if missing:
        banner = [html.Div(className="pvc-monitor-warning", children=[
            html.B("Background job unavailable."),
            html.Div(f"Job storage: {_JOB_STORE_STATUS}. Check the log below for the exact worker and job id."),
            html.Code(" | ".join(missing))])]

    active_jobs = [j for j in (ingest_job, analysis_job, pvpro_job, diag_job) if j and j.get("phase") not in ("done", "error", "cancelled") and not j.get("done")]
    any_missing = bool(missing)
    any_error = any(j and (j.get("phase") == "error" or j.get("error")) for j in (ingest_job, analysis_job, pvpro_job))
    dot_state = "is-error" if any_missing or any_error else ("is-running" if active_jobs else ("is-done" if data.get("loaded") else "is-idle"))
    log_console = _monitor_log_console(state, [
        ("UPLOAD", ingest_id, ingest_job),
        ("ANALYSIS", analysis_id, analysis_job),
        ("PVPRO", pvpro_id, pvpro_job),
        ("DIAGNOSIS", diag_id, diag_job)])
    footer = html.Div(f"Instance {_JOB_INSTANCE} · worker {os.getpid()} · {_JOB_STORE_STATUS} · updates every 0.9s",
                      className="pvc-monitor-footer")
    active_rows = []
    if ingest_id:
        active_rows.append(rows[0])
    if analysis_id:
        active_rows.append(rows[1] if analysis_kind == "advanced_1" else
                           rows[2] if analysis_kind == "advanced_2" else rows[3])
    if pvpro_id:
        active_rows.append(rows[4])
    if diag_id:
        active_rows.append(rows[6])
    if not active_rows:
        active_rows = [html.Div("No background task is running. The complete history remains below.",
                                className="pvc-monitor-idle-summary")]
    row_grid = html.Div(active_rows, className="pvc-monitor-grid")
    return [*banner, row_grid, log_console, footer], f"pvc-monitor-dot {dot_state}"


# Busy overlay: show a spinner + live seconds counter while an action computes,
# hide it when the workflow slot re-renders (i.e. the result is ready).
app.clientside_callback(
    """
    function(runSteps, actClicks) {
        var t = (window.dash_clientside.callback_context.triggered || []);
        if (!t.length || !t[0].value) return window.dash_clientside.no_update;
        var id0 = (t[0].prop_id || '');
        // fold/unfold are instant UI toggles — don't flash the "Analyzing…" spinner.
        if (id0.indexOf('fold-filters') !== -1 || id0.indexOf('fold-metrics') !== -1)
            return window.dash_clientside.no_update;
        var ov = document.getElementById('busy-overlay');
        if (!ov) return window.dash_clientside.no_update;
        ov.classList.add('show');
        var start = Date.now();
        if (window._pvcBusy) clearInterval(window._pvcBusy);
        var tick = function () {
            var s = Math.floor((Date.now() - start) / 1000);
            var tx = document.getElementById('busy-text');
            if (tx) tx.innerText = 'Analyzing\\u2026 ' + s + 's';
        };
        tick();
        window._pvcBusy = setInterval(tick, 1000);
        return 1;
    }
    """,
    Output("busy-sink", "data"),
    Input({"type": "run-step", "index": ALL}, "n_clicks"),
    Input({"type": "act", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(state) {
        if (state && state.analysis_job) return window.dash_clientside.no_update;
        var ov = document.getElementById('busy-overlay');
        if (ov) ov.classList.remove('show');
        if (window._pvcBusy) { clearInterval(window._pvcBusy); window._pvcBusy = null; }
        return 1;
    }
    """,
    Output("busy-sink", "data", allow_duplicate=True),
    Input("app-state", "data"),
    prevent_initial_call=True,
)


# Poll the PVPRO background job: update the progress bar each tick, and when the
# fit finishes, drop its rate + all figures into result-store.
@app.callback(
    Output("app-state", "data", allow_duplicate=True),
    Output("result-store", "data", allow_duplicate=True),
    Output("pvpro-live", "data"),
    Input("pvpro-poll", "n_intervals"),
    State("app-state", "data"), State("filtered-store", "data"), State("result-store", "data"),
    prevent_initial_call=True,
)
def poll_pvpro(_n, state, filtered, result):
    state = state or {}
    # ---- async AI diagnosis ----
    djid = state.get("diag_job")
    if djid:
        dj = _DIAG_JOBS.get(djid)
        if not dj:
            state = dict(state)
            if _defer_missing_job(state, "diagnosis", djid):
                return state, no_update, no_update
            state = dict(state); state["diag_job"] = None
            result = dict(result or {}); result["diagnosing"] = False
            result["diagnosis"] = "AI diagnosis task is unavailable on this server worker."
            return state, result, no_update
        _clear_missing_job(state, "diagnosis", djid)
        if dj and dj.get("done"):
            state = dict(state); state["diag_job"] = None
            result = dict(result or {}); result["diagnosis"] = dj.get("text"); result["diagnosing"] = False
            return state, result, no_update
    # ---- PVPRO fit ----
    jid = state.get("pvpro_job")
    if not jid:
        return no_update, no_update, no_update
    simple = state.get("pvpro_mode") == "simple"
    if not simple and (state.get("adv") or {}).get("3") != "running":
        return no_update, no_update, no_update
    job = _PVPRO_JOBS.get(jid)
    if not job:
        state = dict(state); state["adv"] = dict(state.get("adv") or {})
        if _defer_missing_job(state, "pvpro", jid):
            return state, no_update, no_update
        state["pvpro_job"] = None; state["pvpro_prog"] = None
        if not simple:
            state["adv"]["3"] = "idle"
        message = _missing_job_message("PVPRO", jid, state.get("pvpro_worker_pid"))
        _append_pipeline_log(state, "PVPRO", message, level="error", job_id=jid,
                             worker_pid=state.get("pvpro_worker_pid"))
        state["pvpro_worker_pid"] = None
        return state, dict(result or {}, error=message), {}
    _clear_missing_job(state, "pvpro", jid)
    state = dict(state)
    state["adv"] = dict(state.get("adv") or {})
    if job["phase"] == "error":
        _capture_job_history(state, "PVPRO", jid, job)
        state["pvpro_job"] = None; state["pvpro_prog"] = None
        if not simple:
            state["adv"]["3"] = "idle"
        _update_job_record(job, committed=True)
        state["pvpro_worker_pid"] = None
        return state, dict(result or {}, error=job.get("error")), {}
    if job["phase"] == "done" and job.get("result") is None:
        # Defensive guard for old/custom compute implementations that may
        # publish phase=done before the result payload is attached.
        live_progress = {"phase": "finalizing", "current": job.get("current", 0),
                         "total": job.get("total", 1),
                         "message": "Finalizing PVPRO rates and figures…"}
        return no_update, no_update, live_progress
    if job["phase"] == "done":
        _capture_job_history(state, "PVPRO", jid, job)
        r = job.get("result") or {}
        state["pvpro_job"] = None; state["pvpro_prog"] = None
        res = {"multi": {"PVPRO": {"rate": r.get("rate"), "figs_all": r.get("figs") or {}, "rates": r.get("rates") or {}}},
               "duration_years": state.get("pvpro_dur", 0.0), "n_kept": state.get("pvpro_nkept", 0),
               "window": state.get("pvpro_window", "")}
        if simple:
            state["simple_done"] = True
        else:
            state["adv"]["3"] = "done"; state["adv"]["4"] = "idle"
        _update_job_record(job, committed=True)
        state["pvpro_worker_pid"] = None
        return state, res, {}
    # still running -> update progress
    live_progress = {k: job.get(k) for k in ("phase", "current", "total", "message")}
    return no_update, no_update, live_progress


@app.callback(
    Output("app-state", "data", allow_duplicate=True),
    Output("data-store", "data", allow_duplicate=True),
    Output("filtered-store", "data", allow_duplicate=True),
    Output("result-store", "data", allow_duplicate=True),
    Output("ingest-status", "children", allow_duplicate=True),
    Input("async-job-poll", "n_intervals"),
    State("app-state", "data"), State("data-store", "data"),
    State("filtered-store", "data"), State("result-store", "data"),
    prevent_initial_call=True,
)
def poll_cancellable_jobs(_n, state, data, filtered, result):
    state = dict(state or DEFAULT_STATE)
    state["adv"] = dict(state.get("adv") or {})

    ingest_id = state.get("ingest_job")
    if ingest_id:
        job = _INGEST_JOBS.get(ingest_id)
        if not job:
            if _defer_missing_job(state, "upload", ingest_id):
                return state, no_update, no_update, no_update, no_update
            message = _missing_job_message("Data-loading", ingest_id, state.get("ingest_worker_pid"))
            _append_pipeline_log(state, "UPLOAD", message, level="error", job_id=ingest_id,
                                 worker_pid=state.get("ingest_worker_pid"))
            state["ingest_job"] = None; state["ingest_worker_pid"] = None
            return state, EMPTY_DATA, {}, {}, _alert(message)
        _clear_missing_job(state, "upload", ingest_id)
        if job and job.get("phase") == "done":
            _capture_job_history(state, "UPLOAD", ingest_id, job)
            parsed = job.get("result") or EMPTY_DATA
            _update_job_record(job, committed=True)
            state["ingest_job"] = None
            state["ingest_worker_pid"] = None
            return state, parsed, {}, {}, html.Div()
        if job and job.get("phase") == "error":
            _capture_job_history(state, "UPLOAD", ingest_id, job)
            message = job.get("error") or "Unknown data-loading error."
            _update_job_record(job, committed=True)
            state["ingest_job"] = None
            state["ingest_worker_pid"] = None
            return state, EMPTY_DATA, {}, {}, _alert(f"Could not read the data: {message}")

    analysis_id = state.get("analysis_job")
    if analysis_id:
        job = _ANALYSIS_JOBS.get(analysis_id)
        if not job:
            if _defer_missing_job(state, "analysis", analysis_id):
                return state, no_update, no_update, no_update, no_update
            kind = "Analysis"
            message = _missing_job_message(kind, analysis_id, state.get("analysis_worker_pid"))
            _append_pipeline_log(state, "ANALYSIS", message, level="error", job_id=analysis_id,
                                 worker_pid=state.get("analysis_worker_pid"))
            state["analysis_job"] = None; state["analysis_worker_pid"] = None
            state["analysis_scope"] = None
            if state["adv"].get("3") == "running_async":
                state["adv"]["3"] = "idle"
            result = dict(result or {}); result["error"] = message
            return state, no_update, no_update, result, no_update
        _clear_missing_job(state, "analysis", analysis_id)
        # Jobs created by an older deployment may still point at a daemon
        # thread that vanished with its Gunicorn worker. Never leave the UI in
        # an eternal running state. New Simple YOY runs synchronously and will
        # not enter this branch; this watchdog only cleans up legacy/stale jobs.
        if (job and job.get("kind") == "simple_yoy"
                and job.get("phase") not in ("done", "error", "cancelled")
                and time.time() - float(job.get("updated_at") or job.get("started_at") or time.time()) > 45):
            stale_for = int(time.time() - float(job.get("updated_at") or job.get("started_at") or time.time()))
            _update_job_record(
                job, phase="error",
                message=(f"Legacy Simple YOY worker stopped responding for {stale_for}s. "
                         "Please run it again; Simple YOY now executes in the request worker."),
                error="The background Simple YOY worker stopped responding.")
            _ANALYSIS_JOBS[analysis_id] = job
        if job and job.get("phase") in ("done", "error"):
            kind = job.get("kind")
            payload = job.get("result")
            error = job.get("error")
            _capture_job_history(state, "ANALYSIS", analysis_id, job)
            _update_job_record(job, committed=True)
            state["analysis_job"] = None
            state["analysis_worker_pid"] = None
            state["analysis_scope"] = None
            if error:
                if kind in ("simple_yoy", "simple_prepare_pvpro"):
                    state["simple_done"] = True
                    result = {"simple": {"rate": None, "error": error}}
                else:
                    if kind == "advanced_3":
                        state["adv"]["3"] = "idle"
                    result = dict(result or {})
                    result["error"] = error
                return state, no_update, no_update, result, no_update
            if kind == "simple_yoy":
                state["simple_done"] = True
                payload = payload or {}
                analysis_result = payload.get("result") or {}
                _update_job_record(job, message="Result validated; committing the simple-analysis result to the browser.")
                _ANALYSIS_JOBS[analysis_id] = job
                _append_pipeline_log(state, "COMMIT", "Simple-analysis result committed.",
                                     level="done", job_id=analysis_id, worker_pid=os.getpid())
                # Do not send the complete dataframe back a second time. The
                # multi-megabyte duplicate response was the deployed-version
                # failure point; Advanced Step 1 can identify it again later.
                return state, no_update, no_update, analysis_result, no_update
            if kind == "simple_prepare_pvpro":
                payload = payload or {}
                try:
                    # The preparation worker launches PVPRO itself and returns
                    # only a tiny hand-off record. This avoids moving the full
                    # filtered dataframe through a Dash callback response.
                    jid = payload["pvpro_job_id"]
                    _update_job_record(job, handoff_job_id=jid,
                                       message=f"PVPRO worker {jid} is ready; committing hand-off to the browser.")
                    _ANALYSIS_JOBS[analysis_id] = job
                    state["pvpro_job"] = jid
                    state["pvpro_mode"] = "simple"
                    state["pvpro_worker_pid"] = os.getpid()
                    state["pvpro_dur"] = float(payload.get("duration_years", 0.0) or 0.0)
                    state["pvpro_nkept"] = int(payload.get("n_kept", 0) or 0)
                    state["pvpro_window"] = str(payload.get("window", "") or "")
                    state["pvpro_prog"] = {"phase": "starting", "current": 0, "total": 1,
                                           "message": "Starting PVPRO…"}
                    state["simple_done"] = False
                    _append_pipeline_log(state, "HANDOFF", f"Dataset preparation finished; PVPRO job {jid} started.",
                                         level="running", job_id=jid, worker_pid=os.getpid())
                    return state, no_update, no_update, {}, no_update
                except Exception as exc:
                    detail = traceback.format_exc(limit=12)
                    message = f"Could not start PVPRO after dataset preparation: {type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    _append_pipeline_log(state, "HANDOFF", message, level="error", job_id=analysis_id,
                                         worker_pid=os.getpid(), detail=detail)
                    state["simple_done"] = True
                    return state, no_update, no_update, {"simple": {"rate": None, "error": message}}, no_update
            if kind == "advanced_1":
                payload = payload or {}
                data = dict(payload.get("identified_data") or data or EMPTY_DATA)
                data["prescreen_figs"] = payload.get("figs") or []
                state["adv"]["1"] = "done"
                state["adv"]["2"] = "idle"
                state["adv"]["3"] = state["adv"]["4"] = "locked"
                return state, data, no_update, no_update, no_update
            if kind == "advanced_2":
                state["adv"]["2"] = "done"
                state["adv"]["3"] = "idle"
                state["adv"]["4"] = "locked"
                state["filt_open"] = False
                return state, no_update, payload or {}, no_update, no_update
            if kind == "advanced_3":
                state["adv"]["3"] = "done"
                state["adv"]["4"] = "idle"
                return state, no_update, no_update, payload or {}, no_update

    return (no_update,) * 5


app.clientside_callback(
    """
    function(values, ids, data) {
        var q = (data && data.quality_tags) || {};
        var base = {position: "absolute", right: "58px", top: "50%", transform: "translateY(-50%)",
                    pointerEvents: "none", zIndex: 5, fontSize: "10.5px", fontWeight: 600, lineHeight: "1",
                    borderRadius: "980px", padding: "3px 8px", whiteSpace: "nowrap"};
        var texts = [], styles = [];
        (ids || []).forEach(function (id, i) {
            var v = values[i];
            var tag = v ? (q[v] || "") : "";
            if (!tag) { texts.push(""); styles.push({display: "none"}); return; }
            var neutral = /per-device|^one\\s/i.test(tag);
            var st = Object.assign({}, base, {display: "inline-block",
                color: neutral ? "#57606a" : "#8a6d00",
                background: neutral ? "#f1f3f5" : "#fff6e0",
                border: "1px solid " + (neutral ? "#d7dce0" : "#f0dfa8")});
            texts.push((neutral ? "Note: " : "Warning: ") + tag);
            styles.push(st);
        });
        return [texts, styles];
    }
    """,
    Output({"type": "mappill", "index": ALL}, "children"),
    Output({"type": "mappill", "index": ALL}, "style"),
    Input({"type": "mapsel", "index": ALL}, "value"),
    State({"type": "mapsel", "index": ALL}, "id"),
    State("data-store", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(contents, exClicks) {
        var t = (window.dash_clientside.callback_context.triggered || []);
        if (!t.length || !t[0].value) return window.dash_clientside.no_update;
        var ov = document.getElementById('upload-overlay');
        if (!ov) return window.dash_clientside.no_update;
        ov.classList.add('show');
        var start = Date.now();
        if (window._pvcUp) clearInterval(window._pvcUp);
        var tick = function () {
            var s = Math.floor((Date.now() - start) / 1000);
            var tx = document.getElementById('upload-text');
            if (tx) tx.innerText = 'Loading data\\u2026 ' + s + 's';
        };
        tick();
        window._pvcUp = setInterval(tick, 1000);
        return 1;
    }
    """,
    Output("busy-sink2", "data"),
    Input("upload-data", "contents"),
    Input({"type": "example", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(state) {
        if (state && state.ingest_job) return window.dash_clientside.no_update;
        var ov = document.getElementById('upload-overlay');
        if (ov) ov.classList.remove('show');
        if (window._pvcUp) { clearInterval(window._pvcUp); window._pvcUp = null; }
        return 1;
    }
    """,
    Output("busy-sink2", "data", allow_duplicate=True),
    Input("app-state", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(values, ids) {
        var el = document.getElementById('apply-wrap');
        if (!el) return window.dash_clientside.no_update;
        var orig = {};
        try { orig = JSON.parse(el.getAttribute('data-orig') || '{}'); } catch (e) {}
        var changed = false;
        for (var i = 0; i < (ids || []).length; i++) {
            var role = ids[i].index;
            var cur = (values[i] == null) ? '' : values[i];
            var o = (orig[role] == null) ? '' : orig[role];
            if (cur !== o) { changed = true; break; }
        }
        el.style.display = changed ? 'block' : 'none';
        return window.dash_clientside.no_update;
    }
    """,
    Output("busy-sink3", "data"),
    Input({"type": "mapsel", "index": ALL}, "value"),
    State({"type": "mapsel", "index": ALL}, "id"),
    prevent_initial_call=True,
)


# Live (clientside) update of the variable-mapping status dots + required-warning
# hints the instant a dropdown changes — before the user clicks re-plot.
app.clientside_callback(
    """
    function(values, ids) {
        var req = {"Time": 1, "DC Power": 1};
        return (ids || []).map(function (id, i) {
            var v = values[i];
            var c = v ? "#16a34a" : (req[id.index] ? "#dc2626" : "#a1a1aa");
            return {width: "9px", height: "9px", borderRadius: "50%", background: c,
                    flex: "0 0 auto", marginTop: "10px"};
        });
    }
    """,
    Output({"type": "mapdot", "index": ALL}, "style"),
    Input({"type": "mapsel", "index": ALL}, "value"),
    State({"type": "mapsel", "index": ALL}, "id"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(values, ids) {
        var req = {"Time": 1, "DC Power": 1};
        return (ids || []).map(function (id, i) {
            var show = (!values[i]) && req[id.index];
            if (show) {
                return {display: "block", color: "#b23", fontSize: "11.5px",
                        fontWeight: 600, marginTop: "6px"};
            }
            return {display: "none"};
        });
    }
    """,
    Output({"type": "maphint", "index": ALL}, "style"),
    Input({"type": "mapsel", "index": ALL}, "value"),
    State({"type": "mapsel", "index": ALL}, "id"),
    prevent_initial_call=True,
)


def get_layout():
    return _root_layout()


def _alert(msg, kind="error"):
    c = {"error": ("#8a1c1c", "rgba(220,80,80,0.12)", "rgba(220,80,80,0.3)"),
         "ok": ("#1a8f60", "rgba(52,199,140,0.14)", "rgba(52,199,140,0.32)")}[kind]
    return html.Div(msg, style={"margin": "0 0 20px", "padding": "14px 18px", "borderRadius": "16px",
                    "color": c[0], "background": c[1], "border": f"1px solid {c[2]}", "fontSize": "14px", "fontWeight": 600})


# ---- internal view switch (no URL). Fires only on view change so the nav /
#      hero / upload dropzone stay mounted (buttons stay responsive). ----
@app.callback(Output("pvc-page", "children"),
              Input("pvc-view", "data"),
              prevent_initial_call=True)
def render_view(view):
    return home_body()


# ---- fill the example row + workflow slot from the stores (only these update
#      on data/state changes; the surrounding scaffold is untouched) ----
@app.callback(Output("example-row", "children"), Output("workflow-slot", "children"),
              Input("app-state", "data"), Input("data-store", "data"),
              Input("filtered-store", "data"), Input("result-store", "data"),
              Input("pvc-view", "data"))
def render_workflow(state, data, filtered, result, view):
    if view not in (None, "app"):
        return no_update, no_update
    state = state or DEFAULT_STATE
    data = data or EMPTY_DATA
    return example_cards(state), workflow_layout(state, data, filtered or {}, result or {})


@app.callback(Output("modal-view", "data"),
              Input({"type": "nav", "index": ALL}, "n_clicks"),
              Input({"type": "modalclose", "index": ALL}, "n_clicks"),
              prevent_initial_call=True)
def set_modal(nav_clicks, close_clicks):
    trig = ctx.triggered_id
    val = ctx.triggered[0]["value"] if ctx.triggered else None
    if not isinstance(trig, dict) or not val:
        return no_update
    if trig.get("type") == "nav":
        return trig["index"]
    if trig.get("type") == "modalclose":
        return None
    return no_update


@app.callback(Output("modal-root", "children"), Input("modal-view", "data"))
def show_modal(view):
    return render_modal(view) if view else None


# ---- ingest: upload / example -> parse ----
@app.callback(Output("data-store", "data"),
              Output("app-state", "data"),
              Output("filtered-store", "data"),
              Output("result-store", "data"),
              Output("ingest-status", "children"),
              Input("upload-data", "contents"),
              Input({"type": "example", "index": ALL}, "n_clicks"),
              State("upload-data", "filename"),
              State("app-state", "data"),
              prevent_initial_call=True)
def ingest(upload_contents, ex_clicks, filename, previous_state):
    trig = ctx.triggered_id
    if trig is None:
        return (no_update,) * 5
    fresh = dict(DEFAULT_STATE, adv=dict(DEFAULT_STATE["adv"]), filters=dict(DEFAULT_STATE["filters"]),
                 fparams=dict(FILTER_PARAM_DEFAULTS), methods=["YOY"],
                 mparams={**METHOD_PARAM_DEFAULTS, **PVPRO_PARAM_DEFAULTS})
    fresh["pipeline_log"] = list((previous_state or {}).get("pipeline_log") or [])
    # Preserve the complete visible histories of any previous dataset jobs
    # before replacing their active ids with the new upload job.
    previous_state = previous_state or {}
    for stage, key, registry in (
        ("UPLOAD", "ingest_job", _INGEST_JOBS),
        ("ANALYSIS", "analysis_job", _ANALYSIS_JOBS),
        ("PVPRO", "pvpro_job", _PVPRO_JOBS),
        ("DIAGNOSIS", "diag_job", _DIAG_JOBS),
    ):
        old_jid = previous_state.get(key)
        if old_jid:
            _capture_job_history(fresh, stage, old_jid, registry.get(old_jid))
    try:
        if trig == "upload-data":
            if not upload_contents:
                return (no_update,) * 5
            fresh.update(selected="upload", selected_label=filename or "your file")
            short_name = os.path.splitext(os.path.basename(filename or "upload"))[0][:22]
            fresh["current_dataset"] = short_name
            _append_pipeline_log(fresh, "DATASET", f"Started new dataset: {filename or 'uploaded file'}.",
                                 level="running", dataset=short_name)
            fresh["ingest_job"] = _launch_ingest(contents=upload_contents, filename=filename, dataset=short_name)
            fresh["ingest_worker_pid"] = os.getpid()
        elif isinstance(trig, dict) and trig.get("type") == "example":
            if not any(ex_clicks or []):
                return (no_update,) * 5
            ex = next(e for e in EXAMPLES if e["id"] == trig["index"])
            fresh.update(selected=ex["id"], selected_label=ex["label"])
            short_name = ex["label"].replace("System ", "SYS")
            fresh["current_dataset"] = short_name
            _append_pipeline_log(fresh, "DATASET", f"Started new dataset: {ex['label']} ({ex['file']}).",
                                 level="running", dataset=short_name)
            # Built-in parquet examples are local and small. The prior UI read
            # them in this request and felt immediate; routing them through a
            # background thread + 0.9 s polling added an unnecessary second
            # callback and, on deployment, could queue behind monitor polls.
            # Keep the full frame server-side and return only metadata.
            started = time.perf_counter()
            _append_pipeline_log(fresh, "UPLOAD", "Reading the local example file…",
                                 level="running", dataset=short_name)
            raw = load_raw_data(example_file=ex["file"])
            browser_data = _store_dataset(raw)
            elapsed = time.perf_counter() - started
            _append_pipeline_log(
                fresh, "UPLOAD",
                (f"Example ready in {elapsed:.2f}s: {int(raw.get('rows', 0)):,} rows; "
                 "browser received metadata only."),
                level="done", dataset=short_name)
            fresh["ingest_job"] = None
            fresh["ingest_worker_pid"] = None
            return browser_data, fresh, {}, {}, html.Div()
        else:
            return (no_update,) * 5
    except Exception as e:
        traceback.print_exc()
        return EMPTY_DATA, no_update, {}, {}, _alert(f"Could not read the data: {e}")
    return EMPTY_DATA, fresh, {}, {}, html.Div()


# ---- main interactions (mode / tabs / filters / methods / runs / actions) ----
@app.callback(
    Output("app-state", "data", allow_duplicate=True),
    Output("data-store", "data", allow_duplicate=True),
    Output("filtered-store", "data", allow_duplicate=True),
    Output("result-store", "data", allow_duplicate=True),
    Input({"type": "mode", "index": ALL}, "n_clicks"),
    Input({"type": "adv-tab", "index": ALL}, "n_clicks"),
    Input({"type": "filter", "index": ALL}, "n_clicks"),
    Input({"type": "method", "index": ALL}, "n_clicks"),
    Input({"type": "smpmethod", "index": ALL}, "n_clicks"),
    Input({"type": "pvprofig", "index": ALL}, "n_clicks"),
    Input({"type": "run-step", "index": ALL}, "n_clicks"),
    Input({"type": "act", "index": ALL}, "n_clicks"),
    Input({"type": "fparam", "index": ALL}, "value"),
    Input({"type": "mparam", "index": ALL}, "value"),
    State("app-state", "data"), State("data-store", "data"),
    State("filtered-store", "data"), State("result-store", "data"),
    State({"type": "fparam", "index": ALL}, "id"),
    State({"type": "mparam", "index": ALL}, "id"),
    State({"type": "mapsel", "index": ALL}, "value"), State({"type": "mapsel", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def interactions(mode_c, tab_c, filter_c, method_c, smp_c, pfig_c, step_c, act_c, fpv, mpv,
                 state, data, filtered, result, fpi, mpi, mspv, mspi):
    trig = ctx.triggered_id
    val = ctx.triggered[0]["value"] if ctx.triggered else None
    if not isinstance(trig, dict):
        return (no_update,) * 4
    ttype, idx = trig["type"], trig["index"]
    if ttype not in ("fparam", "mparam") and not val:
        return (no_update,) * 4
    state = dict(state); data = data or EMPTY_DATA
    try:
        data = _materialize_dataset(data)
    except Exception as exc:
        traceback.print_exc()
        result = dict(result or {})
        result["error"] = str(exc)
        return state, no_update, no_update, result
    filtered = dict(filtered or {}); result = dict(result or {})
    state["filters"] = dict(state.get("filters", {}))
    state["adv"] = dict(state.get("adv", {}))
    _enforce_adv_sequence(state)
    # merge any visible param inputs into state
    old_fp = dict(state.get("fparams", FILTER_PARAM_DEFAULTS))
    fp = dict(old_fp)
    for i, idd in zip(fpv or [], fpi or []):
        if i is not None:
            fp[idd["index"]] = i
    old_mp = dict(state.get("mparams", {}))
    mp = dict(old_mp)
    for i, idd in zip(mpv or [], mpi or []):
        if i is not None:
            mp[idd["index"]] = i
    state["fparams"], state["mparams"] = fp, mp

    if ttype == "fparam":
        changed = any(fp.get(idd["index"]) != old_fp.get(idd["index"]) for idd in (fpi or []))
        if not changed:
            return (no_update,) * 4
        _reset_advanced_from(state, 2)
        return state, no_update, {}, {}

    if ttype == "mparam":
        changed = any(mp.get(idd["index"]) != old_mp.get(idd["index"]) for idd in (mpi or []))
        if not changed:
            return (no_update,) * 4
        _reset_advanced_from(state, 3)
        return state, no_update, no_update, {}

    if ttype == "mode":
        state["mode"] = idx
        return state, no_update, no_update, no_update

    if ttype == "adv-tab":
        if state["adv"].get(str(idx)) != "locked":
            state["adv_tab"] = int(idx)
        return state, no_update, no_update, no_update

    if ttype == "filter":
        state["filters"][idx] = not state["filters"].get(idx, True)
        _reset_advanced_from(state, 2)
        return state, no_update, {}, {}

    if ttype == "pvprofig":
        state["pvpro_fig_sel"] = idx
        return state, no_update, no_update, no_update

    if ttype == "smpmethod":
        state["simple_method"] = idx
        return state, no_update, no_update, no_update

    if ttype == "method":
        ms = list(state.get("methods", []))
        if idx == "PVPRO":
            ms = [] if "PVPRO" in ms else ["PVPRO"]          # PVPRO is exclusive
        else:
            if idx in ms:
                ms = [x for x in ms if x != idx]
            else:
                ms = [x for x in ms if x != "PVPRO"] + [idx]   # picking a stat drops PVPRO
        state["methods"] = ms or ["YOY"]
        _reset_advanced_from(state, 3)
        return state, no_update, no_update, {}

    if ttype == "act":
        if idx == "stop-ingest":
            jid = state.get("ingest_job")
            if jid:
                _cancel_async_job(_INGEST_JOBS, jid)
                _INGEST_JOBS.pop(jid, None)
            state["ingest_job"] = None
            return state, no_update, no_update, no_update
        if idx == "stop-analysis":
            jid = state.get("analysis_job")
            if jid:
                _cancel_async_job(_ANALYSIS_JOBS, jid)
                _ANALYSIS_JOBS.pop(jid, None)
            state["analysis_job"] = None
            state["analysis_scope"] = None
            if state.get("adv", {}).get("3") == "running_async":
                state["adv"]["3"] = "idle"
            return state, no_update, no_update, no_update
        if idx == "new-analysis":
            fresh_state = dict(DEFAULT_STATE, adv=dict(DEFAULT_STATE["adv"]), filters=dict(DEFAULT_STATE["filters"]),
                         fparams=dict(FILTER_PARAM_DEFAULTS), methods=["YOY"],
                         mparams={**METHOD_PARAM_DEFAULTS, **PVPRO_PARAM_DEFAULTS})
            fresh_state["pipeline_log"] = list(state.get("pipeline_log") or [])
            fresh_state["current_dataset"] = state.get("current_dataset", "")
            _append_pipeline_log(fresh_state, "SESSION", "Started a new analysis; prior history was retained.")
            return fresh_state, EMPTY_DATA, {}, {}
        if idx == "open-advanced":
            state["mode"] = "advanced"
            return state, no_update, no_update, no_update
        if idx == "toggle-params":
            state["show_params"] = not state.get("show_params", False)
            return state, no_update, no_update, no_update
        if idx == "retune":
            _reset_advanced_from(state, 2); state["adv_tab"] = 2
            return state, no_update, {}, {}
        if idx == "remetric":
            _reset_advanced_from(state, 3); state["adv_tab"] = 3
            return state, no_update, no_update, {}
        if idx == "unfold-filters":
            state["filt_open"] = True
            return state, no_update, no_update, no_update
        if idx == "fold-filters":
            state["filt_open"] = False
            return state, no_update, no_update, no_update
        if idx == "unfold-metrics":
            state["metric_open"] = True
            return state, no_update, no_update, no_update
        if idx == "fold-metrics":
            state["metric_open"] = False
            return state, no_update, no_update, no_update
        if idx == "select-all-methods":
            has_all = all(m in state["methods"] for m in STAT_METHODS)
            state["methods"] = ["YOY"] if has_all else list(STAT_METHODS)
            _reset_advanced_from(state, 3)
            return state, no_update, no_update, {}
        if idx == "simple-reset":
            state["simple_done"] = False; state["pvpro_job"] = None; state["pvpro_prog"] = None
            state["pvpro_est_keys"] = []
            return state, no_update, no_update, {}
        if idx == "goto-next":
            nt = min(int(state.get("adv_tab", 1)) + 1, 4)
            if state["adv"].get(str(nt)) != "locked":
                state["adv_tab"] = nt
            return state, no_update, no_update, no_update
        if idx == "stop-pvpro":
            jid = state.get("pvpro_job")
            if jid:
                _PVPRO_JOBS.pop(jid, None)
            state["pvpro_job"] = None; state["pvpro_prog"] = None; state["adv"]["3"] = "idle"
            return state, no_update, no_update, no_update
        if idx == "estimate-pvpro":
            if not data.get("loaded") or not data.get("identified"):
                return no_update, no_update, no_update, no_update
            result = {}
            try:
                src = _df_from_json(filtered["df_good"]) if filtered.get("df_good") else _df_from_json(data["df"])
                params = estimate_pvpro_params(src, data.get("mapped") or {}, cells_in_series=_num(mp.get("cells"), 60))
                notes = []
                est_keys = []
                if isinstance(params.get("modules_per_string"), dict):
                    state["mparams"]["mps"] = params["modules_per_string"]["value"]; est_keys.append("mps")
                    notes.append(f"modules/string = {params['modules_per_string']['value']} ({params['modules_per_string'].get('basis','')})")
                if isinstance(params.get("parallel_strings"), dict):
                    state["mparams"]["ps"] = params["parallel_strings"]["value"]; est_keys.append("ps")
                    notes.append(f"parallel strings = {params['parallel_strings']['value']} ({params['parallel_strings'].get('basis','')})")
                state["pvpro_est_keys"] = est_keys
                result["pvpro_estimated"] = "\u2713 Estimated from data \u2014 " + ("; ".join(notes) if notes else "no layout could be inferred (check V/I columns).")
                _reset_advanced_from(state, 3)
            except Exception as e:
                result["pvpro_estimated"] = f"Could not estimate: {e}"
            return state, no_update, no_update, result
        if idx == "apply-mapping":
            if not data.get("loaded"):
                return no_update, no_update, no_update, no_update
            try:
                df = _df_from_json(data["df"])
                new_mapped = dict(data.get("mapped") or {})
                for v, idd in zip(mspv or [], mspi or []):
                    new_mapped[idd["index"]] = v if v else None
                data = dict(data)
                data["mapped"] = new_mapped
                if new_mapped.get("Irradiance"):
                    data["irra_key"] = new_mapped["Irradiance"]
                data["mapping"] = _build_mapping(df, new_mapped)
                data["has_vi"] = bool(new_mapped.get("DC Voltage") and new_mapped.get("DC Current"))
                figs, _e = make_overview_figures(df, new_mapped)
                jfigs = []
                for g in (figs or []):
                    try:
                        f = g.figure if hasattr(g, "figure") else g
                        jfigs.append(go.Figure(f).to_json())
                    except Exception:
                        pass
                data["prescreen_figs"] = jfigs
                # a mapping change resets downstream results, but Step 2 stays
                # available as long as the required roles (Time + DC Power) exist
                has_required = bool(new_mapped.get("Time")) and bool(new_mapped.get("DC Power"))
                state["adv"]["3"] = "locked"; state["adv"]["4"] = "locked"
                state["adv"]["2"] = "idle" if has_required else "locked"
                state["adv"]["1"] = "done" if has_required else "idle"; state["adv_tab"] = 1
                return state, _store_dataset(data, data.get("dataset_id")), {}, {}
            except Exception as e:
                traceback.print_exc()
                result["error"] = str(e)
                return state, data, no_update, result
        if idx == "run-simple":
            if not data.get("loaded"):
                return no_update, no_update, no_update, no_update
            if state.get("simple_method") != "PVPRO":
                data_snapshot = dict(data)

                def _simple_work(progress):
                    progress(0, 12, "Preparing the uploaded time series…")
                    identified0 = data_snapshot
                    if not identified0.get("identified"):
                        raw_df0 = _df_from_json(data_snapshot["df"])
                        identified0 = run_parse(
                            df=raw_df0, filename=data_snapshot.get("filename"),
                            progress_callback=lambda c, _t, m: progress(c, 12, m))
                    df0 = _df_from_json(identified0["df"])
                    dg0, n_raw0, n_kept0, _pie0, _power0 = apply_filter_chain(
                        df0, identified0["mapped"], identified0["irra_key"],
                        DEFAULT_STATE["filters"], FILTER_PARAM_DEFAULTS,
                        progress_callback=lambda c, _t, m: progress(c + 4, 12, m))
                    if dg0.empty:
                        raise ValueError("No points survived default filtering.")
                    start0, end0 = dg0.index.min(), dg0.index.max()
                    dur0 = (end0 - start0).days / 365.25 if hasattr(end0 - start0, "days") else 0.0
                    try:
                        win0 = f"{pd.Timestamp(start0):%Y-%m-%d} \u2192 {pd.Timestamp(end0):%Y-%m-%d}"
                    except Exception:
                        win0 = ""
                    progress(9, 10, "Aggregating daily normalized power…")
                    daily0 = aggregate_daily(dg0, identified0["irra_key"])
                    progress(10, 10, "Computing the year-on-year degradation rate…")
                    rate0, fig0 = compute_yoy(daily0, rolling_window=30, iqr_multiplier=1.5)
                    fig_json = fig0.to_json() if fig0 is not None else None
                    analysis_result0 = {"simple": {"rate": float(rate0) if rate0 == rate0 else None,
                                       "fig": fig_json, "duration_years": float(dur0),
                                       "n_kept": n_kept0, "window": win0,
                                       "pct_kept": (n_kept0 / n_raw0 * 100) if n_raw0 else 0.0},
                            "multi": {"YOY": {"rate": float(rate0) if rate0 == rate0 else None,
                                                "fig": fig_json}},
                            "duration_years": float(dur0), "n_kept": n_kept0, "window": win0}
                    # Keep the completed payload deliberately small. The raw
                    # dataframe is already present in data-store.
                    return {"result": analysis_result0}

                # Simple YOY is intentionally executed inside this request.
                # On representative 150k-row uploads the vectorized filter,
                # daily aggregation and YOY fit complete in about one second.
                # Running this short task in a daemon thread is less reliable
                # on Gunicorn/container deployments: a worker recycle can kill
                # the thread while its last diskcache status remains "running"
                # forever. The clientside busy mask still appears immediately,
                # and every stage is copied into the persistent activity log.
                sync_id = "sync-" + uuid.uuid4().hex[:8]
                sync_started = time.perf_counter()

                def _simple_sync_progress(current, total, message):
                    _append_pipeline_log(
                        state, "ANALYSIS", message, level="running",
                        job_id=sync_id, worker_pid=os.getpid(),
                        dataset=state.get("current_dataset"))

                state["analysis_job"] = None
                state["analysis_worker_pid"] = None
                state["analysis_scope"] = None
                state["simple_done"] = False
                try:
                    _append_pipeline_log(
                        state, "ANALYSIS", "Running Simple YOY in the request worker…",
                        level="running", job_id=sync_id, worker_pid=os.getpid(),
                        dataset=state.get("current_dataset"))
                    payload0 = _simple_work(_simple_sync_progress) or {}
                    analysis_result0 = payload0.get("result") or {}
                    elapsed0 = time.perf_counter() - sync_started
                    _append_pipeline_log(
                        state, "ANALYSIS",
                        f"Simple YOY completed in {elapsed0:.2f}s; result committed.",
                        level="done", job_id=sync_id, worker_pid=os.getpid(),
                        dataset=state.get("current_dataset"))
                    state["simple_done"] = True
                    return state, no_update, no_update, analysis_result0
                except Exception as exc:
                    traceback.print_exc()
                    elapsed0 = time.perf_counter() - sync_started
                    message0 = f"Simple YOY failed after {elapsed0:.2f}s: {type(exc).__name__}: {exc}"
                    _append_pipeline_log(
                        state, "ANALYSIS", message0, level="error",
                        job_id=sync_id, worker_pid=os.getpid(),
                        detail=traceback.format_exc(limit=20),
                        dataset=state.get("current_dataset"))
                    state["simple_done"] = True
                    return state, no_update, no_update, {"simple": {"rate": None, "error": message0}}
            data_snapshot = dict(data)
            mp_snapshot = dict(mp)

            def _prepare_pvpro(progress):
                progress(0, 12, "Preparing the uploaded time series…")
                identified0 = data_snapshot
                if not identified0.get("identified"):
                    raw_df0 = _df_from_json(data_snapshot["df"])
                    identified0 = run_parse(
                        df=raw_df0, filename=data_snapshot.get("filename"),
                        progress_callback=lambda c, _t, m: progress(c, 12, m))
                if not identified0.get("has_vi"):
                    raise ValueError("PVPRO needs DC Voltage + DC Current columns, which weren't identified.")
                df0 = _df_from_json(identified0["df"])
                dg0, _n_raw0, n_kept0, _pie0, _power0 = apply_filter_chain(
                    df0, identified0["mapped"], identified0["irra_key"],
                    DEFAULT_STATE["filters"], FILTER_PARAM_DEFAULTS,
                    progress_callback=lambda c, _t, m: progress(c + 4, 12, m))
                if dg0.empty:
                    raise ValueError("No points survived default filtering.")
                start0, end0 = dg0.index.min(), dg0.index.max()
                dur0 = (end0 - start0).days / 365.25 if hasattr(end0 - start0, "days") else 0.0
                try:
                    win0 = f"{pd.Timestamp(start0):%Y-%m-%d} \u2192 {pd.Timestamp(end0):%Y-%m-%d}"
                except Exception:
                    win0 = ""
                try:
                    est0 = estimate_pvpro_params(
                        dg0, identified0["mapped"], cells_in_series=_num(mp_snapshot.get("cells"), 60))
                    if isinstance(est0.get("modules_per_string"), dict):
                        mp_snapshot["mps"] = est0["modules_per_string"]["value"]
                    if isinstance(est0.get("parallel_strings"), dict):
                        mp_snapshot["ps"] = est0["parallel_strings"]["value"]
                except Exception:
                    pass
                kwargs0 = dict(cells_in_series=_num(mp_snapshot.get("cells"), 60),
                               modules_per_string=_num(mp_snapshot.get("mps"), 1),
                               parallel_strings=_num(mp_snapshot.get("ps"), 1),
                               alpha_isc=_num(mp_snapshot.get("alphaisc"), 0.0046),
                               technology=mp_snapshot.get("tech") or "mono-c-Si",
                               days_per_run=_num(mp_snapshot.get("days"), 14),
                               iterations_per_year=_num(mp_snapshot.get("iters"), 12))
                progress(12, 12, "Starting the PVPRO fitting worker…")
                pvpro_job_id0 = _launch_pvpro(
                    dg0, identified0["mapped"], kwargs0,
                    dataset=state.get("current_dataset"))
                return {"pvpro_job_id": pvpro_job_id0,
                        "duration_years": float(dur0), "n_kept": int(n_kept0), "window": win0}

            state["analysis_job"] = _launch_analysis(
                "simple_prepare_pvpro", _prepare_pvpro, dataset=state.get("current_dataset"))
            state["analysis_worker_pid"] = os.getpid()
            state["analysis_scope"] = "simple"
            state["simple_done"] = False
            return state, no_update, no_update, {}
        if idx == "diagnose":
            multi = (result or {}).get("multi") or {}
            uses_pvpro = "PVPRO" in multi
            context = _context_str(data, result)
            jid = uuid.uuid4().hex[:10]
            now = time.time()
            _DIAG_JOBS[jid] = {"done": False, "phase": "running", "message": "Preparing AI diagnosis…",
                               "text": None, "error": None, "current": 0, "total": 1,
                               "started_at": now, "updated_at": now, "worker_pid": os.getpid(),
                               "dataset": state.get("current_dataset") or "dataset",
                               "history": [{"at": now, "phase": "running", "message": "Preparing AI diagnosis…"}]}

            def _dworker(jid=jid, ctx=context, pv=uses_pvpro):
                try:
                    dj = _DIAG_JOBS.get(jid)
                    diag_started = time.time()
                    _update_job_record(
                        dj, message=(f"AI diagnosis request: model={_LLM_MODEL or 'rule-based fallback'}; "
                                     f"context={len(ctx):,} characters; PVPRO context={'yes' if pv else 'no'}."))
                    _DIAG_JOBS[jid] = dj
                    if _LLM_CLIENT is not None and _LLM_MODEL:
                        sysmsg = _DIAG_SYS_PVPRO if pv else _DIAG_SYS
                        resp = _LLM_CLIENT.chat.completions.create(model=_LLM_MODEL, messages=[
                            {"role": "system", "content": sysmsg},
                            {"role": "user", "content": "Analysis context:\n" + ctx + "\n\nGive a concise diagnosis."}],
                            timeout=60)
                        txt = resp.choices[0].message.content.strip()
                        usage = getattr(resp, "usage", None)
                        token_text = (f"; prompt tokens={getattr(usage, 'prompt_tokens', '?')}; "
                                      f"completion tokens={getattr(usage, 'completion_tokens', '?')}") if usage else ""
                    else:
                        txt = "**Rule-based read** (set `OPENAI_API_KEY` for a full AI diagnosis):\n\n" + ctx
                        token_text = "; no LLM call"
                    dj = _DIAG_JOBS.get(jid)
                    _update_job_record(dj, done=True, phase="done", current=1, total=1,
                                       message=(f"AI diagnosis completed in {time.time() - diag_started:.1f}s"
                                                f"{token_text}; response={len(txt):,} characters."), text=txt)
                    _DIAG_JOBS[jid] = dj
                except Exception as e:
                    dj = _DIAG_JOBS.get(jid) or {}
                    _update_job_record(dj, done=True, phase="error",
                                       message=f"AI diagnosis failed: {type(e).__name__}: {e}",
                                       error=str(e), detail=traceback.format_exc(limit=20),
                                       text=f"(AI diagnosis unavailable: {e})")
                    _DIAG_JOBS[jid] = dj

            threading.Thread(target=_dworker, daemon=True).start()
            state["diag_job"] = jid
            result = dict(result or {}); result["diagnosing"] = True; result.pop("diagnosis", None)
            return state, no_update, no_update, result
        return (no_update,) * 4

    if ttype == "run-step":
        n = int(idx)
        if state["adv"].get(str(n)) == "locked" or not data.get("loaded"):
            return no_update, no_update, no_update, no_update
        try:
            if n == 1:
                _reset_advanced_from(state, 1)
                data_snapshot = dict(data)

                def _prescreen_work(progress):
                    progress(0, 6, "Preparing the uploaded time series…")
                    identified0 = data_snapshot
                    if not identified0.get("identified"):
                        raw_df0 = _df_from_json(data_snapshot["df"])
                        identified0 = run_parse(
                            df=raw_df0, filename=data_snapshot.get("filename"),
                            progress_callback=lambda c, _t, m: progress(c, 6, m))
                    df0 = _df_from_json(identified0["df"])
                    progress(5, 6, "Building raw-signal quality figures…")
                    figs0, _e0 = make_overview_figures(df0, identified0["mapped"])
                    jfigs0 = []
                    progress(6, 6, "Serializing prescreening figures…")
                    for graph0 in (figs0 or []):
                        try:
                            fig0 = graph0.figure if hasattr(graph0, "figure") else graph0
                            jfigs0.append(go.Figure(fig0).to_json())
                        except Exception:
                            pass
                    browser_identified0 = _store_dataset(
                        identified0, identified0.get("dataset_id") or data_snapshot.get("dataset_id"))
                    return {"identified_data": browser_identified0, "figs": jfigs0}

                state["analysis_job"] = _launch_analysis(
                    "advanced_1", _prescreen_work, dataset=state.get("current_dataset"))
                state["analysis_worker_pid"] = os.getpid()
                state["analysis_scope"] = "advanced"
                return state, no_update, {}, {}
            if n == 2:
                _reset_advanced_from(state, 2)
                data_snapshot = dict(data)
                filters_snapshot = dict(state["filters"])
                fp_snapshot = dict(fp)

                def _filter_work(progress):
                    progress(0, 7, "Restoring the prescreened time series…")
                    df0 = _df_from_json(data_snapshot["df"])
                    dg0, n_raw0, n_kept0, pie0, power0 = apply_filter_chain(
                        df0, data_snapshot["mapped"], data_snapshot["irra_key"],
                        filters_snapshot, fp_snapshot, progress_callback=progress)
                    if dg0.empty:
                        raise ValueError("No points survived filtering — loosen the thresholds.")
                    return {"df_good": _df_to_json(dg0), "n_raw": n_raw0, "n_kept": n_kept0,
                            "pie": pie0, "power": power0}

                state["analysis_job"] = _launch_analysis(
                    "advanced_2", _filter_work, dataset=state.get("current_dataset"))
                state["analysis_worker_pid"] = os.getpid()
                state["analysis_scope"] = "advanced"
                return state, no_update, {}, {}
            if n == 3:
                if not filtered.get("df_good"):
                    raise ValueError("Run filtering first.")
                _reset_advanced_from(state, 3)
                state["metric_open"] = False
                methods = state.get("methods") or ["YOY"]
                dg = _df_from_json(filtered["df_good"])
                start, end = dg.index.min(), dg.index.max()
                dur = (end - start).days / 365.25 if hasattr(end - start, "days") else 0.0
                try:
                    win = f"{pd.Timestamp(start):%Y-%m-%d} \u2192 {pd.Timestamp(end):%Y-%m-%d}"
                except Exception:
                    win = ""
                if "PVPRO" in methods:
                    if not data.get("has_vi"):
                        raise ValueError("PVPRO needs DC Voltage + DC Current columns, which weren't identified.")
                    kwargs = dict(cells_in_series=_num(mp.get("cells"), 60), modules_per_string=_num(mp.get("mps"), 1),
                                  parallel_strings=_num(mp.get("ps"), 1), alpha_isc=_num(mp.get("alphaisc"), 0.0046),
                                  technology=mp.get("tech") or "mono-c-Si", days_per_run=_num(mp.get("days"), 14),
                                  iterations_per_year=_num(mp.get("iters"), 12))
                    jid = _launch_pvpro(
                        dg, data["mapped"], kwargs, dataset=state.get("current_dataset"))
                    state["adv"]["3"] = "running"; state["pvpro_job"] = jid; state["pvpro_mode"] = "advanced"
                    state["pvpro_worker_pid"] = os.getpid()
                    state["pvpro_dur"] = float(dur); state["pvpro_nkept"] = filtered.get("n_kept", 0); state["pvpro_window"] = win
                    state["pvpro_prog"] = {"phase": "starting", "current": 0, "total": 1, "message": "Starting PVPRO\u2026"}
                    return state, no_update, no_update, {}
                mapped_snapshot = dict(data["mapped"])
                mp_snapshot = dict(mp)
                irra_snapshot = data["irra_key"]
                nkept_snapshot = filtered.get("n_kept", 0)

                def _metric_work(progress):
                    multi0 = run_methods(dg, irra_snapshot, methods, mp_snapshot,
                                         mapped=mapped_snapshot, progress_callback=progress)
                    return {"multi": multi0, "duration_years": float(dur),
                            "n_kept": nkept_snapshot, "window": win}

                state["analysis_job"] = _launch_analysis(
                    "advanced_3", _metric_work, dataset=state.get("current_dataset"))
                state["analysis_worker_pid"] = os.getpid()
                state["analysis_scope"] = "advanced"
                state["adv"]["3"] = "running_async"
                return state, no_update, no_update, {}
            if n == 4:
                sel_filters = [k for k, v in state["filters"].items() if v]
                metric = (state.get("methods") or ["YOY"])[0]
                try:
                    code = get_full_code(data.get("filename", "data"), data["mapped"], sel_filters, metric)
                except Exception:
                    code = _fallback_code(data, state, metric)
                result["code"] = code
                state["adv"]["4"] = "done"
                return state, no_update, no_update, result
        except Exception as e:
            traceback.print_exc()
            result["error"] = str(e)
            return state, no_update, no_update, result

    return (no_update,) * 4


def _fallback_code(data, state, metric):
    mapped = data.get("mapped", {})
    fsel = [k for k, v in state["filters"].items() if v]
    disp = {"YOY": "compute_yoy", "LR": "compute_lr", "HW": "compute_hw",
            "ARIMA": "compute_arima", "CSD": "compute_csd", "PVPRO": "compute_pvpro"}.get(metric, "compute_yoy")
    lines = ["import pandas as pd",
             "from page_supporting_files.analysis_utils_260701 import (",
             f"    normalize, low_irra_power_filter, aggregate_daily, {disp},)",
             "from page_supporting_files.pvcopilot_filter_functions import (",
             "    basic_value_filter, clear_sky_filter, identify_outliers_iqr,)",
             "",
             f'df = pd.read_parquet("{data.get("filename", "data")}")',
             f"mapped = {mapped!r}",
             f'irra_key = "{data.get("irra_key", "")}"',
             "",
             "bv, _ = basic_value_filter(df, mapped); df = df.loc[bv].copy()"]
    if "clearsky" in fsel:
        lines.append("cs, _ = clear_sky_filter(df, irra_key, smoothness_threshold=0.3, energy_threshold=0.5)")
        lines.append("df = df.loc[df.index.isin(cs)]")
    lines.append(f"df = normalize(df, mapped, gamma={-0.004 if 'tempcorr' in fsel else 0.0})")
    if "irr" in fsel:
        lines.append("keep, _ = low_irra_power_filter(df, mapped, irr_thresh=300, power_ratio=0.02, norm_lower=0.01, norm_upper_pct=99)")
        lines.append("df = df.loc[df.index.isin(keep)]")
    lines += ['ni, _ = identify_outliers_iqr(df, "norm", iqr_multiplier=1.5); df = df.loc[df.index.isin(ni)]',
              "daily = aggregate_daily(df, irra_key)",
              f"rate, fig = {disp}(daily)",
              'print(f"Degradation rate: {rate:.2f} %/yr")']
    return "\n".join(lines)


# ---- chat ----
@app.callback(
    Output("chat-state", "data"), Output("chat-drawer", "children"),
    Input("chat-open", "n_clicks"),
    Input({"type": "chatbtn", "index": ALL}, "n_clicks"),
    Input({"type": "chatbox", "index": ALL}, "n_submit"),
    Input({"type": "chipq", "index": ALL}, "n_clicks"),
    State({"type": "chatbox", "index": ALL}, "value"),
    State("chat-state", "data"), State("data-store", "data"), State("result-store", "data"),
    prevent_initial_call=True,
)
def chat_cb(open_c, btn_clicks, submit_list, chip_clicks, text_list, cstate, data, result):
    trig = ctx.triggered_id
    val = ctx.triggered[0]["value"] if ctx.triggered else None
    cstate = dict(cstate); cstate["messages"] = list(cstate["messages"])
    text = (text_list or [None])[0]

    def _ask(q):
        cstate["messages"].append({"role": "user", "text": q})
        cstate["messages"].append({"role": "bot", "text": bot_reply(q, data or {}, result or {})})
        cstate["draft"] = ""

    if trig == "chat-open" and val:
        cstate["open"] = True
    elif isinstance(trig, dict) and trig.get("type") == "chatbtn" and val:
        if trig["index"] == "close":
            cstate["open"] = False
        elif trig["index"] == "send" and text and text.strip():
            _ask(text.strip())
    elif isinstance(trig, dict) and trig.get("type") == "chatbox" and val:
        if text and text.strip():
            _ask(text.strip())
    elif isinstance(trig, dict) and trig.get("type") == "chipq" and val:
        q = _CHAT_EXAMPLES[trig["index"]] if 0 <= trig["index"] < len(_CHAT_EXAMPLES) else None
        if q:
            cstate["draft"] = q          # copy into the input; user presses send
    else:
        return no_update, no_update
    return cstate, chat_drawer(cstate)
