"""
Load and summarise pvtools usage logs written by utils/usage_log.py.
Used by the admin dashboard (pages/admin_usage.py) and from the command line:

    python -m utils.usage_report                     # local ./usage_logs
    python -m utils.usage_report --s3 pvtools        # s3://pvtools/usage-logs/
    python -m utils.usage_report --s3 pvtools --days 90 --csv usage.csv
"""
import argparse
import datetime as dt
import glob
import gzip
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

try:
    from utils import usage_db
except ImportError:          # run as a plain script from inside utils/
    import usage_db

TZ = "America/Los_Angeles"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS = ["ts", "vid", "ip", "tool", "event", "trigger", "status", "ms", "kb",
           "ua", "ref", "country", "city", "region", "lat", "lon", "net", "err", "x"]

# app-level facts attached with usage_log.note(); flattened to x_<key> columns
X_KEYS = ["stage", "mode", "method", "source", "dataset", "rows", "rows_kept", "cols",
          "span_days", "downsized", "filters", "run_s", "status"]

# S3 log objects never change once written, so each one is downloaded once per
# process and kept here; a refresh only fetches the keys that are new.
_OBJ_CACHE = {}
_OBJ_LOCK = threading.Lock()


def _parse(blob):
    rows = []
    for line in blob.decode("utf-8", "ignore").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def load_local(folder=None):
    folder = folder or os.environ.get("USAGE_LOG_DIR", os.path.join(_ROOT, "usage_logs"))
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        with open(path, "rb") as f:
            rows.extend(_parse(f.read()))
    return rows


def load_s3(bucket, prefix="usage-logs", days=30):
    import boto3
    s3 = boto3.client("s3")
    prefix = prefix.strip("/") + "/"
    today = dt.datetime.now(dt.timezone.utc).date()
    keys = []
    pager = s3.get_paginator("list_objects_v2")
    for i in range(days + 1):
        day = today - dt.timedelta(days=i)
        for page in pager.paginate(Bucket=bucket, Prefix=f"{prefix}{day:%Y/%m/%d}/"):
            keys.extend(o["Key"] for o in page.get("Contents", []))

    with _OBJ_LOCK:
        missing = [k for k in keys if k not in _OBJ_CACHE]

    def fetch(key):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return key, _parse(gzip.decompress(body))

    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for key, rows in pool.map(fetch, missing):
                with _OBJ_LOCK:
                    _OBJ_CACHE[key] = rows
    rows = []
    with _OBJ_LOCK:
        for k in keys:
            rows.extend(_OBJ_CACHE.get(k, []))
    return rows


def load_events(days=30, bucket=None, prefix=None, local_dir=None):
    """Rows from S3 when a bucket is configured (USAGE_LOG_S3_BUCKET), else local files."""
    if usage_db.enabled() and not (bucket or local_dir):
        return usage_db.fetch(None if days is None else days)
    bucket = bucket or os.environ.get("USAGE_LOG_S3_BUCKET")
    prefix = prefix or os.environ.get("USAGE_LOG_S3_PREFIX", "usage-logs")
    if bucket:
        return load_s3(bucket, prefix, days)
    return load_local(local_dir)


# ---------------------------------------------------------------------------
# Internal (team) visitors: browsers / networks whose traffic is ours, not users'.
# Stored next to the logs as _internal_visitors.json:
#   {"vids": {"<visitor id>": {"label": "...", "added": "..."}},
#    "ips":  {"<ip hash>":    {"label": "...", "added": "..."}}}
# Matching is applied when the logs are analysed, so marking a browser also
# excludes everything it did before it was marked.
# ---------------------------------------------------------------------------
_INTERNAL_NAME = "_internal_visitors.json"
_internal_cache = {"t": 0.0, "data": None}


def _internal_location():
    bucket = os.environ.get("USAGE_LOG_S3_BUCKET")
    prefix = os.environ.get("USAGE_LOG_S3_PREFIX", "usage-logs").strip("/")
    if bucket:
        return bucket, f"{prefix}/{_INTERNAL_NAME}"
    folder = os.environ.get("USAGE_LOG_DIR", os.path.join(_ROOT, "usage_logs"))
    return None, os.path.join(folder, _INTERNAL_NAME)


def load_internal(max_age_s=60):
    import time
    if _internal_cache["data"] is not None and time.time() - _internal_cache["t"] < max_age_s:
        return _internal_cache["data"]
    bucket, key = _internal_location()
    data = {"vids": {}, "ips": {}}
    try:
        if usage_db.enabled():
            data.update(usage_db.kv_get("internal_visitors") or {})
        elif bucket:
            import boto3
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            data.update(json.loads(body))
        elif os.path.exists(key):
            with open(key) as f:
                data.update(json.load(f))
    except Exception:
        # missing file = nobody marked yet; other errors: keep last good copy
        if _internal_cache["data"] is not None:
            return _internal_cache["data"]
    _internal_cache.update(t=time.time(), data=data)
    return data


def save_internal(data):
    import time
    bucket, key = _internal_location()
    blob = json.dumps(data, indent=1, sort_keys=True).encode()
    if usage_db.enabled():
        usage_db.kv_set("internal_visitors", data)
    elif bucket:
        import boto3
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=blob,
                                      ContentType="application/json")
    else:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        with open(key, "wb") as f:
            f.write(blob)
    _internal_cache.update(t=time.time(), data=data)


def mark_internal(vid=None, ip_hash=None, label=""):
    data = load_internal(max_age_s=0)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if vid:
        data.setdefault("vids", {})[vid] = {"label": label, "added": stamp}
    if ip_hash:
        data.setdefault("ips", {})[ip_hash] = {"label": label, "added": stamp}
    save_internal(data)
    return data


def to_frame(rows, internal=None):
    """Rows -> DataFrame with local time, a visitor key and session numbers."""
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df:
            df[c] = None
    if df.empty:
        df["ts_local"] = pd.Series(dtype="datetime64[ns, America/Los_Angeles]")
        df["who"] = pd.Series(dtype=object)
        df["session"] = pd.Series(dtype=object)
        df["internal"] = pd.Series(dtype=bool)
        df["lbl_net"] = pd.Series(dtype=bool)
        df["is_upload"] = pd.Series(dtype=bool)
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # "x" holds dicts (unhashable): compare duplicates on its JSON text
    _key = df.assign(x=df["x"].map(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v))
    df = df[~_key.duplicated()]
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df["ts_local"] = df["ts"].dt.tz_convert(TZ)
    # a visitor = cookie id, or hashed IP when cookies were refused
    df["who"] = df["vid"].fillna(df["ip"]).fillna("unknown")
    # a session = same visitor with no gap longer than 30 min
    gap = df.groupby("who")["ts"].diff() > pd.Timedelta(minutes=30)
    df["session"] = df["who"] + "#" + gap.groupby(df["who"]).cumsum().astype(str)
    df["status"] = pd.to_numeric(df["status"], errors="coerce")
    df["ms"] = pd.to_numeric(df["ms"], errors="coerce")
    df["kb"] = pd.to_numeric(df["kb"], errors="coerce")
    df["is_upload"] = df["trigger"].fillna("").str.endswith(".contents")
    # error = server crash (5xx) OR an error a callback caught and showed to the user
    df["is_error"] = (df["status"] >= 500) | df["err"].notna()
    xs = df["x"].map(lambda v: v if isinstance(v, dict) else {})
    for k in X_KEYS:
        df["x_" + k] = xs.map(lambda d, k=k: d.get(k))
    internal = internal if internal is not None else {"vids": {}, "ips": {}}
    df["internal"] = (df["vid"].isin(list(internal.get("vids", {})))
                      | df["ip"].isin(list(internal.get("ips", {}))))
    df["lbl_net"] = df["net"].isin(["lbl", "local"])
    return df


def per_tool_summary(df):
    if df.empty:
        return pd.DataFrame(columns=["tool", "visitors", "returning_visitors", "sessions",
                                     "page_views", "actions", "uploads", "errors",
                                     "median_action_ms"])
    days = df.assign(day=df["ts_local"].dt.date).groupby(["tool", "who"])["day"].nunique()
    returning = (days >= 2).groupby("tool").sum()
    g = df.groupby("tool")
    out = pd.DataFrame({
        "visitors": g["who"].nunique(),
        "returning_visitors": returning,
        "sessions": g["session"].nunique(),
        "page_views": g["event"].apply(lambda s: int((s == "page_view").sum())),
        "actions": g["event"].apply(lambda s: int((s == "action").sum())),
        "uploads": g["is_upload"].sum().astype(int),
        "errors": g["is_error"].sum().astype(int),
        "median_action_ms": df[df["event"] == "action"].groupby("tool")["ms"].median(),
    }).fillna(0)
    return (out.astype({"returning_visitors": int}).sort_values("visitors", ascending=False)
               .reset_index())


# ---------------------------------------------------------------------------
# PV-Copilot analysis runs (one row per analysis, plus its timeline)
#
# A run starts when an analysis reads its data: Simple mode's "load" stage
# (each Analyze click) or Advanced mode's "prescreen". The example pick /
# file upload just before it, and the filter / degradation / PVPRO / code /
# AI events of the same mode after it, belong to that run. A data step that
# failed with an error (e.g. an example that could not be read) is a run too.
# ---------------------------------------------------------------------------
_STAGE_ORDER = {"load": 1, "prescreen": 1, "filter": 2, "degradation": 3, "pvpro_done": 3}
_AI_STAGES = {"ai_code", "ai_verify", "ai_diag", "ai_chat"}
_STAGE_NAME = {0: "data step failed", 1: "data loaded", 2: "filtered", 3: "degradation"}
_LANES = ["UI", "Data", "Filter", "Degradation", "AI coding", "AI diagnostic", "AI help chat"]


def _uniq(series):
    vals = [str(v) for v in series.dropna().unique() if str(v) not in ("", "None", "nan")]
    return ", ".join(sorted(set(vals)))


def _trig(r):
    t = r.get("trigger")
    return "" if t is None or (isinstance(t, float) and pd.isna(t)) else str(t)


def _is_data_click(t):
    return "example" in t and t.endswith("n_clicks") or t.endswith(".contents")


def _ai_kind(t):
    """Which AI feature a control belongs to: coding / diagnostic / chat."""
    if "ai-diagnostic" in t:
        return "AI diagnostic"
    if "chat" in t or "composer" in t:
        return "AI help chat"
    if "code" in t or "download" in t:
        return "AI coding"
    return None


def _is_result_click(t):
    return _ai_kind(t) is not None


def assign_runs(df):
    """Series: run id ("<session>|<n>") for each PV-Copilot event, None for
    events outside any run (page views, UI clicks)."""
    run = pd.Series(None, index=df.index, dtype=object)
    c = df[df["tool"] == "pv-copilot"].sort_values("ts")
    for sid, g in c.groupby("session", sort=False):
        cur = {"simple": None, "advanced": None}
        last, n, pending = None, 0, []
        for i, r in g.iterrows():
            st, mode, t = r["x_stage"], r["x_mode"], _trig(r)
            st = None if pd.isna(st) else st
            failed_data = (st is None and pd.notna(r["err"])
                           and (_is_data_click(t) or t.startswith("analyze-btn")
                                or t.startswith("simple-run-trigger")))
            if st in ("load", "prescreen") or failed_data:
                n += 1
                rid = f"{sid}|{n}"
                m = mode if isinstance(mode, str) else ("simple" if t.startswith("simple") else "advanced")
                cur[m] = last = rid
                for j in pending:
                    run[j] = rid
                pending = []
                run[i] = rid
            elif st == "pick" or (st is None and _is_data_click(t)):
                pending.append(i)
            elif st in ("filter", "degradation", "pvpro_done"):
                # Steps follow their OWN mode's current run, so going back and
                # forth between Simple and Advanced (or between Advanced steps)
                # never mixes runs. An Advanced step with no prescreen in this
                # session (e.g. page reload restored the data) starts its own run.
                m = mode if isinstance(mode, str) else None
                if m and cur.get(m) is None:
                    n += 1
                    cur[m] = last = f"{sid}|{n}"
                run[i] = cur.get(m) if m else last
            elif st in _AI_STAGES:
                # AI coding / diagnostic belong to their mode's current run;
                # the help chat is shared, so it goes to the latest run.
                m = mode if isinstance(mode, str) else None
                run[i] = (cur.get(m) if m else None) or last
            elif _is_result_click(t):
                if _ai_kind(t) == "AI help chat":      # shared drawer -> latest run
                    run[i] = last
                else:
                    run[i] = cur["simple" if t.startswith("simple") else "advanced"] or last
            elif pd.notna(r["err"]):
                run[i] = last
    return run


def _count(st, trig, stage, kind):
    """How many times an AI feature was used in a run ('' when never).
    Counts the logged AI work; falls back to clicks for older log lines."""
    n = int((st == stage).sum())
    if not n:
        n = int((trig.map(_ai_kind) == kind).sum())
    return n or ""


def copilot_runs(df):
    """One row per PV-Copilot analysis run, newest first."""
    cols = ["run", "start (PT)", "visitor", "#", "mode", "data", "upload_kb", "rows",
            "span_yr", "method", "kept_%", "reached", "run_s", "ai_code", "ai_diag", "ai_chat",
            "errors", "load_s", "filter_s", "degradation_s", "pvpro_s",
            "ai_code_s", "ai_diag_s", "ai_chat_s", "wall_s", "done"]
    if df.empty or not (df["tool"] == "pv-copilot").any():
        return pd.DataFrame(columns=cols)
    d = df.assign(run=assign_runs(df)).dropna(subset=["run"])
    out = []
    for rid, g in d.groupby("run", sort=False):
        g = g.sort_values("ts")
        st = g["x_stage"]
        main = g[st.isin(["load", "prescreen"])]
        first = main.iloc[0] if not main.empty else g.iloc[0]
        trig = g["trigger"].fillna("")
        up_kb = g.loc[g["is_upload"], "kb"].sum()
        src = first.get("x_source")
        ds = first.get("x_dataset")
        data = (ds if isinstance(ds, str) else
                "upload" if (src == "upload" or up_kb > 0) else
                _uniq(g["x_dataset"]) or
                ("restored session (no data step logged)" if main.empty else "—"))
        filt = g[st == "filter"]
        kept = None
        if not filt.empty:
            fr = filt.iloc[-1]
            if pd.notna(fr["x_rows"]) and fr["x_rows"]:
                kept = round(100 * float(fr["x_rows_kept"]) / float(fr["x_rows"]), 1)
        reached = int(st.map(_STAGE_ORDER).max()) if st.map(_STAGE_ORDER).notna().any() else 0
        ok = ((st.isin(["degradation", "pvpro_done"])) & (g["x_status"] == "done")).any()
        failed = ((st == "pvpro_done") & (g["x_status"] == "failed")).any()
        steps = g[st.notna()]
        def _ai_secs(*stages):
            return round(float(pd.to_numeric(g.loc[st.isin(stages), "x_run_s"],
                                             errors="coerce").fillna(0).sum()), 1)
        ai_code_s, ai_diag_s, ai_chat_s = (_ai_secs("ai_code", "ai_verify"),
                                           _ai_secs("ai_diag"), _ai_secs("ai_chat"))
        if not (st == "ai_diag").any():
            # older log lines: the diagnostic ran inside the click request, so
            # the click's own duration IS the AI time
            ai_diag_s = round(float(g.loc[trig.str.contains("ai-diagnostic-btn") & st.isna(),
                                          "ms"].fillna(0).sum()) / 1000, 1)
        # compute = analysis steps (server time) + background PVPRO + AI work
        run_s = (steps.loc[~steps["x_stage"].isin(_AI_STAGES), "ms"].fillna(0).sum() / 1000
                 + ai_code_s + ai_diag_s + ai_chat_s
                 + pd.to_numeric(g.loc[st == "pvpro_done", "x_run_s"], errors="coerce").fillna(0).sum())
        def _secs(*stages):
            return round(float(g.loc[st.isin(stages), "ms"].fillna(0).sum()) / 1000, 2)
        pvpro_s = pd.to_numeric(g.loc[st == "pvpro_done", "x_run_s"], errors="coerce").fillna(0).sum()
        # wall clock: data step start -> result on screen. ts is written when a
        # request finishes, so the start is that event's ts minus its duration.
        wall_s = None
        fin = g[(st.isin(["degradation", "pvpro_done"])) & (g["x_status"] == "done")]
        if not main.empty and not fin.empty:
            t_start = main["ts"].iloc[0] - pd.Timedelta(milliseconds=float(main["ms"].fillna(0).iloc[0]))
            wall_s = round((fin["ts"].iloc[0] - t_start).total_seconds(), 1)
        span = pd.to_numeric(first.get("x_span_days"), errors="coerce")
        rows = pd.to_numeric(first.get("x_rows"), errors="coerce")
        out.append({
            "run": rid,
            "start (PT)": g["ts_local"].iloc[0].strftime("%Y-%m-%d %H:%M"),
            "visitor": str(g["who"].iloc[0])[:8],
            "#": int(rid.rsplit("|", 1)[1]),
            "mode": first.get("x_mode") if isinstance(first.get("x_mode"), str) else "—",
            "data": data,
            "upload_kb": int(up_kb) if up_kb else None,
            "rows": int(rows) if pd.notna(rows) else None,
            "span_yr": round(float(span) / 365.25, 1) if pd.notna(span) else None,
            "method": _uniq(g.loc[st.isin(["degradation", "pvpro_done"]), "x_method"]) or "—",
            "kept_%": kept,
            "reached": _STAGE_NAME[reached] + (" ✓" if ok else " ✗" if failed else ""),
            "run_s": round(float(run_s), 1),
            "ai_code": _count(st, trig, "ai_code", "AI coding"),
            "ai_diag": _count(st, trig, "ai_diag", "AI diagnostic"),
            "ai_chat": _count(st, trig, "ai_chat", "AI help chat"),
            "errors": int(g["is_error"].sum()),
            "load_s": _secs("load", "prescreen"),
            "filter_s": _secs("filter"),
            "degradation_s": _secs("degradation"),
            "pvpro_s": round(float(pvpro_s), 1),
            "ai_code_s": ai_code_s, "ai_diag_s": ai_diag_s, "ai_chat_s": ai_chat_s,
            "wall_s": wall_s,
            "done": bool(ok),
        })
    return pd.DataFrame(out, columns=cols).sort_values(["start (PT)", "#"], ascending=False)


_FRIENDLY = {
    "simple-analyze-btn": "Click Analyze (Simple)",
    "analyze-btn": "Run prescreening (Advanced)",
    "filter-btn": "Run filtering (Advanced)",
    "run-btn": "Calculate degradation (Advanced)",
    "simple-code-btn": "Click Generate code (Simple)",
    "simple-code-verify-btn": "Click Verify code (Simple)",
    "simple-ai-diagnostic-btn": "Click AI diagnostic (Simple)",
    "advanced-ai-diagnostic-btn": "Click AI diagnostic (Advanced)",
    "generate-code-btn": "Click Generate code (Advanced)",
    "code-verify-btn": "Click Verify code (Advanced)",
    "chat-send": "Send help-chat question",
    "chat-composer": "Send help-chat question (Enter)",
    "chat-example": "Pick example question",
    "chat-drawer-open": "Open help chat",
    "simple-method-radio": "Change method (Simple)",
    "example-tab-list": "Browse examples (list)",
}


def _lane_and_label(r):
    t = _trig(r)
    comp = t.split(".")[0]
    st = r.get("x_stage")
    x = {k: r.get("x_" + k) for k in X_KEYS}

    def _n(v):
        return f"{int(v):,}" if v is not None and pd.notna(v) else "?"

    if st in ("load", "prescreen", "pick"):
        what = x["dataset"] or x["source"] or "data"
        head = {"load": "Load (Simple)", "prescreen": "Prescreen (Advanced)",
                "pick": "Pick example"}[st]
        lab = f"{head}: {what} · {_n(x['rows'])} rows"
        if x["span_days"] is not None and pd.notna(x["span_days"]):
            lab += f" · {float(x['span_days']) / 365.25:.1f} yr"
        if x["downsized"] is not None and pd.notna(x["downsized"]):
            lab += f" · downsized ×{x['downsized']:g}"
        return "Data", lab
    if st == "filter":
        lab = f"Filter: kept {_n(x['rows_kept'])} of {_n(x['rows'])}"
        if x["filters"]:
            lab += f" ({x['filters']})"
        return "Filter", lab
    if st == "degradation":
        return "Degradation", f"{x['method'] or '?'} {x['status'] or ''}".strip()
    if st in _AI_STAGES:
        rs = x["run_s"]
        secs = f" · {float(rs):.1f} s" if rs is not None and pd.notna(rs) else ""
        what = {"ai_code": "Generate code", "ai_verify": "Verify code (test run)",
                "ai_diag": "AI diagnostic", "ai_chat": "Help-chat reply"}[st]
        lane = {"ai_code": "AI coding", "ai_verify": "AI coding",
                "ai_diag": "AI diagnostic", "ai_chat": "AI help chat"}[st]
        status = x["status"] if x["status"] and x["status"] != "done" else ""
        return lane, f"{what} {status}".strip() + secs
    if st == "pvpro_done":
        rs = x["run_s"]
        return "Degradation", (f"PVPRO {x['status']}"
                               + (f" after {float(rs):.0f} s" if rs is not None and pd.notna(rs) else ""))
    if t.endswith(".contents"):
        kb = r.get("kb")
        return "Data", "Upload file" + (f" ({int(kb):,} KB)" if kb is not None and pd.notna(kb) else "")
    if "example" in t:
        return "Data", "Pick example: " + comp
    if _is_result_click(t):
        return _ai_kind(t), _FRIENDLY.get(comp, comp)
    if not t and r.get("event") in ("page_view", "visit"):
        return "UI", "Opened PV-Copilot"
    if x["mode"] is not None and pd.notna(x["mode"]):
        return "UI", f"Switch to {x['mode']} mode"
    return "UI", _FRIENDLY.get(comp, comp) if comp else str(r.get("event"))


def run_timeline(df, run_id):
    """Events of one analysis run with a lane + readable label, oldest first."""
    d = df.assign(run=assign_runs(df))
    g = d[d["run"] == run_id].sort_values("ts").copy()
    if g.empty:
        return g
    ll = g.apply(lambda r: _lane_and_label(r), axis=1, result_type="expand")
    g["lane"], g["label"] = ll[0], ll[1]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", help="bucket name (default: USAGE_LOG_S3_BUCKET or ./usage_logs)")
    ap.add_argument("--prefix", default="usage-logs")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--csv", help="also write all events to this CSV")
    ap.add_argument("--include-internal", action="store_true",
                    help="keep traffic from browsers marked internal (team)")
    a = ap.parse_args()

    if a.s3:
        os.environ.setdefault("USAGE_LOG_S3_BUCKET", a.s3)
    df = to_frame(load_events(a.days, a.s3, a.prefix, a.dir), load_internal())
    if not a.include_internal:
        print(f"(excluding {int(df['internal'].sum())} internal events)")
        df = df[~df["internal"]]
    if df.empty:
        print("No events found.")
        return
    df = df[df["ts"] >= df["ts"].max() - pd.Timedelta(days=a.days)]
    pd.set_option("display.width", 160)
    print(f"\n== {len(df)} events, {df['ts_local'].min():%Y-%m-%d} .. "
          f"{df['ts_local'].max():%Y-%m-%d} ({TZ}) ==\n")
    print(per_tool_summary(df).to_string(index=False), "\n")
    acts = df[df["event"] == "action"]
    if not acts.empty:
        top = acts.groupby(["tool", "trigger"]).size().rename("n").reset_index()
        top = top.sort_values(["tool", "n"], ascending=[True, False]).groupby("tool").head(8)
        print("Top actions per tool\n", top.to_string(index=False), "\n")
    if a.csv:
        df.drop(columns=["who", "session", "ts_local", "is_upload"]).to_csv(a.csv, index=False)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
