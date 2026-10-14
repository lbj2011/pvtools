"""
utils/usage_log.py -- privacy-preserving usage logging for pvtools.

Records WHO (anonymous) used WHICH tool and WHAT they clicked -- never the data.

One JSON line per event:
    ts       UTC timestamp (ISO 8601)
    vid      anonymous visitor id: the random `pvtools_client_id` cookie set in
             app.py, truncated. It is not linked to any name or email.
    ip       keyed hash of the client IP (HMAC-SHA256 with USAGE_LOG_SALT).
             The raw IP is never written.
    tool     tool page: field-degradation, pv-copilot, pv-image, ...
    event    visit      -> the browser loaded a tool URL directly (first hit)
             page_view  -> the router rendered a tool page
             action     -> a user control fired a callback
    trigger  for actions: the component.property that fired, e.g.
             "analyze-button.n_clicks" or "upload-image.contents"
    status   HTTP status of the request (>=500 = server error)
    err      exception CLASS NAME (e.g. "ImportError") when a callback caught an
             error and showed it to the user (HTTP status stays 200). Set by
             note_error(); the exception message is never recorded.
    ms       server time spent on the request
    kb       request size in KB, ONLY for upload triggers (*.contents): says an
             upload happened and how big it was, not what it contained
    ua       device class + browser family, e.g. "desktop/chrome"
    ref      external referrer domain (visit events only), e.g. "google.com"
    country  only if a CDN in front of the site sends CF-IPCountry
    net      "lbl" when the request came from an LBNL network (USAGE_LOG_LBL_NETS),
             "local" for localhost/private addresses (development); else omitted.
             Lets the dashboard separate lab/team traffic without storing IPs.

    x        small app-level facts a page attaches with note(), e.g. for
             PV-Copilot {"stage": "load", "mode": "simple", "method": "YOY",
             "source": "example", "dataset": "ucy_nicosia_monoSi_hourly",
             "rows": 87672}. Keys and values are whitelisted in note(); the
             name of an UPLOADED file is never recorded (source="upload").
             Events carrying x/err are kept even when no user control fired
             them (event="step"), e.g. chained pipeline stages or a PVPRO
             background fit finishing.

NEVER recorded: callback input/output values, uploaded files or images, file
names, chat questions or AI answers, raw IP address, full User-Agent string.

Storage
    Heroku's disk is wiped on every restart, so events are buffered in memory
    and flushed (every USAGE_LOG_FLUSH_S seconds, or every 500 events, and on
    shutdown) to, in order of preference:
      1. Heroku Postgres when DATABASE_URL is set (utils/usage_db.py);
         default flush every 60 s
      2. S3 (default flush every 300 s), as a gzipped JSON-lines object to
        s3://$USAGE_LOG_S3_BUCKET/$USAGE_LOG_S3_PREFIX/YYYY/MM/DD/<time>-<pid>-<rand>.jsonl.gz
    When USAGE_LOG_S3_BUCKET is not set (local development) or an upload
    fails, the lines go to ./usage_logs/usage-YYYY-MM-DD.jsonl instead.

Environment variables
    USAGE_LOG_S3_BUCKET   S3 bucket (must be PRIVATE)     default: unset -> local file
    USAGE_LOG_S3_PREFIX   key prefix                      default: usage-logs
    USAGE_LOG_SALT        secret for hashing IPs           default: Flask secret_key
    USAGE_LOG_FLUSH_S     flush interval in seconds        default: 300
    USAGE_LOG_STDOUT      "1" also prints USAGE lines to stdout (debugging)
    USAGE_LOG_DISABLED    "1" turns logging off entirely
    USAGE_LOG_LBL_NETS    comma-separated CIDRs tagged net="lbl"
                          default: 131.243.0.0/16,128.3.0.0/16 (LBNL)
"""
import atexit
import datetime as _dt
import gzip
import hashlib
import hmac
import ipaddress
import json
import os
import re
import threading
import time
import uuid
from urllib.parse import urlparse

from flask import g, request

from utils import usage_db

# Component properties that mean "a person did something". Anything else
# (dcc.Interval polling, dcc.Store chaining, initial page callbacks) is noise.
_USER_PROPS = {
    "n_clicks", "n_submit", "contents", "value", "active_tab", "active_item",
    "clickData", "selectedData", "selected_rows", "start_date", "end_date",
    "checked",
}
_BOT_RE = re.compile(r"bot|crawl|spider|slurp|curl|wget|python-requests|httpx|"
                     r"headless|scrapy|monitor|uptime", re.I)
_LBL_NETS = []
for _c in os.environ.get("USAGE_LOG_LBL_NETS", "131.243.0.0/16,128.3.0.0/16").split(","):
    try:
        _LBL_NETS.append(ipaddress.ip_network(_c.strip()))
    except ValueError:
        pass
_DEDUPE_S = 1.5          # same visitor + same trigger within this window -> one event
_MAX_BUFFER = 500
_MAX_RETRY_BUFFER = 20000   # events held in memory while the database is unreachable


# ---------------------------------------------------------------------------
# Sink: in-memory buffer -> S3 (or local file)
# ---------------------------------------------------------------------------
class _Sink:
    def __init__(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.bucket = os.environ.get("USAGE_LOG_S3_BUCKET") or None
        self.prefix = os.environ.get("USAGE_LOG_S3_PREFIX", "usage-logs").strip("/") + "/"
        self.local_dir = os.environ.get("USAGE_LOG_DIR", os.path.join(root, "usage_logs"))
        self.db = usage_db.enabled()
        self.flush_s = max(15, int(os.environ.get("USAGE_LOG_FLUSH_S", 60 if self.db else 300)))
        self.stdout = os.environ.get("USAGE_LOG_STDOUT") == "1"
        self._buf = []
        self._lock = threading.Lock()
        self._retry_after = 0.0      # no size-triggered flushes until then (DB outage)
        self._s3 = None
        threading.Thread(target=self._loop, name="usage-log-flush", daemon=True).start()
        atexit.register(self.flush, final=True)

    def add(self, rec):
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
        if self.stdout:
            print("USAGE " + line, flush=True)
        with self._lock:
            self._buf.append(line)
            full = len(self._buf) >= _MAX_BUFFER and time.time() >= self._retry_after
        if full:
            threading.Thread(target=self.flush, daemon=True).start()

    def _loop(self):
        while True:
            time.sleep(self.flush_s)
            self.flush()

    def flush(self, final=False):
        with self._lock:
            lines, self._buf = self._buf, []
        if not lines:
            return
        data = ("\n".join(lines) + "\n").encode("utf-8")
        now = _dt.datetime.now(_dt.timezone.utc)
        if self.db:
            try:
                usage_db.insert([json.loads(l) for l in lines])
                return
            except Exception as e:  # never break the site over logging
                # Database briefly unreachable: keep the events in memory and
                # retry on the next flush (a local file would vanish on the
                # next Heroku restart). Bounded so an outage can't eat RAM;
                # at shutdown (final=True) fall through to S3 / local file.
                with self._lock:
                    if not final and len(self._buf) + len(lines) <= _MAX_RETRY_BUFFER:
                        self._buf[:0] = lines
                        self._retry_after = time.time() + 60
                        print(f"usage_log: Postgres insert failed ({type(e).__name__}); "
                              f"will retry {len(lines)} events", flush=True)
                        return
                print(f"usage_log: Postgres insert failed ({type(e).__name__}: {e}); "
                      f"falling back for {len(lines)} events", flush=True)
        if self.bucket:
            try:
                if self._s3 is None:
                    import boto3
                    self._s3 = boto3.client("s3")
                key = (f"{self.prefix}{now:%Y/%m/%d}/"
                       f"{now:%H%M%S}-{os.getpid()}-{uuid.uuid4().hex[:6]}.jsonl.gz")
                self._s3.put_object(Bucket=self.bucket, Key=key,
                                    Body=gzip.compress(data),
                                    ContentType="application/gzip")
                return
            except Exception as e:  # never break the site over logging
                print(f"usage_log: S3 upload failed ({type(e).__name__}: {e}); "
                      f"writing {len(lines)} events locally", flush=True)
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            path = os.path.join(self.local_dir, f"usage-{now:%Y-%m-%d}.jsonl")
            with open(path, "ab") as f:
                f.write(data)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers (each one deliberately throws information away)
# ---------------------------------------------------------------------------
def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",", 1)[0].strip() or (request.remote_addr or "")


def _hash_ip(ip, salt):
    if not ip:
        return None
    return hmac.new(salt, ip.encode("utf-8"), hashlib.sha256).hexdigest()[:12]


def _net_tag(ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if any(addr in n for n in _LBL_NETS):
        return "lbl"
    if addr.is_loopback or addr.is_private:
        return "local"
    return None


def _ua_class(ua):
    device = "mobile" if re.search(r"Mobi|Android|iPhone|iPad", ua) else "desktop"
    if "Edg/" in ua:
        browser = "edge"
    elif "Firefox/" in ua:
        browser = "firefox"
    elif "Chrome/" in ua or "CriOS/" in ua:
        browser = "chrome"
    elif "Safari/" in ua:
        browser = "safari"
    else:
        browser = "other"
    return f"{device}/{browser}"


def _external_ref_domain(referrer):
    """Keep only the referring site's domain, and only if it is not pvtools itself."""
    if not referrer:
        return None
    host = (urlparse(referrer).hostname or "").lower()
    if not host or host == (request.host.split(":")[0].lower()):
        return None
    return host[4:] if host.startswith("www.") else host


def _trigger_name(prop_id):
    """'analyze-button.n_clicks' -> same; pattern-matching ids keep only their 'type'."""
    comp, _, prop = prop_id.rpartition(".")
    if comp.startswith("{"):
        try:
            d = json.loads(comp)
            comp = str(d.get("type") or "-".join(sorted(map(str, d.keys()))))
        except Exception:
            comp = "pattern"
    return f"{comp[:60]}.{prop}"


def _pick_trigger(changed):
    for pid in changed or []:
        if pid.rpartition(".")[2] in _USER_PROPS:
            return _trigger_name(pid)
    return None


def _dash_input(payload, cid, prop):
    for item in payload.get("inputs", []) or []:
        if isinstance(item, dict) and item.get("id") == cid and item.get("property") == prop:
            return item.get("value")
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_SINK = None  # the process-wide sink, set by init_usage_logging
_SALT = b"pvtools"


# note(): whitelisted keys -> allowed value kind. Strings are cut to 60 chars.
_NOTE_KEYS = {
    "stage": str,      # pick / load / prescreen / filter / degradation / pvpro_done /
                       # ai_code / ai_verify / ai_diag / ai_chat
    "mode": str,       # simple / advanced
    "method": str,     # YOY, PVPRO, "YOY,HW" ...
    "source": str,     # example / upload
    "dataset": str,    # example dataset id ONLY (never an uploaded file name)
    "rows": int, "rows_kept": int, "cols": int,
    "span_days": int, "downsized": float,
    "filters": str,    # filter ids, e.g. "clearsky,outlier"
    "run_s": float,    # compute time measured by the page (e.g. PVPRO thread)
    "status": str,     # started / done / failed / off_topic
}


def note(**fields):
    """Attach small, whitelisted facts to the current request's usage event
    (see the "x" field). No-op outside a Flask request; never raises."""
    try:
        from flask import has_request_context
        if not has_request_context():
            return
        x = getattr(g, "_usage_x", None) or {}
        for k, v in fields.items():
            kind = _NOTE_KEYS.get(k)
            if kind is None or v is None:
                continue
            if kind is str:
                x[k] = str(v)[:60]
            elif kind is int:
                x[k] = int(v)
            else:
                x[k] = round(float(v), 2)
        g._usage_x = x
    except Exception:
        pass


def note_error(exc):
    """Mark the current request as a handled error (shown to the user as an
    alert, so the HTTP status is still 200). Only the exception class name is
    logged. Safe to call anywhere: a no-op outside a Flask request (e.g. in a
    background thread) and it never raises."""
    try:
        from flask import has_request_context
        if has_request_context():
            g._usage_err = type(exc).__name__ if isinstance(exc, BaseException) else str(exc)[:40]
    except Exception:
        pass


def current_visitor():
    """(visitor id, ip hash) of the browser making the current request, in the
    same form as the logged 'vid' and 'ip' fields. Call inside a request."""
    vid = (request.cookies.get("pvtools_client_id") or "")[:16] or None
    return vid, _hash_ip(_client_ip(), _SALT)


def pending_events():
    """Events recorded by this process but not yet flushed to storage
    (used by the admin dashboard so it shows the last few minutes too)."""
    sink = _SINK
    if sink is None:
        return []
    with sink._lock:
        lines = list(sink._buf)
    return [json.loads(line) for line in lines]


def storage_location():
    sink = _SINK
    if sink is None:
        return "usage logging not enabled"
    if sink.db:
        return "Heroku Postgres (DATABASE_URL)"
    return f"s3://{sink.bucket}/{sink.prefix}" if sink.bucket else sink.local_dir


def init_usage_logging(server, routes):
    """Attach usage logging to the Flask server.

    routes: the {pathname: page_module} dict from index.py; the tool name logged
    is the pathname without slashes ("/" -> "home").
    """
    if os.environ.get("USAGE_LOG_DISABLED") == "1":
        print("usage_log: disabled (USAGE_LOG_DISABLED=1)")
        return None

    global _SINK, _SALT
    sink = _SINK = _Sink()
    salt = _SALT = (os.environ.get("USAGE_LOG_SALT") or server.secret_key or "pvtools").encode()
    tools = {p: (p.strip("/") or "home") for p in routes}
    tools["/home"] = "home"
    recent = {}               # (vid, trigger) -> last time, for de-duplication
    recent_lock = threading.Lock()

    def _build(resp):
        ua = request.headers.get("User-Agent", "")
        if not ua or _BOT_RE.search(ua):
            return None
        path = request.path
        ref = kb = trigger = None

        if request.method == "GET" and path in tools:
            event, tool = "visit", tools[path]
            ref = _external_ref_domain(request.referrer)

        elif request.method == "POST" and path == "/_dash-update-component":
            payload = request.get_json(silent=True) or {}
            if "page-content.children" in (payload.get("output") or ""):
                tool = tools.get(_dash_input(payload, "url", "pathname"))
                event = "page_view"
            else:
                trigger = _pick_trigger(payload.get("changedPropIds"))
                tool = tools.get(urlparse(request.referrer or "").path)
                event = "action"
                if trigger is None:
                    # Not fired by a person (store chain, interval poll). Keep
                    # it only when the page attached facts or an error to it.
                    if not (getattr(g, "_usage_x", None) or getattr(g, "_usage_err", None)):
                        return None
                    changed = payload.get("changedPropIds") or []
                    trigger = _trigger_name(changed[0]) if changed else "auto"
                    event = "step"
                if trigger.endswith(".contents") and request.content_length:
                    kb = int(request.content_length // 1024)
            if not tool:
                return None
        else:
            return None

        vid = (request.cookies.get("pvtools_client_id") or "")[:16] or None
        # One click can fire several callbacks (several requests with the same
        # trigger); they are de-duplicated -- except one that hit an error,
        # which must never be dropped.
        if (event == "action" and not getattr(g, "_usage_err", None)
                and not getattr(g, "_usage_x", None)):
            now = time.monotonic()
            key = (vid or _client_ip(), trigger)
            with recent_lock:
                if now - recent.get(key, -1e9) < _DEDUPE_S:
                    return None
                recent[key] = now
                if len(recent) > 5000:
                    recent.clear()

        t0 = getattr(g, "_usage_t0", None)
        ip = _client_ip()
        rec = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "vid": vid,
            "ip": _hash_ip(ip, salt),
            "tool": tool,
            "event": event,
            "trigger": trigger,
            "status": resp.status_code,
            "ms": int((time.perf_counter() - t0) * 1000) if t0 else None,
            "kb": kb,
            "ua": _ua_class(ua),
            "ref": ref,
            "country": request.headers.get("CF-IPCountry"),
            "net": _net_tag(ip),
            "err": getattr(g, "_usage_err", None),
            "x": getattr(g, "_usage_x", None) or None,
        }
        return {k: v for k, v in rec.items() if v is not None}

    @server.before_request
    def _usage_t0():
        g._usage_t0 = time.perf_counter()

    @server.after_request
    def _usage_record(resp):
        try:
            rec = _build(resp)
            if rec:
                sink.add(rec)
        except Exception:
            pass  # logging must never affect the response
        return resp

    where = (f"s3://{sink.bucket}/{sink.prefix}" if sink.bucket else sink.local_dir)
    print(f"usage_log: enabled -> {storage_location()}")
    return sink
