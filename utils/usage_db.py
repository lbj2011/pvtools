"""
utils/usage_db.py -- Postgres storage for the usage log (Heroku Postgres).

Used automatically when the DATABASE_URL environment variable is set (Heroku
sets it when the Heroku Postgres add-on is attached). Otherwise the usage log
falls back to S3 (USAGE_LOG_S3_BUCKET) or local files, as before.

Why: Heroku wipes the dyno's disk on every deploy/restart, so local log files
were lost each time. The database lives outside the dyno and survives.

Tables (created on first use):
    usage_events  one row per event: ts, h (content hash, makes imports
                  idempotent), rec (the event as JSONB, same fields as the
                  JSON-lines log -- see usage_log.py)
    usage_kv      small key/value documents, e.g. the internal-visitors list

Import old log files (JSON lines, .jsonl or .jsonl.gz) -- safe to repeat:
    python -m utils.usage_db import usage_backfill/*.jsonl.gz
    heroku run -a pvtools python -m utils.usage_db import usage_backfill/x.jsonl.gz
"""
import gzip
import hashlib
import json
import os
import sys
import threading

_init_lock = threading.Lock()
_initialised = set()           # DSNs whose tables are known to exist


def enabled():
    return bool(os.environ.get("DATABASE_URL"))


def _dsn():
    dsn = os.environ["DATABASE_URL"]
    if dsn.startswith("postgres://"):          # Heroku's scheme; libpq wants postgresql://
        dsn = "postgresql://" + dsn[len("postgres://"):]
    return dsn


def _connect():
    import psycopg2
    dsn = _dsn()
    # Heroku Postgres requires TLS; a local test database can set PGSSLMODE=disable.
    conn = psycopg2.connect(dsn, sslmode=os.environ.get("PGSSLMODE", "require"),
                            connect_timeout=10, application_name="pvtools-usage")
    if dsn not in _initialised:
        with _init_lock:
            if dsn not in _initialised:
                with conn, conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS usage_events (
                            id   BIGSERIAL PRIMARY KEY,
                            ts   TIMESTAMPTZ NOT NULL,
                            h    TEXT UNIQUE NOT NULL,
                            rec  JSONB NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS usage_events_ts ON usage_events (ts);
                        CREATE TABLE IF NOT EXISTS usage_kv (
                            k        TEXT PRIMARY KEY,
                            v        JSONB NOT NULL,
                            updated  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                    """)
                _initialised.add(dsn)
    return conn


def _hash(rec):
    return hashlib.sha1(json.dumps(rec, sort_keys=True, separators=(",", ":"))
                        .encode("utf-8")).hexdigest()


def insert(records):
    """Insert event dicts; exact duplicates are skipped. Returns rows added."""
    records = [r for r in records if isinstance(r, dict) and r.get("ts")]
    if not records:
        return 0
    from psycopg2.extras import Json, execute_values
    rows = [(r["ts"], _hash(r), Json(r)) for r in records]
    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            added = execute_values(cur, "INSERT INTO usage_events (ts, h, rec) VALUES %s "
                                        "ON CONFLICT (h) DO NOTHING RETURNING 1",
                                   rows, page_size=500, fetch=True)
            return len(added)
    finally:
        conn.close()


def fetch(days=30):
    """Event dicts from the last `days` days (all when days is None), oldest first."""
    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            if days is None:
                cur.execute("SELECT rec FROM usage_events ORDER BY ts")
            else:
                cur.execute("SELECT rec FROM usage_events "
                            "WHERE ts >= now() - %s * interval '1 day' ORDER BY ts",
                            (float(days),))
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def kv_get(key, default=None):
    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT v FROM usage_kv WHERE k = %s", (key,))
            row = cur.fetchone()
            return row[0] if row else default
    finally:
        conn.close()


def kv_set(key, value):
    from psycopg2.extras import Json
    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("INSERT INTO usage_kv (k, v, updated) VALUES (%s, %s, now()) "
                        "ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v, updated = now()",
                        (key, Json(value)))
    finally:
        conn.close()


def _read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    pass


def _main(argv):
    if len(argv) < 2 or argv[0] != "import":
        print(__doc__)
        return 2
    if not enabled():
        print("DATABASE_URL is not set.")
        return 1
    total = 0
    for path in argv[1:]:
        recs = list(_read_jsonl(path))
        added = insert(recs)
        total += added
        print(f"{path}: {len(recs)} events read, {added} new")
    print(f"done: {total} new events")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
