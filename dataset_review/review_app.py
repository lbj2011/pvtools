#!/usr/bin/env python3
"""
Review app for the validated PV degradation dataset (final values only).

    login -> filter papers -> pick a paper -> [key fields + collapsed categories | PDF]
    hover a field for its detail (raw text, PDF quote + page, reason, confidence)
    mark each data point (good / bad / checked / unsure) and leave comments,
    on a whole paper, a data point, or one field.

Run:   python review_app/app.py [--port 8060]   then open the printed address (next free port if busy)
Reads from the parent folder (paper_check/): data_2609_merged_all.pkl (validated_2605 + new_extraction_2609),
data_dictionary.csv, source_pdfs/
Review marks and comments go to review_app/review_log.sqlite (multi-user safe, WAL).

Embedded mode (e.g. inside pvtools): before importing this module, the host puts a config object in
sys.modules["pv_review_host"] with: server (Flask), base ("/dataset-review/"), data, dict_csv, log (paths),
public_mode, password, s3_bucket, s3_log_key. The app then mounts on the host server under `base`,
serves no PDFs, and keeps the review log in S3 (downloaded at start-up, uploaded after every write).
"""
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sqlite3
import sys
import threading
from urllib.parse import quote

import numpy as np
import pandas as pd
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from flask import abort, send_from_directory

# ---------------------------------------------------------------- config
HERE = pathlib.Path(__file__).resolve().parent          # paper_check/review_app/
ROOT = HERE.parent                                       # paper_check/  (data + PDFs live here)
# newest first: rows with a rate only (+ reversible_rate_flag) > faults / stressors / reversible losses re-extracted (extract_faults.py) > per-point
# re-check (recheck_points.py) > added rows repaired (fix_added_rows.py) > merge
DATA = next((ROOT / n for n in ("data_2609_merged_all_faults_with_rate.pkl", "data_2609_merged_all_faults.pkl", "data_2609_merged_all_rechecked.pkl",
                                 "data_2609_merged_all_fixed.pkl",
                                 "data_2609_merged_all.pkl") if (ROOT / n).exists()), ROOT / "data_2609_merged_all.pkl")
DICT_CSV = ROOT / "data_dictionary.csv"
PDF_DIR = ROOT / "source_pdfs"
LOG = HERE / "review_log.sqlite"
PUBLIC_MODE = False            # True -> PDFs are not served; reviewers pick a local folder or open the DOI
LOGIN_PASSWORD = "duramat"
USERS = ["Baojie", "Martin", "Dirk", "Anubhav", "Dax"]
HOST, PORT = "127.0.0.1", 8060          # override: python app.py --port 8070  (or env PORT=...)
EMBED = sys.modules.get("pv_review_host")  # set by a host app (pvtools) -> embedded mode
BASE = "/"
S3_BUCKET = S3_LOG_KEY = None
if EMBED is not None:
    BASE = EMBED.base
    DATA, DICT_CSV, LOG = pathlib.Path(EMBED.data), pathlib.Path(EMBED.dict_csv), pathlib.Path(EMBED.log)
    PDF_DIR = pathlib.Path("/nonexistent")
    PUBLIC_MODE = EMBED.public_mode
    LOGIN_PASSWORD = EMBED.password or LOGIN_PASSWORD
    S3_BUCKET, S3_LOG_KEY = EMBED.s3_bucket, EMBED.s3_log_key
STATUSES = ["good", "bad", "checked", "unsure"]
NA = "__NA__"

# ---------------------------------------------------------------- field layout
# main fields, always shown; each may pull a verbatim "raw text" field into its hover card
MAIN = [
    ("annual_power_deg_rate_in_percent", "Degradation rate"),
    ("pv_tech", "PV technology"),
    ("duration_in_years", "Exposure"),
    ("country", "Location"),
    ("koppen_zone", "Climate zone"),
    ("system_capacity_watts", "System capacity"),
    ("number_of_pv_modules", "Modules"),
    ("mounting_detail", "Mounting"),
    ("scope_of_study", "Scope"),
    ("analysis_method", "Analysis method"),
    ("faults", "Faults (observed)"),
    ("faults_major", "Causal faults"),
    ("stressors_causal", "Stressors"),
    ("reversible_losses", "Reversible losses · observed"),
    ("reversible_affecting_rate", "Reversible losses · affecting rate"),
    ("quality_score", "Quality score"),
]
RAW_OF = {
    "annual_power_deg_rate_in_percent": "annual_power_deg_rate_raw_text",
    "total_power_loss_loss_in_percent": "total_power_loss_loss_raw_text",
    "pv_tech": "pv_tech_raw_text",
    "duration_in_years": "duration_raw_text",
    "country": "location_raw_text",
    "system_capacity_watts": "system_capacity_raw_text",
    "pv_module_nominal_power_watts": "pv_module_nominal_power_raw_text",
    "mounting_detail": "mounting_raw_text",
    "faults": "faults_raw_text",
    "faults_major": "faults_major_raw_text",
    "reversible_losses": "reversible_losses_raw_text",
    "reversible_affecting_rate": "reversible_rate_explanation",
    "stressors_causal": "stressors_raw_text",
    "location_provenance": "location_provenance_raw_text",
    "climate_stressors": "climate_stressors_raw_text",
    "extreme_weather_events": "extreme_weather_raw_text",
    "measurement_correction": "correction_raw_text",
    "uncertainty_reported": "uncertainty_raw_text",
    "goodness_of_fit_reported": "goodness_of_fit_raw_text",
    "preprocessing_steps": "preprocessing_raw_text",
    "voltage_decrease_reported": "vi_decrease_raw_text",
    "current_decrease_reported": "vi_decrease_raw_text",
    "measurement_count": "measurement_span_raw_text",
}
RAW_FIELDS = set(RAW_OF.values())
RAW_HDR = {"reversible_affecting_rate": "Why - per observed loss", "reversible_losses": "Raw text - per loss"}
# related fields shown inside a key field's hover card (after the raw text)
EXTRA_OF = {
    "annual_power_deg_rate_in_percent": [
        ("degradation_metric", "Metric"), ("rate_provenance", "Rate source"), ("analysis_method", "Analysis method"),
        ("rate_is_range", "Given as a range"), ("rate_range_low", "Range low"), ("rate_range_high", "Range high"),
        ("rate_comparability_flag", "Comparability caveats"), ("annual_power_deg_confidence_level", "Extractor confidence"),
        ("annual_power_deg_explanation", "Explanation")],
    "pv_tech": [("pv_tech_detail", "Tech detail"), ("pv_tech_mix", "Technology mix"), ("pv_module_name", "Module"),
                ("bifacial", "Bifacial")],
    "duration_in_years": [("start_year", "Start year"), ("end_year", "End year"), ("measurement_count", "# measurements"),
                          ("sample_size_for_rate", "Sample size for the rate")],
    "country": [("location_latitude", "Latitude"), ("location_longitude", "Longitude"),
                ("location_provenance", "How the location is known"), ("location_site_count", "# sites")],
    "koppen_zone": [("PV zone", "PV zone"), ("pvcz_label", "PVCZ"), ("climate_stressors", "Climate stressors"),
                    ("extreme_weather_events", "Extreme weather")],
    "system_capacity_watts": [("capacity_basis", "Capacity basis"), ("capacity_other_basis_watts", "Other rating"),
                              ("grid_connected", "Grid connected")],
    "number_of_pv_modules": [("pv_module_nominal_power_watts", "Module power"), ("pv_module_name", "Module"),
                             ("pv_module_nominal_power_raw_text", "Module power, raw text")],
    "mounting_detail": [("mounting", "Mounting class"), ("tilt_degrees", "Tilt")],
    "scope_of_study": [("nature_of_study", "Nature of study"), ("sample_size_for_rate", "Sample size for the rate")],
    "analysis_method": [("measurement_approach", "Measurement approach"), ("measurement_correction", "Correction"),
                        ("goodness_of_fit_reported", "Goodness of fit")],
    "faults": [("faults_suspected", "Suspected (hedged)"), ("faults_checked_absent", "Looked for, not found")],
    "faults_major": [("faults_dominant", "Dominant"), ("fault_causality_stated", "Causality stated")],
    "reversible_affecting_rate": [("reversible_rate_flag", "Flag")],
    "stressors_causal": [("stressors_suspected", "Suspected (hedged)"), ("stressors_context", "Context (not linked)")],

    "quality_score": [("quality_design_n", "Design flags met"), ("quality_reporting_n", "Reporting flags met"),
                      ("quality_flags_met", "Flags met"), ("quality_flags_missing", "Flags missing")],
}
# the rest, collapsed by category (raw-text fields live in the hover cards, not here)
GROUPS = [
    ("Degradation rate", [
        "degradation_metric", "rate_provenance", "rate_stated_by_author", "rate_comparability_flag",
        "total_power_loss_loss_in_percent", "rate_is_range", "rate_range_low", "rate_range_high",
        "total_power_loss_type_of_power_loss", "total_power_loss_explanation", "annual_power_deg_location_of_info",
        "annual_power_deg_confidence_level", "annual_power_deg_explanation", "annual_degradation_rate_of_other_params",
        "voltage_decrease_reported", "current_decrease_reported", "vi_decrease_summary"]),
    ("Technology & system", [
        "pv_tech_detail", "pv_tech_mix", "pv_tech_mix_known", "pv_module_name", "bifacial", "materials_and_construction",
        "capacity_basis", "capacity_other_basis_watts", "mounting", "tilt_degrees", "grid_connected", "nature_of_study"]),
    ("Exposure, location & climate", [
        "start_year", "end_year", "location_latitude", "location_longitude", "location_provenance", "location_site_count",
        "PV zone", "pvcz_label", "climate_stressors", "extreme_weather_events", "extreme_weather_impact"]),
    ("Faults & reversible losses", ["fault_causality_stated", "soiling_handling"]),
    ("Measurement & analysis", [
        "measurement_approach", "measurement_correction", "calibration_reported", "source_of_initial_power_value",
        "other_examination_techs", "sample_size_for_rate", "measurement_count",
        "number_of_measurements_for_degradation_analysis", "uncertainty_reported", "goodness_of_fit_reported",
        "data_preprocessing_reported", "preprocessing_steps", "study_limitations_stated"]),
    ("Study quality", ["quality_design_n", "quality_reporting_n", "quality_flags_met", "quality_flags_missing"]),
    ("Notes", ["n_data_points_in_paper", "note", "other_studies"]),
]
UNITS = {"duration_in_years": "yr", "tilt_degrees": "°", "pv_module_nominal_power_watts": "W",
         "capacity_other_basis_watts": "W", "total_power_loss_loss_in_percent": "%", "rate_range_low": "%/yr",
         "rate_range_high": "%/yr", "location_latitude": "°", "location_longitude": "°"}
CHIPPED = {"rate_provenance", "location_provenance", "capacity_basis", "measurement_correction", "measurement_approach",
           "soiling_handling", "data_preprocessing_reported", "uncertainty_reported", "goodness_of_fit_reported",
           "voltage_decrease_reported", "current_decrease_reported", "total_power_loss_type_of_power_loss",
           "extreme_weather_impact", "mounting_detail", "mounting", "scope_of_study", "analysis_method",
           "source_of_initial_power_value", "pv_tech", "PV zone", "nature_of_study", "koppen_zone", "pvcz_label",
           "rate_affected_by_reversible", "soiling_effect_on_rate", "snow_status", "shading_status", "reversible_rate_flag"}
QUIET = {"not_reported", "not reported", "none_reported", "absent", "unclear", "unspecified", "not_addressed", "not_stated", "unknown", "none", "other"}
GLYPH = {"reversible_losses": {"soiling": "∴", "snow or ice cover": "❄︎", "shading": "◐"},
         "mounting_detail": {"ground-rack": "▁", "roof-flush": "⌂", "roof-rack": "⌂", "roof-unspecified": "⌂",
                             "facade": "▯", "tracker": "↻"},
         "rate_provenance": {"stated_by_author": "“", "read_from_figure": "⌗", "computed_from_total_loss": "Σ",
                             "computed_two_point": "Σ", "computed_from_fit": "Σ", "computed_other": "Σ"}}


# ---------------------------------------------------------------- helpers
def plain(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return int(v)                                    # 2014.0 -> 2014 for display
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if hasattr(v, "tolist") and not isinstance(v, str):
        try:
            return plain(v.tolist())
        except Exception:
            pass
    if hasattr(v, "item"):
        try:
            return plain(v.item())
        except Exception:
            pass
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except Exception:
            return v
    return v if isinstance(v, (int, float, bool, str)) else str(v)


def isnil(v):
    return v is None or v == "" or (isinstance(v, list) and not v)


def pretty(s):
    return str(s).replace("_", " ")


def nfmt(v, d=3):
    try:
        return f"{float(v):,.{d}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(v)


def cap_fmt(w):
    w = float(w)
    if w >= 1e6:
        return nfmt(w / 1e6, 2), "MW"
    if w >= 1e3:
        return nfmt(w / 1e3, 2), "kW"
    return nfmt(w, 1), "W"


def first_author(s):
    s = str(s or "")
    if s.upper().startswith("AUTHOR_NAMES:"):
        s = s.split(":", 1)[1]
    return s.split(",")[0].strip()


# ---------------------------------------------------------------- data
raw = pd.read_pickle(DATA)
if "validation_status" in raw.columns:
    raw = raw[raw["validation_status"].notna()]
if "validation_row_verdict" in raw.columns:
    raw = raw[raw["validation_row_verdict"].astype(str) != "drop"]
if "excluded_reason" in raw.columns:
    raw = raw[~raw["eid"].isin(raw.loc[raw["excluded_reason"].notna(), "eid"])]
df = raw.copy()
df["rk"] = [int(i) for i in df.index]                    # stable row key = index in the pkl
df = df.set_index("rk", drop=False)
df["_rate"] = pd.to_numeric(df.get("annual_power_deg_rate_in_percent"), errors="coerce")
df["_dur"] = pd.to_numeric(df.get("duration_in_years"), errors="coerce")
df["_year"] = pd.to_numeric(df.get("year"), errors="coerce")


def _affecting(items):
    """reversible losses inside this point's rate: uncorrected, and unknown marked with '?'"""
    out = []
    for x in plain(items) or []:
        if isinstance(x, dict) and x.get("present") in ("present", "negligible"):
            if x.get("effect_on_rate") == "uncorrected":
                out.append(x.get("type"))
            elif x.get("effect_on_rate") in (None, "unknown"):
                out.append(f"{x.get('type')} (possibly)")
    return out


if "reversible_items" in df.columns and "reversible_affecting_rate" not in df.columns:
    df["reversible_affecting_rate"] = df["reversible_items"].map(_affecting)

def _bounds(col, step, q=0.005):
    x = df[col].dropna()
    if x.empty:
        return 0, 1
    lo, hi = x.quantile(q), x.quantile(1 - q)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


# range filters: (slider id, data column, label, step). A handle left at its end means "no bound",
# so values beyond the slider span (e.g. soiling rates of -30 %/yr) and rows without a value stay in.
RANGES = [("r_rate", "_rate", "Rate (%/yr)", 0.1), ("r_dur", "_dur", "Duration (yr)", 0.5), ("r_year", "_year", "Year", 1)]
FIXED_B = {"r_rate": (-10.0, 2.0), "r_dur": (0.0, 30.0)}     # slider span; values beyond an end are kept while its handle sits there
RANGE_B = {rid: FIXED_B.get(rid) or _bounds(col, st) for rid, col, _, st in RANGES}

FIELD_DESC = {}
if DICT_CSV.exists():
    for _, r in pd.read_csv(DICT_CSV).iterrows():
        FIELD_DESC[r["column"]] = plain(r.get("description")) or ""

FIELD_DESC["reversible_affecting_rate"] = ("observed reversible losses that are still inside this point's rate: the data the rate "
                                           "comes from contain them and they were not cleaned, filtered or corrected. "
                                           "'(possibly)' = observed, but the paper does not say. Candidates to exclude.")
FIELD_DESC["reversible_losses"] = "soiling / snow / shading the paper reports for this point (observed)"
FIELD_DESC["number_of_pv_modules"] = "number of modules × module nameplate power"

# PDFs: the file the validation read, else any file whose name contains the eid
pdf_names = sorted(p.name for p in PDF_DIR.iterdir() if p.suffix.lower() == ".pdf") if PDF_DIR.exists() else []
pdf_set = set(pdf_names)
pdf_map = {}
for eid, g in df.groupby("eid"):
    src = plain(g["validation_source_pdf"].iloc[0]) if "validation_source_pdf" in g else None
    if src and src in pdf_set:
        pdf_map[eid] = src
        continue
    e = str(eid).lower()
    hit = next((n for n in pdf_names if e in n.lower()), None)
    if hit:
        pdf_map[eid] = hit


def _list_col(c):
    return df[c].map(lambda v: plain(v) or []) if c in df.columns else pd.Series([[]] * len(df), index=df.index)


FAULTS = _list_col("faults")
papers = []
for eid, g in df.groupby("eid", sort=False):
    h = g.iloc[0]
    rates = g["_rate"].dropna()
    papers.append({
        "eid": eid, "title": plain(h.get("title")) or eid, "year": plain(h.get("year")),
        "source": plain(h.get("source_title")), "author": first_author(plain(h.get("authors_with_affiliations"))), "doi": plain(h.get("doi")),
        "n": len(g), "rate": ("" if rates.empty else nfmt(rates.iloc[0], 2) if len(rates) == 1 or rates.min() == rates.max()
                              else f"{nfmt(rates.min(), 2)} … {nfmt(rates.max(), 2)}"),
        "pdf": "✓" if eid in pdf_map else "",
    })
PAPERS = pd.DataFrame(papers).set_index("eid", drop=False)
print(f"{DATA.name}: {len(df)} data points, {len(PAPERS)} papers, {len(pdf_map)} with a PDF")


# ---------------------------------------------------------------- review log
# review log in S3 (embedded mode): local sqlite is the working copy, S3 the durable one
_s3_lock = threading.Lock()


def _s3():
    import boto3
    return boto3.client("s3")


def s3_pull():
    if not S3_BUCKET:
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        _s3().download_file(S3_BUCKET, S3_LOG_KEY, str(LOG))
        print(f"review log <- s3://{S3_BUCKET}/{S3_LOG_KEY}")
    except Exception as e:                                 # first run: nothing there yet
        print(f"review log: no copy in S3 yet ({type(e).__name__}); starting a new one")


def s3_push():
    if not S3_BUCKET:
        return
    def _up():
        with _s3_lock:
            try:
                tmp = LOG.with_suffix(".upload")
                with sqlite3.connect(LOG, timeout=10) as src, sqlite3.connect(tmp) as dst:
                    src.backup(dst)                         # consistent snapshot, WAL included
                _s3().upload_file(str(tmp), S3_BUCKET, S3_LOG_KEY)
            except Exception as e:
                print(f"WARNING: review log upload to S3 failed: {e}")
    threading.Thread(target=_up, daemon=True).start()


s3_pull()


def _con():
    c = sqlite3.connect(LOG, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS log(eid TEXT, rk INTEGER, field TEXT, status TEXT, comment TEXT,"
              " reviewer TEXT, label TEXT, ts TEXT)")
    return c


def _migrate():
    """soft delete: deleted rows stay in the file, hidden everywhere (run once at start-up)"""
    with _con() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(log)")}
        for col in ("deleted_ts", "deleted_by"):
            if col not in cols:
                try:
                    c.execute(f"ALTER TABLE log ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass                                   # added meanwhile by another process


_migrate()


def read_log(include_deleted=False):
    try:
        return _read_log(include_deleted)
    except Exception:                                     # e.g. the log file was swapped for an old one
        _migrate()
        return _read_log(include_deleted)


def _read_log(include_deleted=False):
    with _con() as c:
        lg = pd.read_sql("SELECT rowid AS id, * FROM log" + ("" if include_deleted else " WHERE deleted_ts IS NULL")
                         + " ORDER BY ts", c)
    for k in ("field", "status", "comment", "reviewer", "label"):
        lg[k] = lg[k].fillna("")
    return lg


def delete_entries(ids, who):
    ids = [int(i) for i in ids if i is not None]
    if not ids:
        return 0
    with _con() as c:
        c.executemany("UPDATE log SET deleted_ts=?, deleted_by=? WHERE rowid=? AND deleted_ts IS NULL",
                      [(dt.datetime.now().isoformat(timespec="seconds"), who or "", i) for i in ids])
    s3_push()
    return len(ids)


def log_event(eid, rk, field="", status="", comment="", reviewer=""):
    label = plain(df.at[rk, "data_point_label"]) if rk is not None and "data_point_label" in df.columns else ""
    with _con() as c:
        c.execute("INSERT INTO log (eid, rk, field, status, comment, reviewer, label, ts) VALUES (?,?,?,?,?,?,?,?)",
                  (eid, rk, field, status, comment, reviewer, label or "", dt.datetime.now().isoformat(timespec="seconds")))
    s3_push()


def row_status(lg):
    s = lg[(lg.status != "") & lg.rk.notna()]
    return s.groupby("rk").last()[["status", "reviewer"]] if len(s) else pd.DataFrame(columns=["status", "reviewer"])


# ---------------------------------------------------------------- user pills
PALETTE = ["#3b6ea8", "#b0603a", "#3f8a6b", "#8e5a8f", "#a07c2c", "#4f86a0", "#6b5b9a", "#666"]


def cap_name(u):
    """'baojie' / 'BAOJIE' -> 'Baojie' (each word capitalised)"""
    return " ".join(w[:1].upper() + w[1:].lower() for w in str(u or "").split()) or "?"


# fixed colour per reviewer (the first four keep the colours they had); others fall back to a hash
USER_COLOR = {"Baojie": "#b0603a", "Martin": "#4f86a0", "Dirk": "#a07c2c", "Anubhav": "#8e5a8f", "Dax": "#3f8a6b"}


def ucolor(u):
    n = cap_name(u)
    return USER_COLOR.get(n) or PALETTE[int(hashlib.md5(n.encode()).hexdigest(), 16) % len(PALETTE)]


def pill(u):
    return html.Span(cap_name(u), className="pill", style={"background": ucolor(u)})


# ---------------------------------------------------------------- field rendering
def cell(rk, f):
    """(value, evidence) for one field of one data point"""
    if f not in df.columns:
        return None, {}
    v = plain(df.at[rk, f])
    ev = plain(df.at[rk, "validation_field_evidence"]) if "validation_field_evidence" in df.columns else None
    e = (ev or {}).get(f) if isinstance(ev, dict) else None
    e = e if isinstance(e, dict) and e.get("verdict") != "rejected" else {}
    # a field the validation CHANGED keeps its quote / reason in validation_changes, not always in
    # validation_field_evidence - use the change when it produced the value shown
    if not (e.get("evidence_quote") or e.get("reason")) and "validation_changes" in df.columns:
        ch = plain(df.at[rk, "validation_changes"])
        for c in (ch if isinstance(ch, list) else []):
            if isinstance(c, dict) and c.get("field") == f and _same(c.get("new_value"), v):
                e = {"value": v, "evidence_quote": c.get("evidence_quote"), "evidence_page": c.get("evidence_page"),
                     "reason": c.get("reason"), "confidence": c.get("confidence"), "_changed_from": c.get("old_value")}
    # Added data points were created as copies of the paper's first record, so their per-field evidence
    # (and the rate explanation) can still describe that record. Evidence recorded for a different value
    # is therefore not shown as if it supported this one.
    if e and "value" in e and not _same(e.get("value"), v):
        e = {"_foreign": e.get("value")}
    if f == "annual_power_deg_rate_in_percent" and ("_foreign" in e or not e) and _is_added(rk):
        q = plain(df.at[rk, "added_row_evidence"]) if "added_row_evidence" in df.columns else None
        if q:
            e = {"evidence_quote": q, "evidence_page": plain(df.at[rk, "added_row_page"]),
                 "reason": "Row-level evidence recorded when this data point was added.", **e}
    return v, e


def _same(a, b):
    a, b = plain(a), plain(b)
    if isnil(a) and isnil(b):
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        pass
    if isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    return str(a) == str(b)


def _is_added(rk):
    return "row_origin" in df.columns and plain(df.at[rk, "row_origin"]) == "added_by_validation"


def _copied_expl(rk):
    """True when an added data point's rate explanation is the same text as another data point's of the paper"""
    if "annual_power_deg_explanation" not in df.columns or not _is_added(rk):
        return False
    t = plain(df.at[rk, "annual_power_deg_explanation"])
    if isnil(t):
        return False
    same = df[(df.eid == df.at[rk, "eid"]) & (df.rk != rk)]["annual_power_deg_explanation"].map(plain)
    return (same == t).any()


EFFECT_TXT = {"excluded": "not in the rate (excluded)", "corrected": "corrected out of the rate",
              "uncorrected": "inside the rate (uncorrected)", "negligible": "negligible effect on the rate",
              "unknown": "effect on the rate unknown", "none_reported": "none reported"}
EFFECT_LEGEND = ("Effect on the rate - not in the rate: the measurement avoids it (cleaned before measuring, indoor "
                 "STC, filtered days) · corrected out: in the data but corrected in the analysis · inside the rate: in "
                 "the data and not handled, so the rate is likely too steep · negligible: the authors say it is small "
                 "· unknown: present, but unclear whether the rate's data contain it")
VALUE_TXT = {"rate_affected_by_reversible": EFFECT_TXT, "soiling_effect_on_rate": EFFECT_TXT}


def chip(f, x):
    s = str(x)
    g = GLYPH.get(f, {}).get(s)
    txt = VALUE_TXT.get(f, {}).get(s, pretty(s))
    return html.Span(([html.Span(g, className="gl")] if g else []) + [txt], className="chip" + (" mute" if s in QUIET else ""))


def render_value(f, v, rk, big=False):
    if f == "number_of_pv_modules" and big is not None:
        pw = cell(rk, "pv_module_nominal_power_watts")[0]
        if not isnil(v) or not isnil(pw):
            n = html.Span("?" if isnil(v) else nfmt(v, 0), className="num")
            if isnil(pw):
                return n
            return html.Span([n, html.Span(" × ", className="unit"), html.Span([nfmt(pw, 1), html.Span("W", className="unit")], className="num")])
    if f == "reversible_affecting_rate" and isnil(v):
        items = plain(df.at[rk, "reversible_items"]) if "reversible_items" in df.columns else None
        present = [x for x in (items or []) if isinstance(x, dict) and x.get("present") in ("present", "negligible")]
        return html.Span("none" if present else "none reported", className="nil")
    if isnil(v):
        return html.Span("none" if isinstance(v, list) else "—", className="nil")
    if isinstance(v, bool):
        return html.Span("✓ yes", className="bool") if v else html.Span("✕ no", className="bool n")
    if f == "annual_power_deg_rate_in_percent":
        return html.Span([("+" if v > 0 else "") + nfmt(v, 3), html.Span("%/yr", className="unit")], className="num" + (" big" if big else ""))
    if f == "system_capacity_watts":
        n, u = cap_fmt(v)
        return html.Span([n, html.Span(u, className="unit")], className="num")
    if f == "quality_score":
        m = int(plain(df.at[rk, "quality_score_max"]) or 12) if "quality_score_max" in df.columns else 12
        return html.Span([html.Span([str(v), html.Span(f"/ {m}", className="unit")], className="num"),
                          html.Span([html.I(className="on" if i < v else "") for i in range(m)], className="meter")])
    if f == "duration_in_years":
        s, e = cell(rk, "start_year")[0], cell(rk, "end_year")[0]
        return html.Span([html.Span([nfmt(v, 2), html.Span("yr", className="unit")], className="num")]
                         + ([html.Span(f" {s or '?'}–{e or '?'}", className="nil small")] if (s or e) else []))
    if f == "koppen_zone":
        z = cell(rk, "PV zone")[0]
        return html.Span([chip(f, v)] + ([chip(f, z)] if z else []))
    if f == "mounting_detail":
        t = cell(rk, "tilt_degrees")[0]
        return html.Span([chip(f, v)] + ([html.Span(f"{nfmt(t, 1)}° tilt", className="chip mute")] if t is not None else []))
    if f == "country":
        return html.Span(str(v))
    if f == "reversible_affecting_rate":
        return html.Span([html.Span([html.Span(GLYPH["reversible_losses"].get(x.replace(" (possibly)", ""), ""), className="gl"), pretty(x)],
                                    className="chip" + (" mute" if "possibly" in x else "")) for x in v])
    if isinstance(v, list):
        if v and isinstance(v[0], (dict, list)):
            return html.Span("; ".join(json.dumps(x, ensure_ascii=False) if not isinstance(x, dict)
                                       else ", ".join(f"{pretty(k)}: {y}" for k, y in x.items()) for x in v), className="raw")
        return html.Span([chip(f, x) for x in v])
    if isinstance(v, dict):
        return html.Span(", ".join(f"{pretty(k)}: {y}" for k, y in v.items()), className="raw")
    if f in CHIPPED:
        return chip(f, v)
    if isinstance(v, (int, float)):
        return html.Span([nfmt(v, 4 if "itude" in f else 3)] + ([html.Span(UNITS[f], className="unit")] if f in UNITS else []), className="num")
    return html.Span(str(v), className="raw" if len(str(v)) > 60 else "")


HOVER_FIELDS = {xf for lst in EXTRA_OF.values() for xf, _ in lst}      # shown inside another field's hover

# structured items (extract_faults.py) listed in the hover card of the field they back
ITEM_OF = {
    "faults": ("fault_items", lambda x: True),
    "faults_major": ("fault_items", lambda x: x.get("role") == "causal"),
    "stressors_causal": ("stressor_items", lambda x: True),
}
ROLE_ORDER = {"causal": 0, "observed": 1, "suspected": 2, "checked_absent": 3, "context": 4}


def item_rows(f, rk):
    col, keep = ITEM_OF.get(f, (None, None))
    if not col or col not in df.columns:
        return []
    items = [x for x in (plain(df.at[rk, col]) or []) if isinstance(x, dict) and keep(x)]
    items.sort(key=lambda x: (ROLE_ORDER.get(x.get("role"), 9), str(x.get("term") or x.get("type"))))
    out = []
    for x in items:
        name = x.get("term") or x.get("type")
        if f == "__unused__":
            eff = x.get("effect_on_rate") or "unknown"
            tags = [EFFECT_TXT.get(eff, eff)]
            why = [w for w in (x.get("analysis_treatment"), ", ".join(c for c in (x.get("cleaning") or []) if c != "not_reported"))
                   if w and w != "not_reported"]
            meta = [("handling", "; ".join(why)), ("why", x.get("reason")), ("loss", x.get("loss"))]
        elif col == "reversible_items":
            tags = [x.get("present")] + (x.get("cleaning") or []) + [x.get("analysis_treatment")]
            meta = [("source", ", ".join(x.get("source") or [])), ("frequency", x.get("cleaning_frequency")), ("loss", x.get("loss"))]
        elif col == "stressor_items":
            tags = [x.get("role"), ("→ " + ", ".join(x["linked_faults"])) if x.get("linked_faults") else None,
                    "demoted from causal" if x.get("_demoted_from") else None]
            meta = [("level", x.get("level")), ("authors' words", x.get("raw_term"))]
        else:
            tags = [x.get("role"), "dominant" if x.get("dominant") else None] + [d for d in (x.get("detection") or []) if d != "not_reported"]
            meta = [("prevalence", x.get("prevalence")), ("impact", x.get("impact")), ("authors' words", x.get("raw_term"))]
        tags = [t for t in tags if t and t != "not_reported"]
        meta = [(k, v) for k, v in meta if v and str(v).strip().lower() != str(name).lower()]
        q, pg = x.get("quote"), x.get("page")
        out.append(html.Div([
            html.Div([html.B(pretty(name)), " "] + [html.Span(t if t in EFFECT_TXT.values() else pretty(t),
                                                              className="chip" + (" mute" if t in QUIET else "")) for t in tags]),
            html.Div(" · ".join(f"{k}: {v}" for k, v in meta), className="nil small") if meta else None,
            html.Div(f"“{q}”" + (f"  (p. {pg})" if pg not in (None, "") else ""), className="tquote") if q else None,
        ], className="titem"))
    return out


def tip_card(f, label, v, e, rk):
    """hidden detail block; the page JS shows it in a floating card on hover"""
    parts = [html.Button("×", className="tclose", title="close"), html.Div(label, className="tt"), html.Div(f, className="tf")]
    if FIELD_DESC.get(f):
        parts.append(html.Div(FIELD_DESC[f], className="tdesc"))
    parts.append(html.Div([html.Div("Value:", className="th"), render_value(f, v, rk)], className="tsec"))
    rf = RAW_OF.get(f)
    q = e.get("evidence_quote") or e.get("quote")
    pg = e.get("evidence_page", e.get("page"))
    norm = lambda t: re.sub(r"[^0-9a-z%]+", "", str(t).lower())
    rv = cell(rk, rf)[0] if rf else None
    # raw text = the verbatim span stored with the record; quote = the sentence the check cited.
    # When one contains the other they say the same thing: show it once (the longer), with its page.
    dup = bool(q) and not isnil(rv) and (norm(rv) in norm(q) or norm(q) in norm(rv))
    if dup:
        rv = q if len(str(q)) >= len(str(rv)) else rv
        q = None
    items = item_rows(f, rk)
    if items:
        parts.append(html.Div([html.Div("Per item (role · details · quote):", className="th")] + items,
                              className="tsec"))
        rf = None
    if rf:
        raw_hdr = RAW_HDR.get(f, "Raw text") + (f" (p. {pg})" if dup and pg not in (None, "") else "") + ":"
        if f in RAW_HDR and not isnil(rv):
            rv = str(rv).replace(" | ", "\n")                  # one line per loss
        parts.append(html.Div([html.Div([raw_hdr] + ([html.Button("Copy", className="copybtn", **{"data-copy": "traw"})] if not isnil(rv) else []), className="th"),
                               html.Div(str(rv) if not isnil(rv) else "— not recorded", className="traw" + ("" if not isnil(rv) else " nil"))],
                              className="tsec"))
    rows_x = []
    for xf, xl in EXTRA_OF.get(f, []):
        if xf not in df.columns:
            continue
        xv = cell(rk, xf)[0]
        if xf == "soiling_effect_on_rate" and xv == cell(rk, "rate_affected_by_reversible")[0]:
            continue                                            # same as the overall line above: say it once
        if isnil(xv) and xf not in ("degradation_metric", "rate_provenance", "location_latitude", "location_longitude"):
            continue                                            # keep the card short: skip empty extras
        val = render_value(xf, xv, rk)
        if xf == "annual_power_deg_explanation" and _copied_expl(rk):
            val = html.Div([html.Div("⚠ copied from another data point of this paper; it may describe that one, not this", className="warn"), val])
        rows_x.append(html.Tr([html.Td(xl, className="xk"), html.Td(val, className="xv")]))
    if rows_x:
        parts.append(html.Div([html.Div("Related:", className="th"), html.Table(rows_x, className="xtab")], className="tsec"))
    if "_foreign" in e:
        parts.append(html.Div([html.Div("Evidence:", className="th"),
                               html.Div(["⚠ The field-level quote on record belongs to another data point of this paper (value ",
                                         html.B(str(e["_foreign"])), "), so it is not shown here."], className="warn")], className="tsec"))
    if q:
        parts.append(html.Div([html.Div(["Quote from the PDF" + (f" (p. {pg})" if pg not in (None, "") else "") + ":",
                                         html.Button("Copy", className="copybtn", **{"data-copy": "tquote"})], className="th"),
                               html.Div(f"“{q}”", className="tquote", **{"data-text": str(q)})], className="tsec"))
    if e.get("reason") and e.get("verdict") != "fault_extract":      # that reason only names the script
        parts.append(html.Div([html.Div("Reason:", className="th"), html.Div(e["reason"], className="treason")], className="tsec"))
    if e.get("confidence"):
        parts.append(html.Div([html.Span("Confidence: ", className="th inline"), str(e["confidence"])], className="tsec"))
    if len(parts) <= (4 if FIELD_DESC.get(f) else 3) and not rf:
        parts.append(html.Div("no quote or reason recorded", className="nil small"))
    return html.Div(parts, className="tipsrc")


def field_box(rk, f, label, ncomments, main=False):
    v, e = cell(rk, f)
    long = isinstance(v, str) and len(v) > 60
    badge = [html.Span(f"💬{ncomments}", className="cbadge", title=f"{ncomments} comment(s) on this field")] if ncomments else []
    return html.Div([
        html.Div([label] + badge, className="k"),
        html.Div(render_value(f, v, rk, big=main and f == "annual_power_deg_rate_in_percent"), className="v"),
        tip_card(f, label, v, e, rk),
    ], className=("card" if main else "f") + (" empty" if isnil(v) else "") + (" long" if long else "")
       + (" rate" if main and f == "annual_power_deg_rate_in_percent" and not isnil(v) else ""), **{"data-field": f})


def render_record(eid, rk, lg):
    p = PAPERS.loc[eid]
    rows = df[df.eid == eid]
    st = row_status(lg)
    fc = lg[(lg.comment != "") & (lg.eid == eid) & (lg.rk == rk) & (lg.field != "")].groupby("field").size().to_dict()
    head = html.Div([
        html.H2(p.title),
        html.Div([html.Span(f"{p.author} et al." if p.author else ""), html.Span(str(p.year or "")),
                  html.A(f"doi:{p.doi}", href=f"https://doi.org/{p.doi}", target="_blank") if p.doi else None,
                  html.Span(p.source) if p.source else None, html.Span(eid)], className="meta"),
    ] + ([html.Details([html.Summary("paper summary"), html.Div(plain(rows.iloc[0].get("summary")), className="sumbody")], className="sum")]
         if plain(rows.iloc[0].get("summary")) else []), className="ph")
    tabs = []
    n = len(rows)
    cur_i = [int(x) for x in rows.rk].index(int(rk)) + 1 if int(rk) in [int(x) for x in rows.rk] else 1
    cur_lbl = plain(df.at[rk, "data_point_label"]) or ""
    dp_ev = plain(df.at[rk, "added_row_evidence"]) if "added_row_evidence" in df.columns and _is_added(rk) else None
    dp_pg = plain(df.at[rk, "added_row_page"]) if "added_row_page" in df.columns else None
    dp_head = html.Div([
        html.Span(f"{n} data point{'s' if n > 1 else ''} extracted from this paper", className="dpn"),
        html.Span(f"showing #{cur_i}" + (f" of {n}" if n > 1 else "") + (f" · {cur_lbl}" if cur_lbl else ""), className="dpcur"),
    ] + ([html.Div([html.Span("Evidence for this data point" + (f" (p. {dp_pg})" if dp_pg not in (None, "") else "") + ": ", className="dpevk"),
                    html.Span(f"“{dp_ev}”", className="dpevq")], className="dpev")] if dp_ev else []), className="dphead")
    if n > 1:
        for i_, r in enumerate(rows.itertuples(), 1):
            s = st["status"].get(r.rk, "") if len(st) else ""
            rate = plain(df.at[r.rk, "annual_power_deg_rate_in_percent"]) if "annual_power_deg_rate_in_percent" in df.columns else None
            tabs.append(html.Button([html.Span(className=f"sdot {s}"), html.B(f"#{i_} ", className="tn")]
                                    + ([html.Span(f"{nfmt(rate, 2)}%/yr", className="n")] if rate is not None else [])
                                    + [" · ", plain(df.at[r.rk, "data_point_label"]) or f"row {r.rk}"],
                                    id={"type": "dp", "rk": int(r.rk)}, n_clicks=0,
                                    className="tab" + (" active" if r.rk == rk else ""),
                                    title=plain(df.at[r.rk, "data_point_label"]) or ""))
    hero = html.Div([field_box(rk, f, lbl, fc.get(f, 0), main=True) for f, lbl in MAIN
                     if f in df.columns and f not in HOVER_FIELDS], className="hero")
    groups = []
    for title, fs in GROUPS:
        fs = [f for f in fs if f in df.columns and f not in RAW_FIELDS and f not in HOVER_FIELDS]
        filled = [f for f in fs if not isnil(cell(rk, f)[0])]
        if not fs:
            continue
        groups.append(html.Details([
            html.Summary([title, html.Span(f"{len(filled)} / {len(fs)} filled", className="cnt")]),
            html.Div([field_box(rk, f, pretty(f), fc.get(f, 0)) for f in (filled + [f for f in fs if f not in filled])], className="fields"),
        ], className="grp"))
    return [head, dp_head] + ([html.Div(tabs, className="tabs")] if tabs else []), hero, groups


def render_thread(eid, rk, lg):
    cm = lg[((lg.comment != "") | (lg.status != "")) & (lg.eid == eid) & (lg.rk.isna() | (lg.rk == rk))]
    if cm.empty:
        return html.Div("no comments or marks yet", className="hint")
    out = []
    for r in cm.itertuples():
        where = "paper" if pd.isna(r.rk) else (pretty(r.field) if r.field else "data point")
        body = (html.Span(r.comment, className="ctext") if r.comment
                else html.Span(["marked ", html.B(r.status, className=f"stxt {r.status}")], className="ctext"))
        out.append(html.Div([pill(r.reviewer), html.Span(where, className="ctag"), body,
                             html.Span(r.ts[:16].replace("T", " "), className="cts"),
                             html.Button("×", id={"type": "cdel", "id": int(r.id)}, n_clicks=0, className="cdel",
                                         title="delete this entry")], className="cmsg"))
    return out


def log_rows(lg, who=None, kind=None, text=None):
    lg = lg.sort_values("ts", ascending=False)
    if who:
        lg = lg[lg.reviewer.map(cap_name).isin(who)]
    if kind == "comment":
        lg = lg[lg.comment != ""]
    elif kind == "status":
        lg = lg[lg.status != ""]
    if text:
        t = text.lower()
        lg = lg[lg.comment.str.lower().str.contains(t, regex=False) | lg.eid.str.lower().str.contains(t, regex=False)
                | lg.label.str.lower().str.contains(t, regex=False)]
    import html as _h
    esc = lambda t: _h.escape(str(t or "")).replace("*", "&#42;").replace("_", "&#95;").replace("`", "&#96;")  # noqa: E731
    out = []
    for r in lg.itertuples():
        who = cap_name(r.reviewer)
        what = (f'<span class="ctext">{esc(r.comment)}</span>' if r.comment else
                f'<span class="sdot {r.status}"></span><b class="stxt {r.status}">{esc(r.status)}</b>')
        out.append({"id": int(r.id), "time": f'<span class="cts">{r.ts[:16].replace("T", " ")}</span>',
                    "reviewer": f'<span class="pill" style="background:{ucolor(who)}">{esc(who)}</span>',
                    "what": what, "kind": "comment" if r.comment else "status", "paper": esc(r.eid), "eid": r.eid,
                    "point": esc("(whole paper)" if pd.isna(r.rk) else r.label or f"row {int(r.rk)}"),
                    "field": esc(pretty(r.field)) if r.field else "", "rk": None if pd.isna(r.rk) else int(r.rk)})
    return out


# ---------------------------------------------------------------- app
if EMBED is not None:
    app = Dash(__name__, server=EMBED.server, url_base_pathname=BASE, title="PV dataset review",
               suppress_callback_exceptions=True)
else:
    app = Dash(__name__, title="PV dataset review", suppress_callback_exceptions=True)
server = app.server


@server.route(f"{BASE}pdf/<path:fname>")
def review_serve_pdf(fname):
    if PUBLIC_MODE or not (PDF_DIR / fname).exists():
        abort(404)
    return send_from_directory(PDF_DIR, fname, mimetype="application/pdf")


def opts(col, list_col=False):
    if col not in df.columns:
        return []
    vals = set()
    for v in df[col]:
        v = plain(v)
        for x in (v if isinstance(v, list) else [v]):
            if not isnil(x):
                vals.add(str(x))
    o = [{"label": pretty(v), "value": v} for v in sorted(vals)]
    return o + [{"label": "(not reported)", "value": NA}]


def dd(id_, col, ph):
    return dcc.Dropdown(id=id_, options=opts(col), multi=True, placeholder=ph, className="dd")


def num(id_, v=None, step="any"):
    return dcc.Input(id=id_, type="number", value=v, step=step, className="numin", debounce=True)


def range_block(rid, label, step):
    lo, hi = RANGE_B[rid]
    fmt = (lambda v: str(int(v))) if step >= 1 else (lambda v: nfmt(v, 1))
    marks = {float(v): fmt(v) for v in np.linspace(lo, hi, 7 if rid == "r_rate" else 5)}
    marks[float(lo)] = ("≤" if rid == "r_rate" else "") + marks[float(lo)]
    marks[float(hi)] = marks[float(hi)] + ("+" if rid != "r_year" else "")
    return html.Div([
        html.Label(label),
        num(rid + "_lo", lo, step),
        html.Div(dcc.RangeSlider(id=rid, min=lo, max=hi, step=step, value=[lo, hi], marks=marks,
                                 allowCross=False, updatemode="mouseup", tooltip={"placement": "bottom"}), className="slider"),
        num(rid + "_hi", hi, step),
    ], className="rng")


TYPE_LABEL = {"J": "Journal", "C": "Conference", "Review": "Review", "B": "Book", "Other": "Other"}
FILTERS = [("f_type", "type_code", "Document type"), ("f_srcname", "source_title", "Journal / conference"),
           ("f_tech", "pv_tech", "PV tech"), ("f_zone", "PV zone", "Climate zone"), ("f_koppen", "koppen_zone", "Köppen"),
           ("f_mount", "mounting_detail", "Mounting"), ("f_scope", "scope_of_study", "Scope"),
           ("f_country", "country", "Country"), ("f_method", "analysis_method", "Method"),
           ("f_prov", "rate_provenance", "Rate source"), ("f_faults", "faults", "Faults"),
           ("f_major", "faults_major", "Causal faults"), ("f_stress", "stressors_causal", "Stressors"),
           ("f_rev", "reversible_losses", "Reversible losses observed"),
           ("f_inrate", "reversible_affecting_rate", "Reversible losses affecting rate"),
           ("f_rflag", "reversible_rate_flag", "Reversible-loss flag")]
FILTERS = [x for x in FILTERS if x[1] in df.columns or x[0] in ("f_type", "f_srcname")]
LIST_FILTER = {"faults", "faults_major", "stressors_causal", "reversible_losses", "reversible_affecting_rate"}
_PAPER_SRC = df.drop_duplicates("eid")[["eid"] + [c for c in ("type_code", "source_title") if c in df.columns]]


def type_options():
    if "type_code" not in _PAPER_SRC:
        return []
    n = _PAPER_SRC["type_code"].map(plain).value_counts()
    return [{"label": f"{TYPE_LABEL.get(t, t)} ({c})", "value": t} for t, c in n.items() if not isnil(t)]


def source_options(types=None):
    if "source_title" not in _PAPER_SRC:
        return []
    p = _PAPER_SRC
    if types and "type_code" in p:
        p = p[p["type_code"].map(plain).isin(types)]
    n = p["source_title"].map(plain).dropna().value_counts()
    return [{"label": f"{t} ({c})", "value": t} for t, c in sorted(n.items(), key=lambda x: (-x[1], x[0]))]

main = html.Div([
    html.Div([html.H3("PV field degradation dataset · review", className="h3"), html.Div(id="pickslot"),
              html.Button("Export reviews", id="export", className="btn"), dcc.Download(id="dl"),
              html.Span(id="whoami", className="who"), html.Button("Log out", id="logout", className="btn")], className="titlebar"),
    html.Details([
        html.Summary(["All reviews ", html.Span(id="log_count", className="cnt")]),
        html.Div([
            dcc.Dropdown(id="lf_who", multi=True, placeholder="reviewer", className="dd"),
            dcc.Dropdown(id="lf_kind", options=[{"label": "comments and marks", "value": "all"},
                                                {"label": "comments only", "value": "comment"},
                                                {"label": "status marks only", "value": "status"}],
                         value="all", clearable=False, className="dd"),
            dcc.Input(id="lf_text", type="text", placeholder="search comment / eid / data point", debounce=True, className="txt"),
            html.Button("Select all", id="log_all", n_clicks=0, className="btn"),
            html.Button("Clear selection", id="log_none", n_clicks=0, className="btn"),
            dcc.ConfirmDialogProvider(html.Button("Delete selected", className="btn s-bad"), id="log_del",
                                      message="Delete the selected review entries? They are hidden from the app "
                                              "(kept in review_log.sqlite, marked deleted)."),
            html.Span(id="log_msg", className="msg"),
        ], className="logbar"),
        dash_table.DataTable(
            id="logtbl", data=[], row_selectable="multi", selected_rows=[], page_action="none",
            columns=[{"name": n, "id": i, "presentation": "markdown"}
                     for i, n in (("time", "time"), ("reviewer", "reviewer"), ("what", "comment / mark"),
                                  ("paper", "paper"), ("point", "data point"), ("field", "field"))],
            markdown_options={"html": True},
            style_table={"maxHeight": "320px", "overflowY": "auto"},
            style_header={"fontWeight": 600, "fontSize": 10.5, "background": "var(--surface-2)", "border": "none",
                          "textTransform": "uppercase", "letterSpacing": ".05em", "color": "var(--muted)",
                          "fontFamily": 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif'},
            style_cell={"fontSize": 12, "padding": "4px 8px", "textAlign": "left", "whiteSpace": "normal",
                        "fontFamily": 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif', "border": "none",
                        "borderBottom": "1px solid var(--line)", "maxWidth": 420, "cursor": "pointer"},
            style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": "var(--accent-soft)", "border": "none"},
                                    {"if": {"state": "selected"}, "backgroundColor": "var(--accent-soft)", "border": "none"}],
            style_cell_conditional=[{"if": {"column_id": "time"}, "width": "120px"}, {"if": {"column_id": "reviewer"}, "width": "80px"}],
            css=[{"selector": ".dash-spreadsheet-menu", "rule": "display:none"},
                 {"selector": "td, th, .dash-cell-value, .dash-header, .dash-cell-value p",
                  "rule": 'font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif !important'},
                 {"selector": ".dash-cell-value p", "rule": "margin:0"},
                 {"selector": ".cts", "rule": "color:var(--muted);font-size:11px"},
                 {"selector": ".pill", "rule": "font-size:11px;line-height:18px;padding:0 8px"}]),
        html.Div("click a row to open that paper / data point", className="hint small"),
    ], className="logpanel", id="logpanel"),
    html.Details([
        html.Summary("Filters"),
        html.Div([dcc.Dropdown(id="f_type", options=type_options(), multi=True, placeholder="Document type", className="dd"),
                  dcc.Dropdown(id="f_srcname", options=source_options(), multi=True, placeholder="Journal / conference name",
                               className="dd wide", optionHeight=44)]
                 + [dd(i, c, p) for i, c, p in FILTERS if i not in ("f_type", "f_srcname")] + [
            dcc.Dropdown(id="f_status", multi=True, placeholder="Review status", className="dd",
                         options=[{"label": "(unreviewed)", "value": ""}] + [{"label": s, "value": s} for s in STATUSES]),
            dcc.Input(id="f_text", type="text", placeholder="search title / author / eid / doi", debounce=True, className="txt"),
        ], className="filters"),
        html.Div([range_block(rid, lbl, st) for rid, _, lbl, st in RANGES], className="filters ranges"),
        html.Div([html.Button("Apply filters", id="apply", n_clicks=0, className="btn primary"),
                  html.Button("Reset", id="reset", n_clicks=0, className="btn")], className="fbtns"),
    ], open=True, className="fwrap"),
    html.Div([
        html.Div([
            html.Div(id="counter", className="counter"),
            html.Div([html.Span([html.I(style={"background": f"var(--{v})"}), k]) for k, v in
                      [("good", "good"), ("bad", "bad"), ("checked", "chk"), ("unsure", "uns")]], className="slegend"),
            dash_table.DataTable(
                id="tbl", data=[], page_action="none", columns=[{"name": "paper", "id": "card", "presentation": "markdown"}],
                markdown_options={"html": True}, style_table={"height": "calc(100vh - 170px)", "overflowY": "auto"},
                style_header={"display": "none"},
                style_cell={"fontSize": 12, "padding": "6px 10px", "textAlign": "left", "whiteSpace": "normal",
                            "fontFamily": "inherit", "border": "none", "borderBottom": "1px solid var(--line)", "cursor": "pointer"},
                style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": "var(--accent-soft)",
                                         "border": "none", "borderLeft": "3px solid var(--accent)"},
                                        {"if": {"state": "selected"}, "backgroundColor": "var(--accent-soft)", "border": "none"}],
                css=[{"selector": ".dash-spreadsheet-menu", "rule": "display:none"},
                     {"selector": ".dash-cell-value p", "rule": "margin:0"},
                     {"selector": ".pt", "rule": "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.35;color:var(--ink)"},
                     {"selector": ".pm", "rule": "color:var(--muted);font-size:11px;margin-top:2px"},
                     {"selector": ".pm b", "rule": "color:var(--ink-2)"},
                     {"selector": ".rvs", "rule": "display:flex;flex-wrap:wrap;gap:2px 10px;margin-top:3px;font-size:11px;align-items:center"},
                     {"selector": ".rv", "rule": "display:inline-flex;align-items:center;gap:3px"},
                     {"selector": ".rn", "rule": "font-weight:650;margin-right:2px"},
                     {"selector": ".cm", "rule": "color:var(--muted)"},
                     {"selector": ".d", "rule": "display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--muted)"},
                     {"selector": ".d.good", "rule": "background:var(--good)"},
                     {"selector": ".d.bad", "rule": "background:var(--bad)"},
                     {"selector": ".d.checked", "rule": "background:var(--chk)"},
                     {"selector": ".d.unsure", "rule": "background:var(--uns)"}]),
        ], className="plist"),
        html.Div([
            html.Div([html.Button("◀", id="prev", className="btn"), html.Button("▶", id="next", className="btn"),
                      html.Span(id="sel_label", className="sel"),
                      html.Button("Good ✓", id="mk_good", className="btn s-good"), html.Button("Bad ✕", id="mk_bad", className="btn s-bad"),
                      html.Button("Checked", id="mk_checked", className="btn s-checked"), html.Button("Unsure ?", id="mk_unsure", className="btn s-unsure"),
                      html.Span(id="save_msg", className="msg")], className="reviewbar"),
            html.Div(id="rec_head"), html.Div(id="rec_hero"),
            html.Div([
                html.Div(id="thread", className="thread"),
                html.Div([dcc.Dropdown(id="c_field", clearable=False, value="__row__", className="dd cfield"),
                          dcc.Input(id="comment", type="text", placeholder="add a comment… (Enter to send)", debounce=True, className="comment"),
                          html.Button("Send", id="add_c", className="btn primary")], className="crow"),
            ], className="cbox"),
            html.Div(id="rec_groups"),
        ], className="left", id="left"),
        html.Div(className="divider", title="drag to resize · double-click to reset"),
        html.Div(id="pdf", className="right"),
    ], className="panels", id="panels"),
    dcc.Store(id="cur"), dcc.Store(id="pdf_eid"), dcc.Store(id="order"),
], className="root", id="main", style={"display": "none"})

login = html.Div([html.Div([
    html.H2("PV dataset review"),
    html.Label("Who are you?"),
    html.Div([html.Button(u, id={"type": "ubtn", "name": u}, n_clicks=0, className="ubtn",
                          style={"borderColor": ucolor(u), "color": ucolor(u)}) for u in USERS], className="ubtns"),
    dcc.Input(id="login_name", type="text", placeholder="or type your name", className="txt"),
    html.Label("Password"),
    html.Div([dcc.Input(id="login_pw", type="password", placeholder="password", debounce=True, className="txt"),
              html.Button("Show", id="pw_toggle", n_clicks=0, className="btn pwbtn", title="show / hide the password")],
             className="pwrow"),
    html.Div("Hint: which consortium are we all part of? (not case-sensitive)", className="hint small"),
    html.Button("Log in", id="login_btn", n_clicks=0, className="btn primary"),
    html.Div(id="login_msg", className="err"),
], className="logincard")], id="login", className="loginwrap")

app.layout = html.Div([dcc.Store(id="user", storage_type="session"), login, main])


# ---------------------------------------------------------------- login
@app.callback(Output("login_name", "value"), Input({"type": "ubtn", "name": ALL}, "n_clicks"), prevent_initial_call=True)
def pick_user(_):
    t = ctx.triggered_id
    return t["name"] if isinstance(t, dict) and ctx.triggered[0]["value"] else no_update


@app.callback(Output("user", "data"), Output("login_msg", "children"),
              Input("login_btn", "n_clicks"), Input("login_pw", "n_submit"), Input("logout", "n_clicks"),
              State("login_name", "value"), State("login_pw", "value"), prevent_initial_call=True)
def do_login(_b, _e, _o, typed, pw):
    if ctx.triggered_id == "logout":
        return None, ""
    name = cap_name((typed or "").strip()) if (typed or "").strip() else ""
    if not name:
        return no_update, "pick or type a name"
    if (pw or "").strip().lower() != LOGIN_PASSWORD.lower():
        return no_update, "wrong password"
    return name, ""


@app.callback(Output("login_pw", "type"), Output("pw_toggle", "children"), Input("pw_toggle", "n_clicks"))
def pw_toggle(n):
    return ("text", "Hide") if (n or 0) % 2 else ("password", "Show")


@app.callback(Output("login", "style"), Output("main", "style"), Output("whoami", "children"), Input("user", "data"))
def gate(user):
    if user:
        return {"display": "none"}, {"display": "flex"}, pill(user)
    return {"display": "flex"}, {"display": "none"}, ""


# ---------------------------------------------------------------- filters -> paper table
FILTER_IDS = [i for i, _, _ in FILTERS] + ["f_status", "f_text"]
SLIDER_IDS = [rid for rid, *_ in RANGES]


@app.callback(Output("f_srcname", "options"), Output("f_srcname", "value", allow_duplicate=True),
              Input("f_type", "value"), State("f_srcname", "value"), prevent_initial_call=True)
def cascade_source(types, chosen):
    o = source_options(types)
    ok = {x["value"] for x in o}
    keep = [v for v in (chosen or []) if v in ok] or None
    return o, keep


@app.callback([Output(i, "value") for i in FILTER_IDS + SLIDER_IDS], Input("reset", "n_clicks"), prevent_initial_call=True)
def reset(_):
    return [None] * len(FILTER_IDS) + [list(RANGE_B[r]) for r in SLIDER_IDS]


def _sync_range(rid):
    lo0, hi0 = RANGE_B[rid]

    @app.callback(Output(rid, "value", allow_duplicate=True), Output(rid + "_lo", "value"), Output(rid + "_hi", "value"),
                  Input(rid, "value"), Input(rid + "_lo", "value"), Input(rid + "_hi", "value"), prevent_initial_call=True)
    def _sync(val, lo, hi):
        if ctx.triggered_id == rid:
            return no_update, val[0], val[1]
        lo = lo0 if lo is None else max(lo0, min(float(lo), hi0))
        hi = hi0 if hi is None else max(lo0, min(float(hi), hi0))
        lo, hi = min(lo, hi), max(lo, hi)
        return [lo, hi], lo, hi


for _r in SLIDER_IDS:
    _sync_range(_r)


def _match(col, vals):
    real = [v for v in vals if v != NA]
    if col in LIST_FILTER:
        lst = _list_col(col)
        m = lst.map(lambda l: any(str(x) in real for x in l))
        if NA in vals:
            m |= lst.map(lambda l: not l)
        return m
    s = df[col].map(plain)
    m = s.astype(str).isin(real)
    if NA in vals:
        m |= s.map(isnil)
    return m


@app.callback(Output("tbl", "data"), Output("counter", "children"), Output("order", "data"),
              Output("tbl", "active_cell"), Output("tbl", "selected_cells"),
              Input("apply", "n_clicks"), Input("f_text", "value"), Input("save_msg", "children"),
              [Input(r, "value") for r in SLIDER_IDS],
              [State(i, "value") for i in FILTER_IDS if i != "f_text"])
def apply_filters(_n, text, _msg, *vals):
    rv = dict(zip(SLIDER_IDS, vals[:len(SLIDER_IDS)]))
    fv = dict(zip([i for i in FILTER_IDS if i != "f_text"], vals[len(SLIDER_IDS):]))
    m = pd.Series(True, index=df.index)
    for i, col, _ in FILTERS:
        if fv.get(i) and col in df.columns:
            m &= _match(col, fv[i])
    for rid, col, _, _ in RANGES:
        lo0, hi0 = RANGE_B[rid]
        lo, hi = rv.get(rid) or (lo0, hi0)
        if lo > lo0:                     # handle moved off its end -> real bound (rows without a value drop out)
            m &= df[col] >= lo
        if hi < hi0:
            m &= df[col] <= hi
    lg = read_log()
    st = row_status(lg)
    rstat = df["rk"].map(st["status"]) .fillna("") if len(st) else pd.Series("", index=df.index)
    if fv.get("f_status"):
        m &= rstat.isin(fv["f_status"])
    eids = pd.unique(df.loc[m, "eid"])
    P = PAPERS.loc[[e for e in eids]] if len(eids) else PAPERS.iloc[0:0]
    ncm = lg[lg.comment != ""].groupby("eid").size()
    if text:
        t = text.lower()
        P = P[P[["title", "author", "eid", "doi", "year"]].astype(str).apply(lambda s: s.str.lower().str.contains(t, regex=False)).any(axis=1)]
    P = P.sort_values(["year", "title"], ascending=[False, True])
    nrev = rstat[rstat != ""].groupby(df["eid"]).size()
    tab = P[["eid", "year", "author", "title", "n", "rate", "pdf"]].copy()
    # per paper, per reviewer: that reviewer's latest status on each data point -> one dot each
    sv = lg[(lg.status != "") & lg.rk.notna()].copy()
    dots = {}
    if len(sv):
        sv["who"] = sv.reviewer.map(cap_name)
        last = sv.groupby(["eid", "who", "rk"]).last().reset_index().sort_values(["eid", "who", "rk"])
        for (e, who), g in last.groupby(["eid", "who"]):
            ds = "".join(f'<i class="d {st}" title="{st}"></i>' for st in g.status.tolist()[:10]) + (f"+{len(g) - 10}" if len(g) > 10 else "")
            dots.setdefault(e, []).append(f'<span class="rv"><span class="rn" style="color:{ucolor(who)}">{who}</span>{ds}</span>')
    tab["review"] = ["".join(dots.get(e, [])) + (f'<span class="cm">💬{int(ncm[e])}</span>' if e in ncm.index else "")
                     for e in tab.eid]
    tab["year"] = tab["year"].map(lambda y: "" if y is None or (isinstance(y, float) and math.isnan(y)) else int(y))
    esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tab["card"] = [f'<div class="pt">{esc(r.title)}</div><div class="pm"><b>{esc(r.author or "?")}</b> · {r.year}'
                   + (f" · {r.n} data points" if r.n > 1 else "") + (f" · {r.rate} %/yr" if r.rate else "")
                   + (" · pdf" if r.pdf else "") + "</div>"
                   + (f'<div class="rvs">{r.review}</div>' if r.review else "")
                   for r in tab.itertuples()]
    tips = [{} for _ in range(len(tab))]
    counter = [html.B(f"{len(tab)} papers"), f" · {int(PAPERS.loc[tab.eid, 'n'].sum()) if len(tab) else 0} data points · "
               f"{int((rstat != '').sum())} reviewed overall · {int((lg.comment != '').sum())} comments"]
    keep = ctx.triggered_id == "save_msg"
    return (tab[["eid", "card"]].to_dict("records"), counter, tab.eid.tolist(),
            no_update if keep else None, no_update if keep else [])


# ---------------------------------------------------------------- selection
@app.callback(Output("cur", "data"),
              Input("tbl", "active_cell"), Input("prev", "n_clicks"), Input("next", "n_clicks"),
              Input({"type": "dp", "rk": ALL}, "n_clicks"),
              State("tbl", "derived_virtual_data"), State("cur", "data"), prevent_initial_call=True)
def pick(active, _p, _n, _dp, vdata, cur):
    t = ctx.triggered_id
    if isinstance(t, dict):
        if not ctx.triggered[0]["value"]:
            return no_update
        return {"eid": str(df.at[t["rk"], "eid"]), "rk": t["rk"]}
    if not vdata:
        return no_update
    eids = [r["eid"] for r in vdata]
    if t == "tbl":
        if not active:
            return no_update
        e = eids[active["row"]]
    else:
        i = eids.index(cur["eid"]) if cur and cur.get("eid") in eids else -1
        i = i + (1 if t == "next" else -1)
        e = eids[max(0, min(len(eids) - 1, i))]
    if cur and cur.get("eid") == e:
        return no_update
    return {"eid": e, "rk": int(df.index[df.eid == e][0])}


@app.callback(Output("rec_head", "children"), Output("rec_hero", "children"), Output("rec_groups", "children"),
              Output("sel_label", "children"), Output("thread", "children"), Output("c_field", "options"), Output("c_field", "value"),
              Input("cur", "data"), Input("save_msg", "children"), State("c_field", "value"))
def show(cur, _msg, cfield):
    if not cur:
        return html.Div("Pick a paper on the left.", className="hint"), "", "", "", "", [], "__row__"
    eid, rk = cur["eid"], cur["rk"]
    lg = read_log()
    head, hero, groups = render_record(eid, rk, lg)
    st = row_status(lg)
    s = st.loc[rk] if len(st) and rk in st.index else None
    label = [html.Span(eid, className="eidlbl")] + ([" · ", html.B(s.status, className=f"stxt {s.status}"), " ", pill(s.reviewer)] if s is not None else [html.Span(" · not reviewed", className="nil")])
    fopts = [{"label": "on this data point", "value": "__row__"}, {"label": "on the whole paper", "value": "__paper__"}] + \
            [{"label": f"field: {lbl}", "value": f} for f, lbl in MAIN if f in df.columns] + \
            [{"label": f"field: {pretty(f)}", "value": f} for _, fs in GROUPS for f in fs if f in df.columns]
    keep = cfield if ctx.triggered_id == "save_msg" and cfield else "__row__"
    return head, hero, groups, label, render_thread(eid, rk, lg), fopts, keep


@app.callback(Output("pdf", "children"), Output("pdf_eid", "data"), Input("cur", "data"), State("pdf_eid", "data"))
def show_pdf(cur, shown):
    if not cur:
        return html.Div("The PDF appears here.", className="hint"), None
    eid = cur["eid"]
    if eid == shown:
        return no_update, no_update
    fname = None if PUBLIC_MODE else pdf_map.get(eid)
    if fname:
        return html.Iframe(src=f"{BASE}pdf/{quote(fname)}#navpanes=0&view=FitH", className="pdfframe"), eid
    doi = PAPERS.at[eid, "doi"]
    return html.Div([
        html.Div([html.A("Open at publisher ↗", href=f"https://doi.org/{doi}", target="_blank", className="btn primary") if doi else html.Span("no DOI"),
                  html.Span(" · or click “📁 Choose local PDF folder…” (top bar) to pick the folder with your PDFs; the paper then shows here", className="hint")],
                 id="localmsg", className="pubbar"),
        html.Iframe(id="localpdf", className="pdfframe")], className="pubwrap"), eid


# ---------------------------------------------------------------- mark / comment / export
@app.callback(Output("save_msg", "children"), Output("comment", "value"),
              Input("mk_good", "n_clicks"), Input("mk_bad", "n_clicks"), Input("mk_checked", "n_clicks"), Input("mk_unsure", "n_clicks"),
              Input("add_c", "n_clicks"), Input("comment", "n_submit"),
              State("cur", "data"), State("comment", "value"), State("c_field", "value"), State("user", "data"), prevent_initial_call=True)
def mark(_g, _b, _c, _u, _a, _s, cur, comment, cfield, user):
    if not cur:
        return "pick a paper first", no_update
    t, now = ctx.triggered_id, f"{dt.datetime.now():%H:%M:%S}"
    if t in ("add_c", "comment"):
        if not (comment or "").strip():
            return no_update, no_update
        rk = None if cfield == "__paper__" else cur["rk"]
        field = "" if cfield in ("__row__", "__paper__", None) else cfield
        log_event(cur["eid"], rk, field=field, comment=comment.strip(), reviewer=user or "")
        return f"comment saved · {now}", ""
    status = t.replace("mk_", "")
    log_event(cur["eid"], cur["rk"], status=status, reviewer=user or "")
    return f"marked {status} · {now}", no_update


# ---------------------------------------------------------------- review log panel
@app.callback(Output("logtbl", "data"), Output("log_count", "children"), Output("lf_who", "options"),
              Output("logtbl", "selected_rows"),
              Input("save_msg", "children"), Input("lf_who", "value"), Input("lf_kind", "value"), Input("lf_text", "value"),
              Input("user", "data"))
def log_table(_msg, who, kind, text, _u):
    lg = read_log()
    rows = log_rows(lg, who, kind, text)
    names = sorted({cap_name(u) for u in lg.reviewer if u})
    return rows, f"{len(rows)} of {len(lg)}" if (who or text or kind not in (None, "all")) else f"{len(lg)}", \
        [{"label": n, "value": n} for n in names], []


@app.callback(Output("logtbl", "selected_rows", allow_duplicate=True),
              Input("log_all", "n_clicks"), Input("log_none", "n_clicks"), State("logtbl", "data"), prevent_initial_call=True)
def log_select(_a, _n, data):
    return list(range(len(data or []))) if ctx.triggered_id == "log_all" else []


@app.callback(Output("save_msg", "children", allow_duplicate=True), Output("log_msg", "children"),
              Input("log_del", "submit_n_clicks"), State("logtbl", "selected_rows"), State("logtbl", "data"),
              State("user", "data"), prevent_initial_call=True)
def log_delete(_n, sel, data, user):
    ids = [data[i]["id"] for i in (sel or []) if data and i < len(data)]
    if not ids:
        return no_update, "nothing selected"
    n = delete_entries(ids, user)
    now = f"{dt.datetime.now():%H:%M:%S}"
    return f"deleted {n} entr{'y' if n == 1 else 'ies'} · {now}", f"deleted {n}"


@app.callback(Output("save_msg", "children", allow_duplicate=True),
              Input({"type": "cdel", "id": ALL}, "n_clicks"), State("user", "data"), prevent_initial_call=True)
def thread_delete(clicks, user):
    t = ctx.triggered_id
    if not isinstance(t, dict) or not ctx.triggered[0]["value"]:
        return no_update                                  # buttons re-rendered, not clicked
    delete_entries([t["id"]], user)
    return f"deleted 1 entry · {dt.datetime.now():%H:%M:%S}"


@app.callback(Output("cur", "data", allow_duplicate=True), Input("logtbl", "active_cell"), State("logtbl", "data"),
              prevent_initial_call=True)
def log_open(active, data):
    if not active or not data or active["row"] >= len(data):
        return no_update
    r = data[active["row"]]
    e = r.get("eid") or r["paper"]
    if e not in set(df.eid):
        return no_update
    rk = r["rk"] if r["rk"] is not None and r["rk"] in df.index else int(df.index[df.eid == e][0])
    return {"eid": e, "rk": rk}


@app.callback(Output("dl", "data"), Input("export", "n_clicks"), prevent_initial_call=True)
def export(_):
    lg = read_log().drop(columns=["id"])
    lg = lg.merge(PAPERS[["eid", "title", "doi"]].reset_index(drop=True), on="eid", how="left")
    return dcc.send_data_frame(lg.to_csv, f"review_log_{dt.date.today()}.csv", index=False)


# ---------------------------------------------------------------- page shell: style + client JS
app.index_string = r"""<!DOCTYPE html><html><head>{%metas%}<meta name="robots" content="noindex,nofollow"><title>{%title%}</title>{%favicon%}{%css%}
<style>
:root{--bg:#f7f7f5;--surface:#fff;--surface-2:#f2f2ef;--surface-3:#e7e7e3;--ink:#1a1a18;--ink-2:#55544f;--muted:#8d8b84;
--line:#e3e2dc;--accent:#2f6fb8;--accent-soft:#eaf1f9;--good:#3f8a5a;--bad:#b0483f;--chk:#3b6ea8;--uns:#a07c2c;--shadow:rgba(20,20,15,.14)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.root{min-height:100vh;box-sizing:border-box;flex-direction:column;padding:8px 12px;gap:6px}
.titlebar{display:flex;align-items:center;gap:10px}.h3{margin:0 auto 0 0;font-size:15px;font-weight:650}
.who .pill{font-size:12px;padding:2px 10px}
.btn{padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--surface);cursor:pointer;font-size:12px;white-space:nowrap;color:var(--ink)}
.btn:hover{background:var(--surface-2)} .btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.s-good{color:var(--good)} .btn.s-bad{color:var(--bad)} .btn.s-checked{color:var(--chk)} .btn.s-unsure{color:var(--uns)}
.fwrap{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:4px 10px}
.fwrap>summary{cursor:pointer;color:var(--ink-2);font-weight:600;font-size:12px;padding:2px 0}
.filters{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;padding:6px 0 4px}
.dd{min-width:140px;font-size:12px} .dd.wide{min-width:320px}
.slegend{display:flex;gap:10px;padding:4px 10px;font-size:11px;color:var(--muted);border-bottom:1px solid var(--line)}
.slegend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:middle} .dd .Select-control{min-height:30px;height:30px;border-color:var(--line)}
.rng{font-size:11px;color:var(--ink-2);display:flex;align-items:center;gap:6px}
.ranges{gap:6px 22px;padding-top:0} .rng label{min-width:78px;text-align:right;font-size:12px;color:var(--ink-2)}
.fbtns{display:flex;gap:8px;padding:6px 0 4px;border-top:1px solid var(--line);margin-top:4px}
#tip .xtab{border-collapse:collapse;width:100%} #tip .xtab td{padding:2px 0;vertical-align:top;border-bottom:1px solid var(--surface-2)}
#tip .xk{color:var(--muted);font-size:11.5px;width:40%;padding-right:8px !important} #tip .xv{font-size:12.5px} .slider{width:230px;padding-top:14px}
.slider .rc-slider-track{background:var(--accent)} .slider .rc-slider-handle{border-color:var(--accent)}
.slider .rc-slider-mark-text{font-size:10px;color:var(--muted)} .numin{width:62px;padding:3px 5px;border:1px solid var(--line);border-radius:5px}
.txt{width:230px;padding:5px 8px;border:1px solid var(--line);border-radius:6px}
.panels{flex:none;height:calc(100vh - 24px);min-height:520px;display:grid;grid-template-columns:auto minmax(0,1fr) 7px minmax(320px,var(--rightw,42%));gap:0 8px}
.plist{width:300px;display:flex;flex-direction:column;min-height:0;background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.plist .counter{padding:6px 10px;font-size:12px;color:var(--ink-2);border-bottom:1px solid var(--line)}
.plist .dash-table-container{flex:1;min-height:0}
.left{overflow-y:auto;min-width:0;padding:0 2px 60px}
.right{background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.divider{cursor:col-resize;background:var(--surface-3);border-radius:4px} .divider:hover,.panels.dragging .divider{background:var(--accent)}
.panels.dragging .right{pointer-events:none}
.pdfframe{width:100%;height:100%;border:0} .pubwrap{height:100%;display:flex;flex-direction:column} .pubbar{padding:8px 10px;border-bottom:1px solid var(--line)} .pubbar.found .hint{display:none}
.hint{color:var(--muted);padding:8px 12px;font-size:12px}
.reviewbar{position:sticky;top:0;z-index:3;background:var(--bg);display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding:2px 0 8px}
.sel{margin-right:auto;font-size:12px}.eidlbl{color:var(--muted)} .msg{color:var(--good);font-size:12px}
.stxt.good{color:var(--good)}.stxt.bad{color:var(--bad)}.stxt.checked{color:var(--chk)}.stxt.unsure{color:var(--uns)}
.pill{display:inline-block;color:#fff;border-radius:9px;padding:0 7px;font-size:10px;font-weight:600;line-height:16px;vertical-align:middle}
.ph h2{margin:0 0 3px;font-size:16px;font-weight:650;line-height:1.35} .ph .meta{color:var(--muted);font-size:12px;display:flex;gap:4px 14px;flex-wrap:wrap}
.sum{margin-top:6px;font-size:12.5px;color:var(--ink-2)} .sum summary{cursor:pointer;color:var(--muted);font-size:12px}
.sumbody{margin-top:5px;padding:8px 11px;background:var(--surface-2);border-radius:8px;white-space:pre-wrap;max-height:12em;overflow:auto}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 2px}
.dphead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:12px 0 2px;padding:6px 10px;background:var(--accent-soft);border-radius:7px}
.dpn{font-weight:650;color:var(--accent)} .dpev{flex-basis:100%;font-size:12px;color:var(--ink-2);border-left:3px solid var(--accent);padding:2px 0 2px 9px;margin-top:2px}
.dpevk{font-weight:650;color:var(--ink-2)} .dpevq{white-space:pre-wrap} .dpcur{color:var(--ink-2);font-size:12px} .tab .tn{color:var(--muted);font-weight:600}
.tab{flex:none;padding:3px 10px;border:1px solid var(--line);border-radius:999px;background:var(--surface);font-size:12px;cursor:pointer;max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink)}
.tab.active{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)} .tab .n{color:var(--ink-2);font-weight:600}
.sdot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;border:1px solid var(--muted);vertical-align:middle}
.sdot.good{background:var(--good);border-color:var(--good)}.sdot.bad{background:var(--bad);border-color:var(--bad)}.sdot.checked{background:var(--chk);border-color:var(--chk)}.sdot.unsure{background:var(--uns);border-color:var(--uns)}
.hero{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:7px;margin:12px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:8px 11px;min-height:64px;cursor:default}
.card .k,.f .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);white-space:normal;overflow-wrap:anywhere;line-height:1.3}
.f .k{text-transform:none;letter-spacing:0;font-size:11px}
.card .v{margin-top:3px;font-size:14px;font-weight:600;line-height:1.3;word-break:break-word;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.card.rate{border-color:var(--accent)} .card.rate .v{color:var(--accent)} .num.big{font-size:24px;font-weight:700}
.card:hover,.f:hover{background:var(--surface-2)} .card.pinned,.f.pinned{outline:2px solid var(--accent);outline-offset:-2px}
.empty .v{color:var(--muted);font-weight:500}
.cbadge{margin-left:6px;font-size:10px;color:var(--accent);text-transform:none}
.grp{background:var(--surface);border:1px solid var(--line);border-radius:9px;margin:0 0 8px;overflow:hidden}
.grp>summary{cursor:pointer;padding:7px 12px;font-size:12px;font-weight:650;letter-spacing:.03em;text-transform:uppercase;color:var(--ink-2);background:var(--surface-2);display:flex;gap:8px}
.grp[open]>summary{border-bottom:1px solid var(--line)} .grp .cnt{margin-left:auto;font-weight:500;text-transform:none;color:var(--muted)}
.fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}
.f{padding:6px 11px 7px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);min-height:50px;cursor:default}
.f .v{font-size:12.5px;margin-top:2px;word-break:break-word;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.chip{display:inline-flex;gap:4px;align-items:center;padding:0 7px;border-radius:5px;font-size:12px;font-weight:500;line-height:20px;margin:1px 4px 1px 0;white-space:nowrap;background:var(--surface-3)}
.card .chip{font-weight:600} .chip .gl{color:var(--muted)} .chip.mute{background:transparent;color:var(--muted);border:1px dashed var(--line)}
.unit{color:var(--muted);font-size:.78em;font-weight:500;margin-left:3px} .nil{color:var(--muted)} .small{font-size:.8em;font-weight:500}
.bool{font-weight:600}.bool.n{color:var(--muted);font-weight:500} .raw{color:var(--ink-2);font-weight:400}
.meter{display:inline-flex;gap:2px;margin-left:7px;vertical-align:middle}.meter i{width:5px;height:10px;border-radius:2px;background:var(--surface-3)}.meter i.on{background:var(--ink-2)}
.tipsrc{display:none}
#tip{position:fixed;z-index:99;width:min(460px,calc(100vw - 24px));background:var(--surface);border:1px solid var(--line);border-radius:10px;box-shadow:0 10px 32px var(--shadow);padding:11px 13px;font-size:12.5px;display:none;pointer-events:none;max-height:80vh;overflow:auto}
#tip.pinned{pointer-events:auto;border-color:var(--accent)}
#tip .tt{font-weight:650;font-size:13.5px} #tip .tf{font:11px ui-monospace,monospace;color:var(--muted)} #tip .tdesc{color:var(--muted);font-size:11.5px;margin:3px 0 2px}
#tip .tsec{margin-top:8px} #tip .th{font-size:11px;font-weight:700;color:var(--ink-2);text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px} #tip .th.inline{display:inline}
#tip .tclose{position:sticky;top:0;float:right;margin:-4px -4px 0 8px;width:24px;height:24px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink-2);font-size:16px;line-height:20px;cursor:pointer;padding:0}
#tip .tclose:hover{background:var(--surface-2);color:var(--ink)}
#tip .warn{color:#9a6a12;background:#fbf3e0;border-radius:5px;padding:3px 7px;font-size:11.5px;margin:2px 0 4px}
#tip .copybtn{float:right;font:600 10.5px system-ui;text-transform:none;letter-spacing:0;padding:1px 8px;border:1px solid var(--line);border-radius:5px;background:var(--surface);color:var(--accent);cursor:pointer}
#tip .copybtn:hover{background:var(--accent-soft)} #tip .copybtn.done{color:var(--good);border-color:var(--good)}
#tip{pointer-events:auto !important}
#tip .traw{background:var(--surface-2);border-radius:6px;padding:5px 8px;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px}
#tip .tquote{border-left:3px solid var(--accent);padding:2px 0 2px 9px;color:var(--ink-2);white-space:pre-wrap}
#tip .treason{color:var(--ink-2);white-space:pre-wrap}
#tip .titem{padding:5px 0;border-bottom:1px solid var(--surface-2)} #tip .titem:last-child{border-bottom:0}
#tip .titem .tquote{margin-top:3px;font-size:12px}
#tip .hintp{margin-top:8px;color:var(--muted);font-size:11px}
.cbox{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:8px 10px;margin:0 0 12px;display:flex;flex-direction:column;gap:6px}
.thread{max-height:170px;overflow-y:auto;display:flex;flex-direction:column;gap:4px}
.cmsg{display:flex;align-items:baseline;gap:6px;padding:3px 7px;background:var(--surface-2);border-radius:6px}
.cdel{margin-left:auto;border:none;background:transparent;color:var(--muted);cursor:pointer;font-size:14px;line-height:1;padding:0 3px}
.cdel:hover{color:var(--bad)}
.logpanel{margin:6px 0} .logbar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
.logbar .dd{min-width:170px} .hint.small{font-size:11px}
.ctag{font-size:10.5px;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 5px;white-space:nowrap}
.ctext{flex:1;white-space:pre-wrap} .cts{color:var(--muted);font-size:10px;white-space:nowrap}
.crow{display:flex;gap:6px;align-items:center} .cfield{min-width:210px} .comment{flex:1;padding:5px 8px;border:1px solid var(--line);border-radius:6px}
.pwrow{display:flex;gap:6px;align-items:stretch} .pwrow .txt{flex:1;min-width:0} .pwbtn{flex:none}
.loginwrap{justify-content:center;align-items:center;height:100vh}
.logincard{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:26px 30px;width:360px;display:flex;flex-direction:column;gap:8px;box-shadow:0 4px 16px rgba(0,0,0,.06)}
.logincard h2{margin:0 0 8px}.logincard label{font-size:12px;color:var(--ink-2);margin-top:6px}.logincard .txt{width:100%}.err{color:var(--bad);font-size:12px;min-height:16px}
.ubtns{display:flex;flex-wrap:wrap;gap:6px}.ubtn{padding:5px 13px;border:2px solid;border-radius:16px;background:var(--surface);font-weight:600;cursor:pointer}
#pdfpick{font-size:12px} #pdfcount{color:var(--good);margin-left:4px}
.plist .dash-spreadsheet-container .dash-header{display:none}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
<div id="tip"></div>
<div id="pdfpick"><label class="btn" title="Click to choose a folder on your computer that holds the paper PDFs. Each paper is matched to a PDF whose file name contains its EID; files stay on your computer (nothing is uploaded). Only needed when the app cannot show the PDF itself.">📁 Choose local PDF folder… <input type="file" id="pdfdir" webkitdirectory multiple style="display:none"></label><span id="pdfcount"></span></div>
<script>
(function(){ // hover cards: show the field's hidden .tipsrc in one floating card next to the field;
             // the card stays while the mouse is on it (so Copy can be clicked); click a field to pin it
  var tip=document.getElementById('tip'), pinned=null, cur=null, hideT=null;
  function place(el){ var r=el.getBoundingClientRect(), w=tip.offsetWidth, h=tip.offsetHeight;
    var x=r.right+8; if(x+w>innerWidth-10) x=Math.max(10,r.left-w-8);
    var y=Math.min(Math.max(10,r.top), innerHeight-h-10); tip.style.left=x+'px'; tip.style.top=Math.max(10,y)+'px'; }
  function show(el){ var s=el.querySelector('.tipsrc'); if(!s)return; cur=el;
    tip.innerHTML=s.innerHTML+'<div class="hintp">'+(pinned?'pinned · click the field again or Esc to release':'click the field to pin')+'</div>';
    tip.style.display='block'; place(el); }
  function hide(){ if(pinned)return; tip.style.display='none'; cur=null; }
  function later(){ clearTimeout(hideT); hideT=setTimeout(hide,220); }
  function unpin(){ if(pinned){pinned.classList.remove('pinned');pinned=null;} tip.classList.remove('pinned'); hide(); }
  document.addEventListener('mouseover',function(e){ if(!e.target.closest)return;
    if(e.target.closest('#tip')){ clearTimeout(hideT); return; }
    var el=e.target.closest('[data-field]');
    if(el){ clearTimeout(hideT); if(!pinned && el!==cur) show(el); } else later(); });
  document.addEventListener('click',function(e){ if(!e.target.closest)return;
    if(e.target.closest('#tip .tclose')){ if(pinned){pinned.classList.remove('pinned');pinned=null;} tip.classList.remove('pinned'); tip.style.display='none'; cur=null; return; }
    var cb=e.target.closest('#tip .copybtn');
    if(cb){ var sec=cb.closest('.tsec'), box=sec&&sec.querySelector('.'+cb.dataset.copy);
      var txt=box?(box.dataset.text||box.textContent):''; copyText(txt); cb.textContent='Copied ✓'; cb.classList.add('done');
      setTimeout(function(){cb.textContent='Copy';cb.classList.remove('done');},1400); return; }
    if(e.target.closest('#tip')) return;
    var el=e.target.closest('[data-field]');
    if(el){ if(pinned===el){unpin();return;} unpin(); pinned=el; el.classList.add('pinned'); show(el); tip.classList.add('pinned'); return; }
    if(pinned) unpin(); });
  function copyText(t){ if(navigator.clipboard&&window.isSecureContext){ navigator.clipboard.writeText(t).catch(fallback); } else fallback();
    function fallback(){ var ta=document.createElement('textarea'); ta.value=t; ta.style.position='fixed'; ta.style.opacity='0';
      document.body.appendChild(ta); ta.select(); try{document.execCommand('copy');}catch(_){} document.body.removeChild(ta); } }
  document.addEventListener('keydown',function(e){ if(e.key==='Escape')unpin(); });
  new MutationObserver(function(){ if(cur&&!document.body.contains(cur)){ pinned=null; tip.classList.remove('pinned'); tip.style.display='none'; cur=null; } })
    .observe(document.body,{subtree:true,childList:true});
})();
(function(){ // local PDF folder: match by eid in the file name, nothing is uploaded
  var files={}, names=[], lastEid=null;
  document.getElementById('pdfdir').addEventListener('change',function(e){
    for(var i=0;i<e.target.files.length;i++){var f=e.target.files[i]; if(/\.pdf$/i.test(f.name)){files[f.name.toLowerCase()]=f;}}
    names=Object.keys(files); document.getElementById('pdfcount').textContent=names.length+' PDFs loaded'; lastEid=null; show(); });
  function show(){ var lbl=document.querySelector('#sel_label .eidlbl'), fr=document.getElementById('localpdf'); if(!lbl||!fr||!names.length)return;
    var eid=lbl.textContent.trim().toLowerCase(); if(eid===lastEid&&fr.src)return;
    var n=names.find(function(x){return x.indexOf(eid)>=0;}), msg=document.getElementById('localmsg');
    if(n){ if(fr._url)URL.revokeObjectURL(fr._url); fr._url=URL.createObjectURL(files[n]); fr.src=fr._url+'#navpanes=0&view=FitH'; if(msg)msg.classList.add('found'); }
    else { fr.removeAttribute('src'); if(msg)msg.classList.remove('found'); } lastEid=eid; }
  new MutationObserver(function(){ var slot=document.getElementById('pickslot'),pk=document.getElementById('pdfpick');
    if(slot&&pk&&pk.parentElement!==slot)slot.appendChild(pk); show(); }).observe(document.body,{subtree:true,childList:true,characterData:true});
})();
(function(){ // new paper -> scroll the record panel back to the top
  var last=null; new MutationObserver(function(){ var l=document.querySelector('#sel_label .eidlbl'); if(!l)return;
    var e=l.textContent; if(e!==last){ last=e; var p=document.getElementById('left'); if(p)p.scrollTop=0; } }).observe(document.body,{subtree:true,childList:true,characterData:true});
})();
(function(){ // draggable divider between record and PDF
  var drag=false,panels=null,root=document.documentElement;
  try{var s=localStorage.getItem('rightw'); if(s)root.style.setProperty('--rightw',s);}catch(e){}
  document.addEventListener('mousedown',function(e){ if(e.target.classList&&e.target.classList.contains('divider')){drag=true;panels=e.target.parentElement;panels.classList.add('dragging');e.preventDefault();}});
  document.addEventListener('mousemove',function(e){ if(!drag)return; var r=panels.getBoundingClientRect();
    var w=Math.max(320,Math.min(r.width-700,r.right-e.clientX))+'px'; root.style.setProperty('--rightw',w); try{localStorage.setItem('rightw',w);}catch(_){} });
  document.addEventListener('mouseup',function(){ if(drag){drag=false;panels.classList.remove('dragging');} });
  document.addEventListener('dblclick',function(e){ if(e.target.classList&&e.target.classList.contains('divider')){root.style.removeProperty('--rightw');try{localStorage.removeItem('rightw');}catch(_){}} });
})();
</script></body></html>"""

def _free_port(host, start, tries=20):
    import socket
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            if sk.connect_ex((host, p)) != 0:
                return p
    raise SystemExit(f"no free port in {start}-{start + tries - 1}")


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", PORT)))
    ap.add_argument("--host", default=os.environ.get("HOST", HOST))
    a = ap.parse_args()
    port = _free_port(a.host, a.port)
    if port != a.port:
        print(f"port {a.port} is busy, using {port}")
    print(f"open http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{port}")
    app.run(debug=False, host=a.host, port=port)
