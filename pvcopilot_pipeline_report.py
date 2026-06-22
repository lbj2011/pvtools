"""
pvcopilot_pipeline_report.py
============================

End-to-end automation harness for the PVcopilot tool.

For every dataset in a folder (default ./test_datasets/), this drives the SAME
pipeline the website runs -- Analyze -> Apply Filters -> Calculate Degradation --
using the exact functions the app's callbacks call, and records, per dataset:

    GOOD     full pipeline ran and produced a finite degradation rate
    WARNING  ran, but with caveats (no irradiance, no temperature, YoY disabled)
    BLOCKED  tool correctly refused (under 1 year of data -- expected behavior)
    ERROR    the tool failed -- file unreadable, column mapping failed, a filter
             or the degradation method crashed, timed out, or returned no rate

Each run writes a timestamped, machine-readable report AND a human-readable PDF
into ./reports/ , plus a "latest" copy that always points at the most recent run:

    reports/pvcopilot_pipeline_report_<UTC-timestamp>.json   (full detail)
    reports/pvcopilot_pipeline_report_<UTC-timestamp>.csv    (one row per dataset)
    reports/pvcopilot_pipeline_report_<UTC-timestamp>.pdf    (visual summary)
    reports/pvcopilot_pipeline_latest.json / .csv / .pdf

Usage (always with the pvtools env):
    conda activate pvtools
    python pvcopilot_pipeline_report.py                  # scans ./test_datasets/
    python pvcopilot_pipeline_report.py path/to/folder   # scan a different folder
    python pvcopilot_pipeline_report.py --include-examples   # also test data/*.parquet
    python pvcopilot_pipeline_report.py --method LR      # force a degradation method

The classification mirrors the app's own gating:
  * < 1 year of data           -> BLOCKED (degradation not attempted)
  * 1 <= years < 2             -> YoY disabled, falls back to Linear Regression
  * >= 2 years                 -> YoY (the app's default method)
"""

import os
import sys
import csv
import json
import base64
import argparse
import textwrap
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

# --- make the project root importable, then load API keys like index.py does ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# All imports below are third-party / app modules that only exist in the
# `pvtools` conda env. If you launch this with the system Python (e.g.
# `/usr/bin/python3 ...`) they won't be found -- so turn the raw
# ModuleNotFoundError into a clear instruction instead.
try:
    from dotenv import load_dotenv

    import numpy as np
    import pandas as pd

    import matplotlib
    matplotlib.use("Agg")  # headless: write to file, no display
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # The SAME functions PVcopilot's callbacks use.
    from page_supporting_files.analysis_utils import (
        parse_contents, normalize, low_irra_power_filter, aggregate_daily,
        compute_yoy, compute_lr, compute_hw, compute_arima, compute_csd,
    )
    from page_supporting_files.pvcopilot_filter_functions import (
        basic_value_filter, clear_sky_filter, identify_outliers_iqr,
    )
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"\nERROR: missing dependency '{e.name}'.\n"
        f"This script must run inside the 'pvtools' conda env, not the system\n"
        f"Python ({sys.executable}).\n\n"
        f"Fix:\n"
        f"    conda activate pvtools\n"
        f"    python pvcopilot_pipeline_report.py\n\n"
        f"Or call the env's Python directly:\n"
        f"    /opt/anaconda3/envs/pvtools/bin/python pvcopilot_pipeline_report.py\n\n"
    )
    sys.exit(1)

load_dotenv(override=True)  # load API keys like index.py does

SUPPORTED = (".csv", ".xlsx", ".xls", ".parquet")
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "test_datasets")
EXAMPLE_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

# Mirror the app: cap each heavy stage so a pathological dataset can't hang the run.
STEP_TIMEOUT_S = 10
# Default filters match the app's `filter-options` default value.
DEFAULT_FILTERS = ["timezone", "low-irra-power", "outlier", "clearsky"]

STATUS_COLOR = {
    "GOOD":    "#16a34a",  # green
    "WARNING": "#d97706",  # amber
    "BLOCKED": "#2563eb",  # blue (expected refusal, not a failure)
    "ERROR":   "#dc2626",  # red
}
STATUS_ORDER = ["GOOD", "WARNING", "BLOCKED", "ERROR"]

_POOL = ThreadPoolExecutor(max_workers=4)


def _with_timeout(fn, *args, timeout=STEP_TIMEOUT_S, **kwargs):
    return _POOL.submit(fn, *args, **kwargs).result(timeout=timeout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_text(component):
    """Pull readable text out of a Dash component / string / list."""
    if component is None:
        return ""
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(_extract_text(c) for c in component).strip()
    children = getattr(component, "children", None)
    if children is not None:
        return _extract_text(children)
    return ""


def _build_contents(path):
    """Mirror the browser upload: a base64 data-URL string."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:application/octet-stream;base64,{b64}"


def _duration_years(df):
    """Span of the data in years, measured from the time index.

    Only trusts a genuine DatetimeIndex (which parse_contents sets from the
    identified Time column). Returns None when time isn't established -- e.g. no
    Time column was identified -- so we don't mistake a plain integer index for
    a near-zero (1970 epoch) span and block it for the wrong reason.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return None
    return (idx.max() - idx.min()).days / 365.25


def _run_filters(df, mapping):
    """Replicate the app's run_filter compute path with default parameters.
    Returns the filtered, normalized dataframe (rows kept)."""
    power_key = mapping.get("DC Power")
    irra_key = mapping.get("Irradiance")
    has_irr = bool(irra_key) and irra_key in df.columns

    bv_normal, _ = basic_value_filter(df, mapping)
    df = df.loc[bv_normal].copy()

    clearsky_mask = pd.Series(True, index=df.index)
    if "clearsky" in DEFAULT_FILTERS and has_irr:
        normal_idx, _ = clear_sky_filter(df, irra_key,
                                         smoothness_threshold=0.3,
                                         energy_threshold=0.5)
        clearsky_mask = df.index.isin(normal_idx)

    df_f = normalize(df, mapping, gamma=-0.004)
    current_mask = pd.Series(clearsky_mask, index=df_f.index)

    if "timezone" in DEFAULT_FILTERS:
        try:
            df_f.index = pd.to_datetime(df_f.index)
            df_f.index = df_f.index.tz_localize("UTC").tz_convert("US/Pacific")
        except Exception:
            pass  # already tz-aware or non-datetime; app tolerates this too

    if "low-irra-power" in DEFAULT_FILTERS and has_irr:
        normal_idx, _ = low_irra_power_filter(
            df_f, mapping, irr_thresh=300, power_ratio=0.02,
            norm_lower=0.01, norm_upper_pct=99)
        current_mask &= df_f.index.isin(normal_idx)

    if "outlier" in DEFAULT_FILTERS:
        normal_idx, _ = identify_outliers_iqr(df_f, "norm", iqr_multiplier=1.5)
        current_mask &= df_f.index.isin(normal_idx)

    return df_f.loc[df_f.index[current_mask]]


def _run_degradation(df_filtered, mapping, method):
    """aggregate_daily + the chosen statistical method. Returns rd (%/yr)."""
    irra_key = mapping.get("Irradiance")
    if not irra_key or irra_key not in df_filtered.columns:
        irra_key = None
    daily = aggregate_daily(df_filtered, irra_key)
    if method == "YOY":
        rd, _ = compute_yoy(daily)
    elif method == "LR":
        rd, _ = compute_lr(daily)
    elif method == "HW":
        rd, _ = compute_hw(daily, period=12)
    elif method == "ARIMA":
        rd, _ = compute_arima(daily, p=1, d=1, q=0, seasonal_period=12)
    elif method == "CSD":
        rd, _ = compute_csd(daily, period=12)
    else:
        raise ValueError(f"Unknown method: {method}")
    return rd


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------
def run_one(path, forced_method=None):
    """Drive Analyze -> Filter -> Degradation for one dataset.

    Returns a dict capturing the outcome, the stage reached, the reason, and
    useful metadata.
    """
    fname = os.path.basename(path)
    rec = {
        "file": fname, "status": "ERROR", "stage": "load", "reason": "",
        "rows_raw": None, "rows_filtered": None, "duration_years": None,
        "mapped": {}, "method": None, "rd_percent_per_year": None, "notes": [],
    }

    # ---- Stage 1: ANALYZE (load + LLM column identification) ----
    try:
        contents = _build_contents(path)
        df, summary_table, mapping, _code = _with_timeout(
            parse_contents, contents, fname)
    except FutureTimeout:
        rec.update(stage="analyze", reason=f"Analyze exceeded {STEP_TIMEOUT_S}s timeout.")
        return rec
    except Exception as e:
        rec.update(stage="analyze",
                   reason=f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=2))
        return rec

    if df is None:
        rec.update(stage="analyze",
                   reason=_extract_text(summary_table).strip() or "File could not be read/parsed.")
        return rec
    if not mapping:
        rec.update(stage="analyze",
                   reason=_extract_text(summary_table).strip() or "LLM column-mapping failed.")
        return rec

    rec["mapped"] = dict(mapping)
    rec["rows_raw"] = int(len(df))

    # ---- Stage 2 (FIRST GATE): DURATION -------------------------------------
    # Block purely on HOW LONG the data spans, before requiring any specific
    # column. Data under a year can never yield a meaningful degradation rate,
    # so "too short" takes precedence over "missing columns": a single-day file
    # reports BLOCKED (expected refusal), not ERROR. This needs only the time
    # index, which parse_contents sets from the identified Time column.
    dur = _duration_years(df)
    rec["duration_years"] = round(dur, 3) if dur is not None else None
    if dur is not None and dur < 1.0:
        months = int(round(dur * 12))
        rec.update(status="BLOCKED", stage="duration",
                   reason=f"Only ~{months} months (<1 year). Degradation correctly not attempted.")
        return rec

    # ---- Required columns (only reached by data that is long enough) --------
    missing_required = [v for v in ("DC Power", "Time") if v not in mapping]
    if missing_required:
        rec.update(status="ERROR", stage="analyze",
                   reason="Missing required column(s): " + ", ".join(missing_required)
                          + ". Need both Time and DC Power.")
        return rec

    # Data-quality notes (what the app shows in its toggle).
    if mapping.get("DC Power") == "computed_dc_power":
        rec["notes"].append("Power computed as V*I (no direct power channel).")
    has_irr = bool(mapping.get("Irradiance")) and mapping["Irradiance"] in df.columns
    has_temp = bool(mapping.get("Module temperature")) and mapping["Module temperature"] in df.columns
    if not has_irr:
        rec["notes"].append("No irradiance -> not weather-normalized (less reliable).")
    elif not has_temp:
        rec["notes"].append("No module temperature -> no temperature correction.")

    # Method follows the app's gating: YoY needs >= 2 years; else LR.
    if forced_method:
        method = forced_method
    elif dur is not None and dur < 2.0:
        method = "LR"
        rec["notes"].append("Under 2 years -> YoY disabled, using Linear Regression.")
    else:
        method = "YOY"
    rec["method"] = method

    # ---- Stage 3: FILTER ----
    try:
        df_filtered = _with_timeout(_run_filters, df, mapping)
    except FutureTimeout:
        rec.update(stage="filter", reason=f"Filtering exceeded {STEP_TIMEOUT_S}s timeout.")
        return rec
    except Exception as e:
        rec.update(stage="filter",
                   reason=f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=2))
        return rec

    rec["rows_filtered"] = int(len(df_filtered))
    if len(df_filtered) == 0:
        rec.update(status="ERROR", stage="filter",
                   reason="All rows removed by filtering -- nothing left to analyze.")
        return rec

    # ---- Stage 4: DEGRADATION ----
    try:
        rd = _with_timeout(_run_degradation, df_filtered, mapping, method)
    except FutureTimeout:
        rec.update(stage="degradation", reason=f"Degradation exceeded {STEP_TIMEOUT_S}s timeout.")
        return rec
    except Exception as e:
        rec.update(stage="degradation",
                   reason=f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=2))
        return rec

    if rd is None or not np.isfinite(rd):
        rec.update(status="ERROR", stage="degradation",
                   reason=f"{method} returned no finite degradation rate (got {rd}).")
        return rec

    rec["rd_percent_per_year"] = round(float(rd), 4)
    rec["stage"] = "done"
    rec["status"] = "WARNING" if rec["notes"] else "GOOD"
    rec["reason"] = (f"{method} degradation rate = {rd:.3f} %/yr"
                     + (" (with caveats)" if rec["notes"] else ""))
    return rec


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def _counts(results):
    c = {k: 0 for k in STATUS_ORDER}
    for r in results:
        c[r["status"]] = c.get(r["status"], 0) + 1
    return c


def write_json(results, counts, meta, path):
    with open(path, "w") as f:
        json.dump({"meta": meta, "summary": counts, "results": results}, f, indent=2)


def write_csv(results, path):
    cols = ["file", "status", "stage", "duration_years", "method",
            "rd_percent_per_year", "rows_raw", "rows_filtered", "reason"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            row["reason"] = " ".join(str(row.get("reason", "")).split())  # flatten newlines
            w.writerow(row)


def write_pdf(results, counts, meta, path):
    lines = []
    lines.append(("PVcopilot Pipeline Report", "title"))
    lines.append((f"Generated {meta['generated_utc']}  -  {meta['dataset_dir']}", "subtitle"))
    lines.append((f"{meta['n_files']} dataset(s)   "
                  + "   ".join(f"{k}: {counts[k]}" for k in STATUS_ORDER), "subtitle"))
    lines.append(("", "gap"))

    for r in sorted(results, key=lambda x: (STATUS_ORDER.index(x["status"]), x["file"])):
        lines.append((r["file"], "file"))
        lines.append((f"{r['status']}  (stage: {r['stage']})", "status:" + r["status"]))
        meta_bits = []
        if r["duration_years"] is not None:
            meta_bits.append(f"{r['duration_years']} yr")
        if r["method"]:
            meta_bits.append(f"method={r['method']}")
        if r["rd_percent_per_year"] is not None:
            meta_bits.append(f"rd={r['rd_percent_per_year']} %/yr")
        if r["rows_raw"] is not None:
            meta_bits.append(f"rows {r['rows_raw']}->{r['rows_filtered']}")
        if meta_bits:
            lines.append(("    " + "  |  ".join(meta_bits), "detail"))
        for para in str(r["reason"]).splitlines() or [""]:
            for wrapped in (textwrap.wrap(para, width=108) or [""]):
                lines.append(("    " + wrapped, "detail"))
        if r["notes"]:
            for note in r["notes"]:
                for wrapped in textwrap.wrap("- " + note, width=104):
                    lines.append(("      " + wrapped, "detail"))
        lines.append(("", "gap"))

    LINES_PER_PAGE = 50
    y_start, line_h = 0.96, 0.92 / LINES_PER_PAGE
    with PdfPages(path) as pdf:
        i = 0
        while i < len(lines):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            y = y_start
            for text, kind in lines[i:i + LINES_PER_PAGE]:
                if kind == "title":
                    ax.text(0.06, y, text, fontsize=18, fontweight="bold")
                elif kind == "subtitle":
                    ax.text(0.06, y, text, fontsize=10, color="#444444")
                elif kind == "file":
                    ax.text(0.06, y, text, fontsize=11, fontweight="bold", family="monospace")
                elif kind.startswith("status:"):
                    st = kind.split(":", 1)[1]
                    ax.text(0.06, y, text, fontsize=10, fontweight="bold",
                            color=STATUS_COLOR.get(st, "#000000"))
                elif kind == "detail":
                    ax.text(0.06, y, text, fontsize=9, color="#222222", family="monospace")
                y -= line_h
            pdf.savefig(fig); plt.close(fig)
            i += LINES_PER_PAGE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _print_legend():
    """Explain the four verdicts before the run begins."""
    print("=" * 78)
    print("PVcopilot pipeline -- result legend")
    print("=" * 78)
    rows = [
        ("GOOD",    "Full pipeline ran and produced a finite degradation rate (no caveats)."),
        ("WARNING", "Ran and produced a rate, but with caveats (no irradiance/temperature,"),
        ("",        "  or YoY fell back to LR). Tool worked; result is less reliable."),
        ("BLOCKED", "Tool correctly REFUSED -- data is under 1 year, so degradation isn't"),
        ("",        "  meaningful. This is expected behavior, NOT a failure."),
        ("ERROR",   "Tool genuinely FAILED -- unreadable file, no column mapping, a filter"),
        ("",        "  crashed, all rows filtered out, >10s timeout, or no valid rate."),
        ("",        "  Check each ERROR's 'reason' and 'stage' in the report."),
    ]
    for label, text in rows:
        print(f"  {label:<8}{text}")
    print("=" * 78 + "\n")


def collect_files(dataset_dir, include_examples):
    files = []
    if os.path.isdir(dataset_dir):
        files += [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir))
                  if f.lower().endswith(SUPPORTED)]
    if include_examples and os.path.isdir(EXAMPLE_DIR):
        files += [os.path.join(EXAMPLE_DIR, f) for f in sorted(os.listdir(EXAMPLE_DIR))
                  if f.lower().endswith(SUPPORTED)]
    return files


def main():
    ap = argparse.ArgumentParser(description="Run every dataset through the PVcopilot pipeline.")
    ap.add_argument("dataset_dir", nargs="?", default=DEFAULT_DATASET_DIR,
                    help="Folder of datasets to test (default: ./test_datasets/).")
    ap.add_argument("--include-examples", action="store_true",
                    help="Also test the bundled example datasets in ./data/.")
    ap.add_argument("--method", default=None,
                    choices=["YOY", "LR", "HW", "ARIMA", "CSD"],
                    help="Force a degradation method instead of the app's auto-gating.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only test the first N datasets (handy for a quick check).")
    args = ap.parse_args()

    files = collect_files(args.dataset_dir, args.include_examples)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No datasets found in {args.dataset_dir} "
              f"(supported: {', '.join(SUPPORTED)}).")
        return 1

    _print_legend()
    print(f"Running {len(files)} dataset(s) through the PVcopilot pipeline...\n")
    results = []
    for path in files:
        fname = os.path.basename(path)
        print(f"  - {fname} ...", flush=True, end=" ")
        rec = run_one(path, forced_method=args.method)
        print(rec["status"])
        results.append(rec)

    counts = _counts(results)
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": os.path.relpath(args.dataset_dir, PROJECT_ROOT),
        "n_files": len(files),
        "forced_method": args.method,
    }

    base = os.path.join(REPORT_DIR, f"pvcopilot_pipeline_report_{stamp}")
    latest = os.path.join(REPORT_DIR, "pvcopilot_pipeline_latest")
    for target in (base, latest):
        write_json(results, counts, meta, target + ".json")
        write_csv(results, target + ".csv")
        write_pdf(results, counts, meta, target + ".pdf")

    print("\nSummary: " + "   ".join(f"{k}: {counts[k]}" for k in STATUS_ORDER))
    print(f"Reports written to {os.path.relpath(REPORT_DIR, PROJECT_ROOT)}/ "
          f"(this run: ...{stamp}.json/.csv/.pdf, plus ...latest.*)")
    # Non-zero exit if anything genuinely failed -- handy for CI / scripting.
    return 2 if counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
