"""
prototype_energy_ratio.py
=========================

Side-by-side comparison of the CURRENT degradation pipeline vs a proposed
RdTools-style DAILY ENERGY-RATIO method, across the whole dataset library.

CURRENT (what the app/pipeline do today)
    point-wise norm = power / (irr * temp-correction)  per timestamp
    -> heavy point filters (clear-sky, irr>300, bounds, IQR)
    -> daily aggregate -> YoY (LR fallback) -> single rate

PROPOSED (prototyped here, nothing in the app is changed)
    sum each day FIRST:  ratio_day = sum(power) / sum(expected_power)
      expected = irr * (1 + gamma*(Tcell-25))   (per-point, summed per day)
    -> keep days with enough daylight coverage; light ratio sanity trim
    -> YoY on the daily ratio with a BOOTSTRAP CONFIDENCE INTERVAL
       (median of year-apart changes; CI = 5-95% of bootstrapped medians)

Needs irradiance -- no-irradiance datasets are reported as "n/a" for the new
method (they're the NSRDB-backfill candidates, a separate proposal).

Output: reports/energy_ratio_comparison.csv + a printed summary.

Usage:
    /opt/anaconda3/envs/pvtools/bin/python prototype_energy_ratio.py [--match 'my_1*'] [--limit N]
"""

import os
import sys
import csv
import fnmatch
import argparse
import warnings
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

import pvcopilot_pipeline_report as P
from page_supporting_files.analysis_utils import aggregate_daily

GAMMA = -0.004
MIN_DAY_POINTS = 4          # need this many daytime points for a day's sums
MIN_INSOLATION_FRAC = 0.2   # drop days with < 20% of the median daily insolation
RATIO_TRIM = (0.3, 1.7)     # sanity band around the median ratio (x median)
YOY_TOL_DAYS = 7
BOOT_N = 500
RNG = np.random.default_rng(0)   # fixed seed -> deterministic CI


# ---------------------------------------------------------------------------
# Proposed method
# ---------------------------------------------------------------------------
def daily_energy_ratio(df, mapping):
    """Daily ratio of measured energy to plane-of-array-expected energy.
    Returns (series indexed by day, n_span_days) or (None, reason)."""
    pcol = mapping.get("DC Power")
    icol = mapping.get("Irradiance")
    tcol = mapping.get("Module temperature")
    if not pcol or pcol not in df.columns:
        return None, "no power"
    if not icol or icol not in df.columns:
        return None, "no irradiance"

    p = pd.to_numeric(df[pcol], errors="coerce")
    g = pd.to_numeric(df[icol], errors="coerce")
    t = (pd.to_numeric(df[tcol], errors="coerce")
         if tcol and tcol in df.columns else None)

    # Expected power ~ irradiance with temperature correction (arbitrary scale;
    # the scale cancels in a %/yr trend).
    expected = g * (1 + GAMMA * (t - 25)) if t is not None else g.copy()

    # Keep daylight points with sane values. LOW threshold (50 W/m^2): the sums
    # do the denoising, we only exclude true night/dead readings.
    ok = (g >= 50) & (p >= 0) & expected.notna() & p.notna()
    sub = pd.DataFrame({"p": p[ok], "e": expected[ok], "g": g[ok]})
    if len(sub) == 0:
        return None, "no daylight points"
    day = sub.groupby(sub.index.date).agg(
        e_meas=("p", "sum"), e_exp=("e", "sum"),
        insol=("g", "sum"), n=("p", "size"))
    day.index = pd.to_datetime(day.index)

    # Day validity: enough points AND enough insolation that the sum is a real
    # day (kills heavily-gapped days that would distort the ratio).
    med_insol = day["insol"].median()
    day = day[(day["n"] >= MIN_DAY_POINTS)
              & (day["insol"] >= MIN_INSOLATION_FRAC * med_insol)
              & (day["e_exp"] > 0)]
    if len(day) == 0:
        return None, "no valid days"

    ratio = day["e_meas"] / day["e_exp"]
    med = ratio.median()
    ratio = ratio[(ratio >= RATIO_TRIM[0] * med) & (ratio <= RATIO_TRIM[1] * med)]
    if len(ratio) < 2:
        return None, "too few valid days"
    span_days = (ratio.index.max() - ratio.index.min()).days + 1
    return ratio, span_days


def yoy_with_ci(ratio):
    """Median year-over-year change of the daily ratio + bootstrap CI.
    Returns (rd, lo, hi, n_pairs) in %/yr, or (nan, nan, nan, 0)."""
    ratio = ratio.sort_index()
    idx = ratio.index
    targets = idx - pd.DateOffset(years=1)
    pos = idx.get_indexer(targets, method="nearest")
    changes = []
    for i, j in enumerate(pos):
        if j < 0:
            continue
        dt_days = abs((idx[i] - pd.DateOffset(years=1) - idx[j]).days)
        if dt_days > YOY_TOL_DAYS or idx[j] >= idx[i]:
            continue
        prev, curr = ratio.iloc[j], ratio.iloc[i]
        if prev > 0:
            changes.append((curr / prev - 1.0) * 100.0)
    changes = np.asarray(changes)
    if len(changes) < 5:
        return np.nan, np.nan, np.nan, len(changes)
    rd = float(np.median(changes))
    boots = np.median(
        RNG.choice(changes, size=(BOOT_N, len(changes)), replace=True), axis=1)
    lo, hi = float(np.percentile(boots, 5)), float(np.percentile(boots, 95))
    return rd, lo, hi, len(changes)


# ---------------------------------------------------------------------------
# Current method (exactly what the pipeline does)
# ---------------------------------------------------------------------------
def current_method(df, mapping):
    """Returns (retained_pct, n_daily, rd) via today's pipeline path."""
    try:
        filt = P._run_filters(df, mapping)
    except Exception:
        return np.nan, 0, np.nan
    retained = 100.0 * len(filt) / len(df) if len(df) else np.nan
    irr = mapping.get("Irradiance")
    irr = irr if irr and irr in filt.columns else None
    try:
        daily = aggregate_daily(filt, irr)
        s = daily if isinstance(daily, pd.Series) else daily.iloc[:, 0]
        n_daily = int(s.dropna().shape[0])
        rd, _ = P._run_degradation(filt, mapping, "YOY")
    except Exception:
        return retained, 0, np.nan
    return retained, n_daily, (float(rd) if rd is not None else np.nan)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = P.collect_files(P.DEFAULT_DATASET_DIR, include_examples=False,
                            match=args.match)
    if args.limit:
        files = files[: args.limit]

    rows = []
    devnull = open(os.devnull, "w")
    for path in tqdm(files, desc="comparing", unit="ds", file=sys.stderr):
        name = os.path.basename(path)
        try:
            with redirect_stdout(devnull):
                contents = P._build_contents(path)
                df, _, mapping, _, _ = P._with_timeout(
                    P.parse_contents, contents, name,
                    timeout=P.ANALYZE_TIMEOUT_S)
        except Exception as e:
            rows.append({"dataset": name, "error": f"analyze: {type(e).__name__}"})
            continue
        if df is None or not mapping.get("DC Power"):
            rows.append({"dataset": name, "error": "no power mapping"})
            continue

        with redirect_stdout(devnull):
            old_ret, old_nd, old_rd = current_method(df.copy(), mapping)

        ratio, span = daily_energy_ratio(df, mapping)
        if ratio is None:
            new = {"new_rd": np.nan, "new_lo": np.nan, "new_hi": np.nan,
                   "new_days": 0, "new_note": span}   # span holds the reason
        else:
            rd, lo, hi, npairs = yoy_with_ci(ratio)
            new = {"new_rd": rd, "new_lo": lo, "new_hi": hi,
                   "new_days": int(len(ratio)),
                   "new_note": f"{npairs} yoy pairs over {span}d span"}

        rows.append({
            "dataset": name,
            "old_retained_pct": None if np.isnan(old_ret) else round(old_ret, 1),
            "old_daily_points": old_nd,
            "old_rd": None if np.isnan(old_rd) else round(old_rd, 3),
            "new_valid_days": new["new_days"],
            "new_rd": None if np.isnan(new["new_rd"]) else round(new["new_rd"], 3),
            "new_ci": ("" if np.isnan(new["new_rd"]) else
                       f"[{new['new_lo']:.2f}, {new['new_hi']:.2f}]"),
            "new_ci_width": ("" if np.isnan(new["new_rd"]) else
                             round(new["new_hi"] - new["new_lo"], 2)),
            "note": new["new_note"],
            "error": "",
        })

    out = os.path.join(P.REPORT_DIR, "energy_ratio_comparison.csv")
    os.makedirs(P.REPORT_DIR, exist_ok=True)
    cols = ["dataset", "old_retained_pct", "old_daily_points", "old_rd",
            "new_valid_days", "new_rd", "new_ci", "new_ci_width", "note", "error"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- summary ----
    both = [r for r in rows if r.get("old_rd") is not None
            and r.get("new_rd") is not None]
    print(f"\ndatasets: {len(rows)}  |  comparable (both methods produced a rate): {len(both)}")
    if both:
        old_abs = [abs(r["old_rd"]) for r in both]
        new_abs = [abs(r["new_rd"]) for r in both]
        print(f"|rate| > 3%/yr   old: {sum(a > 3 for a in old_abs):3d}   new: {sum(a > 3 for a in new_abs):3d}")
        print(f"median |rate|    old: {np.median(old_abs):.2f}    new: {np.median(new_abs):.2f}")
        oldpts = [r["old_daily_points"] for r in both]
        newpts = [r["new_valid_days"] for r in both]
        print(f"median daily pts old: {np.median(oldpts):.0f}     new: {np.median(newpts):.0f}")
        in_band = [r for r in both
                   if -3 <= r["new_rd"] <= 0.5]
        print(f"new rates in plausible band (-3..0.5): {len(in_band)}/{len(both)}")
    print(f"\nfull table -> {out}")


if __name__ == "__main__":
    main()
