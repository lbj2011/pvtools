"""
build_dkasc_datasets.py
=======================

Batch-import the DKASC Alice Springs PV arrays (https://dkasolarcentre.com.au)
into test_datasets/.

For each site: download the full 5-minute export, resample to HOURLY means
(matching the library's cadence), and save as

    new_my_<N>.csv   when the data spans >= 2 years   ("multiyear")
    new_y_<N>.csv    when it spans under 2 years      ("~1 year")

Only actual PV arrays are imported -- the master meter, weather station,
BESS/microgrid feeds and site totals are skipped (not panel data).

Provenance is recorded in test_datasets/new_manifest.csv (site id, DKASC
export name, span, rows). The script is RESUMABLE: sites already in the
manifest are skipped, so it can be re-run after an interruption.

Usage:
    /opt/anaconda3/envs/pvtools/bin/python build_dkasc_datasets.py
"""

import csv
import os
import re
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(PROJECT_ROOT, "test_datasets")
MANIFEST = os.path.join(DEST, "new_manifest.csv")
BASE = "https://solarcentre.spinifexvalley.com.au/export/"
TMP = "/tmp/dkasc_dl.csv"
MULTIYEAR_MIN_YEARS = 2.0

# PV arrays at Alice Springs (site id -> export filename). Non-PV feeds
# (MasterMeter, WeatherStation, PWC mains, Mutitjulu, microgrid/BESS, totals)
# are deliberately absent. Site 70 was imported manually as new_my_1; sites
# 52-66 were imported by the first batch run (see new_manifest.csv).
#
# CURATED to 25 total new datasets: rather than importing every phase of the
# same module bank (near-duplicates), this list covers ONE array per module
# not yet represented (M1-M11) plus the special/refit arrays (archived
# Q-Cells, Uni of Singapore, BIITE, one _II refit) for technology diversity.
SITES = {
    100: "100-Site_DKA-M1_A-Phase.csv",   # M1
    81:  "81-Site_DKA-M2_A-Phase.csv",    # M2
    90:  "90-Site_DKA-M3_A-Phase.csv",    # M3
    93:  "93-Site_DKA-M4_A-Phase.csv",    # M4
    79:  "79-Site_DKA-M6_A-Phase.csv",    # M6
    85:  "85-Site_DKA-M7_A-Phase.csv",    # M7
    67:  "67-Site_DKA-M8_A-Phase.csv",    # M8
    87:  "87-Site_DKA-M9_A+C-Phases.csv", # M9
    97:  "97-Site_DKA-M10_B+C-Phases.csv",# M10
    78:  "78-Site_DKA-M11_3-Phase.csv",   # M11
    205: "205-Site_Archived_DKA-M15_BPhase_UMG_QCells.csv",  # archived Q-Cells
    215: "215-Site_DKA-Uni_of_Singapore.csv",                # special array
    219: "219-Site_DKA-BIITE_and_CLC.csv",                   # special array
    218: "218-Site_DKA-M4_C-Phase_II.csv",                   # gen-II refit
}


def load_manifest():
    rows = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            rows = list(csv.DictReader(f))
    return rows


def next_counters(manifest):
    """Continue numbering from the manifest AND any files already on disk."""
    my, y = 0, 0
    names = [r["dataset"] for r in manifest] + os.listdir(DEST)
    for n in names:
        m = re.match(r"new_my_(\d+)\.csv$", n)
        if m:
            my = max(my, int(m.group(1)))
        m = re.match(r"new_y_(\d+)\.csv$", n)
        if m:
            y = max(y, int(m.group(1)))
    return my, y


def main():
    manifest = load_manifest()
    done_sites = {r["site_id"] for r in manifest}
    my_n, y_n = next_counters(manifest)
    new_rows = []

    for site_id, fname in sorted(SITES.items()):
        if str(site_id) in done_sites:
            print(f"site {site_id}: already imported, skipping")
            continue
        url = BASE + fname
        print(f"site {site_id}: downloading {fname} ...", flush=True)
        r = subprocess.run(["curl", "-sL", "--max-time", "900", url, "-o", TMP])
        if r.returncode != 0 or not os.path.exists(TMP) or os.path.getsize(TMP) < 1e6:
            print(f"site {site_id}: DOWNLOAD FAILED, skipping", flush=True)
            continue
        try:
            df = pd.read_csv(TMP, parse_dates=["timestamp"], on_bad_lines="skip")
            df = df.set_index("timestamp").sort_index()
            hourly = df.resample("1h").mean().dropna(how="all")
            span = (hourly.index.max() - hourly.index.min()).days / 365.25
        except Exception as e:
            print(f"site {site_id}: CONVERT FAILED ({type(e).__name__}: {e})", flush=True)
            continue
        finally:
            try:
                os.remove(TMP)
            except OSError:
                pass

        if len(hourly) < 100:
            print(f"site {site_id}: only {len(hourly)} hourly rows, skipping", flush=True)
            continue

        if span >= MULTIYEAR_MIN_YEARS:
            my_n += 1
            out = f"new_my_{my_n}.csv"
        else:
            y_n += 1
            out = f"new_y_{y_n}.csv"
        hourly.to_csv(os.path.join(DEST, out))
        row = {"dataset": out, "site_id": str(site_id), "source": fname,
               "span_years": f"{span:.2f}", "hourly_rows": str(len(hourly))}
        new_rows.append(row)
        print(f"site {site_id}: -> {out}  ({span:.1f} yr, {len(hourly)} rows)", flush=True)

        # Append to the manifest as we go so an interruption loses nothing.
        exists = os.path.exists(MANIFEST)
        with open(MANIFEST, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "site_id", "source",
                                              "span_years", "hourly_rows"])
            if not exists:
                w.writeheader()
            w.writerow(row)

    print(f"\nDone. Imported {len(new_rows)} new dataset(s). "
          f"Manifest: {os.path.relpath(MANIFEST, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
