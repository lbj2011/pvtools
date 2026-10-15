"""
build_usable_datasets.py
=========================

Standalone helper (does NOT touch the app). Builds a VARIETY of real
multi-year / multi-day PVcopilot datasets from the public OEDI PVDAQ data lake,
each chosen to exercise a specific data tier / fix.

Each build:
  1. Lists ALL daily CSV files for the system (paginated).
  2. Selects files -- spread across the full range ("multiyear") or a single
     contiguous block ("multiday").
  3. Downloads each, replaces missing-value sentinels (-99999 etc.) with NaN,
     resamples to hourly means, and concatenates them.
  4. Writes one CSV per spec into ./test_datasets/.

Usage:
    conda activate pvtools
    python build_usable_datasets.py
"""

import os
import io
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

BASE = "https://oedi-data-lake.s3.amazonaws.com/"
ROOT = "pvdaq/csv/pvdata/"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_datasets")

# (system_id, output_filename, mode, n_files)
BUILD_SPECS = [
    (3,    "multiyear_full_dcnaming_sys3.csv",        "multiyear", 150),  # full, dc_pos_* naming
    (1199, "multiyear_no_irradiance_sys1199.csv",     "multiyear", 150),  # power+V+I, no irradiance
    (1265, "multiyear_power_only_sys1265.csv",        "multiyear", 150),  # power only (no irr/temp)
    (1429, "multiyear_compute_power_VxI_sys1429.csv", "multiyear", 150),  # V+I, no power -> V*I
    (4,    "multiday_short_span_sys4.csv",            "multiday",   90),  # full, ~90 contiguous days
]
RESAMPLE = "1h"
SENTINELS = [-99999.0, -9999.0, -999.0]
DATE_RE = re.compile(r"date_(\d{4})_(\d{2})_(\d{2})")


def _local(tag):
    return tag.split("}")[-1]


def _get(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read()


def list_csv_keys(system_id):
    prefix = f"{ROOT}system_id={system_id}/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(_get(BASE + "?" + urllib.parse.urlencode(params)))
        token, truncated = None, False
        for el in root:
            tag = _local(el.tag)
            if tag == "Contents":
                key = None
                for child in el:
                    if _local(child.tag) == "Key":
                        key = child.text
                if key and key.lower().endswith(".csv"):
                    keys.append(key)
            elif tag == "IsTruncated":
                truncated = (el.text == "true")
            elif tag == "NextContinuationToken":
                token = el.text
        if not (truncated and token):
            break
    return keys


def sorted_by_date(keys):
    dated = [(m.group(0), k) for k in keys if (m := DATE_RE.search(k))]
    dated.sort()
    return [k for _, k in dated]


def select(keys, mode, n):
    keys = sorted_by_date(keys)
    if len(keys) <= n:
        return keys
    if mode == "multiday":
        # one contiguous block of n days, starting a quarter of the way in
        start = len(keys) // 4
        return keys[start:start + n]
    # multiyear: spread n files evenly across the full date range
    step = len(keys) / n
    return [keys[int(i * step)] for i in range(n)]


def load_one(key):
    raw = _get(BASE + urllib.parse.quote(key))
    df = pd.read_csv(io.BytesIO(raw))
    time_col = next((c for c in df.columns if c.lower() in
                     ("measured_on", "time", "timestamp", "datetime")), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)
    df = df.replace(SENTINELS, np.nan)
    num = df.select_dtypes(include=[np.number])
    return None if num.empty else num.resample(RESAMPLE).mean()


def build(system_id, out_name, mode, n_files):
    print(f"\n=== system_id={system_id}  ->  {out_name}  ({mode}) ===")
    keys = list_csv_keys(system_id)
    sample = select(keys, mode, n_files)
    print(f"  {len(keys)} daily files; using {len(sample)} ({mode})...")

    frames = []
    for i, key in enumerate(sample, 1):
        try:
            part = load_one(key)
            if part is not None and not part.empty:
                frames.append(part)
        except Exception as e:
            print(f"    skip {key.split('/')[-1]}: {type(e).__name__}")
        if i % 30 == 0:
            print(f"    {i}/{len(sample)} processed...")

    if not frames:
        print("  no usable data, skipping.")
        return

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full.index.name = "measured_on"
    out = os.path.join(DEST, out_name)
    full.reset_index().to_csv(out, index=False)

    span = f"{full.index.min().date()} -> {full.index.max().date()}"
    cols = [c for c in full.columns if c != "system_id"]
    print(f"  WROTE {out_name}: rows={len(full):,}  span={span}")
    print(f"  columns: {', '.join(cols)}")


def main():
    os.makedirs(DEST, exist_ok=True)
    for sid, name, mode, n in BUILD_SPECS:
        try:
            build(sid, name, mode, n)
        except Exception as e:
            print(f"  system {sid}: ERROR {type(e).__name__}: {e}")
    print(f"\nDone. New datasets are in {DEST}/")


if __name__ == "__main__":
    main()
