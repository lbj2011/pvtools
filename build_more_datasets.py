"""
build_more_datasets.py
======================

Builds N additional COMBINED multiyear datasets from the public OEDI PVDAQ data
lake (s3://oedi-data-lake/pvdaq/csv/pvdata/system_id=*/), in the SAME style as
the existing test_datasets/my_01..my_50 files.

For each NEW system (one not already represented in the manifest), it:
  1. Lists every daily CSV the system has (paginated).
  2. Selects files spread evenly across the full date range ("multiyear").
  3. Downloads each, replaces missing-value sentinels with NaN, resamples to
     hourly means, and concatenates them -- i.e. the real data that lives in
     that folder, not synthetic iterations.
  4. Writes test_datasets/my_<NN>_csv_<system_id>.csv  (NN continues from 51).
  5. Appends a row to test_datasets/GENERATED_MANIFEST.md.

Systems are taken in the lake's listing order, skipping any already used and any
that yield no usable data, until N new datasets exist.

Usage (pvtools env):
    conda activate pvtools
    python build_more_datasets.py            # 50 new datasets (default)
    python build_more_datasets.py 25         # custom count
"""

import os
import re
import io
import sys
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"\nERROR: missing dependency '{e.name}'. Run inside the pvtools env:\n"
        f"    conda activate pvtools\n"
        f"    python build_more_datasets.py\n"
        f"(not the system Python: {sys.executable})\n\n")
    sys.exit(1)

BASE = "https://oedi-data-lake.s3.amazonaws.com/"
ROOT = "pvdaq/csv/pvdata/"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(PROJECT_ROOT, "test_datasets")
MANIFEST = os.path.join(DEST, "GENERATED_MANIFEST.md")

N_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_FILES = 150          # daily files sampled per system (matches the original my_* builds)
RESAMPLE = "1h"
SENTINELS = [-99999.0, -9999.0, -999.0]
DATE_RE = re.compile(r"date_(\d{4})_(\d{2})_(\d{2})")


def _local(tag):
    return tag.split("}")[-1]


def _get(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read()


def existing_systems():
    """System ids already represented in the manifest (so we don't repeat them)."""
    used = set()
    if os.path.exists(MANIFEST):
        for line in open(MANIFEST):
            m = re.match(r"\|\s*my_\d+\S*\.csv\s*\|\s*\w+\s*\|\s*(\d+)\s*\|", line)
            if m:
                used.add(m.group(1))
    return used


def next_index():
    """Continue numbering from the highest existing my_<NN>_* file (so re-runs
    extend the set instead of clobbering my_51, my_52, ...)."""
    mx = 0
    for f in os.listdir(DEST):
        m = re.match(r"my_(\d+)(_csv_\d+)?\.csv$", f)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def iter_systems():
    """Yield every system_id (as a string) in the lake, in listing order."""
    token = None
    while True:
        params = {"list-type": "2", "prefix": ROOT, "delimiter": "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(_get(BASE + "?" + urllib.parse.urlencode(params)))
        token = None
        truncated = False
        for el in root:
            tag = _local(el.tag)
            if tag == "CommonPrefixes":
                for child in el:
                    if _local(child.tag) == "Prefix":
                        # ".../system_id=10019/" -> "10019"
                        m = re.search(r"system_id=([^/]+)/", child.text)
                        if m:
                            yield m.group(1)
            elif tag == "IsTruncated":
                truncated = (el.text == "true")
            elif tag == "NextContinuationToken":
                token = el.text
        if not (truncated and token):
            break


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


def select_multiyear(keys, n):
    keys = sorted_by_date(keys)
    if len(keys) <= n:
        return keys
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


def build_system(system_id, index):
    keys = list_csv_keys(system_id)
    sample = select_multiyear(keys, N_FILES)
    if not sample:
        print(f"  system {system_id}: no dated daily CSVs, skipping.")
        return None

    frames = []
    for k in sample:
        try:
            part = load_one(k)
            if part is not None and not part.empty:
                frames.append(part)
        except Exception:
            pass
    if not frames:
        print(f"  system {system_id}: no usable data, skipping.")
        return None

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full.index.name = "measured_on"

    out_name = f"my_{index:02d}_csv_{system_id}.csv"
    full.reset_index().to_csv(os.path.join(DEST, out_name), index=False)

    span = f"{full.index.min().date()}→{full.index.max().date()}"
    cols = ", ".join(c for c in full.columns if c != "system_id")
    print(f"  [{index}] system {system_id}: {len(keys)} daily files -> "
          f"{out_name}  rows={len(full):,}  span={span}")
    # Manifest row: | file | kind | system | span | rows | columns | renamed |
    return f"| {out_name} | multiyear | {system_id} | {span} | {len(full)} | {cols} | no |\n"


def main():
    os.makedirs(DEST, exist_ok=True)
    used = existing_systems()
    print(f"{len(used)} systems already represented; building {N_NEW} new dataset(s)...\n")

    manifest_rows = []
    built = 0
    index = next_index()
    scanned = 0
    for system_id in iter_systems():
        if built >= N_NEW:
            break
        if system_id in used:
            continue
        scanned += 1
        try:
            row = build_system(system_id, index)
        except Exception as e:
            print(f"  system {system_id}: ERROR {type(e).__name__}: {e}")
            row = None
        if row:
            manifest_rows.append(row)
            used.add(system_id)
            built += 1
            index += 1

    if manifest_rows and os.path.exists(MANIFEST):
        with open(MANIFEST, "a") as f:
            f.writelines(manifest_rows)

    print(f"\nDone. Built {built} new dataset(s) (scanned {scanned} new systems). "
          f"Folder now has {len([f for f in os.listdir(DEST) if f.startswith('my_') and f.endswith('.csv')])} my_* datasets.")


if __name__ == "__main__":
    main()
