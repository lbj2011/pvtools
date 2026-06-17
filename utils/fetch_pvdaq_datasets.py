"""
fetch_pvdaq_datasets.py
=======================

Standalone helper (does NOT touch the app). Downloads one representative CSV
from each of N different PVDAQ systems in the public OEDI data lake into
./test_datasets/, so test_datasets_report.py can run them through PVcopilot's
column-identification step.

- Only CSV files are considered (the change_log_*.pdf files are ignored).
- Near-empty files (header only) are skipped by picking the LARGEST csv on the
  first listing page for each system.

Usage:
    conda activate pvtools
    python fetch_pvdaq_datasets.py            # default 15 systems
    python fetch_pvdaq_datasets.py 30         # custom count
"""

import os
import sys
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

BASE = "https://oedi-data-lake.s3.amazonaws.com/"
ROOT_PREFIX = "pvdaq/csv/pvdata/"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_datasets")
N_SYSTEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def _local(tag):
    """Strip the XML namespace from a tag name."""
    return tag.split("}")[-1]


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def list_xml(prefix, delimiter=None, max_keys=1000):
    params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
    if delimiter:
        params["delimiter"] = delimiter
    url = BASE + "?" + urllib.parse.urlencode(params)
    return ET.fromstring(_get(url))


def iter_systems():
    """Yield every system prefix, paginating through CommonPrefixes."""
    token = None
    while True:
        params = {"list-type": "2", "prefix": ROOT_PREFIX,
                  "delimiter": "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = BASE + "?" + urllib.parse.urlencode(params)
        root = ET.fromstring(_get(url))
        token = None
        truncated = False
        for el in root:
            tag = _local(el.tag)
            if tag == "CommonPrefixes":
                for child in el:
                    if _local(child.tag) == "Prefix":
                        yield child.text
            elif tag == "IsTruncated":
                truncated = (el.text == "true")
            elif tag == "NextContinuationToken":
                token = el.text
        if not (truncated and token):
            break


def biggest_csv(system_prefix):
    """Return (key, size) of the largest .csv on the first listing page."""
    root = list_xml(system_prefix, max_keys=1000)
    best = None
    for el in root:
        if _local(el.tag) != "Contents":
            continue
        key = size = None
        for child in el:
            if _local(child.tag) == "Key":
                key = child.text
            elif _local(child.tag) == "Size":
                size = int(child.text)
        if key and key.lower().endswith(".csv"):
            if best is None or size > best[1]:
                best = (key, size)
    return best


def main():
    os.makedirs(DEST, exist_ok=True)
    # Existing files stay untouched; we only ADD new ones and never overwrite.
    existing = set(os.listdir(DEST))
    print(f"{len([f for f in existing if f.endswith('.csv')])} CSV(s) already "
          f"in {DEST} (kept as-is).")
    print(f"Adding {N_SYSTEMS} varied raw-format systems...\n")

    downloaded = 0
    scanned = 0
    for sysprefix in iter_systems():
        if downloaded >= N_SYSTEMS:
            break
        scanned += 1
        sys_label = sysprefix.rstrip("/").split("/")[-1]  # e.g. system_id=34
        try:
            best = biggest_csv(sysprefix)
            if not best:
                continue  # empty system, skip silently
            key, size = best
            fname = key.split("/")[-1]
            # "Varied" = the raw per-day format (system_N__date_...), whose
            # column layout differs system-to-system. Skip the AC-aggregate
            # systems (already represented) and anything already on disk.
            if "__date_" not in fname:
                continue
            if fname in existing:
                continue
            dest_path = os.path.join(DEST, fname)
            urllib.request.urlretrieve(BASE + urllib.parse.quote(key), dest_path)
            existing.add(fname)
            downloaded += 1
            print(f"  [{downloaded}/{N_SYSTEMS}] {sys_label}: {fname}  ({size:,} bytes)")
        except Exception as e:
            print(f"  {sys_label}: ERROR {type(e).__name__}: {e}")

    print(f"\nAdded {downloaded} new CSV(s) (scanned {scanned} systems). "
          f"Folder now has {len([f for f in os.listdir(DEST) if f.endswith('.csv')])} CSVs.")
    print("Next: python test_datasets_report.py")


if __name__ == "__main__":
    main()
