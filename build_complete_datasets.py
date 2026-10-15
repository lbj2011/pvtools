"""
build_complete_datasets.py
==========================

Like build_more_datasets.py, but PREFERS systems that have all the variables we
care about: DC power, voltage, current, irradiance, and module temperature
(Time is always present). For each unused PVDAQ system it downloads one sample
daily file, scores how many of the 5 variable types its columns contain, and:

  * score 5 (complete)  -> build immediately
  * score 3-4           -> held as a backup
After scanning, if fewer than TARGET complete systems were found, it fills the
remainder from the highest-scoring backups -- i.e. "all variables if possible,
otherwise maximize, otherwise best effort".

Output continues the my_<NN>_csv_<system_id>.csv naming and appends to the
manifest, exactly like build_more_datasets.py.

Usage (pvtools env):
    conda activate pvtools
    python build_complete_datasets.py          # 25 datasets (default)
    python build_complete_datasets.py 10
"""

import io
import re
import sys
import urllib.parse

try:
    import pandas as pd
    import build_more_datasets as B   # reuse iter_systems/list_csv_keys/build_system/...
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"\nERROR: missing dependency '{e.name}'. Run inside the pvtools env:\n"
        f"    conda activate pvtools\n"
        f"    python build_complete_datasets.py\n\n")
    sys.exit(1)

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SCAN_CAP = 500            # max systems to sample-inspect before filling from backups
SAMPLE_NROWS = 5

# Name patterns for each required variable type (AC or DC both count for scoring;
# the tool handles AC fallback downstream). Matched as substrings, case-insensitive.
VAR_PATTERNS = {
    "power":       re.compile(r"pow|pwr|p_?dc|pdc|p_?mp|pmpp?|ppv|p_?out|p_?ac|kw|watt", re.I),
    "voltage":     re.compile(r"volt|v_?dc|vdc|v_?mp|vmpp?|vpv|v_?ac|vac|v_?pos", re.I),
    "current":     re.compile(r"curr|i_?dc|idc|i_?mp|impp?|ipv|i_?ac|iac|amp|a_avg", re.I),
    "irradiance":  re.compile(r"irr|poa|ghi|dni|pyran|insol|solar|irrad|g_?poa", re.I),
    "temperature": re.compile(r"temp|t_?mod|tmod|t_?cell|tcell|module_temp|bom", re.I),
}


def _score_columns(columns):
    present = set()
    for c in columns:
        s = str(c)
        for var, rgx in VAR_PATTERNS.items():
            if rgx.search(s):
                present.add(var)
    return len(present), present


def _sample_columns(dated_keys):
    """Download the first daily file just to read its column names (cheap)."""
    raw = B._get(B.BASE + urllib.parse.quote(dated_keys[0]))
    return list(pd.read_csv(io.BytesIO(raw), nrows=SAMPLE_NROWS).columns)


def main():
    used = B.existing_systems()
    index = B.next_index()
    print(f"{len(used)} systems already used; finding {TARGET} with as many of "
          f"{list(VAR_PATTERNS)} as possible...\n")

    built = 0
    scanned = 0
    backups = []          # (score, system_id) for partial systems, used to fill
    manifest_rows = []

    def _do_build(system_id):
        nonlocal built, index
        try:
            row = B.build_system(system_id, index)
        except Exception as e:
            print(f"  system {system_id}: ERROR {type(e).__name__}: {e}")
            return False
        if row:
            manifest_rows.append(row)
            used.add(system_id)
            built += 1
            index += 1
            return True
        return False

    for system_id in B.iter_systems():
        if built >= TARGET or scanned >= SCAN_CAP:
            break
        if system_id in used:
            continue
        keys = B.list_csv_keys(system_id)
        dated = B.sorted_by_date(keys)
        if not dated:
            continue
        scanned += 1
        try:
            cols = _sample_columns(dated)
        except Exception:
            continue
        score, present = _score_columns(cols)
        missing = [v for v in VAR_PATTERNS if v not in present]
        print(f"  scan system {system_id}: {score}/5 "
              f"[{', '.join(sorted(present))}]" + (f"  missing {missing}" if missing else ""))
        if score >= 5:
            _do_build(system_id)
        else:
            backups.append((score, system_id))

    # Fill the remainder from the most-complete partial systems.
    if built < TARGET:
        backups.sort(reverse=True)   # highest score first
        print(f"\nOnly {built} complete (5/5) systems found; filling "
              f"{TARGET - built} more from the most-complete partials...")
        for score, system_id in backups:
            if built >= TARGET:
                break
            print(f"  fill system {system_id} (score {score}/5)")
            _do_build(system_id)

    if manifest_rows and B.os.path.exists(B.MANIFEST):
        with open(B.MANIFEST, "a") as f:
            f.writelines(manifest_rows)

    n_my = len([f for f in B.os.listdir(B.DEST)
                if f.startswith("my_") and f.endswith(".csv")])
    print(f"\nDone. Built {built} dataset(s) (scanned {scanned} systems). "
          f"Folder now has {n_my} my_* datasets.")


if __name__ == "__main__":
    main()
