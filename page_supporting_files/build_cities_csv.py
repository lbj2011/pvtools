"""
One-time script to build data/cities15000.csv from GeoNames.

Run this ONCE locally to produce the CSV used by field_degradation.py:

    python build_cities_csv.py

It fetches the cities15000.zip dump from GeoNames (~4 MB), keeps only the
columns we need (name, ascii name, country code, lat, lon, population), and
writes data/cities15000.csv (~1 MB).

GeoNames is licensed CC BY 4.0. https://download.geonames.org/export/dump/
"""
import io
import os
import zipfile
import urllib.request

import pandas as pd

URL = "https://download.geonames.org/export/dump/cities15000.zip"
OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "cities15000.csv")

# GeoNames file format (tab-separated, no header). See:
# https://download.geonames.org/export/dump/readme.txt
COLS_FULL = [
    "geonameid", "name", "asciiname", "alternatenames",
    "latitude", "longitude", "feature_class", "feature_code",
    "country_code", "cc2", "admin1_code", "admin2_code",
    "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        zip_bytes = resp.read()
    print(f"  got {len(zip_bytes) / 1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The archive contains cities15000.txt
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(name) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=COLS_FULL,
                low_memory=False,
                keep_default_na=False,   # 'NA' is Namibia's country code, don't NaN it
                na_values=[""],
            )

    print(f"  parsed {len(df):,} rows")

    # Keep only what we need
    keep = ["name", "asciiname", "country_code", "latitude", "longitude", "population"]
    df = df[keep].copy()

    # Sort by population desc so when we display matches the biggest city wins
    df = df.sort_values("population", ascending=False).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"  wrote {OUT_PATH}  ({size_mb:.2f} MB, {len(df):,} cities)")


if __name__ == "__main__":
    main()
