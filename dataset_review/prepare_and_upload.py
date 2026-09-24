#!/usr/bin/env python3
"""
Put the dataset (and optionally the existing review log) where the pvtools review page reads it.

    python dataset_review/prepare_and_upload.py SRC.pkl DICT.csv [--log review_log.sqlite] [--local]
      SRC.pkl   e.g. paper_check/data_2609_merged_all_faults_with_rate.pkl
      DICT.csv  paper_check/data_dictionary.csv
      --log     seed / replace the review log in S3 with this sqlite file (e.g. the local review log)
      --local   only write dataset_review/data/ (git-ignored) for local testing, no upload
    python dataset_review/prepare_and_upload.py --download-log OUT.sqlite    # get the web review log

Only the columns the review app uses are kept (keeps the dyno's memory low). Needs AWS credentials
with write access to the bucket (same as the other pvtools S3 files).
"""
import argparse
import pathlib
import shutil

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
BUCKET, PREFIX = "pvtools", "dataset_review/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dict_csv", nargs="?")
    ap.add_argument("--log")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--download-log")
    a = ap.parse_args()
    if a.download_log:
        import boto3
        boto3.client("s3").download_file(BUCKET, PREFIX + "review_log.sqlite", a.download_log)
        print(f"s3://{BUCKET}/{PREFIX}review_log.sqlite -> {a.download_log}")
        return
    df = pd.read_pickle(a.src)
    code = (HERE / "review_app.py").read_text()
    keep = [c for c in df.columns if f'"{c}"' in code or f"'{c}'" in code]
    out_dir = HERE / "data"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "dataset_review.pkl.xz"
    df[keep].to_pickle(out)
    shutil.copy(a.dict_csv, out_dir / "data_dictionary.csv")
    print(f"{len(df)} rows, {len(keep)}/{len(df.columns)} columns -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if a.local:
        return
    import boto3
    s3 = boto3.client("s3")
    for f in (out, out_dir / "data_dictionary.csv"):
        s3.upload_file(str(f), BUCKET, PREFIX + f.name)
        print(f"uploaded s3://{BUCKET}/{PREFIX}{f.name}")
    if a.log:
        s3.upload_file(a.log, BUCKET, PREFIX + "review_log.sqlite")
        print(f"uploaded s3://{BUCKET}/{PREFIX}review_log.sqlite (review log replaced)")


if __name__ == "__main__":
    main()
