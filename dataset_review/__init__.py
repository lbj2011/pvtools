"""
Internal review tool for the PV field-degradation dataset, mounted inside pvtools at /dataset-review/.

Deliberately NOT linked from any page or the navbar (internal testing only): reviewers get the URL
directly. The page sends noindex; login is required (password: env DATASET_REVIEW_PASSWORD).

Data and the review log live in the private S3 bucket (never in git):
    s3://pvtools/dataset_review/dataset_review.pkl.xz     the dataset shown (slimmed, see prepare_and_upload.py)
    s3://pvtools/dataset_review/data_dictionary.csv       field descriptions for the hover cards
    s3://pvtools/dataset_review/review_log.sqlite         marks + comments; pulled at start-up, pushed after each write
For local development put the first two in dataset_review/data/ (git-ignored) and no S3 access is needed
for them; the log is then kept in DATASET_REVIEW_CACHE only if S3 is unreachable.

The review app itself is review_app.py (the same file as paper_check/review_app/app.py), which runs
standalone too. PDFs are not served here (copyright): reviewers open the DOI or pick a local PDF folder.
"""
import os
import pathlib
import sys
import types

BASE = "/dataset-review/"
BUCKET = os.environ.get("DATASET_REVIEW_BUCKET", "pvtools")
PREFIX = "dataset_review/"
HERE = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path(os.environ.get("DATASET_REVIEW_CACHE", "/tmp/dataset_review"))
FILES = {"data": "dataset_review.pkl.xz", "dict_csv": "data_dictionary.csv"}


def _fetch(name):
    """local copy in dataset_review/data/ if present, else download from S3 into the cache"""
    local = HERE / "data" / name
    if local.exists():
        return local
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / name
    if not dst.exists():
        import boto3
        boto3.client("s3").download_file(BUCKET, PREFIX + name, str(dst))
    return dst


def mount(server):
    """register the review app on the pvtools Flask server; returns True if mounted"""
    try:
        paths = {k: _fetch(v) for k, v in FILES.items()}
    except Exception as e:
        print(f"dataset review not mounted: data not available ({type(e).__name__}: {e})")
        return False
    CACHE.mkdir(parents=True, exist_ok=True)
    sys.modules["pv_review_host"] = types.SimpleNamespace(
        server=server, base=BASE, data=str(paths["data"]), dict_csv=str(paths["dict_csv"]),
        log=str(CACHE / "review_log.sqlite"), public_mode=True,
        password=os.environ.get("DATASET_REVIEW_PASSWORD", "duramat"),
        s3_bucket=None if os.environ.get("DATASET_REVIEW_NO_S3") else BUCKET, s3_log_key=PREFIX + "review_log.sqlite")
    from . import review_app  # noqa: F401  (builds the Dash app and its callbacks)
    from flask import redirect
    for k, alias in enumerate(("/dataset-review", "/dataset_review", "/dataset_review/")):   # typos / no slash
        server.add_url_rule(alias, f"dataset_review_alias_{k}", lambda: redirect(BASE))
    print(f"dataset review mounted at {BASE}")
    return True
