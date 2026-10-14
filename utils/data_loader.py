import pandas as pd
import logging

logger = logging.getLogger(__name__)

_df_cache = None

NEW_DATA = "data/data_2609_w_rate.parquet"     # validated 2026-09 dataset (rows with a rate only; git-ignored)
OLD_DATA = "data/data_2605_w_pvcz.parquet"


def _compat(df):
    """new dataset -> the column names the field-degradation page was written for"""
    df = df.copy()
    df["row_key"] = df.index.astype(int)
    alias = {"paper id": "eid", "confidence level": "annual_power_deg_confidence_level",
             "scope of study": "scope_of_study", "system capacity in watts": "system_capacity_watts"}
    for old, new in alias.items():
        if new in df.columns:
            df[old] = df[new]
    for c in ("rate", "rate_abs", "duration", "latitude", "longitude", "publish year", "system capacity",
              "system capacity in watts"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    import json

    def _lst(v):
        try:
            return json.loads(v) if isinstance(v, str) and v.startswith("[") else ([] if v is None else v)
        except Exception:
            return []
    df["faults_list"] = df["faults"].map(_lst)   # verified faults of each data point (old column held unverified ones)
    df["document type"] = df["type_code"].map({"J": "Journal", "C": "Conference", "B": "Book",
                                               "Review": "Review"}).fillna("Other")
    return df


def get_df():
    global _df_cache
    if _df_cache is None:
        import os
        if os.path.exists(NEW_DATA):
            _df_cache = _compat(pd.read_parquet(NEW_DATA))
        else:
            logger.warning("%s not found, using %s", NEW_DATA, OLD_DATA)
            _df_cache = pd.read_parquet(OLD_DATA)
    return _df_cache

def safe_get_df():
    try:
        return get_df()
    except Exception:
        logger.exception("Failed to load parquet data")
        return pd.DataFrame()