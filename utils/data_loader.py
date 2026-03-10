import pandas as pd
import logging

logger = logging.getLogger(__name__)

_df_cache = None

def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_parquet("data/data_250924.parquet")
    return _df_cache

def safe_get_df():
    try:
        return get_df()
    except Exception:
        logger.exception("Failed to load parquet data")
        return pd.DataFrame()
