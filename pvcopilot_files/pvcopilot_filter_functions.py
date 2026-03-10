import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Import for plotting

def auto_fix_timezone(df, time_key, power_key, target_tz="local"):
    """
    Automatically detect and correct timezone and Daylight Saving Time (DST)
    issues in PV system timeseries data.

    This function assumes that:
        • `df.index` is already a pandas.DatetimeIndex.
        • The index name equals `time_key`.
        • `power_key` refers to a column representing PV power output.
    
    The function applies three layers of detection:
        1. Detect whether timestamps are tz-naive and localize them as UTC.
        2. Detect whether the daily solar peak appears shifted by ±1 hour.
           If the average daily peak hour is outside the expected solar window,
           it applies a +1 or -1 hour correction.
        3. Detect DST jump irregularities:
               - Missing hour (DST forward shift)
               - Repeated hour (DST backward shift)
           and adjust the timestamps accordingly.
    
    After corrections, the function optionally converts the index timezone
    into the desired target timezone.

    Parameters
    ----------
    df : pandas.DataFrame
        Input PV timeseries data. Must have a DatetimeIndex.
    
    time_key : str
        Name of the DatetimeIndex (for reference/logging purposes).

    power_key : str
        Column name containing PV power data. Used to detect solar noon shifts.
    
    target_tz : str, default "local"
        If a valid timezone string is provided (e.g., "America/Los_Angeles"),
        the timestamps are converted to that timezone.
        If "local", no timezone conversion is applied.

    Returns
    -------
    df_fixed : pandas.DataFrame
        The corrected DataFrame with updated DatetimeIndex.

    message : str
        Human-readable summary describing which corrections were applied.

    Notes
    -----
    • This method uses heuristics and solar-peak assumptions.
    • Designed for PV datasets with at least several days of data.
    • Does NOT drop any data, only shifts timestamps.

    Examples
    --------
    >>> df_fixed, msg = auto_fix_timezone(
    ...     df,
    ...     time_key="time",
    ...     power_key="p_mp_ref",
    ...     target_tz="America/Los_Angeles"
    ... )
    >>> print(msg)
    Index was tz-naive → localized as UTC.
    Detected early solar peak → applied +1 hour correction.
    Converted index timezone to America/Los_Angeles.
    """

    # Make a safe copy
    df = df.copy()
    messages = []

    # -------------------------------------
    # 1. Ensure index is a DatetimeIndex
    # -------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a pandas.DatetimeIndex")

    ts = df.index

    # -------------------------------------
    # 2. Localize if timezone-naive
    # -------------------------------------
    if ts.tz is None:
        df.index = df.index.tz_localize("UTC")
        ts = df.index
        messages.append("Index was tz-naive → localized as UTC.")

    # -------------------------------------
    # 3. Detect ±1 hour solar peak shift
    # -------------------------------------
    peak_based_shift = detect_timezone_offset(df, power_key)

    if peak_based_shift == "+1 hour":
        df.index = df.index + pd.Timedelta(hours=1)
        messages.append("Detected early solar peak → applied +1 hour correction.")
    elif peak_based_shift == "-1 hour":
        df.index = df.index - pd.Timedelta(hours=1)
        messages.append("Detected late solar peak → applied -1 hour correction.")
    else:
        messages.append("Solar peak time appears normal → no ±1 hour correction applied.")

    # -------------------------------------
    # 4. Detect DST irregularities
    # -------------------------------------
    missing_hours, duplicated_hours = detect_dst_jump(df)

    if 2 in missing_hours:
        df.index = df.index + pd.Timedelta(hours=1)
        messages.append("Detected DST forward jump (missing 2:00) → applied +1 hour shift.")

    if 2 in duplicated_hours:
        df.index = df.index - pd.Timedelta(hours=1)
        messages.append("Detected DST backward jump (duplicated 2:00) → applied -1 hour shift.")

    # -------------------------------------
    # 5. Convert to target timezone (optional)
    # -------------------------------------
    if target_tz != "local":
        df.index = df.index.tz_convert(target_tz)
        messages.append(f"Converted index timezone to {target_tz}.")
    else:
        messages.append("Timezone kept unchanged (target_tz='local').")

    return df, "\n".join(messages)



def detect_timezone_offset(df, power_key):
    daily_peak = df.groupby(df.index.date)[power_key].idxmax()
    hours = [t.hour for t in daily_peak]
    avg_peak_hour = sum(hours) / len(hours)

    if avg_peak_hour < 9:
        return "+1 hour"
    elif avg_peak_hour > 15:
        return "-1 hour"
    else:
        return "OK"


def detect_dst_jump(df):
    h = df.index.hour
    missing_hours = set(range(24)) - set(h)
    duplicated_hours = [x for x in range(24) if (h == x).sum() > 120]
    return missing_hours, duplicated_hours

def identify_outliers_iqr(df: pd.DataFrame, power_key: str, time_key: str, iqr_multiplier: float = 1.5):
    """
    Identifies outliers in a specified column of a DataFrame using the Interquartile Range (IQR) method.

    Outliers are defined as data points that are less than Q1 - (IQR * iqr_multiplier) or
    greater than Q3 + (IQR * iqr_multiplier).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        power_key (str): The column name to use for outlier detection (e.g., 'p_mp_ref').
        time_key (str): The column name for the timestamp (used for context, not calculation).
        iqr_multiplier (float): The multiplier for the IQR range. Default is 1.5 (Tukey's Fences).

    Returns:
        tuple: A tuple containing two pandas Index objects:
               - normal_indices: The indices of the normal data points.
               - outlier_indices: The indices of the outlier data points.
    """
    if power_key not in df.columns:
        print(f"Error: The specified power column '{power_key}' does not exist in the DataFrame.")
        return pd.Index([]), pd.Index([])

    # Ensure the data is numeric and drop NaNs for quantile calculation
    data = df[power_key].dropna()

    # 1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # 2. Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1

    # 3. Define the lower and upper bounds (Fences)
    lower_bound = Q1 - (IQR * iqr_multiplier)
    upper_bound = Q3 + (IQR * iqr_multiplier)

    print(f"\n--- Outlier Detection Metrics for '{power_key}' ---")
    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print("---------------------------------------------------\n")

    # 4. Identify Outliers
    # Outliers: Data points below the lower bound or above the upper bound
    is_outlier = (df[power_key] < lower_bound) | (df[power_key] > upper_bound)
    outlier_indices = df.index[is_outlier]

    # 5. Identify Normal Data
    # Normal points: Data points within the bounds
    is_normal = ~is_outlier
    normal_indices = df.index[is_normal]

    return normal_indices, outlier_indices