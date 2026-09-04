import pandas as pd
import numpy as np

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

def identify_outliers_iqr(df: pd.DataFrame, power_key: str, iqr_multiplier: float = 1.5):
    """
    Identifies outliers in a specified column of a DataFrame using the Interquartile Range (IQR) method.

    Outliers are defined as data points that are less than Q1 - (IQR * iqr_multiplier) or
    greater than Q3 + (IQR * iqr_multiplier).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        power_key (str): The column name to use for outlier detection (e.g., 'p_mp_ref').
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


def basic_value_filter(df: pd.DataFrame, mapped_variables_dict: dict,
                       irr_min: float = 0.0,
                       irr_max: float = 1500.0,
                       temp_min: float = -40.0,
                       temp_max: float = 100.0,
                       power_min: float = -1.0):
    """
    Removes physically implausible sensor readings before any analysis.

    Applies hard range bounds to irradiance, module temperature, and DC power.
    Any row where at least one variable falls outside its valid range is removed.
    This catches sensor faults (e.g. irradiance=34000 W/m², temperature=1800°C)
    that would otherwise corrupt normalization and smoothness scoring.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe with DatetimeIndex.
    mapped_variables_dict : dict
        Column name mapping, e.g. {'DC Power': ..., 'Irradiance': ..., 'Module temperature': ...}.
    irr_min : float, default 0.0
        Minimum plausible irradiance (W/m²). Negative values are unphysical.
    irr_max : float, default 1500.0
        Maximum plausible POA irradiance (W/m²). Exceeding ~1200 W/m² indicates sensor fault.
    temp_min : float, default -40.0
        Minimum plausible module temperature (°C).
    temp_max : float, default 100.0
        Maximum plausible module temperature (°C). Exceeding ~80°C suggests sensor fault.
    power_min : float, default -1.0
        Minimum plausible DC power (W). Small negatives tolerated for noise; large negatives excluded.

    Returns
    -------
    normal_indices, outlier_indices : pd.Index
    """
    irr_key   = mapped_variables_dict.get('Irradiance')
    temp_key  = mapped_variables_dict.get('Module temperature')
    power_key = mapped_variables_dict.get('DC Power')

    mask = pd.Series(True, index=df.index)

    if irr_key and irr_key in df.columns:
        mask &= df[irr_key].between(irr_min, irr_max)
        print(f"  Irradiance [{irr_min}, {irr_max}] W/m²    : removed {(~df[irr_key].between(irr_min, irr_max)).sum()} pts")

    if temp_key and temp_key in df.columns:
        mask &= df[temp_key].between(temp_min, temp_max)
        print(f"  Temperature [{temp_min}, {temp_max}] °C  : removed {(~df[temp_key].between(temp_min, temp_max)).sum()} pts")

    if power_key and power_key in df.columns:
        mask &= df[power_key] >= power_min
        print(f"  Power >= {power_min} W               : removed {(df[power_key] < power_min).sum()} pts")

    normal_indices  = df.index[mask]
    outlier_indices = df.index[~mask]
    return normal_indices, outlier_indices


def detect_clear_days(df: pd.DataFrame, irradiance_key: str,
                      smoothness_threshold: float = 0.3,
                      energy_threshold: float = 0.5,
                      window_days: int = 30):
    """
    Simplified clear-day detection based on Meyers et al. (Solar Data Tools, IEEE PVSC 2020).

    Resolution-aware: automatically detects whether data is sub-daily or
    already downsampled/daily-aggregated and adjusts the algorithm accordingly.

    Sub-daily mode (>=4 readings/day on average):
      1. Smoothness score: 1 - normalized L1-norm of the 2nd-order temporal
         difference of the intraday irradiance signal.
      2. Energy score: daily irradiance sum divided by a rolling 90th-percentile
         seasonal baseline (±window_days).
      Clear if BOTH scores >= thresholds (AND rule).

    Low-resolution / daily mode (<4 readings/day):
      Energy-only: keeps days where seasonally-normalized irradiance exceeds
      energy_threshold. Smoothness is skipped.

    Key implementation detail: only dates that actually have data in df are
    scored. Dates with no readings are skipped entirely and never marked clear,
    avoiding the problem where empty-day energy = 0 corrupts the seasonal
    baseline or produces false exclusions.
    """
    # ── Clip irradiance to a physically plausible range ─────────────────────
    # Sensor faults (e.g. irr=34000 W/m², temp=1800°C) create enormous
    # 2nd-order differences that dominate max_smooth and crush all good-day
    # scores to ~0. Clip to realistic PV range before scoring.
    IRR_MAX_PHYSICAL = 1500.0   # W/m²  (upper bound for POA irradiance)
    irr = df[irradiance_key].copy().clip(lower=0, upper=IRR_MAX_PHYSICAL)

    # ── Only operate on dates that actually have data ─────────────────────────
    date_groups = {}
    for ts, val in irr.items():
        d = ts.date()
        date_groups.setdefault(d, []).append(val)

    unique_dates = np.array(sorted(date_groups.keys()), dtype=object)
    n_days = len(unique_dates)

    if n_days == 0:
        return unique_dates, np.array([], dtype=bool), "no data"

    # ── Detect temporal resolution (avg readings per day with data) ───────────
    pts_per_day = len(df) / n_days
    use_smoothness = pts_per_day >= 4

    smoothness_raw = np.full(n_days, np.nan)
    energy_raw = np.zeros(n_days)

    for i, d in enumerate(unique_dates):
        day_vals = np.array(date_groups[d], dtype=float)
        energy_raw[i] = np.nansum(day_vals)
        if use_smoothness and len(day_vals) >= 3:
            d2 = day_vals[:-2] - 2 * day_vals[1:-1] + day_vals[2:]
            smoothness_raw[i] = np.sum(np.abs(d2))

    # ── Seasonally normalize energy via rolling 90th-percentile ──────────────
    energy_series = pd.Series(energy_raw, index=pd.to_datetime(unique_dates))
    win = max(2 * window_days + 1, 5)
    rolling_90 = (
        energy_series
        .rolling(window=win, center=True, min_periods=3)
        .quantile(0.90)
    )
    rolling_90 = rolling_90.ffill().bfill().replace(0, np.nan)
    energy_score = np.clip(energy_series.values / rolling_90.values, 0, 1)

    if use_smoothness:
        # ── Normalize smoothness using 95th percentile (robust to outlier days) ──
        valid_smooth = smoothness_raw[~np.isnan(smoothness_raw)]
        ref_smooth = np.percentile(valid_smooth, 95) if len(valid_smooth) > 0 else 1.0
        if ref_smooth == 0:
            ref_smooth = 1.0
        smoothness_score = np.clip(1.0 - smoothness_raw / ref_smooth, 0, 1)
        smoothness_score = np.where(np.isnan(smoothness_raw), 0.0, smoothness_score)

        # ── Adapt thresholds to dataset if fixed ones yield 0 clear days ─────
        # Some datasets (hourly, noisy) have low absolute smoothness scores
        # even on genuinely clear days. If no days pass the fixed thresholds,
        # fall back to selecting the top fraction of days by each score,
        # equivalent to a per-dataset percentile-based threshold.
        joint = (smoothness_score >= smoothness_threshold) & (energy_score >= energy_threshold)
        if joint.sum() == 0:
            # Use top 30% by smoothness AND top 40% by energy as adaptive fallback
            smooth_adaptive = np.percentile(smoothness_score, 70)
            energy_adaptive = np.percentile(energy_score, 60)
            joint = (smoothness_score >= smooth_adaptive) & (energy_score >= energy_adaptive)
            mode = f"sub-daily (adaptive: smooth>={smooth_adaptive:.2f}, energy>={energy_adaptive:.2f})"
        else:
            mode = "sub-daily"

        clear_day_mask = joint
    else:
        clear_day_mask = energy_score >= energy_threshold
        mode = "daily (energy-only)"

    return unique_dates, clear_day_mask, mode


def clear_sky_filter(df: pd.DataFrame, irradiance_key: str,
                     smoothness_threshold: float = 0.3,
                     energy_threshold: float = 0.5,
                     window_days: int = 30):
    """
    Returns normal_indices and outlier_indices based on clear-day detection.
    Points on clear days are kept; all others are excluded.

    Automatically adapts to data resolution — see detect_clear_days for details.
    """
    unique_dates, clear_mask, mode = detect_clear_days(
        df, irradiance_key, smoothness_threshold, energy_threshold, window_days
    )
    print(f"[clear_sky_filter] mode={mode}, clear_days={clear_mask.sum()}/{len(clear_mask)}")

    # unique_dates are Python date objects
    clear_dates = set(d for d, c in zip(unique_dates, clear_mask) if c)

    is_clear = pd.Series(
        [ts.date() in clear_dates for ts in df.index],
        index=df.index
    )

    normal_indices = df.index[is_clear]
    outlier_indices = df.index[~is_clear]
    return normal_indices, outlier_indices