"""
analysis_utils_pkg.py
=====================

Package-backed replacements for the hand-written analysis functions used by
``pages/pvcopilot.py``.  Every public function here keeps the SAME signature
and the SAME return shape as its counterpart in ``analysis_utils.py`` /
``pvcopilot_filter_functions.py``, so switching the page over is an import
swap, nothing more.

Tier 0  (pvlib, already installed)
    compute_pvpro  -> single-diode model via pvlib.pvsystem.calcparams_desoto
                      + pvlib.singlediode.bishop88_mpp / _v_from_i / _i_from_v

Tier 1  (rdtools 2.1.8)
    normalize            -> rdtools.normalization.pvwatts_dc_power
    basic_value_filter   -> rdtools.filtering.poa_filter + tcell_filter
    low_irra_power_filter-> rdtools.filtering.normalized_filter (norm range)
    aggregate_daily      -> rdtools.aggregation.aggregation_insol
    compute_yoy          -> rdtools.degradation.degradation_year_on_year
    compute_lr           -> rdtools.degradation.degradation_ols
    compute_csd          -> rdtools.degradation.degradation_classical_decomposition

Tier 2  (pvanalytics 0.2.2, optional -- functions fall back to pandas
         when pvanalytics is not importable)
    identify_outliers_iqr-> pvanalytics.quality.outliers.tukey
    auto_fix_timezone    -> pvanalytics.features.daytime + quality.time.shifts_ruptures
                            (shifts_ruptures needs the optional `ruptures` package)

Install (see requirements_pkg.txt):
    pip install xgboost-cpu h5py matplotlib          # rdtools runtime deps, ~40 MB
    pip install --no-deps rdtools==2.1.8             # skip the 190-450 MB GPU xgboost wheel
    pip install pvanalytics==0.2.2 ruptures          # tier 2 (optional)

Behavioural differences vs. the hand-written versions are documented in the
docstring of each function under "Differences".
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Tier 0: pvlib (already a dependency of the app) ------------------------
import pvlib
from pvlib.pvsystem import calcparams_desoto
from pvlib.singlediode import bishop88_mpp, bishop88_v_from_i, bishop88_i_from_v
from scipy.optimize import minimize as _scipy_minimize

# --- Tier 1: rdtools --------------------------------------------------------
import rdtools
from rdtools.normalization import pvwatts_dc_power
from rdtools.filtering import poa_filter, tcell_filter, normalized_filter
from rdtools.aggregation import aggregation_insol
from rdtools.degradation import (degradation_year_on_year,
                                 degradation_ols,
                                 degradation_classical_decomposition)

# --- Tier 2: pvanalytics (optional) ----------------------------------------
try:
    from pvanalytics.quality.outliers import tukey as _pva_tukey
    from pvanalytics.features import daytime as _pva_daytime
    _HAS_PVANALYTICS = True
except ImportError:          # pragma: no cover
    _HAS_PVANALYTICS = False
try:
    from pvanalytics.quality.time import shifts_ruptures as _pva_shifts_ruptures
    import ruptures  # noqa: F401  (optional dependency of shifts_ruptures)
    _HAS_RUPTURES = True
except ImportError:          # pragma: no cover
    _HAS_RUPTURES = False


PACKAGE_VERSIONS = {
    "pvlib": pvlib.__version__,
    "rdtools": rdtools.__version__,
    "pvanalytics": (__import__("pvanalytics").__version__ if _HAS_PVANALYTICS else None),
}

# Same palette as analysis_utils.VAR_COLORS (duplicated so this module does
# not import analysis_utils, which instantiates an OpenAI client on import).
VAR_COLORS = {
    "power":       "#0064AB",
    "irradiance":  "#5b9bd5",
    "temperature": "#8ec4e8",
    "voltage":     "#2a8e7a",
    "current":     "#9bcc4e",
}

_EPS = 1e-9   # used to turn rdtools' strict inequalities into inclusive ones


def _time_to_years(index):
    return (index - index[0]).days / 365.25


def _numeric(s):
    return s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")


# =============================================================================
# TIER 1 -- rdtools-backed filters / normalisation / aggregation
# =============================================================================
def normalize(df, mapped_variables_dict, gamma=-0.004):
    """PVWatts-style normalisation via ``rdtools.normalization.pvwatts_dc_power``.

    norm = P_dc / P_expected, with
    P_expected = pvwatts_dc_power(G, power_dc_rated=1, T_cell, gamma_pdc=gamma)
               = G/1000 * (1 + gamma*(T-25))
    which is algebraically identical to the hand-written
    ``P / (G * (1 + gamma*(T-25))) * 1000``.

    Differences: none in the numbers.  Module temperature is optional exactly
    as before (rdtools skips the temperature term when temperature_cell=None).
    Rows with G < 50 W/m² are set to NaN, as before.
    """
    if not mapped_variables_dict.get("Irradiance"):
        raise KeyError("normalize() requires 'Irradiance' to be mapped")
    if not mapped_variables_dict.get("DC Power"):
        raise KeyError("normalize() requires 'DC Power' to be mapped")

    irr_key = mapped_variables_dict["Irradiance"]
    power_key = mapped_variables_dict["DC Power"]
    temp_key = mapped_variables_dict.get("Module temperature")

    irr = _numeric(df[irr_key])
    power = _numeric(df[power_key])
    df[irr_key] = irr
    df[power_key] = power

    temp_cell = None
    if temp_key and temp_key in df.columns:
        temp_cell = _numeric(df[temp_key])
        df[temp_key] = temp_cell

    # power_dc_rated=1 -> P_expected = G/1000 * (1 + gamma*(T-25)), so
    # P / P_expected == P / (G * (1 + gamma*(T-25))) * 1000  (the old formula).
    p_expected = pvwatts_dc_power(irr, 1.0,
                                  temperature_cell=temp_cell,
                                  gamma_pdc=gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["norm"] = power / p_expected
    df.loc[irr < 50, "norm"] = np.nan
    return df


def basic_value_filter(df, mapped_variables_dict,
                       irr_min=0.0, irr_max=1500.0,
                       temp_min=-40.0, temp_max=100.0,
                       power_min=-1.0):
    """Physical-range filter via ``rdtools.filtering.poa_filter`` and
    ``rdtools.filtering.tcell_filter``.

    Differences: rdtools uses strict inequalities (low < x < high); the
    hand-written version used inclusive bounds.  A tiny epsilon is applied
    to the bounds so that behaviour stays inclusive (irradiance == 0 at
    night is kept, as before).  The DC-power lower bound has no rdtools
    equivalent and is a plain pandas comparison.
    Returns (normal_indices, outlier_indices).
    """
    irr_key = mapped_variables_dict.get("Irradiance")
    temp_key = mapped_variables_dict.get("Module temperature")
    power_key = mapped_variables_dict.get("DC Power")

    mask = pd.Series(True, index=df.index)

    if irr_key and irr_key in df.columns:
        ok = poa_filter(_numeric(df[irr_key]), irr_min - _EPS, irr_max + _EPS)
        mask &= ok.fillna(False)
    if temp_key and temp_key in df.columns:
        ok = tcell_filter(_numeric(df[temp_key]), temp_min - _EPS, temp_max + _EPS)
        mask &= ok.fillna(False)
    if power_key and power_key in df.columns:
        mask &= (_numeric(df[power_key]) >= power_min).fillna(False)

    return df.index[mask], df.index[~mask]


def low_irra_power_filter(df, mapped_variables_dict,
                          irr_thresh=300, power_ratio=0.02,
                          norm_lower=0.01, norm_upper_pct=99):
    """Low-irradiance / low-power / normalised-range filter.

    The normalised-range part uses ``rdtools.filtering.normalized_filter``
    (the same filter rdtools' TrendAnalysis applies); the irradiance
    threshold uses ``poa_filter`` with an open upper bound.  The
    power > ratio*irradiance check has no package equivalent and stays a
    pandas expression.

    Differences: strict vs inclusive bounds on the norm range (epsilon-
    adjusted, so effectively none).
    """
    irr_key = mapped_variables_dict.get("Irradiance")
    power_key = mapped_variables_dict.get("DC Power")
    if not irr_key:
        raise KeyError("low_irra_power_filter() requires 'Irradiance' to be mapped")
    if not power_key:
        raise KeyError("low_irra_power_filter() requires 'DC Power' to be mapped")

    irr = _numeric(df[irr_key])
    power = _numeric(df[power_key])

    mask = poa_filter(irr, irr_thresh, np.inf).fillna(False)
    mask &= (power > power_ratio * irr).fillna(False)

    upper = df["norm"].quantile(norm_upper_pct / 100)
    mask &= normalized_filter(df["norm"], norm_lower - _EPS, upper + _EPS).fillna(False)

    return df.index[mask], df.index[~mask]


def aggregate_daily(df_f, irradiance_col):
    """Insolation-weighted daily aggregation via ``rdtools.aggregation_insol``.

    Differences: none.  rdtools computes
    sum(norm*insol)/sum(insol) per day, exactly the hand-written formula;
    rows where either value is NaN are dropped first so both numerator and
    denominator see the same rows, and days without data are dropped (rdtools
    would return NaN for them).  With regular sampling, weighting by
    irradiance is equivalent to weighting by insolation.
    """
    sub = df_f[["norm", irradiance_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    daily = aggregation_insol(sub["norm"], sub[irradiance_col], frequency="D")
    daily = daily.dropna()
    daily.index = pd.to_datetime(daily.index).tz_localize(None) \
        if getattr(daily.index, "tz", None) is not None else pd.to_datetime(daily.index)
    daily.name = None
    return daily


# -----------------------------------------------------------------------------
# shared plotting helper (same look as the hand-written compute_* figures)
# -----------------------------------------------------------------------------
def _trend_figure(series, trend_x, trend_y, trend_label, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="markers",
                             marker=dict(size=8, opacity=0.7, color="#A6CAEC"),
                             name="Daily-aggragated Power"))
    if trend_x is not None:
        fig.add_trace(go.Scatter(x=trend_x, y=trend_y, mode="lines",
                                 line=dict(color="#0070C0", width=2),
                                 name=trend_label))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Power (W)",
                      template="plotly_white", height=350,
                      margin=dict(l=40, r=20, t=50, b=40),
                      legend=dict(orientation="h", yanchor="top", y=-0.2,
                                  xanchor="center", x=0.5))
    return fig


# =============================================================================
# TIER 1 -- rdtools-backed degradation metrics
# =============================================================================
def compute_yoy(series, eps=1e-6, rolling_window=30, iqr_multiplier=1.5,
                confidence_level=68.2, uncertainty_method="simple",
                return_ci=False):
    """Year-on-year degradation via ``rdtools.degradation_year_on_year``.

    Returns (rd, fig) like before; with ``return_ci=True`` returns
    (rd, fig, ci) where ci is the rdtools confidence interval [lo, hi] in %/yr.

    Differences (all improvements):
      * pairs are matched with an 8-day tolerance instead of requiring the
        exact calendar date one year earlier to exist in the index;
      * the rate is the median of per-pair slopes relative to the first-year
        median (rdtools ``recenter=True``) rather than the IQR-trimmed median
        of raw ratios;
      * a bootstrap confidence interval is available;
      * rdtools raises if the series spans < 2 years -> we return NaN like
        the old code did when it found no pairs.
    ``eps`` and ``iqr_multiplier`` are accepted for signature compatibility
    and ignored.
    """
    series = series.dropna().sort_index()
    rd, ci = np.nan, np.array([np.nan, np.nan])
    if len(series) >= 2:
        kw = dict(recenter=True, confidence_level=confidence_level)
        # `uncertainty_method` only exists in rdtools >= 3 (2.x always bootstraps)
        if "uncertainty_method" in inspect.signature(degradation_year_on_year).parameters:
            kw["uncertainty_method"] = uncertainty_method
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rd, ci, _info = degradation_year_on_year(series, **kw)
            rd = float(rd)
            ci = np.asarray(ci, dtype=float)
        except ValueError as e:      # e.g. "must provide at least two years"
            print(f"[compute_yoy] rdtools: {e}")

    trend = series.rolling(rolling_window, center=True).mean()
    label = f"Trend ({rolling_window}-day rolling)"
    if np.isfinite(rd) and np.all(np.isfinite(ci)):
        label += f"  YoY {rd:+.2f} %/yr [{ci[0]:+.2f}, {ci[1]:+.2f}]"
    fig = _trend_figure(series, trend.index, trend, label, "Power Trend")
    return (rd, fig, ci) if return_ci else (rd, fig)


def compute_lr(series, confidence_level=68.2, return_ci=False):
    """Linear-trend degradation via ``rdtools.degradation_ols``.

    Differences: rdtools expresses the slope relative to the OLS intercept
    (year-0 capacity) — ``slope / intercept * 100`` — whereas the hand-
    written version divided by the series mean.  For a few %/yr of
    degradation the two differ by well under 0.1 %/yr.
    """
    series = series.dropna().sort_index()
    if len(series) < 2:
        return (np.nan, None, None) if return_ci else (np.nan, None)
    rd, ci, info = degradation_ols(series, confidence_level=confidence_level)
    rd = float(rd)
    # rdtools uses years = days/365 internally; rebuild its line for the plot
    t365 = (series.index - series.index[0]).days / 365.0
    trend = info["intercept"] + info["slope"] * t365
    fig = _trend_figure(series, series.index, trend,
                        f"LR Trend ({rd:.2f}%/yr)", "Power and Linear Regression Trend")
    return (rd, fig, np.asarray(ci, float)) if return_ci else (rd, fig)


def compute_csd(series, period=12, confidence_level=68.2, return_ci=False):
    """Classical-decomposition degradation via
    ``rdtools.degradation_classical_decomposition``.

    Differences: rdtools isolates the trend with a centred 365-day moving
    average and fits OLS to it (with a Mann-Kendall test), instead of
    ``statsmodels.seasonal_decompose(period=12)`` on daily data.  rdtools
    requires a complete, regular daily series and >= 2 years, so the input is
    reindexed to daily frequency and gaps are linearly interpolated before
    the call (the plotted points are still the real data).  ``period`` is
    accepted for signature compatibility and ignored.
    """
    series = series.dropna().sort_index()
    if len(series) < 2:
        return (np.nan, None, None) if return_ci else (np.nan, None)
    full = series.asfreq("D")
    full = full.interpolate(method="linear", limit_direction="both").asfreq("D")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rd, ci, info = degradation_classical_decomposition(
                full, confidence_level=confidence_level)
        rd = float(rd)
        ci = np.asarray(ci, float)
        trend = info["series"].dropna()
    except ValueError as e:
        print(f"[compute_csd] rdtools: {e}")
        return (np.nan, None, None) if return_ci else (np.nan, None)
    fig = _trend_figure(series, trend.index, trend.values,
                        f"CSD Trend ({rd:.2f}%/yr)", "Power and CSD Trend")
    return (rd, fig, ci) if return_ci else (rd, fig)


# =============================================================================
# TIER 2 -- pvanalytics-backed outlier & time-shift detection (optional)
# =============================================================================
def identify_outliers_iqr(df, power_key, iqr_multiplier=1.5):
    """Tukey/IQR outliers via ``pvanalytics.quality.outliers.tukey``.

    Falls back to the equivalent pandas expression when pvanalytics is not
    installed.  Differences: none (both use Q1 - k*IQR, Q3 + k*IQR and treat
    NaN as non-outliers).  Returns (normal_indices, outlier_indices).
    """
    if power_key not in df.columns:
        print(f"Error: The specified power column '{power_key}' does not exist in the DataFrame.")
        return pd.Index([]), pd.Index([])
    data = _numeric(df[power_key])
    if _HAS_PVANALYTICS:
        is_outlier = _pva_tukey(data, k=iqr_multiplier)
    else:
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        is_outlier = (data < q1 - iqr_multiplier * iqr) | (data > q3 + iqr_multiplier * iqr)
    is_outlier = is_outlier.fillna(False).astype(bool)
    return df.index[~is_outlier], df.index[is_outlier]


def detect_time_shifts(df, power_key, period_min=60, shift_min=60):
    """Per-day clock shifts (DST / logger jumps) from the power signal alone,
    via ``pvanalytics.features.daytime`` (sunrise/sunset from power) and
    ``pvanalytics.quality.time.shifts_ruptures`` (change-point detection).

    The reference is the dataset's own median mid-day time, so the result is
    the shift RELATIVE to the majority of the record — no site location is
    needed.  Returns a Series of shift minutes per day (0 = no shift), or None
    if pvanalytics/ruptures are unavailable or the data is too short.
    Sign convention (rdtools/pvanalytics): ADD the returned minutes to the
    timestamps of that day to align them with the reference.
    """
    if not (_HAS_PVANALYTICS and _HAS_RUPTURES):
        return None
    power = _numeric(df[power_key]).clip(lower=0).sort_index()
    power = power[~power.index.duplicated()]
    # pvanalytics needs the sampling frequency; gappy indices have none inferred
    step = power.index.to_series().diff().median()
    freq = pd.tseries.frequencies.to_offset(step) if pd.notna(step) else None
    try:
        daytime = _pva_daytime.power_or_irradiance(power, freq=freq)
        sunrise = _pva_daytime.get_sunrise(daytime, freq=freq)
        sunset = _pva_daytime.get_sunset(daytime, freq=freq)
    except Exception as e:
        print(f"[detect_time_shifts] daytime detection failed: {e}")
        return None
    mid = (sunrise + (sunset - sunrise) / 2)
    mid = mid.dropna()
    if mid.empty:
        return None
    mid_min = (mid.dt.hour * 60 + mid.dt.minute).groupby(mid.index.date).first()
    mid_min.index = pd.to_datetime(mid_min.index)
    mid_min = mid_min.asfreq("D").interpolate(limit_direction="both").round().astype(int)
    if len(mid_min) <= period_min:
        return None
    reference = pd.Series(int(mid_min.median()), index=mid_min.index)
    try:
        _shifted, shift_amount = _pva_shifts_ruptures(
            mid_min, reference, period_min=period_min, shift_min=shift_min)
    except Exception as e:
        print(f"[detect_time_shifts] shifts_ruptures failed: {e}")
        return None
    return shift_amount


def auto_fix_timezone(df, time_key, power_key, target_tz="local"):
    """Drop-in for ``pvcopilot_filter_functions.auto_fix_timezone``.

    1. tz-naive index -> localised as UTC (unchanged);
    2. ±1 h absolute offset from the average daily solar-peak hour (unchanged
       heuristic — no package can do this without site coordinates);
    3. DST / logger jumps detected with pvanalytics ``shifts_ruptures`` and
       corrected per day (replaces the crude "hour 2 missing / duplicated"
       check).  Falls back to no jump correction if pvanalytics or ruptures
       are not installed.
    Returns (df_fixed, message).
    """
    df = df.copy()
    messages = []
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a pandas.DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        messages.append("Index was tz-naive → localized as UTC.")

    daily_peak = df.groupby(df.index.date)[power_key].idxmax().dropna()
    avg_peak_hour = np.mean([t.hour for t in daily_peak]) if len(daily_peak) else 12
    if avg_peak_hour < 9:
        df.index = df.index + pd.Timedelta(hours=1)
        messages.append("Detected early solar peak → applied +1 hour correction.")
    elif avg_peak_hour > 15:
        df.index = df.index - pd.Timedelta(hours=1)
        messages.append("Detected late solar peak → applied -1 hour correction.")
    else:
        messages.append("Solar peak time appears normal → no ±1 hour correction applied.")

    shifts = detect_time_shifts(df.tz_localize(None) if df.index.tz else df, power_key)
    if shifts is None:
        messages.append("Time-shift detection skipped (pvanalytics/ruptures not available or too little data).")
    else:
        nz = shifts[shifts != 0]
        if nz.empty:
            messages.append("No DST / clock jumps detected (pvanalytics shifts_ruptures).")
        else:
            day_shift = shifts.reindex(pd.to_datetime(df.index.date)).fillna(0).to_numpy()
            df.index = df.index + pd.to_timedelta(day_shift, unit="m")
            n_days = int((nz != 0).sum())
            messages.append(f"Corrected clock shifts on {n_days} days "
                            f"(pvanalytics shifts_ruptures; amounts: "
                            f"{sorted(set(int(v) for v in nz.unique()))} min).")

    if target_tz != "local":
        df.index = df.index.tz_convert(target_tz)
        messages.append(f"Converted index timezone to {target_tz}.")
    else:
        messages.append("Timezone kept unchanged (target_tz='local').")
    return df, "\n".join(messages)


# =============================================================================
# TIER 0 -- PVPRO-style SDM degradation with pvlib doing the physics
# =============================================================================
_Q_E, _K_B = 1.602176634e-19, 1.380649e-23
_T_REF_K = 25.0 + 273.15
_G_REF = 1000.0

# Same values as pvpro.modeling.estimate_Eg_dEgdT
_TECH_TABLE = {
    "mono-c-Si":  (1.121, -0.0002677),
    "multi-c-Si": (1.121, -0.0002677),
    "GaAs":       (1.424, -0.000433),
    "CIGS":       (1.15,  -0.00001),
    "CdTe":       (1.475, -0.0003),
}

_FIT_PARAMS = ("photocurrent_ref", "saturation_current_ref",
               "resistance_series_ref", "resistance_shunt_ref", "diode_factor")


def _estimate_Eg_dEgdT(technology):
    if technology not in _TECH_TABLE:
        raise ValueError(f"Unknown technology '{technology}'. Valid: {sorted(_TECH_TABLE)}.")
    return _TECH_TABLE[technology]


def _sdm_at_conditions(params, G, T, cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """Reference SDM params -> (IL, I0, Rs, Rsh, nNsVth) at (G, T) using
    ``pvlib.pvsystem.calcparams_desoto`` (replaces _calcparams_desoto_lite)."""
    a_ref = params["diode_factor"] * cells_in_series * _K_B * _T_REF_K / _Q_E
    return calcparams_desoto(
        effective_irradiance=G, temp_cell=T,
        alpha_sc=alpha_isc, a_ref=a_ref,
        I_L_ref=params["photocurrent_ref"],
        I_o_ref=params["saturation_current_ref"],
        R_sh_ref=params["resistance_shunt_ref"],
        R_s=params["resistance_series_ref"],
        EgRef=Eg_ref, dEgdT=dEgdT,
        irrad_ref=_G_REF, temp_ref=25.0)


def _predict_mpp(params, G, T, cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """(V_mp, I_mp) for a whole window via ``pvlib.singlediode.bishop88_mpp``
    (replaces _mpp_vectorised)."""
    IL, I0, Rs, Rsh, nNsVth = _sdm_at_conditions(params, G, T, cells_in_series,
                                                 alpha_isc, Eg_ref, dEgdT)
    with np.errstate(all="ignore"):
        i_mp, v_mp, _p_mp = bishop88_mpp(IL, I0, Rs, Rsh, nNsVth, method="newton")
    return np.asarray(v_mp, float), np.asarray(i_mp, float)


def _stc_points(params, cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """P_mp, V_mp, I_mp, V_oc, I_sc at STC via pvlib bishop88 functions
    (replaces _mpp/_voc/_isc_vectorised)."""
    IL, I0, Rs, Rsh, nNsVth = _sdm_at_conditions(
        params, np.array([_G_REF]), np.array([25.0]),
        cells_in_series, alpha_isc, Eg_ref, dEgdT)
    with np.errstate(all="ignore"):
        i_mp, v_mp, p_mp = bishop88_mpp(IL, I0, Rs, Rsh, nNsVth, method="newton")
        v_oc = bishop88_v_from_i(0.0, IL, I0, Rs, Rsh, nNsVth, method="newton")
        i_sc = bishop88_i_from_v(0.0, IL, I0, Rs, Rsh, nNsVth, method="newton")
    return (float(np.ravel(p_mp)[0]), float(np.ravel(v_mp)[0]), float(np.ravel(i_mp)[0]),
            float(np.ravel(v_oc)[0]), float(np.ravel(i_sc)[0]))


# numerical parameter transforms (same as pvpro / the lite version)
def _p_to_x(p, key):
    if key == "saturation_current_ref":
        return np.log(p) + 23.0
    if key == "resistance_shunt_ref":
        return np.log(p) / 2.0 + 1.0
    if key == "resistance_series_ref":
        return p * 2.2
    return p


def _x_to_p(x, key):
    if key == "saturation_current_ref":
        return float(np.exp(x - 23.0))
    if key == "resistance_shunt_ref":
        return float(np.exp(2.0 * (x - 1.0)))
    if key == "resistance_series_ref":
        return float(x / 2.2)
    return float(x)


def _loss(x, G, T, V_meas, I_meas, V_scale, I_scale,
          cells_in_series, alpha_isc, Eg_ref, dEgdT):
    params = {k: _x_to_p(x[i], k) for i, k in enumerate(_FIT_PARAMS)}
    try:
        V_pred, I_pred = _predict_mpp(params, G, T, cells_in_series, alpha_isc, Eg_ref, dEgdT)
    except Exception:
        return 1e6
    v_err = (V_pred - V_meas) / V_scale
    i_err = (I_pred - I_meas) / I_scale
    val = np.nanmean(v_err ** 2 + i_err ** 2)
    return float(val) if np.isfinite(val) else 1e6


def _fit_window(G, T, V_meas, I_meas, p0, lower_bounds, upper_bounds,
                cells_in_series, alpha_isc, Eg_ref, dEgdT,
                saturation_current_multistart=(0.2, 0.5, 1.0, 2.0, 5.0)):
    x_lo = np.array([_p_to_x(lower_bounds[k], k) for k in _FIT_PARAMS])
    x_hi = np.array([_p_to_x(upper_bounds[k], k) for k in _FIT_PARAMS])
    bounds = list(zip(x_lo, x_hi))
    V_scale = max(float(np.nanmedian(V_meas)), 1e-3)
    I_scale = max(float(np.nanmedian(I_meas)), 1e-3)

    best, best_loss = None, np.inf
    for mult in saturation_current_multistart:
        p0_try = dict(p0)
        p0_try["saturation_current_ref"] = p0["saturation_current_ref"] * mult
        x0 = np.clip(np.array([_p_to_x(p0_try[k], k) for k in _FIT_PARAMS]), x_lo, x_hi)
        try:
            res = _scipy_minimize(
                _loss, x0=x0, bounds=bounds, method="L-BFGS-B",
                args=(G, T, V_meas, I_meas, V_scale, I_scale,
                      cells_in_series, alpha_isc, Eg_ref, dEgdT),
                options={"maxiter": 80, "ftol": 1e-7, "disp": False})
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < best_loss:
            best, best_loss = res, res.fun
    if best is None:
        return None
    fit = {k: _x_to_p(best.x[i], k) for i, k in enumerate(_FIT_PARAMS)}
    fit["loss"] = float(best.fun)
    return fit


def _estimate_p0_simple(G, T, V, I, cells_in_series):
    """Same textbook seed as the lite version (top-decile irradiance points)."""
    if len(G) < 10:
        return None
    mask = G >= np.nanquantile(G, 0.9)
    if mask.sum() < 5:
        mask = np.ones_like(G, dtype=bool)
    Vmp_med, Imp_med, Gmed = (float(np.nanmedian(V[mask])), float(np.nanmedian(I[mask])),
                              float(np.nanmedian(G[mask])))
    if Vmp_med <= 0 or Imp_med <= 0 or Gmed <= 0:
        return None
    IL_ref_guess = Imp_med * (_G_REF / Gmed)
    n_guess = 1.03
    nNsVth_ref = n_guess * cells_in_series * (_K_B * _T_REF_K / _Q_E)
    I0_ref_guess = max(IL_ref_guess * np.exp(-Vmp_med / nNsVth_ref), 1e-13)
    return dict(photocurrent_ref=float(np.clip(IL_ref_guess, 0.1, 20.0)),
                saturation_current_ref=float(np.clip(I0_ref_guess, 1e-13, 1e-5)),
                resistance_series_ref=0.4, resistance_shunt_ref=600.0,
                diode_factor=n_guess)


def compute_pvpro(df, mapped_variables_dict,
                  cells_in_series=60, modules_per_string=1, parallel_strings=1,
                  alpha_isc=0.0046, technology="mono-c-Si",
                  days_per_run=14, iterations_per_year=12,
                  resistance_shunt_ref=600.0, delta_T=3.0,
                  irradiance_threshold=200.0, min_points_per_window=20,
                  progress_callback=None):
    """PVPRO-style windowed single-diode-model fit.

    Identical algorithm and return values (rd, figs, rates) to
    ``analysis_utils.compute_pvpro``; the single-diode physics is now done by
    pvlib:
      * De Soto translation  -> ``pvlib.pvsystem.calcparams_desoto``
      * MPP / Voc / Isc      -> ``pvlib.singlediode.bishop88_mpp``,
                                ``bishop88_v_from_i(0,…)``, ``bishop88_i_from_v(0,…)``
    Differences: the lite version added a fixed 1e-5 S extra shunt
    conductance for numerical safety; pvlib uses the exact De Soto
    R_sh = R_sh_ref * G_ref / G.  Effect on fitted P_mp,ref is negligible at
    the irradiance levels kept (> irradiance_threshold).
    Trend extraction (IQR-trimmed linear fit per quantity) is unchanged.
    """
    def _report(stage, current=0, total=1, message=""):
        if progress_callback is None:
            return
        try:
            progress_callback(stage, current, total, message)
        except Exception:
            pass

    _report("prepare", 0, 1, "Validating inputs and pulling V/I/G/T columns")
    required = ["DC Voltage", "DC Current", "Irradiance", "Module temperature"]
    missing = [r for r in required
               if mapped_variables_dict.get(r) is None or mapped_variables_dict[r] not in df.columns]
    if missing:
        raise ValueError("PVPRO requires the following columns to be identified in Step 1: "
                         + ", ".join(missing) + ". They were not found in the dataset.")
    v_key, i_key = mapped_variables_dict["DC Voltage"], mapped_variables_dict["DC Current"]
    irr_key, tm_key = mapped_variables_dict["Irradiance"], mapped_variables_dict["Module temperature"]

    df_p = df[[v_key, i_key, irr_key, tm_key]].copy()
    for c in (v_key, i_key, irr_key, tm_key):
        df_p[c] = _numeric(df_p[c])
    df_p.index = pd.to_datetime(df_p.index)
    df_p = df_p.dropna()
    df_p = df_p[df_p[irr_key] > irradiance_threshold]
    df_p = df_p[(df_p[v_key] > 0) & (df_p[i_key] > 0)]
    if len(df_p) < 100:
        raise ValueError(f"After dropping NaNs and points with irradiance ≤ {irradiance_threshold} "
                         f"W/m², only {len(df_p)} rows remain — too few for PVPRO. "
                         "Loosen the Step 2 filters or supply a longer dataset.")

    V_arr = df_p[v_key].to_numpy(float) / max(modules_per_string, 1)
    I_arr = df_p[i_key].to_numpy(float) / max(parallel_strings, 1)
    G_arr = df_p[irr_key].to_numpy(float)
    Tc_arr = df_p[tm_key].to_numpy(float) + delta_T

    Eg_ref, dEgdT = _estimate_Eg_dEgdT(technology)
    lower_bounds = dict(photocurrent_ref=0.01, saturation_current_ref=1e-13,
                        resistance_series_ref=0.0, resistance_shunt_ref=10.0, diode_factor=0.5)
    upper_bounds = dict(photocurrent_ref=20.0, saturation_current_ref=1e-5,
                        resistance_series_ref=1.0, resistance_shunt_ref=5000.0, diode_factor=2.0)

    _report("p0", 0, 1, "Estimating starting parameters from top-decile irradiance")
    p0_global = _estimate_p0_simple(G_arr, Tc_arr, V_arr, I_arr, cells_in_series)
    if p0_global is None:
        raise ValueError("Could not derive a starting point for the SDM fit. "
                         "Dataset may have too few high-irradiance points.")
    p0_global["resistance_shunt_ref"] = float(resistance_shunt_ref)

    t_start_all, t_end_all = df_p.index.min(), df_p.index.max()
    step_days = max(int(round(365.25 / max(iterations_per_year, 1))), 1)
    window_starts = []
    cur = t_start_all
    while cur + pd.Timedelta(days=days_per_run) <= t_end_all + pd.Timedelta(days=1):
        window_starts.append(cur)
        cur = cur + pd.Timedelta(days=step_days)
    n_total = len(window_starts)

    rows = []
    p0_warm = dict(p0_global)
    for w_idx, cur in enumerate(window_starts):
        _report("fitting", w_idx, n_total,
                f"Fitting window {w_idx + 1} / {n_total} ({cur.strftime('%Y-%m-%d')})")
        idx_w = (df_p.index >= cur) & (df_p.index < cur + pd.Timedelta(days=days_per_run))
        n_w = int(idx_w.sum())
        if n_w >= min_points_per_window:
            fit = _fit_window(G_arr[idx_w], Tc_arr[idx_w], V_arr[idx_w], I_arr[idx_w],
                              p0_warm, lower_bounds, upper_bounds,
                              cells_in_series, alpha_isc, Eg_ref, dEgdT)
            if fit is not None:
                p0_warm = {k: fit[k] for k in _FIT_PARAMS}
                p_mp, v_mp, i_mp, v_oc, i_sc = _stc_points(fit, cells_in_series,
                                                           alpha_isc, Eg_ref, dEgdT)
                rows.append({"t_mid": cur + pd.Timedelta(days=days_per_run / 2),
                             "p_mp_ref": p_mp, "v_mp_ref": v_mp, "i_mp_ref": i_mp,
                             "v_oc_ref": v_oc, "i_sc_ref": i_sc,
                             "loss": fit["loss"], "n_points": n_w,
                             **{k: fit[k] for k in _FIT_PARAMS}})
            else:
                p0_warm = dict(p0_global)

    if len(rows) < 4:
        raise ValueError(f"PVPRO produced only {len(rows)} successful window fits. "
                         "Need at least 4. Try a longer dataset or fewer/looser filters.")
    pfit = pd.DataFrame(rows).set_index("t_mid").sort_index()

    # ---- trend extraction (unchanged from the lite version) ----------------
    _report("trend", 0, 1, "Computing degradation rates")
    from sklearn.linear_model import LinearRegression as _LR

    def _linear_rate(series):
        s = series.dropna()
        if len(s) < 4:
            return np.nan, s
        q1, q3 = np.nanpercentile(s.values, [25, 75])
        iqr = q3 - q1
        keep = (s.values >= q1 - 1.5 * iqr) & (s.values <= q3 + 1.5 * iqr)
        s_clean = s.loc[keep] if keep.sum() >= 4 else s
        t_years = _time_to_years(s_clean.index).values.reshape(-1, 1)
        lr = _LR().fit(t_years, s_clean.values)
        med = float(np.nanmedian(s_clean.values))
        if med == 0 or not np.isfinite(med):
            return np.nan, s_clean
        return float(lr.coef_[0] / med * 100.0), s_clean

    quantities = [("p_mp_ref", "<b>Pmp</b> (ref)", "W", "Pmp"),
                  ("v_mp_ref", "<b>Vmp</b> (ref)", "V", "Vmp"),
                  ("i_mp_ref", "<b>Imp</b> (ref)", "A", "Imp"),
                  ("v_oc_ref", "<b>Voc</b> (ref)", "V", "Voc"),
                  ("i_sc_ref", "<b>Isc</b> (ref)", "A", "Isc")]
    rates, cleaned = {}, {}
    for col, _, _, _ in quantities:
        rates[col], cleaned[col] = _linear_rate(pfit[col])
    rd = rates["p_mp_ref"]
    if not np.isfinite(rd):
        # fallback: rdtools YoY on the per-window P_mp,ref series
        try:
            rd = float(degradation_year_on_year(pfit["p_mp_ref"].dropna())[0])
        except Exception:
            rd = np.nan

    # ---- figures (unchanged look) -----------------------------------------
    color_key = {"p_mp_ref": "power", "v_mp_ref": "voltage", "i_mp_ref": "current",
                 "v_oc_ref": "voltage", "i_sc_ref": "current"}

    def _rgba(hex_str, alpha=0.3):
        h = hex_str.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"

    def _one_panel(col, label, units, short, height):
        series, s_clean, rate = pfit[col].dropna(), cleaned[col], rates[col]
        rate_str = f"({rate:+.2f} %/yr)" if np.isfinite(rate) else "(n/a)"
        trend_color = VAR_COLORS[color_key[col]]
        fig = go.Figure()
        if len(series):
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="markers",
                                     marker=dict(size=8, color=_rgba(trend_color), line=dict(width=0)),
                                     showlegend=False,
                                     hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3g} " + units + "<extra></extra>"))
            if len(s_clean) >= 2 and np.isfinite(rate):
                t_years_arr = _time_to_years(s_clean.index).values
                med = float(np.nanmedian(s_clean.values))
                trend = med + rate / 100.0 * med * (t_years_arr - np.nanmean(t_years_arr))
                fig.add_trace(go.Scatter(x=s_clean.index, y=trend, mode="lines",
                                         line=dict(color=trend_color, width=2.5), showlegend=False))
        fig.update_layout(
            title=dict(text=f"{label} &nbsp;<span style='color:#475569;font-weight:400'>{rate_str}</span>",
                       x=0.5, xanchor="center", y=0.97, yanchor="top",
                       font=dict(size=14, family="Arial", color="#0f172a")),
            template="plotly_white", height=height, margin=dict(l=55, r=14, t=36, b=30),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(title_text=f"{short} ({units})", title_font=dict(size=11),
                         showgrid=True, gridcolor="#e2e8f0", zeroline=False)
        fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
        return fig

    heights = {"p_mp_ref": 210}
    figs = {col: _one_panel(col, label, units, short, heights.get(col, 180))
            for col, label, units, short in quantities}
    # Per-window fit table, attached to the Pmp figure's metadata so callers
    # (tests, CSV export) can reach it without changing the return signature.
    _tbl = pfit.reset_index()
    _tbl["t_mid"] = _tbl["t_mid"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    figs["p_mp_ref"].layout.meta = {"pfit": _tbl.to_dict(orient="list")}

    _report("done", n_total, n_total, "Done")
    return rd, figs, rates


# =============================================================================
# Step 4 -- code export (package-based snippet, no LLM call)
# =============================================================================
_PKG_IMPORTS_CODE = "import pandas as pd\nimport numpy as np"

_PKG_MAIN_CODE = """# ---------------- Main code ----------------
# Basic physical-range filter
normal_idx, _ = basic_value_filter(df, mapped_variables_dict)
df = df.loc[normal_idx].copy()

# Normalize DC power by irradiance (and module temperature if available)
df_filtered = normalize(df, mapped_variables_dict)

# Filters selected in the app
selected_filters = {selected_filters}
normal_idx = df_filtered.index

if 'low-irra-power' in selected_filters:
    idx, _ = low_irra_power_filter(df_filtered, mapped_variables_dict)
    normal_idx = normal_idx.intersection(idx)

if 'outlier' in selected_filters:
    idx, _ = identify_outliers_iqr(df_filtered, 'norm')
    normal_idx = normal_idx.intersection(idx)

print('Total number of points:', len(df_filtered))
print('Number of normal points:', len(normal_idx))
print('Number of outliers:', len(df_filtered) - len(normal_idx))

df_filtered_final = df_filtered.loc[normal_idx]

# Insolation-weighted daily aggregation (rdtools)
daily_data = aggregate_daily(df_filtered_final, mapped_variables_dict['Irradiance'])

# Degradation rate selected in the app (rdtools) -> rate [%/yr], confidence interval
selected_metric = {selected_metric}
if 'YOY' in selected_metric:
    rd, ci = compute_yoy(daily_data)
elif 'LR' in selected_metric:
    rd, ci = compute_lr(daily_data)
elif 'CSD' in selected_metric:
    rd, ci = compute_csd(daily_data)
else:
    raise ValueError('HW / ARIMA are not available in the package-based export; '
                     'use YOY, LR or CSD.')

print(f'Degradation rate: {{rd:+.2f}} %/yr  (68% CI {{ci[0]:+.2f}} .. {{ci[1]:+.2f}})')
"""


def get_full_code(filename, mapped_variables_dict, selected_filters, selected_metric):
    """Same role as ``analysis_utils.get_full_code`` but the emitted script
    calls rdtools / pvanalytics directly (see pvcopilot_functions_code_pkg.txt).
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "pvcopilot_functions_code_pkg.txt"), "r", encoding="utf-8") as f:
        functions_code = f.read().replace('"', "'")

    fname = filename or "data.csv"
    if "csv" in fname:
        code_read = f"df = pd.read_csv('{fname}', index_col=0, parse_dates=True)  # adjust the path if needed"
    elif "xls" in fname:
        code_read = f"df = pd.read_excel('{fname}', index_col=0, parse_dates=True)  # adjust the path if needed"
    elif "parquet" in fname:
        code_read = f"df = pd.read_parquet('{fname}')  # adjust the path if needed"
    else:
        code_read = f"df = pd.read_csv('{fname}', index_col=0, parse_dates=True)  # adjust the path if needed"

    metric = list(selected_metric) if isinstance(selected_metric, (list, tuple, set)) else [selected_metric]
    main_code = _PKG_MAIN_CODE.format(selected_filters=list(selected_filters or []),
                                      selected_metric=metric)
    return "\n\n".join([_PKG_IMPORTS_CODE, functions_code, code_read,
                        f"mapped_variables_dict = {mapped_variables_dict}", main_code])


# =============================================================================
# CLEAR-SKY FILTERING
#
# The published clear-sky detectors (pvlib's Reno detector, rdtools' clear-sky
# index filter, pvanalytics' `features.clearsky.reno`) all compare the measured
# irradiance against a CLEAR-SKY REFERENCE series.  Producing that reference is
# the only part that needs site information, so it is separated out here:
#
#   * `latitude` + `longitude` given -> a real clear-sky model:
#       pvlib.location.Location.get_clearsky (Ineichen-Perez with pvlib's
#       bundled Linke-turbidity climatology) transposed to the array plane
#       with pvlib.irradiance.get_total_irradiance.
#   * no coordinates -> an EMPIRICAL envelope derived from the measurements
#       themselves (a high quantile of each time-of-day over a +/- window_days
#       window), so the filter still works with no user input.
#
# The classification itself is always a package function:
#   * sub-hourly data  -> pvlib.clearsky.detect_clearsky (Reno & Hansen 2016),
#     whose five criteria are defined for 1-minute GHI;
#   * coarser data     -> rdtools.filtering.csi_filter (clear-sky index within
#     +/- csi_threshold of 1), which is resolution independent.  The Reno
#     line-length / slope criteria are calibrated for 1-minute data and reject
#     almost everything on hourly records, so `method="auto"` avoids them
#     there.
# =============================================================================
from rdtools.filtering import csi_filter as _rd_csi_filter
from rdtools.normalization import irradiance_rescale as _rd_irradiance_rescale
from pvlib.clearsky import detect_clearsky as _pvlib_detect_clearsky
from pvlib.location import Location as _PvlibLocation
from pvlib.irradiance import get_total_irradiance as _pvlib_get_total_irradiance
try:
    from pvanalytics.system import infer_orientation_fit_pvwatts as _pva_infer_orientation
except ImportError:          # pragma: no cover
    _pva_infer_orientation = None

_RENO_MAX_INTERVAL_MIN = 15.0   # above this, Reno's criteria are out of calibration


def _median_interval_minutes(index):
    try:
        step = pd.to_datetime(pd.Series(index)).diff().median()
        m = step.total_seconds() / 60.0
        return m if np.isfinite(m) and m > 0 else None
    except Exception:
        return None


def _fixed_offset_tz(longitude):
    """Standard-time zone nearest a longitude, as an Etc/GMT name.

    Logger timestamps are normally local standard time, so when the index is
    tz-naive this is what aligns the modelled clear-sky curve with the data's
    own clock.  Note Etc/GMT signs are inverted (Etc/GMT+7 is UTC-7).
    """
    offset = int(round(float(longitude) / 15.0))
    return f"Etc/GMT{'+' if offset <= 0 else '-'}{abs(offset)}"


def modeled_clear_sky_poa(index, latitude, longitude, tilt=None, azimuth=None,
                          altitude=0.0, model="ineichen", power=None):
    """Clear-sky plane-of-array irradiance for `index`, entirely from pvlib.

    The array geometry is resolved in this order:
      1. `tilt` / `azimuth` as supplied by the caller;
      2. otherwise fitted from `power` (a measured power Series on the same
         index) with ``pvanalytics.system.infer_orientation_fit_pvwatts``,
         which regresses a PVWatts model against the clear-sky components;
      3. otherwise the rule of thumb: tilt = |latitude|, equator-facing.

    Returns ``(poa_global, tilt, azimuth, geometry_source)``.
    """
    idx = pd.to_datetime(index)
    tz = idx.tz if idx.tz is not None else _fixed_offset_tz(longitude)
    loc = _PvlibLocation(float(latitude), float(longitude), tz=tz,
                         altitude=float(altitude or 0.0))
    if idx.tz is None:
        idx_aware = idx.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        idx_aware = idx
    cs = loc.get_clearsky(idx_aware, model=model)
    solpos = loc.get_solarposition(idx_aware)

    source = "as supplied"
    if tilt is None or azimuth is None:
        fitted = _infer_orientation(cs, solpos, power, index)
        if fitted is not None:
            f_tilt, f_az, r2 = fitted
            tilt = f_tilt if tilt is None else tilt
            azimuth = f_az if azimuth is None else azimuth
            source = f"fitted from power with pvanalytics (R2 {r2:.3f})"
        else:
            source = "assumed (tilt = |latitude|, equator-facing)"
    if tilt is None:
        tilt = abs(float(latitude))
    if azimuth is None:
        azimuth = 180.0 if float(latitude) >= 0 else 0.0
    poa = _pvlib_get_total_irradiance(
        surface_tilt=float(tilt), surface_azimuth=float(azimuth),
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=cs["dni"], ghi=cs["ghi"], dhi=cs["dhi"])["poa_global"]
    poa.index = index          # back to the caller's (possibly naive) index
    return poa, float(tilt), float(azimuth), source


def _infer_orientation(cs, solpos, power, index, max_points=6000, min_points=200):
    """Fit surface tilt/azimuth from measured power with pvanalytics.

    Restricted to bright, clear-sky-model hours (the regime the PVWatts fit
    assumes) and subsampled so the fit stays well under a second even on a
    decade of data.  Returns ``(tilt, azimuth, r2)``, or None when it cannot
    be done (no pvanalytics, no power column, too few usable points, or a
    fit that explains too little of the variance to be trusted).
    """
    if _pva_infer_orientation is None or power is None:
        return None
    try:
        pw = pd.Series(power).astype(float)
        pw.index = index
        ok = (cs["ghi"].to_numpy() > 200) & np.isfinite(pw.to_numpy()) & (pw.to_numpy() > 0)
        if int(ok.sum()) < min_points:
            return None
        pos = np.flatnonzero(ok)
        if len(pos) > max_points:                       # even thinning, keeps all seasons
            pos = pos[:: int(np.ceil(len(pos) / max_points))]
        sl = np.zeros(len(pw), dtype=bool)
        sl[pos] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tilt, azimuth, r2 = _pva_infer_orientation(
                pd.Series(pw.to_numpy()[sl], index=cs.index[sl]),
                cs["ghi"][sl], cs["dhi"][sl], cs["dni"][sl],
                solpos["apparent_zenith"][sl], solpos["azimuth"][sl])
        if not np.isfinite(r2) or r2 < 0.5:
            return None
        return float(tilt), float(azimuth), float(r2)
    except Exception as e:
        print("[clear_sky] orientation fit failed (%s: %s)" % (type(e).__name__, e))
        return None


def empirical_clear_sky_envelope(irradiance, window_days=30, quantile=0.90):
    """Clear-sky reference derived from the measurements themselves.

    For each time-of-day slot, take a centred rolling `quantile` over the
    surrounding +/- `window_days` days.  On a site with some clear days in
    every season this tracks the true clear-sky curve closely, and it needs no
    site information at all.

    Limitation: a window with NO clear day at all yields an envelope that sits
    below the real clear-sky curve, which lets mediocre days pass.  Supplying
    coordinates (the modelled reference above) removes that failure mode.
    """
    s = pd.Series(irradiance).astype(float).clip(lower=0)
    step = _median_interval_minutes(s.index) or 60.0
    tod = (s.index.hour * 60 + s.index.minute + s.index.second / 60.0)
    tod = (np.round(tod / step) * step).astype(int)     # snap jittered stamps
    frame = pd.DataFrame({"v": s.to_numpy(), "d": s.index.normalize(), "t": tod})
    piv = frame.pivot_table(index="d", columns="t", values="v", aggfunc="mean")
    if piv.empty:
        return pd.Series(np.nan, index=s.index)
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
    env = (piv.rolling(2 * int(window_days) + 1, center=True, min_periods=3)
              .quantile(float(quantile)).ffill().bfill())
    env = env.T.rolling(3, center=True, min_periods=1).mean().T   # smooth in time-of-day
    flat = env.stack(dropna=False)
    flat.index = (flat.index.get_level_values(0)
                  + pd.to_timedelta(flat.index.get_level_values(1), unit="m"))
    flat = flat[~flat.index.duplicated()]
    return flat.reindex(s.index)


def _peak_hour_offset(measured, reference, min_days=20):
    """Median (measured peak time - reference peak time) per day, in hours.

    A modelled clear-sky curve is only useful if it is on the SAME CLOCK as
    the data, and logger timestamps are routinely offset (DST, a logger left
    on UTC, a wrong longitude).  Comparing the two daily peak times measures
    that offset directly, without needing to know which of those caused it.
    """
    ok = measured.notna() & reference.notna() & (measured > 50)
    if int(ok.sum()) < 10:
        return 0.0
    m, r = measured[ok], reference[ok]
    m_peak = m.groupby(m.index.normalize()).idxmax()
    r_peak = r.groupby(r.index.normalize()).idxmax()
    common = m_peak.index.intersection(r_peak.index)
    if len(common) < min_days:
        return 0.0

    def _hours(ts):
        ts = pd.to_datetime(pd.Series(ts.values))
        return ts.dt.hour + ts.dt.minute / 60.0

    diff = _hours(m_peak.loc[common]).to_numpy() - _hours(r_peak.loc[common]).to_numpy()
    return float(np.median(diff))


def _calibrate_level(measured, reference):
    """Put the modelled reference on the measurement's own scale.

    Array tilt/azimuth, sensor calibration and soiling all shift the measured
    plane-of-array level away from a textbook clear-sky model, and the
    clear-sky index is a RATIO, so that bias would move every csi.
    ``rdtools.normalization.irradiance_rescale`` exists for exactly this (it
    rescales a modelled series to the measured clear-sky periods); a plain
    high-percentile ratio is the fallback when it cannot converge.
    """
    ok = measured.notna() & reference.notna() & (measured > 50) & (reference > 50)
    if int(ok.sum()) < 50:
        return reference, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rescaled = _rd_irradiance_rescale(measured[ok], reference[ok], method="iterative")
        factor = float(np.nanmedian(np.asarray(rescaled) / reference[ok].to_numpy()))
        if np.isfinite(factor) and 0.2 < factor < 5.0:
            return reference * factor, factor
    except Exception as e:
        print("[clear_sky] irradiance_rescale failed (%s: %s); using a percentile ratio"
              % (type(e).__name__, e))
    hi_m = float(np.nanpercentile(measured[ok], 98))
    hi_r = float(np.nanpercentile(reference[ok], 98))
    factor = hi_m / hi_r if hi_r > 0 else 1.0
    if not np.isfinite(factor) or not (0.2 < factor < 5.0):
        factor = 1.0
    return reference * factor, factor


def clear_sky_reference(df, irradiance_key, latitude=None, longitude=None,
                        tilt=None, azimuth=None, altitude=0.0,
                        window_days=30, quantile=0.90, power_key=None):
    """Clear-sky reference for `df[irradiance_key]`, plus a one-line
    description of how it was produced (for the UI).  Uses the modelled path
    when coordinates are supplied and falls back to the empirical envelope —
    including when the model fails for any reason."""
    irr = _numeric(df[irradiance_key])
    if latitude is not None and longitude is not None:
        try:
            power = (_numeric(df[power_key])
                     if power_key and power_key in df.columns else None)
            notes = []
            step_h = (_median_interval_minutes(df.index) or 60.0) / 60.0

            # Pass 1 - a cheap probe with rule-of-thumb geometry, used only to
            # measure how far the record's timestamps sit from true solar time
            # (DST, a logger left on UTC, a clock never set).  Daily peak times
            # are all it needs, so a sample of days is enough.
            probe_index = df.index
            if len(probe_index) > 4000:
                keep_days = pd.Index(probe_index.normalize().unique())[::5]
                sel = probe_index.normalize().isin(keep_days)
                if sel.sum() > 500:
                    probe_index = probe_index[sel]
            probe, _t, _a, _g = modeled_clear_sky_poa(
                probe_index, latitude, longitude, None, None, altitude)
            shift_h = _peak_hour_offset(irr.reindex(probe_index), probe)
            if abs(shift_h) > 3.0:
                notes.append(f"WARNING: modelled solar noon is {shift_h:+.1f} h from the "
                             "measured peak -- check the coordinates and the timestamps")
                shift_h = 0.0
            elif abs(shift_h) < step_h:
                shift_h = 0.0

            # Pass 2 - model each sample at its TRUE solar time, so both the
            # orientation fit and the clear-sky curve land on the record's own
            # clock.  (Fitting the orientation before this correction pulls the
            # azimuth toward the clock error.)
            solar_index = df.index - pd.Timedelta(hours=float(shift_h))
            ref, used_tilt, used_az, geom = modeled_clear_sky_poa(
                solar_index, latitude, longitude, tilt, azimuth, altitude, power=power)
            ref.index = df.index
            if shift_h:
                notes.append(f"timestamps are {shift_h:+.1f} h off solar time; corrected")

            if ref.notna().any() and float(np.nanmax(ref.to_numpy())) > 0:
                ref, factor = _calibrate_level(irr, ref)
                if abs(factor - 1.0) > 0.02:
                    notes.append(f"level calibrated x{factor:.2f}")
                how = ("pvlib Ineichen clear-sky at "
                       f"{float(latitude):.3f}, {float(longitude):.3f} "
                       f"(tilt {used_tilt:.0f}°, azimuth {used_az:.0f}°"
                       f", {geom})"
                       + ("; " + "; ".join(notes) if notes else ""))
                return ref, how
        except Exception as e:
            print(f"[clear_sky_reference] modelled clear-sky failed ({type(e).__name__}: {e});"
                  " falling back to the empirical envelope")
    ref = empirical_clear_sky_envelope(irr, window_days=window_days, quantile=quantile)
    return ref, (f"empirical envelope ({int(quantile * 100)}th percentile over "
                 f"±{int(window_days)} days; no site coordinates given)")


def clear_sky_filter(df, irradiance_key,
                     smoothness_threshold=0.3,      # legacy arg, see Notes
                     energy_threshold=0.5,          # -> day_fraction
                     window_days=30,
                     latitude=None, longitude=None, tilt=None, azimuth=None,
                     altitude=0.0,
                     csi_threshold=0.15, day_fraction=None, method="auto",
                     min_irradiance=50.0, quantile=0.90, power_key=None,
                     return_info=False):
    """Keep the points that fall on CLEAR DAYS — same contract as the
    hand-written ``pvcopilot_filter_functions.clear_sky_filter``: returns
    ``(normal_indices, outlier_indices)`` and keeps every point of a day it
    judges clear.

    How a day is judged has changed.  Instead of the hand-written
    smoothness + energy scores, each DAYTIME sample is classified by a package
    function against a clear-sky reference (see `clear_sky_reference`):

    ``method="csi"``   rdtools.filtering.csi_filter — the measured/clear-sky
                       ratio must be within ``csi_threshold`` of 1.
    ``method="reno"``  pvlib.clearsky.detect_clearsky — the five Reno & Hansen
                       criteria, which are calibrated for 1-minute GHI.
    ``method="auto"``  reno for data sampled at <= 15 min, csi above that.

    A day is clear when at least ``day_fraction`` of its daytime samples
    (irradiance above ``min_irradiance``) are classified clear.

    Notes
    -----
    ``energy_threshold`` keeps its old role — the share of the day that must
    be good — and is used as ``day_fraction`` when that is not given
    explicitly (same 0.5 default, same direction: higher is stricter).
    ``smoothness_threshold`` has no counterpart in the package criteria and is
    accepted only so existing call sites keep working; it is ignored.
    With ``return_info=True`` a third element is returned: a dict with
    ``reference`` (how the reference was built), ``method``, ``n_clear_days``,
    ``n_days`` and ``interval_min``, for display in the UI.
    """
    irr = _numeric(df[irradiance_key]).clip(lower=0, upper=1500)
    day_fraction = float(energy_threshold if day_fraction is None else day_fraction)
    interval = _median_interval_minutes(df.index)

    info = {"reference": None, "method": None, "n_clear_days": 0, "n_days": 0,
            "interval_min": interval}
    empty = (df.index[:0], df.index)
    if irr.notna().sum() == 0:
        info["reference"] = "no usable irradiance"
        return (*empty, info) if return_info else empty

    reference, how = clear_sky_reference(
        df, irradiance_key, latitude=latitude, longitude=longitude,
        tilt=tilt, azimuth=azimuth, altitude=altitude,
        window_days=window_days, quantile=quantile, power_key=power_key)
    info["reference"] = how

    if method == "auto":
        method = "reno" if (interval is not None and interval <= _RENO_MAX_INTERVAL_MIN) else "csi"
    info["method"] = method

    daytime = (irr > min_irradiance).fillna(False)

    if method == "reno":
        # detect_clearsky needs an evenly spaced index; window_length is in
        # minutes and must span more than two samples.
        step = interval or 1.0
        freq = pd.tseries.frequencies.to_offset(pd.Timedelta(minutes=step))
        meas_r = irr.asfreq(freq) if irr.index.freq is None else irr
        ref_r = reference.reindex(meas_r.index)
        window_length = int(max(10, np.ceil(4 * step)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clear_r = _pvlib_detect_clearsky(meas_r.fillna(0.0), ref_r.fillna(0.0),
                                                 window_length=window_length)
            clear = pd.Series(clear_r, index=meas_r.index).reindex(irr.index).fillna(False)
        except Exception as e:
            print(f"[clear_sky_filter] detect_clearsky failed ({type(e).__name__}: {e});"
                  " falling back to the clear-sky-index filter")
            method = info["method"] = "csi"
            clear = None
    else:
        clear = None

    if clear is None:
        ref_safe = reference.where(reference > 1.0)
        clear = _rd_csi_filter(irr, ref_safe, threshold=csi_threshold).fillna(False)

    clear = clear.astype(bool) & daytime
    by_day_clear = clear.groupby(clear.index.normalize()).sum()
    by_day_total = daytime.groupby(daytime.index.normalize()).sum()
    frac = (by_day_clear / by_day_total.replace(0, np.nan))
    clear_days = set(frac.index[frac >= day_fraction])

    info["n_days"] = int(frac.notna().sum())
    info["n_clear_days"] = len(clear_days)

    keep = pd.Series(df.index.normalize().isin(list(clear_days)), index=df.index)
    result = (df.index[keep], df.index[~keep])
    return (*result, info) if return_info else result
