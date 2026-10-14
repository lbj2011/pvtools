"""
export_lib.py — the functions PV-Copilot puts into an exported analysis script.

Every function here is a commented, stand-alone copy of the code the app itself
runs (pages/pvcopilot.py, analysis_utils.py, analysis_utils_pkg.py), with the
UI-only parts (progress callbacks, the app's own figure styling) taken out.
The NUMERICS ARE IDENTICAL: tests/test_export_lib.py checks that the exported
pipeline reproduces the app's degradation rates on every example dataset, and
each exported script is re-checked when the user clicks "Verify code".

The code export (code_export.py) copies only the functions a given run needs,
plus whatever they depend on, into the script.  Plot functions at the end
build the figures the script shows when it finishes.

Sections
    1. Loading and preparing the data
    2. Filters (data quality, time alignment, clipping, clear sky, outliers)
    3. Performance normalisation and daily aggregation
    4. Degradation-rate methods (YoY, LR, CSD, Holt-Winters, ARIMA)
    5. PVPRO (single-diode-model fit per time window)
    6. Plots
"""
import inspect
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pvlib.clearsky import detect_clearsky
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location
from pvlib.pvsystem import calcparams_desoto
from pvlib.singlediode import bishop88_i_from_v, bishop88_mpp, bishop88_v_from_i
from rdtools.aggregation import aggregation_insol
from rdtools.degradation import (degradation_classical_decomposition,
                                 degradation_ols, degradation_year_on_year)
from rdtools.filtering import (csi_filter, normalized_filter, poa_filter,
                               quantile_clip_filter, tcell_filter)
from rdtools.normalization import irradiance_rescale, pvwatts_dc_power
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# pvanalytics is optional: without it the IQR filter uses the same formula in
# pandas, and the clear-sky model cannot fit the array orientation.
try:
    from pvanalytics.quality.outliers import tukey
except ImportError:
    tukey = None
try:
    from pvanalytics.system import infer_orientation_fit_pvwatts
except ImportError:
    infer_orientation_fit_pvwatts = None


# =============================================================================
# Constants
# =============================================================================
# Mapped columns that must be numeric.
NUMERIC_ROLES = ("DC Power", "Irradiance", "Module temperature",
                 "DC Voltage", "DC Current")

# Words in a column name that mean Fahrenheit (module_temp_f, T_degF, "Temp (°F)").
FAHRENHEIT_WORDS = {"f", "degf", "fahrenheit", "°f"}

# rdtools' range filters use strict bounds (low < x < high); widening them by
# EPS makes them inclusive, e.g. irradiance == 0 at night is kept.
EPS = 1e-9

# pvlib's Reno & Hansen clear-sky test is calibrated for data at <= 15 min.
RENO_MAX_INTERVAL_MIN = 15.0

# rdtools' year-on-year method needs at least two years of data.
MIN_YEARS_FOR_YOY = 2.0

# Single-diode model (PVPRO): physical constants and STC reference conditions.
Q_E, K_B = 1.602176634e-19, 1.380649e-23     # elementary charge (C), Boltzmann (J/K)
T_REF_K = 25.0 + 273.15                      # STC cell temperature (K)
G_REF = 1000.0                               # STC irradiance (W/m²)
# Band gap (eV) and its temperature coefficient (eV/K) per technology.
BAND_GAP_BY_TECHNOLOGY = {
    "mono-c-Si":  (1.121, -0.0002677),
    "multi-c-Si": (1.121, -0.0002677),
    "GaAs":       (1.424, -0.000433),
    "CIGS":       (1.15,  -0.00001),
    "CdTe":       (1.475, -0.0003),
}
# The five fitted single-diode-model parameters (at reference conditions).
SDM_PARAMS = ("photocurrent_ref", "saturation_current_ref",
              "resistance_series_ref", "resistance_shunt_ref", "diode_factor")

# Plot colours.
BLUE, LIGHT_BLUE, GREY, RED = "#0070C0", "#A6CAEC", "#B8C2CC", "#D9534F"


# =============================================================================
# Small utilities
# =============================================================================


def as_numeric(s):
    """A column as numbers (text that isn't a number becomes NaN)."""
    return s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")


def years_since_start(index):
    """Elapsed time since the first timestamp, in years."""
    return (index - index[0]).days / 365.25


# =============================================================================
# Loading and preparing the data
# =============================================================================
def coerce_numeric_columns(df, mapping):
    """Make sure every mapped measurement column is numeric.

    Some loggers export numbers as text; those become real numbers here
    (anything unreadable becomes NaN) so later arithmetic can't fail."""
    if df is None or not mapping:
        return df
    for role in NUMERIC_ROLES:
        col = mapping.get(role)
        if col and col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def name_says_fahrenheit(col):
    """True when the column name contains a Fahrenheit word, e.g.
    module_temp_f, T_degF or "Module temp (°F)"."""
    words = re.split(r"[\s_()\[\]-]+", str(col).lower())
    return any(w in FAHRENHEIT_WORDS for w in words)


def fix_temperature_units(df, mapping):
    """Convert Fahrenheit temperature columns to °C and blank logger sentinels.

    Everything downstream (value ranges, the temperature correction, PVPRO)
    works in °C. A column is taken to be °F when
      * its name says so AND its 99.9th percentile is above 95 (a °C module
        temperature never gets that hot, so an already-converted column with
        an `_f` name is left alone), or
      * its values can only be °F: median above 60 and nothing above 212.
    Values below -60 are logger error codes (e.g. -327.67) and become NaN.
    Returns (df, list of notes describing what was changed)."""
    notes = []
    copied = False
    for role in ("Module temperature", "Ambient temperature"):
        col = (mapping or {}).get(role)
        if not col or col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        n_sent = int((v < -60).sum())
        if n_sent:
            v = v.where(v >= -60)
        med, mx = v.median(), v.max()
        p999 = v.quantile(0.999) if v.notna().any() else np.nan
        name_says_f = name_says_fahrenheit(col) and np.isfinite(p999) and p999 > 95
        values_say_f = (np.isfinite(med) and np.isfinite(mx) and med > 60 and mx <= 212)
        if name_says_f or values_say_f:
            v = (v - 32.0) * 5.0 / 9.0
            why = "its name" if name_says_f else f"its values (median {med:.0f})"
            notes.append(f"{role} column '{col}' is in °F (from {why}) — "
                         "converted to °C for filtering and temperature correction.")
        if n_sent:
            notes.append(f"{role} column '{col}': {n_sent:,} logger sentinel values "
                         "(below −60) treated as missing.")
        if name_says_f or values_say_f or n_sent:
            if not copied:
                df, copied = df.copy(), True
            df[col] = v
    return df, notes


def drop_duplicate_timestamps(df, mapping=None):
    """Remove repeated timestamps, keeping the first reading of each.

    Daily aggregation and the clear-sky test assume one reading per time
    stamp. Returns (df, note or None)."""
    try:
        time_key = (mapping or {}).get("Time")
        if time_key and time_key != "__index__" and time_key in df.columns:
            dup_mask = df[time_key].duplicated(keep="first")
        else:
            dup_mask = df.index.duplicated(keep="first")
        n_dupes = int(dup_mask.sum())
        if n_dupes == 0:
            return df, None
        cleaned = df.loc[~np.asarray(dup_mask)].copy()
        note = (f"Duplicate timestamps — {n_dupes:,} duplicate reading(s) "
                "found and removed (kept the first occurrence of each).")
        return cleaned, note
    except Exception:
        return df, None


def downsize_block_mean(df, factor):
    """Average every `factor` consecutive rows into one (block averaging).

    Used by PV-Copilot to keep large, high-resolution records fast. Works for
    decimal factors too: row i goes to block floor(i / factor). Numeric
    columns are averaged, other columns keep the block's first value, and
    each new timestamp is the mean time of its block (the full time span is
    kept)."""
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 1:
        raise ValueError(f"Downsize factor must be a finite number > 1 (got {factor}).")

    blocks = np.floor(np.arange(len(df)) / factor).astype(np.int64)

    out = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.groupby(blocks).mean()
        else:
            out[col] = s.groupby(blocks).first()
    df_out = pd.DataFrame(out)

    if isinstance(df.index, pd.DatetimeIndex):
        tz = df.index.tz
        base_index = df.index.tz_localize(None) if tz is not None else df.index
        # Average the timestamps as int64 nanoseconds (cast to [ns] first so
        # the unit is unambiguous on every pandas version).
        ns = pd.Series(base_index.astype("datetime64[ns]").astype("int64"))
        mean_ns = ns.groupby(blocks).mean().astype("int64")
        new_index = pd.DatetimeIndex(pd.to_datetime(mean_ns.values, unit="ns"))
        if tz is not None:
            new_index = new_index.tz_localize(tz)
        new_index.name = df.index.name
        df_out.index = new_index
    else:
        df_out.index.name = df.index.name

    df_out.attrs = dict(getattr(df, "attrs", {}))
    return df_out


# =============================================================================
# Filters
#    Each filter returns (kept_index, removed_index[, info]) so the steps can
#    be combined and reported.
# =============================================================================
def basic_value_filter(df, mapping, irr_min=0.0, irr_max=1500.0,
                       temp_min=-40.0, temp_max=100.0, power_min=-1.0):
    """Physically possible values only.

    Irradiance 0-1500 W/m² (rdtools poa_filter), module temperature
    -40..100 °C (rdtools tcell_filter) and DC power >= -1 W. Missing values
    fail the check."""
    irr_key = mapping.get("Irradiance")
    temp_key = mapping.get("Module temperature")
    power_key = mapping.get("DC Power")

    mask = pd.Series(True, index=df.index)
    if irr_key and irr_key in df.columns:
        ok = poa_filter(as_numeric(df[irr_key]), irr_min - EPS, irr_max + EPS)
        mask &= ok.fillna(False)
    if temp_key and temp_key in df.columns:
        ok = tcell_filter(as_numeric(df[temp_key]), temp_min - EPS, temp_max + EPS)
        mask &= ok.fillna(False)
    if power_key and power_key in df.columns:
        mask &= (as_numeric(df[power_key]) >= power_min).fillna(False)
    return df.index[mask], df.index[~mask]


def align_to_solar_day(df, irradiance_key, target_hour=12.5, tolerance_h=1.75):
    """Shift the clock by whole hours so solar noon falls near 12:00.

    Daily aggregation and the clear-day test cut the record at midnight, so
    a "day" must be a solar day. Loggers may record in UTC, standard time or
    daylight-saving time; instead of assuming one, the offset is measured:
    the irradiance-weighted centre of each day (which IS solar noon for a
    roughly symmetric day, at any time resolution), median over the record.
    If that is within `tolerance_h` of `target_hour` nothing changes.
    Returns (df, shift_hours, measured_centre_hour)."""
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)          # keep the wall-clock values
    g = as_numeric(df[irradiance_key]) if irradiance_key in df.columns else None
    peak = np.nan
    if g is not None:
        s = pd.Series(np.asarray(g, float), index=idx).clip(lower=0)
        s = s[s > 50]                        # daylight readings only
        if len(s) > 20:
            hrs = s.index.hour + s.index.minute / 60.0
            day = s.index.normalize()
            w = pd.DataFrame({"h": hrs * s.values, "g": s.values, "n": 1}, index=s.index)
            agg = w.groupby(day).sum()
            agg = agg[agg["n"] >= 3]         # days with at least 3 daylight readings
            if len(agg) >= 10:
                peak = float(np.median(agg["h"] / agg["g"]))
    shift = 0
    if np.isfinite(peak) and abs(peak - target_hour) > tolerance_h:
        shift = int(np.clip(round(target_hour - peak), -12, 12))
    out = df.copy()
    out.index = idx + pd.Timedelta(hours=shift) if shift else idx
    return out, shift, peak


def clipping_filter(df, power_key, spread_max=0.04, flat_top_min=0.03, quantile=0.99):
    """Remove inverter-clipped points — only if the record is actually clipped.

    A clipped array has a flat power ceiling. It is detected from the top of
    the power distribution:
      * (q99.9 - q99) / q99.9 < `spread_max`    (an unclipped tail is >= ~5 %)
      * more than `flat_top_min` of the top-5 % points lie within ±0.5 % of q99.9
    Only then is rdtools' quantile_clip_filter applied at `quantile`.
    Returns (kept_index, clipped_index, info)."""
    p = as_numeric(df[power_key]) if power_key in df.columns else None
    info = {"clipped": False, "spread": np.nan, "flat_top": np.nan, "ceiling": np.nan}
    if p is None:
        return df.index, df.index[:0], info
    pos = p[p > 0].dropna()
    if len(pos) < 200:
        return df.index, df.index[:0], info
    q999, q99, q95 = pos.quantile([0.999, 0.99, 0.95])
    top = pos[pos >= q95]
    spread = float((q999 - q99) / q999) if q999 > 0 else np.nan
    flat = float(((top >= 0.995 * q999) & (top <= 1.005 * q999)).mean())
    info.update(spread=spread, flat_top=flat, ceiling=float(q999))
    if not (spread < spread_max and flat > flat_top_min):
        return df.index, df.index[:0], info
    info["clipped"] = True
    keep = quantile_clip_filter(p.fillna(0.0), quantile=quantile).reindex(df.index).fillna(True)
    keep = keep | ~(p > 0)          # never drop night / zero-power rows here
    return df.index[keep.to_numpy()], df.index[~keep.to_numpy()], info


# Clear-sky filter:
# Keeps only CLEAR DAYS. Each daytime reading is compared with a clear-sky
# reference; a day is clear when enough of its readings match it.
def median_interval_minutes(index):
    """Typical time step of the record, in minutes."""
    try:
        step = pd.to_datetime(pd.Series(index)).diff().median()
        m = step.total_seconds() / 60.0
        return m if np.isfinite(m) and m > 0 else None
    except Exception:
        return None


def standard_time_zone(longitude):
    """Standard-time zone nearest a longitude (Etc/GMT signs are inverted:
    Etc/GMT+7 is UTC-7)."""
    offset = int(round(float(longitude) / 15.0))
    return f"Etc/GMT{'+' if offset <= 0 else '-'}{abs(offset)}"


def infer_orientation(cs, solpos, power, index, max_points=6000, min_points=200):
    """Fit the array tilt/azimuth from measured power (pvanalytics PVWatts
    fit on bright hours). Returns (tilt, azimuth, r2) or None."""
    if infer_orientation_fit_pvwatts is None or power is None:
        return None
    try:
        pw = pd.Series(power).astype(float)
        pw.index = index
        ok = (cs["ghi"].to_numpy() > 200) & np.isfinite(pw.to_numpy()) & (pw.to_numpy() > 0)
        if int(ok.sum()) < min_points:
            return None
        pos = np.flatnonzero(ok)
        if len(pos) > max_points:                       # even thinning keeps all seasons
            pos = pos[:: int(np.ceil(len(pos) / max_points))]
        sl = np.zeros(len(pw), dtype=bool)
        sl[pos] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tilt, azimuth, r2 = infer_orientation_fit_pvwatts(
                pd.Series(pw.to_numpy()[sl], index=cs.index[sl]),
                cs["ghi"][sl], cs["dhi"][sl], cs["dni"][sl],
                solpos["apparent_zenith"][sl], solpos["azimuth"][sl])
        if not np.isfinite(r2) or r2 < 0.5:
            return None
        return float(tilt), float(azimuth), float(r2)
    except Exception as e:
        print("[clear_sky] orientation fit failed (%s: %s)" % (type(e).__name__, e))
        return None


def modeled_clear_sky_poa(index, latitude, longitude, tilt=None, azimuth=None,
                          altitude=0.0, model="ineichen", power=None):
    """Clear-sky plane-of-array irradiance from pvlib (Ineichen model).

    Array geometry: the given tilt/azimuth; otherwise fitted from `power`
    with pvanalytics; otherwise tilt = |latitude|, facing the equator.
    Returns (poa_global, tilt, azimuth, how_the_geometry_was_chosen)."""
    idx = pd.to_datetime(index)
    tz = idx.tz if idx.tz is not None else standard_time_zone(longitude)
    loc = Location(float(latitude), float(longitude), tz=tz,
                         altitude=float(altitude or 0.0))
    if idx.tz is None:
        idx_aware = idx.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        idx_aware = idx
    cs = loc.get_clearsky(idx_aware, model=model)
    solpos = loc.get_solarposition(idx_aware)

    source = "as supplied"
    if tilt is None or azimuth is None:
        fitted = infer_orientation(cs, solpos, power, index)
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
    poa = get_total_irradiance(
        surface_tilt=float(tilt), surface_azimuth=float(azimuth),
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=cs["dni"], ghi=cs["ghi"], dhi=cs["dhi"])["poa_global"]
    poa.index = index          # back to the record's own (possibly naive) index
    return poa, float(tilt), float(azimuth), source


def empirical_clear_sky_envelope(irradiance, window_days=30, quantile=0.90):
    """Clear-sky reference built from the measurements alone.

    For each time-of-day slot: a rolling `quantile` over ±`window_days`
    days. Tracks the true clear-sky curve on any site with some clear days
    in every season; needs no coordinates."""
    s = pd.Series(irradiance).astype(float).clip(lower=0)
    step = median_interval_minutes(s.index) or 60.0
    tod = (s.index.hour * 60 + s.index.minute + s.index.second / 60.0)
    tod = (np.round(tod / step) * step).astype(int)     # snap jittered stamps to slots
    frame = pd.DataFrame({"v": s.to_numpy(), "d": s.index.normalize(), "t": tod})
    piv = frame.pivot_table(index="d", columns="t", values="v", aggfunc="mean")
    if piv.empty:
        return pd.Series(np.nan, index=s.index)
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
    env = (piv.rolling(2 * int(window_days) + 1, center=True, min_periods=3)
              .quantile(float(quantile)).ffill().bfill())
    env = env.T.rolling(3, center=True, min_periods=1).mean().T   # smooth across time of day
    flat = env.stack(dropna=False)
    flat.index = (flat.index.get_level_values(0)
                  + pd.to_timedelta(flat.index.get_level_values(1), unit="m"))
    flat = flat[~flat.index.duplicated()]
    # Look each reading up by its snapped slot, not its raw timestamp.
    key = pd.DatetimeIndex(frame["d"]) + pd.to_timedelta(frame["t"], unit="m")
    return pd.Series(flat.reindex(key).to_numpy(), index=s.index)


def peak_hour_offset(measured, reference, min_days=20):
    """Median daily (measured peak time - modelled peak time), in hours —
    how far the record's clock is from solar time."""
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


def calibrate_level(measured, reference):
    """Scale the modelled reference to the sensor's own level (rdtools
    irradiance_rescale; a 98th-percentile ratio as fallback). The clear-sky
    index is a ratio, so a level bias would shift every reading."""
    ok = measured.notna() & reference.notna() & (measured > 50) & (reference > 50)
    if int(ok.sum()) < 50:
        return reference, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rescaled = irradiance_rescale(measured[ok], reference[ok], method="iterative")
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
    """Clear-sky reference irradiance for the record, and how it was made.

    With coordinates: the pvlib model, corrected for clock offset and
    sensor level. Without coordinates (or if the model fails): the
    empirical envelope."""
    irr = as_numeric(df[irradiance_key])
    if latitude is not None and longitude is not None:
        try:
            power = (as_numeric(df[power_key])
                     if power_key and power_key in df.columns else None)
            notes = []
            step_h = (median_interval_minutes(df.index) or 60.0) / 60.0

            # Pass 1: a quick model on a sample of days, only to measure how
            # far the timestamps are from solar time.
            probe_index = df.index
            if len(probe_index) > 4000:
                keep_days = pd.Index(probe_index.normalize().unique())[::5]
                sel = probe_index.normalize().isin(keep_days)
                if sel.sum() > 500:
                    probe_index = probe_index[sel]
            probe, _t, _a, _g = modeled_clear_sky_poa(
                probe_index, latitude, longitude, None, None, altitude)
            shift_h = peak_hour_offset(irr.reindex(probe_index), probe)
            if abs(shift_h) > 3.0:
                notes.append(f"WARNING: modelled solar noon is {shift_h:+.1f} h from the "
                             "measured peak -- check the coordinates and the timestamps")
                shift_h = 0.0
            elif abs(shift_h) < step_h:
                shift_h = 0.0

            # Pass 2: model every reading at its true solar time.
            solar_index = df.index - pd.Timedelta(hours=float(shift_h))
            ref, used_tilt, used_az, geom = modeled_clear_sky_poa(
                solar_index, latitude, longitude, tilt, azimuth, altitude, power=power)
            ref.index = df.index
            if shift_h:
                notes.append(f"timestamps are {shift_h:+.1f} h off solar time; corrected")

            if ref.notna().any() and float(np.nanmax(ref.to_numpy())) > 0:
                ref, factor = calibrate_level(irr, ref)
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
    return empirical_clear_sky_reference(df, irradiance_key, window_days, quantile)


def empirical_clear_sky_reference(df, irradiance_key, window_days=30, quantile=0.90):
    """Clear-sky reference without site coordinates (the empirical envelope
    of the record's own irradiance), and a description of it."""
    ref = empirical_clear_sky_envelope(as_numeric(df[irradiance_key]),
                                       window_days=window_days, quantile=quantile)
    return ref, (f"empirical envelope ({int(quantile * 100)}th percentile over "
                 f"±{int(window_days)} days; no site coordinates given)")


def clear_sky_filter(df, irradiance_key, reference, how="", csi_threshold=0.15,
                     day_fraction=0.5, method="auto", min_irradiance=50.0,
                     return_info=False):
    """Keep every reading of the days judged CLEAR.

    `reference` is the clear-sky irradiance for each reading (from
    empirical_clear_sky_reference or clear_sky_reference). Each daytime
    reading (irradiance > `min_irradiance`) is classified against it:
      "csi"  rdtools csi_filter: measured / clear-sky within `csi_threshold` of 1
      "reno" pvlib detect_clearsky (Reno & Hansen; for data at <= 15 min)
      "auto" reno for fast data, csi otherwise
    A day is clear when at least `day_fraction` of its daytime readings are
    clear. Returns (kept_index, removed_index[, info])."""
    irr = as_numeric(df[irradiance_key]).clip(lower=0, upper=1500)
    day_fraction = float(day_fraction)
    interval = median_interval_minutes(df.index)

    info = {"reference": how, "method": None, "n_clear_days": 0, "n_days": 0,
            "interval_min": interval}
    empty = (df.index[:0], df.index)
    if irr.notna().sum() == 0:
        info["reference"] = "no usable irradiance"
        return (*empty, info) if return_info else empty

    if method == "auto":
        method = "reno" if (interval is not None and interval <= RENO_MAX_INTERVAL_MIN) else "csi"
    info["method"] = method

    daytime = (irr > min_irradiance).fillna(False)

    if method == "reno":
        # detect_clearsky needs an evenly spaced index and a window of more
        # than two samples (window_length is in minutes).
        step = interval or 1.0
        freq = pd.tseries.frequencies.to_offset(pd.Timedelta(minutes=step))
        meas_r = irr.asfreq(freq) if irr.index.freq is None else irr
        ref_r = reference.reindex(meas_r.index)
        window_length = int(max(10, np.ceil(4 * step)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clear_r = detect_clearsky(meas_r.fillna(0.0), ref_r.fillna(0.0),
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
        clear = csi_filter(irr, ref_safe, threshold=csi_threshold).fillna(False)

    # Share of each day's daytime readings that are clear -> clear days.
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


# =============================================================================
# Performance normalisation and daily aggregation
# =============================================================================
def normalize(df, mapping, gamma=-0.004):
    """Add a "norm" column: measured / expected DC power (PVWatts model).

    expected = G/1000 * (1 + gamma * (T_module - 25))    (rdtools pvwatts_dc_power)
    so "norm" is the array's performance with the weather taken out; a
    degrading array shows a slowly falling "norm". Without a module
    temperature the temperature term is skipped. Readings below 50 W/m² get
    NaN (the ratio is meaningless in low light)."""
    if not mapping.get("Irradiance"):
        raise KeyError("normalize() requires 'Irradiance' to be mapped")
    if not mapping.get("DC Power"):
        raise KeyError("normalize() requires 'DC Power' to be mapped")

    irr_key = mapping["Irradiance"]
    power_key = mapping["DC Power"]
    temp_key = mapping.get("Module temperature")

    irr = as_numeric(df[irr_key])
    power = as_numeric(df[power_key])
    df[irr_key] = irr
    df[power_key] = power

    temp_cell = None
    if temp_key and temp_key in df.columns:
        temp_cell = as_numeric(df[temp_key])
        df[temp_key] = temp_cell

    p_expected = pvwatts_dc_power(irr, 1.0, temperature_cell=temp_cell, gamma_pdc=gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["norm"] = power / p_expected
    df.loc[irr < 50, "norm"] = np.nan
    return df


def low_irra_power_filter(df, mapping, irr_thresh=300, power_ratio=0.02,
                          norm_lower=0.01, norm_upper_pct=99):
    """Keep bright, producing, plausible readings:
      * irradiance >= `irr_thresh` W/m²                 (rdtools poa_filter)
      * DC power > `power_ratio` x irradiance            (the array is producing)
      * `norm_lower` <= norm <= its `norm_upper_pct`-th percentile
                                                         (rdtools normalized_filter)
    Needs the "norm" column from normalize()."""
    irr_key = mapping.get("Irradiance")
    power_key = mapping.get("DC Power")
    if not irr_key:
        raise KeyError("low_irra_power_filter() requires 'Irradiance' to be mapped")
    if not power_key:
        raise KeyError("low_irra_power_filter() requires 'DC Power' to be mapped")

    irr = as_numeric(df[irr_key])
    power = as_numeric(df[power_key])

    mask = poa_filter(irr, irr_thresh, np.inf).fillna(False)
    mask &= (power > power_ratio * irr).fillna(False)

    upper = df["norm"].quantile(norm_upper_pct / 100)
    mask &= normalized_filter(df["norm"], norm_lower - EPS, upper + EPS).fillna(False)
    return df.index[mask], df.index[~mask]


def identify_outliers_iqr(df, column, iqr_multiplier=1.5):
    """Tukey outliers of `column` (usually "norm"): outside
    [Q1 - k*IQR, Q3 + k*IQR] with k = `iqr_multiplier` (pvanalytics tukey,
    or the same formula in pandas). Returns (kept_index, outlier_index)."""
    if column not in df.columns:
        print(f"Error: The specified power column '{column}' does not exist in the DataFrame.")
        return pd.Index([]), pd.Index([])
    data = as_numeric(df[column])
    if tukey is not None:
        is_outlier = tukey(data, k=iqr_multiplier)
    else:
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        is_outlier = (data < q1 - iqr_multiplier * iqr) | (data > q3 + iqr_multiplier * iqr)
    is_outlier = is_outlier.fillna(False).astype(bool)
    return df.index[~is_outlier], df.index[is_outlier]


def aggregate_daily(df_good, irradiance_col):
    """Daily performance index: insolation-weighted mean of "norm" per day
    (rdtools aggregation_insol: sum(norm * G) / sum(G)). Days without data
    are dropped."""
    sub = df_good[["norm", irradiance_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    daily = aggregation_insol(sub["norm"], sub[irradiance_col], frequency="D")
    daily = daily.dropna()
    daily.index = pd.to_datetime(daily.index).tz_localize(None) \
        if getattr(daily.index, "tz", None) is not None else pd.to_datetime(daily.index)
    daily.name = None
    return daily


# =============================================================================
# Degradation-rate methods
#    Each takes the daily series and returns (rate in %/yr, details); the
#    details feed the plots. NaN means the method could not produce a rate.
# =============================================================================
def daily_span_years(daily):
    """Length of the daily series in years (None if it is empty)."""
    try:
        idx = pd.to_datetime(daily.dropna().index)
        if len(idx) < 2:
            return None
        return (idx.max() - idx.min()).days / 365.25
    except Exception:
        return None


def yoy_possible(daily):
    """Year-on-year needs at least two years of data."""
    span = daily_span_years(daily)
    return span is not None and span >= MIN_YEARS_FOR_YOY


def compute_yoy(daily, rolling_window=30, confidence_level=68.2):
    """Year-on-year degradation (rdtools degradation_year_on_year).

    Each day is compared with the same day one year later; the rate is the
    median of those year-over-year changes, and rdtools bootstraps a
    confidence interval. Details: ci, the individual YoY values and a
    `rolling_window`-day moving average for the plot."""
    series = daily.dropna().sort_index()
    rd, ci, yoy_values = np.nan, np.array([np.nan, np.nan]), None
    if len(series) >= 2:
        kw = dict(recenter=True, confidence_level=confidence_level)
        # `uncertainty_method` only exists in rdtools >= 3 (2.x always bootstraps)
        if "uncertainty_method" in inspect.signature(degradation_year_on_year).parameters:
            kw["uncertainty_method"] = "simple"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rd, ci, yoy_info = degradation_year_on_year(series, **kw)
            rd = float(rd)
            ci = np.asarray(ci, dtype=float)
            yoy_values = yoy_info.get("YoY_values")
        except ValueError as e:      # e.g. "must provide at least two years"
            print(f"[compute_yoy] rdtools: {e}")
    trend = series.rolling(rolling_window, center=True).mean()
    return rd, {"ci": ci, "yoy_values": yoy_values, "trend": trend,
                "trend_label": f"{rolling_window}-day rolling mean"}


def compute_lr(daily, confidence_level=68.2):
    """Linear regression (rdtools degradation_ols): the slope of the daily
    series relative to its intercept (year-0 performance), in %/yr."""
    series = daily.dropna().sort_index()
    if len(series) < 2:
        return np.nan, {}
    rd, ci, info = degradation_ols(series, confidence_level=confidence_level)
    t365 = (series.index - series.index[0]).days / 365.0     # rdtools' own time axis
    trend = pd.Series(info["intercept"] + info["slope"] * t365, index=series.index)
    return float(rd), {"ci": np.asarray(ci, float), "trend": trend,
                       "trend_label": "linear fit"}


def compute_csd(daily, confidence_level=68.2):
    """Classical seasonal decomposition (rdtools
    degradation_classical_decomposition): a centred 365-day moving average
    removes the seasonal cycle, then a linear fit to that trend. Needs a
    regular daily series, so gaps are filled by linear interpolation first."""
    series = daily.dropna().sort_index()
    if len(series) < 2:
        return np.nan, {}
    full = series.asfreq("D")
    full = full.interpolate(method="linear", limit_direction="both").asfreq("D")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rd, ci, info = degradation_classical_decomposition(
                full, confidence_level=confidence_level)
    except ValueError as e:
        print(f"[compute_csd] rdtools: {e}")
        return np.nan, {}
    return float(rd), {"ci": np.asarray(ci, float), "trend": info["series"].dropna(),
                       "trend_label": "365-day trend"}


def compute_hw(daily, period=12):
    """Holt-Winters exponential smoothing (statsmodels, additive trend and
    season of `period` samples); the rate is the linear slope of the fitted
    values relative to their mean."""
    series = daily.dropna()
    if len(series) < 2 * period:
        return np.nan, {}
    model = ExponentialSmoothing(series, trend='add', seasonal='add',
                                 seasonal_periods=period).fit()
    fitted = model.fittedvalues
    t = years_since_start(fitted.index).values.reshape(-1, 1)
    y = fitted.values
    slope = LinearRegression().fit(t, y).coef_[0]
    rd = slope / np.mean(y) * 100
    return rd, {"trend": fitted, "trend_label": "Holt-Winters fit"}


def compute_arima(daily, p=1, d=1, q=0, seasonal_period=12):
    """Seasonal ARIMA (statsmodels SARIMAX, order (p, d, q), seasonal
    (0, 1, 1, `seasonal_period`)); the rate is the linear slope of the fitted
    values relative to their mean, after dropping the model's warm-up."""
    series = daily.dropna()
    if len(series) < 24:
        return np.nan, {}
    res = SARIMAX(series, order=(p, d, q),
                  seasonal_order=(0, 1, 1, seasonal_period)).fit(disp=False)
    fitted = res.fittedvalues
    # The first fitted values are unreliable until the Kalman filter settles:
    # drop about one seasonal cycle (at most 10 % of the record).
    warmup = min(seasonal_period + d, max(1, len(fitted) // 10))
    fitted = fitted.iloc[warmup:]
    t = years_since_start(fitted.index).values.reshape(-1, 1)
    y = fitted.values
    slope = LinearRegression().fit(t, y).coef_[0]
    rd = slope / np.mean(y) * 100
    smooth = fitted.rolling(window=31, center=True, min_periods=1).mean()   # for the plot
    return rd, {"trend": smooth, "trend_label": "ARIMA fit (31-day smoothed)"}


# =============================================================================
# PVPRO — single-diode-model (SDM) parameters per time window
#    The measured maximum-power voltage and current are fitted with a De Soto
#    single-diode model in consecutive windows; each fit is translated to STC,
#    and the degradation rate is the trend of the STC maximum power.
# =============================================================================
def band_gap(technology):
    """(band gap in eV, its temperature coefficient in eV/K) for a technology."""
    if technology not in BAND_GAP_BY_TECHNOLOGY:
        raise ValueError(f"Unknown technology '{technology}'. Valid: {sorted(BAND_GAP_BY_TECHNOLOGY)}.")
    return BAND_GAP_BY_TECHNOLOGY[technology]


def sdm_at_conditions(params, G, T, cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """Translate reference SDM parameters to irradiance G and cell
    temperature T (pvlib calcparams_desoto) -> (IL, I0, Rs, Rsh, nNsVth)."""
    a_ref = params["diode_factor"] * cells_in_series * K_B * T_REF_K / Q_E
    return calcparams_desoto(
        effective_irradiance=G, temp_cell=T,
        alpha_sc=alpha_isc, a_ref=a_ref,
        I_L_ref=params["photocurrent_ref"],
        I_o_ref=params["saturation_current_ref"],
        R_sh_ref=params["resistance_shunt_ref"],
        R_s=params["resistance_series_ref"],
        EgRef=Eg_ref, dEgdT=dEgdT,
        irrad_ref=G_REF, temp_ref=25.0)


def predict_mpp(params, G, T, cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """Model maximum-power voltage and current at each (G, T) (pvlib bishop88_mpp)."""
    IL, I0, Rs, Rsh, nNsVth = sdm_at_conditions(params, G, T, cells_in_series,
                                                 alpha_isc, Eg_ref, dEgdT)
    with np.errstate(all="ignore"):
        i_mp, v_mp, _p_mp = bishop88_mpp(IL, I0, Rs, Rsh, nNsVth, method="newton")
    return np.asarray(v_mp, float), np.asarray(i_mp, float)


def stc_points(params, cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """Pmp, Vmp, Imp, Voc, Isc of a fitted model at STC (1000 W/m², 25 °C)."""
    IL, I0, Rs, Rsh, nNsVth = sdm_at_conditions(
        params, np.array([G_REF]), np.array([25.0]),
        cells_in_series, alpha_isc, Eg_ref, dEgdT)
    with np.errstate(all="ignore"):
        i_mp, v_mp, p_mp = bishop88_mpp(IL, I0, Rs, Rsh, nNsVth, method="newton")
        v_oc = bishop88_v_from_i(0.0, IL, I0, Rs, Rsh, nNsVth, method="newton")
        i_sc = bishop88_i_from_v(0.0, IL, I0, Rs, Rsh, nNsVth, method="newton")
    return (float(np.ravel(p_mp)[0]), float(np.ravel(v_mp)[0]), float(np.ravel(i_mp)[0]),
            float(np.ravel(v_oc)[0]), float(np.ravel(i_sc)[0]))


# The optimiser works on rescaled parameters (log for I0 and Rsh) so all five
# have comparable magnitudes.
def to_fit_scale(p, key):
    if key == "saturation_current_ref":
        return np.log(p) + 23.0
    if key == "resistance_shunt_ref":
        return np.log(p) / 2.0 + 1.0
    if key == "resistance_series_ref":
        return p * 2.2
    return p


def from_fit_scale(x, key):
    if key == "saturation_current_ref":
        return float(np.exp(x - 23.0))
    if key == "resistance_shunt_ref":
        return float(np.exp(2.0 * (x - 1.0)))
    if key == "resistance_series_ref":
        return float(x / 2.2)
    return float(x)


def mpp_fit_loss(x, G, T, V_meas, I_meas, V_scale, I_scale,
          cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """Mean squared relative error of the modelled vs measured Vmp and Imp."""
    params = {k: from_fit_scale(x[i], k) for i, k in enumerate(SDM_PARAMS)}
    try:
        V_pred, I_pred = predict_mpp(params, G, T, cells_in_series, alpha_isc, Eg_ref, dEgdT)
    except Exception:
        return 1e6
    v_err = (V_pred - V_meas) / V_scale
    i_err = (I_pred - I_meas) / I_scale
    val = np.nanmean(v_err ** 2 + i_err ** 2)
    return float(val) if np.isfinite(val) else 1e6


def fit_sdm_window(G, T, V_meas, I_meas, p0, lower_bounds, upper_bounds,
                cells_in_series, alpha_isc, Eg_ref, dEgdT,
                saturation_current_multistart=(0.2, 0.5, 1.0, 2.0, 5.0)):
    """Fit the five SDM parameters to one window (scipy L-BFGS-B, several
    starting values of I0; the best fit wins). None if every start fails."""
    x_lo = np.array([to_fit_scale(lower_bounds[k], k) for k in SDM_PARAMS])
    x_hi = np.array([to_fit_scale(upper_bounds[k], k) for k in SDM_PARAMS])
    bounds = list(zip(x_lo, x_hi))
    V_scale = max(float(np.nanmedian(V_meas)), 1e-3)
    I_scale = max(float(np.nanmedian(I_meas)), 1e-3)

    best, best_loss = None, np.inf
    for mult in saturation_current_multistart:
        p0_try = dict(p0)
        p0_try["saturation_current_ref"] = p0["saturation_current_ref"] * mult
        x0 = np.clip(np.array([to_fit_scale(p0_try[k], k) for k in SDM_PARAMS]), x_lo, x_hi)
        try:
            res = minimize(
                mpp_fit_loss, x0=x0, bounds=bounds, method="L-BFGS-B",
                args=(G, T, V_meas, I_meas, V_scale, I_scale,
                      cells_in_series, alpha_isc, Eg_ref, dEgdT),
                options={"maxiter": 80, "ftol": 1e-7, "disp": False})
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < best_loss:
            best, best_loss = res, res.fun
    if best is None:
        return None
    fit = {k: from_fit_scale(best.x[i], k) for i, k in enumerate(SDM_PARAMS)}
    fit["loss"] = float(best.fun)
    return fit


def initial_sdm_guess(G, T, V, I, cells_in_series):
    """Textbook starting values from the brightest 10 % of readings."""
    if len(G) < 10:
        return None
    mask = G >= np.nanquantile(G, 0.9)
    if mask.sum() < 5:
        mask = np.ones_like(G, dtype=bool)
    Vmp_med, Imp_med, Gmed = (float(np.nanmedian(V[mask])), float(np.nanmedian(I[mask])),
                              float(np.nanmedian(G[mask])))
    if Vmp_med <= 0 or Imp_med <= 0 or Gmed <= 0:
        return None
    IL_ref_guess = Imp_med * (G_REF / Gmed)
    n_guess = 1.03
    nNsVth_ref = n_guess * cells_in_series * (K_B * T_REF_K / Q_E)
    I0_ref_guess = max(IL_ref_guess * np.exp(-Vmp_med / nNsVth_ref), 1e-13)
    return dict(photocurrent_ref=float(np.clip(IL_ref_guess, 0.1, 20.0)),
                saturation_current_ref=float(np.clip(I0_ref_guess, 1e-13, 1e-5)),
                resistance_series_ref=0.4, resistance_shunt_ref=600.0,
                diode_factor=n_guess)


def compute_pvpro(df, mapping, cells_in_series=60, modules_per_string=1,
                  parallel_strings=1, alpha_isc=0.0046, technology="mono-c-Si",
                  days_per_run=14, iterations_per_year=12,
                  resistance_shunt_ref=600.0, delta_T=3.0,
                  irradiance_threshold=200.0, min_points_per_window=20):
    """PVPRO degradation rate (%/yr) from DC voltage and current.

    1. Keep readings above `irradiance_threshold` with positive V and I; scale
       V and I to one module (÷ modules per string, ÷ parallel strings); cell
       temperature = module temperature + `delta_T`.
    2. Windows of `days_per_run` days, started `iterations_per_year` times a
       year; each window with >= `min_points_per_window` readings gets an SDM
       fit (warm-started from the previous window).
    3. Every fit is translated to STC -> Pmp, Vmp, Imp, Voc, Isc per window.
    4. Rate of each quantity: linear fit (after an IQR outlier trim) relative
       to its median. The headline rate is that of Pmp.
    Returns (rate, details) with details["windows"] (one row per window) and
    details["rates"] (%/yr of each STC quantity)."""
    required = ["DC Voltage", "DC Current", "Irradiance", "Module temperature"]
    missing = [r for r in required if mapping.get(r) is None or mapping[r] not in df.columns]
    if missing:
        raise ValueError("PVPRO needs these columns: " + ", ".join(missing))
    v_key, i_key = mapping["DC Voltage"], mapping["DC Current"]
    irr_key, tm_key = mapping["Irradiance"], mapping["Module temperature"]

    # 1. the readings the fit uses
    df_p = df[[v_key, i_key, irr_key, tm_key]].copy()
    for c in (v_key, i_key, irr_key, tm_key):
        df_p[c] = as_numeric(df_p[c])
    df_p.index = pd.to_datetime(df_p.index)
    df_p = df_p.dropna()
    df_p = df_p[df_p[irr_key] > irradiance_threshold]
    df_p = df_p[(df_p[v_key] > 0) & (df_p[i_key] > 0)]
    if len(df_p) < 100:
        raise ValueError(f"Only {len(df_p)} readings above {irradiance_threshold} W/m² "
                         "remain — too few for PVPRO.")

    V_arr = df_p[v_key].to_numpy(float) / max(modules_per_string, 1)
    I_arr = df_p[i_key].to_numpy(float) / max(parallel_strings, 1)
    G_arr = df_p[irr_key].to_numpy(float)
    Tc_arr = df_p[tm_key].to_numpy(float) + delta_T

    Eg_ref, dEgdT = band_gap(technology)
    lower_bounds = dict(photocurrent_ref=0.01, saturation_current_ref=1e-13,
                        resistance_series_ref=0.0, resistance_shunt_ref=10.0, diode_factor=0.5)
    upper_bounds = dict(photocurrent_ref=20.0, saturation_current_ref=1e-5,
                        resistance_series_ref=1.0, resistance_shunt_ref=5000.0, diode_factor=2.0)

    p0_global = initial_sdm_guess(G_arr, Tc_arr, V_arr, I_arr, cells_in_series)
    if p0_global is None:
        raise ValueError("Could not derive a starting point for the SDM fit "
                         "(too few high-irradiance readings).")
    p0_global["resistance_shunt_ref"] = float(resistance_shunt_ref)

    # 2. the windows
    t_start_all, t_end_all = df_p.index.min(), df_p.index.max()
    step_days = max(int(round(365.25 / max(iterations_per_year, 1))), 1)
    window_starts = []
    cur = t_start_all
    while cur + pd.Timedelta(days=days_per_run) <= t_end_all + pd.Timedelta(days=1):
        window_starts.append(cur)
        cur = cur + pd.Timedelta(days=step_days)
    n_total = len(window_starts)
    print(f"PVPRO: fitting {n_total} windows of {days_per_run} days …")

    rows = []
    p0_warm = dict(p0_global)
    for cur in window_starts:
        idx_w = (df_p.index >= cur) & (df_p.index < cur + pd.Timedelta(days=days_per_run))
        n_w = int(idx_w.sum())
        if n_w >= min_points_per_window:
            fit = fit_sdm_window(G_arr[idx_w], Tc_arr[idx_w], V_arr[idx_w], I_arr[idx_w],
                              p0_warm, lower_bounds, upper_bounds,
                              cells_in_series, alpha_isc, Eg_ref, dEgdT)
            if fit is not None:
                p0_warm = {k: fit[k] for k in SDM_PARAMS}        # warm start the next window
                # 3. this window's model at STC
                p_mp, v_mp, i_mp, v_oc, i_sc = stc_points(fit, cells_in_series,
                                                           alpha_isc, Eg_ref, dEgdT)
                rows.append({"t_mid": cur + pd.Timedelta(days=days_per_run / 2),
                             "p_mp_ref": p_mp, "v_mp_ref": v_mp, "i_mp_ref": i_mp,
                             "v_oc_ref": v_oc, "i_sc_ref": i_sc,
                             "loss": fit["loss"], "n_points": n_w,
                             **{k: fit[k] for k in SDM_PARAMS}})
            else:
                p0_warm = dict(p0_global)
    if len(rows) < 4:
        raise ValueError(f"PVPRO produced only {len(rows)} successful window fits (need at "
                         "least 4). Try longer windows (days_per_run) or looser filters.")
    pfit = pd.DataFrame(rows).set_index("t_mid").sort_index()

    # 4. trend of each STC quantity
    def _linear_rate(series):
        s = series.dropna()
        if len(s) < 4:
            return np.nan, s
        q1, q3 = np.nanpercentile(s.values, [25, 75])
        iqr = q3 - q1
        keep = (s.values >= q1 - 1.5 * iqr) & (s.values <= q3 + 1.5 * iqr)
        s_clean = s.loc[keep] if keep.sum() >= 4 else s
        t_years = years_since_start(s_clean.index).values.reshape(-1, 1)
        lr = LinearRegression().fit(t_years, s_clean.values)
        med = float(np.nanmedian(s_clean.values))
        if med == 0 or not np.isfinite(med):
            return np.nan, s_clean
        return float(lr.coef_[0] / med * 100.0), s_clean

    rates = {}
    for col in ("p_mp_ref", "v_mp_ref", "i_mp_ref", "v_oc_ref", "i_sc_ref"):
        rates[col], _ = _linear_rate(pfit[col])
    rd = rates["p_mp_ref"]
    if not np.isfinite(rd):
        # fallback: rdtools YoY on the per-window Pmp series
        try:
            rd = float(degradation_year_on_year(pfit["p_mp_ref"].dropna())[0])
        except Exception:
            rd = np.nan
    return rd, {"windows": pfit, "rates": rates}


# =============================================================================
# Plots
#    Each returns a plotly figure; the script shows them at the end.
# =============================================================================
def style_figure(fig, title, ytitle, height=380):
    """Common look of the figures."""
    fig.update_layout(title=title, template="plotly_white", height=height,
                      margin=dict(l=60, r=20, t=50, b=40), yaxis_title=ytitle,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                  xanchor="right", x=1.0))
    return fig


def plot_filtering(df_all, df_good, max_points=20000):
    """Normalised performance of every reading: kept (blue) vs removed (grey).
    At most `max_points` readings are drawn (evenly thinned) to keep it fast."""
    def _thin(s):
        s = s.dropna()
        return s.iloc[:: int(np.ceil(len(s) / max_points))] if len(s) > max_points else s
    removed = df_all.loc[~df_all.index.isin(df_good.index), "norm"]
    fig = go.Figure()
    r = _thin(removed)
    k = _thin(df_good["norm"])
    fig.add_trace(go.Scattergl(x=r.index, y=r.values, mode="markers", name="removed",
                               marker=dict(size=3, color=GREY, opacity=0.4)))
    fig.add_trace(go.Scattergl(x=k.index, y=k.values, mode="markers", name="kept",
                               marker=dict(size=3, color=BLUE, opacity=0.6)))
    if len(k) > 10:                     # frame the kept data, not the outliers
        lo, hi = np.nanpercentile(k.values, [1, 99])
        pad = 0.6 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad])
    kept_pct = 100.0 * len(df_good) / max(len(df_all), 1)
    return style_figure(fig, f"Filtering — {len(df_good):,} of {len(df_all):,} readings kept "
                       f"({kept_pct:.1f} %)", "Normalised power (W at 1000 W/m², 25 °C)")


def plot_daily_trend(daily, rate, method, details):
    """Daily performance index with the method's trend line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="markers", name="daily",
                             marker=dict(size=6, color=LIGHT_BLUE, opacity=0.8)))
    trend = (details or {}).get("trend")
    if trend is not None and len(trend):
        fig.add_trace(go.Scatter(x=trend.index, y=np.asarray(trend), mode="lines",
                                 name=(details or {}).get("trend_label", "trend"),
                                 line=dict(color=BLUE, width=2.5)))
    return style_figure(fig, f"{method}: {rate:+.2f} %/yr", "Daily normalised power (W)")


def plot_yoy_distribution(details, rate):
    """Histogram of the individual year-on-year rates; the median is the result."""
    vals = (details or {}).get("yoy_values")
    fig = go.Figure()
    if vals is not None and len(vals):
        v = pd.Series(vals).dropna()
        lo, hi = np.nanpercentile(v, [1, 99])           # hide the extreme 2 %
        fig.add_trace(go.Histogram(x=v[(v >= lo) & (v <= hi)], nbinsx=60,
                                   marker_color=LIGHT_BLUE, name="YoY rates"))
        fig.add_vline(x=rate, line_color=RED, line_width=2,
                      annotation_text=f"median {rate:+.2f} %/yr")
    fig.update_xaxes(title_text="Year-on-year rate (%/yr)")
    return style_figure(fig, "Year-on-year rate distribution", "Count", height=320)


def plot_method_comparison(results):
    """Bar chart of the rate from each method."""
    names = [m for m, r in results.items() if r is not None and np.isfinite(r)]
    vals = [results[m] for m in names]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=BLUE,
                           text=[f"{v:+.2f}" for v in vals], textposition="outside"))
    return style_figure(fig, "Degradation rate by method", "%/yr", height=320)


def plot_pvpro(details):
    """STC parameters of every PVPRO window with their trends."""
    pfit, rates = details["windows"], details["rates"]
    cols = [("p_mp_ref", "Pmp (W)"), ("v_mp_ref", "Vmp (V)"), ("i_mp_ref", "Imp (A)"),
            ("v_oc_ref", "Voc (V)"), ("i_sc_ref", "Isc (A)")]
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[f"{label} — {rates.get(c, np.nan):+.2f} %/yr"
                                        for c, label in cols])
    for i, (c, label) in enumerate(cols, start=1):
        s = pfit[c].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="markers", showlegend=False,
                                 marker=dict(size=7, color=BLUE, opacity=0.6)), row=i, col=1)
        if len(s) >= 2 and np.isfinite(rates.get(c, np.nan)):
            t = years_since_start(s.index).values
            med = float(np.nanmedian(s.values))
            line = med + rates[c] / 100.0 * med * (t - np.nanmean(t))
            fig.add_trace(go.Scatter(x=s.index, y=line, mode="lines", showlegend=False,
                                     line=dict(color=RED, width=2)), row=i, col=1)
    fig.update_layout(template="plotly_white", height=900, title="PVPRO: STC parameters per window",
                      margin=dict(l=60, r=20, t=70, b=40))
    return fig
