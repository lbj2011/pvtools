import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc
from dash import html, dcc
import base64, os, json
import openai
# import rdtools
import ast
import plotly.express as px
import plotly.graph_objects as go
from page_supporting_files.pvcopilot_filter_functions import auto_fix_timezone
import traceback
from dotenv import load_dotenv
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import time


load_dotenv(override=True)

cborg_API_KEY = os.getenv("cborg_api_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configuration (from your prompt) ---
# client = openai.OpenAI(
#     api_key=cborg_API_KEY,
#     base_url="https://api.cborg.lbl.gov"
# )

# client_gpt = openai.OpenAI(
#     api_key= OPENAI_API_KEY
# )

client = openai.OpenAI(
    api_key= OPENAI_API_KEY
)

# Model used for column identification. (Kept as a constant so parse_contents
# and any future caller reference one place.)
LLM_MODEL = "gpt-5.4-mini"

def _time_to_years(index):
    return (index - index[0]).days / 365.25


# =============================================================================
# FIXED-TIME DOWNSAMPLING for very large datasets
#
# Keeps the SAME clock times every day across the whole record (e.g. always
# 09:30, 11:00, 12:30, 14:00) instead of truncating rows or averaging. This
# preserves the full multi-year span (which degradation analysis needs) and
# keeps the sampling bias identical in every year, so it cancels out of
# year-over-year ratios, normalization, and trend fits.
# =============================================================================
def downsample_fixed_times(df, mapped_variables_dict=None,
                           max_points=10000, min_per_day=6):
    """
    Downsample a large sub-daily dataset to <= max_points rows by keeping a
    fixed set of times of day, the same ones for every day in the record.

    The kept clock times are daylight-weighted: they're placed at evenly
    spaced quantiles of the mean daily power (or irradiance) profile, so
    samples concentrate in producing hours while still spanning morning to
    evening. Night rows carry near-zero information and are the first to go.

    A floor of `min_per_day` samples/day is enforced even if that exceeds
    max_points — sub-daily analyses (clear-sky smoothness, time-shift
    detection, clipping) need >= 4 readings/day EVEN AFTER the basic value
    filter removes some points, so the floor is 6: measured on 12-17-year
    records, 4/day dropped the clear-sky filter into its degraded energy-only
    mode while 6/day kept full fidelity (rates within ~0.1 %/yr of the
    complete dataset, ~38k rows instead of 152k).

    Whole days are never dropped, and every kept value is a real measurement
    (no averaging/interpolation).

    Returns
    -------
    df_out : pd.DataFrame
        The downsampled frame (or the input unchanged when small enough).
    note : str or None
        Human-readable description of what was done; None if unchanged.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) <= max_points:
        return df, None
    n_days = len(np.unique(df.index.date))
    if n_days == 0:
        return df, None
    target = max(int(min_per_day), int(max_points // n_days))
    if len(df) / n_days <= target:
        return df, None  # already at/below the target density

    tod = (df.index.hour * 60 + df.index.minute).values

    # Daylight weighting from the mean power (or irradiance) profile by
    # time-of-day; uniform if neither column is known.
    weight_col = None
    for role in ("DC Power", "Irradiance"):
        key = (mapped_variables_dict or {}).get(role)
        if key and key in df.columns:
            weight_col = key
            break
    if weight_col:
        w = pd.to_numeric(df[weight_col], errors="coerce").clip(lower=0).fillna(0)
        profile = pd.Series(w.values, index=tod).groupby(level=0).mean().sort_index()
    else:
        profile = pd.Series(1.0, index=np.unique(tod))
    if profile.sum() <= 0:
        profile[:] = 1.0

    # Pick `target` clock times at evenly spaced quantiles of the cumulative
    # profile — dense where the plant produces, sparse at the edges.
    cum = profile.cumsum() / profile.sum()
    quantiles = (np.arange(target) + 0.5) / target
    positions = np.minimum(np.searchsorted(cum.values, quantiles), len(cum) - 1)
    picked = sorted(set(int(cum.index[p]) for p in positions))

    keep = np.isin(tod, picked)
    df_out = df[keep].copy()
    df_out.attrs = dict(getattr(df, "attrs", {}))

    if len(picked) <= 8:
        times_txt = ", ".join(f"{t // 60:02d}:{t % 60:02d}" for t in picked)
    else:
        times_txt = (f"{picked[0] // 60:02d}:{picked[0] % 60:02d} to "
                     f"{picked[-1] // 60:02d}:{picked[-1] % 60:02d}")
    note = (f"Large dataset downsampled from {len(df):,} to {int(keep.sum()):,} rows by "
            f"keeping the same {len(picked)} times of day across all days ({times_txt}); "
            f"the full {n_days:,}-day span is preserved and every kept value is a real "
            f"measurement.")
    print(f"  {note}")
    return df_out, note


# Column usability is judged by how much REAL data it carries -- both an
# absolute count and the missingness rate. The ceiling is deliberately high: a
# column that is, say, 64% missing but still has >1000 real values is usable and
# must NOT be discarded in favor of a worse fallback. Only genuinely dead /
# near-empty columns are dropped.
MAX_MISSING_FRAC = 0.9      # drop only near-dead columns (>90% missing)
FLAG_MISSING_FRAC = 0.05    # flag columns 5-90% missing as gappy
MIN_VALID_POINTS = 10       # absolute floor: fewer real values than this -> drop
# Min plausible irradiance peak (95th pct, W/m^2). Below this the column is the
# wrong units / a dead sensor, so we ignore it rather than void every row.
MIN_IRRADIANCE_PEAK = 50.0


# A per-device channel names a specific inverter / mppt / string / combiner /
# meter / zone by NUMBER (inv1, mppt_2, ...). A system-level column has no such
# device index; for whole-system degradation the system total is preferred.
_DEVICE_LEVEL_RE = re.compile(
    r'(?:inv|inverter|mppt|string|combiner|cb|ch|meter|zone|block|array)[\s_]*\d',
    re.I)


def _is_device_level(col):
    """True if a column name targets one numbered device (inv1, mppt_2, ...)."""
    return bool(_DEVICE_LEVEL_RE.search(str(col)))


# Friendly, concrete labels for a numbered device channel, so the mapping pill
# reads "one inverter" (for inv1_dc_power) instead of the vaguer "per-device".
# Matched only when _is_device_level() is already true; order matters only in
# that the first hit wins. Keep these device words in sync with _DEVICE_LEVEL_RE.
_DEVICE_NOUNS = [
    (re.compile(r'(?:inv|inverter)[\s_]*\d', re.I), "one inverter"),
    (re.compile(r'mppt[\s_]*\d',             re.I), "one MPPT"),
    (re.compile(r'string[\s_]*\d',           re.I), "one string"),
    (re.compile(r'(?:combiner|cb)[\s_]*\d',  re.I), "one combiner"),
    (re.compile(r'meter[\s_]*\d',            re.I), "one meter"),
    (re.compile(r'zone[\s_]*\d',             re.I), "one zone"),
    (re.compile(r'block[\s_]*\d',            re.I), "one block"),
    (re.compile(r'array[\s_]*\d',            re.I), "one array"),
    (re.compile(r'ch[\s_]*\d',               re.I), "one channel"),
]


def _device_scope_label(col):
    """A concrete 'one <device>' note for a numbered device channel (e.g.
    'one inverter' for inv1_dc_power). Falls back to 'one device' if the device
    word isn't one we specifically name. This is CONTEXT, not a data defect —
    it tells the user the column is a single device, not the system total."""
    s = str(col)
    for rx, noun in _DEVICE_NOUNS:
        if rx.search(s):
            return noun
    return "one device"


def _missing_frac(df, col):
    """Fraction of a column that is missing (non-numeric/blank counts as missing)."""
    if col not in df.columns or len(df) == 0:
        return 1.0
    return float(pd.to_numeric(df[col], errors="coerce").isna().mean())


def _best_by_data(df, cols):
    """Pick the column that actually carries data among candidate column names.

    A candidate is usable only if it is at most MAX_MISSING_FRAC missing, has at
    least MIN_VALID_POINTS real values, and isn't all-zero/constant (nunique<=1).
    A name match is NOT enough. Among the usable ones, prefer system-level over
    per-device, then most non-null values, then highest variance, then order.

    Returns (best_col_or_None, viable_cols).
    """
    total = len(df)
    viable, stats = [], {}
    for c in cols or []:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        non_null = int(s.notna().sum())
        missing_frac = 1.0 - (non_null / total) if total else 1.0
        if missing_frac > MAX_MISSING_FRAC:     # too sparse -> drop
            continue
        if non_null < MIN_VALID_POINTS:         # absolute floor for tiny frames
            continue
        nunique = int(s.nunique(dropna=True))
        if nunique <= 1:            # drops constant / all-zero (all-NaN handled above)
            continue
        viable.append(c)
        stats[c] = (non_null, float(s.var(skipna=True) or 0.0))
    if not viable:
        return None, []
    order = {c: i for i, c in enumerate(cols)}
    best = min(viable, key=lambda c: (
        _is_device_level(c),        # False(0) = system-level sorts first
        -stats[c][0],               # more non-null first
        -stats[c][1],               # higher variance first
        order.get(c, 1 << 30),      # then the candidate's listed order
    ))
    return best, viable


# Deterministic power-column name scan — a safety net for when the (small,
# non-deterministic) LLM omits an obvious power column from its candidates.
# Mirrors the irradiance name-scan below. A name matches "power" if it contains
# 'power'/'pwr' or a p-symbol token (pdc, p_dc, pmp, ppv, p_array, ...); it's AC
# if it carries an AC/output/grid marker, otherwise DC.
_POWER_NAME_RE = re.compile(
    r'power|pwr|(?:^|[^a-z])p_?(?:dc|ac|mpp?|pv|out|in|array|string)(?:[^a-z]|$)',
    re.I)
_AC_MARKER_RE = re.compile(r'(?:^|[^a-z])ac(?:[^a-z]|$)|out|grid|phase', re.I)
# Unambiguous DC-side markers: a "dc" token, or the inverter INPUT side
# (input / in_ / _in). Deliberately conservative — it excludes softer hints
# like "pv"/"mp" that also appear on system-level columns — so it only ever
# fires when a name is clearly the DC side. Mirrors _AC_MARKER_RE and is used
# only to reject a candidate from the OPPOSITE electrical role.
_DC_MARKER_RE = re.compile(
    r'(?:^|[^a-z])dc(?:[^a-z]|$)|input|(?:^|[^a-z])in_|_in(?:[^a-z]|$)', re.I)


def _name_side(col):
    """'ac' | 'dc' | None from unambiguous electrical-side markers in the NAME.

    Returns None when the name carries neither marker OR carries both (an
    ambiguous name like `ac_dc_*`); in those cases we defer to the LLM / data
    and never filter. Only a clean one-sided name yields 'ac' or 'dc'."""
    s = str(col)
    ac = bool(_AC_MARKER_RE.search(s))
    dc = bool(_DC_MARKER_RE.search(s))
    if ac and not dc:
        return "ac"
    if dc and not ac:
        return "dc"
    return None


def _drop_wrong_side(cols, want):
    """Drop candidates whose NAME unambiguously marks the opposite electrical
    side of `want` ('dc' or 'ac'). This is a deterministic guard against the
    (non-deterministic) LLM occasionally listing an obviously-AC column such as
    `ac_power` under a DC role — such a column must never be selected for, or
    offered as an alternative under, the wrong side. Names with no clear side
    marker (or an ambiguous one) are always kept."""
    opp = "ac" if want == "dc" else "dc"
    return [c for c in (cols or []) if _name_side(c) != opp]


def _scan_power_names(df):
    """Return (dc_power_cols, ac_power_cols) found purely by column name."""
    dc, ac = [], []
    for c in df.columns:
        if not _POWER_NAME_RE.search(str(c)):
            continue
        (ac if _AC_MARKER_RE.search(str(c)) else dc).append(c)
    return dc, ac


def _least_bad(df, cols):
    """Fallback for when NO candidate passes the strict quality gates in
    _best_by_data, but the role still has a proposed column present in the data.

    The strict gates are there to CHOOSE among several candidates — but when a
    role has only a poor column (or a single low-quality one), dropping it
    leaves the role unmapped and the user sees nothing. Instead we keep the
    least-bad candidate (most data, then most distinct values, then variance),
    so it's still shown, and return a reason the caller surfaces as a data-
    quality warning. Requires at least one real numeric value (a fully dead /
    all-text column is still unusable). Returns (col_or_None, reason)."""
    present = []
    for c in cols or []:
        if c not in df.columns:
            continue
        if int(pd.to_numeric(df[c], errors="coerce").notna().sum()) < 1:
            continue                     # totally dead / non-numeric — unusable
        present.append(c)
    if not present:
        return None, None

    def _key(c):
        s = pd.to_numeric(df[c], errors="coerce")
        return (_is_device_level(c), -int(s.notna().sum()),
                -int(s.nunique(dropna=True)), -float(s.var(skipna=True) or 0.0))

    best = min(present, key=_key)
    s = pd.to_numeric(df[best], errors="coerce")
    non_null = int(s.notna().sum())
    nunique = int(s.nunique(dropna=True))
    mf = _missing_frac(df, best)
    if nunique <= 1:
        reason = "is constant / all-zero"
    elif non_null < MIN_VALID_POINTS:
        reason = f"has only {non_null} valid values"
    elif mf > MAX_MISSING_FRAC:
        reason = f"is {mf:.0%} missing"
    else:
        reason = "is low quality"
    return best, reason


def _quality_tag(df, col, role=None):
    """A short parenthetical quality/context label for a candidate column, shown
    after its name in the mapping dropdown (e.g. 'all-zero', '94% missing',
    'per-device', 'wrong units'). Returns None for a clean, system-level column
    that needs no annotation."""
    if not col or col == "__index__" or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    non_null = int(s.notna().sum())
    if non_null == 0:
        return "no numeric data"
    # Irradiance must be in W/m^2 — flag a mis-scaled channel.
    if role == "Irradiance":
        peak = s.quantile(0.95)
        if not (np.isfinite(peak) and peak >= MIN_IRRADIANCE_PEAK):
            return "wrong units"
    nunique = int(s.nunique(dropna=True))
    if nunique <= 1:
        return "all-zero" if float(s.max()) == 0 else "constant"
    if non_null < MIN_VALID_POINTS:
        return f"only {non_null} pts"
    mf = _missing_frac(df, col)
    if mf > MAX_MISSING_FRAC:
        return f"{mf:.0%} missing"
    if mf > FLAG_MISSING_FRAC:
        return f"{mf:.0%} missing"
    if _is_device_level(col):
        return _device_scope_label(col)
    return None


_TIME_NAME_RE = re.compile(
    r'(?:^|[^a-z])(timestamp|datetime|measured_on|meas_on|date|time|utc|epoch)(?:[^a-z]|$)',
    re.I)

_IRR_NAME_RE = re.compile(
    r'irradian|irrad|poa|ghi|dni|pyran|insol|solar|g_poa|ref_cell|isc_ref', re.I)


def _name_looks_like_time(name):
    """True if a column NAME reads like a timestamp, matched as whole tokens so
    'update'/'runtime' don't false-positive."""
    return bool(_TIME_NAME_RE.search(str(name)))


def _looks_like_time(series, sample=20, min_frac=0.8):
    """True if a column's VALUES look like timestamps: sample the first non-null
    values, parse as datetimes, require most to parse AND not all identical.

    Numeric columns are rejected outright -- pd.to_datetime would read integers
    as nanoseconds-since-epoch. A datetime-typed column passes immediately.
    """
    if pd.api.types.is_numeric_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dropna().nunique() > 1
    vals = series.dropna().head(sample)
    if len(vals) == 0:
        return False
    parsed = pd.to_datetime(vals, errors="coerce")
    frac_ok = parsed.notna().mean()
    return bool(frac_ok >= min_frac and parsed.nunique(dropna=True) > 1)

# ================================
# Read data
# ================================

def parse_contents(contents=None, filename=None, df=None, progress=None):
    """
    Parses uploaded file OR an existing DataFrame (example dataset),
    identifies PV variables via LLM, and returns:

        (df, summary_table_div, mapped_variables_dict, code_read)

    `progress` is an optional callable taking a single human-readable
    message; it is invoked at each stage boundary so the UI can show a
    live "what am I doing right now" status line while this runs.
    """

    def _p(message):
        if progress is not None:
            try:
                progress(message)
            except Exception:
                pass  # a broken status line must never break parsing

    # -----------------------------
    # 1. Load dataframe
    # -----------------------------
    if df is None:

        _p("Reading your file…")

        if contents is None:
            return None, html.Div("Please upload a file to analyze."), {}, None, []

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                code_read = f"df = pd.read_csv('{filename}')"

            elif 'xls' in filename or 'xlsx' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
                code_read = f"df = pd.read_excel('{filename}')"

            elif 'parquet' in filename:
                df = pd.read_parquet(io.BytesIO(decoded))
                code_read = f"df = pd.read_parquet('{filename}')"

            else:
                return None, html.Div(
                    f"Unsupported file type: {filename}",
                    className="alert alert-danger"
                ), {}, None, []

        except Exception as e:
            return None, html.Div(
                f"There was an error processing the file: {e}",
                className="alert alert-danger"
            ), {}, None, []

    else:
        # Example dataset case
        code_read = "df = pd.read_csv('data/pmp.csv')"

    # ----------------------------------
    # 1.5 Detect if time is in index
    # ----------------------------------
    _p("Checking timestamps…")
    time_in_index = False

    if isinstance(df.index, pd.DatetimeIndex):
        time_in_index = True
    elif not pd.api.types.is_numeric_dtype(df.index):
        # Try converting index to datetime (non-destructive check).
        # Guard: only attempt this for a NON-numeric index. A plain integer
        # RangeIndex (0,1,2,...) -- which pd.read_csv produces for every
        # uploaded CSV whose time lives in a column -- would otherwise be
        # parsed by pd.to_datetime as nanoseconds-since-epoch, collapsing every
        # row onto 1970-01-01 and discarding the real time column.
        try:
            converted_index = pd.to_datetime(df.index, errors='coerce')
            if converted_index.notna().sum() > 0.9 * len(df):
                df.index = converted_index
                time_in_index = True
        except Exception:
            pass

    # ----------------------------------
    # 2A. Validate column names
    # ----------------------------------
    colnames = df.columns.tolist()

    # Case 1: pandas auto-generated integer column names
    case1_no_headers = colnames == list(range(len(colnames)))

    # Case 2: column names contain no alphabetic letters (numeric-only headers)
    case2_no_real_names = all(not any(c.isalpha() for c in str(name)) for name in colnames)

    if case1_no_headers or case2_no_real_names:
        return None, html.Div(
            "Uploaded file does not contain valid column names. "
            "Column names must include descriptive text (e.g. 'power', 'time').",
            className="alert alert-danger"
        ), {}, None, []

    # ----------------------------------
    # 3. Prepare LLM identification
    # ----------------------------------
    # NOTE: "DC Voltage" and "DC Current" are required by the PVPRO degradation
    # method (it does single-diode-model fitting on V and I).  They are NOT
    # required by YoY / LR / HW / ARIMA / CSD (those only need DC Power), so
    # they may safely come back as "N/A" without blocking the workflow.
    # Canonical roles the rest of the app keys on. (DC Voltage/Current are only
    # required by PVPRO; they may be absent for the statistical methods.)
    canonical_vars = [
        "DC Power",
        "DC Voltage",
        "DC Current",
        "Irradiance",
        "Module temperature",
    ]
    if not time_in_index:
        canonical_vars.insert(1, "Time")

    # Roles we ask the LLM about -- includes the AC counterparts so we can fall
    # back to AC when no DC column exists.
    llm_roles = [
        "DC Power", "AC Power",
        "DC Voltage", "AC Voltage",
        "DC Current", "AC Current",
        "Irradiance", "Module temperature",
    ]
    if not time_in_index:
        llm_roles.append("Time")

    prompt = f"""
    You are identifying physical quantities from PV (photovoltaic) system data
    column names. Below are the columns from the data file:

    {colnames}

    For each physical quantity below, list the columns that could represent it,
    best match first, AT MOST 3 per quantity (empty list if none match):
    {', '.join(llm_roles)}.

    List more than one only when a file genuinely has several channels for the
    same quantity (per inverter / string / MPPT); otherwise a single best match
    is fine. Keep DC and AC strictly separate (see rules below). Irradiance,
    Module temperature, and Time have no AC/DC distinction. Be concise.

    =======================================================================
    NAMING CONVENTIONS YOU MUST KNOW
    =======================================================================

    PV inverters expose two electrical sides:
      • DC side = "input" side  (panel array -> inverter input). On the DC
        side you find DC Voltage, DC Current, DC Power.
      • AC side = "output" side (inverter output -> grid). On the AC side
        you find AC Voltage, AC Current, AC Power.

    So if you see a column like `inv1_input_current`, `inv2_in_voltage`, or
    `dc_input_power`, those are the DC side. Columns like `inv1_output_*`,
    `ac_out_*`, `grid_*` are the AC side.

    Symbol-style names follow the same logic:
      • DC quantities:  v_dc, vdc, V_dc, v_mp, vmp, vpv, v_array, v_string,
                        v_panel, v_input, vin, idc, i_dc, i_mp, imp, ipv,
                        i_array, i_string, i_input, iin, pdc, p_dc, p_in,
                        p_mp, pmpp, p_pv, p_array.
      • AC quantities:  v_ac, vac, v_out, vout, v_grid, v_phase, v_line,
                        iac, i_ac, i_out, iout, i_grid, pac, p_ac, p_out,
                        p_grid.

    Real-world column-name patterns to recognise as DC Voltage:
        v_dc, vdc, V_DC, Vdc, dcvolt, dc_volt, dc_voltage, v_mp, vmp,
        v_mpp, vmpp, v_pv, vpv, v_array, v_string, v_panel,
        inv1_input_voltage, inv_input_voltage, inv1_v_in, inv1_vin,
        inverter1_dc_voltage, dc_in_voltage, mppt1_voltage,
        v_pos_neg, U_dc, Udc, panel_voltage, string_voltage.

    Real-world column-name patterns to recognise as DC Current:
        i_dc, idc, I_DC, Idc, dccurr, dc_curr, dc_current, i_mp, imp,
        i_mpp, impp, i_pv, ipv, i_array, i_string,
        inv1_input_current, inv_input_current, inv1_i_in, inv1_iin,
        inverter1_dc_current, dc_in_current, mppt1_current, I_pos_neg,
        string_current.

    Real-world column-name patterns to recognise as DC Power:
        p_dc, pdc, dc_power, P_DC, Pdc, p_mp, pmp, pmpp, p_pv, ppv,
        p_array, p_string, inv1_input_power, inv1_dc_power,
        inverter1_dc_power, mppt_power, panel_power, string_power.

    IMPORTANT — DO NOT MISS POWER. A column whose name contains "power" (or
    "pwr", or a p-symbol like pdc/p_dc/pmp/ppv) is a POWER column and MUST be
    listed under DC Power (or AC Power if it carries an AC/output/grid marker).
    Never leave DC Power empty when such a column exists. This is the single
    most important quantity — err toward including a plausible power column.

    AC columns (output / grid / inverter output) belong under the AC roles
    (AC Power / AC Voltage / AC Current) -- list them there, NOT under the DC
    roles. We prefer DC and only fall back to AC when no DC column exists.

    Patterns for Irradiance (plane-of-array):
        poa, poa_irradiance, ghi, dni, irr, irrad, irradiance, g_poa,
        g_pyranometer, g_silicon, isc_ref_cell, ref_cell, pyranometer,
        plane_of_array, in_plane_irradiance, solar_radiation.

    Patterns for Module temperature:
        module_temperature, mod_temp, t_mod, tmod, t_module, tmodule,
        cell_temperature, t_cell, tcell, panel_temp, back_of_module,
        BOM, module_temp_C, mod_T, T_pv, temperature_module.

    =======================================================================
    DECISION RULES (apply IN THIS ORDER)
    =======================================================================

    1) SYMBOL FIRST. If a column literally contains "dc" or "DC" anywhere,
       it is on the DC side. If it contains "ac" or "AC", it is on the AC
       side. (Treat these as substring matches but be careful with words
       like "factor", "track" — only match when "dc"/"ac" is a separate
       token or clearly an electrical-side label.)

    2) INVERTER INPUT/OUTPUT. If a column contains "input", "in_", or "_in"
       on an inverter / mppt / converter, it is DC. If it contains "output",
       "out_", "_out", "grid", or "phase", it is AC.

    3) CONSISTENCY. If you identified one DC quantity from an inverter
       (e.g. DC Current = `inv1_input_current`), look HARD for the matching
       DC Voltage from the SAME inverter (e.g. `inv1_input_voltage`,
       `inv1_v_in`, `inv1_dc_voltage`). It is very rare to have DC Current
       without DC Voltage in the same file — assume it is there and find it.
       Same in reverse: if you found a DC Voltage, look for the matching
       DC Current.

    4) UNITS HINTS. If the column name carries units (e.g. `*_V`, `*_kV`,
       `*_A`, `*_kA`, `*_W`, `*_kW`, `*_kWh`, `*_W_m2`, `*_degC`), use them
       to disambiguate quantity type (V -> voltage, A -> current, W -> power,
       W/m^2 -> irradiance, degC/C -> temperature), I is also current, remember this, 
       since there is an equation that is V = IR.

    5) DON'T GUESS THE QUANTITY TYPE. If you genuinely cannot tell what a column
       is, leave it out. But DO list every column you ARE confident matches a
       role -- multiple matches are expected and wanted.

    =======================================================================
    OUTPUT FORMAT
    =======================================================================

    Return ONLY a JSON object — no prose, no markdown fences. Each value is a
    list of matching column names, best match first (use [] if none match):

    {{
      "candidates": {{
        "DC Power": ["col", ...],
        "AC Power": ["col", ...],
        "DC Voltage": ["col", ...],
        "AC Voltage": ["col", ...],
        "DC Current": ["col", ...],
        "AC Current": ["col", ...],
        "Irradiance": ["col", ...],
        "Module temperature": ["col", ...],
        "Time": ["col", ...]
      }}
    }}
    """

    # Default return values
    mapped_variables_dict = {}
    mapping_notes = []   # AC-fallback / ambiguous-column / time-by-value warnings

    try:
        # Identification calls the LLM every run (no caching). temperature=0 +
        # seed=0 give best-effort determinism, but the gateway doesn't honor the
        # seed, so the same file CAN map to different columns run-to-run.
        _p("Identifying data columns…")
        _p("Asking AI to identify your columns…")
        # Call LLM. Pin temperature=0 + a fixed seed (best-effort determinism).
        # Some models reject these params -- fall back to a plain call if so.
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=0,
            )
        except (openai.BadRequestError, openai.UnprocessableEntityError):
            # Only retry without the params when the model REJECTS them (a 4xx).
            # A timeout / connection / rate-limit error must NOT trigger a second
            # full call here -- that would stack another timeout on top.
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )

        res_text = response.choices[0].message.content.strip()
        cleaned = res_text.lstrip("`").lstrip("json").rstrip("`")
        result = json.loads(cleaned)
        raw_candidates = result.get("candidates", {}) or {}

        def _cands(role):
            """Candidate column names the LLM proposed for a role (best-first),
            keeping only real, non-'N/A' strings."""
            v = raw_candidates.get(role, [])
            if isinstance(v, str):
                v = [v]
            return [c for c in v if isinstance(c, str) and c and c != "N/A"]
        # Per-role list of the OTHER valid columns the user could switch to
        # (keyed by the canonical mapping-table role). Surfaced inline under the
        # matching dropdown in Advanced mode.
        alternatives = {}
        _CANON_ROLE = {"AC Power": "DC Power", "AC Voltage": "DC Voltage",
                       "AC Current": "DC Current"}

        def _ambiguity_note(role, chosen, viable, ac=False):
            # Record the OTHER valid columns for this role. These are surfaced
            # ONLY inline under the matching dropdown in Advanced mode -- we
            # deliberately do NOT add them to mapping_notes (the consolidated
            # warning panel), so the message lives in exactly one place.
            if chosen and len(viable) > 1:
                alternatives[_CANON_ROLE.get(role, role)] = [c for c in viable if c != chosen]

        # ----------------------------------
        # Deterministic safety net: pair up DC Voltage / DC Current.
        #
        # If only one of {DC Voltage, DC Current} resolved, try to find the
        # partner by simple name-pattern substitution on the identified column.
        # Handles consistently-named sensor pairs (inv1_input_current ↔
        # inv1_input_voltage, vdc ↔ idc, v_mp ↔ i_mp, ...).
        # ----------------------------------
        def _pair_partner(known_col, want):
            """Find a column whose name is `known_col` with current<->voltage
            tokens swapped. `want` is 'voltage' or 'current'."""
            if not known_col or known_col == "N/A":
                return None
            # Token-replacement pairs (case-sensitive matches first, then
            # fall back to a case-insensitive scan).
            if want == "voltage":
                swaps = [("current", "voltage"), ("Current", "Voltage"),
                         ("CURRENT", "VOLTAGE"), ("curr",    "volt"),
                         ("Curr",    "Volt"),    ("CURR",    "VOLT"),
                         ("amps",    "volts"),   ("Amps",    "Volts"),
                         ("amp",     "volt"),    ("Amp",     "Volt"),
                         # symbol swaps: idc<->vdc, imp<->vmp, ipv<->vpv,
                         # i_dc<->v_dc, i_mp<->v_mp, etc.
                         ("i_dc",    "v_dc"),    ("I_dc",    "V_dc"),
                         ("I_DC",    "V_DC"),    ("idc",     "vdc"),
                         ("Idc",     "Vdc"),     ("IDC",     "VDC"),
                         ("i_mp",    "v_mp"),    ("I_mp",    "V_mp"),
                         ("imp",     "vmp"),     ("Imp",     "Vmp"),
                         ("IMP",     "VMP"),     ("ipv",     "vpv"),
                         ("Ipv",     "Vpv"),     ("IPV",     "VPV"),
                         ("i_in",    "v_in"),    ("I_in",    "V_in"),
                         ("iin",     "vin"),     ("Iin",     "Vin")]
            else:  # want == 'current'
                swaps = [("voltage", "current"), ("Voltage", "Current"),
                         ("VOLTAGE", "CURRENT"), ("volt",    "curr"),
                         ("Volt",    "Curr"),    ("VOLT",    "CURR"),
                         ("volts",   "amps"),    ("Volts",   "Amps"),
                         ("volt",    "amp"),     ("Volt",    "Amp"),
                         ("v_dc",    "i_dc"),    ("V_dc",    "I_dc"),
                         ("V_DC",    "I_DC"),    ("vdc",     "idc"),
                         ("Vdc",     "Idc"),     ("VDC",     "IDC"),
                         ("v_mp",    "i_mp"),    ("V_mp",    "I_mp"),
                         ("vmp",     "imp"),     ("Vmp",     "Imp"),
                         ("VMP",     "IMP"),     ("vpv",     "ipv"),
                         ("Vpv",     "Ipv"),     ("VPV",     "IPV"),
                         ("v_in",    "i_in"),    ("V_in",    "I_in"),
                         ("vin",     "iin"),     ("Vin",     "Iin")]
            for src, tgt in swaps:
                if src in known_col:
                    candidate = known_col.replace(src, tgt)
                    if candidate in colnames and candidate != known_col:
                        return candidate
            # Case-insensitive fallback: replace 'current'/'voltage' as words.
            lc = known_col.lower()
            if want == "voltage" and "current" in lc:
                # Map back to the actual column with the same case structure.
                for c in colnames:
                    cl = c.lower()
                    if cl == lc.replace("current", "voltage") and c != known_col:
                        return c
            if want == "current" and "voltage" in lc:
                for c in colnames:
                    cl = c.lower()
                    if cl == lc.replace("voltage", "current") and c != known_col:
                        return c
            return None

        # ----------------------------------
        # Resolve each role: prefer the candidate with the MOST REAL DATA, fall
        # back DC -> AC for the electrical quantities, and warn on ambiguity.
        # ----------------------------------
        def _resolve_dc_ac(dc_role, ac_role, label):
            """Return (chosen_col_or_None, 'dc'|'ac'|None) preferring DC.

            Candidates whose NAME clearly marks the wrong electrical side are
            dropped first (see _drop_wrong_side) so an AC-named column can never
            be chosen for — or offered as an alternative under — a DC role, and
            vice versa."""
            dc_cands = _drop_wrong_side(_cands(dc_role), "dc")
            ac_cands = _drop_wrong_side(_cands(ac_role), "ac")
            best, viable = _best_by_data(df, dc_cands)
            if best:
                _ambiguity_note(dc_role, best, viable)
                return best, "dc"
            best, viable = _best_by_data(df, ac_cands)
            if best:
                mapping_notes.append(
                    f"No usable DC {label} column — using AC {label} '{best}' "
                    f"(includes inverter effects, so it's an approximation).")
                _ambiguity_note(ac_role, best, viable, ac=True)
                return best, "ac"
            # Lenient fallback: nothing passed the quality gates, but if a
            # candidate column exists, keep the least-bad one (shown + warned)
            # rather than leaving the role unmapped.
            lb, reason = _least_bad(df, dc_cands)
            side = "dc"
            if lb is None:
                lb, reason = _least_bad(df, ac_cands)
                side = "ac"
            if lb is not None:
                mapping_notes.append(
                    f"{label.capitalize()} column '{lb}' {reason} — kept anyway "
                    f"so you can review it; results may be unreliable.")
                return lb, side
            return None, None

        # Voltage / Current (needed for PVPRO and for V*I power).
        _p("Mapping variables to columns…")
        v_col, v_side = _resolve_dc_ac("DC Voltage", "AC Voltage", "voltage")
        i_col, i_side = _resolve_dc_ac("DC Current", "AC Current", "current")

        # Pair recovery: if exactly one of V/I is still missing, try the name-swap.
        if v_col and not i_col:
            p = _pair_partner(v_col, want="current")
            if p:
                i_col, i_side = p, v_side
        elif i_col and not v_col:
            p = _pair_partner(i_col, want="voltage")
            if p:
                v_col, v_side = p, i_side

        # Power resolution priority (DC preferred; real/good preferred over
        # computed or low-quality; role never left empty when a column exists):
        #   1 strict DC  ->  2 DC V×I  ->  3 lenient DC (low-quality, shown+warned)
        #   -> 4 strict AC  ->  5 AC V×I  ->  6 lenient AC.
        # Supplement the LLM's candidates with a deterministic name scan so an
        # obvious column like 'dc_power' is caught even when the LLM omits it.
        scan_dc_p, scan_ac_p = _scan_power_names(df)
        # Guard each side against wrong-side names before combining with the
        # deterministic scan (which is already side-correct). This is what stops
        # an LLM-listed `ac_power` from surfacing under DC Power (selected or as
        # an "also detected" alternative), and vice versa.
        dc_p_cands = list(dict.fromkeys(
            _drop_wrong_side(_cands("DC Power"), "dc") + scan_dc_p))
        ac_p_cands = list(dict.fromkeys(
            _drop_wrong_side(_cands("AC Power"), "ac") + scan_ac_p))

        def _compute_power(side):
            df["computed_dc_power"] = (pd.to_numeric(df[v_col], errors="coerce")
                                       * pd.to_numeric(df[i_col], errors="coerce"))
            return "computed_dc_power", side

        p_col, p_side = None, None

        # 1) strict DC direct
        best_dc_p, viable_dc_p = _best_by_data(df, dc_p_cands)
        if best_dc_p:
            p_col, p_side = best_dc_p, "dc"
            _ambiguity_note("DC Power", best_dc_p, viable_dc_p)

        # 2) computed DC (V×I) from good V/I
        if p_col is None and v_col and i_col and v_side == "dc" and i_side == "dc":
            p_col, p_side = _compute_power("dc")
            mapping_notes.append("DC Power computed as Voltage × Current (no direct power column).")

        # 3) lenient DC: a single low-quality DC power column is still shown + warned
        if p_col is None:
            lb, reason = _least_bad(df, dc_p_cands)
            if lb is not None:
                p_col, p_side = lb, "dc"
                mapping_notes.append(
                    f"DC Power column '{lb}' {reason} — kept anyway so you can "
                    f"review it; results may be unreliable.")

        # 4) strict AC direct
        if p_col is None:
            best_ac_p, viable_ac_p = _best_by_data(df, ac_p_cands)
            if best_ac_p:
                p_col, p_side = best_ac_p, "ac"
                mapping_notes.append(
                    f"No usable DC power column — using AC power '{best_ac_p}' "
                    f"(includes inverter effects, so it's an approximation).")
                _ambiguity_note("AC Power", best_ac_p, viable_ac_p, ac=True)

        # 5) computed AC (V×I)
        if p_col is None and v_col and i_col:
            p_col, p_side = _compute_power("ac")
            mapping_notes.append("Power computed as Voltage × Current on the AC side "
                                 "(no DC power) — includes inverter effects, approximate.")

        # 6) lenient AC: last resort, low-quality AC power column shown + warned
        if p_col is None:
            lb, reason = _least_bad(df, ac_p_cands)
            if lb is not None:
                p_col, p_side = lb, "ac"
                mapping_notes.append(
                    f"AC Power column '{lb}' {reason} — kept anyway so you can "
                    f"review it; results may be unreliable (AC also includes "
                    f"inverter effects).")

        # Fill the third electrical quantity from the other two (P = V * I).
        # Exact on the DC side; on the AC side it's approximate (ignores power
        # factor) but consistent with the rest of the AC fallback. Only derive
        # from a SAME-SIDE partner so we never mix DC and AC. Only matters for
        # PVPRO. Division guards: zero/near-zero denominators -> NaN (dropped).
        if p_col:
            _pnum = pd.to_numeric(df[p_col], errors="coerce")
            if v_col is None and i_col is not None and i_side == p_side:
                df["computed_dc_voltage"] = _pnum / pd.to_numeric(
                    df[i_col], errors="coerce").replace(0, np.nan)
                v_col, v_side = "computed_dc_voltage", p_side
                mapping_notes.append(
                    "DC Voltage computed as Power / Current." if p_side == "dc"
                    else "Voltage computed as AC Power / AC Current — approximate "
                         "(ignores power factor).")
            elif i_col is None and v_col is not None and v_side == p_side:
                df["computed_dc_current"] = _pnum / pd.to_numeric(
                    df[v_col], errors="coerce").replace(0, np.nan)
                i_col, i_side = "computed_dc_current", p_side
                mapping_notes.append(
                    "DC Current computed as Power / Voltage." if p_side == "dc"
                    else "Current computed as AC Power / AC Voltage — approximate "
                         "(ignores power factor).")

        # Irradiance. The filters assume W/m^2 (clear-sky peaks ~800-1200), so a
        # column whose peak is far below that is the wrong units / a dead sensor
        # and would void every row. Many sites expose several irradiance channels
        # at different scales, so: (1) prefer an LLM candidate that's in plausible
        # W/m^2 units; (2) if none are, scan ALL irradiance-named columns for a
        # properly-scaled one; (3) only if nothing is usable, drop irradiance and
        # fall back to the power-only path.
        def _irr_peak(c):
            return pd.to_numeric(df[c], errors="coerce").quantile(0.95)

        def _wm2_ok(c):
            return c in df.columns and np.isfinite(_irr_peak(c)) and _irr_peak(c) >= MIN_IRRADIANCE_PEAK

        # Consider EVERY irradiance column -- the LLM's candidates plus a scan of
        # all irradiance-named columns -- but keep only the ones actually in W/m^2.
        # Pick the one with the most data; the ambiguity warning then offers ONLY
        # the other VALID columns as alternatives (never the mis-scaled ones).
        irr_named = [c for c in df.columns if _IRR_NAME_RE.search(str(c))]
        all_irr = list(dict.fromkeys([c for c in _cands("Irradiance") if c in df.columns]
                                     + irr_named))
        valid_irr = [c for c in all_irr if _wm2_ok(c)]
        irr_col, irr_viable = _best_by_data(df, valid_irr)

        if irr_col is not None:
            # Warn (with ONLY the valid alternatives) when more than one works.
            _ambiguity_note("Irradiance", irr_col, irr_viable)
            # If the LLM's own picks were mis-scaled and we recovered a valid one,
            # say so (so a single valid column is still explained).
            if not any(_wm2_ok(c) for c in _cands("Irradiance") if c in df.columns):
                mapping_notes.append(
                    f"Auto-selected irradiance '{irr_col}' (in W/m²); other irradiance "
                    "columns were the wrong units and were skipped.")
        elif all_irr:
            # Irradiance-named columns exist but none were usable. Say WHY: a
            # column in the right W/m² range that still got dropped was too
            # sparse (mostly missing); otherwise it was the wrong units.
            if valid_irr:
                mapping_notes.append(
                    "Irradiance column(s) present and in W/m² but too sparse "
                    "(mostly missing) — ignoring irradiance and using raw power.")
            else:
                mapping_notes.append(
                    "Irradiance column(s) present but not in W/m² (wrong units / dead sensor) "
                    "— ignoring irradiance and using raw power.")
        temp_col, temp_viable = _best_by_data(df, _cands("Module temperature"))
        _ambiguity_note("Module temperature", temp_col, temp_viable)
        if temp_col is None:
            lb, reason = _least_bad(df, _cands("Module temperature"))
            if lb is not None:
                temp_col = lb
                mapping_notes.append(
                    f"Module temperature column '{lb}' {reason} — kept anyway; "
                    f"temperature correction may be unreliable.")

        # Time: detect BY NAME first (the LLM's name-based pick, then any column
        # whose name reads like time); only if no time-like NAME exists, fall
        # back to scanning column VALUES for one that parses as datetimes.
        if time_in_index:
            time_col = df.index.name if df.index.name not in [None, ""] else "__index__"
        else:
            # A time column must also actually contain timestamps -- a name match
            # on an empty column is not enough.
            def _has_time_data(col):
                return (col in df.columns
                        and pd.to_datetime(df[col], errors="coerce").notna().sum() >= MIN_VALID_POINTS)

            time_col = None
            # 1) BY NAME — trust the LLM's Time candidate if it's a real, populated column.
            for c in _cands("Time"):
                if _has_time_data(c):
                    time_col = c
                    break
            # 2) BY NAME — backup: any column whose NAME looks like time AND has data.
            if time_col is None:
                for c in df.columns:
                    if _name_looks_like_time(c) and _has_time_data(c):
                        time_col = c
                        break
            # 3) BY VALUE — no time-like name with data; scan values.
            if time_col is None:
                for c in df.columns:
                    if _looks_like_time(df[c]):
                        time_col = c
                        mapping_notes.append(f"Time column detected from values: '{c}'.")
                        break

        # Canonical mapping (keys stay DC-named so downstream is unchanged; the
        # chosen column may be an AC column when DC was absent).
        mapped_variables_dict = {}
        if time_col:
            mapped_variables_dict["Time"] = time_col
        if p_col:
            mapped_variables_dict["DC Power"] = p_col
        if v_col:
            mapped_variables_dict["DC Voltage"] = v_col
        if i_col:
            mapped_variables_dict["DC Current"] = i_col
        if irr_col:
            mapped_variables_dict["Irradiance"] = irr_col
        if temp_col:
            mapped_variables_dict["Module temperature"] = temp_col

        # Flag kept-but-gappy columns (5-50% missing): usable, but the user
        # should treat the result carefully. (>50% missing was already dropped.)
        for _role, _col in mapped_variables_dict.items():
            if _role == "Time" or _col == "computed_dc_power":
                continue
            _mf = _missing_frac(df, _col)
            if _mf > FLAG_MISSING_FRAC:
                mapping_notes.append(
                    f"{_role} column '{_col}' is {_mf:.0%} missing — kept, but treat "
                    "the result carefully (gaps were filled/skipped).")

        # Promote a time COLUMN to the DatetimeIndex (downstream operates on the
        # index). Computed power already lives in df; nothing else to add.
        if not time_in_index and time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).set_index(time_col)

        # Comprehensive per-role candidate lists for the UI: EVERY relevant
        # column (LLM picks + name-scan), INCLUDING low-quality ones, so they all
        # appear under "LLM-detected {role}" rather than being hidden. Each gets
        # a short quality tag (e.g. 'all-zero', '94% missing', 'per-device') the
        # UI appends after the name.
        #
        # Same-side only: a DC role lists AC columns as alternatives ONLY when
        # the chosen column is itself an AC fallback (no DC column existed). When
        # a real DC column was selected we never offer an AC column beside it —
        # DC and AC power are different physical quantities, so e.g. `ac_power`
        # must not appear under DC Power. (dc_*_cands are already wrong-side
        # filtered; filter the V/I lists here too before pairing them by side.)
        dc_v_cands = _drop_wrong_side(_cands("DC Voltage"), "dc")
        ac_v_cands = _drop_wrong_side(_cands("AC Voltage"), "ac")
        dc_i_cands = _drop_wrong_side(_cands("DC Current"), "dc")
        ac_i_cands = _drop_wrong_side(_cands("AC Current"), "ac")
        _role_cands = {
            "DC Power":           (ac_p_cands + dc_p_cands) if p_side == "ac"
                                  else dc_p_cands,
            "DC Voltage":         (ac_v_cands + dc_v_cands) if v_side == "ac"
                                  else dc_v_cands,
            "DC Current":         (ac_i_cands + dc_i_cands) if i_side == "ac"
                                  else dc_i_cands,
            "Irradiance":         all_irr,
            "Module temperature": _cands("Module temperature"),
            "Time":               ([c for c in _cands("Time") if c in df.columns]
                                   + [c for c in df.columns if _name_looks_like_time(c)]),
        }
        comprehensive_alts, quality_tags = {}, {}
        for _role, _cand_list in _role_cands.items():
            seen = []
            for c in _cand_list:
                if c in df.columns and c not in seen:
                    seen.append(c)
            if seen:
                comprehensive_alts[_role] = seen
                for c in seen:
                    t = _quality_tag(df, c, _role)
                    if t:
                        quality_tags[c] = t

        # Very large datasets are thinned to fixed times-of-day BEFORE anything
        # downstream (figures, filters, degradation, the dcc.Store payload) —
        # same clock times every day, full span preserved.
        df, _ds_note = downsample_fixed_times(df, mapped_variables_dict)
        if _ds_note:
            mapping_notes.append(_ds_note)

        # Stash the per-role candidates + quality tags on the frame so the
        # Advanced UI can render them under each dropdown (set AFTER any re-index
        # so it lands on the final df object the caller receives).
        df.attrs["mapping_alternatives"] = comprehensive_alts
        df.attrs["mapping_quality_tags"] = quality_tags

        # Build the read-only mapping table for display (canonical roles).
        _p("Building data summary…")
        display_roles = ["Time", "DC Power", "DC Voltage", "DC Current",
                         "Irradiance", "Module temperature"]
        mapping_df = pd.DataFrame(
            [{"Metric": r, "Variable Name": mapped_variables_dict.get(r, "N/A")}
             for r in display_roles])

        summary_table = html.Div([
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(c) for c in mapping_df.columns])),
                    html.Tbody([
                        html.Tr([html.Td(mapping_df.iloc[i][col]) for col in mapping_df.columns])
                        for i in range(len(mapping_df))
                    ])
                ],
                className="table table-striped"
            )
        ])

        # ----------------------------------
        # Check for missing Power/Time
        # ----------------------------------
        missing_msgs = []
        if "DC Power" not in mapped_variables_dict:
            missing_msgs.append("⚠️ Power column not identified.")
        if "Time" not in mapped_variables_dict:
            missing_msgs.append("⚠️ Time column not identified.")

        if missing_msgs:
            summary_table = html.Div([
                summary_table,
                html.Div(
                    "Degradation analysis requires both Time and Power columns.",
                    className="alert alert-warning"
                ),
                html.Div([html.Div(msg) for msg in missing_msgs])
            ])

    except Exception as e:
        summary_table = html.Div(
            f"Error during LLM analysis or parsing: {e}",
            className="alert alert-warning"
        )
        mapped_variables_dict = {}
        mapping_notes = []

    return df, summary_table, mapped_variables_dict, code_read, mapping_notes


# ================================
# FIGURES OF RAW DATA
# ================================

# Module-level color palette for every physical variable rendered in the app.
# Both `make_overview_figures` (Step 1 raw-data preview) and `compute_pvpro`
# (Step 3 PVPRO trends) consume this so the user sees the same color for
# the same physical quantity no matter which step they're looking at.
#
# Convention:  blues for environmental/meteorological inputs (power,
# irradiance, temperature); greens for the electrical DC measurements
# (voltage, current).  Adjust here and the change ripples everywhere.
VAR_COLORS = {
    "power":       "#0064AB",   # navy
    "irradiance":  "#5b9bd5",   # mid blue
    "temperature": "#8ec4e8",   # light blue
    "voltage":     "#2a8e7a",   # teal
    "current":     "#9bcc4e",   # lime green
}


def make_overview_figures(df, mapped_variables_dict, temp_col="temp_C"):

    figures = []
    errors = []

    # Local alias so existing reference names in the function keep working.
    COLORS = {
        "power":      VAR_COLORS["power"],
        "irradiance": VAR_COLORS["irradiance"],
        "temp_raw":   VAR_COLORS["temperature"],
        "voltage":    VAR_COLORS["voltage"],
        "current":    VAR_COLORS["current"],
    }

    # -------------------------
    # Shared layout config -- compact: lower height, no redundant x-title.
    # The subplot title is rendered in BOLD (HTML <b>) to give it more
    # presence at this compact height.  The y-axis carries the unit only
    # (e.g. "(W)", "(°C)") to keep the left gutter narrow.
    # -------------------------
    def apply_layout(fig, title, y_label):
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.01,
                       font=dict(size=14, family="Arial", color="#0f172a")),
            template="plotly_white",
            height=130,
            margin=dict(l=50, r=20, t=28, b=24),
            xaxis_title=None,
            yaxis_title=y_label,
            hovermode="x unified",
            showlegend=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
        fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0",
                         title_font=dict(size=11))
        return fig

    # -------------------------
    # 1. Power (blue)
    # -------------------------
    try:
        power_key = mapped_variables_dict.get("DC Power")

        if not power_key:
            raise ValueError("DC Power key not found")

        if power_key not in df.columns:
            raise ValueError(f"Column '{power_key}' not found")

        fig_power = go.Figure()

        fig_power.add_trace(go.Scattergl(
            x=df.index,
            y=df[power_key],
            mode="markers",
            name="Power Output",
            opacity=0.18,
            marker=dict(color=COLORS["power"], size=4)
        ))

        fig_power = apply_layout(fig_power, "Power vs Time", "Power (W)")
        figures.append(dcc.Graph(figure=fig_power))

    except Exception as e:
        errors.append(f"[Power Plot] {str(e)}")

    # -------------------------
    # 2. Irradiance (orange)
    # -------------------------
    try:
        irr_key = mapped_variables_dict.get("Irradiance")

        if not irr_key:
            raise ValueError("Irradiance key not found")

        if irr_key not in df.columns:
            raise ValueError(f"Column '{irr_key}' not found")

        fig_irr = go.Figure()

        fig_irr.add_trace(go.Scattergl(
            x=df.index,
            y=df[irr_key],
            mode="markers",
            name="Irradiance",
            opacity=0.18,
            marker=dict(color=COLORS["irradiance"], size=4)
        ))

        fig_irr = apply_layout(fig_irr, "Irradiance", "Irrad. (W/m²)")

        # --- Apply irradiance limit ---
        ymax = df[irr_key].max()
        ymin = df[irr_key].min()

        y_lower = ymin
        y_upper = ymax

        if ymax > 1500:
            y_upper = 1500

        if ymin < 0:
            y_lower = 0

        fig_irr.update_yaxes(range=[y_lower, y_upper])

        figures.append(dcc.Graph(figure=fig_irr))

    except Exception as e:
        errors.append(f"[Irradiance Plot] {str(e)}")

    # -------------------------
    # 3. Temperature (red + purple)
    # -------------------------
    try:
        temp_raw = mapped_variables_dict.get("Module temperature")

        if not temp_raw:
            raise ValueError("Module temperature key not found")

        if temp_raw not in df.columns:
            raise ValueError(f"Column '{temp_raw}' not found")

        fig_temp = go.Figure()

        fig_temp.add_trace(go.Scattergl(
            x=df.index,
            y=df[temp_raw],
            mode="markers",
            name="Module Temp raw",
            opacity=0.18,
            marker=dict(color=COLORS["temp_raw"], size=4)
        ))

        fig_temp.update_yaxes(range=[y_lower - 20, y_upper + 20])

        fig_temp = apply_layout(fig_temp, "Temperature", "Temp (°C)")

        # --- Apply temperature limit ---
        ymax = df[temp_raw].max()
        ymin = df[temp_raw].min()

        y_lower = ymin
        y_upper = ymax

        if ymax > 150:
            y_upper = 80

        if ymin < -50:
            y_lower = -40

        fig_temp.update_yaxes(range=[y_lower - 20, y_upper + 20])

        figures.append(dcc.Graph(figure=fig_temp))

    except Exception as e:
        errors.append(f"[Temperature Plot] {str(e)}")

    # -------------------------
    # 4. DC Voltage (purple) -- optional (only if mapped, used by PVPRO)
    # -------------------------
    try:
        v_key = mapped_variables_dict.get("DC Voltage")

        if v_key and v_key in df.columns:
            fig_v = go.Figure()
            fig_v.add_trace(go.Scattergl(
                x=df.index,
                y=df[v_key],
                mode="markers",
                name="DC Voltage",
                opacity=0.18,
                marker=dict(color=COLORS["voltage"], size=4)
            ))
            fig_v = apply_layout(fig_v, "DC Voltage", "Voltage (V)")
            figures.append(dcc.Graph(figure=fig_v))

    except Exception as e:
        errors.append(f"[DC Voltage Plot] {str(e)}")

    # -------------------------
    # 5. DC Current (emerald) -- optional (only if mapped, used by PVPRO)
    # -------------------------
    try:
        i_key = mapped_variables_dict.get("DC Current")

        if i_key and i_key in df.columns:
            fig_i = go.Figure()
            fig_i.add_trace(go.Scattergl(
                x=df.index,
                y=df[i_key],
                mode="markers",
                name="DC Current",
                opacity=0.18,
                marker=dict(color=COLORS["current"], size=4)
            ))
            fig_i = apply_layout(fig_i, "DC Current", "Current (A)")
            figures.append(dcc.Graph(figure=fig_i))

    except Exception as e:
        errors.append(f"[DC Current Plot] {str(e)}")

    return figures, errors


# ================================
# NORMALIZATION
# ================================
def normalize(df, mapped_variables_dict, gamma=-0.004):
    # Adaptive normalization based on which columns are available:
    #   power + irradiance + temp -> full temperature-corrected normalization
    #   power + irradiance         -> irradiance-only normalization
    #   power alone                -> raw power (no normalization possible)
    power_key  = mapped_variables_dict.get("DC Power")
    irr_key    = mapped_variables_dict.get("Irradiance")
    temp_C_key = mapped_variables_dict.get("Module temperature")

    has_irr  = bool(irr_key) and irr_key in df.columns
    has_temp = bool(temp_C_key) and temp_C_key in df.columns

    if has_irr and has_temp:
        df['norm'] = df[power_key] / (
            df[irr_key] * (1 + gamma * (df[temp_C_key] - 25)))*1000
        df.loc[df[irr_key] < 50, 'norm'] = np.nan
    elif has_irr:
        df['norm'] = df[power_key] / df[irr_key] * 1000
        df.loc[df[irr_key] < 50, 'norm'] = np.nan
    else:
        # No irradiance: trend the raw power directly (un-normalized).
        df['norm'] = df[power_key]

    return df


# ================================
# Low irradiance & power filter
# ================================
def low_irra_power_filter(df, mapped_variables_dict,
                          irr_thresh=300, power_ratio=0.02,
                          norm_lower=0.01, norm_upper_pct=99):
    mask = pd.Series(True, index=df.index)

    irr_key = mapped_variables_dict["Irradiance"]
    power_key = mapped_variables_dict["DC Power"]

    # irradiance filter
    mask &= df[irr_key] > irr_thresh

    # Power filter — UNIT-INDEPENDENT. The old form (power > ratio * irradiance)
    # compared the power column's native units against W/m², which only worked
    # when power happened to be in watts; a kW-scaled file (e.g. DKASC) failed
    # for every daytime row and 0% survived. Scale by the system's own
    # spike-robust peak instead: at full sun (1000 W/m²) a point must produce
    # at least `power_ratio` of peak power, proportionally less at lower sun.
    p = pd.to_numeric(df[power_key], errors="coerce")
    capacity = p.quantile(0.99)
    if np.isfinite(capacity) and capacity > 0:
        mask &= p > power_ratio * (df[irr_key] / 1000.0) * capacity

    # norm range filter
    upper = df['norm'].quantile(norm_upper_pct / 100)
    mask &= df['norm'].between(norm_lower, upper)

    # ✅ indices
    normal_indices = df.index[mask]
    outlier_indices = df.index[~mask]

    return normal_indices, outlier_indices

# ================================
# DAILY AGGREGATION
# ================================
def aggregate_daily(df_f, irradiance_col=None):
    # Group by the post-dropna frame's OWN index. Grouping by df_f.index.date
    # (the full, original-length index) breaks with "Grouper and axis must be
    # same length" whenever dropna() removes rows -- which happens on any real
    # data that has gaps/NaNs.
    if irradiance_col and irradiance_col in df_f.columns:
        # Irradiance available: irradiance-weighted daily mean of 'norm'.
        sub = df_f[['norm', irradiance_col]].dropna()
        daily = (
            sub
            .groupby(sub.index.date)
            .apply(lambda x: np.sum(x['norm'] * x[irradiance_col]) / np.sum(x[irradiance_col]))
        )
        daily.index = pd.to_datetime(daily.index)
        return daily

    # No irradiance: daily PEAK power, then drop low-peak (cloudy / partial /
    # sparsely-sampled) days so the trend is built from comparable clear days.
    sub = df_f[['norm']].dropna()
    daily = sub.groupby(sub.index.date)['norm'].quantile(PEAK_QUANTILE)
    daily.index = pd.to_datetime(daily.index)

    if len(daily) >= CLEAR_DAY_MIN_DAYS:
        reference_peak = daily.quantile(0.90)   # robust "clear-sky" peak level
        if reference_peak and np.isfinite(reference_peak):
            daily = daily[daily >= CLEAR_DAY_FRAC * reference_peak]

    return daily

# ================================
# YoY
# ================================
def compute_yoy(series, eps=1e-6, rolling_window=30, iqr_multiplier=1.5,
                tolerance_days=15):
    series = series.dropna().sort_index()
    yoy = []

    # Denominator guard: a year-over-year ratio is curr/prev, so a small `prev`
    # explodes it (e.g. a 257 W partial-day vs a normal day -> +465). Skip any
    # pair whose prior-year value is implausibly low relative to the series'
    # typical level, not just ~zero. This kills the artifact where un-normalized
    # low-output days act as tiny denominators and send the rate to hundreds %.
    median_level = float(np.median(series.values)) if len(series) else 0.0
    prev_floor = max(eps, DENOM_FLOOR_FRAC * median_level)

    # Pair each point with the value closest to exactly one year earlier,
    # accepting any match within +/- tolerance_days rather than requiring the
    # exact same calendar date. This makes YoY robust to irregular/sparse daily
    # sampling, where the identical calendar date rarely recurs across years.
    targets = series.index - pd.DateOffset(years=1)
    nearest_pos = series.index.get_indexer(
        targets, method="nearest", tolerance=pd.Timedelta(days=tolerance_days)
    )

    for i in range(len(series)):
        j = nearest_pos[i]
        if j == -1:
            continue  # no data point within tolerance of one year earlier

        prev = series.iloc[j]
        curr = series.iloc[i]

        if prev < prev_floor:
            continue  # prior-year value too low to be a trustworthy denominator

        ratio = curr / prev - 1

        if np.isfinite(ratio):
            yoy.append(ratio)

    yoy = np.array(yoy)

    # --- Remove outliers using IQR ---
    if len(yoy) > 0:
        q1 = np.percentile(yoy, 25)
        q3 = np.percentile(yoy, 75)
        iqr = q3 - q1

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        yoy = yoy[(yoy >= lower) & (yoy <= upper)]

    rd = np.median(yoy) * 100 if len(yoy) > 0 else np.nan

    # ===============
    # plot - YOY
    # ===============

    trend = series.rolling(rolling_window, center=True).mean()

    fig = go.Figure()

    # Daily points
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#A6CAEC"),
            name="Daily-aggragated Power"
        )
    )

    # Trend line
    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend,
            mode="lines",
            line=dict(color="#0070C0", width=2),
            name=f"Trend ({rolling_window}-day rolling)"
        )
    )

    fig.update_layout(
        title="Power Trend",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return rd, fig


# ================================
# LR
# ================================
def compute_lr(series):
    series = series.dropna()

    # Mathematical minimum for a slope. Too-few-points still produces a number;
    # the caller flags it as unreliable (see degradation_reliability) rather than
    # the tool refusing outright.
    if len(series) < 2:
        return np.nan, None

    t = _time_to_years(series.index).values
    y = series.values

    # Theil-Sen (median of pairwise slopes) instead of ordinary least squares:
    # OLS is dragged by a few outlier days (e.g. a handful of high-output days at
    # the end read as steep "improvement"), while the median slope is robust to
    # them and recovers the true gentle trend.
    from scipy import stats as _scipy_stats
    slope, intercept, _lo, _hi = _scipy_stats.theilslopes(y, t)
    trend = slope * t + intercept

    rd = slope / np.mean(y) * 100

    fig = go.Figure()

    # Raw data (scatter)
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#A6CAEC"),
            name="Daily-aggragated Power"
        )
    )

    # Trend line
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=trend,
            mode="lines",
            line=dict(color="#0070C0", width=2),
            name=f"LR Trend ({rd:.2f}%/yr)"
        )
    )

    fig.update_layout(
        title="Power and Linear Regression Trend",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return rd, fig


# ================================
# HW
# ================================
def compute_hw(series, period=12):
    series = series.dropna()

    if len(series) < 2 * period:
        return np.nan, None

    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=period
    ).fit()

    fitted = model.fittedvalues

    t = _time_to_years(fitted.index).values.reshape(-1, 1)
    y = fitted.values

    lr = LinearRegression().fit(t, y)
    slope = lr.coef_[0]
    rd = slope / np.mean(y)*100

    fig = go.Figure()

    # Raw data
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#A6CAEC"),
            name="Daily-aggragated Power"
        )
    )

    # HW fitted line
    fig.add_trace(
        go.Scatter(
            x=fitted.index,
            y=fitted.values,
            mode="lines",
            line=dict(color="#0070C0", width=2),
            name=f"HW Fit ({rd:.2f}%/yr)"
        )
    )

    fig.update_layout(
        title="Power and Holt-Winters Trend",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return rd, fig


# ================================
# ARIMA
# ================================

def compute_arima(series, order=(1,1,0), seasonal_order=(0,1,1,12), p=1, d=1, q=0, seasonal_period=12):
    # Allow flat params to override tuple args
    order = (p, d, q)
    seasonal_order = (0, 1, 1, seasonal_period)
    series = series.dropna()

    if len(series) < 24:
        return np.nan, None

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    res = model.fit(disp=False)

    fitted = res.fittedvalues

    t = _time_to_years(fitted.index).values.reshape(-1, 1)
    y = fitted.values

    lr = LinearRegression().fit(t, y)
    slope = lr.coef_[0]
    rd = slope / np.mean(y) * 100

    fig = go.Figure()

    # Raw data (scatter)
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#A6CAEC"),
            name="Daily-aggragated Power"
        )
    )

    # ARIMA fitted line
    fig.add_trace(
        go.Scatter(
            x=fitted.index,
            y=fitted.values,
            mode="lines",
            line=dict(color="#0070C0", width=2),
            name=f"ARIMA Fit ({rd:.2f}%/yr)"
        )
    )

    fig.update_layout(
        title="Power and ARIMA Trend",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    return rd, fig


# ================================
# CSD
# ================================
def compute_csd(series, period=12):
    series = series.dropna()

    if len(series) < 2 * period:
        return np.nan, None

    decomposition = seasonal_decompose(series, model='additive', period=period)
    trend = decomposition.trend.dropna()

    t = _time_to_years(trend.index).values.reshape(-1, 1)
    y = trend.values

    model = LinearRegression().fit(t, y)
    slope = model.coef_[0]
    rd = slope / np.mean(y)*100

    trend = decomposition.trend

    fig = go.Figure()

    # Raw data
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#A6CAEC"),
            name="Daily-aggragated Power"
        )
    )

    # Trend (drop NaNs from edges)
    fig.add_trace(
        go.Scatter(
            x=trend.dropna().index,
            y=trend.dropna().values,
            mode="lines",
            line=dict(color="#0070C0", width=2),
            name=f"CSD Trend ({rd:.2f}%/yr)"
        )
    )

    fig.update_layout(
        title="Power and CSD Trend",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return rd, fig

# ================================
# get full code
# ================================
def get_full_code(filename, mapped_variables_dict,selected_filters, selected_metric):

    with open("page_supporting_files/pvcopilot_functions_code.txt", "r", encoding="utf-8") as f:
            pvcopilot_functions_code = f.read().replace('"', "'")

    with open("page_supporting_files/pvcopilot_packages_code.txt", "r", encoding="utf-8") as f:
            pvcopilot_packages_code = f.read().replace('"', "'")

    with open("page_supporting_files/pvcopilot_main_code.txt", "r", encoding="utf-8") as f:
            main_code = f.read().replace('"', "'")

    prompt = f"""
        Your task is to generate Python code and return it as a JSON object with TWO keys:
        - "packages_code": only import / package-related code
        - "main_code": the rest of the logic

        Requirements:

        1) packages_code:
        - include packages: 
        - import pandas as pd
        - import numpy as np

        2) main_code:
        - add comments like 'Main code'
        - load data as df where filename is {filename}
        (add a comment that user may need to provide full file path)
        - define a dict 'mapped_variables_dict' from:
        {mapped_variables_dict}
        - use function:
            df_filtered = normalize(df, mapped_variables_dict)

        - define selected_filters (list) from {selected_filters}
        
        - if "low-irra-power" in selected_filters:
            use:
            normal_idx, outlier_idx = low_irra_power_filter(df_filtered, mapped_variables_dict)

        - if "outlier" in selected_filters:
            use:
            normal_idx, outlier_idx = identify_outliers_iqr(df_filtered, "norm")

        - merge all normal_idx
        - print:
            * total number of points
            * number of normal points
            * number of outliers

        - define:
            df_filtered_final = df_filtered.loc[normal_idx]

        - use:
            daily_data = aggregate_daily(df_filtered_final, irra_key)

        - define selected_metric (list) from {selected_metric}

        - depending on selected_metric:
            if 'YOY':
                rd, _ = compute_yoy(daily_data)
            if 'LR':
                rd, _ = compute_lr(daily_data)
            if 'ARIMA':
                rd, _ = compute_arima(daily_data)
            if 'CSD':
                rd, _ = compute_csd(daily_data)
            if 'HW':
                rd, _ = compute_hw(daily_data)

        - print rd

        Notes:
        - all functions are already defined, just call them
        - add concise comments for each step
        - do NOT include imports in main_code
        - do NOT include explanations outside the code

        Return format EXACTLY:
        {{
        "packages_code": "...",
        "main_code": "..."
        }}
        """

    # Call LLM
    # start_time = time.time()

    # response = client.chat.completions.create(
    #     # model="openai/gpt-4.1-mini",
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=1000,
    # )

    # end_time = time.time()


    # print(f"Time taken: {end_time - start_time:.2f} seconds")

    # res_text = response.choices[0].message.content.strip()

    # # Remove possible markdown wrappers
    # clean_text = re.sub(r"```json\n(.*?)```", r"\1", res_text, flags=re.DOTALL).strip()
    # clean_text = re.sub(r"```python\n(.*?)```", r"\1", clean_text, flags=re.DOTALL).strip()

    # Parse JSON
    # try:
    #     parsed = json.loads(clean_text)
    #     packages_code = parsed.get("packages_code", "")
    #     main_code = parsed.get("main_code", "")
    #     full_code = packages_code + "\n\n" + pvcopilot_functions_code + "\n\n" + main_code

    if 'csv' in filename:
        code_read = f"df = pd.read_csv('{filename}')"

    elif 'xls' in filename or 'xlsx' in filename:
        code_read = f"df = pd.read_excel('{filename}')"

    elif 'parquet' in filename:
        code_read = f"df = pd.read_parquet('{filename}')"

    code_variable_mapping = f"mapped_variables_dict = {mapped_variables_dict}"

    try:
        full_code = (
            pvcopilot_packages_code + "\n\n"
            + pvcopilot_functions_code + "\n\n"
            + code_read + "\n\n"
            + code_variable_mapping + "\n\n"
            + main_code
        )

    except json.JSONDecodeError:
        raise ValueError("Response is not valid JSON")

    # with open("llm_response.txt", "w", encoding="utf-8") as f:
    #     f.write(clean_text)

    return full_code


# =============================================================================
# PVPRO-LITE
#
# A self-contained, lightweight implementation of the PVPRO degradation
# analysis approach (Meyers et al., IEEE JPV 10(2), 2020).  The full pvpro
# package depends on solar-data-tools, rdtools, pvanalytics, pvlib and MOSEK;
# this version is implemented in pure NumPy / SciPy / scikit-learn so that the
# web app has zero extra dependencies beyond what the other compute_* methods
# already need.
#
# What we keep (the science):
#   1. De Soto single-diode-model translation of reference parameters
#      (IL_ref, I0_ref, Rs_ref, Rsh_ref, n) to operating conditions (G, T_cell).
#   2. Bishop88-style maximum-power-point solve via SciPy's Brent root-finder.
#   3. Iteration of short (e.g. 14-day) windows over the dataset, fitting the
#      five reference SDM parameters in each window by minimising the L2
#      residual between predicted and measured (V_mp, I_mp).
#   4. Year-over-year degradation extraction (rdtools-style) on the resulting
#      P_mp,ref time series.
#
# What we drop (the heavy infrastructure):
#   - solar-data-tools data classification & clear-sky detection (we rely on
#     Step 2's filters in the web app instead).
#   - pvanalytics operating-mode classifier (we assume every kept point is at
#     MPP after Step 2 filtering).
#   - rdtools.degradation_year_on_year (re-implemented inline, ~15 lines).
#   - pvlib calcparams_desoto / bishop88_mpp (re-implemented inline).
#   - MOSEK (no convex optimisation needed; L-BFGS-B is enough for SDM fits).
# =============================================================================
from scipy.optimize import minimize as _scipy_minimize, brentq as _scipy_brentq

# ----- Physical constants -----
_Q_E   = 1.602176634e-19       # electron charge, C
_K_B   = 1.380649e-23          # Boltzmann constant, J/K
_T_REF = 25.0 + 273.15         # STC cell temperature, K
_G_REF = 1000.0                # STC irradiance, W/m^2

# ----- Technology lookup (same values as pvpro.modeling.estimate_Eg_dEgdT) ---
_TECH_TABLE = {
    "mono-c-Si":  (1.121,   -0.0002677),
    "multi-c-Si": (1.121,   -0.0002677),
    "GaAs":       (1.424,   -0.000433),
    "CIGS":       (1.15,    -0.00001),
    "CdTe":       (1.475,   -0.0003),
}


def _estimate_Eg_dEgdT(technology):
    if technology not in _TECH_TABLE:
        raise ValueError(
            f"Unknown technology '{technology}'. "
            f"Valid choices: {sorted(_TECH_TABLE)}."
        )
    return _TECH_TABLE[technology]


def _calcparams_desoto_lite(effective_irradiance,
                            temperature_cell_C,
                            photocurrent_ref,
                            saturation_current_ref,
                            resistance_series_ref,
                            resistance_shunt_ref,
                            diode_factor,
                            cells_in_series,
                            alpha_isc,
                            Eg_ref,
                            dEgdT):
    """
    Translate reference (STC) single-diode-model parameters to the operating
    irradiance and temperature, following the standard De Soto / pvlib
    formulation.

    Returns
    -------
    IL, I0, Rs, Rsh, nNsVth : ndarray
        SDM coefficients at each operating point (vectorised over time).
    """
    G    = np.asarray(effective_irradiance, dtype=float)
    Tc   = np.asarray(temperature_cell_C,   dtype=float) + 273.15  # to Kelvin

    # Band gap shifts linearly with temperature: Eg = Eg_ref * (1 + dEg/dT * dT).
    dT  = Tc - _T_REF
    Eg  = Eg_ref * (1.0 + dEgdT * dT)

    # Light-generated current: linear in G, plus a small temperature term.
    IL  = (G / _G_REF) * (photocurrent_ref + alpha_isc * dT)

    # Saturation current: cubic-T scaling plus an Eg/kT exponential.
    Vth_ratio = (_Q_E / _K_B) * (Eg_ref / _T_REF - Eg / Tc)
    I0 = saturation_current_ref * (Tc / _T_REF) ** 3 * np.exp(Vth_ratio)

    # Series resistance is temperature/irradiance-independent in De Soto.
    Rs = np.full_like(G, resistance_series_ref, dtype=float)

    # Shunt resistance scales as G_ref / G (with a tiny extra conductance to
    # keep it bounded at very low irradiance — same trick pvpro uses).
    Gsh_extra = 1e-5
    Rsh_inv = (G / _G_REF) / resistance_shunt_ref + Gsh_extra
    Rsh = 1.0 / Rsh_inv

    # nNsVth: ideality factor × cells in series × thermal voltage.
    nNsVth = diode_factor * cells_in_series * (_K_B * Tc / _Q_E)

    return IL, I0, Rs, Rsh, nNsVth


def _bishop88_iv_at_vd(Vd, IL, I0, Rs, Rsh, nNsVth):
    """Current and terminal voltage as a function of the diode voltage Vd."""
    # I = IL - I0*(exp(Vd/nNsVth) - 1) - Vd/Rsh
    # The clip on Vd/nNsVth prevents overflow when L-BFGS-B explores ugly x.
    arg = np.clip(Vd / nNsVth, -100.0, 100.0)
    I_diode = I0 * (np.exp(arg) - 1.0)
    I_term  = IL - I_diode - Vd / Rsh
    V_term  = Vd - I_term * Rs
    return V_term, I_term


def _bishop88_dpdvd(Vd, IL, I0, Rs, Rsh, nNsVth):
    """dP/dVd of the terminal power, used as the root for MPP."""
    # P = V*I where both depend on Vd.  dP/dVd = dV/dVd * I + V * dI/dVd.
    arg = np.clip(Vd / nNsVth, -100.0, 100.0)
    g   = I0 / nNsVth * np.exp(arg)            # dI_diode/dVd
    # dI_term/dVd  = -g - 1/Rsh
    dI = -g - 1.0 / Rsh
    # dV_term/dVd = 1 - Rs*dI/dVd
    dV = 1.0 - Rs * dI
    V_term, I_term = _bishop88_iv_at_vd(Vd, IL, I0, Rs, Rsh, nNsVth)
    return dV * I_term + V_term * dI


def _mpp_one_point(IL, I0, Rs, Rsh, nNsVth):
    """Brent root-find for a single operating point. Returns (Vmp, Imp).

    This scalar path is kept as a fallback / debugging aid.  The fitting
    loop uses the vectorised Newton solver below.
    """
    if IL <= 0:
        return 0.0, 0.0
    Voc_est = nNsVth * np.log(max(IL / I0, 1.0) + 1.0)
    if not np.isfinite(Voc_est) or Voc_est <= 0:
        return 0.0, 0.0
    lo, hi = 1e-6, Voc_est * 1.001
    try:
        f_lo = _bishop88_dpdvd(lo, IL, I0, Rs, Rsh, nNsVth)
        f_hi = _bishop88_dpdvd(hi, IL, I0, Rs, Rsh, nNsVth)
        if f_lo * f_hi > 0:
            for hi in (Voc_est * 1.5, Voc_est * 2.0, Voc_est * 5.0):
                f_hi = _bishop88_dpdvd(hi, IL, I0, Rs, Rsh, nNsVth)
                if f_lo * f_hi <= 0:
                    break
            else:
                return 0.0, 0.0
        Vd_star = _scipy_brentq(_bishop88_dpdvd, lo, hi,
                                args=(IL, I0, Rs, Rsh, nNsVth),
                                xtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        return 0.0, 0.0
    V, I = _bishop88_iv_at_vd(Vd_star, IL, I0, Rs, Rsh, nNsVth)
    return float(V), float(I)


def _mpp_vectorised(IL, I0, Rs, Rsh, nNsVth, n_iter=20):
    """Vectorised MPP solver via Newton's method on dP/dVd = 0.

    All inputs are 1-D arrays of equal length.  For each point we iterate
    Vd ← Vd − f(Vd)/f'(Vd) where f = dP/dVd, then evaluate V_term, I_term
    at the converged Vd.  Newton is well-behaved on this problem when
    started from V_oc/2, which is the standard textbook initial guess.

    n_iter=20 is a safe upper bound; convergence typically happens in 5–8
    iterations and the extras cost almost nothing on vectors.
    """
    IL  = np.asarray(IL,  dtype=float)
    I0  = np.asarray(I0,  dtype=float)
    Rs  = np.asarray(Rs,  dtype=float)
    Rsh = np.asarray(Rsh, dtype=float)
    nNsVth = np.asarray(nNsVth, dtype=float)

    # Open-circuit voltage estimate -> initial guess Vd ~ Voc/2.
    with np.errstate(divide="ignore", invalid="ignore"):
        Voc_est = nNsVth * np.log(np.maximum(IL / np.maximum(I0, 1e-30), 1.0) + 1.0)
    valid = (IL > 0) & np.isfinite(Voc_est) & (Voc_est > 0)

    Vd = np.where(valid, 0.7 * Voc_est, 0.0)

    # Vectorised Newton iteration on f(Vd) = dP/dVd.
    for _ in range(n_iter):
        arg = np.clip(Vd / np.maximum(nNsVth, 1e-30), -100.0, 100.0)
        exp_a = np.exp(arg)
        # I_term and its Vd-derivative
        I_diode = I0 * (exp_a - 1.0)
        I_term  = IL - I_diode - Vd / Rsh
        # d/dVd of I_diode = I0/nNsVth * exp(Vd/nNsVth)
        g = I0 / np.maximum(nNsVth, 1e-30) * exp_a
        dI = -g - 1.0 / Rsh
        # V_term and its Vd-derivative
        V_term = Vd - I_term * Rs
        dV = 1.0 - Rs * dI
        # f(Vd) = dP/dVd, where P = V*I.
        f = dV * I_term + V_term * dI
        # f'(Vd) = d2V/dVd2 * I + 2 dV dI + V d2I/dVd2.
        # d2I/dVd2 = -g/nNsVth.   d2V/dVd2 = -Rs * d2I/dVd2 = Rs*g/nNsVth.
        d2I = -g / np.maximum(nNsVth, 1e-30)
        d2V =  Rs * g / np.maximum(nNsVth, 1e-30)
        f_prime = d2V * I_term + 2.0 * dV * dI + V_term * d2I
        # Newton step, guarded against f' ~ 0.
        step = np.where(np.abs(f_prime) > 1e-30, f / f_prime, 0.0)
        Vd_new = Vd - step
        # Keep Vd in (0, 1.05 * Voc_est).
        Vd = np.where(valid, np.clip(Vd_new, 1e-6, 1.05 * Voc_est), Vd)

    # Final evaluation.
    arg = np.clip(Vd / np.maximum(nNsVth, 1e-30), -100.0, 100.0)
    I_diode = I0 * (np.exp(arg) - 1.0)
    I_term  = IL - I_diode - Vd / Rsh
    V_term  = Vd - I_term * Rs

    # Invalid points -> (0, 0).
    V_term = np.where(valid, V_term, 0.0)
    I_term = np.where(valid, I_term, 0.0)
    return V_term, I_term


def _voc_vectorised(IL, I0, Rs, Rsh, nNsVth, n_iter=20):
    """Open-circuit voltage. Solves 0 = IL - I0*(exp(Voc/nNsVth) - 1) - Voc/Rsh
    by Newton iteration. R_s does not appear (current is zero).
    """
    IL  = np.asarray(IL,  dtype=float)
    I0  = np.asarray(I0,  dtype=float)
    Rsh = np.asarray(Rsh, dtype=float)
    nNsVth = np.asarray(nNsVth, dtype=float)

    # Initial guess: ignore shunt branch -> Voc ~ nNsVth * ln(IL/I0 + 1).
    with np.errstate(divide="ignore", invalid="ignore"):
        V = nNsVth * np.log(np.maximum(IL / np.maximum(I0, 1e-30), 0.0) + 1.0)
    valid = (IL > 0) & np.isfinite(V) & (V > 0)
    V = np.where(valid, V, 0.0)

    for _ in range(n_iter):
        arg = np.clip(V / np.maximum(nNsVth, 1e-30), -100.0, 100.0)
        ea  = np.exp(arg)
        f   = IL - I0 * (ea - 1.0) - V / Rsh
        # df/dV = -I0/nNsVth * exp(V/nNsVth) - 1/Rsh
        fp  = -I0 / np.maximum(nNsVth, 1e-30) * ea - 1.0 / Rsh
        step = np.where(np.abs(fp) > 1e-30, f / fp, 0.0)
        V = np.where(valid, np.maximum(V - step, 1e-6), V)

    return np.where(valid, V, 0.0)


def _isc_vectorised(IL, I0, Rs, Rsh, nNsVth, n_iter=20):
    """Short-circuit current. Solves Isc = IL - I0*(exp(Isc*Rs/nNsVth) - 1)
    - Isc*Rs/Rsh by Newton iteration. V at the terminals is zero, but a
    small voltage drop Isc*Rs lives across the diode.
    """
    IL  = np.asarray(IL,  dtype=float)
    I0  = np.asarray(I0,  dtype=float)
    Rs  = np.asarray(Rs,  dtype=float)
    Rsh = np.asarray(Rsh, dtype=float)
    nNsVth = np.asarray(nNsVth, dtype=float)

    valid = IL > 0
    # First-order guess assuming Rs is small: Isc ~ IL.
    I = np.where(valid, IL, 0.0)

    for _ in range(n_iter):
        arg = np.clip(I * Rs / np.maximum(nNsVth, 1e-30), -100.0, 100.0)
        ea  = np.exp(arg)
        f   = I - IL + I0 * (ea - 1.0) + I * Rs / Rsh
        # df/dI = 1 + I0*Rs/nNsVth * exp(I*Rs/nNsVth) + Rs/Rsh
        fp  = 1.0 + I0 * Rs / np.maximum(nNsVth, 1e-30) * ea + Rs / Rsh
        step = np.where(np.abs(fp) > 1e-30, f / fp, 0.0)
        I = np.where(valid, np.maximum(I - step, 0.0), I)

    return np.where(valid, I, 0.0)


# ---------- numerical parameter transforms (same as pvpro for stability) -----
def _p_to_x(p, key):
    """Physical parameter -> numerical fit variable."""
    if key == "saturation_current_ref":
        return np.log(p) + 23.0
    if key == "resistance_shunt_ref":
        return np.log(p) / 2.0 + 1.0
    if key == "resistance_series_ref":
        return p * 2.2
    return p


def _x_to_p(x, key):
    """Numerical fit variable -> physical parameter."""
    if key == "saturation_current_ref":
        return float(np.exp(x - 23.0))
    if key == "resistance_shunt_ref":
        return float(np.exp(2.0 * (x - 1.0)))
    if key == "resistance_series_ref":
        return float(x / 2.2)
    return float(x)


# ---------- single-window fit ------------------------------------------------
_FIT_PARAMS = (
    "photocurrent_ref",
    "saturation_current_ref",
    "resistance_series_ref",
    "resistance_shunt_ref",
    "diode_factor",
)


def _predict_vi(params, G, T, cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """Run the SDM forward for an entire window given reference parameters."""
    IL, I0, Rs, Rsh, nNsVth = _calcparams_desoto_lite(
        G, T,
        photocurrent_ref       = params["photocurrent_ref"],
        saturation_current_ref = params["saturation_current_ref"],
        resistance_series_ref  = params["resistance_series_ref"],
        resistance_shunt_ref   = params["resistance_shunt_ref"],
        diode_factor           = params["diode_factor"],
        cells_in_series        = cells_in_series,
        alpha_isc              = alpha_isc,
        Eg_ref                 = Eg_ref,
        dEgdT                  = dEgdT,
    )
    V_pred, I_pred = _mpp_vectorised(IL, I0, Rs, Rsh, nNsVth)
    return V_pred, I_pred


def _loss(x, fit_params, G, T, V_meas, I_meas,
          V_scale, I_scale,
          cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """L2 loss in scaled (V, I) coords, same form as the original PVPRO loss."""
    params = {k: _x_to_p(x[i], k) for i, k in enumerate(fit_params)}
    V_pred, I_pred = _predict_vi(params, G, T,
                                 cells_in_series, alpha_isc, Eg_ref, dEgdT)
    v_err = (V_pred - V_meas) / V_scale
    i_err = (I_pred - I_meas) / I_scale
    return float(np.nanmean(v_err ** 2 + i_err ** 2))


def _fit_window(G, T, V_meas, I_meas,
                p0, lower_bounds, upper_bounds,
                cells_in_series, alpha_isc, Eg_ref, dEgdT,
                saturation_current_multistart=(0.2, 0.5, 1.0, 2.0, 5.0)):
    """Fit the five reference SDM parameters in one window via L-BFGS-B with
    multistart over the saturation-current start value (mirrors pvpro)."""
    fit_params = list(_FIT_PARAMS)

    x_lo = np.array([_p_to_x(lower_bounds[k], k) for k in fit_params])
    x_hi = np.array([_p_to_x(upper_bounds[k], k) for k in fit_params])
    bounds = list(zip(x_lo, x_hi))

    Io_ref_seed = p0["saturation_current_ref"]
    V_scale = max(float(np.nanmedian(V_meas)), 1e-3)
    I_scale = max(float(np.nanmedian(I_meas)), 1e-3)

    best = None
    best_loss = np.inf
    for mult in saturation_current_multistart:
        p0_try = dict(p0)
        p0_try["saturation_current_ref"] = Io_ref_seed * mult
        x0 = np.array([_p_to_x(p0_try[k], k) for k in fit_params])
        # Clip x0 into bounds.
        x0 = np.minimum(np.maximum(x0, x_lo), x_hi)
        try:
            res = _scipy_minimize(
                _loss, x0=x0, bounds=bounds, method="L-BFGS-B",
                args=(fit_params, G, T, V_meas, I_meas, V_scale, I_scale,
                      cells_in_series, alpha_isc, Eg_ref, dEgdT),
                options={"maxiter": 80, "ftol": 1e-7, "disp": False},
            )
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < best_loss:
            best = res
            best_loss = res.fun
        # Yield the GIL momentarily between multistart attempts so a
        # concurrent Dash polling callback can read progress without
        # waiting for the entire window's fits to finish.
        time.sleep(0)

    if best is None:
        return None
    fit = {k: _x_to_p(best.x[i], k) for i, k in enumerate(fit_params)}
    fit["loss"] = float(best.fun)
    return fit


# ---------- simple starting-point estimator ----------------------------------
def _estimate_p0_simple(G, T, V, I,
                        cells_in_series, alpha_isc, Eg_ref, dEgdT):
    """Quick-and-dirty initial-guess routine that does NOT need pvanalytics.

    Uses the highest-irradiance subset (~top 10 %) as approximate STC points,
    then derives the standard textbook seeds:
        photocurrent_ref      ~ I_mp(STC) (Isc ~ Imp at high G)
        saturation_current_ref ~ I_mp * exp(-V_mp / nNsVth) / Ns
        resistance_series_ref  = 0.4
        resistance_shunt_ref   = 600
        diode_factor           = 1.0
    """
    if len(G) < 10:
        return None
    g_hi = np.nanquantile(G, 0.9)
    mask = G >= g_hi
    if mask.sum() < 5:
        mask = np.ones_like(G, dtype=bool)
    Vmp_med = float(np.nanmedian(V[mask]))
    Imp_med = float(np.nanmedian(I[mask]))
    Gmed    = float(np.nanmedian(G[mask]))
    if Vmp_med <= 0 or Imp_med <= 0 or Gmed <= 0:
        return None

    # Scale photocurrent up to STC: I_L,ref ~ Imp * (G_ref / G_obs)
    IL_ref_guess = Imp_med * (_G_REF / Gmed)

    # Rough saturation current from V_mp,ref ~ nNsVth * ln(IL/I0)
    n_guess = 1.03
    nNsVth_ref = n_guess * cells_in_series * (_K_B * _T_REF / _Q_E)
    # Scale Vmp to STC roughly (drop temperature dependence): Vmp,ref ~ Vmp + small offset
    Vmp_ref_guess = Vmp_med
    I0_ref_guess = max(IL_ref_guess * np.exp(-Vmp_ref_guess / nNsVth_ref), 1e-13)

    return dict(
        photocurrent_ref       = float(np.clip(IL_ref_guess, 0.1, 20.0)),
        saturation_current_ref = float(np.clip(I0_ref_guess, 1e-13, 1e-5)),
        resistance_series_ref  = 0.4,
        resistance_shunt_ref   = 600.0,
        diode_factor           = n_guess,
    )


# ---------- rdtools-style year-on-year degradation ---------------------------
def _yoy_degradation(series, eps=1e-9):
    """Year-on-year degradation rate, in %/yr, computed rdtools-style.

    For every timestamp t, compute the relative change to the value at t+365d
    (when available).  The reported rate is the median of all such pairs.
    """
    s = series.dropna()
    if len(s) < 4:
        return np.nan
    idx = pd.to_datetime(s.index)
    vals = s.values
    ratios = []
    one_year = pd.Timedelta(days=365)
    half_window = pd.Timedelta(days=15)  # tolerate ±15 days for nearest match
    for k in range(len(idx)):
        target = idx[k] + one_year
        # nearest neighbour search
        diffs = np.abs((idx - target).total_seconds())
        j = int(np.argmin(diffs))
        if abs(idx[j] - target) <= half_window and vals[k] > eps:
            ratios.append(vals[j] / vals[k] - 1.0)
    if not ratios:
        return np.nan
    return float(np.median(ratios) * 100.0)  # %/yr


# ---------------------------------------------------------------------------
# PVPRO parameter estimation FROM THE DATA ITSELF.
#
# Typical crystalline-module operating values used to back out the array
# layout from measured string voltage/current. Rough by nature -- the UI
# marks these as auto-filled estimates the user should review.
# ---------------------------------------------------------------------------
VMP_PER_CELL = 0.51      # V at max power per crystalline cell (standard)
MODULE_IMP_TYP = 8.5     # A at max power for a typical crystalline string
VI_P_TOL = 0.35          # measured V*I must agree with measured P within 35%


def estimate_pvpro_params(df, mapped_variables_dict, cells_in_series=60):
    """Estimate PVPRO array-layout parameters FROM THE MEASUREMENTS.

    All electrical quantities are measured from the dataset itself: the median
    voltage/current/power over the TOP-DECILE power points (the array's actual
    max-power operating region). The layout then follows arithmetically:

        modules_per_string = V_measured / (cells_in_series x 0.51 V/cell)
        parallel_strings   = I_measured / I_string
            where I_measured is the measured current, or P_measured/V_measured
            when no current column exists.

    Before estimating, the measurements are cross-checked against each other:
    if measured V x I disagrees with measured P by more than 35%, the columns
    are inconsistent (e.g. an AC current fallback on a DC-side power) and NO
    estimate is produced -- better empty than wrong.

    Returns {param_name: {"value": int, "basis": str}} for the parameters the
    data supports; {} when it can't be done honestly.
    """
    out = {}
    pcol = mapped_variables_dict.get("DC Power")
    vcol = mapped_variables_dict.get("DC Voltage")
    icol = mapped_variables_dict.get("DC Current")
    if not pcol or pcol not in df.columns:
        return out
    p = pd.to_numeric(df[pcol], errors="coerce")
    p_hi_cut = p.quantile(0.90)
    if not np.isfinite(p_hi_cut) or p_hi_cut <= 0:
        return out
    high = p >= p_hi_cut            # the array's real max-power operating region
    if int(high.sum()) < 30:
        return out
    p_med = p[high].median()

    v_med = None
    if vcol and vcol in df.columns:
        v = pd.to_numeric(df[vcol], errors="coerce")[high].median()
        # Plausible DC string voltage only (rejects mis-scaled/AC-ish values).
        if np.isfinite(v) and 15 <= v <= 1500:
            v_med = float(v)

    i_med, i_from = None, "measured"
    if icol and icol in df.columns:
        i = pd.to_numeric(df[icol], errors="coerce")[high].median()
        if np.isfinite(i) and 0.3 <= i <= 2000:
            i_med = float(i)

    # Sanity-check V, I, P against each other. A measured V*I can be at most
    # the reported power; it is often much LESS, because V and I are frequently
    # per-inverter (or per-string) while DC Power is the whole-system total, so
    # V*I is only a 1/N fraction of P. That is a LEGITIMATE case -- the
    # per-device voltage/current still yield a correct modules-per-string /
    # parallel-strings for that device -- so a ratio well below 1 must be
    # allowed. We reject ONLY when V*I is materially GREATER than P, which is
    # physically impossible for a sub-measurement and signals genuinely
    # inconsistent columns (mis-scaled units, etc.). Wrong electrical side is
    # already prevented upstream by the DC/AC name filter.
    if v_med is not None and i_med is not None and p_med > 0:
        ratio = (v_med * i_med) / p_med
        if ratio > 1 + VI_P_TOL:
            return out

    # No current column: CALCULATE it from the measured power and voltage.
    if i_med is None and v_med is not None and p_med > 0:
        i_calc = p_med / v_med
        if 0.3 <= i_calc <= 2000:
            i_med, i_from = i_calc, "P/V"

    module_vmp = cells_in_series * VMP_PER_CELL
    if v_med is not None and module_vmp > 0:
        mps = max(1, int(round(v_med / module_vmp)))
        if mps <= 40:
            out["modules_per_string"] = {
                "value": mps,
                "basis": (f"measured {v_med:.0f} V / "
                          f"{module_vmp:.1f} V/module "
                          f"({cells_in_series} cells x {VMP_PER_CELL} V)")}

    if i_med is not None:
        ps = max(1, int(round(i_med / MODULE_IMP_TYP)))
        if ps <= 200:
            src = (f"measured {i_med:.1f} A" if i_from == "measured"
                   else f"{i_med:.1f} A calculated as P/V")
            out["parallel_strings"] = {
                "value": ps,
                "basis": f"{src} / {MODULE_IMP_TYP} A/string"}
    return out


# =============================================================================
# PVPRO  (single-diode-model based degradation) -- PUBLIC ENTRY POINT
# =============================================================================
def compute_pvpro(df,
                  mapped_variables_dict,
                  cells_in_series=60,
                  modules_per_string=1,
                  parallel_strings=1,
                  alpha_isc=0.0046,
                  technology="mono-c-Si",
                  days_per_run=14,
                  iterations_per_year=12,
                  resistance_shunt_ref=600.0,
                  delta_T=3.0,
                  irradiance_threshold=200.0,
                  min_points_per_window=20,
                  progress_callback=None):
    """
    Lightweight PVPRO degradation analysis.

    Walks the dataset in `days_per_run`-day windows (spaced so there are
    `iterations_per_year` windows per year), fits the five reference
    single-diode-model parameters in each window by minimising the L2
    residual between measured and predicted (V_mp, I_mp), reconstructs
    P_mp,ref, V_mp,ref, I_mp,ref, V_oc,ref, I_sc,ref at STC in each
    window, and reports the long-term degradation rate of each via a
    linear-trend fit.

    Required mapped variables:
        'DC Voltage', 'DC Current', 'Irradiance', 'Module temperature'.

    Parameters
    ----------
    ...  (see web-app help panel) ...
    progress_callback : callable or None
        If supplied, called once per window as
            progress_callback(stage, current, total, message)
        where:
            stage   in {"prepare", "p0", "fitting", "trend", "done"}
            current is the zero-based index of the window just finished
            total   is the total number of windows that will be fit
            message is a short human-readable string
        Exceptions raised by the callback are swallowed.

    Returns
    -------
    rd : float
        Annual degradation rate of P_mp,ref in %/yr (negative = power loss).
    figs : dict[str, plotly.graph_objects.Figure]
        One small Figure per quantity, keyed by column name.  The caller
        arranges them in a grid (in pvcopilot.py each goes inside its own
        rounded-corner card).  Keys: ``"p_mp_ref"``, ``"v_mp_ref"``,
        ``"i_mp_ref"``, ``"v_oc_ref"``, ``"i_sc_ref"``.
    rates : dict
        Per-quantity degradation rates in %/yr:
        ``{"p_mp_ref": ..., "v_mp_ref": ..., "i_mp_ref": ...,
           "v_oc_ref": ..., "i_sc_ref": ...}``.
    """
    def _report(stage, current=0, total=1, message=""):
        if progress_callback is None:
            return
        try:
            progress_callback(stage, current, total, message)
        except Exception:
            pass

    _report("prepare", 0, 1, "Validating inputs and pulling V/I/G/T columns")
    # -------------------------------------------------------------
    # 1. Validate inputs and pull the four required columns.
    # -------------------------------------------------------------
    required = ["DC Voltage", "DC Current", "Irradiance", "Module temperature"]
    missing = [r for r in required
               if (mapped_variables_dict.get(r) is None
                   or mapped_variables_dict[r] not in df.columns)]
    if missing:
        raise ValueError(
            "PVPRO requires the following columns to be identified in Step 1: "
            + ", ".join(missing)
            + ". They were not found in the dataset."
        )

    v_key   = mapped_variables_dict["DC Voltage"]
    i_key   = mapped_variables_dict["DC Current"]
    irr_key = mapped_variables_dict["Irradiance"]
    tm_key  = mapped_variables_dict["Module temperature"]

    df_p = df[[v_key, i_key, irr_key, tm_key]].copy()
    df_p.index = pd.to_datetime(df_p.index)
    df_p = df_p.dropna()
    df_p = df_p[df_p[irr_key] > irradiance_threshold]
    # Drop obviously non-operating points (V*I <= 0).
    df_p = df_p[(df_p[v_key] > 0) & (df_p[i_key] > 0)]

    if len(df_p) < 100:
        raise ValueError(
            f"After dropping NaNs and points with irradiance ≤ {irradiance_threshold} "
            f"W/m², only {len(df_p)} rows remain — too few for PVPRO. "
            "Loosen the Step 2 filters or supply a longer dataset."
        )

    # Per-module / per-string normalisation (same convention as full pvpro).
    V_arr = df_p[v_key].to_numpy(dtype=float) / max(modules_per_string, 1)
    I_arr = df_p[i_key].to_numpy(dtype=float) / max(parallel_strings, 1)
    G_arr = df_p[irr_key].to_numpy(dtype=float)
    Tc_arr = df_p[tm_key].to_numpy(dtype=float) + delta_T  # cell temperature, °C
    t_arr  = df_p.index.to_numpy()  # ns-precision datetime64

    # -------------------------------------------------------------
    # 2. Look up technology constants and define bounds for the fit.
    # -------------------------------------------------------------
    Eg_ref, dEgdT = _estimate_Eg_dEgdT(technology)

    lower_bounds = dict(
        photocurrent_ref       = 0.01,
        saturation_current_ref = 1e-13,
        resistance_series_ref  = 0.0,
        resistance_shunt_ref   = 10.0,
        diode_factor           = 0.5,
    )
    upper_bounds = dict(
        photocurrent_ref       = 20.0,
        saturation_current_ref = 1e-5,
        resistance_series_ref  = 1.0,
        resistance_shunt_ref   = 5000.0,
        diode_factor           = 2.0,
    )

    # -------------------------------------------------------------
    # 3. Use the whole-dataset top-decile points to get a stable p0.
    # -------------------------------------------------------------
    _report("p0", 0, 1, "Estimating starting parameters from top-decile irradiance")
    p0_global = _estimate_p0_simple(
        G_arr, Tc_arr, V_arr, I_arr,
        cells_in_series, alpha_isc, Eg_ref, dEgdT,
    )
    if p0_global is None:
        raise ValueError(
            "Could not derive a starting point for the SDM fit. "
            "Dataset may have too few high-irradiance points."
        )
    # Override Rsh seed with the user-supplied value (matches PVPRO behaviour).
    p0_global["resistance_shunt_ref"] = float(resistance_shunt_ref)

    # -------------------------------------------------------------
    # 4. Walk the dataset in windows and fit each one.
    # -------------------------------------------------------------
    t_start_all = df_p.index.min()
    t_end_all   = df_p.index.max()
    span_days   = (t_end_all - t_start_all).days + 1
    step_days   = max(int(round(365.25 / max(iterations_per_year, 1))), 1)

    # Pre-compute the list of window start-times so we know the total
    # count up front (needed by progress_callback for a tqdm-style bar).
    window_starts = []
    cur = t_start_all
    while cur + pd.Timedelta(days=days_per_run) <= t_end_all + pd.Timedelta(days=1):
        window_starts.append(cur)
        cur = cur + pd.Timedelta(days=step_days)
    n_total_windows = len(window_starts)

    rows = []
    # Warm-start state: after the first successful fit, reuse that fit's
    # parameters as the starting point for the next window's optimisation.
    # Adjacent ~14-day windows have very similar SDM parameters (modules
    # don't change much in two weeks), so a warm p0 typically cuts each
    # window's scipy.least_squares iteration count in half.  We keep the
    # original `p0_global` as a fallback in case a warm start drifts too
    # far and a later window fit fails -- on failure we fall back to the
    # global p0 for the *next* attempt rather than propagating bad state.
    p0_warm = dict(p0_global)
    for w_idx, cur in enumerate(window_starts):
        # Report progress at the START of this window so the UI advances
        # as soon as work begins, not only after it finishes.  Each report
        # call is also a momentary GIL yield (it acquires a short lock),
        # which lets the polling callback get a fresh read.
        _report(
            "fitting", w_idx, n_total_windows,
            f"Fitting window {w_idx + 1} / {n_total_windows} "
            f"({cur.strftime('%Y-%m-%d')})",
        )

        win_end = cur + pd.Timedelta(days=days_per_run)
        # Boolean window mask.
        idx_w = (df_p.index >= cur) & (df_p.index < win_end)
        n_w = int(idx_w.sum())
        if n_w >= min_points_per_window:
            G_w  = G_arr[idx_w]
            Tc_w = Tc_arr[idx_w]
            V_w  = V_arr[idx_w]
            I_w  = I_arr[idx_w]
            fit = _fit_window(
                G_w, Tc_w, V_w, I_w,
                p0=p0_warm,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                cells_in_series=cells_in_series,
                alpha_isc=alpha_isc,
                Eg_ref=Eg_ref,
                dEgdT=dEgdT,
            )
            if fit is not None:
                # Warm-start update: feed this fit's parameters to the
                # next window's optimisation.  Only the 5 SDM params --
                # don't carry over "loss" or anything else.
                p0_warm = {k: fit[k] for k in _FIT_PARAMS}
                # Reconstruct V_mp, I_mp, V_oc, I_sc at STC for this window.
                IL, I0, Rs, Rsh, nNsVth = _calcparams_desoto_lite(
                    np.array([_G_REF]), np.array([25.0]),
                    photocurrent_ref       = fit["photocurrent_ref"],
                    saturation_current_ref = fit["saturation_current_ref"],
                    resistance_series_ref  = fit["resistance_series_ref"],
                    resistance_shunt_ref   = fit["resistance_shunt_ref"],
                    diode_factor           = fit["diode_factor"],
                    cells_in_series        = cells_in_series,
                    alpha_isc              = alpha_isc,
                    Eg_ref                 = Eg_ref,
                    dEgdT                  = dEgdT,
                )
                Vmp_ref, Imp_ref = _mpp_vectorised(IL, I0, Rs, Rsh, nNsVth)
                Voc_ref = _voc_vectorised(IL, I0, Rs, Rsh, nNsVth)
                Isc_ref = _isc_vectorised(IL, I0, Rs, Rsh, nNsVth)
                p_mp_ref = float(Vmp_ref[0] * Imp_ref[0])
                rows.append({
                    "t_mid":   cur + pd.Timedelta(days=days_per_run / 2),
                    "p_mp_ref": p_mp_ref,
                    "v_mp_ref": float(Vmp_ref[0]),
                    "i_mp_ref": float(Imp_ref[0]),
                    "v_oc_ref": float(Voc_ref[0]),
                    "i_sc_ref": float(Isc_ref[0]),
                    "loss":     fit["loss"],
                    "n_points": n_w,
                    **{k: fit[k] for k in _FIT_PARAMS},
                })
            else:
                # Fit failed -- the warm-start p0 might have drifted far
                # enough from this window's data that the optimiser can't
                # find a basin.  Reset to the global p0 so the next
                # window starts from a known-stable point instead of
                # propagating the bad state forward.
                p0_warm = dict(p0_global)

        # Yield the GIL so the polling HTTP callback in the main Dash thread
        # can be served between windows. Without this, scipy's L-BFGS-B holds
        # the GIL almost continuously and the progress bar appears frozen
        # until the whole fit completes.
        time.sleep(0.01)

    if len(rows) < 4:
        raise ValueError(
            f"PVPRO produced only {len(rows)} successful window fits. "
            "Need at least 4. Try a longer dataset or fewer/looser filters."
        )

    pfit = pd.DataFrame(rows).set_index("t_mid").sort_index()

    # -------------------------------------------------------------
    # 5. Degradation rate for every reconstructed quantity.
    #
    # The per-window series are short (one point per ~14 days), have already
    # averaged out diurnal & seasonal variability, and are essentially
    # monotonic with small noise.  In that regime a linear fit on the trend
    # (after IQR outlier rejection) is a more reliable estimator than
    # rdtools-style YoY pairing, which assumes noisy daily data.  We compute
    # the linear-slope rate for each electrical quantity and report them
    # all in the result figure.
    # -------------------------------------------------------------
    _report("trend", 0, 1, "Computing degradation rates")
    from sklearn.linear_model import LinearRegression as _LR

    def _linear_rate(series):
        """Returns (rd_pct_per_yr, clean_series_kept_after_iqr).
        Linear regression slope normalised by the median, in %/yr."""
        s = series.dropna()
        if len(s) < 4:
            return np.nan, s
        q1, q3 = np.nanpercentile(s.values, [25, 75])
        iqr = q3 - q1
        keep = (s.values >= q1 - 1.5 * iqr) & (s.values <= q3 + 1.5 * iqr)
        s_clean = s.loc[keep]
        if len(s_clean) < 4:
            s_clean = s
        t_years = _time_to_years(s_clean.index).values.reshape(-1, 1)
        lr = _LR().fit(t_years, s_clean.values)
        med = float(np.nanmedian(s_clean.values))
        if med == 0 or not np.isfinite(med):
            return np.nan, s_clean
        return float(lr.coef_[0] / med * 100.0), s_clean

    quantities = [
        # (column key, HTML title label, units, plain short name for axis).
        ("p_mp_ref", "<b>Pmp</b> (ref)", "W", "Pmp"),
        ("v_mp_ref", "<b>Vmp</b> (ref)", "V", "Vmp"),
        ("i_mp_ref", "<b>Imp</b> (ref)", "A", "Imp"),
        ("v_oc_ref", "<b>Voc</b> (ref)", "V", "Voc"),
        ("i_sc_ref", "<b>Isc</b> (ref)", "A", "Isc"),
    ]

    rates = {}
    cleaned = {}
    for col, _, _, _ in quantities:
        r, c = _linear_rate(pfit[col])
        rates[col] = r
        cleaned[col] = c

    # Headline rate (the one returned to the caller) is P_mp,ref's.
    rd = rates["p_mp_ref"]
    if not np.isfinite(rd):
        rd_yoy = _yoy_degradation(pfit["p_mp_ref"].dropna())
        rd = rd_yoy if np.isfinite(rd_yoy) else np.nan

    # -------------------------------------------------------------
    # 6. Build the result figures.
    #
    # NOTE on layout: we produce one small Plotly Figure PER QUANTITY,
    # returned as a dict keyed by the quantity column name.  The Dash
    # callback in pvcopilot.py arranges them in a CSS grid where each
    # figure sits inside its own rounded-corner card div.
    #
    # Going through HTML rather than Plotly subplot shapes gives us
    # pixel-perfect rounded corners, clean axis-label margins, and no
    # coordinate-system bleed-through into adjacent subplots.
    # -------------------------------------------------------------
    qmap = {col: (label, units, short)
            for col, label, units, short in quantities}

    # Map each PVPRO quantity to a key in the shared VAR_COLORS palette
    # (defined at module scope).  Pmp -> 'power', Vmp/Voc -> 'voltage',
    # Imp/Isc -> 'current'.  This is the SAME color the user saw for the
    # same physical variable on the Step 1 raw-data preview.
    PVPRO_COLOR_KEY = {
        "p_mp_ref": "power",
        "v_mp_ref": "voltage",
        "i_mp_ref": "current",
        "v_oc_ref": "voltage",
        "i_sc_ref": "current",
    }

    def _hex_to_rgba(hex_str, alpha=0.4):
        """'#0064AB' + 0.4 -> 'rgba(0,100,171,0.4)' for translucent scatter."""
        h = hex_str.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def _one_panel(col, height):
        label, units, short = qmap[col]
        series = pfit[col].dropna()
        s_clean = cleaned[col]
        rate = rates[col]
        rate_str = f"({rate:+.2f} %/yr)" if np.isfinite(rate) else "(n/a)"

        # Per-quantity color from the shared palette.
        trend_color   = VAR_COLORS[PVPRO_COLOR_KEY[col]]
        # Scatter is a translucent shade of the trend color.  Alpha 0.3
        # keeps individual points faint so the trend line stands out.
        scatter_color = _hex_to_rgba(trend_color, alpha=0.3)

        fig = go.Figure()
        if len(series) > 0:
            # Scatter (translucent shade of the trend color).
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="markers",
                marker=dict(size=8, color=scatter_color,
                            line=dict(width=0)),
                showlegend=False,
                hovertemplate=("%{x|%Y-%m-%d}<br>"
                               "%{y:.3g} " + units + "<extra></extra>"),
            ))
            # Trend line (full saturation of the same color).
            if len(s_clean) >= 2 and np.isfinite(rate):
                t_years_arr = _time_to_years(s_clean.index).values
                med = float(np.nanmedian(s_clean.values))
                slope_abs = rate / 100.0 * med
                trend = med + slope_abs * (t_years_arr - np.nanmean(t_years_arr))
                fig.add_trace(go.Scatter(
                    x=s_clean.index, y=trend,
                    mode="lines",
                    line=dict(color=trend_color, width=2.5),
                    showlegend=False,
                ))

        fig.update_layout(
            # Panel title carries the quantity name (bold) + rate inline.
            # The label string itself already includes <b>Pmp</b> etc.,
            # so we do NOT wrap it again.
            title=dict(
                text=f"{label} &nbsp;"
                     f"<span style='color:#475569;font-weight:400'>"
                     f"{rate_str}</span>",
                x=0.5, xanchor="center", y=0.97, yanchor="top",
                font=dict(size=14, family="Arial", color="#0f172a"),
            ),
            template="plotly_white",
            height=height,
            margin=dict(l=55, r=14, t=36, b=30),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",   # transparent: card BG shows through
            plot_bgcolor="rgba(0,0,0,0)",
        )
        # Y-axis label: short quantity name + unit (e.g. "Pmp (W)").
        # Plotly axis titles render as plain text -- can't use <b> here, so
        # we use the plain short name passed in via the quantities tuple.
        fig.update_yaxes(title_text=f"{short} ({units})",
                         title_font=dict(size=11),
                         showgrid=True, gridcolor="#e2e8f0", zeroline=False)
        fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
        return fig

    # Pmp gets a taller panel (full-width row); the four secondary panels
    # are shorter and arranged in two rows of two.
    figs = {
        "p_mp_ref": _one_panel("p_mp_ref", height=210),
        "v_mp_ref": _one_panel("v_mp_ref", height=180),
        "i_mp_ref": _one_panel("i_mp_ref", height=180),
        "v_oc_ref": _one_panel("v_oc_ref", height=180),
        "i_sc_ref": _one_panel("i_sc_ref", height=180),
    }

    _report("done", n_total_windows, n_total_windows, "Done")
    return rd, figs, rates

# ===========================================================================
# PORTED FROM THE bugFixes BRANCH — energy-ratio YoY method, rate-
# reliability helpers, and low_power_filter. Required by
# pvcopilot_pipeline_report.py. Purely additive: nothing above is modified.
# ===========================================================================

# ================================
# DAILY AGGREGATION
# ================================
# When there's no irradiance, the daily value is the day's PEAK power (a high
# quantile) rather than the mean -- and low-peak days are dropped. Rationale:
# without irradiance we can't weather-normalize, so a daily MEAN collapses on
# sparsely/partially-sampled days (e.g. a dawn-only day averages near zero),
# which then blows up year-over-year ratios. A clear day's PEAK power is roughly
# comparable year-to-year (same solar geometry), so trending daily peaks of
# clear/high-output days is the standard fallback and keeps the signal stable.
PEAK_QUANTILE = 0.95          # "peak" = 95th percentile of a day's power


CLEAR_DAY_FRAC = 0.6          # keep days whose peak >= 60% of the reference peak


CLEAR_DAY_MIN_DAYS = 10       # only apply the clear-day drop when we have enough days


DENOM_FLOOR_FRAC = 0.2        # YoY: ignore pairs whose prior value < 20% of median


# Plausible annual degradation band (%/yr). Real PV degradation is a small
# NEGATIVE number (~0 to -3 %/yr typically). Anything positive beyond noise, or
# steeper than ~-3 %/yr, almost always means the inputs were un-normalizable
# (no irradiance), too sparse, or otherwise unreliable -- we flag those as
# rate_reliable = no. (Tunable: loosen MIN to e.g. -5 if you want to allow
# genuinely fast-degrading systems through unflagged.)
DEGRADATION_PLAUSIBLE_MIN = -3.0


DEGRADATION_PLAUSIBLE_MAX = 0.5


# A per-year degradation trend fit to a handful of daily points is unreliable --
# a few noisy days swing the slope to absurd values (+33%/yr from 4 points). We
# still REPORT a rate below this many daily points, but flag it as unreliable
# (with the reason) rather than presenting it as trustworthy.
MIN_TREND_POINTS = 20


MAX_CI_WIDTH = 3.0           # internal band wider than this (%/yr) -> unreliable


# ---------------------------------------------------------------------------
# ENERGY-RATIO METHOD (RdTools-style)
#
# Sum each day FIRST (measured energy vs plane-of-array-expected energy), then
# trend the daily ratio year-over-year. Summing before dividing keeps dawn/dusk
# and transient points from poisoning the ratio, so far less data is discarded
# and the trend is steadier. Validated against synthetic data with known
# injected rates (recovered within ~0.03 %/yr; truth inside the internal
# uncertainty band in 12/12 runs).
# ---------------------------------------------------------------------------
ER_MIN_IRR = 50.0            # daylight gate for the daily sums (W/m^2)


ER_MIN_DAY_POINTS = 4        # a day needs this many daylight points


ER_MIN_INSOL_FRAC = 0.2      # ...and >=20% of the median daily insolation


ER_RATIO_TRIM = (0.3, 1.7)   # sanity band around the median daily ratio


ER_YOY_TOL_DAYS = 7          # match days within +/- a week of one year apart


ER_MIN_PAIRS = 5             # need this many year-apart pairs for a rate


ER_BOOT_N = 500              # bootstrap resamples for the internal band


# ================================
# Low power filter (no-irradiance fallback)
# ================================
def low_power_filter(df, mapped_variables_dict, peak_quantile=0.99, min_frac=0.15):
    """Power-only cleanup used when there's no irradiance (so the irradiance-based
    clear-sky / low-irradiance filters can't run).

    Keeps points producing at least `min_frac` of the system's peak power and
    drops the rest -- i.e. removes night, dawn/dusk, and heavily-clouded low-output
    readings, which otherwise stay in (raw power has no irradiance to gate on) and
    contaminate the trend. `peak_quantile` (99th pct) is a spike-robust stand-in
    for the true peak.

    Returns (normal_indices, outlier_indices), matching the other filters.
    """
    power_key = mapped_variables_dict.get("DC Power")
    if not power_key or power_key not in df.columns:
        return df.index, df.index[[]]
    p = pd.to_numeric(df[power_key], errors="coerce")
    peak = p.quantile(peak_quantile)
    if not np.isfinite(peak) or peak <= 0:
        return df.index, df.index[[]]
    mask = p >= (min_frac * peak)
    # Safety: never drop every row. If the threshold would remove everything
    # (e.g. an outlier-inflated peak), keep all rather than zero out the dataset.
    if not mask.any():
        return df.index, df.index[[]]
    return df.index[mask], df.index[~mask]


def rate_is_plausible(rd):
    """True if a degradation rate (%/yr) is finite and within the plausible band
    (roughly a small negative number). Used to flag unreliable results rather
    than presenting them as trustworthy."""
    try:
        return bool(np.isfinite(rd) and
                    DEGRADATION_PLAUSIBLE_MIN <= rd <= DEGRADATION_PLAUSIBLE_MAX)
    except Exception:
        return False


def degradation_reliability(rd, n_points=None, has_irradiance=True,
                            duration_years=None, ci_width=None):
    """Decide whether a (computed) degradation rate is trustworthy, and if not,
    WHY -- so the UI can show the rate alongside 'unreliable because ...'.

    `ci_width` (energy-ratio YoY only) is the width of the internal bootstrap
    uncertainty band in %/yr. It is NEVER displayed as a range -- it only feeds
    this flag: a band wider than MAX_CI_WIDTH means the spread of year-over-year
    changes is too large for the single number to be trusted.

    Returns (is_reliable, reasons). reasons is a list of plain-English strings;
    an empty list means the rate looks reliable.
    """
    reasons = []
    if n_points is not None and n_points < MIN_TREND_POINTS:
        reasons.append(
            f"only {n_points} daily data point(s) survived filtering "
            f"(a reliable trend needs at least {MIN_TREND_POINTS})")
    if not has_irradiance:
        reasons.append(
            "there is no irradiance column, so power isn't weather-normalized "
            "(year-to-year weather then looks like degradation)")
    if duration_years is not None and duration_years < 2:
        reasons.append(
            f"the data spans only {duration_years:.1f} year(s) — under 2 years is "
            "too short to separate real aging from seasonal weather")
    if not rate_is_plausible(rd):
        reasons.append(
            f"the computed rate ({rd:+.1f}%/yr) is outside the physically "
            "plausible range (real degradation is roughly 0 to −3%/yr)")
    if ci_width is not None and np.isfinite(ci_width) and ci_width > MAX_CI_WIDTH:
        reasons.append(
            f"the year-over-year changes disagree with each other by about "
            f"±{ci_width / 2:.1f}%/yr — too scattered for this single number "
            "to be trusted")
    return (len(reasons) == 0), reasons


def daily_energy_ratio(df, mapped_variables_dict, gamma=-0.004):
    """Daily measured-vs-expected energy ratio. Returns a Series indexed by
    day, or None when it can't be built (no power/irradiance, no valid days)."""
    pcol = mapped_variables_dict.get("DC Power")
    icol = mapped_variables_dict.get("Irradiance")
    tcol = mapped_variables_dict.get("Module temperature")
    if not pcol or pcol not in df.columns or not icol or icol not in df.columns:
        return None
    p = pd.to_numeric(df[pcol], errors="coerce")
    g = pd.to_numeric(df[icol], errors="coerce")
    t = pd.to_numeric(df[tcol], errors="coerce") if tcol and tcol in df.columns else None

    expected = g * (1 + gamma * (t - 25)) if t is not None else g.copy()
    ok = (g >= ER_MIN_IRR) & (p >= 0) & expected.notna() & p.notna()
    sub = pd.DataFrame({"p": p[ok], "e": expected[ok], "g": g[ok]})
    if len(sub) == 0:
        return None
    day = sub.groupby(sub.index.date).agg(
        e_meas=("p", "sum"), e_exp=("e", "sum"),
        insol=("g", "sum"), n=("p", "size"))
    day.index = pd.to_datetime(day.index)

    med_insol = day["insol"].median()
    day = day[(day["n"] >= ER_MIN_DAY_POINTS)
              & (day["insol"] >= ER_MIN_INSOL_FRAC * med_insol)
              & (day["e_exp"] > 0)]
    if len(day) == 0:
        return None
    ratio = day["e_meas"] / day["e_exp"]
    med = ratio.median()
    ratio = ratio[(ratio >= ER_RATIO_TRIM[0] * med)
                  & (ratio <= ER_RATIO_TRIM[1] * med)]
    return ratio if len(ratio) >= 2 else None


def compute_yoy_energy(df, mapped_variables_dict, gamma=-0.004):
    """Energy-ratio YoY degradation rate.

    Returns (rd, fig, ci_width, n_days):
      rd        median year-over-year change of the daily energy ratio (%/yr)
      fig       scatter of the daily ratio with the trend annotated
      ci_width  width of the internal bootstrap band (%/yr) — used ONLY for
                the reliability flag, never displayed as a range
      n_days    number of valid daily points the estimate is built from
    (np.nan, None, np.nan, 0) when the method can't run on this data.
    """
    ratio = daily_energy_ratio(df, mapped_variables_dict, gamma=gamma)
    if ratio is None:
        return np.nan, None, np.nan, 0
    ratio = ratio.sort_index()

    idx = ratio.index
    targets = idx - pd.DateOffset(years=1)
    pos = idx.get_indexer(targets, method="nearest")
    changes = []
    for i, j in enumerate(pos):
        if j < 0 or idx[j] >= idx[i]:
            continue
        if abs((idx[i] - pd.DateOffset(years=1) - idx[j]).days) > ER_YOY_TOL_DAYS:
            continue
        prev, curr = ratio.iloc[j], ratio.iloc[i]
        if prev > 0:
            changes.append((curr / prev - 1.0) * 100.0)
    changes = np.asarray(changes)
    if len(changes) < ER_MIN_PAIRS:
        return np.nan, None, np.nan, int(len(ratio))

    rd = float(np.median(changes))
    rng = np.random.default_rng(0)   # fixed seed -> same answer every run
    boots = np.median(
        rng.choice(changes, size=(ER_BOOT_N, len(changes)), replace=True), axis=1)
    ci_width = float(np.percentile(boots, 95) - np.percentile(boots, 5))

    # Figure: daily ratio scatter + the median trend through it.
    t_yr = np.asarray((ratio.index - ratio.index[0]).days) / 365.25
    trend = ratio.median() * (1 + rd / 100.0) ** t_yr
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values, mode="markers",
        marker=dict(size=6, opacity=0.6, color="#A6CAEC"),
        name="Daily energy ratio"))
    fig.add_trace(go.Scatter(
        x=ratio.index, y=trend, mode="lines",
        line=dict(color="#0070C0", width=2),
        name=f"YoY Trend ({rd:.2f}%/yr)"))
    fig.update_layout(
        title="Daily Energy Ratio and YoY Trend",
        xaxis_title="Time", yaxis_title="Measured / expected energy",
        template="plotly_white", height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.2,
                    xanchor="center", x=0.5))
    return rd, fig, ci_width, int(len(ratio))
