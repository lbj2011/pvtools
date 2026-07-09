import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc
from dash import html, dcc
import base64, os, json
import hashlib
import threading
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
LLM_MODEL = "gpt-5.4-nano"

# ---------------------------------------------------------------------------
# Deterministic LLM column-identification cache.
#
# The LLM (even at temperature=0 / seed=0) does NOT return the same candidate
# lists every call, so the SAME file could map to different columns run-to-run.
# The prompt depends only on the column NAMES (not the data), so we hash it and
# cache the result on disk: the first run for a given column-signature calls the
# LLM; every run after that reuses it. Delete the cache file to force a refresh.
# (The mapping is only committed once the caller confirms a good run — see
# commit_llm_cache — so a wrong mapping from a bad dataset is never locked in.)
# ---------------------------------------------------------------------------
_LLM_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".llm_id_cache.json")
_LLM_CACHE_LOCK = threading.Lock()
_LLM_CACHE = None  # loaded lazily


def _llm_cache_load():
    global _LLM_CACHE
    if _LLM_CACHE is None:
        try:
            with open(_LLM_CACHE_PATH, "r") as f:
                _LLM_CACHE = json.load(f)
        except Exception:
            _LLM_CACHE = {}
    return _LLM_CACHE


def _llm_cache_get(key):
    with _LLM_CACHE_LOCK:
        return _llm_cache_load().get(key)


def _llm_cache_put(key, value):
    # Load-modify-write under a lock, with an atomic replace, so parallel
    # workers can't corrupt the file.
    with _LLM_CACHE_LOCK:
        cache = _llm_cache_load()
        cache[key] = value
        try:
            tmp = _LLM_CACHE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cache, f)
            os.replace(tmp, _LLM_CACHE_PATH)
        except Exception:
            pass


def commit_llm_cache(df):
    """Save the column-mapping the LLM produced for this dataframe to the cache,
    but ONLY when the caller has decided the run was GOOD. parse_contents stashes
    the freshly-identified mapping as 'pending' on df.attrs; nothing is written
    until this is called. A no-op on a cache hit or when nothing is pending."""
    pending = getattr(df, "attrs", {}).get("_llm_cache_pending") if df is not None else None
    if pending:
        key, value = pending
        _llm_cache_put(key, value)


def _time_to_years(index):
    return (index - index[0]).days / 365.25


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
        # Deterministic identification: reuse the cached candidate lists for this
        # exact column-signature if we've seen it. temperature=0/seed=0 alone is
        # NOT enough (the gateway doesn't honor the seed), so the cache is what
        # guarantees the same file maps the same way every run.
        _p("Identifying data columns…")
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        raw_candidates = _llm_cache_get(cache_key)
        # Only SAVE the mapping once the dataset is confirmed GOOD downstream.
        # On a miss we hold the freshly-identified result "pending" and let the
        # caller commit it (commit_llm_cache) iff the run was clean -- so a wrong
        # mapping from a failed/iffy dataset never gets locked into the cache.
        _cache_miss = raw_candidates is None

        if _cache_miss:
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
            # NOTE: deliberately NOT saved here -- see commit_llm_cache().

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
            """Return (chosen_col_or_None, 'dc'|'ac'|None) preferring DC."""
            best, viable = _best_by_data(df, _cands(dc_role))
            if best:
                _ambiguity_note(dc_role, best, viable)
                return best, "dc"
            best, viable = _best_by_data(df, _cands(ac_role))
            if best:
                mapping_notes.append(
                    f"No usable DC {label} column — using AC {label} '{best}' "
                    f"(includes inverter effects, so it's an approximation).")
                _ambiguity_note(ac_role, best, viable, ac=True)
                return best, "ac"
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

        # Power: DC direct -> DC V*I -> AC direct -> AC V*I.
        best_dc_p, viable_dc_p = _best_by_data(df, _cands("DC Power"))
        p_col, p_side = None, None
        if best_dc_p:
            p_col, p_side = best_dc_p, "dc"
            _ambiguity_note("DC Power", best_dc_p, viable_dc_p)
        elif v_col and i_col and v_side == "dc" and i_side == "dc":
            df["computed_dc_power"] = (pd.to_numeric(df[v_col], errors="coerce")
                                       * pd.to_numeric(df[i_col], errors="coerce"))
            p_col, p_side = "computed_dc_power", "dc"
            mapping_notes.append("DC Power computed as Voltage × Current (no direct power column).")
        else:
            best_ac_p, viable_ac_p = _best_by_data(df, _cands("AC Power"))
            if best_ac_p:
                p_col, p_side = best_ac_p, "ac"
                mapping_notes.append(
                    f"No usable DC power column — using AC power '{best_ac_p}' "
                    f"(includes inverter effects, so it's an approximation).")
                _ambiguity_note("AC Power", best_ac_p, viable_ac_p, ac=True)
            elif v_col and i_col:
                df["computed_dc_power"] = (pd.to_numeric(df[v_col], errors="coerce")
                                           * pd.to_numeric(df[i_col], errors="coerce"))
                p_col, p_side = "computed_dc_power", "ac"
                mapping_notes.append("Power computed as Voltage × Current on the AC side "
                                     "(no DC power) — includes inverter effects, approximate.")

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

        # Stash the per-role alternatives on the frame so the Advanced UI can
        # render them inline under each dropdown (set AFTER any re-index so it
        # lands on the final df object the caller receives).
        df.attrs["mapping_alternatives"] = {
            r: cols for r, cols in alternatives.items() if cols}

        # Hold a freshly-identified (not-yet-cached) mapping as "pending" so the
        # caller can commit it to the cache only if the run is GOOD.
        df.attrs["_llm_cache_pending"] = (
            (cache_key, raw_candidates) if _cache_miss else None)

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

    irr_key = mapped_variables_dict["Irradiance"]
    power_key = mapped_variables_dict["DC Power"]
    temp_C_key = mapped_variables_dict["Module temperature"]

    df['norm'] = df[power_key] / (
        df[irr_key] * (1 + gamma * (df[temp_C_key] - 25)))*1000

    df.loc[df[irr_key] < 50, 'norm'] = np.nan

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

    # power filter
    mask &= df[power_key] > power_ratio * df[irr_key]

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
def aggregate_daily(df_f, irradiance_col):
    daily = (
        df_f[['norm', irradiance_col]]
        .dropna()
        .groupby(df_f.index.date)
        .apply(lambda x: np.sum(x['norm'] * x[irradiance_col]) / np.sum(x[irradiance_col]))
    )

    daily.index = pd.to_datetime(daily.index)

    return daily

# ================================
# YoY
# ================================
def compute_yoy(series, eps=1e-6, rolling_window=30, iqr_multiplier=1.5):
    series = series.dropna()
    yoy = []

    for t in series.index:
        t_prev = t - pd.DateOffset(years=1)

        if t_prev in series.index:
            prev = series.loc[t_prev]
            curr = series.loc[t]

            if prev < eps:
                continue

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

    if len(series) < 2:
        return np.nan, None

    t = _time_to_years(series.index).values.reshape(-1, 1)
    y = series.values

    model = LinearRegression().fit(t, y)
    trend = model.predict(t)

    slope = model.coef_[0]
    rd = slope / np.mean(y)*100

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