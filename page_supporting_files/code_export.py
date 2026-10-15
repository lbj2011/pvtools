"""
code_export.py — a runnable, verified Python script for one PV-Copilot run.

Pipeline
--------
1. ``build_draft_script(cfg)``: assemble a script from the app's OWN functions
   (pulled with ``inspect.getsource`` together with everything they depend on,
   so the script runs the same code the app ran) plus a main section that
   replays the exact sequence of the mode that produced the result (Simple or
   Advanced) with the settings of that run.
2. ``tidy_with_llm(draft, cfg)``: an LLM rewrites the draft for a human reader
   — structure, names, comments — keeping the numerics.
3. ``run_script(code, df)``: execute a script in a separate, restricted Python
   process on the data the app analysed and read back its ``results`` dict.
   LLM-written code is statically checked first (``check_script_safety``).
4. ``generate_verified_script(...)``: draft -> verify -> LLM -> verify (one
   repair round) and return whichever version reproduces the app's rate(s).

The script contract (the LLM prompt restates it):
    * a top-level ``DATA_FILE = "..."`` line — the harness points it at the
      data under test;
    * a top-level ``results`` dict {method: rate in %/yr} when it finishes.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import pandas as pd

RATE_TOLERANCE = 0.01          # %/yr: how close a script must reproduce the app
RUN_TIMEOUT_S = 240            # per script execution (PVPRO needs the headroom)
_INTERNAL_MODULE_PREFIXES = ("page_supporting_files", "pages", "app", "utils", "index")


# =============================================================================
# 1. Extracting the app's own functions (with their dependencies)
# =============================================================================
def _defined_names(node):
    """Top-level names a module statement defines."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        out = set()
        for t in node.targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    out.add(n.id)
        return out
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    if isinstance(node, ast.Import):
        return {(a.asname or a.name.split(".")[0]) for a in node.names}
    if isinstance(node, ast.ImportFrom):
        return {(a.asname or a.name) for a in node.names}
    if isinstance(node, (ast.Try, ast.If)):
        out = set()
        blocks = [node.body, getattr(node, "orelse", []), getattr(node, "finalbody", [])]
        blocks += [h.body for h in getattr(node, "handlers", [])]
        for b in blocks:
            for sub in b:
                out |= _defined_names(sub)
        return out
    return set()


def _referenced_names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class _ModuleSource:
    _cache = {}

    def __init__(self, module):
        self.module = module
        self.src = inspect.getsource(module)
        self.tree = ast.parse(self.src)
        self.defs = {}
        for node in self.tree.body:
            for name in _defined_names(node):
                self.defs.setdefault(name, []).append(node)

    @classmethod
    def of(cls, module):
        key = module.__name__
        if key not in cls._cache:
            cls._cache[key] = cls(module)
        return cls._cache[key]

    def segment(self, node):
        """Source of a top-level statement, with the comment lines directly
        above it (they explain constants and small helpers)."""
        lines = self.src.splitlines()
        start = node.lineno - 1
        while start > 0 and lines[start - 1].startswith("#") and not lines[start - 1].startswith("# ===="):
            start -= 1
        return "\n".join(lines[start:node.end_lineno])


def _is_internal(modname, level=0):
    return level > 0 or (modname or "").split(".")[0] in _INTERNAL_MODULE_PREFIXES


def extract_functions(specs, with_lines=False):
    """specs: list of (module, name). Returns (import_lines, code_blocks)
    covering those names and everything they reference, resolved through the
    app's internal imports to where each name is really defined."""
    imports, blocks, seen, owner = [], [], set(), {}
    todo = list(specs)
    while todo:
        module, name = todo.pop()
        ms = _ModuleSource.of(module)
        for node in ms.defs.get(name, []):
            key = (module.__name__, node.lineno,
                   name if isinstance(node, (ast.Import, ast.ImportFrom)) else None)
            if key in seen:
                continue
            seen.add(key)
            if isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                if _is_internal(node.module, node.level):
                    target = importlib.import_module(node.module)
                    for a in node.names:
                        if (a.asname or a.name) == name:
                            todo.append((target, a.name))
                            if a.asname and a.asname != a.name:
                                blocks.append((module.__name__, node.lineno,
                                               f"{a.asname} = {a.name}"))
                    continue
                # only the name that is actually used, not the whole statement
                for a in node.names:
                    if (a.asname or a.name) == name:
                        imports.append(("from", node.module, a.name, a.asname))
                continue
            if isinstance(node, ast.Import):
                for a in node.names:
                    if (a.asname or a.name.split(".")[0]) == name:
                        imports.append(("import", a.name, None, a.asname))
                continue
            for defined in _defined_names(node):
                prev = owner.get(defined)
                if prev and prev != module.__name__:
                    raise RuntimeError(f"code export: '{defined}' is defined in both "
                                       f"{prev} and {module.__name__}")
                owner[defined] = module.__name__
            blocks.append((module.__name__, node.lineno, ms.segment(node)))
            for ref in _referenced_names(node):
                if ref in ms.defs:
                    todo.append((module, ref))
    # stable order: by module, then by position in that module
    order = {}
    for m, ln, _ in blocks:
        order.setdefault(m, len(order))
    blocks.sort(key=lambda b: (order[b[0]], b[1]))
    if with_lines:
        return list(dict.fromkeys(imports)), blocks
    return list(dict.fromkeys(imports)), [b[2] for b in blocks]


_STDLIB = {"warnings", "inspect", "re", "math", "json", "datetime", "collections",
           "itertools", "functools", "typing", "dataclasses", "copy", "textwrap",
           "numbers", "statistics"}


def imports_from_code(code):
    """Structured imports of a script's top-level import statements."""
    out = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.Import):
            out += [("import", a.name, None, a.asname) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module != "__future__":
            out += [("from", node.module, a.name, a.asname) for a in node.names]
    return out


def render_imports(imports):
    """One tidy import block: standard library, then third-party packages,
    `import x` before `from x import ...`, one line per module, sorted."""
    plain, froms = {}, {}
    for kind, mod, name, asname in imports:
        if kind == "import":
            plain[(mod, asname)] = True
        else:
            froms.setdefault(mod, {})[(name, asname)] = True

    def _line_from(mod, names):
        items = sorted(f"{n} as {a}" if a and a != n else n for n, a in names)
        line = f"from {mod} import {', '.join(items)}"
        if len(line) <= 88:
            return line
        return f"from {mod} import (\n" + "".join(f"    {i},\n" for i in items) + ")"

    groups = {True: [], False: []}
    for mod, asname in sorted(plain):
        groups[mod.split(".")[0] in _STDLIB].append(
            (mod, f"import {mod}" + (f" as {asname}" if asname else "")))
    for mod in sorted(froms):
        groups[mod.split(".")[0] in _STDLIB].append((mod, _line_from(mod, froms[mod])))
    blocks = []
    for std in (True, False):
        lines = [l for _m, l in sorted(groups[std], key=lambda t: (t[0].split(".")[0],
                                                                   not t[1].startswith("import"),
                                                                   t[0]))]
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def banner(title, char="="):
    """A section header: a rule, the title, a rule (79 characters wide)."""
    rule = "# " + char * 77
    return f"{rule}\n# {title}\n{rule}"


def _section_titles(module):
    """[(line number, title)] of the section headers of a module:
    a '# ====' rule, '# Title', optional '#   description' lines, a rule."""
    lines = inspect.getsource(module).splitlines()
    out, i = [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("# ====") and lines[i + 1].startswith("# ") \
                and not lines[i + 1].startswith("# ===="):
            out.append((i + 2, lines[i + 1][2:].strip()))
            i += 2
            while i < len(lines) and not lines[i].startswith("# ===="):
                i += 1
        i += 1
    return out


# =============================================================================
# 2. The draft script
# =============================================================================
_METHOD_FUNCS = {"YOY": "compute_yoy", "LR": "compute_lr", "CSD": "compute_csd",
                 "HW": "compute_hw", "ARIMA": "compute_arima", "PVPRO": "compute_pvpro"}


def _fmt(v):
    return repr(v)


HELPER_MARKER = "# >>> PV-Copilot helper functions (inserted here verbatim) <<<"

_FILTER_PARAM_NOTES = {
    "irr_thresh": "W/m², readings below this irradiance are dropped",
    "power_ratio": "DC power must exceed this × irradiance (array producing)",
    "gamma": "1/°C, power temperature coefficient for the normalisation",
    "iqr": "Tukey multiplier for the outlier filter",
    "norm_lower": "lowest normalised performance kept",
    "norm_upper_pct": "percentile of normalised performance above which readings are dropped",
}
_METHOD_DEFAULTS = {"yoy_window": 30, "hw_period": 12, "arima_p": 1, "arima_d": 1,
                    "arima_q": 0, "arima_s": 12}
_METHOD_PARAM_NOTES = {
    "yoy_window": "days, moving average drawn over the daily series (plot only)",
    "hw_period": "Holt-Winters seasonal period (samples)",
    "arima_p": "ARIMA order p", "arima_d": "ARIMA order d", "arima_q": "ARIMA order q",
    "arima_s": "ARIMA seasonal period (samples)",
}
_METHOD_PARAM_USED_BY = {"yoy_window": "YOY", "hw_period": "HW", "arima_p": "ARIMA",
                         "arima_d": "ARIMA", "arima_q": "ARIMA", "arima_s": "ARIMA"}


def _stub(block):
    """Signature + first docstring line of a helper, for the LLM to read."""
    try:
        node = ast.parse(block).body[0]
    except Exception:
        return block.splitlines()[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        lines = block.splitlines()
        head = "\n".join(lines[:node.body[0].lineno - 1]) if node.body else lines[0]
        doc = (ast.get_docstring(node) or "").strip().splitlines()
        return head + (f'\n    """{doc[0]}"""' if doc else "") + "\n    ..."
    return block if len(block.splitlines()) <= 3 else block.splitlines()[0] + "  # ..."


def _lit(v):
    """A Python literal, strings in double quotes."""
    if isinstance(v, str) and '"' not in v and "\\" not in v:
        return f'"{v}"'
    return repr(v)


def _dict_literal(name, d, notes):
    """NAME = {...} with one key per line and aligned comments."""
    if not d:
        return f"{name} = {{}}"
    items = [(f"    {_lit(k)}: {_lit(v)},", notes.get(k)) for k, v in d.items()]
    width = max(len(i) for i, _n in items) + 2
    lines = [f"{name} = {{"]
    lines += [f"{i:<{width}}# {n}" if n else i for i, n in items]
    lines.append("}")
    return "\n".join(lines)


def _helper_code(blocks_with_lines, module):
    """The helper functions, grouped under the section titles of export_lib:
    two blank lines around functions, none between consecutive constants."""
    titles = _section_titles(module)
    text, current, prev_def = "", None, None
    for _mod, lineno, code in blocks_with_lines:
        title = "Optional packages"
        for ln, t in titles:
            if ln <= lineno:
                title = t
        is_def = code.lstrip().startswith(("def ", "class ", "@", "try:"))
        if title != current:
            sep = "\n\n\n" if text else ""
            text += sep + f"# --- {title} " + "-" * max(3, 72 - len(title)) + "\n"
            current = title
        elif is_def or prev_def:
            text += "\n\n\n"
        else:
            text += "\n\n" if code.lstrip().startswith("#") else "\n"
        text += code
        prev_def = is_def
    return text


def build_draft_script(cfg, helpers="full"):
    """cfg keys:
        mode            "simple" | "advanced"
        data_file       file name shown to the user
        mapping         the column mapping the app used
        downsize_factor float | None
        methods         list of methods to compute, e.g. ["YOY"] or ["YOY","LR"]
        filter_params   dict(gamma, irr_thresh, power_ratio, iqr, norm_lower, norm_upper_pct)
        filters         (advanced) list of enabled filter keys
        clearsky        (advanced) dict(csi_threshold, day_fraction, latitude, longitude, tilt, azimuth)
        method_params   dict(yoy_window, hw_period, arima_p, arima_d, arima_q, arima_s)
        pvpro_kwargs    dict for compute_pvpro (PVPRO only)
        expected        {method: rate %/yr} the app showed (documented in the header)

    helpers="full"  -> the complete script;
    helpers="split" -> (body without the helper functions, helper_code,
                       helper imports, signatures-only listing) for the LLM.
    """
    import page_supporting_files.export_lib as EL

    methods = [m for m in cfg.get("methods") or ["YOY"]]
    advanced = cfg.get("mode") == "advanced"
    pvpro = methods == ["PVPRO"]
    single = not pvpro and (not advanced or len(methods) == 1)
    need = set(methods) | ({"LR"} if single else set())
    enabled = set(cfg.get("filters") or []) if advanced else {"timezone", "clearsky",
                                                                "low-irra-power", "outlier"}
    cs = dict(cfg.get("clearsky") or {}) if advanced else {}
    modelled_cs = cs.get("latitude") is not None and cs.get("longitude") is not None

    names = ["coerce_numeric_columns", "fix_temperature_units", "drop_duplicate_timestamps",
             "basic_value_filter", "clipping_filter", "normalize", "plot_filtering"]
    if cfg.get("downsize_factor"):
        names.append("downsize_block_mean")
    if "timezone" in enabled:
        names.append("align_to_solar_day")
    if "clearsky" in enabled:
        names += ["clear_sky_filter",
                  "clear_sky_reference" if modelled_cs else "empirical_clear_sky_reference"]
    if "low-irra-power" in enabled:
        names.append("low_irra_power_filter")
    if "outlier" in enabled:
        names.append("identify_outliers_iqr")
    if pvpro:
        names += ["compute_pvpro", "plot_pvpro"]
    else:
        names += ["aggregate_daily", "plot_daily_trend"] + [_METHOD_FUNCS[m] for m in sorted(need)]
        if "YOY" in need:
            names += ["yoy_possible", "plot_yoy_distribution"]
        if len(methods) > 1:
            names.append("plot_method_comparison")
    imports, blocks = extract_functions([(EL, n) for n in names], with_lines=True)
    imports = list(imports) + [("import", "warnings", None, None),
                               ("import", "numpy", None, "np"),
                               ("import", "pandas", None, "pd")]

    fp = dict(cfg.get("filter_params") or {})
    mp = {k: v for k, v in (cfg.get("method_params") or {}).items()
          if k in _METHOD_DEFAULTS and _METHOD_PARAM_USED_BY[k] in need}
    for k, v in _METHOD_DEFAULTS.items():
        if _METHOD_PARAM_USED_BY[k] in need:
            mp[k] = mp.get(k) if mp.get(k) not in (None, "") else v
    exp = cfg.get("expected") or {}
    pkgs = ["pandas", "numpy", "scipy", "pvlib", "rdtools", "pvanalytics", "plotly"]
    if need & {"HW", "ARIMA"}:
        pkgs.append("statsmodels")
    if pvpro or need & {"HW", "ARIMA"}:
        pkgs.append("scikit-learn")
    header = textwrap.dedent(f'''\
        """
        PV-Copilot analysis script ({cfg.get("mode", "simple")} mode)

        Reproduces the degradation analysis PV-Copilot ran on {cfg.get("data_file") or "your data"}.
        Result shown in the app: {", ".join(f"{m} {r:+.2f} %/yr" for m, r in exp.items()) or "n/a"}
        Requires: {", ".join(pkgs)}
        Run it with the data file next to it:  python this_script.py
        """''')

    settings = [banner("Settings"),
                f"DATA_FILE = {_lit(cfg.get('data_file') or 'data.csv')}",
                _dict_literal("MAPPING", cfg.get("mapping") or {},
                              {"Time": "'__index__' = the file's index holds the timestamps"}
                              if (cfg.get("mapping") or {}).get("Time") == "__index__" else {}),
                f"DOWNSIZE_FACTOR = {_lit(cfg.get('downsize_factor'))}   "
                "# average this many consecutive readings into one (None = off)",
                _dict_literal("FILTER_PARAMS", fp, _FILTER_PARAM_NOTES)]
    if advanced:
        settings.append("ENABLED_FILTERS = [" + ", ".join(_lit(x) for x in sorted(enabled)) + "]")
    if "clearsky" in enabled and advanced:
        settings.append(_dict_literal("CLEAR_SKY", {
            "csi_threshold": cs.get("csi_threshold") if cs.get("csi_threshold") is not None else 0.15,
            "day_fraction": cs.get("day_fraction") if cs.get("day_fraction") is not None else 0.5,
            **({k: cs.get(k) for k in ("latitude", "longitude", "tilt", "azimuth")}
               if modelled_cs else {})},
            {"csi_threshold": "a reading is clear when measured / clear-sky is within this of 1",
             "day_fraction": "share of a day's readings that must be clear",
             "tilt": "degrees (None = fitted from the data)",
             "azimuth": "degrees (None = fitted from the data)"}))
    settings.append("METHODS = [" + ", ".join(_lit(m) for m in methods) + "]")
    if mp:
        settings.append(_dict_literal("METHOD_PARAMS", mp, _METHOD_PARAM_NOTES))
    if pvpro:
        settings.append(_dict_literal("PVPRO_KWARGS", cfg.get("pvpro_kwargs") or {}, {
            "days_per_run": "length of each fitting window (days)",
            "iterations_per_year": "windows started per year"}))
    settings.append("SHOW_PLOTS = True   # open the figures in a browser at the end")
    settings = "\n".join(settings)

    loader = banner("Pipeline") + "\n" + textwrap.dedent('''\
        def load_data(path):
            """Read the data file and put the timestamps on the index."""
            time_col = MAPPING.get("Time")
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif path.endswith((".xls", ".xlsx")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, index_col=0 if time_col in (None, "__index__") else None)
            if time_col and time_col != "__index__" and time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.dropna(subset=[time_col]).set_index(time_col)
            df.index = pd.to_datetime(df.index)
        ''')
    mapping = cfg.get("mapping") or {}
    derived = [(r, c) for r, c in mapping.items() if str(c).startswith("computed_dc_")]
    if derived:
        loader += "    # PV-Copilot derived the missing electrical quantity from the other two.\n"
        v, i, p = (f"MAPPING[{k!r}]" for k in ("DC Voltage", "DC Current", "DC Power"))
        num = lambda x: f"pd.to_numeric(df[{x}], errors='coerce')"
        for role, _c in derived:
            if role == "DC Power":
                loader += f"    df[{p}] = {num(v)} * {num(i)}\n"
            elif role == "DC Voltage":
                loader += f"    df[{v}] = {num(p)} / {num(i)}.replace(0, np.nan)\n"
            elif role == "DC Current":
                loader += f"    df[{i}] = {num(p)} / {num(v)}.replace(0, np.nan)\n"
    loader += "    return df\n\n\n" + textwrap.dedent(f'''\
        def prepare(df):
            """Numeric columns, °F -> °C and logger error codes, duplicate
            timestamps{", then the same block averaging the app applied" if cfg.get("downsize_factor") else ""}."""
            df = coerce_numeric_columns(df, MAPPING)
            df, notes = fix_temperature_units(df, MAPPING)
            df, note = drop_duplicate_timestamps(df, MAPPING)
            for n in notes + ([note] if note else []):
                print("note:", n)
        ''')
    if cfg.get("downsize_factor"):
        loader += "    df = downsize_block_mean(df, DOWNSIZE_FACTOR)\n"
    loader += "    return df\n"

    # ---- filters: the app's exact order and combination rules -------------
    f = ['def run_filters(df):',
         '    """Step 2 in the app\'s order. Returns (every reading with its',
         '    normalised performance "norm", the readings kept)."""',
         '    irr = MAPPING["Irradiance"]',
         '    ok, _ = basic_value_filter(df, MAPPING)',
         '    df = df.loc[ok].copy()']
    if "timezone" in enabled:
        f += ['    df, shift_h, centre_h = align_to_solar_day(df, irr)',
              '    print(f"time alignment: shift {shift_h:+d} h (irradiance centre {centre_h:.1f} h)")']
    f += ['    keep, _, clip = clipping_filter(df, MAPPING.get("DC Power"))',
          '    if clip["clipped"] and len(keep):',
          '        df = df.loc[keep].copy()',
          '        print(f"clipping: readings at the ~{clip[\'ceiling\']:,.0f} W ceiling removed")']
    if "clearsky" in enabled:
        if modelled_cs:
            ref_call = ('    reference, how = clear_sky_reference(\n'
                        '        df, irr, latitude=CLEAR_SKY["latitude"], longitude=CLEAR_SKY["longitude"],\n'
                        '        tilt=CLEAR_SKY["tilt"], azimuth=CLEAR_SKY["azimuth"],\n'
                        '        power_key=MAPPING.get("DC Power"))')
        else:
            ref_call = '    reference, how = empirical_clear_sky_reference(df, irr)'
        cs_args = ('csi_threshold=CLEAR_SKY["csi_threshold"], day_fraction=CLEAR_SKY["day_fraction"], '
                   if advanced else '')
        if advanced:
            f += [ref_call,
                  f'    clear_kept, _, cs_info = clear_sky_filter(df, irr, reference, how, {cs_args}return_info=True)',
                  '    clear_ok = df.index.isin(clear_kept)',
                  '    print(f"clear sky: {cs_info[\'n_clear_days\']} of {cs_info[\'n_days\']} days clear ({how})")']
        else:
            f += ['    # Simple mode skips a filter that would remove every remaining reading.',
                  '    keep_mask = np.ones(len(df), dtype=bool)',
                  '    try:',
                  '        ' + ref_call.strip().replace("\n", "\n    "),
                  '        clear_kept, _, cs_info = clear_sky_filter(df, irr, reference, how, return_info=True)',
                  '        clear_ok = np.asarray(df.index.isin(clear_kept), dtype=bool)',
                  '        if clear_ok.sum() > 0:',
                  '            keep_mask = clear_ok',
                  '        print(f"clear sky: {cs_info[\'n_clear_days\']} of {cs_info[\'n_days\']} days clear")',
                  '    except Exception as e:',
                  '        print(f"clear-sky filter skipped ({type(e).__name__})")']
    f += ['    df = normalize(df, MAPPING, gamma=FILTER_PARAMS["gamma"])']
    if advanced:
        f += ['    # Each filter is evaluated on the whole series and removes readings still kept.',
              '    keep_mask = np.ones(len(df), dtype=bool)']
        if "clearsky" in enabled:
            f += ['    keep_mask &= clear_ok']
        if "low-irra-power" in enabled:
            f += ['    ok, _ = low_irra_power_filter(',
                  '        df, MAPPING, irr_thresh=FILTER_PARAMS["irr_thresh"],',
                  '        power_ratio=FILTER_PARAMS["power_ratio"], norm_lower=FILTER_PARAMS["norm_lower"],',
                  '        norm_upper_pct=FILTER_PARAMS["norm_upper_pct"])',
                  '    keep_mask &= df.index.isin(ok)']
        if "outlier" in enabled:
            f += ['    ok, _ = identify_outliers_iqr(df, "norm", iqr_multiplier=FILTER_PARAMS["iqr"])',
                  '    keep_mask &= df.index.isin(ok)']
    else:
        f += ['    ok, _ = low_irra_power_filter(',
              '        df, MAPPING, irr_thresh=FILTER_PARAMS["irr_thresh"],',
              '        power_ratio=FILTER_PARAMS["power_ratio"], norm_lower=0.01, norm_upper_pct=99)',
              '    candidate = keep_mask & np.asarray(df.index.isin(ok), dtype=bool)',
              '    if candidate.sum() > 0 or keep_mask.sum() == 0:',
              '        keep_mask = candidate',
              '    ok, _ = identify_outliers_iqr(df, "norm", iqr_multiplier=FILTER_PARAMS["iqr"])',
              '    candidate = keep_mask & np.asarray(df.index.isin(ok), dtype=bool)',
              '    if candidate.sum() > 0 or keep_mask.sum() == 0:',
              '        keep_mask = candidate']
    f += ['    return df, df.loc[df.index[keep_mask]]']
    filters = "\n".join(f) + "\n"

    call = {
        "YOY": "compute_yoy(daily, rolling_window=METHOD_PARAMS['yoy_window'])",
        "LR": "compute_lr(daily)",
        "CSD": "compute_csd(daily)",
        "HW": "compute_hw(daily, period=METHOD_PARAMS['hw_period'])",
        "ARIMA": ("compute_arima(daily, p=METHOD_PARAMS['arima_p'], d=METHOD_PARAMS['arima_d'],\n"
                  "                                       q=METHOD_PARAMS['arima_q'], "
                  "seasonal_period=METHOD_PARAMS['arima_s'])"),
    }
    c = ['def run_degradation(df_good):',
         '    """Returns ({method: rate in %/yr}, {method: details for the plots}, daily series)."""']
    if pvpro:
        c += ['    rate, info = compute_pvpro(df_good, MAPPING, **PVPRO_KWARGS)',
              "    return {'PVPRO': float(rate)}, {'PVPRO': info}, None"]
    else:
        c += ["    daily = aggregate_daily(df_good, MAPPING['Irradiance'])   # daily performance index"]
        if single:
            m = methods[0]
            cond = "yoy_possible(daily)" if m == "YOY" else None
            c += [f"    method, rate, info = {m!r}, np.nan, {{}}"]
            body = [f"        rate, info = {call[m]}"] if m != "LR" else []
            if m == "LR":
                c += ["    rate, info = compute_lr(daily)"]
            else:
                if cond:
                    c += [f"    if {cond}:                 # YoY needs at least two years of data",
                          "        try:", "    " + body[0],
                          "        except Exception as e:",
                          "            print(f'YOY failed: {e}')"]
                else:
                    c += ["    try:", body[0],
                          "    except Exception as e:",
                          f"        print(f'{m} failed: {{e}}')"]
                c += ["    if not np.isfinite(rate):   # no rate -> linear regression, as the app does",
                      f"        print('{m} gave no rate; using linear regression instead')",
                      "        method = 'LR'",
                      "        rate, info = compute_lr(daily)"]
            c += ["    return {method: float(rate)}, {method: info}, daily"]
        else:
            c += ["    results, details = {}, {}",
                  "    for method in METHODS:",
                  "        if method == 'YOY' and not yoy_possible(daily):",
                  "            print('YOY skipped: it needs at least two years of data')",
                  "            results[method] = None",
                  "            continue",
                  "        try:"]
            for k, m in enumerate(methods):
                c += [f"            {'if' if k == 0 else 'elif'} method == {m!r}:",
                      f"                rate, info = {call[m]}"]
            c += ["            results[method] = float(rate) if np.isfinite(rate) else None",
                  "            details[method] = info",
                  "        except Exception as e:",
                  "            print(f'{method} failed: {e}')",
                  "            results[method] = None",
                  "    return results, details, daily"]
    calc = "\n".join(c) + "\n"

    if pvpro:
        plots = ["        figures.append(plot_pvpro(details['PVPRO']))"]
    else:
        plots = ["        for method, rate in results.items():",
                 "            if rate is not None:",
                 "                figures.append(plot_daily_trend(daily, rate, method, details.get(method)))"]
        if "YOY" in need:
            plots += ["        if results.get('YOY') is not None:",
                      "            figures.append(plot_yoy_distribution(details['YOY'], results['YOY']))"]
        if len(methods) > 1:
            plots += ["        figures.append(plot_method_comparison(results))"]
    main = banner("Run") + "\n" + "\n".join([
        "def main():",
        '    """Load, filter, compute the degradation rate(s), report and plot."""',
        "    df = prepare(load_data(DATA_FILE))",
        "    df_all, df_good = run_filters(df)",
        '    print(f"{len(df_good):,} of {len(df):,} readings kept after filtering")',
        "    results, details, daily = run_degradation(df_good)",
        "    for method, rate in results.items():",
        '        print(f"{method}: {rate:+.3f} %/yr" if rate is not None else f"{method}: n/a")',
        "    if SHOW_PLOTS:",
        "        figures = [plot_filtering(df_all, df_good)]",
        *plots,
        "        for fig in figures:",
        "            fig.show()",
        "    return results",
        "", "",
        'if __name__ == "__main__":',
        "    results = main()",
    ]) + "\n"

    helper_code = (banner("PV-Copilot helper functions") + "\n"
                   "# PV-Copilot's own analysis code, called by the pipeline above.\n\n"
                   + _helper_code(blocks, EL))
    body = header + "\n\n" + "\n\n\n".join([settings, loader, filters, calc, main])
    if helpers == "split":
        stubs = "\n\n".join(_stub(b[2]) for b in blocks)
        return body, helper_code, imports, stubs
    return splice_helpers(body, helper_code, imports)


def splice_helpers(code, helper_code, helper_imports=()):
    """Assemble the final script from `code` (the pipeline — written by the
    LLM or by build_draft_script) and PV-Copilot's helper functions:

        docstring
        one merged, sorted import block (+ warnings filter)
        the pipeline code (settings, pipeline functions)
        the helper functions
        the `if __name__ == "__main__":` block / other top-level statements

    Import statements and a top-level warnings.filterwarnings() in `code` are
    moved into the merged block, so nothing is imported twice or mid-file."""
    tree = ast.parse(code)
    lines = code.splitlines()
    drop, docstring_end = set(), 0
    for i, node in enumerate(tree.body):
        is_doc = (i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None),
                                                                     ast.Constant)
                  and isinstance(node.value.value, str))
        if is_doc:
            docstring_end = node.end_lineno
        is_filter = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and ast.unparse(node.value.func) == "warnings.filterwarnings")
        if isinstance(node, (ast.Import, ast.ImportFrom)) or is_filter:
            drop.update(range(node.lineno - 1, node.end_lineno))
    imports = list(helper_imports) + imports_from_code(code) + [("import", "warnings", None, None)]
    import_block = render_imports(imports) + '\n\nwarnings.filterwarnings("ignore")'

    # First statement that runs something (not a def, class, assignment or the
    # docstring): the helpers go right before it so everything is defined.
    insert_at = len(lines)
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                             ast.AnnAssign, ast.Import, ast.ImportFrom)):
            continue
        if node.end_lineno <= docstring_end:
            continue
        if isinstance(node, ast.Expr) and ast.unparse(node.value.func if isinstance(
                node.value, ast.Call) else node.value) == "warnings.filterwarnings":
            continue
        start = node.lineno - 1
        # keep a comment/banner that sits directly above the statement with it
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        insert_at = start
        break

    head = [l for j, l in enumerate(lines[:docstring_end]) if j not in drop]
    mid = [l for j, l in enumerate(lines[docstring_end:insert_at], docstring_end)
           if j not in drop and l.strip() != HELPER_MARKER]
    tail = [l for j, l in enumerate(lines[insert_at:], insert_at)
            if j not in drop and l.strip() != HELPER_MARKER]

    def _clean(block):
        text = "\n".join(block).strip("\n")
        return re.sub(r"\n{4,}", "\n\n\n", text)

    parts = [import_block, _clean(mid), helper_code.strip("\n"), _clean(tail)]
    body = "\n\n\n".join(p for p in parts if p.strip()) + "\n"
    return (_clean(head) + "\n\n" + body) if _clean(head) else body


# =============================================================================
# 3. Running a script safely and reading back its results
# =============================================================================
_ALLOWED_IMPORT_ROOTS = {
    "pandas", "numpy", "scipy", "rdtools", "pvlib", "pvanalytics", "statsmodels",
    "sklearn", "plotly", "warnings", "math", "inspect", "re", "datetime", "json",
    "collections", "itertools", "functools", "typing", "dataclasses", "__future__",
    "copy", "textwrap", "numbers", "statistics",
}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
                    "globals", "locals", "vars", "setattr", "delattr", "exit", "quit"}
_FORBIDDEN_ATTRS = {"system", "popen", "spawnl", "spawnv", "fork", "remove", "unlink",
                    "rmdir", "rmtree", "kill", "environ", "getenv", "putenv", "chmod",
                    "chown", "_exit", "execv", "execve", "Popen", "check_output",
                    "check_call", "urlopen", "socket", "connect", "to_pickle", "read_pickle"}
_ALLOWED_DUNDERS = {"__name__", "__doc__", "__init__", "__future__"}


def check_script_safety(code):
    """Static screen for code an LLM wrote before it is executed. Returns a
    list of problems (empty when the script is acceptable)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
                    problems.append(f"import of '{a.name}' is not allowed")
        elif isinstance(n, ast.ImportFrom):
            if n.level or (n.module or "").split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
                problems.append(f"import from '{n.module}' is not allowed")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in _FORBIDDEN_CALLS:
                problems.append(f"call to {n.func.id}() is not allowed")
            if n.func.id == "getattr" and len(n.args) >= 2:
                a = n.args[1]
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str)
                        and not a.value.startswith("__")):
                    problems.append("getattr() with a non-literal or dunder name is not allowed")
        elif isinstance(n, ast.Attribute):
            if n.attr in _FORBIDDEN_ATTRS:
                problems.append(f"attribute '.{n.attr}' is not allowed")
            if n.attr.startswith("__") and n.attr not in _ALLOWED_DUNDERS:
                problems.append(f"dunder attribute '.{n.attr}' is not allowed")
        elif isinstance(n, ast.Name) and n.id.startswith("__") and n.id not in _ALLOWED_DUNDERS:
            problems.append(f"name '{n.id}' is not allowed")
    if not re.search(r"^DATA_FILE\s*=", code, re.M):
        problems.append("the top-level `DATA_FILE = ...` line is missing")
    return sorted(set(problems))


_HARNESS = r'''
import json, runpy, sys, warnings
warnings.filterwarnings("ignore")
# Capture the figures the script shows instead of opening a browser.
_FIGS = []
try:
    import plotly.basedatatypes as _bdt
    def _capture(self, *a, **k):
        if len(_FIGS) < 8:
            _FIGS.append(self.to_json())
    _bdt.BaseFigure.show = _capture
except Exception:
    pass
g = runpy.run_path(sys.argv[1], run_name="__main__")
res = g.get("results")
if not isinstance(res, dict):
    raise SystemExit("script did not define a `results` dict")
def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None
with open(sys.argv[2], "w") as fh:
    json.dump(_FIGS, fh)
sys.stdout.flush()
print("__PVC_RESULTS__" + json.dumps({str(k): _f(v) for k, v in res.items()}))
'''


def _limit_resources():                      # pragma: no cover (child process)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
        resource.setrlimit(resource.RLIMIT_CPU, (RUN_TIMEOUT_S + 20, RUN_TIMEOUT_S + 20))
    except Exception:
        pass


def run_script(code, df_input, timeout=RUN_TIMEOUT_S, full=False):
    """Execute `code` on `df_input` in a separate Python process.

    Returns (results_dict | None, error_text | None, seconds); with full=True
    a 4th element: {"stdout": what the script printed, "figures": [plotly
    JSON of each figure it showed]} — i.e. the output of THIS code."""
    t0 = time.perf_counter()
    extra = {"stdout": "", "figures": []}

    def _ret(res, err):
        secs = time.perf_counter() - t0
        return (res, err, secs, extra) if full else (res, err, secs)

    with tempfile.TemporaryDirectory(prefix="pvc_export_") as tmp:
        data_path = os.path.join(tmp, "input.parquet")
        df_input.to_parquet(data_path)
        runnable, n = re.subn(r"^DATA_FILE\s*=.*$", f"DATA_FILE = {data_path!r}", code,
                              count=1, flags=re.M)
        if not n:
            return _ret(None, "no top-level DATA_FILE line")
        script_path = os.path.join(tmp, "analysis.py")
        harness_path = os.path.join(tmp, "harness.py")
        figs_path = os.path.join(tmp, "figures.json")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(runnable)
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(_HARNESS)
        # A bare environment: no API keys or credentials reach the child.
        env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")
               if k in os.environ}
        env.update(HOME=tmp, PYTHONHASHSEED="0", MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        try:
            proc = subprocess.run([sys.executable, "-E", harness_path, script_path, figs_path],
                                  cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=timeout,
                                  preexec_fn=_limit_resources if os.name == "posix" else None)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
            extra["stdout"] = out.replace(data_path, "DATA_FILE")[-6000:]
            return _ret(None, f"timed out after {timeout} s")
        lines = proc.stdout.splitlines()
        shown = [l for l in lines if not l.startswith("__PVC_RESULTS__")]
        extra["stdout"] = "\n".join(shown).replace(data_path, "DATA_FILE")[-6000:]
        try:
            with open(figs_path, encoding="utf-8") as f:
                extra["figures"] = json.load(f)
        except Exception:
            pass
        for line in reversed(lines):
            if line.startswith("__PVC_RESULTS__"):
                return _ret(json.loads(line[len("__PVC_RESULTS__"):]), None)
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return _ret(None, "\n".join(err[-12:]).replace(data_path, "DATA_FILE")
                    or f"exit code {proc.returncode}")


def compare_results(got, expected, tol=RATE_TOLERANCE):
    """(ok, human-readable differences)."""
    if not got:
        return False, "no results"
    diffs = []
    for m, want in (expected or {}).items():
        have = got.get(m)
        if want is None:
            continue
        if have is None:
            diffs.append(f"{m}: script gave no rate (app {want:+.3f})")
        elif abs(have - want) > tol:
            diffs.append(f"{m}: script {have:+.3f} vs app {want:+.3f} %/yr")
    return (not diffs), "; ".join(diffs)


# =============================================================================
# 4. LLM rewrite
# =============================================================================
_SYSTEM_PROMPT = """You are a senior photovoltaic data scientist and Python engineer.
You rewrite auto-generated analysis scripts into clean, well-documented code a PV
engineer can read, run and adapt. Numerical behaviour must not change."""

_USER_PROMPT = """Below is the pipeline part of a script PV-Copilot assembled to reproduce
one degradation analysis. Polish it for a human reader.

Goals
- A module docstring that explains, in a few sentences, what the analysis does
  and lists the key settings of this run (data file, filters, method, and the
  rate the app reported). Keep it short; the settings are in the code below it.
- Keep the layout: settings, then the pipeline functions (load_data, prepare,
  run_filters, run_degradation), then main() and the
  `if __name__ == "__main__": results = main()` block at the end.
- Section headers exactly in this form (79 characters wide):
  # =============================================================================
  # Title
  # =============================================================================
- Short comments that explain the PV meaning of each step. Base them on the
  helper docstrings listed below; name a package (rdtools, pvlib, pvanalytics,
  statsmodels) only where that docstring does. Never guess.
- Keep the comments on the settings (units, meaning) and SHOW_PLOTS with the
  figures at the end of main().
- Simplify where it helps, but keep every step.

Hard rules (the result is executed automatically and checked against the app):
1. EXACTLY the same numbers: same filters, same order, same thresholds, same
   helper calls with the same arguments. Do not "improve" the method.
2. Keep the top-level line `DATA_FILE = "..."` (same value); it is swapped for
   the test data before running.
3. Keep `results = main()` under `if __name__ == "__main__":`; main() returns
   the dict mapping each method to its rate in %/yr (same keys as now).
4. The helper functions listed below are added to the script automatically,
   together with their imports. Call them as they are; do NOT write, copy or
   re-implement them, and do NOT import anything for them. Import only what
   YOUR code uses, with plain import statements (never __import__, importlib or
   dynamic imports). Allowed: pandas, numpy, scipy, rdtools, pvlib, pvanalytics,
   statsmodels, sklearn, plotly, warnings, math, re, datetime, json, collections,
   functools. No file writing, no network, no subprocess/os/sys, no eval/exec/open().
5. Return ONLY the complete pipeline script inside one ```python code block.

{repair}Script:
```python
{code}
```

Helper functions (signatures and first docstring line only; do not output them):
```python
{stubs}
```"""


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip() + "\n"


def tidy_with_llm(code, repair_note=None, timeout=150, stubs=""):
    """Return the LLM's rewrite of `code` (raises on API errors)."""
    from page_supporting_files.analysis_utils import client, LLM_MODEL
    repair = ""
    if repair_note:
        repair = ("A previous rewrite of this script FAILED verification:\n"
                  f"{repair_note}\nStart again from the script below and keep the "
                  "numerical behaviour identical.\n\n")
    msgs = [{"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT.format(
                code=code, repair=repair, stubs=stubs or "(none)")}]
    api = client.with_options(timeout=timeout).chat.completions
    try:
        # A rewrite needs little deliberation; low effort keeps it fast.
        resp = api.create(model=LLM_MODEL, messages=msgs, reasoning_effort="low")
    except Exception as e:
        if "reasoning" not in str(e).lower():
            raise
        resp = api.create(model=LLM_MODEL, messages=msgs)
    return _extract_code(resp.choices[0].message.content)


# =============================================================================
# 5. The whole thing
# =============================================================================
_METHOD_LABELS = {"YOY": "YoY", "LR": "LR", "CSD": "CSD", "HW": "Holt-Winters",
                  "ARIMA": "ARIMA", "PVPRO": "PVPRO"}


def _rate_text(expected):
    return ", ".join(f"{_METHOD_LABELS.get(m, m)} {r:+.2f} %/yr"
                     for m, r in (expected or {}).items() if r is not None)


def generate_verified_script(cfg, df_input, progress=None, use_llm=True):
    """One-shot generate + verify (kept for scripts/tests; the UI calls the two
    phases separately)."""
    g = generate_script(cfg, progress=progress, use_llm=use_llm)
    v = verify_script(g["code"], df_input, g["expected"], progress=progress)
    notes = list(g["notes"])
    if not v["verified"]:
        notes.append(f"Verification failed: {v.get('diff') or v.get('error')}")
    return dict(code=g["code"], source=g["source"], verified=v["verified"], got=v["got"],
                expected=g["expected"], notes=notes, secs=g["secs"] + v["secs"])


# =============================================================================
# 6. Two-phase flow used by the UI: generate (fast) -> the user clicks Verify
# =============================================================================
def generate_script(cfg, progress=None, use_llm=True):
    """Build the script and let the LLM rewrite it, WITHOUT running anything.
    The rewrite only has to pass the static safety check here (with one AI
    retry); whether it reproduces the app is checked later by verify_script.

    Returns dict(code, source, expected, notes, secs)."""
    def _p(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    t0 = time.perf_counter()
    expected = {k: v for k, v in (cfg.get("expected") or {}).items() if v is not None}
    notes = []
    _p("Collecting the settings of your run")
    draft = build_draft_script(cfg)
    body, helper_code, helper_imports, stubs = build_draft_script(cfg, helpers="split")
    code, source = draft, "draft"
    if use_llm:
        repair = None
        for attempt in (1, 2):
            _p("AI is writing the script" if attempt == 1
               else "AI is fixing a problem in its script")
            try:
                ai_body = tidy_with_llm(body, repair_note=repair, stubs=stubs)
            except Exception as e:
                notes.append(f"AI rewrite unavailable ({type(e).__name__}); showing the "
                             "script assembled from PV-Copilot's functions.")
                break
            _p("Checking the AI script is safe to run")
            try:
                ai_code = splice_helpers(ai_body, helper_code, helper_imports)
            except SyntaxError as e:
                problems = [f"it is not valid Python (line {e.lineno}: {e.msg})"]
                repair = "It was rejected before running: " + problems[0]
                continue
            problems = check_script_safety(ai_code)
            if not problems:
                code, source = ai_code, "ai"
                break
            repair = "It was rejected before running: " + "; ".join(problems[:6])
        else:
            notes.append("The AI script didn't pass the safety check after a retry "
                         "(" + "; ".join(problems[:3]) + "); showing the assembled script.")
    return dict(code=code, source=source, draft=draft, expected=expected,
                notes=notes, secs=time.perf_counter() - t0)


def verify_script(code, df_input, expected, progress=None):
    """Run `code` on the app's input and compare with the app's rates.
    Returns dict(verified, got, diff, error, secs)."""
    if progress:
        try:
            progress(f"Running the script on your data to check it reproduces "
                     f"{_rate_text(expected) or 'the app result'}")
        except Exception:
            pass
    problems = check_script_safety(code)
    if problems:
        return dict(verified=False, got=None, diff=None,
                    error="Rejected by the safety check: " + "; ".join(problems[:4]), secs=0.0)
    got, err, secs, extra = run_script(code, df_input, full=True)
    if got is None:
        return dict(verified=False, got=None, diff=None, error=err, secs=secs, **extra)
    ok, diff = compare_results(got, expected)
    return dict(verified=ok, got=got, diff=diff or None, error=None, secs=secs, **extra)
