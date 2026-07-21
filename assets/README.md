# PV Copilot — new liquid-glass UI, wired to the real pipeline

The Apple "liquid-glass" mockup is now the entire UI (design, structure, colors),
with the **real** analysis functions from `page_supporting_files` wired in.

## Integration (pvtools page)

`pvcopilot.py` is a **page** inside the pvtools multi-page site. `index.py` loads it via
`module.get_layout()`, so this file:

- exposes `get_layout()` (returns the page body),
- does **not** set `app.layout` and does **not** use `dcc.Location` — the site owns the URL,
- switches the top-nav (Methods / Cite / Team) through an internal `dcc.Store("pvc-view")`
  instead of the browser URL, so it never collides with the site router,
- attaches onto the existing instance via `from app import app`.

Drop it in as `pages/pvcopilot.py` (or wherever your pages live) and put
`pvcopilot_styles.css` + `pv-array.png` in the site's `assets/`. No `app.py` ships here —
the site already has one.

## File structure

- `pvcopilot.py` — the whole page: layouts + callbacks, wired to the real pipeline.
- `page_supporting_files/` — `analysis_utils.py`, `pvcopilot_filter_functions.py` (unchanged).
- `assets/` — `pvcopilot_styles.css`, `pv-array.png`.

## What's wired to real code

| UI surface | Real function(s) |
|---|---|
| Upload / example load | `parse_contents` (LLM column identification) |
| Step 1 · Prescreening | `make_overview_figures` + real completeness / mapping / quality tags (`_quality_tag`) |
| Step 2 · Filtering | `basic_value_filter` → `clear_sky_filter` → `normalize` → `low_irra_power_filter` → `identify_outliers_iqr` (each UI toggle maps to a real op) |
| Step 3 · Model | `aggregate_daily` → `compute_yoy` / `compute_lr` / `compute_hw` / `compute_arima` / `compute_csd` / `compute_pvpro` |
| Step 4 · Code | `get_full_code` (with a local reproducible-script fallback) |
| Simple mode | full default pipeline, YoY |
| Ask Copilot chat | the same OpenAI client `analysis_utils` configures |

`analysis_utils.py` and `pvcopilot_filter_functions.py` are unchanged.

## Environment this expects

- **`.env`** with `OPENAI_API_KEY=...` — needed for column identification (upload/example
  parsing), the chat, and Step-4 code generation. The rest of the pipeline runs without it.
- **`data/`** folder next to `app.py` containing the example parquet files
  (`sys_1278_downsampled_with_VI.parquet`, `sys_1403_part1_downsampled_with_VI.parquet`,
  `sys_1422_downsampled.parquet`).
- **`page_supporting_files/diagnostic_prompts.py`** — optional; if absent, the chat uses a
  built-in system prompt.
- **`page_supporting_files/pvcopilot_functions_code.txt`, `_packages_code.txt`, `_main_code.txt`**
  — optional; if absent, Step 4 falls back to a locally-generated reproducible script.

## Notes

- **PVPRO** runs synchronously here and can take 1–3 minutes; it needs DC voltage + current.
  The non-blocking background-job infrastructure from the old `pvcopilot.py` can be layered
  back in if you want progress polling.
