# PV-Copilot: switching the analysis code to pvlib / rdtools / pvanalytics

Status: `pages/pvcopilot.py` now imports the package-backed functions
(`analysis_utils_pkg`). The previous page is kept verbatim as
`pages/pvcopilot_legacy.py` (not imported by `index.py` — it registers the
same component ids, so the two cannot be loaded in one Dash app at once; to go
back, swap the two file names). `analysis_utils.py` and
`pvcopilot_filter_functions.py` are unchanged and still provide HW, ARIMA,
clear-sky, column identification, overview figures and the PVPRO layout
estimate.

## New files

| File | What it is |
|---|---|
| `page_supporting_files/analysis_utils_pkg.py` | Drop-in replacements (same names, signatures, return shapes) backed by pvlib, rdtools 2.1.8 and (optionally) pvanalytics 0.2.2 |
| `page_supporting_files/pvcopilot_functions_code_pkg.txt` | Package-based version of the code snippet exported to users in Step 4 (`get_full_code`) |
| `tests/test_analysis_utils_pkg.py` | Old-vs-new comparison tests on `data/sys_1278…` and `data/sys_1403…`; run as pytest or as a script for a comparison table |
| `requirements_pkg.txt` | The extra dependencies and the install trick that avoids the 190–450 MB xgboost wheel |

## Install (local + Dockerfile)

```
pip install -r requirements_pkg.txt          # xgboost-cpu, h5py, matplotlib, pvanalytics, ruptures
pip install --no-deps rdtools==2.1.8         # rdtools' metadata insists on the GPU xgboost wheel
```

Verified in a clean Python 3.11 venv with the exact pins of `requirements.txt`
(numpy 1.26.4, pandas 2.0.3, scipy 1.13.1, pvlib 0.9.5): no pin changes, no
conflicts. On-disk cost: rdtools 7 MB + xgboost-cpu 19 MB + h5py 11 MB +
matplotlib/pillow 42 MB ≈ 80 MB; pvanalytics + scikit-image stack ≈ 45 MB.
(`pip install rdtools` without the trick would add ≈ 450 MB to the image.)

## What changed in `pages/pvcopilot.py`

Only the import block (L17–L21 of the legacy file). Diff:

```python
# removed (legacy imports)
# from page_supporting_files.analysis_utils import make_overview_figures, normalize, low_irra_power_filter, aggregate_daily, compute_yoy, get_full_code
# from page_supporting_files.analysis_utils import compute_lr, compute_hw, compute_arima, compute_csd, compute_pvpro
# from page_supporting_files.pvcopilot_filter_functions import identify_outliers_iqr, clear_sky_filter, basic_value_filter

# added
from page_supporting_files.analysis_utils import make_overview_figures, compute_hw, compute_arima
from page_supporting_files.pvcopilot_filter_functions import clear_sky_filter
from page_supporting_files.analysis_utils_pkg import (
    normalize, basic_value_filter, low_irra_power_filter, identify_outliers_iqr,
    aggregate_daily, compute_yoy, compute_lr, compute_csd, compute_pvpro,
    get_full_code,
)
# (estimate_pvpro_params, downsize_block_mean, parse_contents: unchanged imports)
```

Everything downstream (`_do_filter`, the metric dispatcher `_dispatch_stat_method`,
the PVPRO background job) is untouched — signatures and return values are
identical. Verified: the new page imports, registers its 85 callbacks and
builds its layout in the test env; PVPRO figures round-trip through the
diskcache job store and Plotly's JSON encoder.

`get_full_code` is now `analysis_utils_pkg.get_full_code`: the Step 4 export
emits `pvcopilot_functions_code_pkg.txt` (rdtools / pvanalytics calls) and a
main block whose `compute_*` return `(rd, ci)`. HW / ARIMA are not offered in
the exported script (no package equivalent). The old LLM-templated exporter
still exists in `analysis_utils.get_full_code`.

Not changed on purpose: the hard-coded `UTC → US/Pacific` block in
`_do_filter`. `analysis_utils_pkg.auto_fix_timezone(df, time_key, power_key)`
is ready if you want to replace it, but that changes behaviour, not just the
backend.

Optional: `compute_yoy(daily, return_ci=True)` → `(rd, fig, ci)`; the YoY
figure legend already shows the 68 % CI.

### UI changes for the 2-year YoY requirement

rdtools' `degradation_year_on_year` refuses records shorter than two years,
so the page was adjusted (all in `pages/pvcopilot.py`):

* `_MIN_YEARS_FOR_YOY` 1.0 → 2.0.
* `gate_yoy_by_duration` now also listens to `dataframe-filtered` (Step 2
  output). The raw record can be long enough while the filtered series is not
  (sys_1403: 3.1 yr raw, 1.8 yr after filtering because DC power is NaN before
  2016-04). The gate uses the shorter of the two spans: when < 2 yr the YoY
  checkbox is disabled, YoY is dropped from the selection (LR becomes the
  selection if nothing else is checked) and a note appears under the method
  list: "Year-over-Year (rdtools) needs at least 2 years of usable data; this
  record spans 1.8 years after filtering, so YoY is disabled and Linear
  regression is selected instead."
* Safety net at calculation time (`_daily_span_years` / `_yoy_possible` on
  the daily series): Advanced single method → LR with a warning callout;
  Advanced multi-method → YoY n/a + callout; Simple mode → LR with a callout,
  the summary card labelled "Linear regression", stash `method="LR"`,
  `method_requested="YOY"`, `daily_span_years`.
* Simple mode bug fixed on the way: `toggle_simple_start_and_result` only
  revealed the result view for `method in ("YOY","PVPRO")`, so an LR fallback
  produced a blank Simple-mode result. Now `("YOY","LR","PVPRO")`.

## Upstream-package logos

`assets/function_logos/` (rdtools_logo.png, pvlib_logo.png,
pvanalytics_logo.png) is surfaced wherever that package does the work, the way
the PVPRO mark already was:

| Where | Marks |
|---|---|
| Step 2, clear-sky settings panel | RdTools, pvlib, PVAnalytics |
| Step 2, low irradiance / power panel | RdTools |
| Step 2, outlier removal panel | PVAnalytics |
| Step 2, time zone panel | none — still the hand-written correction |
| Step 3, YoY / LR / CSD options | RdTools |
| Step 3, HW / ARIMA options | none — statsmodels, no mark |
| Step 3, PVPRO option | pvlib beside the existing PVPRO mark (the single-diode maths is pvlib's) |

In Step 3 the marks sit on the CATEGORY lines, not on each option: one
RdTools mark on "statistical / trend methods" and pvlib + PV-Pro on
"single-diode-model fitting". Per-option marks meant three identical RdTools
marks in one column. Both category headings are now 14 px / weight 800 (were
11 px / 600). Note the category line credits the category, not every member:
HW and ARIMA are statsmodels — the per-method attribution lives in "Metric
details".

Helpers: `_PKG_LOGOS` (file, alt, project URL, optical scale) and
`_pkg_logo` / `_pkg_logo_row`. Two details worth keeping:

* The marks are sized by HEIGHT with a per-logo scale (RdTools 0.72, pvlib
  1.30, PVAnalytics 1.05), because equal height is not equal weight: the
  aspect ratios differ wildly (6.6:1 / 2.4:1 / 4.2:1) and RdTools is a heavy
  wordmark while PVAnalytics is a light one. The scales were set by rendering
  the three side by side and matching them by eye.
* In the Step 3 single-diode heading the PV-Pro mark comes first and pvlib
  second: the method is PVPRO's, pvlib supplies the maths underneath it.
* Marks inside a checkbox's `<label>` (the Step 3 options) are plain images,
  not links — a link there would both open the project page and toggle the
  method.


## Step 2 filter details

The single "Filter details (equations & references)" accordion under the Apply
button is gone. Each filter's explanation now sits in its own settings panel,
under that filter's parameters, as a collapsed "How this filter works"
disclosure (`_filter_detail_body` / `_filter_detail_panel`), and each filter
carries its OWN reference list (`_filter_references`), numbered from [1] within
that filter — nothing is left at the bottom of the step, and no reader has to
match a superscript against a list several screens away. The expanded
disclosure is the same soft-blue card (`_exp_outer_style`) the shared list used
to have, so an open explanation reads as one block.

The first tab's panel has no parameters, so instead of the "no adjustable
settings" one-liner it shows a short bullet summary of what always runs
(`_filter_summary_body` / `_filter_summary_bullets`), with the full text in the
details disclosure below it.

The first tab is "Basic checks & time correction": the physical range limits
(which never had a tab, only a paragraph in the accordion) and the time-zone /
DST correction, together, because neither is a judgement call about which
points are interesting — they are what has to be true before the other filters
mean anything. The row is locked on: `FILTER_TABS` carries a `locked` flag, the
tab shows an "always on" pill and its checkbox is rendered checked and
`disabled`. The checkbox component stays (id `cb-timezone`) so the clientside
sync still puts "timezone" into `filter-options` and `_do_filter` keeps running
that step — no pipeline change, only the UI stops offering to turn it off.

The clear-sky entry was rewritten while moving: it still described the
hand-written smoothness + energy scoring. It now documents the reference
(modelled vs empirical, the clock and level corrections) and the two
detectors, and the reference list gained Ineichen & Perez (2002) and
Reno & Hansen (2016).

Still stale, not touched: the YoY / LR / CSD entries in *Metric* details
describe the pre-rdtools formulas.


## What maps to what

| Old (hand-written) | New | Result |
|---|---|---|
| `normalize` | `rdtools.normalization.pvwatts_dc_power` (rated power = 1) | identical (rtol 1e-12) |
| `basic_value_filter` | `rdtools.filtering.poa_filter` + `tcell_filter` (+ pandas for P ≥ min) | identical index sets |
| `low_irra_power_filter` | `poa_filter` + `rdtools.filtering.normalized_filter` (+ pandas for P > r·G) | identical index sets |
| `identify_outliers_iqr` | `pvanalytics.quality.outliers.tukey` (pandas fallback) | identical index sets |
| `aggregate_daily` | `rdtools.aggregation.aggregation_insol` | identical (max Δ 6e-11) |
| `compute_yoy` | `rdtools.degradation.degradation_year_on_year` | 8-day pair tolerance, first-year recentering, bootstrap CI; **returns NaN for < 2 years** (old code returned a number from a few months of pairs) |
| `compute_lr` | `rdtools.degradation.degradation_ols` | slope/intercept instead of slope/mean; Δ ≈ 0.03–0.05 %/yr on the examples |
| `compute_csd` | `rdtools.degradation.degradation_classical_decomposition` | 365-day centred MA + OLS + Mann-Kendall, needs a complete daily series (gaps interpolated); differs from `seasonal_decompose(period=12)` by 0.2–0.3 %/yr on the examples |
| `compute_hw`, `compute_arima` | — | no package equivalent, keep in `analysis_utils.py` |
| `clear_sky_filter`, `detect_clear_days` | `rdtools.filtering.csi_filter` / `pvlib.clearsky.detect_clearsky`, against a reference from `pvlib.location.Location.get_clearsky` + `pvlib.irradiance.get_total_irradiance` (coordinates given) or a data-derived envelope (not given); geometry fitted with `pvanalytics.system.infer_orientation_fit_pvwatts`, level calibrated with `rdtools.normalization.irradiance_rescale` | precision 0.99 vs 0.92 for the hand-written scoring on synthetic data with known clear days, same recall |
| `_calcparams_desoto_lite`, `_mpp/_voc/_isc_vectorised` (~250 lines) | `pvlib.pvsystem.calcparams_desoto`, `pvlib.singlediode.bishop88_mpp / bishop88_v_from_i / bishop88_i_from_v` | forward model agrees within 0.2 % (lite version had an extra 1e-5 S shunt term); P_mp,ref rate within 0.06 %/yr on the examples; ~15 % slower per window |
| `auto_fix_timezone` DST heuristic | `pvanalytics.features.daytime` + `quality.time.shifts_ruptures` | detects and corrects an injected 1-h jump on sys_1403 (needs `ruptures`) |

Note the sign convention of `shifts_ruptures`: the returned minutes are
*added* to the timestamps to align them with the reference.

## Clear-sky filtering

The published detectors all compare the measurement against a CLEAR-SKY
REFERENCE, and only that reference needs site information — so it is produced
separately (`clear_sky_reference`) and the classification is always a package
call:

* **Reference, coordinates given** — `pvlib.location.Location.get_clearsky`
  (Ineichen-Perez, with pvlib's bundled Linke-turbidity climatology) transposed
  to the array plane by `pvlib.irradiance.get_total_irradiance`. Tilt/azimuth
  come from the user, else are fitted from the measured power with
  `pvanalytics.system.infer_orientation_fit_pvwatts` (recovered 13°/207° from a
  synthetic 15°/210° array), else fall back to tilt = |latitude|, equator-facing.
* **Reference, no coordinates** — an empirical envelope: a high quantile of each
  time-of-day over a ±30-day window. Needs no input at all; its weakness is a
  window with no clear day in it, where the envelope sits too low.
* **Two corrections before use.** The modelled curve is first put on the
  record's own clock — the median offset between measured and modelled daily
  peak times, which absorbs DST, a logger left on UTC and longitude error alike
  (an offset above 3 h is reported as a warning instead, since that means the
  coordinates are wrong). It is then put on the record's own scale with
  `rdtools.normalization.irradiance_rescale`, because the clear-sky index is a
  ratio and array geometry, sensor calibration and soiling all bias the level.
  The orientation fit runs *after* the clock correction: fitting it first pulls
  the azimuth toward the clock error (249° instead of 210° in the test case).
* **Classification** — `rdtools.filtering.csi_filter` (measured/clear-sky within
  ±0.15 of 1) by default, and `pvlib.clearsky.detect_clearsky` (the five Reno &
  Hansen criteria) for data sampled at 15 min or finer. The Reno thresholds are
  calibrated for 1-minute GHI; on the hourly example records they reject almost
  everything (only 4.8 % of samples pass `slope_max`), so `method="auto"` does
  not use them there.
* A day is kept when at least `day_fraction` (default 0.5) of its daytime
  samples are classified clear — the same whole-day contract as before.

Measured on synthetic data with known clear days (55 % clear, 2 years hourly):

| reference | precision | recall |
|---|---|---|
| hand-written smoothness + energy | 0.92 | 1.00 |
| empirical envelope | 0.99 | 1.00 |
| pvlib model, coordinates only | 0.97 | 0.76 |
| pvlib model + geometry fitted from power | 0.97 | 1.00 |
| pvlib model + true geometry | 0.97 | 1.00 |

(the "coordinates only" row is why the orientation fit is worth having: the
rule-of-thumb tilt was 35° against a real 15°, and the seasonal shape error
costs recall.)

**UI.** Step 2 is now master/detail: the four filters are a list on the left
(checkbox = on/off, click = select) and the selected filter's settings fill a
panel on the right, so the clear-sky section no longer stretches a grid row.
Every parameter input stays mounted and hidden when its filter is not selected,
because the Apply-filters callback reads them all as `State`s. The layout lives
in `FILTER_TABS` / `_filter_tab` / `_filter_panel`, the selection in
`select_filter_tab`, and the two-column CSS in `assets/pvcopilot_styles.css`
(`.pvc-advanced-filter-split`, stacks below 900 px).

The clear-sky settings panel offers a **City** search above the coordinate
boxes, backed by the same GeoNames `data/cities15000.csv` the Field
Degradation page uses; picking a city writes into the latitude/longitude
inputs (it is a shortcut, not a separate setting, so a typed correction
always wins). Chrome's oversized number steppers are suppressed page-wide in
the CSS, matching what the app already did for inputs tagged `.pnum`.

The panel offers the
clear-sky index tolerance, the minimum clear share of a day, and four optional
site fields — latitude, longitude, tilt, azimuth. Coordinates are optional: left
blank, the empirical envelope is used. The filter summary line reports which
reference was built, which detector ran, and how many days came out clear.
Simple mode has no site fields and uses the empirical path.


## Test results (`python tests/test_analysis_utils_pkg.py`)

```
dataset   metric                        old (hand-written)    new (packages)  note
sys_1278  filter pipeline (kept pts)                 20096             20096
sys_1278  daily points                                2384              2384  max|Δ|=5.8e-11
sys_1278  YoY %/yr                                  -0.086            +0.025  CI68 [-0.10,+0.14]
sys_1278  LR %/yr                                   +0.868            +0.893
sys_1278  CSD %/yr                                  +0.872            +0.566
sys_1403  filter pipeline (kept pts)                  4760              4760
sys_1403  daily points                                 644               644  max|Δ|=2.7e-12
sys_1403  YoY %/yr                                  -2.247               nan  < 2 yr of filtered data (2016-04 → 2018-02)
sys_1403  LR %/yr                                   -2.353            -2.302
sys_1403  CSD %/yr                                  -2.349            -2.570
sys_1278  PVPRO Pmp %/yr                            -0.017            +0.004  layout 11 mod/str × 30 str, 66 s / 76 s
sys_1403  PVPRO Pmp %/yr                            +0.757            +0.695  layout 11 mod/str × 1 str, 27 s / 32 s
```

`pytest tests/test_analysis_utils_pkg.py` → 15 passed (≈ 50 s).

## Things noticed on the way (pre-existing, not changed)

* `requirements.txt` has `# rdtools` / `# xgboost==1.7.6` commented out and
  `base_env.yml` lists rdtools 3.0.0 / pvanalytics / pvpro / solar-data-tools —
  the two dependency lists have diverged; Docker only uses `requirements.txt`.
* `compute_pvpro` with the default layout (`modules_per_string=1,
  parallel_strings=1`) on sys_1278 pushes every window fit to the parameter
  bounds and reports exactly 0.000 %/yr (old and new alike). The Step 3
  auto-fill (`estimate_pvpro_params`) avoids that; worth guarding for in the UI.
* `page_supporting_files/rdtools_function_description.txt` documents the
  rdtools **3.x** signature (`uncertainty_method`); 2.1.8 does not have that
  argument. `analysis_utils_pkg.compute_yoy` handles both.

## Landing hero — floating package marks (UI)

The landing header (`common_header`) is now a two-column flex row: the text
column keeps everything it had (`flex: 1 1 520px; min-width: 0`), and a
250 × 128 px orbit cloud of the four package marks fills the blank right half.

* `_HERO_LOGOS` holds, per mark: file, alt, GitHub URL, width, left/top inside
  the box, which keyframe it uses, its duration and a negative delay.
* Widths are set by **equal optical area**, not equal height — `w = sqrt(A·AR)`
  with `A ≈ 2200 px²`. RdTools is a 6.6:1 wordmark and pvlib a 2.4:1 lockup, so
  matching heights would have made RdTools roughly three times the ink; it is
  held a little under its equal-area width (120 px) because a wide wordmark
  still reads as the biggest thing in the group at equal area.
* Positions are absolute px inside a fixed box, not percentages: the wordmarks
  are fixed-width, so a percentage grid would let them collide at some widths.
* The marks sit **on two tilted ellipses** (rx/ry 108/37 and 70/24, both rotated
  -16 deg about the box centre), drawn behind them as 1 px rings — one solid,
  one dashed — with a third, purely decorative inner ring (96 × 42) and a small
  amber "sun" at the centre. The five angles were solved by search: on-ring, at
  least 8 px apart, inside the box, and leaving the centre clear.
  The rings are plain divs with `border-radius: 50%`; an ellipse costs nothing
  that way and needs no SVG component type that Dash's `html` namespace lacks.
* Nothing travels the rings. Each ring instead **glows**: its border fades
  between rgba(47,107,255,0.10) and 0.38 while a soft halo (outer and inset
  box-shadow) swells and fades, on staggered 5.2 / 6.1 / 7.4 s periods so the
  three never pulse together. The rings do not rotate — a tilted ellipse in
  rotation reads as tumbling rather than orbiting.
* The credit is carried by the subtitle itself ("Upload a PV time-series — PV
  Copilot computes the degradation rate on established open-source PV
  packages") rather than by a separate caption under the marks:
  one line of prose people read, instead of two lines competing for the same
  corner.
* The five angles are solved against an **even 72 deg spread** (each mark may
  jitter +/- 16 deg off its slot) with a global rotation offset, then scored on
  largest angular gap and pairwise clearance. That is what keeps the marks from
  bunching into two corners: the accepted layout has a largest gap of 76 deg
  and at least 9 px of clearance between any two marks.
* The card's inner block is now a top row (eyebrow + headline + subtitle, with
  the cloud beside them, `align-items: center`) followed by the development
  note, which spans the full card width. The cloud therefore lines up with the
  three text lines rather than being centred against the note as well.
* Three keyframes (`pvcFloatA/B/C`) with durations 5.9–8.3 s, `alternate`,
  `ease-in-out`, and staggered negative delays so the five never swing in
  phase. The drift is small on purpose — about ±3 px and ±0.7° — so the marks
  breathe without the group looking unsettled. Transform-only, so the whole
  thing stays on the compositor.
* `:hover` pauses the mark and lifts it to full opacity, so the GitHub link is
  hittable on a moving target.
* `@media (max-width: 980px)` hides the cloud — below that the card is one
  column and the cloud would squeeze the headline.
* `@media (prefers-reduced-motion: reduce)` disables the animation.

The CSS is appended to the `<style>` element the clientside callback already
injects into `<head>` for the number-spinner fix, for the same reason:
`assets/pvcopilot_styles.css` has not been reaching the browser reliably.

**New assets.** The shipped marks carry an opaque white plate (RdTools has no
alpha at all), which showed as four pale rectangles on the glass card.
`mix-blend-mode: multiply` does not fix it — an animated, `will-change`
element forms its own stacking context, so it blends against nothing. So the
four marks were re-cut with the white made transparent on a soft 232→250 ramp
(the ramp keeps antialiased glyph edges from turning into a staircase) and
saved beside the originals, which are untouched:

```
assets/function_logos/rdtools_logo_flat.png
assets/function_logos/pvlib_logo_flat.png
assets/function_logos/pvanalytics_logo_flat.png
assets/function_logos/sdt_logo_flat.png
assets/pvpro_logo_flat.png
```

**Solar Data Tools** is shown at the user's request even though the app does
not import it (it was rejected during the dependency survey — 114 packages and
a forced numpy 2.x). Its mark is credit, not a dependency claim; nothing in
Step 2 or Step 3 attributes a filter or a metric to it.
