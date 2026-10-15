"""Regression tests for page_supporting_files/analysis_utils.py.

Each degradation method must recover the rate injected into the synthetic
data (see conftest.py), and the bug fixes merged in PR #1 must keep working.
"""
import numpy as np
import pandas as pd
import pytest

from page_supporting_files import analysis_utils as au

TOL = 0.15  # %/yr


@pytest.fixture(scope="module")
def daily(synthetic_df, mapping):
    keep, _ = au.low_irra_power_filter(au.normalize(synthetic_df.copy(), mapping), mapping)
    good = au.normalize(synthetic_df.loc[keep].copy(), mapping)
    return au.aggregate_daily(good, mapping["Irradiance"])


# ---------------------------------------------------------------- normalize
def test_normalize_full(synthetic_df, mapping):
    out = au.normalize(synthetic_df.copy(), mapping)
    assert out["norm"].notna().sum() > 0


def test_normalize_without_temperature(synthetic_df):
    out = au.normalize(synthetic_df[["p", "poa"]].copy(),
                       {"DC Power": "p", "Irradiance": "poa"})
    assert out["norm"].notna().sum() > 0


def test_normalize_power_only_trends_raw_power(synthetic_df):
    out = au.normalize(synthetic_df[["p"]].copy(), {"DC Power": "p"})
    assert (out["norm"] == out["p"]).all()


def test_normalize_requires_power(synthetic_df):
    with pytest.raises(KeyError):
        au.normalize(synthetic_df.copy(), {"Irradiance": "poa"})


def test_normalize_coerces_text_columns(synthetic_df, mapping):
    df = synthetic_df.copy()
    df["p"] = df["p"].astype(str)
    assert au.normalize(df, mapping)["norm"].notna().sum() > 0


# ------------------------------------------------------------------ filters
def test_irradiance_power_filter_is_unit_independent(synthetic_df, mapping):
    """A kW-scaled power column must keep the same rows as the W version
    (the old ratio-vs-W/m2 check dropped every row for kW files)."""
    w = synthetic_df.copy()
    kw = synthetic_df.copy()
    kw["p"] = kw["p"] / 1000
    keep_w, _ = au.low_irra_power_filter(au.normalize(w, mapping), mapping)
    keep_kw, _ = au.low_irra_power_filter(au.normalize(kw, mapping), mapping)
    assert len(keep_kw) > 1000
    assert abs(len(keep_w) - len(keep_kw)) / len(keep_w) < 0.01


def test_low_power_filter_drops_low_output(synthetic_df):
    keep, drop = au.low_power_filter(synthetic_df.copy(), {"DC Power": "p"})
    assert 0 < len(keep) < len(synthetic_df)
    assert len(keep) + len(drop) == len(synthetic_df)


# -------------------------------------------------------------- aggregation
def test_aggregate_daily_irradiance_weighted(daily):
    assert len(daily) > 1500
    assert isinstance(daily.index, pd.DatetimeIndex)


def test_aggregate_daily_power_only_path(synthetic_df):
    m = {"DC Power": "p"}
    df = au.normalize(synthetic_df[["p"]].copy(), m)
    keep, _ = au.low_power_filter(df, m)
    assert len(au.aggregate_daily(df.loc[keep])) > 100


# ------------------------------------------------------ degradation methods
def test_yoy_recovers_rate(daily, true_rd):
    rd, _ = au.compute_yoy(daily)
    assert abs(rd - true_rd) < TOL


def test_yoy_handles_sparse_sampling(daily, true_rd):
    rng = np.random.default_rng(1)
    rd, _ = au.compute_yoy(daily[rng.random(len(daily)) > 0.6])
    assert np.isfinite(rd) and abs(rd - true_rd) < 2 * TOL


def test_lr_recovers_rate(daily, true_rd):
    rd, _ = au.compute_lr(daily)
    assert abs(rd - true_rd) < TOL


def test_lr_robust_to_outlier_days(daily, true_rd):
    spiked = daily.copy()
    spiked.iloc[-10:] *= 1.5
    rd, _ = au.compute_lr(spiked)
    assert abs(rd - true_rd) < TOL


def test_energy_ratio_yoy_recovers_rate(synthetic_df, mapping, true_rd):
    rd, _fig, ci_width, n_days = au.compute_yoy_energy(synthetic_df.copy(), mapping)
    assert abs(rd - true_rd) < TOL
    assert n_days > 1500
    assert ci_width < au.MAX_CI_WIDTH


# -------------------------------------------------------- reliability checks
def test_rate_is_plausible():
    assert au.rate_is_plausible(-0.8)
    assert not au.rate_is_plausible(12.0)
    assert not au.rate_is_plausible(float("nan"))


def test_reliability_flags_too_few_points():
    ok, reasons = au.degradation_reliability(33.0, n_points=4)
    assert not ok and reasons


def test_reliability_accepts_normal_result():
    ok, _ = au.degradation_reliability(-0.8, n_points=1500, duration_years=6)
    assert ok


# ------------------------------------------------------------------ figures
def test_overview_figures_with_power_spike(synthetic_df, mapping):
    df = synthetic_df.iloc[:20000].copy()
    df.iloc[100, df.columns.get_loc("p")] = 2.2e7  # garbage spike
    assert au.make_overview_figures(df, mapping) is not None
