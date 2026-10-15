"""Shared pytest setup for pvtools.

* Puts the repo root on sys.path so `page_supporting_files`, `pages`, `utils`
  import the same way they do when running `python index.py`.
* Sets dummy LLM API keys so modules that build an OpenAI client at import
  time don't fail. No test calls an LLM.
* Provides a synthetic PV dataset with a KNOWN degradation rate, so the
  analysis functions can be checked against ground truth.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OPENAI_API_KEY", "ci-dummy")
os.environ.setdefault("cborg_api_key", "ci-dummy")

TRUE_RD = -0.8  # %/yr injected into the synthetic data


@pytest.fixture(scope="session")
def true_rd():
    return TRUE_RD


@pytest.fixture(scope="session")
def mapping():
    return {"DC Power": "p", "Irradiance": "poa", "Module temperature": "tmod"}


@pytest.fixture(scope="session")
def synthetic_df():
    """6 years of 15-min data: clear/cloudy days, module temperature, a 5 kW
    system degrading at TRUE_RD %/yr, plus 1% measurement noise."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", "2023-12-31 23:45", freq="15min")
    doy = idx.dayofyear.values
    hr = idx.hour.values + idx.minute.values / 60
    sun = np.clip(np.sin((hr - 6) / 12 * np.pi), 0, None) * (
        0.85 + 0.15 * np.cos((doy - 172) / 365 * 2 * np.pi))
    cloud = rng.choice([1, 1, 1, 0.5, 0.2], size=len(idx))
    poa = 1000 * sun * cloud
    tmod = 25 + 25 * sun + rng.normal(0, 1, len(idx))
    years = (idx - idx[0]).days.values / 365.25
    p = (5000 * poa / 1000 * (1 - 0.004 * (tmod - 25))
         * (1 + TRUE_RD / 100) ** years * (1 + rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"poa": poa, "tmod": tmod, "p": p}, index=idx)
