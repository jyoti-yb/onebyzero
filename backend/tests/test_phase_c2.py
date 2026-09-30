"""Expanded atlas sample gates and day-block statistical comparisons."""
import numpy as np
import pandas as pd
import xarray as xr

from monsoonpp.phase_c2 import _decision, compare_synoptic_errors


def _case(days=12):
    rng = np.random.default_rng(7)
    lat, lon = np.arange(4), np.arange(6)
    shape = (days, len(lat), len(lon))
    obs = np.full(shape, 20.0, dtype="float32")
    error = rng.normal(0, 2, shape).astype("float32")
    synoptic = np.zeros(shape, dtype="int8")
    synoptic[:, :, :2] = 1
    error[:, :, :2] += 5.0
    paired = xr.Dataset(
        {"fc": (("time", "lat", "lon"), obs + error), "obs": (("time", "lat", "lon"), obs)},
        coords={"time": pd.date_range("2024-07-01", periods=days), "lat": lat, "lon": lon},
    )
    regimes = xr.Dataset({"synoptic": (("time", "lat", "lon"), synoptic)}, coords=paired.coords)
    return paired, regimes


def test_matched_day_bootstrap_finds_material_low_proxy_shift():
    paired, regimes = _case()
    effects = compare_synoptic_errors(
        paired, regimes, np.ones((4, 6), dtype=bool),
        {"bootstrap_replicates": 200, "min_comparison_days": 10, "min_comparison_cells": 50,
         "material_hedges_g": 0.2, "material_mae_mm": 1.0, "bootstrap_seed": 3},
    )
    low = effects[effects.target_regime == "low_proxy"].iloc[0]
    assert low.matched_days == 12
    assert low.statistically_usable and low.material_difference
    assert low.mean_error_ci_low > 0 and low.hedges_g_ci_low > 0
    depression = effects[effects.target_regime == "depression_proxy"].iloc[0]
    assert not depression.statistically_usable and not depression.material_difference


def test_proxy_observations_never_open_ml_gate():
    effects = pd.DataFrame([{"target_regime": "low_proxy", "material_difference": True}])
    move, reasons = _decision(122, 4, "SMOKE_TEST_PROXY", effects, 1, {})
    assert not move
    assert any("not final IMD truth" in reason for reason in reasons)


def test_one_day_depression_interval_is_not_estimable():
    paired, regimes = _case(days=1)
    regimes["synoptic"] = xr.where(regimes.synoptic == 1, 2, regimes.synoptic)
    effects = compare_synoptic_errors(
        paired, regimes, np.ones((4, 6), dtype=bool),
        {"bootstrap_replicates": 20, "min_comparison_days": 2, "min_comparison_cells": 1},
    )
    depression = effects[effects.target_regime == "depression_proxy"].iloc[0]
    assert np.isnan(depression.hedges_g_ci_low)
    assert not depression.statistically_usable


def test_length_and_effect_gates_are_explicit():
    effects = pd.DataFrame([{"target_regime": "low_proxy", "material_difference": False}])
    move, reasons = _decision(31, 1, "FINAL_TRUTH", effects, 1, {})
    assert not move
    assert len(reasons) == 3
