"""JJAS exact pairing and independent proxy-system evidence."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsoonpp.phase_c import PhaseCError
from monsoonpp.phase_c3 import (compare_synoptic_errors_by_system,
                                independent_event_counts,
                                track_synoptic_centres,
                                validate_exact_period)


def _paired(days=3):
    starts = pd.date_range("2024-06-01 03:00", periods=days, freq="D")
    labels = starts.normalize()
    shape = (days, 3, 4)
    obs = np.full(shape, 10.0, dtype="float32")
    fc = obs.copy()
    return xr.Dataset(
        {"fc": (("time", "lat", "lon"), fc), "obs": (("time", "lat", "lon"), obs)},
        coords={"time": labels, "lat": [18.0, 19.0, 20.0], "lon": [80.0, 81.0, 82.0, 83.0],
                "valid_start": ("time", starts), "valid_end": ("time", starts + pd.Timedelta(days=1))},
    )


def test_exact_period_rejects_a_gap():
    paired = _paired()
    facts = validate_exact_period(paired, "2024-06-01 03:00", "2024-06-04 03:00")
    assert facts["windows"] == 3 and facts["gaps"] == 0
    broken = paired.isel(time=[0, 2])
    with pytest.raises(PhaseCError, match="requires 3 paired windows"):
        validate_exact_period(broken, "2024-06-01 03:00", "2024-06-04 03:00")


def test_tracker_counts_one_intensifying_system_once():
    centres = pd.DataFrame([
        {"time": "2024-06-01 03:00", "lat": 20.0, "lon": 84.0, "class": "low_proxy"},
        {"time": "2024-06-01 09:00", "lat": 20.4, "lon": 84.5, "class": "depression_proxy"},
        {"time": "2024-06-03 03:00", "lat": 18.0, "lon": 88.0, "class": "low_proxy"},
    ])
    tracked, events = track_synoptic_centres(
        centres, {"track_max_gap_hours": 12, "track_max_speed_kmh": 65}
    )
    assert tracked.system_id.nunique() == 2
    first = events[events.contains_depression_proxy].iloc[0]
    assert first.primary_class == "depression_proxy"
    counts = independent_event_counts(tracked, events).set_index("regime")
    assert counts.loc["low_proxy", "independent_systems_with_class"] == 2
    assert counts.loc["low_proxy", "primary_class_systems"] == 1
    assert counts.loc["depression_proxy", "independent_systems_with_class"] == 1


def test_system_block_effects_keep_systems_separate_from_cells():
    paired = _paired(days=2)
    paired["fc"].values[:, :, :2] += 5.0
    synoptic = np.zeros(paired.fc.shape, dtype="int8")
    synoptic[:, :, :2] = 1
    regimes = xr.Dataset({"synoptic": (("time", "lat", "lon"), synoptic)}, coords={
        "time": paired.time, "lat": paired.lat, "lon": paired.lon,
    })
    tracked = pd.DataFrame([
        {"time": pd.Timestamp("2024-06-01 03:00"), "lat": 19.0, "lon": 80.5,
         "class": "low_proxy", "system_id": "SYS0001"},
        {"time": pd.Timestamp("2024-06-02 03:00"), "lat": 19.0, "lon": 80.5,
         "class": "low_proxy", "system_id": "SYS0002"},
    ])
    effects = compare_synoptic_errors_by_system(
        paired, regimes, tracked, np.ones((3, 4), dtype=bool),
        {"bootstrap_replicates": 30, "bootstrap_seed": 3,
         "min_comparison_days": 2, "min_comparison_cells": 2},
    )
    low = effects[effects.comparison == "low_proxy_vs_none"].iloc[0]
    assert low.independent_systems == 2
    assert low.independent_days == 2
    assert low.target_cells > low.independent_systems
    assert low.ci_estimable and low.meets_existing_day_cell_gate
    assert low.mean_error_ci_low == pytest.approx(5.0)
