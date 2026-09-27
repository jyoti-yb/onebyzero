import numpy as np
import pytest
import xarray as xr

from src.align import AlignmentError, normalize_and_select_exact_grid, validate_temporal_pair


def test_temporal_pair_matches_imd_ending_date() -> None:
    result = validate_temporal_pair(
        {
            "initialization_time": "2024-08-31T00:00:00Z",
            "valid_start": "2024-08-31T03:00:00Z",
            "valid_end": "2024-09-01T03:00:00Z",
        },
        np.datetime64("2024-09-01"),
    )

    assert result["period_match"] is True
    assert result["imd_observation_date"] == "2024-09-01"


def test_temporal_pair_rejects_mismatched_period() -> None:
    with pytest.raises(AlignmentError, match="Temporal alignment failed"):
        validate_temporal_pair(
            {
                "initialization_time": "2024-08-31T00:00:00Z",
                "valid_start": "2024-08-31T03:00:00Z",
                "valid_end": "2024-09-01T00:00:00Z",
            },
            np.datetime64("2024-09-01"),
        )


def test_exact_grid_selection_sorts_and_normalizes_without_interpolation() -> None:
    forecast = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("latitude", "longitude"),
        coords={
            "latitude": [7.0, 6.75, 6.5],
            "longitude": [66.5, 66.75, 67.0, 359.75],
        },
    )

    aligned, diagnostics = normalize_and_select_exact_grid(
        forecast,
        np.array([6.5, 6.75, 7.0]),
        np.array([66.5, 66.75, 67.0]),
    )

    np.testing.assert_array_equal(aligned.latitude, [6.5, 6.75, 7.0])
    np.testing.assert_array_equal(aligned.longitude, [66.5, 66.75, 67.0])
    np.testing.assert_array_equal(aligned.values, [[8, 9, 10], [4, 5, 6], [0, 1, 2]])
    assert diagnostics["interpolation_performed"] is False
    assert diagnostics["maximum_latitude_difference"] == 0.0
    assert diagnostics["maximum_longitude_difference"] == 0.0
