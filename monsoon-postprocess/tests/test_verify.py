import numpy as np
import pytest

from src.verify import VerificationError, calculate_continuous_metrics


def test_continuous_metrics_use_only_valid_mask_cells() -> None:
    forecast = np.array([[2.0, 4.0], [999.0, 6.0]])
    observed = np.array([[1.0, 2.0], [np.nan, 3.0]])
    valid_mask = np.array([[1, 1], [0, 1]])

    result = calculate_continuous_metrics(
        forecast,
        observed,
        valid_mask,
        latitude=np.array([10.0, 11.0]),
        longitude=np.array([75.0, 76.0]),
    )

    assert result["valid_cell_count"] == 3
    assert result["bias_mm"] == 2.0
    assert result["mae_mm"] == 2.0
    assert result["rmse_mm"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert result["pearson_correlation"] == pytest.approx(1.0)
    assert result["observed_mean_mm"] == 2.0
    assert result["forecast_mean_mm"] == 4.0
    assert result["observed_maximum_mm"] == 3.0
    assert result["forecast_maximum_mm"] == 6.0
    assert result["observed_maximum_location"] == {"latitude": 11.0, "longitude": 76.0}
    assert result["forecast_maximum_location"] == {"latitude": 11.0, "longitude": 76.0}


def test_wet_area_fractions_use_strictly_positive_rainfall() -> None:
    result = calculate_continuous_metrics(
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 2.0]]),
        np.ones((2, 2), dtype=np.int8),
        latitude=np.array([10.0, 11.0]),
        longitude=np.array([75.0, 76.0]),
    )

    assert result["observed_wet_area_fraction"] == 0.5
    assert result["forecast_wet_area_fraction"] == 0.5


def test_valid_mask_cannot_select_nan_observation() -> None:
    with pytest.raises(VerificationError, match="selects 1 non-finite"):
        calculate_continuous_metrics(
            np.array([[1.0]]),
            np.array([[np.nan]]),
            np.array([[1]]),
            latitude=np.array([10.0]),
            longitude=np.array([75.0]),
        )
