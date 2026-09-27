"""Continuous rainfall verification over the paired GFS-IMD valid mask."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


class VerificationError(RuntimeError):
    """Raised when paired rainfall cannot be verified safely."""


METRIC_DEFINITIONS = {
    "bias_mm": "mean(forecast_rainfall - observed_rainfall)",
    "mae_mm": "mean(abs(forecast_rainfall - observed_rainfall))",
    "rmse_mm": "sqrt(mean((forecast_rainfall - observed_rainfall)^2))",
    "pearson_correlation": "Pearson correlation between forecast and observed rainfall",
    "observed_wet_area_fraction": "fraction of valid cells with observed rainfall > 0 mm",
    "forecast_wet_area_fraction": "fraction of valid cells with forecast rainfall > 0 mm",
}


def _as_datetime_string(value: Any) -> str:
    return str(np.datetime64(value, "D"))


def _maximum_location(
    values: np.ndarray,
    mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    times: np.ndarray | None,
) -> dict[str, Any]:
    masked = np.where(mask, values, -np.inf)
    index = np.unravel_index(int(np.argmax(masked)), values.shape)
    location: dict[str, Any] = {
        "latitude": float(latitude[index[-2]]),
        "longitude": float(longitude[index[-1]]),
    }
    if values.ndim == 3:
        if times is None:
            raise VerificationError("Pooled maximum location requires time coordinates.")
        location["date"] = _as_datetime_string(times[index[0]])
    return location


def calculate_continuous_metrics(
    forecast: Any,
    observed: Any,
    valid_mask: Any,
    *,
    latitude: Any,
    longitude: Any,
    times: Any | None = None,
    wet_threshold_mm: float = 0.0,
) -> dict[str, Any]:
    """Calculate continuous metrics using only cells selected by ``valid_mask``."""

    forecast_values = np.asarray(forecast, dtype=np.float64)
    observed_values = np.asarray(observed, dtype=np.float64)
    mask_values = np.asarray(valid_mask) == 1
    latitude_values = np.asarray(latitude, dtype=np.float64)
    longitude_values = np.asarray(longitude, dtype=np.float64)
    time_values = None if times is None else np.asarray(times)
    if forecast_values.shape != observed_values.shape or forecast_values.shape != mask_values.shape:
        raise VerificationError("Forecast, observation, and valid_mask shapes must match.")
    if forecast_values.ndim not in {2, 3}:
        raise VerificationError("Rainfall arrays must be two-dimensional or three-dimensional.")
    if forecast_values.shape[-2:] != (latitude_values.size, longitude_values.size):
        raise VerificationError("Latitude/longitude lengths do not match rainfall dimensions.")
    if forecast_values.ndim == 3 and (
        time_values is None or time_values.size != forecast_values.shape[0]
    ):
        raise VerificationError("Three-dimensional rainfall requires matching time coordinates.")
    if not math.isfinite(wet_threshold_mm) or wet_threshold_mm < 0:
        raise VerificationError("wet_threshold_mm must be finite and non-negative.")

    valid_count = int(mask_values.sum())
    if valid_count == 0:
        raise VerificationError("valid_mask selects no cells.")
    invalid_selected = mask_values & (
        ~np.isfinite(forecast_values) | ~np.isfinite(observed_values)
    )
    if invalid_selected.any():
        raise VerificationError(
            f"valid_mask selects {int(invalid_selected.sum())} non-finite forecast/observation cells."
        )

    selected_forecast = forecast_values[mask_values]
    selected_observed = observed_values[mask_values]
    error = selected_forecast - selected_observed
    forecast_std = float(selected_forecast.std())
    observed_std = float(selected_observed.std())
    correlation = (
        float(np.corrcoef(selected_forecast, selected_observed)[0, 1])
        if valid_count >= 2 and forecast_std > 0 and observed_std > 0
        else None
    )
    observed_maximum = float(selected_observed.max())
    forecast_maximum = float(selected_forecast.max())
    return {
        "valid_cell_count": valid_count,
        "bias_mm": float(error.mean()),
        "mae_mm": float(np.abs(error).mean()),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "pearson_correlation": correlation,
        "observed_mean_mm": float(selected_observed.mean()),
        "forecast_mean_mm": float(selected_forecast.mean()),
        "observed_maximum_mm": observed_maximum,
        "forecast_maximum_mm": forecast_maximum,
        "observed_maximum_location": _maximum_location(
            observed_values, mask_values, latitude_values, longitude_values, time_values
        ),
        "forecast_maximum_location": _maximum_location(
            forecast_values, mask_values, latitude_values, longitude_values, time_values
        ),
        "wet_threshold_mm": wet_threshold_mm,
        "observed_wet_area_fraction": float(
            (selected_observed > wet_threshold_mm).mean()
        ),
        "forecast_wet_area_fraction": float(
            (selected_forecast > wet_threshold_mm).mean()
        ),
    }


def _csv_row(scope: str, date_label: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    observed_location = metrics["observed_maximum_location"]
    forecast_location = metrics["forecast_maximum_location"]
    return {
        "scope": scope,
        "date": date_label,
        "valid_cell_count": metrics["valid_cell_count"],
        "bias_mm": metrics["bias_mm"],
        "mae_mm": metrics["mae_mm"],
        "rmse_mm": metrics["rmse_mm"],
        "pearson_correlation": metrics["pearson_correlation"],
        "observed_mean_mm": metrics["observed_mean_mm"],
        "forecast_mean_mm": metrics["forecast_mean_mm"],
        "observed_maximum_mm": metrics["observed_maximum_mm"],
        "forecast_maximum_mm": metrics["forecast_maximum_mm"],
        "observed_maximum_date": observed_location.get("date", date_label),
        "observed_maximum_latitude": observed_location["latitude"],
        "observed_maximum_longitude": observed_location["longitude"],
        "forecast_maximum_date": forecast_location.get("date", date_label),
        "forecast_maximum_latitude": forecast_location["latitude"],
        "forecast_maximum_longitude": forecast_location["longitude"],
        "observed_wet_area_fraction": metrics["observed_wet_area_fraction"],
        "forecast_wet_area_fraction": metrics["forecast_wet_area_fraction"],
        "wet_threshold_mm": metrics["wet_threshold_mm"],
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_json(payload: Mapping[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def verify_paired_dataset(
    paired_path: Path,
    metrics: list[str] | None = None,
    *,
    wet_threshold_mm: float = 0.0,
) -> dict[str, Any]:
    """Calculate daily and pooled continuous verification for one paired dataset."""

    del metrics  # Continuous smoke-test verification always emits the canonical set.
    if not paired_path.is_file():
        raise VerificationError(f"Paired dataset does not exist: {paired_path}")
    with xr.open_dataset(paired_path) as dataset:
        required = {"forecast_rainfall", "observed_rainfall", "valid_mask"}
        missing = required - set(dataset.data_vars)
        if missing:
            raise VerificationError(f"Paired dataset is missing variables: {sorted(missing)}")
        forecast = dataset["forecast_rainfall"].load()
        observed = dataset["observed_rainfall"].load()
        valid_mask = dataset["valid_mask"].load()
        latitude = np.asarray(dataset.latitude.values)
        longitude = np.asarray(dataset.longitude.values)
        times = np.asarray(dataset.time.values)

    daily_results = []
    csv_rows = []
    for index, time_value in enumerate(times):
        date_label = _as_datetime_string(time_value)
        result = calculate_continuous_metrics(
            forecast.values[index],
            observed.values[index],
            valid_mask.values[index],
            latitude=latitude,
            longitude=longitude,
            wet_threshold_mm=wet_threshold_mm,
        )
        result = {"scope": "daily", "date": date_label, **result}
        daily_results.append(result)
        csv_rows.append(_csv_row("daily", date_label, result))

    pooled = calculate_continuous_metrics(
        forecast.values,
        observed.values,
        valid_mask.values,
        latitude=latitude,
        longitude=longitude,
        times=times,
        wet_threshold_mm=wet_threshold_mm,
    )
    pooled_label = f"POOLED_{_as_datetime_string(times[0])}_{_as_datetime_string(times[-1])}"
    pooled = {"scope": "pooled_7_day", "date": pooled_label, **pooled}
    csv_rows.append(_csv_row("pooled_7_day", pooled_label, pooled))
    return {
        "status": "PASS",
        "paired_dataset": str(paired_path),
        "mask_rule": (
            "Metrics include only cells where valid_mask == 1; selected forecast and "
            "observation values must both be finite."
        ),
        "metric_definitions": METRIC_DEFINITIONS,
        "wet_threshold_mm": wet_threshold_mm,
        "daily_valid_cell_counts": [
            result["valid_cell_count"] for result in daily_results
        ],
        "pooled_valid_cell_count": pooled["valid_cell_count"],
        "daily_results": daily_results,
        "pooled_result": pooled,
        "csv_rows": csv_rows,
    }


def write_verification_reports(
    paired_path: Path,
    csv_path: Path,
    summary_path: Path,
    *,
    wet_threshold_mm: float = 0.0,
) -> dict[str, Any]:
    """Calculate metrics and write the requested CSV and JSON reports."""

    results = verify_paired_dataset(
        paired_path, wet_threshold_mm=wet_threshold_mm
    )
    csv_rows = results.pop("csv_rows")
    _write_csv(csv_rows, csv_path)
    results["outputs"] = {"daily_metrics_csv": str(csv_path), "summary_json": str(summary_path)}
    _write_json(results, summary_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-file",
        type=Path,
        default=Path("data/processed/paired/gfs_imd_20240901_20240907.nc"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("reports/smoke_test/06_daily_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/smoke_test/06_verification_summary.json"),
    )
    args = parser.parse_args()
    result = write_verification_reports(
        args.paired_file, args.csv_output, args.summary_output
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
