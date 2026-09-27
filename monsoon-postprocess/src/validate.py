"""Quality checks for paired rainfall forecast-observation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def _missing_fraction(values: xr.DataArray) -> float:
    return float((values.isnull().sum() / values.size).item())


def validate_paired_dataset(paired_path: Path, quality_config: dict[str, Any]) -> dict[str, Any]:
    """Run smoke-test quality checks and return a report dictionary."""

    dataset = xr.open_dataset(paired_path)
    checks: dict[str, Any] = {"path": str(paired_path), "passed": True, "checks": {}}
    min_precip = quality_config["rainfall_min_mm"]
    max_precip = quality_config["rainfall_max_reasonable_mm"]
    min_valid_fraction = quality_config["min_valid_grid_fraction"]

    for name in ("forecast_precip", "observation_precip"):
        data = dataset[name]
        missing_fraction = _missing_fraction(data)
        below_min = bool((data < min_precip).any().item())
        above_max = bool((data > max_precip).any().item())
        finite_or_missing = bool(np.isfinite(data.fillna(0)).all().item())
        valid_fraction = 1.0 - missing_fraction
        passed = (
            valid_fraction >= min_valid_fraction
            and not below_min
            and not above_max
            and finite_or_missing
        )
        checks["checks"][name] = {
            "passed": passed,
            "missing_fraction": missing_fraction,
            "valid_grid_fraction": valid_fraction,
            "has_negative_precip": below_min,
            "has_extreme_precip": above_max,
            "finite_or_missing": finite_or_missing,
        }
        checks["passed"] = checks["passed"] and passed

    same_grid = (
        "lat" in dataset.coords
        and "lon" in dataset.coords
        and dataset["forecast_precip"].sizes.get("lat") == dataset["observation_precip"].sizes.get("lat")
        and dataset["forecast_precip"].sizes.get("lon") == dataset["observation_precip"].sizes.get("lon")
    )
    time_count = int(dataset.sizes.get("time", 0))
    checks["checks"]["alignment"] = {
        "passed": same_grid and time_count > 0,
        "same_grid_shape": same_grid,
        "paired_time_steps": time_count,
    }
    checks["passed"] = checks["passed"] and checks["checks"]["alignment"]["passed"]

    dataset.close()
    return checks
