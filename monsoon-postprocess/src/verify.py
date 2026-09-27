"""Basic deterministic verification metrics for rainfall forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def verify_paired_dataset(paired_path: Path, metrics: list[str]) -> dict[str, Any]:
    """Compute simple verification metrics from paired precipitation fields."""

    dataset = xr.open_dataset(paired_path)
    forecast = dataset["forecast_precip"]
    observed = dataset["observation_precip"]
    error = forecast - observed

    results: dict[str, Any] = {"path": str(paired_path), "metrics": {}}
    if "bias" in metrics:
        results["metrics"]["bias"] = float(error.mean(skipna=True))
    if "mae" in metrics:
        results["metrics"]["mae"] = float(abs(error).mean(skipna=True))
    if "rmse" in metrics:
        results["metrics"]["rmse"] = float(np.sqrt((error**2).mean(skipna=True)))
    if "correlation" in metrics:
        results["metrics"]["correlation"] = float(xr.corr(forecast, observed).mean(skipna=True))

    dataset.close()
    return results
