"""Load local IMD precipitation samples into a normalized NetCDF product."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def _select_precip_variable(dataset: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in dataset.data_vars:
            return name
    available = ", ".join(dataset.data_vars)
    raise KeyError(f"No IMD precipitation variable found. Available variables: {available}")


def load_imd_precip(
    input_files: list[Path],
    output_path: Path,
    variable_candidates: list[str],
    domain: dict[str, float],
) -> Path:
    """Normalize local IMD observations to `observation_precip(time, lat, lon)`.

    NetCDF files are handled directly. CSV files are expected to contain
    `time`, `lat`, `lon`, and one configured precipitation column.
    """

    if not input_files:
        raise ValueError("No IMD input files were provided.")

    suffixes = {path.suffix.lower() for path in input_files}
    if suffixes <= {".nc", ".nc4"}:
        dataset = xr.open_mfdataset([str(path) for path in input_files], combine="by_coords")
        variable = _select_precip_variable(dataset, variable_candidates)
        precip = dataset[variable].rename("observation_precip")
        normalized = precip.to_dataset()
    elif suffixes <= {".csv"}:
        frame = pd.concat((pd.read_csv(path) for path in input_files), ignore_index=True)
        variable = next((name for name in variable_candidates if name in frame.columns), None)
        if variable is None:
            raise KeyError("No configured precipitation column found in IMD CSV files.")
        required = {"time", "lat", "lon", variable}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"IMD CSV files are missing columns: {sorted(missing)}")
        frame["time"] = pd.to_datetime(frame["time"])
        normalized = (
            frame.set_index(["time", "lat", "lon"])[variable]
            .to_xarray()
            .rename("observation_precip")
            .to_dataset()
        )
    else:
        raise ValueError(
            "Unsupported IMD smoke-test format. Provide local NetCDF or CSV files for now."
        )

    precip = normalized["observation_precip"]
    if "lat" not in precip.coords or "lon" not in precip.coords:
        raise ValueError("IMD precipitation must expose lat and lon coordinates.")
    normalized = normalized.where(
        (normalized.lat >= domain["south"])
        & (normalized.lat <= domain["north"])
        & (normalized.lon >= domain["west"])
        & (normalized.lon <= domain["east"]),
        drop=True,
    )
    normalized["observation_precip"].attrs.setdefault("units", "mm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_netcdf(output_path)
    return output_path
