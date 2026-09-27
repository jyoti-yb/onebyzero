"""Build daily precipitation products from forecast accumulation fields."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def _select_precip_variable(dataset: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in dataset.data_vars:
            return name
        case_insensitive_match = next(
            (available for available in dataset.data_vars if available.lower() == name.lower()),
            None,
        )
        if case_insensitive_match is not None:
            return case_insensitive_match
    available = ", ".join(dataset.data_vars)
    raise KeyError(f"No precipitation variable found. Available variables: {available}")


def build_precip_accumulation(
    forecast_files: list[Path],
    output_path: Path,
    variable_candidates: list[str],
    domain: dict[str, float],
    rainfall_window_start_utc: int,
    rainfall_window_hours: int,
) -> Path:
    """Create daily forecast precipitation totals from local GRIB files.

    This function assumes the smoke-test files are already present locally.
    """

    if not forecast_files:
        raise ValueError("No forecast files were provided.")

    dataset = xr.open_mfdataset(
        [str(path) for path in forecast_files],
        engine="cfgrib",
        combine="by_coords",
        backend_kwargs={"indexpath": ""},
    )
    variable = _select_precip_variable(dataset, variable_candidates)
    precip = dataset[variable]

    latitude_name = next((name for name in ("lat", "latitude") if name in precip.coords), None)
    longitude_name = next((name for name in ("lon", "longitude") if name in precip.coords), None)
    if latitude_name is None or longitude_name is None:
        raise ValueError("Forecast precipitation must expose latitude and longitude coordinates.")
    precip = precip.where(
        (precip[latitude_name] >= domain["south"])
        & (precip[latitude_name] <= domain["north"])
        & (precip[longitude_name] >= domain["west"])
        & (precip[longitude_name] <= domain["east"]),
        drop=True,
    )

    if "step" in precip.dims:
        valid_time = dataset.get("valid_time")
        if valid_time is not None:
            precip = precip.assign_coords(time=valid_time)

    if "time" not in precip.coords:
        raise ValueError("Forecast precipitation does not expose a usable time coordinate.")

    daily = precip.resample(
        time=f"{rainfall_window_hours}h",
        origin="start_day",
        offset=f"{rainfall_window_start_utc}h",
    ).sum(keep_attrs=True)
    daily.name = "forecast_precip"
    daily.attrs.update({"units": "mm", "source_variable": variable})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_dataset().to_netcdf(output_path)
    dataset.close()
    return output_path
