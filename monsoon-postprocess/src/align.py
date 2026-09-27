"""Time and grid alignment for forecast-observation rainfall pairs."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def _rename_lat_lon(dataset: xr.Dataset, latitude_names: list[str], longitude_names: list[str]) -> xr.Dataset:
    rename: dict[str, str] = {}
    for name in latitude_names:
        if name in dataset.coords and name != "lat":
            rename[name] = "lat"
            break
    for name in longitude_names:
        if name in dataset.coords and name != "lon":
            rename[name] = "lon"
            break
    return dataset.rename(rename)


def align_forecast_observation(
    forecast_path: Path,
    observation_path: Path,
    output_path: Path,
    alignment_config: dict,
) -> Path:
    """Align daily forecast and observed precipitation by time and grid."""

    forecast = xr.open_dataset(forecast_path)
    observation = xr.open_dataset(observation_path)
    forecast = _rename_lat_lon(
        forecast,
        alignment_config["latitude_names"],
        alignment_config["longitude_names"],
    )
    observation = _rename_lat_lon(
        observation,
        alignment_config["latitude_names"],
        alignment_config["longitude_names"],
    )

    common_times = sorted(set(forecast.time.values).intersection(set(observation.time.values)))
    if not common_times:
        raise ValueError("Forecast and observation datasets have no common timestamps.")

    forecast = forecast.sel(time=common_times)
    observation = observation.sel(time=common_times)
    method = alignment_config.get("regrid_method", "nearest")
    forecast_on_obs_grid = forecast.interp(lat=observation.lat, lon=observation.lon, method=method)

    paired = xr.Dataset(
        {
            "forecast_precip": forecast_on_obs_grid["forecast_precip"],
            "observation_precip": observation["observation_precip"],
        }
    )
    paired.attrs["alignment_method"] = method

    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_netcdf(output_path)
    forecast.close()
    observation.close()
    return output_path
