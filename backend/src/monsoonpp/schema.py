"""Canonical data schema. Every adapter must emit these names/units/dims.

The rest of the system never sees source-specific names (APCP, tp, RAINFALL ...).
This is the contract that lets GFS (development) be swapped for NCUM (operational).

Dimensions
----------
forecast : (lead, time, lat, lon)   time = forecast label D: window 03 UTC D -> 03 UTC D+1
obs      : (time, lat, lon)         time = SOURCE date label; its 24 h period is given only by
                                    the configured, verified obs_time_convention (data/obs_time.py)
analysis : (time, lat, lon)
static   : (lat, lon)
"""
from __future__ import annotations

import numpy as np
import xarray as xr

FORECAST_VARS = {
    "tp": "mm/day — 24h accumulated precip over the IMD verification window",
    "u850": "m/s — zonal wind at 850 hPa (daily mean)",
    "v850": "m/s — meridional wind at 850 hPa (daily mean)",
    "tcwv": "kg/m2 (= mm) — total column water vapour / precipitable water",
    "cape": "J/kg — convective available potential energy (daily max or mean)",
    "mslp": "hPa — mean sea level pressure (daily mean)",
}
FORECAST_OPTIONAL_VARS = {
    **{f"{component}{level}": f"validated GFS {component}-wind at {level} hPa"
       for level in (700, 500, 200) for component in ("u", "v")},
    **{f"rh{level}": f"validated GFS relative humidity at {level} hPa" for level in (850, 700, 500, 200)},
    **{f"hgt{level}": f"validated GFS geopotential height at {level} hPa" for level in (850, 700, 500, 200)},
    "wspd850": "derived 850 hPa wind speed",
    "vo850": "derived 850 hPa relative vorticity",
    "moisture_transport": "derived PWAT x 850 hPa wind-speed proxy",
    "pwat_anom": "derived PWAT anomaly from large-scale spatial background",
    "mslp_anom": "derived MSLP anomaly from large-scale spatial background",
}
ANALYSIS_VARS = {k: v for k, v in FORECAST_VARS.items() if k != "tp"}
OBS_VARS = {"rain": "mm/day — observed gridded rainfall (IMD)"}
STATIC_VARS = {
    "slope": "degrees - terrain slope",
    "aspect": "degrees clockwise from north - downslope aspect",
    "elev": "m — terrain height",
    "land": "bool — verification mask (cells with observations)",
    "coast_dist": "km — distance from the coast (0 over sea)",
}


class SchemaError(ValueError):
    pass


def _check(ds: xr.Dataset, required: dict, dims: tuple, kind: str):
    missing = [v for v in required if v not in ds]
    if missing:
        raise SchemaError(f"{kind}: missing variables {missing}")
    for v in required:
        if tuple(ds[v].dims) != dims:
            raise SchemaError(f"{kind}.{v}: dims {ds[v].dims} != {dims}")
    for c in ("lat", "lon"):
        vals = ds[c].values
        if not np.all(np.diff(vals) > 0):
            raise SchemaError(f"{kind}: coordinate {c} must be strictly increasing")


def validate_forecast(ds: xr.Dataset) -> xr.Dataset:
    _check(ds, FORECAST_VARS, ("lead", "time", "lat", "lon"), "forecast")
    if float(ds["tp"].min()) < 0:
        raise SchemaError("forecast.tp has negative values")
    return ds


def validate_obs(ds: xr.Dataset) -> xr.Dataset:
    _check(ds, OBS_VARS, ("time", "lat", "lon"), "obs")
    return ds


def validate_analysis(ds: xr.Dataset) -> xr.Dataset:
    _check(ds, ANALYSIS_VARS, ("time", "lat", "lon"), "analysis")
    return ds


def validate_static(ds: xr.Dataset) -> xr.Dataset:
    _check(ds, STATIC_VARS, ("lat", "lon"), "static")
    return ds
