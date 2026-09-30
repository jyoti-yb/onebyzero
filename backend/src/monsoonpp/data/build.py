"""Build the aligned, schema-validated training/verification cube.

One netCDF with:
  fc_<var>  (lead, time, lat, lon)   forecast (model input)
  obs_rain  (time, lat, lon)         truth
  an_<var>  (time, lat, lon)         analysis (regime labels only)
  elev, land, coast_dist (lat, lon)  static
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import xarray as xr

from .. import schema
from ..adapters import get_adapter
from ..config import Config
from ..grid import build_static, check_imd_compatible, grid_role
from .align import forecast_label_window
from .pairing import align_to_windows, forecast_windows, obs_labels_for_windows, obs_windows
from .obs_source import ObsSourceMeta, ProductRole, attach, require_daily_truth_capable
from .obs_time import (ObsTimeConvention, ObsTimeConventionError, decode_provenance, interpret_obs_times,
                       parse_convention, reject_legacy_keys, require_verified)
from .static_geo import load_real_static, require_real_static

log = logging.getLogger(__name__)


def season_dates(cfg: Config) -> pd.DatetimeIndex:
    parts = [pd.date_range(f"{y}-{cfg.season_start}", f"{y}-{cfg.season_end}", freq="D") for y in cfg.years]
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def build_dataset(cfg: Config, save: bool = True) -> xr.Dataset:
    # Pairing forecasts with observations requires a VERIFIED observation time convention.
    reject_legacy_keys(cfg.adapter_options)
    conv = require_verified(cfg.obs_time_convention, "build forecast/observation pairs", cfg.obs_source)
    # IMD rainfall must sit on its own 0.25 deg lattice (canonical grid or aligned crop).
    role = check_imd_compatible(cfg.grid) if cfg.obs_source == "imd" else grid_role(cfg.grid)
    log.info("runtime grid role: %s (%s)", role, cfg.grid)
    dates = season_dates(cfg)
    fc = schema.validate_forecast(get_adapter("forecast", cfg.forecast_source, cfg).load(dates, cfg.leads))
    f_start, f_end = forecast_windows(fc)
    # request exactly the observation labels whose (verified) windows equal the forecast windows
    obs = schema.validate_obs(get_adapter("obs", cfg.obs_source, cfg).load(obs_labels_for_windows(f_start, f_end, conv)))
    if "source_date_label" not in obs.coords:   # adapters without their own provenance (synthetic)
        obs = decode_provenance(obs, source=cfg.obs_source, product_name=f"{cfg.obs_source} observations",
                                source_filenames="<generated in memory>", units="mm", grid_role=role)
    if cfg.obs_source == "synthetic":
        obs = attach(obs, ObsSourceMeta(
            source_name="synthetic", product_name="synthetic monsoon world (CI fixture)",
            product_role=ProductRole.SYNTHETIC_TEST, source_filename="<generated in memory>",
            source_url_or_origin="monsoonpp.adapters.synthetic", observation_resolution=f"{cfg.grid.res} deg, daily",
            native_grid=f"runtime grid {cfg.grid}", units="mm", time_convention="UNVERIFIED", is_proxy=True))
    obs_meta = require_daily_truth_capable(obs, "build forecast/observation pairs")
    obs = interpret_obs_times(obs, conv)        # the ONLY place a label gets a 24 h window
    obs_meta = replace(obs_meta, time_convention=conv.value)   # record the convention actually applied
    ana = schema.validate_analysis(get_adapter("analysis", cfg.analysis_source, cfg).load(dates))

    # PAIR BY EXACT WINDOW (valid_start, valid_end), never by equal labels
    o_start, o_end = obs_windows(obs)
    obs, obs_ok = align_to_windows(obs, o_start, o_end, f_start, f_end, fc.time.values)
    a_win = [forecast_label_window(t) for t in pd.DatetimeIndex(ana.time.values)]
    ana, ana_ok = align_to_windows(ana, [w[0] for w in a_win], [w[1] for w in a_win], f_start, f_end, fc.time.values)
    # drop windows without an observation / analysis match, or where any lead is missing (failed downloads)
    ok = obs_ok & ana_ok & fc["tp"].notnull().any(("lat", "lon")).all("lead").values
    common = fc.time.values[ok]
    log.info("paired windows: %d / %d forecast windows", len(common), len(f_start))
    fc, obs, ana = fc.sel(time=common), obs.sel(time=common), ana.sel(time=common)

    if cfg.obs_source == "synthetic":
        static = build_static(cfg.grid)
    else:
        static = require_real_static(load_real_static(cfg))
    schema.validate_static(static)

    ds = xr.Dataset(coords={"lead": fc.lead.values, "time": common, "lat": fc.lat.values, "lon": fc.lon.values})
    for v in schema.FORECAST_VARS:
        ds[f"fc_{v}"] = fc[v]
    for v in schema.FORECAST_OPTIONAL_VARS:
        if v in fc:
            ds[f"fc_{v}"] = fc[v]
    ds["obs_rain"] = obs["rain"].reset_coords(drop=True)
    ds["obs_rain"].attrs = {**obs["rain"].attrs, **{k: obs.attrs[k] for k in
                            ("source", "product_name", "time_convention", "units", "grid_role")}}
    ds["obs_rain"].attrs.update(obs_meta.to_attrs())
    for c in ("source_filename", "source_date_label", "valid_start", "valid_end"):
        ds.coords[f"obs_{c}"] = ("time", obs[c].values)   # original metadata kept apart from normalized times
    for v in schema.ANALYSIS_VARS:
        ds[f"an_{v}"] = ana[v]
    for v in schema.STATIC_VARS:
        ds[v] = static[v]
    static_attrs = {k if k.startswith("static_") else f"static_{k}": v for k, v in static.attrs.items()}
    ds.attrs.update(forecast_source=cfg.forecast_source, obs_source=cfg.obs_source,
                    analysis_source=cfg.analysis_source, nwp_model_version=cfg.nwp_model_version,
                    grid_role=role, grid_res=cfg.grid.res, obs_time_convention=conv.value,
                    obs_product_role=obs_meta.product_role.value, obs_is_proxy=str(obs_meta.is_proxy).lower(),
                    grid_bounds=f"{cfg.grid.lat_min}-{cfg.grid.lat_max}N {cfg.grid.lon_min}-{cfg.grid.lon_max}E",
                    **static_attrs)
    if save:
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 3} for v in ds.data_vars if ds[v].dtype != bool}
        ds.assign(land=ds.land.astype("int8")).to_netcdf(cfg.dataset_path, encoding=enc)
        log.info("wrote %s", cfg.dataset_path)
    return ds


def load_dataset(cfg: Config, workflow: str = "use paired forecast/observation dataset") -> xr.Dataset:
    ds = xr.open_dataset(cfg.dataset_path).load()
    ds["land"] = ds.land.astype(bool)
    check_dataset_convention(ds, cfg, workflow)
    return ds


def check_dataset_convention(ds: xr.Dataset, cfg: Config | None, workflow: str) -> ObsTimeConvention:
    """A paired dataset must carry a verified convention matching the run config."""
    if "obs_rain" not in ds:
        return ObsTimeConvention.UNVERIFIED     # forecast-only data: nothing paired
    stored = ds.attrs.get("obs_time_convention")
    if stored is None:
        raise ObsTimeConventionError(f"BLOCKED: '{workflow}': dataset has no obs_time_convention record "
                                     f"(built before conventions were tracked). Rebuild it.")
    conv = require_verified(stored, workflow, ds.attrs.get("obs_source", ""))
    if cfg is not None:
        want = require_verified(cfg.obs_time_convention, workflow, cfg.obs_source)
        if want is not conv:
            raise ObsTimeConventionError(f"BLOCKED: '{workflow}': dataset built with {conv.value}, "
                                         f"config says {want.value}. Rebuild the dataset.")
    return conv
