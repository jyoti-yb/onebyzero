"""NCUM (NCMRWF Unified Model) and any other NWP delivered as NetCDF files.

`NetCDFForecastAdapter` is fully generic: point it at a directory of per-init files
and give it a variable-name map. `NCUMAdapter` is the same with NCUM defaults.
The NCUM names/units below are PLACEHOLDERS — confirm them against the files
NCMRWF actually provides and update `NCUM_VARMAP` (the ML side does not change).

Expected layout (configurable via adapter_options.filename_pattern):
    {raw_dir}/{source}/{init:%Y%m%d}.nc   with a `time` (valid-time) dimension,
    precip given as accumulation since init (mm) and state vars instantaneous.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..data.align import FORECAST_LABEL_RULE, forecast_window, to_grid
from ..grid import make_coords
from .base import ForecastAdapter, register

# canonical -> (source name, scale, offset)
CANONICAL_VARMAP = {
    "tp_accum": ("tp_accum", 1.0, 0.0), "u850": ("u850", 1.0, 0.0), "v850": ("v850", 1.0, 0.0),
    "tcwv": ("tcwv", 1.0, 0.0), "cape": ("cape", 1.0, 0.0), "mslp": ("mslp", 1.0, 0.0),
}
NCUM_VARMAP = {   # PLACEHOLDER — verify with NCMRWF file headers
    "tp_accum": ("precipitation_amount", 1.0, 0.0),
    "u850": ("x_wind_850", 1.0, 0.0),
    "v850": ("y_wind_850", 1.0, 0.0),
    "tcwv": ("atmosphere_mass_content_of_water_vapor", 1.0, 0.0),
    "cape": ("atmosphere_convective_available_potential_energy", 1.0, 0.0),
    "mslp": ("air_pressure_at_sea_level", 0.01, 0.0),
}


@register("forecast", "netcdf")
class NetCDFForecastAdapter(ForecastAdapter):
    source_name = "netcdf"
    default_varmap = CANONICAL_VARMAP

    def __init__(self, cfg):
        super().__init__(cfg)
        opts = cfg.adapter_options
        self.lat, self.lon = make_coords(cfg.grid)
        self.dir = Path(opts.get("forecast_dir", Path(cfg.raw_dir) / self.source_name))
        self.pattern = opts.get("filename_pattern", "{init:%Y%m%d}.nc")
        self.varmap = {**self.default_varmap, **opts.get("varmap", {})}
        self.label_rule = FORECAST_LABEL_RULE   # forecast label = window-start date; never an IMD assumption

    def _open(self, init: pd.Timestamp) -> xr.Dataset:
        return xr.open_dataset(self.dir / self.pattern.format(init=init))

    def _get(self, ds, key, t):
        name, scale, off = self.varmap[key]
        da = ds[name].sel(time=t, method="nearest", tolerance=pd.Timedelta(hours=1))
        return to_grid(da, self.lat, self.lon).values * scale + off

    def load(self, dates, leads):
        keys = ["tp", "u850", "v850", "tcwv", "cape", "mslp"]
        arrs = {k: np.full((len(leads), len(dates), len(self.lat), len(self.lon)), np.nan, "float32") for k in keys}
        for li, L in enumerate(leads):
            for ti, d in enumerate(dates):
                w = forecast_window(d, L, self.label_rule)
                try:
                    ds = self._open(w.init)
                except FileNotFoundError:
                    continue
                t0, t1 = w.init + pd.Timedelta(hours=w.f_start), w.init + pd.Timedelta(hours=w.f_end)
                arrs["tp"][li, ti] = np.clip(self._get(ds, "tp_accum", t1) - self._get(ds, "tp_accum", t0), 0, None)
                for k in keys[1:]:
                    arrs[k][li, ti] = np.mean([self._get(ds, k, w.init + pd.Timedelta(hours=h)) for h in w.mean_hours], 0)
        out = xr.Dataset({k: (("lead", "time", "lat", "lon"), v) for k, v in arrs.items()},
                         coords={"lead": leads, "time": dates, "lat": self.lat, "lon": self.lon})
        out.attrs.update(source=self.source_name, model_version=self.model_version)
        return out


@register("forecast", "ncum")
class NCUMAdapter(NetCDFForecastAdapter):
    source_name = "ncum"
    default_varmap = NCUM_VARMAP
