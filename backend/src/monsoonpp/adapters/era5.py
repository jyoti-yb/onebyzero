"""ERA5 via the Copernicus CDS API — atmospheric state for regime LABELS.

pip install cdsapi ; configure ~/.cdsapirc with your CDS key.
Requested 6-hourly (03/09/15/21 UTC) and averaged over the IMD window.
NOT exercised in the build sandbox (no network).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from ..data.align import FORECAST_LABEL_RULE, daily_window_mean, to_grid
from ..grid import make_coords
from .base import AnalysisAdapter, register

PL = {"u_component_of_wind": "u850", "v_component_of_wind": "v850"}
SL = {"total_column_water_vapour": "tcwv",
      "convective_available_potential_energy": "cape",
      "mean_sea_level_pressure": "mslp"}
SHORT = {"u": "u850", "v": "v850", "tcwv": "tcwv", "cape": "cape", "msl": "mslp"}


@register("analysis", "era5")
class ERA5Adapter(AnalysisAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lat, self.lon = make_coords(cfg.grid)
        self.dir = Path(cfg.raw_dir) / "era5"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.label_rule = FORECAST_LABEL_RULE   # analysis label = window-start date; never an IMD assumption

    def _area(self):
        g = self.cfg.grid
        return [g.lat_max + 1, g.lon_min - 1, g.lat_min - 1, g.lon_max + 1]

    def _fetch(self, year: int, months: list[str]) -> Path:
        import cdsapi
        c = cdsapi.Client()
        base = dict(product_type="reanalysis", year=str(year), month=months,
                    day=[f"{d:02d}" for d in range(1, 32)], time=["03:00", "09:00", "15:00", "21:00"],
                    area=self._area(), grid=[self.cfg.grid.res] * 2, format="netcdf")
        pl, sl = self.dir / f"pl_{year}.nc", self.dir / f"sl_{year}.nc"
        if not pl.exists():
            c.retrieve("reanalysis-era5-pressure-levels", {**base, "variable": list(PL), "pressure_level": "850"}, str(pl))
        if not sl.exists():
            c.retrieve("reanalysis-era5-single-levels", {**base, "variable": list(SL)}, str(sl))
        return pl, sl

    def load(self, dates: pd.DatetimeIndex) -> xr.Dataset:
        parts = []
        for y in sorted(set(dates.year)):
            months = sorted({f"{m:02d}" for m in dates[dates.year == y].month} |
                            {f"{m:02d}" for m in (dates[dates.year == y] + pd.Timedelta(days=1)).month})
            pl, sl = self._fetch(y, months)
            ds = xr.merge([xr.open_dataset(pl), xr.open_dataset(sl)], compat="override")
            if "valid_time" in ds.dims:
                ds = ds.rename(valid_time="time")
            ds = ds.squeeze(drop=True)
            ds = ds.rename({k: v for k, v in SHORT.items() if k in ds})
            ds = ds[list(SHORT.values())]
            ds["mslp"] = ds["mslp"] / 100.0
            parts.append(daily_window_mean(ds, self.label_rule))
        ds = xr.concat(parts, "time")
        ds = ds.sel(time=ds.time.isin(dates))
        return to_grid(ds, self.lat, self.lon).transpose("time", "lat", "lon").astype("float32")
