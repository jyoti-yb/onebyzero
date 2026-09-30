"""IMD 0.25 deg daily gridded rainfall (Pai et al. 2014) via imdlib — DECODING ONLY.

pip install imdlib
Files are yearwise binary .grd from IMD Pune. -999 = missing -> NaN. The land mask for
verification comes from here (cells with data).

This adapter does not decide which 24 h period an IMD date label represents.
`time` is left as the source date label (as decoded by imdlib), provenance is attached,
and `time_convention` is UNVERIFIED. The meaning of the label is applied later by
data.obs_time.interpret_obs_times() from the run config's `obs_time_convention`,
which must be set explicitly after checking the official product documentation.
Nothing here parses filenames, years or other products to guess it.
NOT exercised in the build sandbox (no network).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..data.align import to_grid
from ..data.obs_source import ObsSourceMeta, ProductRole, attach
from ..data.obs_time import decode_provenance
from ..grid import grid_role, make_coords
from .base import ObsAdapter, register

IMD_SOURCE = "IMD"
IMD_PRODUCT_NAME = "IMD daily gridded rainfall, 0.25 deg (Pai et al. 2014); imdlib 'rain' yearwise .grd"
IMD_UNITS = "mm"   # 24 h accumulation per date label; which 24 h is NOT decided here


def decode_imd(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray, source_filenames, grid_role_name: str) -> xr.Dataset:
    """Pure decode: missing-value handling, placement on the runtime grid, provenance.
    No temporal interpretation happens here."""
    da = da.where(da > -998)
    da = to_grid(da, lat, lon).astype("float32")
    obs = xr.Dataset({"rain": da.transpose("time", "lat", "lon")})
    return decode_provenance(obs, source=IMD_SOURCE, product_name=IMD_PRODUCT_NAME,
                             source_filenames=source_filenames, units=IMD_UNITS, grid_role=grid_role_name)


@register("obs", "imd")
class IMDAdapter(ObsAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lat, self.lon = make_coords(cfg.grid)
        self.dir = Path(cfg.raw_dir) / "imd"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _open_source(self, y0: int, y1: int) -> xr.DataArray:
        import imdlib
        if not any(self.dir.glob("rain/*.grd")):
            imdlib.get_data("rain", y0, y1, fn_format="yearwise", file_dir=str(self.dir))
        return imdlib.open_data("rain", y0, y1, "yearwise", str(self.dir)).get_xarray()

    def _source_file_for_year(self, year: int) -> str:
        # provenance only (imdlib yearwise layout); never used to infer a time convention
        p = self.dir / "rain" / f"{year}.grd"
        return str(p) if p.exists() else f"<imdlib yearwise file for {year} not found at {p}>"

    def load(self, dates: pd.DatetimeIndex) -> xr.Dataset:
        y0, y1 = int(dates.year.min()), int(dates.year.max())
        da = self._open_source(y0, y1)
        da = da.sel(time=da.time.isin(dates))   # selection by source label only
        files = [self._source_file_for_year(t.year) for t in pd.DatetimeIndex(da.time.values)]
        obs = decode_imd(da, self.lat, self.lon, files, grid_role(self.cfg.grid))
        return attach(obs, imd_grd_meta(sorted(set(files)), self.cfg.adapter_options.get("obs", {}).get("origin")))


def imd_grd_meta(files, origin=None) -> ObsSourceMeta:
    return ObsSourceMeta(
        source_name="IMD", product_name=IMD_PRODUCT_NAME, product_role=ProductRole.FINAL_TRUTH,
        source_filename="; ".join(files) or "unknown", source_url_or_origin=origin or "imdlib.get_data (IMD Pune)",
        observation_resolution="0.25 deg, daily (24 h accumulation per date label)",
        native_grid="regular lat-lon 0.25 deg, 6.5..38.5 N x 66.5..100.0 E (129 x 135), per imdlib layout",
        units=IMD_UNITS, time_convention="UNVERIFIED", is_proxy=False)


def land_mask_from_obs(obs: xr.Dataset, min_frac: float = 0.9) -> np.ndarray:
    return (obs["rain"].notnull().mean("time") >= min_frac).values
