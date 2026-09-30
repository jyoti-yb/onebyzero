"""Forecast-side time windows and grid alignment.

FORECAST / ANALYSIS LABEL RULE (fixed, independent of any observation product):
  forecast and analysis `time` label D  <=>  window 03 UTC D -> 03 UTC D+1
  (FORECAST_LABEL_RULE = "starting"). Lead L uses the 00 UTC run initialised L-1 days
  before D, i.e. lead 1 = f003..f027.

What an OBSERVATION date label means is a separate, explicitly configured and
verified property (data/obs_time.py: ObsTimeConvention). Nothing in this module
states or assumes the IMD convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class Window:
    init: pd.Timestamp       # 00 UTC init time
    f_start: int             # forecast hour at window start
    f_end: int               # forecast hour at window end

    @property
    def mean_hours(self) -> list[int]:
        """Forecast hours used for daily-mean state variables (6-hourly)."""
        return [self.f_start + 3, self.f_start + 9, self.f_start + 15, self.f_start + 21]


FORECAST_LABEL_RULE = "starting"


def forecast_label_window(label) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(valid_start, valid_end) of a forecast/analysis `time` label under FORECAST_LABEL_RULE."""
    d = pd.Timestamp(label).normalize()
    return d + pd.Timedelta(hours=3), d + pd.Timedelta(hours=27)


def forecast_window(valid_date: pd.Timestamp, lead: int, label_rule: str = FORECAST_LABEL_RULE) -> Window:
    """Forecast window for a label. `label_rule` is a forecast-side labelling rule
    ("starting": label = window-start date; "ending": label = window-end date);
    adapters use FORECAST_LABEL_RULE. It says nothing about observation products."""
    day0 = pd.Timestamp(valid_date).normalize()
    if label_rule == "ending":
        day0 = day0 - pd.Timedelta(days=1)
    elif label_rule != "starting":
        raise ValueError(label_rule)
    init = day0 - pd.Timedelta(days=lead - 1)
    f_start = 3 + 24 * (lead - 1)
    return Window(init=init, f_start=f_start, f_end=f_start + 24)


def gfs_apcp_primary(w: Window) -> list[tuple[int, int, int, int]]:
    """PRIMARY daily construction: difference of two running totals from init.

    00Z run, D+1: APCP(0-27) - APCP(0-3) = 03 UTC D -> 03 UTC D+1.
    Returns (fxx_file, acc_start_h, acc_end_h, sign). `fxx_file` only says which file
    to open; the accumulation interval is always verified from GRIB metadata.
    """
    s, e = w.f_start, w.f_end
    return [(e, 0, e, +1), (s, 0, s, -1)]


def gfs_apcp_qa_pieces(w: Window) -> list[tuple[int, int, int, int]]:
    """INDEPENDENT QA construction from 6-hourly buckets (never the primary).

    D+1: (APCP(0-6) - APCP(0-3)) + APCP(6-12) + APCP(12-18) + APCP(18-24) + APCP(24-27)
    General (window start s = 3 mod 6):
      [B(s-3,s+3) - B(s-3,s)] + B(s+3,s+9) + B(s+9,s+15) + B(s+15,s+21) + B(s+21,s+24)
    Returns (fxx_file, acc_start_h, acc_end_h, sign).
    """
    s = w.f_start
    if s % 6 != 3:
        raise ValueError(f"window must start at a 3-mod-6 forecast hour, got f{s:03d}")
    return [
        (s + 3, s - 3, s + 3, +1),
        (s, s - 3, s, -1),
        (s + 9, s + 3, s + 9, +1),
        (s + 15, s + 9, s + 15, +1),
        (s + 21, s + 15, s + 21, +1),
        (s + 24, s + 21, s + 24, +1),
    ]


def gfs_apcp_pieces(w: Window) -> list[tuple[int, str, int]]:
    """QA pieces as (fxx, "start-end", sign) strings (kept for compatibility)."""
    return [(f, f"{a}-{b}", sg) for f, a, b, sg in gfs_apcp_qa_pieces(w)]


def standardise_latlon(da: xr.DataArray | xr.Dataset):
    """Rename to lat/lon, convert 0..360 lon to -180..180, make lat ascending."""
    ren = {}
    for a, b in (("latitude", "lat"), ("longitude", "lon"), ("LATITUDE", "lat"), ("LONGITUDE", "lon")):
        if a in da.dims or a in da.coords:
            ren[a] = b
    da = da.rename(ren)
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
    return da.sortby("lat").sortby("lon")


def to_grid(da, lat: np.ndarray, lon: np.ndarray, tol: float = 1e-3):
    """Put a field on the target grid. Exact/nearest selection when the target
    is a subset of the source grid (IMD 0.25 <-> GFS 0.25), else bilinear interp.
    For coarse->fine or very different grids use conservative remapping (xESMF)."""
    da = standardise_latlon(da)
    src_lat, src_lon = da.lat.values, da.lon.values
    def _subset(src, tgt):
        idx = np.abs(src[:, None] - tgt[None, :]).argmin(0)
        return np.all(np.abs(src[idx] - tgt) < tol)
    if _subset(src_lat, lat) and _subset(src_lon, lon):
        return da.sel(lat=lat, lon=lon, method="nearest").assign_coords(lat=lat, lon=lon)
    return da.interp(lat=lat, lon=lon, method="linear")


def daily_window_mean(ds: xr.Dataset, label_rule: str = FORECAST_LABEL_RULE) -> xr.Dataset:
    """Hourly/sub-daily analysis -> daily mean over 03Z->03Z windows, labelled per the
    forecast-side label rule (not per any observation convention)."""
    shift = pd.Timedelta(hours=3)
    t = pd.DatetimeIndex(ds.time.values) - shift
    lab = t.normalize()
    if label_rule == "ending":
        lab = lab + pd.Timedelta(days=1)
    return ds.assign_coords(day=("time", lab)).groupby("day").mean("time").rename(day="time")
