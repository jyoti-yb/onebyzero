"""IMERG (SMOKE-TEST PROXY) -> exact 03Z -> 03Z daily rainfall, then 0.1 -> 0.25 deg.

Per window W -> W+24h (W must be 03:00 UTC):
  * exactly 48 granules whose verified source intervals are W + k*30min .. W + (k+1)*30min,
    k = 0..47. Any gap -> the window is REJECTED (recorded, not filled). Overlaps and
    duplicates are already fatal in IMERGAdapter.
  * depth (mm) = sum_k rate_k (mm/hr) * 0.5 h. A cell missing in any granule is missing
    in the daily total (no partial sums).
Then first-order conservative remapping to the canonical grid (data/regrid.py).
The result stays SMOKE_TEST_PROXY and keeps IMERG provenance; it is never IMD.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import xarray as xr

from .obs_source import attach, read
from .regrid import conservative_regrid

HALF_HOUR = pd.Timedelta(minutes=30)
N_GRANULES = 48


class IMERGDailyError(RuntimeError):
    pass


def imerg_daily_03z(native: xr.Dataset, window_starts) -> tuple[xr.Dataset, list[dict]]:
    meta = read(native)
    st = pd.DatetimeIndex(native.source_interval_start.values)
    en = pd.DatetimeIndex(native.source_interval_end.values)
    if not ((en - st) == HALF_HOUR).all():
        raise IMERGDailyError("non-30-minute granule in input")
    days, rejected, kept = [], [], []
    for W in pd.DatetimeIndex(window_starts):
        if (W.hour, W.minute, W.second) != (3, 0, 0):
            raise IMERGDailyError(f"window start {W} is not 03:00 UTC")
        expected = pd.date_range(W, periods=N_GRANULES, freq="30min")
        sel = np.nonzero((st >= W) & (st < W + pd.Timedelta(hours=24)))[0]
        have = st[sel]
        missing = expected.difference(have)
        off_grid = have.difference(expected)
        if len(off_grid):
            raise IMERGDailyError(f"granules not aligned to the 30-min lattice from {W}: {list(off_grid[:3])}")
        if len(missing):
            rejected.append({"valid_start": str(W), "valid_end": str(W + pd.Timedelta(hours=24)),
                             "granules_found": int(len(have)), "granules_missing": int(len(missing)),
                             "first_missing": str(missing[0]), "reason": "gap: fewer than 48 verified granules"})
            continue
        sub = native.precipitation_rate.isel(time=sel)
        depth = (sub * 0.5).sum("time", skipna=False)          # any NaN granule -> NaN day
        days.append(depth)
        kept.append((W, len(sel)))
    if not days:
        raise IMERGDailyError(f"no complete 03Z->03Z windows ({len(rejected)} rejected)")
    starts = pd.DatetimeIndex([k[0] for k in kept])
    daily = xr.concat(days, "time").assign_coords(
        time=starts, valid_start=("time", starts.values),
        valid_end=("time", (starts + pd.Timedelta(hours=24)).values),
        n_granules=("time", np.array([k[1] for k in kept])))
    out = xr.Dataset({"rain": daily.astype("float32")})
    out["rain"].attrs["units"] = "mm"
    out.attrs.update({k: v for k, v in native.attrs.items() if not k.startswith("obs_meta_")})
    m = replace(meta, product_name=meta.product_name + "; 03Z-03Z daily depth from 48 verified 30-min granules",
                observation_resolution="0.1 deg native, 24 h 03Z->03Z (constructed)", units="mm")
    out = attach(out, m)
    out.attrs["time_construction"] = "sum of 48 x (rate mm/hr * 0.5 h), intervals from /Grid/time_bnds"
    return out, rejected


def imerg_daily_on_grid(native: xr.Dataset, window_starts, lat, lon, min_coverage: float = 1.0):
    daily, rejected = imerg_daily_03z(native, window_starts)
    rg = conservative_regrid(daily.rain, lat, lon, min_coverage=min_coverage)
    out = xr.Dataset({"rain": rg}).assign_coords(valid_start=daily.valid_start, valid_end=daily.valid_end,
                                                 n_granules=daily.n_granules)
    out["rain"].attrs["units"] = "mm"
    out.attrs.update({k: v for k, v in daily.attrs.items() if not k.startswith("obs_meta_")})
    res = float(np.median(np.diff(lat)))
    m = replace(read(daily), observation_resolution=f"{res} deg (first-order conservative from 0.1 deg), 24 h 03Z->03Z")
    return attach(out, m), rejected
