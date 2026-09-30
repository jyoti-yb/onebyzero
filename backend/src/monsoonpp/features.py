"""Feature engineering: forecast-only predictors + static geography + forecast-side regime.

Rule: nothing derived from observations or analysis may enter the predictor set
(those are only targets / labels). Neighbourhood features let the model deal with
displacement error, not just intensity error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter, uniform_filter

from .grid import region_of
from .regimes.engine import RegimeEngine

STATE_VARS = ["u850", "v850", "tcwv", "cape", "mslp"]

BASE_FEATURES = [
    "tp", "tp_log", "tp_n3_mean", "tp_n5_mean", "tp_n5_max", "tp_n9_mean", "tp_n9_max", "tp_grad",
    "u850", "v850", "wspd", "tcwv", "cape", "moist_flux", "upslope", "onshore",
    "elev", "coast_dist", "lat", "lon", "doy_sin", "doy_cos", "lead",
]
REGIME_FEATURES = [
    "monsoon_z", "cmz_tp", "dist_system_km", "synoptic", "oro_score", "coast_score", "conv_score",
    "mslp_spatial_residual", "vo850", "fc_primary",
]
ALL_FEATURES = BASE_FEATURES + REGIME_FEATURES
META = ["lead", "time", "iy", "ix", "lat", "lon", "year", "region"]


def _nb(a, k, fn):
    size = (1, k, k)
    return uniform_filter(a, size=size, mode="nearest") if fn == "mean" else maximum_filter(a, size=size, mode="nearest")


def _grad_mag(a, res, lat):
    gy, gx = np.gradient(a, axis=(1, 2))
    return np.hypot(gx / np.cos(np.deg2rad(lat))[None, :, None], gy) / res


class RegimeBundle:
    """Fitted regime engines: one for labels, one per lead for the forecast side."""

    def __init__(self):
        self.label_engine: RegimeEngine | None = None
        self.fc_engines: dict[int, RegimeEngine] = {}

    def fit(self, ds: xr.Dataset, train_times) -> "RegimeBundle":
        synthetic_fixture = str(ds.attrs.get("forecast_source", "")) == "synthetic"
        self.label_engine = RegimeEngine().fit(ds.obs_rain.sel(time=train_times), ds.land,
                                                allow_short_climatology=synthetic_fixture)
        for L in ds.lead.values:
            self.fc_engines[int(L)] = RegimeEngine().fit(ds.fc_tp.sel(lead=L, time=train_times), ds.land,
                                                         allow_short_climatology=synthetic_fixture)
        return self

    def labels(self, ds: xr.Dataset) -> xr.Dataset:
        f = {k: ds[f"an_{k}"] for k in STATE_VARS}
        return self.label_engine.transform(ds.obs_rain, f, ds, run_filter=True)

    def forecast(self, ds: xr.Dataset, lead: int) -> xr.Dataset:
        f = {k: ds[f"fc_{k}"].sel(lead=lead) for k in STATE_VARS}
        return self.fc_engines[int(lead)].transform(ds.fc_tp.sel(lead=lead), f, ds, run_filter=False)


def feature_cube(ds: xr.Dataset, lead: int, fc_reg: xr.Dataset) -> dict[str, np.ndarray]:
    """All predictors on the full grid for one lead: name -> (time, lat, lon)."""
    lat, lon = ds.lat.values, ds.lon.values
    res = float(lat[1] - lat[0])
    nt = ds.sizes["time"]
    shp = (nt, len(lat), len(lon))
    g = lambda v: ds[f"fc_{v}"].sel(lead=lead).values.astype("float32")
    tp = g("tp")
    u, v, tcwv, cape = g("u850"), g("v850"), g("tcwv"), g("cape")

    elev, cdist = ds.elev.values, ds.coast_dist.values
    dy = res * 111e3
    dx = dy * np.cos(np.deg2rad(lat))[:, None]
    hy, hx = np.gradient(elev)
    cy, cx = np.gradient(cdist)
    cn = np.hypot(cx, cy) + 1e-12
    doy = pd.DatetimeIndex(ds.time.values).dayofyear.values
    b = lambda a: np.broadcast_to(a, shp).astype("float32")

    from .grid import CMZ_BOX, box_mask
    cmz = box_mask(lat, lon, CMZ_BOX)
    out = {
        "tp": tp,
        "tp_log": np.log1p(tp),
        "tp_n3_mean": _nb(tp, 3, "mean"),
        "tp_n5_mean": _nb(tp, 5, "mean"),
        "tp_n5_max": _nb(tp, 5, "max"),
        "tp_n9_mean": _nb(tp, 9, "mean"),
        "tp_n9_max": _nb(tp, 9, "max"),
        "tp_grad": _grad_mag(tp, res, lat),
        "u850": u, "v850": v, "wspd": np.hypot(u, v), "tcwv": tcwv, "cape": cape,
        "moist_flux": np.hypot(u, v) * tcwv,
        "upslope": u * (hx / dx)[None] + v * (hy / dy)[None],
        "onshore": u * (cx / cn)[None] + v * (cy / cn)[None],
        "elev": b(elev[None]), "coast_dist": b(cdist[None]),
        "lat": b(lat[None, :, None]), "lon": b(lon[None, None, :]),
        "doy_sin": b(np.sin(2 * np.pi * doy / 365.25)[:, None, None]),
        "doy_cos": b(np.cos(2 * np.pi * doy / 365.25)[:, None, None]),
        "lead": np.full(shp, lead, "float32"),
        "monsoon_z": b(fc_reg.monsoon_z.values[:, None, None]),
        "cmz_tp": b(tp[:, cmz].mean(1)[:, None, None]),
        "dist_system_km": np.minimum(fc_reg.dist_system_km.values, 3000),
        "synoptic": fc_reg.synoptic.values.astype("float32"),
        "oro_score": fc_reg.oro_score.values, "coast_score": fc_reg.coast_score.values,
        "conv_score": fc_reg.conv_score.values,
        "mslp_spatial_residual": fc_reg.mslp_spatial_residual.values,
        "vo850": fc_reg.vo850.values, "fc_primary": fc_reg.primary.values.astype("float32"),
    }
    return out


def build_table(ds: xr.Dataset, bundle: RegimeBundle, labels: xr.Dataset | None = None) -> pd.DataFrame:
    """Long table over land cells: one row per (lead, time, cell)."""
    if "obs_rain" in ds:      # rows pair forecasts with observations -> convention must be verified
        from .data.build import check_dataset_convention
        check_dataset_convention(ds, None, "create training samples")
    land = ds.land.values
    iy, ix = np.nonzero(land)
    lat, lon = ds.lat.values, ds.lon.values
    times = pd.DatetimeIndex(ds.time.values)
    frames = []
    for L in ds.lead.values:
        cube = feature_cube(ds, int(L), bundle.forecast(ds, int(L)))
        cols = {k: v[:, iy, ix].ravel().astype("float32") for k, v in cube.items()}
        nt = len(times)
        cols["time"] = np.repeat(times.values, len(iy))
        cols["iy"] = np.tile(iy, nt).astype("int16")
        cols["ix"] = np.tile(ix, nt).astype("int16")
        cols["year"] = np.repeat(times.year.values, len(iy)).astype("int16")
        if "obs_rain" in ds:
            cols["obs"] = ds.obs_rain.values[:, iy, ix].ravel().astype("float32")
        if labels is not None:
            cols["label_primary"] = labels.primary.values[:, iy, ix].ravel()
            cols["label_state"] = np.repeat(labels.monsoon_state.values, len(iy))
        frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, ignore_index=True)
    df["region"] = region_of(df["lat"].values, df["lon"].values)
    keep = np.isfinite(df["tp"])
    if "obs" in df:
        keep &= np.isfinite(df["obs"])
    df = df[keep].reset_index(drop=True)
    return df
