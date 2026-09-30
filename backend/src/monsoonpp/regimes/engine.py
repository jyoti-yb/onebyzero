"""Hierarchical objective regime proxies with explicit capability limits.

Nothing emitted here is an official IMD classification. Active/break follows the
published Rajeevan et al. (2010) objective rainfall criterion only when an adequate
daily climatology is fitted. Low/depression labels are NWP-field proxies. Local
forcing values are dimensionless heuristic scores for stratified verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, uniform_filter1d

from ..grid import box_mask

MONSOON_STATES = ["break", "normal", "active"]
SYNOPTIC = ["none", "low_proxy", "depression_proxy"]
PRIMARY = ["low_depression_proxy", "orographic", "coastal", "active", "normal", "break"]
CMZ_BOX = (18.0, 28.0, 65.0, 88.0)
SYSTEM_SEARCH_BOX = (15.0, 28.0, 74.0, 93.0)


class RegimeCapabilityError(RuntimeError):
    pass


@dataclass
class RegimeParams:
    z_thresh: float = 1.0
    min_run_days: int = 3
    min_climatology_years: int = 20
    background_sigma_cells: float = 8.0
    low_spatial_residual_hpa: float = 1.5
    low_closed_depth_hpa: float = 2.0
    low_vorticity_1e5: float = 1.0
    depression_closed_depth_hpa: float = 4.0
    depression_vorticity_1e5: float = 2.0
    depression_wind_ms: float = 8.7
    closed_ring_radius_deg: float = 3.0
    min_center_separation_deg: float = 4.0
    max_systems_per_day: int = 12
    influence_km: dict = field(default_factory=lambda: {"low_proxy": 350.0, "depression_proxy": 550.0})
    oro_ref: float = 0.12
    onshore_ref: float = 8.0
    coast_decay_km: float = 150.0
    convective_cape_ref: float = 2500.0
    convective_pwat_ref: float = 60.0
    primary_score_thresh: float = 0.5


def _grad(field2d, res_deg, lat):
    dy = res_deg * 111e3
    dx = dy * np.cos(np.deg2rad(lat))[:, None]
    gy, gx = np.gradient(field2d)
    return gx / dx, gy / dy


def _run_filter(state: np.ndarray, times, min_run: int | None = None) -> np.ndarray:
    """Keep only same-state runs on consecutive calendar days."""
    if min_run is None:  # compatibility with the original _run_filter(state, min_run) helper
        min_run = int(times)
        times = pd.date_range("2000-01-01", periods=len(state), freq="D")
    dates = pd.DatetimeIndex(times).normalize()
    out = np.zeros_like(state)
    i, n = 0, len(state)
    while i < n:
        j = i + 1
        while (j < n and state[j] == state[i]
               and dates[j] - dates[j - 1] == pd.Timedelta(days=1)):
            j += 1
        if state[i] != 0 and (j - i) >= min_run:
            out[i:j] = state[i]
        i = j
    return out


class RegimeEngine:
    def __init__(self, params: RegimeParams | None = None):
        self.p = params or RegimeParams()
        self.clim_mean: np.ndarray | None = None
        self.clim_std: np.ndarray | None = None
        self.climatology_years: list[int] = []

    @staticmethod
    def _cmz_series(rain: xr.DataArray, land: xr.DataArray | np.ndarray | None = None) -> xr.DataArray:
        mask = box_mask(rain.lat.values, rain.lon.values, CMZ_BOX)
        if land is not None:
            mask &= np.asarray(land, dtype=bool)
        weights = xr.DataArray(
            np.cos(np.deg2rad(rain.lat.values))[:, None] * np.ones((1, rain.sizes["lon"])),
            dims=("lat", "lon"), coords={"lat": rain.lat, "lon": rain.lon},
        ).where(mask)
        valid_weights = weights.where(np.isfinite(rain))
        return (rain * valid_weights).sum(("lat", "lon"), skipna=True) / valid_weights.sum(("lat", "lon"), skipna=True)

    def fit(self, rain: xr.DataArray, land=None, *, allow_short_climatology: bool = False) -> "RegimeEngine":
        years = sorted(set(pd.DatetimeIndex(rain.time.values).year))
        if len(years) < self.p.min_climatology_years and not allow_short_climatology:
            raise RegimeCapabilityError(
                f"active/break proxy needs at least {self.p.min_climatology_years} years of daily rainfall; "
                f"only {len(years)} supplied"
            )
        series = self._cmz_series(rain, land)
        doy = series.time.dt.dayofyear.values
        mean = np.full(367, np.nan)
        std = np.full(367, np.nan)
        for day in np.unique(doy):
            values = series.values[doy == day]
            mean[day] = np.nanmean(values)
            std[day] = np.nanstd(values, ddof=1) if np.isfinite(values).sum() > 1 else np.nan
        idx = np.arange(367)
        for values, fallback, positive in ((mean, float(np.nanmean(series.values)), False),
                                           (std, float(np.nanstd(series.values)), True)):
            valid = np.isfinite(values) & ((values > 0) if positive else True)
            if valid.sum() < 2:
                values[:] = fallback
            else:
                values[:] = np.interp(idx, idx[valid], values[valid])
        self.clim_mean = uniform_filter1d(mean, 31, mode="nearest")
        self.clim_std = np.maximum(uniform_filter1d(std, 31, mode="nearest"), 1e-6)
        self.climatology_years = years
        return self

    def monsoon_index(self, rain: xr.DataArray, land=None) -> np.ndarray:
        if self.clim_mean is None or self.clim_std is None:
            raise RegimeCapabilityError("active/break proxy unavailable: no validated climatology is fitted")
        series = self._cmz_series(rain, land)
        doy = series.time.dt.dayofyear.values
        return (series.values - self.clim_mean[doy]) / self.clim_std[doy]

    def classify_monsoon(self, rain: xr.DataArray, land=None, *, run_filter: bool = True):
        z = self.monsoon_index(rain, land)
        times = pd.DatetimeIndex(rain.time.values)
        state = np.where(z >= self.p.z_thresh, 1, np.where(z <= -self.p.z_thresh, -1, 0))
        state[~np.isin(times.month, (7, 8))] = 0
        if run_filter:
            state = _run_filter(state, times, self.p.min_run_days)
        return z, state

    def transform_atmosphere(self, fields: dict[str, xr.DataArray], static: xr.Dataset) -> xr.Dataset:
        """Classify synoptic proxies and local forcing without claiming a monsoon state."""
        required = ("u850", "v850", "mslp", "cape", "tcwv")
        missing = [name for name in required if name not in fields]
        if missing:
            raise RegimeCapabilityError(f"atmospheric regime inputs missing: {missing}")
        ref = fields["mslp"]
        lat, lon = ref.lat.values, ref.lon.values
        res = float(np.median(np.diff(lat)))
        nt = ref.sizes["time"]
        u, v = fields["u850"].values, fields["v850"].values
        mslp, cape, tcwv = ref.values, fields["cape"].values, fields["tcwv"].values
        elev, coast_dist = static.elev.values, static.coast_dist.values
        hx, hy = _grad(elev, res, lat)
        cx, cy = _grad(coast_dist, res, lat)
        coast_norm = np.hypot(cx, cy) + 1e-12
        cx, cy = cx / coast_norm, cy / coast_norm

        shape = (nt, len(lat), len(lon))
        distance = np.full(shape, 9999.0, "float32")
        synoptic = np.zeros(shape, "int8")
        oro = np.zeros(shape, "float32")
        coast = np.zeros(shape, "float32")
        conv = np.zeros(shape, "float32")
        residual = np.zeros(shape, "float32")
        vorticity = np.zeros(shape, "float32")
        search = box_mask(lat, lon, SYSTEM_SEARCH_BOX)
        lon2, lat2 = np.meshgrid(lon, lat)
        radius = max(1, int(round(self.p.closed_ring_radius_deg / res)))
        footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        if radius > 1:
            footprint[1:-1, 1:-1] = False
        systems = []
        detection_audit = []

        for t in range(nt):
            background = gaussian_filter(mslp[t], self.p.background_sigma_cells, mode="nearest")
            spatial_residual = mslp[t] - background
            residual[t] = spatial_residual
            ux, uy = _grad(u[t], res, lat)
            vx, _ = _grad(v[t], res, lat)
            vort = (vx - uy) * 1e5
            vorticity[t] = vort
            wind = np.hypot(u[t], v[t])
            ring_min = minimum_filter(mslp[t], footprint=footprint, mode="constant", cval=-np.inf)
            closed_depth = ring_min - mslp[t]
            max_wind_field = maximum_filter(wind, size=2 * radius + 1, mode="nearest")
            residual_candidates = ((spatial_residual == minimum_filter(spatial_residual, size=7)) & search
                                   & (spatial_residual <= -self.p.low_spatial_residual_hpa))
            vorticity_candidates = residual_candidates & (vort >= self.p.low_vorticity_1e5)
            candidates = vorticity_candidates & (closed_depth >= self.p.low_closed_depth_hpa)
            points = list(zip(*np.nonzero(candidates)))
            points.sort(key=lambda point: spatial_residual[point])
            accepted = []
            for iy, ix in points:
                if any(np.hypot(lat[iy] - lat[ay], lon[ix] - lon[ax]) < self.p.min_center_separation_deg
                       for ay, ax in accepted):
                    continue
                accepted.append((iy, ix))
                if len(accepted) >= self.p.max_systems_per_day:
                    break
            for iy, ix in accepted:
                depth = float(closed_depth[iy, ix])
                max_wind = float(max_wind_field[iy, ix])
                is_depression = (depth >= self.p.depression_closed_depth_hpa
                                 and vort[iy, ix] >= self.p.depression_vorticity_1e5
                                 and max_wind >= self.p.depression_wind_ms)
                code = 2 if is_depression else 1
                label = SYNOPTIC[code]
                systems.append({
                    "time": str(pd.Timestamp(ref.time.values[t])), "lat": float(lat[iy]), "lon": float(lon[ix]),
                    "class": label, "mslp_hpa": float(mslp[t, iy, ix]),
                    "mslp_spatial_residual_hpa": float(spatial_residual[iy, ix]),
                    "closed_ring_depth_hpa": depth, "vorticity_1e5_s-1": float(vort[iy, ix]),
                    "max_850_wind_ms": max_wind,
                })
                d_km = np.hypot((lat2 - lat[iy]) * 111.0,
                                (lon2 - lon[ix]) * 111.0 * np.cos(np.deg2rad(lat[iy])))
                closer = d_km < distance[t]
                influence = self.p.influence_km[label]
                distance[t] = np.where(closer, d_km, distance[t])
                synoptic[t] = np.where(closer & (d_km <= influence), code,
                                       np.where(closer, 0, synoptic[t]))

            detection_audit.append({
                "time": str(pd.Timestamp(ref.time.values[t])),
                "spatial_residual_minima": int(residual_candidates.sum()),
                "with_vorticity": int(vorticity_candidates.sum()),
                "with_closed_ring": int(candidates.sum()),
                "accepted_separated_centres": len(accepted),
                "min_spatial_residual_hpa": float(np.nanmin(spatial_residual[search])),
                "max_vorticity_1e5_s-1": float(np.nanmax(vort[search])),
                "max_closed_ring_depth_hpa": (
                    float(np.nanmax(closed_depth[vorticity_candidates])) if vorticity_candidates.any() else float("nan")
                ),
            })

            upslope = u[t] * hx + v[t] * hy
            oro[t] = np.clip(upslope / self.p.oro_ref, 0, 1)
            onshore = u[t] * cx + v[t] * cy
            coast[t] = (np.clip(onshore / self.p.onshore_ref, 0, 1)
                        * np.exp(-coast_dist / self.p.coast_decay_km) * (coast_dist > 0))
            conv[t] = (np.clip(cape[t] / self.p.convective_cape_ref, 0, 1)
                       * np.clip(tcwv[t] / self.p.convective_pwat_ref, 0, 1))

        dims = ("time", "lat", "lon")
        out = xr.Dataset(
            {
                "synoptic": (dims, synoptic),
                "dist_system_km": (dims, distance),
                "oro_score": (dims, oro),
                "coast_score": (dims, coast),
                "conv_score": (dims, conv),
                "mslp_spatial_residual": (dims, residual),
                "vo850": (dims, vorticity),
            },
            coords={"time": ref.time.values, "lat": lat, "lon": lon},
        )
        out.synoptic.attrs.update(
            classes=",".join(SYNOPTIC), classification_kind="objective NWP proxy, not official IMD classification"
        )
        out.mslp_spatial_residual.attrs.update(
            units="hPa", long_name="MSLP minus Gaussian spatial background",
            warning="spatial residual; not a climatological anomaly",
        )
        out.oro_score.attrs["classification_kind"] = "heuristic terrain-flow forcing score"
        out.coast_score.attrs["classification_kind"] = "heuristic onshore-flow coastal forcing score"
        out.conv_score.attrs["classification_kind"] = "heuristic CAPE-PWAT convective-environment score"
        out.attrs.update(
            synoptic_classes=",".join(SYNOPTIC), wd_proxy_available="false",
            wd_proxy_reason="no validated upstream track and 350-450 hPa vorticity-anomaly climatology",
            official_classification="none; all synoptic/local outputs are objective proxies or heuristic scores",
        )
        self.last_systems = systems
        self.last_detection_audit = detection_audit
        return out

    def transform(self, rain: xr.DataArray, fields: dict[str, xr.DataArray], static: xr.Dataset,
                  run_filter: bool) -> xr.Dataset:
        z, state = self.classify_monsoon(rain, static.get("land"), run_filter=run_filter)

        out = self.transform_atmosphere(fields, static)
        shape = out.synoptic.shape
        state3 = np.broadcast_to(state[:, None, None], shape)
        threshold = self.p.primary_score_thresh
        primary = np.where(out.synoptic.values > 0, 0,
                  np.where(out.oro_score.values >= threshold, 1,
                  np.where(out.coast_score.values >= threshold, 2,
                  np.where(state3 == 1, 3, np.where(state3 == -1, 5, 4))))).astype("int8")
        out["monsoon_z"] = (("time",), z.astype("float32"))
        out["monsoon_state"] = (("time",), (state + 1).astype("int8"))
        out["primary"] = (("time", "lat", "lon"), primary)
        out.monsoon_state.attrs.update(
            classes=",".join(MONSOON_STATES),
            classification_kind="Rajeevan-2010-style objective proxy; not an official IMD designation",
            climatology_years=",".join(map(str, self.climatology_years)),
        )
        out.attrs.update(primary_classes=",".join(PRIMARY), monsoon_states=",".join(MONSOON_STATES))
        return out
