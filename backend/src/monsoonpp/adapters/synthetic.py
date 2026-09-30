"""Synthetic monsoon world with KNOWN, regime-dependent NWP errors.

Purpose: exercise the full pipeline offline and give the regime-aware models
something real to learn. It is not a climate model. What is built in, on purpose:

truth
  * seasonal cycle + 30–50 day intraseasonal oscillation (active / break)
  * monsoon lows/depressions forming over the head of the Bay, moving WNW,
    heaviest rain in the SW quadrant
  * orographic rain = upslope flow x moisture (Western Ghats, NE hills, Himalaya)
  * break-monsoon rainfall shift to Himalayan foothills / NE / south peninsula
NWP forecast errors (what post-processing should learn)
  * damped intraseasonal amplitude -> under-forecast in active, over-forecast in break
  * coarse terrain -> strong under-forecast of windward orographic rain
  * depression rain centred on the low (no SW offset), smeared, plus track error ~ lead
  * newly forming lows missed at longer leads (unrecoverable error)
  * drizzle bias: too many light-rain days, worse in break
  * noise decorrelating with lead
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from ..config import Config
from ..grid import build_static, make_coords
from .base import AnalysisAdapter, ForecastAdapter, ObsAdapter, register


@dataclass
class _System:
    lat: float
    lon: float
    deficit: float  # hPa
    age: int
    life: int
    dlon: float
    dlat: float
    peak: float


def _smooth_noise(rng, shape, sigma):
    f = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return f / (f.std() + 1e-9)


class SyntheticWorld:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lat, self.lon = make_coords(cfg.grid)
        self.static = build_static(cfg.grid)
        self.lon2, self.lat2 = np.meshgrid(self.lon, self.lat)
        self.elev = self.static["elev"].values.astype("float64")
        self.elev_coarse = gaussian_filter(self.elev, 2.0)        # what the NWP model "sees"
        self.land = self.static["land"].values
        self.foot = np.interp(self.lon2, [72, 74, 77, 80, 84, 88, 92, 97],
                              [34.5, 33.0, 31.0, 29.5, 28.0, 27.2, 27.5, 28.2])
        dy = cfg.grid.res * 111e3
        dx = dy * np.cos(np.deg2rad(self.lat2))
        self._grads = {}
        for key, h in (("true", self.elev), ("coarse", self.elev_coarse)):
            gy, gx = np.gradient(h)
            self._grads[key] = (gx / dx, gy / dy)
        self._generate()

    # ------------------------------------------------------------------ fields
    def _fields(self, s, m, systems, rng, noise_scale):
        lat2, lon2 = self.lat2, self.lon2
        u = 12 * s * (1 + 0.35 * m) * np.exp(-((lat2 - 15) / 5) ** 2) + 2.0
        v = np.full_like(u, 1.0)
        mslp = 1006 - 4 * np.exp(-((lat2 - 28) / 5) ** 2 - ((lon2 - 72) / 8) ** 2)
        tcwv = 28 + 32 * s * (1 + 0.25 * m) * np.exp(-((lat2 - 20) / 10) ** 2)
        tcwv -= 14 * ((lat2 > 24) & (lon2 < 75))
        for sy in systems:
            dxs, dys = lon2 - sy.lon, lat2 - sy.lat
            r = np.hypot(dxs, dys) + 1e-6
            R = 3.0
            vt = 2.5 * sy.deficit * (r / R) * np.exp(0.5 * (1 - (r / R) ** 2))
            u += -vt * dys / r
            v += vt * dxs / r
            mslp -= sy.deficit * np.exp(-r ** 2 / (2 * 2.0 ** 2))
            tcwv += 8 * np.exp(-r ** 2 / 8)
        cape = 700 + 1000 * s + 350 * np.clip(-m, 0, None) * self.land
        shp = u.shape
        u = u + noise_scale * 1.5 * _smooth_noise(rng, shp, 2)
        v = v + noise_scale * 1.5 * _smooth_noise(rng, shp, 2)
        tcwv = tcwv + noise_scale * 3 * _smooth_noise(rng, shp, 2)
        cape = np.clip(cape + noise_scale * 300 * _smooth_noise(rng, shp, 2), 0, None)
        mslp = mslp + noise_scale * 0.8 * _smooth_noise(rng, shp, 3)
        return dict(u850=u, v850=v, tcwv=tcwv, cape=cape, mslp=mslp)

    def _upslope(self, u, v, which):
        gx, gy = self._grads[which]
        return u * gx + v * gy   # m/s vertical motion forced by terrain

    # ------------------------------------------------------------------ truth
    def _mu_truth(self, s, m, systems, f):
        lat2, lon2 = self.lat2, self.lon2
        cmz = 14 * s * np.maximum(0.05, 1 + 0.7 * m) * np.exp(-((lat2 - 22.5) / 4.5) ** 2) * ((lon2 > 73) & (lon2 < 88))
        ne = 18 * s * np.maximum(0.2, 1 - 0.3 * m) * np.exp(-(((lat2 - 26) / 2.5) ** 2 + ((lon2 - 92.5) / 3) ** 2))
        hf = 10 * s * np.maximum(0.2, 1 - 0.4 * m) * np.exp(-((lat2 - self.foot) / 1.2) ** 2)
        sp = 5 * s * np.clip(-m, 0, None) * np.exp(-((lat2 - 12) / 4) ** 2) * (f["cape"] / 1500)
        oro = 120 * np.clip(self._upslope(f["u850"], f["v850"], "true"), 0, None) / 0.26 * (f["tcwv"] / 55)
        dep = np.zeros_like(lat2)
        for sy in systems:
            dep += 6 * sy.deficit * np.exp(-((lon2 - sy.lon + 1.0) ** 2 + (lat2 - sy.lat + 1.0) ** 2) / (2 * 1.8 ** 2))
        return 1.0 * s + cmz + ne + hf + sp + oro + dep

    def _mu_nwp(self, s, m, systems, f):
        lat2, lon2 = self.lat2, self.lon2
        cmz = 14 * s * np.maximum(0.05, 1 + 0.45 * m) * np.exp(-((lat2 - 22.5) / 4.5) ** 2) * ((lon2 > 73) & (lon2 < 88))
        ne = 16 * s * np.maximum(0.2, 1 - 0.15 * m) * np.exp(-(((lat2 - 26) / 2.5) ** 2 + ((lon2 - 92.5) / 3) ** 2))
        hf = 9 * s * np.maximum(0.2, 1 - 0.2 * m) * np.exp(-((lat2 - self.foot) / 1.2) ** 2)
        sp = 3 * s * np.clip(-m, 0, None) * np.exp(-((lat2 - 12) / 4) ** 2)
        oro = 0.55 * 120 * np.clip(self._upslope(f["u850"], f["v850"], "coarse"), 0, None) / 0.26 * (f["tcwv"] / 55)
        dep = np.zeros_like(lat2)
        for sy in systems:
            dep += 4.5 * sy.deficit * np.exp(-((lon2 - sy.lon) ** 2 + (lat2 - sy.lat) ** 2) / (2 * 2.3 ** 2))
        drizzle = (1.5 * s + 2.5 * np.clip(-m, 0, None)) * self.land
        return 1.0 * s + cmz + ne + hf + sp + oro + dep + drizzle

    @staticmethod
    def _realise(mu, z1, z2, wet_scale, amp_sigma):
        pwet = 1 - np.exp(-mu / wet_scale)
        wet = norm.cdf(z1) < pwet
        amt = mu / np.maximum(pwet, 1e-3) * np.exp(amp_sigma * z2 - 0.5 * amp_sigma ** 2)
        return np.clip(amt * wet, 0, 600)

    # ------------------------------------------------------------------ driver
    def _generate(self):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        shape = self.lat2.shape
        leads = cfg.leads
        times, rain, ana, fc = [], [], [], {L: [] for L in leads}
        sys_log = []
        for year in cfg.years:
            dates = pd.date_range(f"{year}-{cfg.season_start}", f"{year}-{cfg.season_end}", freq="D")
            n = len(dates)
            i = np.arange(n)
            s = 0.35 + 0.65 * np.sin(np.pi * (i + 10) / (n + 20))
            P, ph = rng.uniform(35, 50), rng.uniform(0, 2 * np.pi)
            ar = np.zeros(n)
            for k in range(1, n):
                ar[k] = 0.8 * ar[k - 1] + rng.normal(0, 0.6)
            m = np.sin(2 * np.pi * i / P + ph) + 0.5 * ar
            m = (m - m.mean()) / m.std()

            systems: list[_System] = []
            for d in range(n):
                # genesis
                if rng.random() < 0.10 * (1 + 0.6 * max(m[d], 0)):
                    systems.append(_System(lat=rng.uniform(19, 22), lon=rng.uniform(86.5, 90), deficit=0.0,
                                           age=0, life=int(rng.integers(3, 8)), dlon=-rng.uniform(0.8, 1.3),
                                           dlat=rng.uniform(0, 0.4), peak=rng.uniform(2, 9)))
                for sy in systems:
                    frac = sy.age / max(sy.life - 1, 1)
                    sy.deficit = sy.peak * np.sin(np.pi * (0.25 + 0.6 * frac))
                live = [sy for sy in systems if sy.age < sy.life]
                sys_log.append([(dates[d], sy.lat, sy.lon, sy.deficit) for sy in live])

                f_true = self._fields(s[d], m[d], live, rng, noise_scale=1.0)
                mu = self._mu_truth(s[d], m[d], live, f_true)
                z1, z2 = _smooth_noise(rng, shape, 1.2), _smooth_noise(rng, shape, 1.2)
                truth = self._realise(mu, z1, z2, wet_scale=4.0, amp_sigma=0.8)
                rain.append(np.where(self.land, truth, np.nan))
                ana.append({k: v + 0.05 * np.std(v) * _smooth_noise(rng, shape, 2) for k, v in f_true.items()})
                times.append(dates[d])

                for L in leads:
                    m_f = m[d] + rng.normal(0, 0.25 * L)
                    fsys = []
                    for sy in live:
                        if sy.age < L and rng.random() < 0.5:
                            continue   # forecast missed the genesis
                        fsys.append(_System(lat=sy.lat + rng.normal(0, 0.4 * L), lon=sy.lon + rng.normal(0, 0.4 * L),
                                            deficit=sy.deficit * rng.uniform(0.6, 0.9), age=sy.age, life=sy.life,
                                            dlon=0, dlat=0, peak=sy.peak))
                    f_nwp = self._fields(s[d], m_f, fsys, rng, noise_scale=0.8 + 0.3 * L)
                    mu_f = self._mu_nwp(s[d], m_f, fsys, f_nwp)
                    rho = max(0.3, 0.75 - 0.15 * (L - 1))
                    z1f = rho * z1 + np.sqrt(1 - rho ** 2) * _smooth_noise(rng, shape, 1.5)
                    z2f = rho * z2 + np.sqrt(1 - rho ** 2) * _smooth_noise(rng, shape, 1.5)
                    tp = self._realise(mu_f, z1f, z2f, wet_scale=2.5, amp_sigma=0.5)
                    fc[L].append({"tp": gaussian_filter(tp, 0.7), **f_nwp})

                for sy in systems:
                    sy.age += 1
                    sy.lon += sy.dlon
                    sy.lat += sy.dlat
                systems = [sy for sy in systems if sy.age < sy.life]

        t = pd.DatetimeIndex(times)
        coords = {"time": t, "lat": self.lat, "lon": self.lon}
        self.obs = xr.Dataset({"rain": (("time", "lat", "lon"), np.stack(rain).astype("float32"))}, coords=coords)
        self.analysis = xr.Dataset(
            {k: (("time", "lat", "lon"), np.stack([a[k] for a in ana]).astype("float32")) for k in ana[0]},
            coords=coords)
        fvars = {}
        for k in fc[leads[0]][0]:
            fvars[k] = (("lead", "time", "lat", "lon"),
                        np.stack([np.stack([x[k] for x in fc[L]]) for L in leads]).astype("float32"))
        self.forecast = xr.Dataset(fvars, coords={"lead": leads, **coords})
        self.systems = sys_log


@lru_cache(maxsize=4)
def _world(cfg_key: str, cfg_json: str) -> SyntheticWorld:
    import json
    from ..config import load_config
    d = json.loads(cfg_json)
    cfg = load_config(None, **{k: v for k, v in d.items() if k in ("name", "years", "leads", "seed", "season_start", "season_end")})
    cfg.grid.__dict__.update(d["grid"])
    return SyntheticWorld(cfg)


def get_world(cfg: Config) -> SyntheticWorld:
    import json
    d = cfg.to_dict()
    key = json.dumps({k: d[k] for k in ("years", "leads", "seed", "grid", "season_start", "season_end")}, sort_keys=True)
    return _world(key, json.dumps(d, default=str))


def _sel(ds, dates):
    return ds.sel(time=ds.time.isin(dates))


@register("forecast", "synthetic")
class SyntheticForecastAdapter(ForecastAdapter):
    source_name = "synthetic"

    def load(self, dates, leads):
        return _sel(get_world(self.cfg).forecast, dates).sel(lead=leads)


@register("obs", "synthetic")
class SyntheticObsAdapter(ObsAdapter):
    def load(self, dates):
        return _sel(get_world(self.cfg).obs, dates)


@register("analysis", "synthetic")
class SyntheticAnalysisAdapter(AnalysisAdapter):
    def load(self, dates):
        return _sel(get_world(self.cfg).analysis, dates)
