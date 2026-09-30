"""Regime x Error Atlas (does NWP error depend on regime? — the project's core hypothesis)
and a drift monitor (does the error/feature distribution shift, e.g. after an NWP upgrade?)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Config
from ..grid import region_of
from ..regimes.engine import PRIMARY
from .metrics import contingency


def error_atlas(cfg: Config, ds: xr.Dataset) -> dict:
    labels = xr.open_dataset(cfg.product_dir / "regime_labels.nc")
    land = ds.land.values
    prim = labels.primary.values
    obs = ds.obs_rain.values
    lat2, lon2 = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    reg = region_of(lat2, lon2)
    rows = []
    bias_maps = np.full((ds.sizes["lead"], len(PRIMARY), *land.shape), np.nan, "float32")
    count_maps = np.zeros_like(bias_maps)
    for li, L in enumerate(ds.lead.values):
        fc = ds.fc_tp.sel(lead=L).values
        err = fc - obs
        for c, name in enumerate(PRIMARY):
            m = (prim == c) & land[None]
            n = m.sum(0)
            s = np.where(m, err, 0).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                bias_maps[li, c] = np.where(n >= 10, s / n, np.nan)
            count_maps[li, c] = n
            for r in np.unique(reg[land]):
                mm = m & (reg == r)[None]
                if mm.sum() < 100:
                    continue
                f, o = fc[mm], obs[mm]
                h = contingency(f, o, 64.5)
                rows.append({"lead": int(L), "regime": name, "region": r, "n": int(mm.sum()),
                             "bias": float((f - o).mean()), "rmse": float(np.sqrt(((f - o) ** 2).mean())),
                             "rel_bias_pct": float(100 * (f.sum() - o.sum()) / max(o.sum(), 1e-6)),
                             "pod_64.5": h["pod"], "csi_64.5": h["csi"], "obs_freq_64.5": h["base_rate"]})
    atlas = xr.Dataset({"bias": (("lead", "regime", "lat", "lon"), bias_maps),
                        "count": (("lead", "regime", "lat", "lon"), count_maps)},
                       coords={"lead": ds.lead.values, "regime": PRIMARY, "lat": ds.lat.values, "lon": ds.lon.values})
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_netcdf(cfg.report_dir / "error_atlas.nc")
    t = pd.DataFrame(rows)
    # the hypothesis test in one number: spread of regime-mean bias vs overall bias
    by_regime = t.groupby(["lead", "regime"]).apply(lambda g: np.average(g.bias, weights=g.n), include_groups=False)
    return {"table": rows, "regime_mean_bias": {f"lead{k[0]}_{k[1]}": round(float(v), 2) for k, v in by_regime.items()}}


def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """Population stability index of b relative to a."""
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    pa = np.histogram(np.clip(a, q[0], q[-1]), q)[0] / len(a) + 1e-6
    pb = np.histogram(np.clip(b, q[0], q[-1]), q)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def drift_check(cfg: Config, pred: pd.DataFrame, features=("tp", "tcwv", "cape", "u850", "oro_score", "monsoon_z")) -> dict:
    """Compare new-period raw-NWP error & inputs against the training archive.
    Flags: |PSI| > 0.25 (major shift) or per-regime bias change > 3 mm/day."""
    arch = pd.read_parquet(cfg.product_dir / "train_archive.parquet")
    out = {"feature_psi": {}, "regime_bias_shift": {}, "flags": []}
    tp_new = pred["tp"].values
    for f in features:
        if f in arch and f in pred:
            out["feature_psi"][f] = round(psi(arch[f].values, pred[f].values), 3)
    if "tp" in arch:
        out["feature_psi"]["tp"] = round(psi(arch["tp"].values, tp_new), 3)
    for c, name in enumerate(PRIMARY):
        a = arch[arch.label_primary == c]
        b = pred[pred.label_primary == c]
        if len(a) > 500 and len(b) > 500:
            shift = float((b.tp - b.obs).mean() - (a.tp - a.obs).mean())
            out["regime_bias_shift"][name] = round(shift, 2)
            if abs(shift) > 3:
                out["flags"].append(f"raw bias in '{name}' shifted {shift:+.1f} mm/day vs training")
    out["flags"] += [f"PSI {k}={v}" for k, v in out["feature_psi"].items() if v > 0.25]
    out["retrain_recommended"] = bool(out["flags"])
    return out
