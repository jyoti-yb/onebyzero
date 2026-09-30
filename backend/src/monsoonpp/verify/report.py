"""Verification report: the skill cube REGIME x REGION x LEAD x THRESHOLD x SCALE."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..data.build import check_dataset_convention, load_dataset
from ..data.obs_time import require_verified
from ..pipeline import MODEL_ORDER
from ..regimes.engine import PRIMARY
from .metrics import contingency, continuous, fss, fss_useful, probabilistic

log = logging.getLogger(__name__)


def _to_grid(sub: pd.DataFrame, col: str, times, ny, nx) -> np.ndarray:
    arr = np.full((len(times), ny, nx), np.nan, "float32")
    tix = pd.Index(times).get_indexer(sub["time"])
    arr[tix, sub["iy"].values, sub["ix"].values] = sub[col].values
    return arr


def verification_tables(cfg: Config, pred: pd.DataFrame, ds=None) -> dict:
    require_verified(cfg.obs_time_convention, "run verification", cfg.obs_source)
    thr = cfg.thresholds
    models = [m for m in MODEL_ORDER if m in pred]
    res: dict = {"overall": [], "categorical": [], "by_regime": [], "by_region": [], "fss": [], "prob": []}

    for L, g in pred.groupby("lead"):
        o = g["obs"].values
        for m in models:
            f = g[m].values
            res["overall"].append({"lead": int(L), "model": m, **continuous(f, o)})
            for t in thr:
                res["categorical"].append({"lead": int(L), "model": m, **contingency(f, o, t)})
        # regime & region stratification
        for key, col in (("by_regime", "label_primary"), ("by_region", "region")):
            for val, gg in g.groupby(col):
                if len(gg) < 200:
                    continue
                name = PRIMARY[int(val)] if col == "label_primary" else val
                for m in models:
                    c = continuous(gg[m].values, gg["obs"].values)
                    h = contingency(gg[m].values, gg["obs"].values, 64.5)
                    res[key].append({"lead": int(L), "stratum": name, "model": m, "n": c["n"], "bias": c["bias"],
                                     "rmse": c["rmse"], "corr": c["corr"], "pod_64.5": h["pod"],
                                     "far_64.5": h["far"], "csi_64.5": h["csi"], "ets_64.5": h["ets"]})
        # probabilistic heads vs deterministic-threshold reference
        for pt in cfg.prob_thresholds:
            col = f"p_gt_{pt:g}"
            if col not in g:
                continue
            ev = o > pt
            res["prob"].append({"lead": int(L), "threshold": pt, "source": "heads (calibrated)",
                                **probabilistic(g[col].values, ev)})
            for m in ("L0_raw", "L5_moe"):
                res["prob"].append({"lead": int(L), "threshold": pt, "source": f"{m} >= thr (0/1)",
                                    **probabilistic((g[m].values > pt).astype(float), ev)})

    # FSS on reconstructed grids
    if ds is None:
        ds = load_dataset(cfg, "run verification")
    check_dataset_convention(ds, cfg, "run verification")
    ny, nx = ds.sizes["lat"], ds.sizes["lon"]
    land = ds.land.values
    for L, g in pred.groupby("lead"):
        times = np.sort(g["time"].unique())
        ob = _to_grid(g, "obs", times, ny, nx)
        for m in models:
            fc = _to_grid(g, m, times, ny, nx)
            fc = np.nan_to_num(fc)
            for t in (15.6, 64.5, 115.6):
                base = float(np.nanmean(ob[:, land] >= t))
                for s in cfg.fss_scales:
                    res["fss"].append({"lead": int(L), "model": m, "threshold": t, "scale_cells": s,
                                       "scale_km": round(s * float(ds.lat[1] - ds.lat[0]) * 111),
                                       "fss": fss(fc, ob, land, t, s), "useful": fss_useful(base)})
    return res


def headline(res: dict) -> pd.DataFrame:
    ov = pd.DataFrame(res["overall"])
    cat = pd.DataFrame(res["categorical"])
    rows = []
    for (L, m), g in ov.groupby(["lead", "model"]):
        c = cat[(cat.lead == L) & (cat.model == m)].set_index("threshold")
        rows.append({"lead": L, "model": m, "rmse": g["rmse"].iloc[0], "bias": g["bias"].iloc[0], "corr": g["corr"].iloc[0],
                     "ETS_15.6": c.loc[15.6, "ets"], "CSI_64.5": c.loc[64.5, "csi"], "POD_64.5": c.loc[64.5, "pod"],
                     "FAR_64.5": c.loc[64.5, "far"], "CSI_115.6": c.loc[115.6, "csi"], "FB_64.5": c.loc[64.5, "freq_bias"]})
    h = pd.DataFrame(rows)
    h["model"] = pd.Categorical(h["model"], [m for m in MODEL_ORDER if m in set(h.model)])
    return h.sort_values(["lead", "model"]).reset_index(drop=True)


def _md(df: pd.DataFrame, floatfmt=3) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values) + "\n"


def write_report(cfg: Config, split: str = "test") -> dict:
    require_verified(cfg.obs_time_convention, "run verification", cfg.obs_source)
    pred = pd.read_parquet(cfg.product_dir / "predictions.parquet")
    pred = pred[pred["split"] == split]
    ds = load_dataset(cfg, "run verification")
    res = verification_tables(cfg, pred, ds)
    from .atlas import error_atlas, drift_check
    res["atlas"] = error_atlas(cfg, ds)
    res["drift"] = drift_check(cfg, pred)
    clf = json.loads((cfg.product_dir / "regime_classifier_eval.json").read_text())
    res["regime_classifier"] = clf

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    (cfg.report_dir / f"verification_{split}.json").write_text(json.dumps(res, indent=1, default=float))

    h = headline(res)
    fss_df = pd.DataFrame(res["fss"])
    fss_tab = fss_df[fss_df.threshold == 64.5].pivot_table(index=["lead", "model"], columns="scale_km", values="fss").reset_index()
    reg = pd.DataFrame(res["by_regime"])
    reg_tab = reg[reg.model.isin(["L0_raw", "L3_gbm_generic", "L4_gbm_regime", "L5_moe"])].pivot_table(
        index=["lead", "stratum"], columns="model", values="rmse").reset_index()
    reg_csi = reg[reg.model.isin(["L0_raw", "L3_gbm_generic", "L4_gbm_regime", "L5_moe"])].pivot_table(
        index=["lead", "stratum"], columns="model", values="csi_64.5").reset_index()
    prob = pd.DataFrame(res["prob"])[["lead", "threshold", "source", "brier", "bss", "roc_auc", "base_rate"]]
    atlas = pd.DataFrame(res["atlas"]["table"])
    atlas_tab = atlas[atlas.lead == atlas.lead.min()].pivot_table(index="regime", columns="region", values="bias").round(1).reset_index()

    src_note = ("**SYNTHETIC DATA** — numbers validate the pipeline, not forecast skill. "
                if cfg.forecast_source == "synthetic" else "")
    md = [f"# Verification report — {cfg.name} ({split} years {cfg.split.test_years if split == 'test' else cfg.split.val_years})\n",
          f"{src_note}NWP: `{cfg.forecast_source}` ({cfg.nwp_model_version}); obs: `{cfg.obs_source}`; "
          f"rows: {len(pred):,}; thresholds follow IMD categories (64.5 heavy, 115.6 very heavy).\n",
          "## 1. Headline: correction ladder\n", _md(h),
          "## 2. Main experiment: generic vs regime-aware, RMSE by analysis regime\n", _md(reg_tab),
          "### CSI (>=64.5 mm) by regime\n", _md(reg_csi),
          "## 3. FSS, threshold 64.5 mm, by neighbourhood scale (km)\n", _md(fss_tab),
          "## 4. Heavy-rain probabilities\n", _md(prob),
          "## 5. Regime x Error Atlas (raw NWP mean error, mm/day, shortest lead, all years)\n", _md(atlas_tab),
          "## 6. Regime classifier (forecast fields -> analysis regime)\n",
          f"test accuracy {clf['test']['accuracy']:.3f}, macro-F1 {clf['test']['macro_f1']:.3f}; "
          f"per class: {clf['test']['per_class_f1']}\n",
          "## 7. Drift check\n", "```\n" + json.dumps(res["drift"], indent=1, default=float) + "\n```\n"]
    (cfg.report_dir / f"report_{split}.md").write_text("\n".join(md))
    h.to_csv(cfg.report_dir / f"headline_{split}.csv", index=False)
    _figures(cfg, res, split)
    return res


def _figures(cfg: Config, res: dict, split: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reliability
    prob = res["prob"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    for p in prob:
        if p["source"].startswith("heads"):
            r = p["reliability"]
            ax.plot([x["mean_p"] for x in r], [x["obs_freq"] for x in r], marker="o",
                    label=f"lead {p['lead']} >{p['threshold']:g} mm")
    ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency"); ax.set_title("Reliability")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(cfg.report_dir / f"reliability_{split}.png", dpi=120); plt.close(fig)

    # CSI by threshold per model (lead min)
    cat = pd.DataFrame(res["categorical"])
    cat = cat[cat.lead == cat.lead.min()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, g in cat.groupby("model", sort=False):
        ax.plot(g.threshold.astype(str), g.csi, marker="o", label=m)
    ax.set_xlabel("threshold (mm/day)"); ax.set_ylabel("CSI"); ax.set_title("CSI by threshold, shortest lead")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(cfg.report_dir / f"csi_{split}.png", dpi=120); plt.close(fig)

    # atlas maps
    import xarray as xr
    p = cfg.report_dir / "error_atlas.nc"
    if p.exists():
        a = xr.open_dataset(p)
        L = int(a.lead.min())
        regs = [r for r in PRIMARY if r in a.regime.values]
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for ax, r in zip(axs.ravel(), regs):
            a.bias.sel(lead=L, regime=r).plot(ax=ax, cmap="BrBG_r", vmin=-30, vmax=30, add_colorbar=True)
            ax.set_title(f"{r}: raw NWP bias (mm/d)")
        fig.tight_layout(); fig.savefig(cfg.report_dir / "error_atlas_maps.png", dpi=110); plt.close(fig)
