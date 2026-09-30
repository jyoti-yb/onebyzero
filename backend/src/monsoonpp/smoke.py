"""Real-data smoke test: GFS 00Z D+L daily rain vs one observation product.

Validates the ENGINEERING chain (ingest -> exact windows -> canonical grid -> pairing ->
metrics/maps). It is not a skill estimate; every output is labelled with the observation
product role (FINAL_TRUTH / OFFICIAL_ALTERNATIVE / SMOKE_TEST_PROXY).

Config (`smoke:` block):
  start, end         forecast labels (window 03Z label -> 03Z label+1), e.g. 2024-07-15 .. 2024-07-21
  lead               forecast lead in days (default 1)
  gfs_local_dir      optional: pre-downloaded GFS files; else Herbie (AWS)
  gfs_local_pattern  default "{root}/gfs.{init:%Y%m%d}/00/atmos/gfs.t00z.pgrb2.0p25.f{fxx:03d}"
  mask               obs_valid (default for IMD products) | india_outline (default for IMERG)
  min_coverage       conservative-regrid coverage threshold for IMERG (default 1.0)
Observation source = cfg.obs_source: imd | imd_netcdf | imd_ncmrwf_merged | imerg.
Label-based products require a VERIFIED obs_time_convention (UNVERIFIED stays blocked).
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Timestamp as T
import xarray as xr

from .adapters import get_adapter
from .adapters.gfs_precip import GFSDailyPrecip, Tolerances
from .adapters.imerg import IMERGAdapter
from .config import Config
from .data.align import FORECAST_LABEL_RULE, forecast_window
from .data.imerg_daily import imerg_daily_on_grid
from .data.obs_source import ProductRole, read
from .data.obs_time import interpret_obs_times, reject_legacy_keys, require_verified
from .data.pairing import align_to_windows, forecast_windows, obs_labels_for_windows, obs_windows
from .data.regrid import strict_select
from .grid import grid_role, india_mask, make_coords
from .verify.metrics import contingency, continuous, fss

log = logging.getLogger(__name__)

ROLE_LABEL = {
    ProductRole.FINAL_TRUTH: "FINAL_TRUTH — IMD 0.25° gauge rainfall",
    ProductRole.OFFICIAL_ALTERNATIVE: "OFFICIAL_ALTERNATIVE — IMD–NCMRWF merged rainfall (not gauge-only IMD)",
    ProductRole.SMOKE_TEST_PROXY: "SMOKE_TEST_PROXY — NASA GPM IMERG satellite estimate. NOT IMD. NOT observational truth.",
}
ALWAYS = "Real-data smoke test: validates the pipeline; these numbers are not evidence of forecast skill."
SUPPORTED_OBS = ("imd", "imd_netcdf", "imd_ncmrwf_merged", "imerg")


class SmokeError(RuntimeError):
    pass


# ------------------------------------------------------------------ forecast
def _gfs_fetch(cfg: Config):
    sm = cfg.smoke
    if sm.get("gfs_local_dir"):
        pat = sm.get("gfs_local_pattern", "{root}/gfs.{init:%Y%m%d}/00/atmos/gfs.t00z.pgrb2.0p25.f{fxx:03d}")
        root = sm["gfs_local_dir"]
        def fetch(init, fxx):
            p = Path(pat.format(root=root, init=init, fxx=fxx))
            if not p.exists():
                raise FileNotFoundError(f"GFS file missing: {p}")
            return p
        return fetch
    from .adapters.gfs import GFSAdapter
    return GFSAdapter(cfg)._fetch_apcp_file


def load_gfs_daily(cfg: Config, labels: pd.DatetimeIndex, lead: int) -> xr.Dataset:
    lat, lon = make_coords(cfg.grid)
    tol = Tolerances(**cfg.adapter_options.get("gfs_precip_tolerances", {}))
    builder = GFSDailyPrecip(_gfs_fetch(cfg), tol, model="GFS", product=cfg.adapter_options.get("gfs_product", "pgrb2.0p25"))
    fields, prov = [], []
    for d in labels:
        res = builder.build(forecast_window(d, lead, FORECAST_LABEL_RULE), lead)   # frozen construction + checks
        fields.append(strict_select(res.field, lat, lon).values)
        prov.append({**res.provenance, **{k: v for k, v in res.qa.items() if k.startswith("qa_") and not isinstance(v, (list, dict))}})
    ds = xr.Dataset({"tp": (("time", "lat", "lon"), np.stack(fields).astype("float32"))},
                    coords={"time": labels, "lat": lat, "lon": lon,
                            "tp_valid_start": ("time", [p["valid_start"] for p in prov]),
                            "tp_valid_end": ("time", [p["valid_end"] for p in prov]),
                            "init_time": ("time", [p["init_time"] for p in prov]),
                            "qa_max_abs_mm": ("time", [float(p.get("qa_max_abs_mm", np.nan)) for p in prov])})
    ds["tp"].attrs.update(units="mm", **{k: prov[0][k] for k in ("model", "product", "cycle", "construction_method",
                                                                 "qa_method", "source_units", "output_units")})
    return ds


# --------------------------------------------------------------- observations
def load_obs_daily(cfg: Config, f_start, f_end):
    src = cfg.obs_source
    lat, lon = make_coords(cfg.grid)
    if src not in SUPPORTED_OBS:
        raise SmokeError(f"smoke test needs a real observation product {SUPPORTED_OBS}, not {src!r}")
    if src == "imerg":
        start, end = pd.Timestamp(min(f_start)), pd.Timestamp(max(f_end))
        native = IMERGAdapter(cfg).load(start, end)
        obs, rejected = imerg_daily_on_grid(native, f_start, lat, lon, cfg.smoke.get("min_coverage", 1.0))
        return obs, read(obs), rejected
    conv = require_verified(cfg.obs_time_convention, "smoke-test pairing", src)       # frozen guard
    labels = obs_labels_for_windows(f_start, f_end, conv)
    obs = get_adapter("obs", src, cfg).load(labels)
    obs = interpret_obs_times(obs, conv)
    meta = replace(read(obs), time_convention=conv.value)
    return obs, meta, []


# -------------------------------------------------------------------- pairing
def pair(fc: xr.Dataset, obs: xr.Dataset, rejected: list) -> tuple[xr.Dataset, list[dict]]:
    f_start, f_end = forecast_windows(fc)
    o_start, o_end = obs_windows(obs)
    al, ok = align_to_windows(obs, o_start, o_end, f_start, f_end, fc.time.values)
    rej = {r["valid_start"]: r["reason"] for r in rejected}
    unpaired = [{"forecast_label": str(pd.Timestamp(t).date()), "valid_start": str(pd.Timestamp(s)),
                 "reason": rej.get(str(pd.Timestamp(s)), "no observation record with this exact window")}
                for t, s, k in zip(fc.time.values, f_start, ok) if not k]
    keep = np.nonzero(ok)[0]
    if not keep.size:
        raise SmokeError("no forecast window has an observation with the identical window")
    p = xr.Dataset({"fc": fc.tp.isel(time=keep), "obs": al.rain.isel(time=keep)})
    p = p.assign_coords(valid_start=("time", f_start[keep]), valid_end=("time", f_end[keep]),
                        init_time=("time", fc.init_time.values[keep]),
                        obs_label=("time", al.source_date_label.values[keep] if "source_date_label" in al.coords
                                   else np.array(["explicit interval"] * keep.size)))
    return p, unpaired


# -------------------------------------------------------------------- metrics
def _mask(cfg: Config, paired: xr.Dataset) -> tuple[np.ndarray, str]:
    kind = cfg.smoke.get("mask", "india_outline" if cfg.obs_source == "imerg" else "obs_valid")
    if kind == "obs_valid":
        return np.ones((paired.sizes["lat"], paired.sizes["lon"]), bool), "cells with a valid observation"
    if kind == "india_outline":
        return india_mask(paired.lat.values, paired.lon.values), "rough India outline polygon (grid.INDIA_OUTLINE)"
    raise SmokeError(f"unknown smoke.mask {kind!r}")


def _block(f: np.ndarray, o: np.ndarray, thresholds) -> dict:
    out = {**continuous(f, o)}
    for t in thresholds:
        c = contingency(f, o, t)
        for k in ("pod", "far", "csi", "ets", "freq_bias", "hits", "misses", "false_alarms", "base_rate"):
            out[f"{k}_{t:g}"] = c[k]
    return out


def compute_metrics(cfg: Config, paired: xr.Dataset, base_mask: np.ndarray) -> dict:
    thr, scales = cfg.thresholds, cfg.fss_scales
    fc, ob = paired.fc.values, paired.obs.values
    valid = base_mask[None] & np.isfinite(fc) & np.isfinite(ob)
    daily = []
    for i in range(fc.shape[0]):
        m = valid[i]
        row = {"forecast_label": str(pd.Timestamp(paired.time.values[i]).date()),
               "valid_start": str(pd.Timestamp(paired.valid_start.values[i])),
               "valid_end": str(pd.Timestamp(paired.valid_end.values[i])),
               "obs_label": str(paired.obs_label.values[i]),
               "cells_used": int(m.sum()), "obs_missing_in_mask": int((base_mask & ~np.isfinite(ob[i])).sum())}
        if m.sum() > 2:
            row.update(_block(fc[i][m], ob[i][m], thr))
            for t in thr:
                for s in scales:
                    row[f"fss_{t:g}_s{s}"] = fss(fc[i:i + 1], ob[i:i + 1], m, t, s)
        daily.append(row)
    pooled_m = valid.all(0)                                    # cells valid on every day (for FSS)
    pooled = {"days": int(fc.shape[0]), "cells_used_total": int(valid.sum()),
              "cells_valid_all_days": int(pooled_m.sum()), **_block(fc[valid], ob[valid], thr)}
    for t in thr:
        for s in scales:
            pooled[f"fss_{t:g}_s{s}"] = fss(np.nan_to_num(fc), ob, pooled_m, t, s) if pooled_m.any() else float("nan")
    return {"daily": daily, "pooled": pooled}


# ----------------------------------------------------------------------- maps
def make_maps(paired: xr.Dataset, mask: np.ndarray, outdir: Path, title: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    m = xr.DataArray(mask, dims=("lat", "lon"))
    def panel(fc, ob, name, sub):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
        vmax = float(np.nanpercentile(np.concatenate([fc.values[np.isfinite(fc.values)], ob.values[np.isfinite(ob.values)]]), 99)) or 1.0
        fc.plot(ax=axs[0], cmap="Blues", vmin=0, vmax=vmax, cbar_kwargs={"label": "mm"}); axs[0].set_title("GFS forecast")
        ob.plot(ax=axs[1], cmap="Blues", vmin=0, vmax=vmax, cbar_kwargs={"label": "mm"}); axs[1].set_title("observation")
        err = fc - ob
        e = float(np.nanpercentile(np.abs(err.values), 99)) if np.isfinite(err.values).any() else 1.0
        err.plot(ax=axs[2], cmap="BrBG_r", vmin=-e, vmax=e, cbar_kwargs={"label": "forecast - obs (mm)"})
        axs[2].set_title("error")
        fig.suptitle(f"{title}\n{sub}", fontsize=9)
        fig.tight_layout(); p = outdir / name; fig.savefig(p, dpi=110); plt.close(fig)
        files.append(str(p))
    for i in range(paired.sizes["time"]):
        fc = paired.fc.isel(time=i).where(m); ob = paired.obs.isel(time=i).where(m)
        s = pd.Timestamp(paired.valid_start.values[i]); e = pd.Timestamp(paired.valid_end.values[i])
        panel(fc, ob, f"day_{s:%Y%m%d}_03Z.png", f"window {s:%Y-%m-%d %H:%MZ} -> {e:%Y-%m-%d %H:%MZ}; obs label {paired.obs_label.values[i]}")
    both = paired.obs.notnull().all("time")
    panel(paired.fc.sum("time").where(m & both), paired.obs.sum("time", skipna=False).where(m),
          "pooled_total.png", f"{paired.sizes['time']}-day total (cells missing on any day are blank)")
    return files


# --------------------------------------------------------------------- driver
def _md_table(rows: list[dict], cols: list[str]) -> str:
    def f(v):
        return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else ("" if isinstance(v, float) else str(v))
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + \
        "\n".join("| " + " | ".join(f(r.get(c, "")) for c in cols) + " |" for r in rows) + "\n"


def run_smoke(cfg: Config) -> dict:
    reject_legacy_keys(cfg.adapter_options)
    sm = cfg.smoke
    if not sm.get("start") or not sm.get("end"):
        raise SmokeError("smoke.start and smoke.end are required")
    if cfg.obs_source not in SUPPORTED_OBS:            # fail fast, before any download
        raise SmokeError(f"smoke test needs a real observation product {SUPPORTED_OBS}, not {cfg.obs_source!r}")
    labels = pd.date_range(sm["start"], sm["end"], freq="D")
    lead = int(sm.get("lead", 1))
    out = cfg.report_dir / "smoke"
    out.mkdir(parents=True, exist_ok=True)

    fc = load_gfs_daily(cfg, labels, lead)
    f_start, f_end = forecast_windows(fc)
    obs, meta, rejected = load_obs_daily(cfg, f_start, f_end)
    paired, unpaired = pair(fc, obs, rejected)
    mask, mask_desc = _mask(cfg, paired)
    metrics = compute_metrics(cfg, paired, mask)

    label = ROLE_LABEL[meta.product_role]
    comparison = f"GFS {fc.tp.attrs['cycle']} D+{lead} vs {meta.source_name} [{meta.product_role.value}]"
    tag = {"comparison": comparison, "obs_source_name": meta.source_name, "obs_product_role": meta.product_role.value,
           "obs_is_proxy": meta.is_proxy, "role_label": label}
    for r in metrics["daily"]:
        r.update(tag)
    metrics["pooled"].update(tag)
    maps = make_maps(paired, mask, out / "maps", f"{comparison} — {label}")

    summary = {
        "banner": [label, ALWAYS], "comparison": comparison,
        "observation_provenance": {k: (v.value if hasattr(v, "value") else v) for k, v in meta.__dict__.items()},
        "forecast_provenance": {k: fc.tp.attrs[k] for k in ("model", "product", "cycle", "construction_method", "qa_method")},
        "grid": {"role": grid_role(cfg.grid), "bounds": str(cfg.grid)}, "mask": mask_desc, "mask_cells": int(mask.sum()),
        "windows_requested": len(labels), "windows_paired": int(paired.sizes["time"]), "unpaired": unpaired,
        "imerg_rejected_windows": rejected, "metrics": metrics, "maps": maps,
        "phase_a_note": ("Real-data smoke test only if all inputs above are real files; "
                         "Phase A closes only with FINAL_TRUTH or OFFICIAL_ALTERNATIVE observations and a verified convention."),
    }
    (out / "smoke_metrics.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.DataFrame(metrics["daily"]).to_csv(out / "smoke_daily.csv", index=False)
    pd.DataFrame([metrics["pooled"]]).to_csv(out / "smoke_pooled.csv", index=False)
    paired.assign_attrs(**meta.to_attrs(), role_label=label, smoke_note=ALWAYS).to_netcdf(out / "smoke_paired.nc")

    pairs_rows = [{"forecast_label": r["forecast_label"], "valid_start": r["valid_start"], "valid_end": r["valid_end"],
                   "obs_label": r["obs_label"], "init": str(fc.init_time.sel(time=T(r["forecast_label"])).values),
                   "cells_used": r["cells_used"], "obs_missing_in_mask": r["obs_missing_in_mask"]} for r in metrics["daily"]]
    cont = ["forecast_label", "bias", "mae", "rmse", "corr", "pod_64.5", "far_64.5", "csi_64.5", "ets_64.5", "ets_15.6"]
    fss_cols = ["forecast_label"] + [f"fss_{t:g}_s{s}" for t in (15.6, 64.5) for s in cfg.fss_scales]
    md = [f"# Smoke test — {comparison}", "", f"> **{label}**", f"> {ALWAYS}", "",
          "## Observation provenance", "", _md_table([summary["observation_provenance"]], list(summary["observation_provenance"])),
          "## Forecast provenance", "", _md_table([summary["forecast_provenance"]], list(summary["forecast_provenance"])),
          f"Grid: {summary['grid']}. Mask: {mask_desc} ({int(mask.sum())} cells).", "",
          "## Pairing (exact valid_start / valid_end)", "", _md_table(pairs_rows, list(pairs_rows[0])),
          "### Unpaired windows", "", _md_table(unpaired, ["forecast_label", "valid_start", "reason"]) if unpaired else "none\n",
          "## Pooled metrics", "", _md_table([metrics["pooled"]], [c for c in cont if c != "forecast_label"] + ["cells_used_total"]),
          "## Daily metrics", "", _md_table(metrics["daily"], cont),
          "## FSS (daily; scales in 0.25° cells)", "", _md_table(metrics["daily"], fss_cols),
          "Pooled FSS: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics["pooled"].items() if k.startswith("fss_") and np.isfinite(v)), "",
          "## Maps", ""] + [f"- `{Path(p).name}`" for p in maps] + ["", f"_{summary['phase_a_note']}_", ""]
    (out / "smoke_report.md").write_text("\n".join(md))
    summary["outputs"] = [str(out / n) for n in ("smoke_report.md", "smoke_metrics.json", "smoke_daily.csv",
                                                  "smoke_pooled.csv", "smoke_paired.nc")]
    return summary
