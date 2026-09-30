"""Phase C: objective regime proxies and a pre-ML real-data error atlas."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

from .adapters.gfs import GFSAdapter
from .adapters.gfs_atmos import REGIME_FIELDS, decode_gfs_atmos_file
from .config import Config
from .data.align import FORECAST_LABEL_RULE, forecast_window
from .data.static_geo import load_real_static, require_real_static
from .grid import india_mask, region_of
from .regimes.engine import MONSOON_STATES, SYNOPTIC, RegimeCapabilityError, RegimeEngine, RegimeParams
from .verify.metrics import contingency, continuous


RAJEEVAN_URL = "https://repository.ias.ac.in/15911/"
IITM_URL = "https://www.tropmet.res.in/erpas/files/active_break_selection.php"
IMD_SYSTEM_URL = "https://rsmcnewdelhi.imd.gov.in/images/pdf/ogni.pdf"
WD_REFERENCE_URL = "https://doi.org/10.1029/2018GL077734"


class PhaseCError(RuntimeError):
    pass


def _load_atmosphere(cfg: Config, times: pd.DatetimeIndex, lead: int):
    adapter = GFSAdapter(cfg)
    snapshots, sources, checks = [], [], []
    bbox = [float(adapter.lat[0]), float(adapter.lat[-1]), float(adapter.lon[0]), float(adapter.lon[-1])]
    profile = cfg.phase_c.get("atmosphere_profile", "phase_b_full")
    if profile not in ("phase_b_full", "regime_required"):
        raise PhaseCError(f"unknown phase_c.atmosphere_profile {profile!r}")
    fetch = adapter._fetch_regime_atmos_file if profile == "regime_required" else adapter._fetch_atmos_file
    required = REGIME_FIELDS if profile == "regime_required" else None
    for label in times:
        window = forecast_window(label, lead, FORECAST_LABEL_RULE)
        for fxx in window.mean_hours:
            path = fetch(window.init, fxx)
            sample, facts = decode_gfs_atmos_file(
                path, window.init, fxx, bbox=bbox, required_fields=required
            )
            sample = sample.sel(lat=adapter.lat, lon=adapter.lon)
            valid_time = pd.Timestamp(facts["valid_time"])
            sample = sample.expand_dims(time=[valid_time]).assign_coords(
                forecast_label=("time", [pd.Timestamp(label).normalize()])
            )
            snapshots.append(sample)
            sources.append(str(path))
            checks.append(facts)
    return xr.concat(snapshots, dim="time"), sources, checks


def _daily_regimes(snapshot_regimes: xr.Dataset, labels: pd.DatetimeIndex) -> xr.Dataset:
    """Any synoptic proxy during a day; mean local forcing and diagnostics."""
    daily = []
    forecast_labels = pd.DatetimeIndex(snapshot_regimes.forecast_label.values)
    for label in labels:
        subset = snapshot_regimes.isel(time=np.nonzero(forecast_labels == label.normalize())[0])
        day = xr.Dataset(
            {
                "synoptic": subset.synoptic.max("time"),
                "dist_system_km": subset.dist_system_km.min("time"),
                "oro_score": subset.oro_score.mean("time"),
                "coast_score": subset.coast_score.mean("time"),
                "conv_score": subset.conv_score.mean("time"),
                "mslp_spatial_residual": subset.mslp_spatial_residual.mean("time"),
                "vo850": subset.vo850.mean("time"),
            }
        ).expand_dims(time=[label.normalize()])
        daily.append(day)
    out = xr.concat(daily, dim="time")
    for name in out.data_vars:
        out[name].attrs = snapshot_regimes[name].attrs
    out.attrs = {
        **snapshot_regimes.attrs,
        "daily_aggregation": (
            "synoptic=max class over four instantaneous samples; distance=min; "
            "local scores and diagnostics=mean of samples"
        ),
    }
    return out


def _grouped_fss(fc, obs, valid, eval_mask, threshold: float, scale: int) -> float:
    numerator = denominator = 0.0
    for t in range(fc.shape[0]):
        domain = valid[t]
        centres = eval_mask[t] & domain
        if not centres.any():
            continue
        normalizer = uniform_filter(domain.astype("float64"), scale, mode="constant")
        fc_event = ((fc[t] >= threshold) & domain).astype("float64")
        ob_event = ((obs[t] >= threshold) & domain).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            pf = np.where(normalizer > 0, uniform_filter(fc_event, scale, mode="constant") / normalizer, 0)
            po = np.where(normalizer > 0, uniform_filter(ob_event, scale, mode="constant") / normalizer, 0)
        numerator += float(np.sum((pf[centres] - po[centres]) ** 2))
        denominator += float(np.sum(pf[centres] ** 2 + po[centres] ** 2))
    return 1.0 - numerator / denominator if denominator > 0 else float("nan")


def _regime_masks(regimes: xr.Dataset, score_threshold: float, monsoon_available: bool):
    shape = regimes.synoptic.shape
    groups: list[tuple[str, str, np.ndarray, bool]] = []
    for code, name in enumerate(MONSOON_STATES):
        if monsoon_available:
            mask = np.broadcast_to((regimes.monsoon_state.values == code)[:, None, None], shape)
        else:
            mask = np.zeros(shape, dtype=bool)
        groups.append(("monsoon_state", name, mask, monsoon_available))
    for code, name in enumerate(SYNOPTIC):
        groups.append(("synoptic", name, regimes.synoptic.values == code, True))
    score_masks = {
        "orographic": regimes.oro_score.values >= score_threshold,
        "coastal": regimes.coast_score.values >= score_threshold,
        "convective": regimes.conv_score.values >= score_threshold,
    }
    for name, mask in score_masks.items():
        groups.append(("local_forcing", name, mask, True))
    groups.append(("local_forcing", "none", ~np.logical_or.reduce(list(score_masks.values())), True))
    return groups


def build_atlas_table(cfg: Config, paired: xr.Dataset, regimes: xr.Dataset, base_mask: np.ndarray,
                      monsoon_available: bool) -> pd.DataFrame:
    opts = cfg.phase_c
    fc, obs = paired.fc.values, paired.obs.values
    valid = base_mask[None] & np.isfinite(fc) & np.isfinite(obs)
    times = pd.DatetimeIndex(paired.time.values)
    lat2, lon2 = np.meshgrid(paired.lat.values, paired.lon.values, indexing="ij")
    regions = region_of(lat2, lon2)
    region_names = ["all", *sorted(set(regions[base_mask].tolist()))]
    months = sorted(set(times.to_period("M").astype(str)))
    groups = _regime_masks(regimes, float(opts.get("local_score_threshold", 0.5)), monsoon_available)
    rows = []

    for axis, regime, regime_mask, available in groups:
        for region in region_names:
            region_mask = base_mask if region == "all" else base_mask & (regions == region)
            for month in months:
                month_mask = np.asarray(times.to_period("M").astype(str) == month)
                selected = regime_mask & region_mask[None] & month_mask[:, None, None] & valid
                n_cells = int(selected.sum())
                n_days = int(np.sum(np.any(selected, axis=(1, 2))))
                if n_cells:
                    f, o = fc[selected], obs[selected]
                    cont = continuous(f, o)
                else:
                    f = o = np.array([], dtype="float32")
                    cont = {key: float("nan") for key in ("bias", "mae", "rmse", "corr", "mean_fc", "mean_obs")}
                for threshold in cfg.thresholds:
                    cat = contingency(f, o, float(threshold)) if n_cells else {
                        key: (0 if key in ("hits", "false_alarms", "misses", "correct_neg") else float("nan"))
                        for key in ("hits", "false_alarms", "misses", "correct_neg", "pod", "far", "csi", "ets",
                                    "freq_bias", "base_rate")
                    }
                    obs_events = int(np.sum(o >= threshold)) if n_cells else 0
                    warnings = []
                    if not available:
                        warnings.append("classification unavailable")
                    if n_days < int(opts.get("min_group_days", 30)):
                        warnings.append(f"only {n_days} independent days")
                    if n_cells < int(opts.get("min_group_cells", 1000)):
                        warnings.append(f"only {n_cells} space-time cells")
                    if obs_events < int(opts.get("min_events", 20)):
                        warnings.append(f"only {obs_events} observed events")
                    row = {
                        "axis": axis, "regime": regime, "region": region, "month": month,
                        "threshold_mm": float(threshold), "classification_available": available,
                        "n_cells": n_cells, "n_days": n_days, "obs_events": obs_events,
                        **{key: cont[key] for key in ("bias", "mae", "rmse", "corr", "mean_fc", "mean_obs")},
                        **{key: cat[key] for key in ("hits", "false_alarms", "misses", "correct_neg",
                                                     "pod", "far", "csi", "ets", "freq_bias", "base_rate")},
                        "low_sample": bool(warnings), "warning": "; ".join(warnings),
                    }
                    for scale in cfg.fss_scales:
                        row[f"fss_s{scale}"] = _grouped_fss(
                            fc, obs, valid & region_mask[None] & month_mask[:, None, None], selected,
                            float(threshold), int(scale),
                        ) if n_cells else float("nan")
                    rows.append(row)
    return pd.DataFrame(rows)


def _atlas_maps(regimes: xr.Dataset, paired: xr.Dataset, base_mask: np.ndarray, path: Path) -> None:
    groups = [("synoptic", name, regimes.synoptic.values == code) for code, name in enumerate(SYNOPTIC)]
    local_vars = {"orographic": "oro_score", "coastal": "coast_score", "convective": "conv_score"}
    groups += [("local", name, regimes[var].values >= 0.5) for name, var in local_vars.items()]
    bias, count, labels = [], [], []
    error = paired.fc.values - paired.obs.values
    for axis, name, mask in groups:
        selected = mask & base_mask[None] & np.isfinite(error)
        n = selected.sum(0)
        total = np.where(selected, error, 0).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bias.append(np.where(n > 0, total / n, np.nan))
        count.append(n)
        labels.append(f"{axis}:{name}")
    xr.Dataset(
        {"bias": (("regime", "lat", "lon"), np.asarray(bias, "float32")),
         "count": (("regime", "lat", "lon"), np.asarray(count, "int16"))},
        coords={"regime": labels, "lat": paired.lat, "lon": paired.lon},
        attrs={"warning": (
            f"{paired.sizes['time']}-day exploratory atlas using the declared observation role; "
            "not a final skill estimate"
        )},
    ).to_netcdf(path)


def _plot_regimes(regimes: xr.Dataset, base_mask: np.ndarray, path: Path) -> None:
    indices = sorted(set((0, regimes.sizes["time"] // 2, regimes.sizes["time"] - 1)))
    fig, axes = plt.subplots(2, len(indices), figsize=(5 * len(indices), 8), constrained_layout=True, squeeze=False)
    syn_cmap = ListedColormap(["#eeeeee", "#3f8fc5", "#c63d2f"])
    local_cmap = ListedColormap(["#eeeeee", "#2f7d32", "#166d91", "#d88c16"])
    scores = np.stack((regimes.oro_score.values, regimes.coast_score.values, regimes.conv_score.values), axis=0)
    dominant = np.argmax(scores, axis=0) + 1
    dominant[np.max(scores, axis=0) < 0.5] = 0
    for col, index in enumerate(indices):
        date = str(pd.Timestamp(regimes.time.values[index]).date())
        synoptic = np.where(base_mask, regimes.synoptic[index].values, np.nan)
        local = np.where(base_mask, dominant[index], np.nan)
        a = axes[0, col].pcolormesh(regimes.lon, regimes.lat, synoptic, shading="auto",
                                    cmap=syn_cmap, vmin=-0.5, vmax=2.5)
        axes[0, col].set_title(f"{date}: synoptic proxy")
        b = axes[1, col].pcolormesh(regimes.lon, regimes.lat, local, shading="auto",
                                    cmap=local_cmap, vmin=-0.5, vmax=3.5)
        axes[1, col].set_title(f"{date}: dominant local score >= 0.5")
        for ax in axes[:, col]:
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.colorbar(a, ax=axes[0], ticks=[0, 1, 2], label="0 none, 1 low proxy, 2 depression proxy")
    fig.colorbar(b, ax=axes[1], ticks=[0, 1, 2, 3], label="0 none, 1 orographic, 2 coastal, 3 convective")
    fig.suptitle("Phase C objective regime proxies")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_atlas(table: pd.DataFrame, bias_path: Path, skill_path: Path) -> None:
    data = table[(table.region == "all") & (table.classification_available)]
    data = data[data.threshold_mm == 64.5].copy()
    data["label"] = data.axis + ":" + data.regime
    data = data[data.n_cells > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(data.label, data.bias, color=np.where(data.bias >= 0, "#b9473f", "#397ba8"))
    axes[0].axvline(0, color="black", linewidth=0.8); axes[0].set_xlabel("Bias (mm/day)")
    axes[1].barh(data.label, data.rmse, color="#5a7d55"); axes[1].set_xlabel("RMSE (mm/day)")
    period = ", ".join(sorted(data.month.unique()))
    fig.suptitle(f"Exploratory Regime x Error Atlas (all regions, {period})")
    fig.savefig(bias_path, dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(data.label, data.csi.fillna(0), color="#4b82a8"); axes[0].set_xlabel("CSI at 64.5 mm/day")
    fss_col = "fss_s5" if "fss_s5" in data else next(c for c in data if c.startswith("fss_s"))
    axes[1].barh(data.label, data[fss_col].fillna(0), color="#a96d31"); axes[1].set_xlabel(f"{fss_col} at 64.5 mm/day")
    fig.suptitle("Categorical and spatial scores (see sample gates in the C2 report)")
    fig.savefig(skill_path, dpi=150); plt.close(fig)


def _definitions(params: RegimeParams) -> str:
    return f"""# Phase C regime definitions and threshold audit

No output from this engine is an official IMD event declaration.

| Axis | Previous implementation | Phase C definition | Status |
|---|---|---|---|
| Active / break | CMZ box 18-28 deg N, 69-88 deg E; one residual standard deviation; any climatology length | Area-weighted land rainfall over 18-28 deg N, 65-88 deg E; day-of-year 31-day-smoothed mean and standard deviation; `|z| >= {params.z_thresh}` for at least `{params.min_run_days}` consecutive July/August days; at least `{params.min_climatology_years}` years required | Published-style objective proxy, not official |
| Low | Gaussian MSLP departure <= -1.5 hPa and positive 850-vorticity | At any of four validated six-hourly snapshots: local minimum; MSLP spatial residual <= `-{params.low_spatial_residual_hpa}` hPa; complete {params.closed_ring_radius_deg:g} deg ring depth >= `{params.low_closed_depth_hpa}` hPa; 850-vorticity >= `{params.low_vorticity_1e5:g}e-5 s-1` | Objective proxy |
| Depression | MSLP departure <= -3 hPa | At any snapshot: low criteria plus ring depth >= `{params.depression_closed_depth_hpa}` hPa, 850-vorticity >= `{params.depression_vorticity_1e5:g}e-5 s-1`, and maximum 850-wind >= `{params.depression_wind_ms}` m/s | Objective proxy; not IMD intensity |
| Western disturbance | Mentioned but not detected | Disabled: daily 500-hPa fields alone do not provide a validated upstream track or 350-450-hPa vorticity-anomaly climatology | Unsupported |
| Orographic | Positive terrain-following vertical-motion proxy / 0.12 m/s | Same formula with real ETOPO slope/aspect; score 1 at `{params.oro_ref}` m/s | Heuristic score |
| Coastal | Onshore wind / 8 m/s with 150-km decay | Same formula with ETOPO-derived coast geometry | Heuristic score |
| Convective | `(CAPE/2500) * (PWAT/60)`, clipped | Same formula: CAPE scale `{params.convective_cape_ref:g}` J/kg and PWAT scale `{params.convective_pwat_ref:g}` kg/m^2 | Heuristic score |

`mslp_anom` has been renamed `mslp_spatial_residual` in the regime engine because it is
MSLP minus an 8-cell Gaussian spatial background, not a climatological anomaly. Phase B's
validated artifact names remain unchanged as required.

References: [Rajeevan et al. 2010]({RAJEEVAN_URL}), [IITM active/break criteria]({IITM_URL}),
[IMD low-pressure-system discussion]({IMD_SYSTEM_URL}), [WD vertical structure]({WD_REFERENCE_URL}).
"""


def _capability_markdown(capability: dict) -> str:
    rows = "\n".join(
        f"| {name} | {item['available']} | {item.get('detected', '')} | {item['kind']} | {item['reason']} |"
        for name, item in capability["capabilities"].items()
    )
    return f"""# Phase C capability report

**Status: {capability['status']}**<br>
**Move to regime-aware ML: {str(capability['move_to_ml']).upper()}**

| Capability | Available | Detected | Classification type | Reason |
|---|---:|---:|---|---|
{rows}

## Data

- Paired days: `{capability['data']['paired_days']}`
- Months: `{', '.join(capability['data']['months'])}`
- Observation role: `{capability['data']['observation_role']}`
- Validated GFS atmosphere files: `{capability['data']['gfs_atmosphere_files']}`
- ETOPO synthetic flag: `{capability['data']['static_is_synthetic']}`

## Decision

{capability['decision']}
"""


def run_phase_c(cfg: Config) -> dict:
    opts = cfg.phase_c
    paired_path = Path(opts.get("paired_path", ""))
    if not paired_path.exists():
        raise PhaseCError(f"paired real forecast/observation file missing: {paired_path}")
    paired = xr.open_dataset(paired_path).load()
    if paired.attrs.get("obs_meta_product_role") not in ("FINAL_TRUTH", "OFFICIAL_ALTERNATIVE", "SMOKE_TEST_PROXY"):
        raise PhaseCError("paired file lacks real observation-product provenance")
    if paired.fc.attrs.get("construction_method", "").find("APCP(0-27) - APCP(0-3)") < 0:
        raise PhaseCError("paired forecast does not carry the frozen GFS precipitation construction")

    static = require_real_static(load_real_static(cfg))
    if not (np.array_equal(paired.lat, static.lat) and np.array_equal(paired.lon, static.lon)):
        raise PhaseCError("paired, atmospheric, and static grids must match exactly")
    times = pd.DatetimeIndex(paired.time.values)
    atmosphere, source_files, validation = _load_atmosphere(cfg, times, int(opts.get("lead", 1)))
    if not (np.array_equal(paired.lat, atmosphere.lat) and np.array_equal(paired.lon, atmosphere.lon)):
        raise PhaseCError("paired and validated atmospheric grids differ")

    engine = RegimeEngine()
    fields = {name: atmosphere[name] for name in ("u850", "v850", "mslp", "cape", "tcwv")}
    snapshot_regimes = engine.transform_atmosphere(fields, static)
    snapshot_regimes = snapshot_regimes.assign_coords(forecast_label=atmosphere.forecast_label)
    regimes = _daily_regimes(snapshot_regimes, times)
    climatology_path = opts.get("climatology_path")
    monsoon_available = False
    monsoon_reason = "no >=20-year daily land-rainfall climatology supplied"
    regimes["monsoon_state"] = (("time",), np.full(len(times), -1, "int8"))
    if climatology_path:
        clim = xr.open_dataset(climatology_path).load()
        rain_name = "rain" if "rain" in clim else next(iter(clim.data_vars))
        engine.fit(clim[rain_name], static.land)
        z, signed_state = engine.classify_monsoon(paired.obs, static.land, run_filter=True)
        regimes["monsoon_state"] = (("time",), (signed_state + 1).astype("int8"))
        regimes["monsoon_z"] = (("time",), z.astype("float32"))
        monsoon_available = True
        monsoon_reason = "validated climatology supplied"
    regimes.monsoon_state.attrs.update(classes="-1:not_assessed,0:break,1:normal,2:active",
                                       classification_kind="objective proxy, not official IMD designation")

    base_mask = india_mask(paired.lat.values, paired.lon.values) & static.land.values
    atlas = build_atlas_table(cfg, paired, regimes, base_mask, monsoon_available)
    out = Path(opts.get("out_dir", cfg.report_dir / "phase_c"))
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    atlas_path = out / "regime_error_atlas.csv"
    regimes_path = out / "phase_c_regimes.nc"
    map_path = out / "regime_error_maps.nc"
    systems_path = out / "synoptic_proxy_centres.csv"
    detection_path = out / "synoptic_detection_audit.csv"
    definitions_path = out / "regime_definitions.md"
    capability_json = out / "capability_report.json"
    capability_md = out / "capability_report.md"
    report_path = out / "phase_c_report.md"
    regime_figure = figures / "regime_maps.png"
    bias_figure = figures / "atlas_bias_rmse.png"
    skill_figure = figures / "atlas_csi_fss.png"

    atlas.to_csv(atlas_path, index=False)
    regimes.attrs.update(
        source="validated NOAA GFS pgrb2.0p25 daily atmospheric state",
        source_files=";".join(source_files), static_source=static.attrs["static_source"],
        observation_role=paired.attrs["obs_meta_product_role"], monsoon_state_reason=monsoon_reason,
        correction_ml_trained="false",
    )
    regimes.to_netcdf(regimes_path)
    system_columns = ["time", "lat", "lon", "class", "mslp_hpa", "mslp_spatial_residual_hpa",
                      "closed_ring_depth_hpa", "vorticity_1e5_s-1", "max_850_wind_ms"]
    pd.DataFrame(engine.last_systems, columns=system_columns).to_csv(systems_path, index=False)
    pd.DataFrame(engine.last_detection_audit).to_csv(detection_path, index=False)
    _atlas_maps(regimes, paired, base_mask, map_path)
    _plot_regimes(regimes, base_mask, regime_figure)
    _plot_atlas(atlas, bias_figure, skill_figure)
    definitions_path.write_text(_definitions(engine.p), encoding="utf-8")

    role = paired.attrs["obs_meta_product_role"]
    months = sorted(set(times.to_period("M").astype(str)))
    system_counts = pd.Series([item["class"] for item in engine.last_systems], dtype="string").value_counts()
    move_to_ml = False
    decision_reasons = []
    if len(times) < 90:
        decision_reasons.append(f"Only {len(times)} independent paired days are available")
    if len(months) < 3:
        decision_reasons.append(f"Only {len(months)} calendar month(s) are represented")
    if role != "FINAL_TRUTH":
        decision_reasons.append(f"The observation role is {role}, not final IMD truth")
    if not monsoon_available:
        decision_reasons.append(
            "Active/normal/break cannot be assessed without a multi-decadal daily rainfall climatology"
        )
    decision = (
        "Do not move to regime-aware correction ML. " + "; ".join(decision_reasons) + "."
        if decision_reasons else
        "Phase C capability gates are satisfied, but correction ML remains outside this command."
    )
    capability = {
        "status": "PASS_WITH_DATA_LIMITATIONS",
        "move_to_ml": move_to_ml,
        "data": {
            "paired_days": len(times), "atmospheric_snapshots": int(atmosphere.sizes["time"]),
            "months": months, "observation_role": role,
            "gfs_atmosphere_files": len(source_files),
            "gfs_messages_validated": sum(len(item["fields"]) for item in validation),
            "static_source": static.attrs["static_source"], "static_is_synthetic": static.attrs["is_synthetic"],
            "atlas_rows": len(atlas), "all_rows_low_sample": bool(atlas.low_sample.all()),
        },
        "capabilities": {
            "monsoon_active_normal_break": {"available": monsoon_available, "kind": "published-style objective proxy",
                                               "reason": monsoon_reason},
            "low_proxy": {"available": True, "kind": "objective NWP proxy", "detected": int(system_counts.get("low_proxy", 0)),
                          "reason": "MSLP closure + residual + 850-vorticity"},
            "depression_proxy": {"available": True, "kind": "objective NWP proxy",
                                 "detected": int(system_counts.get("depression_proxy", 0)),
                                 "reason": "adds 850-wind and stronger closure/vorticity; not official intensity"},
            "wd_proxy": {"available": False, "kind": "unsupported", "reason": regimes.attrs["wd_proxy_reason"]},
            "orographic_score": {"available": True, "kind": "heuristic score", "reason": "validated 850-wind plus ETOPO terrain gradient"},
            "coastal_score": {"available": True, "kind": "heuristic score", "reason": "validated 850-wind plus ETOPO coast geometry"},
            "convective_score": {"available": True, "kind": "heuristic score", "reason": "validated CAPE and PWAT"},
            "error_atlas": {"available": True, "kind": "exploratory", "reason": f"{len(times)} paired days against {role}"},
        },
        "thresholds": asdict(engine.p),
        "decision": decision,
        "protected_logic": {"gfs_precipitation": "unchanged", "observation_pipeline": "unchanged",
                            "phase_b_validation": "unchanged", "correction_ml_trained": False},
    }
    capability_json.write_text(json.dumps(capability, indent=2), encoding="utf-8")
    capability_md.write_text(_capability_markdown(capability), encoding="utf-8")
    report_path.write_text(
        "# Phase C report\n\n"
        f"**Status: {capability['status']}**\n\n"
        f"Real paired data: {len(times)} days ({times[0].date()} through {times[-1].date()}), "
        f"GFS D+1 versus {role}. Validated atmospheric files: {len(source_files)}. "
        f"Static source: {static.attrs['static_source']}.\n\n"
        f"Synoptic proxy centres detected: {len(engine.last_systems)}. Atlas rows: {len(atlas)}; "
        f"low-sample rows: {int(atlas.low_sample.sum())}.\n\n"
        "## ML readiness\n\n"
        f"**NO.** {capability['decision']}\n\n"
        "See `regime_definitions.md`, `capability_report.md`, and `regime_error_atlas.csv` for details.\n",
        encoding="utf-8",
    )
    outputs = [report_path, definitions_path, capability_md, capability_json, atlas_path, regimes_path,
               map_path, systems_path, detection_path, regime_figure, bias_figure, skill_figure]
    return {"status": capability["status"], "move_to_ml": move_to_ml,
            "outputs": [str(path) for path in outputs], "capability": capability}
