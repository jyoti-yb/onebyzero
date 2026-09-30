"""Phase C2: expanded real-data atlas and clustered error-distribution evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mannwhitneyu

from .config import Config
from .data.static_geo import load_real_static, require_real_static
from .grid import india_mask
from .phase_c import PhaseCError, run_phase_c
from .regimes.engine import SYNOPTIC


def _sufficient(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype="float64")
    return {
        "n": int(values.size),
        "sum": float(values.sum()),
        "sumsq": float(np.square(values).sum()),
        "abs_sum": float(np.abs(values).sum()),
    }


def _combine(records: list[dict[str, float]], indices=None) -> dict[str, float]:
    if indices is None:
        indices = range(len(records))
    selected = [records[int(i)] for i in indices]
    return {key: sum(row[key] for row in selected) for key in ("n", "sum", "sumsq", "abs_sum")}


def _hedges_g(target: dict[str, float], reference: dict[str, float]) -> float:
    n1, n0 = int(target["n"]), int(reference["n"])
    if n1 < 2 or n0 < 2:
        return float("nan")
    v1 = max(0.0, (target["sumsq"] - target["sum"] ** 2 / n1) / (n1 - 1))
    v0 = max(0.0, (reference["sumsq"] - reference["sum"] ** 2 / n0) / (n0 - 1))
    df = n1 + n0 - 2
    pooled = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / df)
    if pooled == 0:
        return 0.0 if target["sum"] / n1 == reference["sum"] / n0 else float("nan")
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(correction * ((target["sum"] / n1) - (reference["sum"] / n0)) / pooled)


def _effect_values(target: dict[str, float], reference: dict[str, float]) -> tuple[float, float, float]:
    mean_diff = target["sum"] / target["n"] - reference["sum"] / reference["n"]
    mae_diff = target["abs_sum"] / target["n"] - reference["abs_sum"] / reference["n"]
    return float(mean_diff), float(mae_diff), _hedges_g(target, reference)


def _interval(values: list[float], confidence: float) -> tuple[float, float]:
    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.quantile(clean, [alpha, 1.0 - alpha]))


def compare_synoptic_errors(paired: xr.Dataset, regimes: xr.Dataset, base_mask: np.ndarray,
                             options: dict) -> pd.DataFrame:
    """Compare proxy errors with no-synoptic errors on matched forecast days.

    Confidence intervals use a forecast-day block bootstrap. Grid cells contribute to
    the effect estimate but are never treated as independent bootstrap replicates.
    """
    fc, obs = paired.fc.values, paired.obs.values
    error = fc - obs
    valid = base_mask[None] & np.isfinite(fc) & np.isfinite(obs)
    reps = int(options.get("bootstrap_replicates", 1000))
    confidence = float(options.get("confidence", 0.95))
    min_days = int(options.get("min_comparison_days", 10))
    min_cells = int(options.get("min_comparison_cells", 5000))
    g_threshold = float(options.get("material_hedges_g", 0.2))
    mae_threshold = float(options.get("material_mae_mm", 1.0))
    max_cliff_cells = int(options.get("max_cliffs_cells", 100000))
    rng = np.random.default_rng(int(options.get("bootstrap_seed", 42)))
    rows = []

    for code, name in enumerate(SYNOPTIC[1:], start=1):
        target_days, reference_days = [], []
        target_values, reference_values = [], []
        matched_dates = []
        for day in range(error.shape[0]):
            target_mask = valid[day] & (regimes.synoptic.values[day] == code)
            reference_mask = valid[day] & (regimes.synoptic.values[day] == 0)
            if not target_mask.any() or not reference_mask.any():
                continue
            tv, rv = error[day][target_mask], error[day][reference_mask]
            target_days.append(_sufficient(tv))
            reference_days.append(_sufficient(rv))
            target_values.append(np.asarray(tv, dtype="float64"))
            reference_values.append(np.asarray(rv, dtype="float64"))
            matched_dates.append(str(pd.Timestamp(paired.time.values[day]).date()))

        matched = len(target_days)
        if matched:
            target = _combine(target_days)
            reference = _combine(reference_days)
            mean_diff, mae_diff, hedges_g = _effect_values(target, reference)
            boot_mean, boot_mae, boot_g = [], [], []
            for _ in range(reps):
                indices = rng.integers(0, matched, matched)
                values = _effect_values(_combine(target_days, indices), _combine(reference_days, indices))
                boot_mean.append(values[0])
                boot_mae.append(values[1])
                boot_g.append(values[2])
            mean_lo, mean_hi = _interval(boot_mean, confidence)
            mae_lo, mae_hi = _interval(boot_mae, confidence)
            g_lo, g_hi = _interval(boot_g, confidence)
            if matched < 2:
                mean_lo = mean_hi = mae_lo = mae_hi = g_lo = g_hi = float("nan")
            tv, rv = np.concatenate(target_values), np.concatenate(reference_values)
            cliff_sampled = len(tv) > max_cliff_cells or len(rv) > max_cliff_cells
            if len(tv) > max_cliff_cells:
                tv = rng.choice(tv, max_cliff_cells, replace=False)
            if len(rv) > max_cliff_cells:
                rv = rng.choice(rv, max_cliff_cells, replace=False)
            u = mannwhitneyu(tv, rv, alternative="two-sided").statistic
            cliffs_delta = float(2.0 * u / (len(tv) * len(rv)) - 1.0)
        else:
            target = reference = {"n": 0}
            mean_diff = mae_diff = hedges_g = cliffs_delta = float("nan")
            mean_lo = mean_hi = mae_lo = mae_hi = g_lo = g_hi = float("nan")
            cliff_sampled = False

        usable = matched >= min_days and target["n"] >= min_cells and reference["n"] >= min_cells
        g_ci_excludes_zero = bool(np.isfinite(g_lo) and (g_lo > 0 or g_hi < 0))
        mae_ci_excludes_zero = bool(np.isfinite(mae_lo) and (mae_lo > 0 or mae_hi < 0))
        material = bool(usable and g_ci_excludes_zero and abs(hedges_g) >= g_threshold
                        and mae_ci_excludes_zero and abs(mae_diff) >= mae_threshold)
        rows.append({
            "comparison": f"{name}_vs_none", "target_regime": name, "reference_regime": "none",
            "matched_days": matched, "matched_dates": ";".join(matched_dates),
            "target_cells": int(target["n"]), "reference_cells": int(reference["n"]),
            "mean_error_difference_mm": mean_diff, "mean_error_ci_low": mean_lo, "mean_error_ci_high": mean_hi,
            "mae_difference_mm": mae_diff, "mae_ci_low": mae_lo, "mae_ci_high": mae_hi,
            "hedges_g": hedges_g, "hedges_g_ci_low": g_lo, "hedges_g_ci_high": g_hi,
            "cliffs_delta": cliffs_delta, "cliffs_delta_sampled": cliff_sampled,
            "confidence": confidence, "bootstrap_unit": "forecast_day", "bootstrap_replicates": reps,
            "statistically_usable": usable, "material_difference": material,
            "warning": "" if usable else f"needs >= {min_days} matched days and >= {min_cells} cells per group",
        })
    return pd.DataFrame(rows)


def _regime_counts(regimes: xr.Dataset, base_mask: np.ndarray, score_threshold: float) -> pd.DataFrame:
    rows = []
    valid = np.broadcast_to(base_mask[None], regimes.synoptic.shape)
    for code, name in enumerate(SYNOPTIC):
        selected = valid & (regimes.synoptic.values == code)
        rows.append({"axis": "synoptic", "regime": name,
                     "days_present": int(np.any(selected, axis=(1, 2)).sum()),
                     "cell_samples": int(selected.sum())})
    local_masks = []
    for name, variable in (("orographic", "oro_score"), ("coastal", "coast_score"),
                           ("convective", "conv_score")):
        selected = valid & (regimes[variable].values >= score_threshold)
        local_masks.append(selected)
        rows.append({"axis": "local_forcing", "regime": name,
                     "days_present": int(np.any(selected, axis=(1, 2)).sum()),
                     "cell_samples": int(selected.sum())})
    local_none = valid & ~np.logical_or.reduce(local_masks)
    rows.append({"axis": "local_forcing", "regime": "none",
                 "days_present": int(np.any(local_none, axis=(1, 2)).sum()),
                 "cell_samples": int(local_none.sum())})
    rows.extend({"axis": "monsoon_state", "regime": name, "days_present": 0, "cell_samples": 0,
                 "classification_available": False}
                for name in ("break", "normal", "active"))
    result = pd.DataFrame(rows)
    result["classification_available"] = result.classification_available.fillna(True).astype(bool)
    return result


def _decision(days: int, months: int, role: str, effects: pd.DataFrame,
              usable_atlas_rows: int, options: dict) -> tuple[bool, list[str]]:
    reasons = []
    min_days = int(options.get("min_ml_days", 90))
    min_months = int(options.get("min_ml_months", 3))
    if days < min_days:
        reasons.append(f"{days} paired days are below the {min_days}-day gate")
    if months < min_months:
        reasons.append(f"{months} month(s) are below the {min_months}-month gate")
    low = effects[effects.target_regime == "low_proxy"]
    if low.empty or not bool(low.iloc[0].material_difference):
        reasons.append("the low-proxy comparison does not yet show a statistically usable material difference")
    if usable_atlas_rows == 0:
        reasons.append("no atlas row meets the configured minimum-sample requirements")
    if role != "FINAL_TRUTH":
        reasons.append(f"{role} is proxy-based exploration, not final IMD truth")
    return not reasons, reasons


def run_phase_c2(cfg: Config) -> dict:
    if not cfg.phase_c2:
        raise PhaseCError("phase_c2 configuration is required")
    phase_c_result = run_phase_c(cfg)
    phase_c_out = Path(cfg.phase_c.get("out_dir", cfg.report_dir / "phase_c"))
    paired = xr.open_dataset(cfg.phase_c["paired_path"]).load()
    regimes = xr.open_dataset(phase_c_out / "phase_c_regimes.nc").load()
    static = require_real_static(load_real_static(cfg))
    base_mask = india_mask(paired.lat.values, paired.lon.values) & static.land.values
    atlas = pd.read_csv(phase_c_out / "regime_error_atlas.csv")
    atlas["meets_minimum_sample"] = atlas.classification_available.eq(True) & ~atlas.low_sample.eq(True)
    effects = compare_synoptic_errors(paired, regimes, base_mask, cfg.phase_c2)
    score_threshold = float(cfg.phase_c.get("local_score_threshold", 0.5))
    counts = _regime_counts(regimes, base_mask, score_threshold)
    usable_groups = (
        atlas.groupby(["axis", "regime"], sort=False).meets_minimum_sample
        .agg(usable_rows="sum", total_rows="size").reset_index()
    )

    out = Path(cfg.phase_c2.get("out_dir", cfg.report_dir / "phase_c2"))
    out.mkdir(parents=True, exist_ok=True)
    sample_path = out / "atlas_sample_counts.csv"
    effects_path = out / "synoptic_error_effects.csv"
    counts_path = out / "regime_counts.csv"
    usable_groups_path = out / "usable_atlas_groups.csv"
    report_path = out / "phase_c2_report.md"
    json_path = out / "phase_c2_summary.json"
    atlas.to_csv(sample_path, index=False)
    effects.to_csv(effects_path, index=False)
    counts.to_csv(counts_path, index=False)
    usable_groups.to_csv(usable_groups_path, index=False)

    role = paired.attrs["obs_meta_product_role"]
    months = len(set(pd.DatetimeIndex(paired.time.values).to_period("M")))
    usable_rows = int(atlas.meets_minimum_sample.sum())
    move_to_ml, reasons = _decision(
        paired.sizes["time"], months, role, effects, usable_rows, cfg.phase_c2
    )
    summary = {
        "status": "PASS" if move_to_ml else "PASS_WITH_DATA_LIMITATIONS",
        "move_to_ml": move_to_ml,
        "correction_ml_trained": False,
        "paired_days": int(paired.sizes["time"]), "months": months, "observation_role": role,
        "atlas_rows": int(len(atlas)), "statistically_usable_atlas_rows": usable_rows,
        "minimum_sample_requirements": {
            "days": int(cfg.phase_c.get("min_group_days", 30)),
            "cells": int(cfg.phase_c.get("min_group_cells", 1000)),
            "observed_events": int(cfg.phase_c.get("min_events", 20)),
        },
        "regime_counts": counts.to_dict(orient="records"),
        "usable_atlas_groups": usable_groups.to_dict(orient="records"),
        "synoptic_effects": effects.drop(columns="matched_dates").to_dict(orient="records"),
        "decision_reasons": reasons,
        "phase_c_status": phase_c_result["status"],
        "capability_gates": {"active_normal_break": "disabled without >=20-year climatology",
                             "wd": "disabled without validated upstream track and 350-450 hPa climatology"},
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    count_lines = "\n".join(
        f"| {r.axis} | {r.regime} | {r.days_present} | {r.cell_samples} | {r.classification_available} |"
        for r in counts.itertuples()
    )
    threshold_rows = atlas[(atlas.axis == "synoptic") & (atlas.region == "all")]
    threshold_lines = "\n".join(
        f"| {r.regime} | {r.threshold_mm:g} | {r.n_days} | {r.n_cells} | {r.obs_events} | "
        f"{r.meets_minimum_sample} |"
        for r in threshold_rows.itertuples()
    )
    usable_lines = "\n".join(
        f"| {r.axis} | {r.regime} | {r.usable_rows} | {r.total_rows} |"
        for r in usable_groups.itertuples()
    )

    def fmt(value, digits):
        return f"{value:.{digits}f}" if np.isfinite(value) else "not estimable"

    effects_lines = "\n".join(
        f"| {r.target_regime} | {r.matched_days} | {r.target_cells} | {fmt(r.mean_error_difference_mm, 2)} "
        f"[{fmt(r.mean_error_ci_low, 2)}, {fmt(r.mean_error_ci_high, 2)}] | {fmt(r.mae_difference_mm, 2)} "
        f"[{fmt(r.mae_ci_low, 2)}, {fmt(r.mae_ci_high, 2)}] | {fmt(r.hedges_g, 3)} "
        f"[{fmt(r.hedges_g_ci_low, 3)}, {fmt(r.hedges_g_ci_high, 3)}] | {fmt(r.cliffs_delta, 3)} | "
        f"{r.statistically_usable} | {r.material_difference} |"
        for r in effects.itertuples()
    )
    report_path.write_text(
        "# Phase C2 expanded regime evidence report\n\n"
        f"**Status: {summary['status']}**  \n"
        f"**Proceed to regime-aware correction ML: {str(move_to_ml).upper()}**  \n"
        "**Correction ML trained: FALSE**\n\n"
        f"Paired real-data days: `{summary['paired_days']}` across `{months}` month(s). Observation role: `{role}`. "
        f"Statistically usable atlas rows: `{usable_rows}` / `{len(atlas)}`.\n\n"
        "## Regime counts\n\n| Axis | Regime | Days present | Cell samples | Available |\n|---|---|---:|---:|---:|\n"
        f"{count_lines}\n\n"
        "## Synoptic threshold and event counts\n\n"
        "| Regime | Threshold mm/day | Days | Cell samples | Observed events | Meets minimum |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"{threshold_lines}\n\n"
        "## Atlas rows meeting configured minimums\n\n"
        "| Axis | Regime | Usable rows | Total rows |\n|---|---|---:|---:|\n"
        f"{usable_lines}\n\n"
        "## Matched-day error-distribution comparisons\n\n"
        "Confidence intervals use a forecast-day block bootstrap; pixels are not independent bootstrap units.\n\n"
        "| Target vs none | Matched days | Target cells | Mean-error difference mm [CI] | MAE difference mm [CI] | Hedges g [CI] | Cliff delta | Usable | Material |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{effects_lines}\n\n"
        "## Decision\n\n" + ("All configured evidence gates pass.\n" if move_to_ml else
        "Do not train regime-aware correction ML yet:\n\n" + "\n".join(f"- {reason}" for reason in reasons) + "\n") +
        "\nActive/normal/break and WD remain disabled for the scientific prerequisites recorded above.\n",
        encoding="utf-8",
    )
    outputs = [report_path, json_path, counts_path, usable_groups_path, sample_path, effects_path,
               *map(Path, phase_c_result["outputs"])]
    return {"status": summary["status"], "move_to_ml": move_to_ml,
            "outputs": [str(path) for path in outputs], "summary": summary}
