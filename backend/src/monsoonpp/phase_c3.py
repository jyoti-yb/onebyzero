"""Phase C3: JJAS exact-pair audit and independent synoptic-system evidence."""
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
from .phase_c import PhaseCError
from .phase_c2 import (_combine, _effect_values, _interval, _sufficient,
                       run_phase_c2)
from .regimes.engine import RegimeParams, SYNOPTIC


def validate_exact_period(paired: xr.Dataset, start, end) -> dict:
    """Require every contiguous 03Z-to-03Z window in [start, end)."""
    expected_start = pd.Timestamp(start)
    expected_end = pd.Timestamp(end)
    starts = pd.DatetimeIndex(paired.valid_start.values)
    ends = pd.DatetimeIndex(paired.valid_end.values)
    expected = pd.date_range(expected_start, expected_end, freq="D", inclusive="left")
    if len(starts) != len(expected):
        raise PhaseCError(f"exact period requires {len(expected)} paired windows; found {len(starts)}")
    if starts.has_duplicates or not starts.equals(expected):
        missing = expected.difference(starts)
        extra = starts.difference(expected)
        raise PhaseCError(f"paired valid_start sequence is not exact; missing={list(missing)}, extra={list(extra)}")
    expected_ends = expected + pd.Timedelta(days=1)
    if not ends.equals(expected_ends):
        raise PhaseCError("paired valid_end values are not exact 24-hour windows")
    labels = pd.DatetimeIndex(paired.time.values)
    if labels.has_duplicates or not labels.equals(expected.normalize()):
        raise PhaseCError("forecast labels do not map one-to-one to the exact valid windows")
    return {
        "windows": len(expected), "valid_start": str(starts[0]), "valid_end": str(ends[-1]),
        "gaps": 0, "duplicates": 0,
    }


def _distance_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    mean_lat = np.deg2rad((lat0 + lat1) / 2.0)
    return float(np.hypot((lat1 - lat0) * 111.0, (lon1 - lon0) * 111.0 * np.cos(mean_lat)))


def track_synoptic_centres(centres: pd.DataFrame, options: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link already-classified centres with a deterministic nearest-neighbour tracker.

    Tracking changes no regime label. It only supplies independent proxy-system units
    for counts and bootstrap resampling.
    """
    columns = list(centres.columns) + ["system_id"]
    if centres.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=[
            "system_id", "primary_class", "first_time", "last_time", "snapshots",
            "independent_days", "contains_low_proxy", "contains_depression_proxy",
        ])
    data = centres.copy()
    data["time"] = pd.to_datetime(data.time)
    data = data.sort_values(["time", "lat", "lon"]).reset_index(drop=True)
    max_gap = pd.Timedelta(hours=float(options.get("track_max_gap_hours", 12)))
    max_speed = float(options.get("track_max_speed_kmh", 65.0))
    active: dict[int, dict] = {}
    assignments: dict[int, int] = {}
    next_id = 1

    for time, group in data.groupby("time", sort=True):
        active = {sid: item for sid, item in active.items() if time - item["time"] <= max_gap}
        candidates = []
        for row_index, row in group.iterrows():
            for sid, item in active.items():
                hours = (time - item["time"]).total_seconds() / 3600.0
                if hours <= 0:
                    continue
                distance = _distance_km(item["lat"], item["lon"], float(row.lat), float(row.lon))
                if distance <= max_speed * hours:
                    candidates.append((distance, row_index, sid))
        used_rows, used_systems = set(), set()
        for _, row_index, sid in sorted(candidates):
            if row_index in used_rows or sid in used_systems:
                continue
            assignments[row_index] = sid
            used_rows.add(row_index)
            used_systems.add(sid)
        for row_index, row in group.iterrows():
            sid = assignments.get(row_index)
            if sid is None:
                sid = next_id
                next_id += 1
                assignments[row_index] = sid
            active[sid] = {"time": time, "lat": float(row.lat), "lon": float(row.lon)}

    data["system_id"] = [f"SYS{assignments[i]:04d}" for i in data.index]
    events = []
    severity = {"low_proxy": 1, "depression_proxy": 2}
    for system_id, group in data.groupby("system_id", sort=True):
        classes = set(group["class"])
        primary = max(classes, key=lambda name: severity.get(name, 0))
        events.append({
            "system_id": system_id, "primary_class": primary,
            "first_time": str(group.time.min()), "last_time": str(group.time.max()),
            "snapshots": int(len(group)),
            "independent_days": int(group.time.dt.normalize().nunique()),
            "contains_low_proxy": "low_proxy" in classes,
            "contains_depression_proxy": "depression_proxy" in classes,
        })
    return data, pd.DataFrame(events)


def independent_event_counts(tracked: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SYNOPTIC[1:]:
        contains = f"contains_{name}"
        rows.append({
            "regime": name,
            "independent_systems_with_class": int(events[contains].sum()) if contains in events else 0,
            "primary_class_systems": int((events.primary_class == name).sum()) if not events.empty else 0,
            "independent_days": int(tracked.loc[tracked["class"] == name, "time"].dt.normalize().nunique())
            if not tracked.empty else 0,
            "snapshot_centres": int((tracked["class"] == name).sum()) if not tracked.empty else 0,
        })
    return pd.DataFrame(rows)


def compare_synoptic_errors_by_system(paired: xr.Dataset, regimes: xr.Dataset,
                                      tracked: pd.DataFrame, base_mask: np.ndarray,
                                      options: dict) -> pd.DataFrame:
    """Estimate error differences with independent tracked systems as bootstrap blocks."""
    fc, obs = paired.fc.values, paired.obs.values
    error = fc - obs
    valid = base_mask[None] & np.isfinite(fc) & np.isfinite(obs)
    dates = pd.DatetimeIndex(paired.time.values).normalize()
    date_index = {date: i for i, date in enumerate(dates)}
    lat2, lon2 = np.meshgrid(paired.lat.values, paired.lon.values, indexing="ij")
    params = RegimeParams()
    reps = int(options.get("bootstrap_replicates", 1000))
    confidence = float(options.get("confidence", 0.95))
    min_days = int(options.get("min_comparison_days", 10))
    min_cells = int(options.get("min_comparison_cells", 5000))
    rng = np.random.default_rng(int(options.get("bootstrap_seed", 42)) + 1009)
    rows = []

    for code, name in enumerate(SYNOPTIC[1:], start=1):
        target_records, reference_records, used_days = [], [], set()
        class_centres = tracked[tracked["class"] == name]
        for _, event in class_centres.groupby("system_id", sort=True):
            target_values, reference_values = [], []
            for day, day_centres in event.groupby(event.time.dt.normalize()):
                if day not in date_index:
                    continue
                index = date_index[day]
                distance = np.full(base_mask.shape, np.inf)
                for centre in day_centres.itertuples():
                    candidate = np.hypot(
                        (lat2 - centre.lat) * 111.0,
                        (lon2 - centre.lon) * 111.0 * np.cos(np.deg2rad(centre.lat)),
                    )
                    distance = np.minimum(distance, candidate)
                target_mask = (valid[index] & (regimes.synoptic.values[index] == code)
                               & (distance <= params.influence_km[name]))
                reference_mask = valid[index] & (regimes.synoptic.values[index] == 0)
                if target_mask.any() and reference_mask.any():
                    target_values.append(error[index][target_mask])
                    reference_values.append(error[index][reference_mask])
                    used_days.add(day)
            if target_values and reference_values:
                target_records.append(_sufficient(np.concatenate(target_values)))
                reference_records.append(_sufficient(np.concatenate(reference_values)))

        systems = len(target_records)
        if systems:
            target, reference = _combine(target_records), _combine(reference_records)
            mean_diff, mae_diff, hedges_g = _effect_values(target, reference)
            boot_mean, boot_mae, boot_g = [], [], []
            for _ in range(reps):
                indices = rng.integers(0, systems, systems)
                values = _effect_values(_combine(target_records, indices), _combine(reference_records, indices))
                boot_mean.append(values[0]); boot_mae.append(values[1]); boot_g.append(values[2])
            mean_lo, mean_hi = _interval(boot_mean, confidence)
            mae_lo, mae_hi = _interval(boot_mae, confidence)
            g_lo, g_hi = _interval(boot_g, confidence)
            if systems < 2:
                mean_lo = mean_hi = mae_lo = mae_hi = g_lo = g_hi = float("nan")
        else:
            target = reference = {"n": 0}
            mean_diff = mae_diff = hedges_g = float("nan")
            mean_lo = mean_hi = mae_lo = mae_hi = g_lo = g_hi = float("nan")
        rows.append({
            "comparison": f"{name}_vs_none", "bootstrap_unit": "tracked_proxy_system",
            "independent_systems": systems, "independent_days": len(used_days),
            "target_cells": int(target["n"]), "reference_cells": int(reference["n"]),
            "mean_error_difference_mm": mean_diff, "mean_error_ci_low": mean_lo,
            "mean_error_ci_high": mean_hi, "mae_difference_mm": mae_diff,
            "mae_ci_low": mae_lo, "mae_ci_high": mae_hi, "hedges_g": hedges_g,
            "hedges_g_ci_low": g_lo, "hedges_g_ci_high": g_hi,
            "confidence": confidence, "bootstrap_replicates": reps,
            "meets_existing_day_cell_gate": (
                len(used_days) >= min_days and target["n"] >= min_cells and reference["n"] >= min_cells
            ),
            "ci_estimable": systems >= 2,
        })
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "none"
    header = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    rows = []
    for item in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in item) + " |")
    return header + "\n" + "\n".join(rows)


def run_phase_c3(cfg: Config) -> dict:
    if not cfg.phase_c3:
        raise PhaseCError("phase_c3 configuration is required")
    opts = cfg.phase_c3
    paired_path = Path(cfg.phase_c["paired_path"])
    if not paired_path.exists():
        raise PhaseCError(f"paired real-data artifact missing: {paired_path}")
    paired = xr.open_dataset(paired_path).load()
    pairing = validate_exact_period(paired, opts["period_start"], opts["period_end"])
    if pairing["windows"] != 122:
        raise PhaseCError(f"Phase C3 requires all 122 JJAS windows; found {pairing['windows']}")
    c2 = run_phase_c2(cfg)

    phase_c_out = Path(cfg.phase_c["out_dir"])
    centres = pd.read_csv(phase_c_out / "synoptic_proxy_centres.csv")
    tracked, events = track_synoptic_centres(centres, opts)
    counts = independent_event_counts(tracked, events)
    regimes = xr.open_dataset(phase_c_out / "phase_c_regimes.nc").load()
    static = require_real_static(load_real_static(cfg))
    base_mask = india_mask(paired.lat.values, paired.lon.values) & static.land.values
    system_effects = compare_synoptic_errors_by_system(paired, regimes, tracked, base_mask, cfg.phase_c2)

    atlas = pd.read_csv(phase_c_out / "regime_error_atlas.csv")
    threshold_counts = (
        atlas[(atlas.axis == "synoptic") & (atlas.region == "all")]
        .groupby(["regime", "threshold_mm"], sort=False)
        .agg(independent_days=("n_days", "sum"), grid_cell_samples=("n_cells", "sum"),
             observed_events=("obs_events", "sum"), usable_month_rows=("low_sample", lambda x: int((~x).sum())),
             total_month_rows=("low_sample", "size"))
        .reset_index()
    )

    out = Path(opts.get("out_dir", cfg.report_dir / "phase_c3"))
    out.mkdir(parents=True, exist_ok=True)
    tracked_path = out / "tracked_synoptic_proxy_centres.csv"
    events_path = out / "independent_synoptic_proxy_systems.csv"
    counts_path = out / "independent_event_counts.csv"
    effects_path = out / "synoptic_system_block_effects.csv"
    thresholds_path = out / "synoptic_threshold_counts.csv"
    summary_path = out / "phase_c3_summary.json"
    report_path = out / "phase_c3_report.md"
    tracked.to_csv(tracked_path, index=False)
    events.to_csv(events_path, index=False)
    counts.to_csv(counts_path, index=False)
    system_effects.to_csv(effects_path, index=False)
    threshold_counts.to_csv(thresholds_path, index=False)

    decision = "PROCEED_TO_ML" if c2["move_to_ml"] else "DO_NOT_PROCEED_TO_ML"
    summary = {
        "status": "PASS",
        "decision": decision, "correction_ml_trained": False,
        "pairing": pairing, "observation_role": paired.attrs["obs_meta_product_role"],
        "regime_counts": c2["summary"]["regime_counts"],
        "atlas_rows": c2["summary"]["atlas_rows"],
        "statistically_usable_atlas_rows": c2["summary"]["statistically_usable_atlas_rows"],
        "usable_atlas_groups": c2["summary"]["usable_atlas_groups"],
        "independent_event_counts": counts.to_dict(orient="records"),
        "day_block_effects": c2["summary"]["synoptic_effects"],
        "system_block_effects": system_effects.to_dict(orient="records"),
        "decision_reasons": c2["summary"]["decision_reasons"],
        "capability_gates": c2["summary"]["capability_gates"],
        "tracking_method": {
            "kind": "deterministic nearest-neighbour linkage of unchanged objective proxy centres",
            "max_gap_hours": float(opts.get("track_max_gap_hours", 12)),
            "max_speed_kmh": float(opts.get("track_max_speed_kmh", 65.0)),
            "official_classification": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report_path.write_text(
        "# Phase C3 JJAS 2024 regime evidence report\n\n"
        f"**Strict decision: `{decision}`**  \n"
        "**Correction ML trained: FALSE**\n\n"
        f"Exact pairing: `{pairing['windows']}` contiguous 03Z-to-03Z windows from "
        f"`{pairing['valid_start']}` through `{pairing['valid_end']}`; no gaps or duplicates. "
        f"Observation role: `{summary['observation_role']}` (proxy truth, not final IMD truth).\n\n"
        "## Independent proxy systems\n\n"
        + _markdown_table(counts, list(counts.columns)) + "\n\n"
        "Grid-cell samples are not independent events. Systems above are nearest-neighbour tracks of the "
        "unchanged objective proxy centres, not official IMD systems.\n\n"
        "## Threshold counts\n\n"
        + _markdown_table(threshold_counts, list(threshold_counts.columns)) + "\n\n"
        "## System-block uncertainty\n\n"
        + _markdown_table(system_effects, [
            "comparison", "independent_systems", "independent_days", "target_cells",
            "mae_difference_mm", "mae_ci_low", "mae_ci_high", "hedges_g",
            "hedges_g_ci_low", "hedges_g_ci_high", "meets_existing_day_cell_gate", "ci_estimable",
        ]) + "\n\n"
        "Day-block results remain in the Phase C2 evidence output. The strict decision reuses only the "
        "existing C2 gates; independent-system counts are reported without inventing a new gate.\n\n"
        "## Decision basis\n\n" + "\n".join(f"- {reason}" for reason in summary["decision_reasons"]) + "\n\n"
        "Active/normal/break and WD remain disabled because their recorded prerequisites are unavailable.\n",
        encoding="utf-8",
    )
    outputs = [report_path, summary_path, counts_path, events_path, tracked_path, effects_path, thresholds_path,
               *map(Path, c2["outputs"])]
    return {"status": summary["status"], "decision": decision,
            "outputs": [str(path) for path in outputs], "summary": summary}
