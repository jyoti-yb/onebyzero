"""Export compact serving-only artifacts from an existing completed run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import Config
from .serving.bundle import BundleError, write_bundle


class DemoExportError(RuntimeError):
    pass


def _clean(value):
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.integer, np.floating)):
        return _clean(value.item())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _grid(values, digits: int = 2):
    array = np.asarray(values, dtype="float64")
    return [[None if not np.isfinite(value) else round(float(value), digits) for value in row] for row in array]


def _stats(values) -> dict:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if not array.size:
        return {"cells": 0, "min": None, "max": None, "mean": None}
    return {
        "cells": int(array.size), "min": round(float(array.min()), 3),
        "max": round(float(array.max()), 3), "mean": round(float(array.mean()), 3),
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise DemoExportError(f"required validated-run artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoExportError(f"invalid JSON artifact {path}: {exc}") from exc


def _phase_c_summary(cfg: Config) -> tuple[dict, str]:
    candidates = []
    if cfg.phase_c3:
        out = Path(cfg.phase_c3.get("out_dir", cfg.report_dir / "phase_c3"))
        candidates.append((out / "phase_c3_summary.json", "phase_c3"))
    if cfg.phase_c2:
        out = Path(cfg.phase_c2.get("out_dir", cfg.report_dir / "phase_c2"))
        candidates.append((out / "phase_c2_summary.json", "phase_c2"))
    for path, phase in candidates:
        if path.is_file():
            summary = _read_json(path)
            if not str(summary.get("status", "")).startswith("PASS"):
                raise DemoExportError(f"validated-run summary is not passing: {path}")
            return summary, phase
    raise DemoExportError("no passing Phase C2/C3 summary is available for export")


def _phase_c_payloads(cfg: Config) -> tuple[dict, dict]:
    if not cfg.phase_c or not cfg.phase_c.get("paired_path"):
        raise DemoExportError("phase_c.paired_path is required for a validated real-data bundle")
    paired_path = Path(cfg.phase_c["paired_path"])
    phase_c_out = Path(cfg.phase_c.get("out_dir", cfg.report_dir / "phase_c"))
    regimes_path = phase_c_out / "phase_c_regimes.nc"
    atlas_path = phase_c_out / "regime_error_atlas.csv"
    capability_path = phase_c_out / "capability_report.json"
    for path in (paired_path, regimes_path, atlas_path, capability_path):
        if not path.is_file():
            raise DemoExportError(f"required validated-run artifact is missing: {path}")
    summary, phase = _phase_c_summary(cfg)
    capability = _read_json(capability_path)

    with xr.open_dataset(paired_path) as source:
        paired = source.isel(time=-1).load()
        paired_days = int(source.sizes["time"])
        latest = pd.Timestamp(source.time.values[-1])
        role = str(source.attrs.get("obs_meta_product_role", ""))
        if role not in ("FINAL_TRUTH", "OFFICIAL_ALTERNATIVE", "SMOKE_TEST_PROXY"):
            raise DemoExportError("paired artifact lacks validated observation-role provenance")
        construction = str(source.fc.attrs.get("construction_method", ""))
        if "APCP(0-27) - APCP(0-3)" not in construction:
            raise DemoExportError("paired artifact lacks the frozen GFS precipitation construction")
        paired_attrs = dict(source.attrs)
        fc_attrs = dict(source.fc.attrs)
    with xr.open_dataset(regimes_path) as source:
        regime = source.sel(time=latest).load()
        regime_attrs = dict(source.attrs)

    atlas = pd.read_csv(atlas_path)
    latest_month = str(latest.to_period("M"))
    atlas_latest = atlas[(atlas.month.astype(str) == latest_month) & (atlas.region == "all")]
    centres_path = phase_c_out / "synoptic_proxy_centres.csv"
    centres = pd.read_csv(centres_path) if centres_path.is_file() else pd.DataFrame()
    if not centres.empty:
        times = pd.to_datetime(centres.time)
        centres = centres[times.dt.normalize() == latest.normalize()]

    valid = np.isfinite(paired.fc.values) & np.isfinite(paired.obs.values)
    synoptic = np.asarray(regime.synoptic.values)
    class_names = str(regime.synoptic.attrs.get("classes", "none,low_proxy,depression_proxy")).split(",")
    synoptic_counts = {
        name: int(np.sum(valid & (synoptic == code))) for code, name in enumerate(class_names)
    }
    monsoon_code = int(np.asarray(regime.monsoon_state.values).item())
    monsoon_classes = {0: "break", 1: "normal", 2: "active"}
    proxy = role == "SMOKE_TEST_PROXY"
    label = str(latest.date())
    cycle = {
        "forecast_label": label,
        "init_time": str(np.asarray(paired.init_time.values).item()) if "init_time" in paired.coords else None,
        "valid_start": str(pd.Timestamp(np.asarray(paired.valid_start.values).item())),
        "valid_end": str(pd.Timestamp(np.asarray(paired.valid_end.values).item())),
        "lead_days": int(cfg.phase_c.get("lead", 1)),
        "units": "mm/day",
        "lat": np.asarray(paired.lat.values).round(3).tolist(),
        "lon": np.asarray(paired.lon.values).round(3).tolist(),
        "forecast_mm": _grid(paired.fc.values),
        "observation_mm": _grid(paired.obs.values),
        "forecast_statistics": _stats(paired.fc.values[valid]),
        "observation_statistics": _stats(paired.obs.values[valid]),
        "observation_role": role,
        "observation_is_proxy": proxy,
        "observation_warning": "Proxy truth for exploration; not final IMD truth." if proxy else None,
        "forecast_provenance": {key: fc_attrs.get(key) for key in (
            "model", "product", "cycle", "construction_method", "qa_method", "source_units", "output_units"
        )},
    }
    regimes = {
        "forecast_label": label,
        "classification_scope": "objective proxies and heuristic scores; no official IMD classification",
        "monsoon_state": {
            "available": monsoon_code in monsoon_classes,
            "value": monsoon_classes.get(monsoon_code, "not_assessed"),
            "reason": regime_attrs.get("monsoon_state_reason"),
        },
        "western_disturbance": {
            "available": False,
            "reason": regime_attrs.get("wd_proxy_reason", "scientific prerequisites unavailable"),
        },
        "synoptic_classes": class_names,
        "synoptic_cell_counts": synoptic_counts,
        "synoptic_code": _grid(synoptic, 0),
        "orographic_score": _grid(regime.oro_score.values, 3),
        "coastal_score": _grid(regime.coast_score.values, 3),
        "convective_score": _grid(regime.conv_score.values, 3),
        "proxy_centres": _clean(centres.to_dict("records")),
    }
    decision = summary.get("decision") or (
        "PROCEED_TO_ML" if summary.get("move_to_ml") else "DO_NOT_PROCEED_TO_ML"
    )
    verification = {
        "period_days": paired_days,
        "latest_month": latest_month,
        "observation_role": role,
        "observation_is_proxy": proxy,
        "decision": decision,
        "summary": _clean(summary),
        "latest_month_all_region_atlas": _clean(atlas_latest.to_dict("records")),
        "warning": "Exploratory verification against proxy truth; not final IMD verification." if proxy else None,
    }
    status = {
        "status": "ready",
        "run": cfg.name,
        "bundle_mode": "validated_regime_error_atlas",
        "source_phase": phase,
        "validated_run_status": summary["status"],
        "latest_cycle": label,
        "paired_days": paired_days,
        "observation_role": role,
        "observation_is_proxy": proxy,
        "scientific_decision": decision,
        "correction_ml_trained": bool(summary.get("correction_ml_trained", False)),
        "capabilities": capability.get("capabilities", {}),
        "network_required_at_runtime": False,
        "raw_data_in_bundle": False,
        "source_provenance": {
            "observation": paired_attrs.get("obs_meta_source_url_or_origin"),
            "static": regime_attrs.get("static_source"),
        },
    }
    payloads = {"status": status, "cycle": cycle, "regimes": regimes, "verification": verification}
    metadata = {
        "source_run": cfg.name, "source_phase": phase, "latest_cycle": label,
        "observation_role": role, "network_required_at_runtime": False,
    }
    return payloads, metadata


def _trained_payloads(cfg: Config) -> tuple[dict, dict]:
    from .products.service import ForecastService

    verification_path = cfg.report_dir / "verification_test.json"
    if not cfg.dataset_path.is_file() or not (cfg.model_dir / "predictor.joblib").is_file():
        raise DemoExportError("completed dataset and predictor are required for a trained-run bundle")
    verification = _read_json(verification_path)
    service = ForecastService(cfg)
    date = service.dates[-1]
    lead = service.leads[0]
    cycle = service.grid(date, lead)
    cycle.update({"forecast_label": str(date.date()), "observation_is_proxy": None})
    regimes = service.regimes(date, lead)
    status = {
        "status": "ready", "run": cfg.name, "bundle_mode": "trained_demo",
        "validated_run_status": "TRAINED_AND_EVALUATED", "latest_cycle": str(date.date()),
        "leads": service.leads, "network_required_at_runtime": False, "raw_data_in_bundle": False,
    }
    verification_payload = {
        "decision": "DEMO_MODEL_AVAILABLE", "summary": _clean(verification),
        "warning": "Synthetic/demo verification; consult source-run provenance before scientific use.",
    }
    payloads = {
        "status": status, "cycle": _clean(cycle), "regimes": _clean(regimes),
        "verification": verification_payload,
    }
    metadata = {
        "source_run": cfg.name, "source_phase": "trained_evaluated",
        "latest_cycle": str(date.date()), "network_required_at_runtime": False,
    }
    return payloads, metadata


def export_demo(cfg: Config, destination: str | Path) -> dict:
    """Export a serving bundle without copying raw/rebuild inputs."""
    if cfg.phase_c and cfg.phase_c.get("paired_path") and Path(cfg.phase_c["paired_path"]).is_file():
        payloads, metadata = _phase_c_payloads(cfg)
    else:
        payloads, metadata = _trained_payloads(cfg)
    metadata = {
        **metadata,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "producer": "monsoonpp export-demo",
    }
    try:
        path = write_bundle(destination, _clean(payloads), _clean(metadata))
    except BundleError as exc:
        raise DemoExportError(str(exc)) from exc
    manifest = _read_json(path / "manifest.json")
    return {"bundle": str(path), "bundle_id": manifest["bundle_id"],
            "files": ["manifest.json", *manifest["files"].keys()]}
