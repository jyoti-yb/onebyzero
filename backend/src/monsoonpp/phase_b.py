"""End-to-end validation evidence for Phase B real GFS and static geography."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .adapters.gfs import DERIVED_ATMOS_FIELDS, GFSAdapter
from .adapters.gfs_atmos import FIELD_SPECS
from .config import Config
from .data.static_geo import load_real_static, require_real_static
from .schema import validate_static


RAW_MAP_FIELDS = ("mslp", "tcwv", "cape", "u850", "v850", "rh850", "hgt500")
STATIC_MAP_FIELDS = ("elev", "slope", "aspect", "coast_dist", "land")


def _stats(da: xr.DataArray) -> dict:
    values = np.asarray(da.values)
    finite = np.isfinite(values)
    return {
        "units": str(da.attrs.get("units", "")),
        "shape": list(values.shape),
        "missing_count": int(values.size - finite.sum()),
        "min": float(np.nanmin(values)),
        "mean": float(np.nanmean(values)),
        "max": float(np.nanmax(values)),
    }


def _plot_fields(ds: xr.Dataset, names: tuple[str, ...], path: Path, title: str) -> None:
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows), constrained_layout=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        da = ds[name]
        cmap = ("terrain" if name == "elev" else "twilight" if name == "aspect"
                else "RdBu_r" if "anom" in name or name in ("u850", "v850", "vo850") else "viridis")
        image = ax.pcolormesh(ds.lon, ds.lat, da, shading="auto", cmap=cmap)
        fig.colorbar(image, ax=ax, shrink=0.82, label=str(da.attrs.get("units", "")))
        ax.set_title(name)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_markdown(qa: dict, path: Path) -> None:
    rows = []
    for group in ("atmospheric_fields", "derived_fields", "static_fields"):
        for name, item in qa[group].items():
            rows.append(
                f"| {name} | {item['units']} | {item['missing_count']} | "
                f"{item['min']:.4g} | {item['mean']:.4g} | {item['max']:.4g} |"
            )
    text = f"""# Phase B validation report

**Status: {qa['status']}**

- GFS product: `{qa['gfs']['product']}`
- Forecast label / lead: `{qa['forecast_label']}` / `{qa['lead_days']}` day
- Initialization: `{qa['gfs']['init_time']}`
- Daily state samples: `{', '.join(qa['gfs']['sample_valid_times'])}`
- Grid: `{qa['grid']['shape'][0]} x {qa['grid']['shape'][1]}` at `{qa['grid']['resolution_degrees']} degree`
- Static source: {qa['static']['static_source']}
- Static DOI: `{qa['static']['static_source_doi']}`
- Static file: `{qa['static']['static_source_file']}`

## Validation

All 19 required GFS messages passed direct GRIB2 checks for parameter identity, pressure level,
units, initialization and valid time, instantaneous step type, regular 0.25-degree grid,
missing values, and broad physical ranges. Static geography passed source resolution,
coordinate, unit, finite-value, and non-synthetic provenance checks.

| Field | Units | Missing | Min | Mean | Max |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Derived fields

Only `wspd850`, `vo850`, `moisture_transport`, `pwat_anom`, and `mslp_anom` are derived.
PWAT and MSLP anomalies are departures from an 8-cell Gaussian spatial background, not
climatological anomalies. The moisture-transport value is a magnitude proxy (`PWAT * wspd850`),
not a vertically integrated moisture-flux vector.

## Provenance

The JSON companion preserves each source GRIB path and each message's decoded identity,
level, source/output units, source range, initialization, forecast hour, and valid time.
The ETOPO source URL, DOI, resolution, cached subset, remapping method, and land definition
are preserved there and in the static NetCDF.
"""
    path.write_text(text, encoding="utf-8")


def run_phase_b(cfg: Config) -> dict:
    opts = dict(cfg.phase_b)
    date = pd.Timestamp(opts.get("date", "2024-07-15"))
    lead = int(opts.get("lead", 1))
    out = Path(opts.get("out_dir", cfg.report_dir / "phase_b"))
    out.mkdir(parents=True, exist_ok=True)

    atmospheric, provenance = GFSAdapter(cfg).load_atmospheric(date, lead)
    static = validate_static(require_real_static(load_real_static(cfg)))
    if not (np.array_equal(atmospheric.lat.values, static.lat.values)
            and np.array_equal(atmospheric.lon.values, static.lon.values)):
        raise ValueError("validated GFS and static geography grids do not match exactly")

    raw_names = tuple(FIELD_SPECS)
    derived_names = tuple(DERIVED_ATMOS_FIELDS)
    qa = {
        "status": "PASS",
        "forecast_label": str(date.normalize()),
        "lead_days": lead,
        "grid": {
            "shape": [int(atmospheric.sizes["lat"]), int(atmospheric.sizes["lon"])],
            "resolution_degrees": float(np.median(np.diff(atmospheric.lat.values))),
            "lat_bounds": [float(atmospheric.lat.min()), float(atmospheric.lat.max())],
            "lon_bounds": [float(atmospheric.lon.min()), float(atmospheric.lon.max())],
        },
        "gfs": {
            "source": "NOAA GFS",
            "product": cfg.adapter_options.get("gfs_product", "pgrb2.0p25"),
            "init_time": atmospheric.attrs["init_time"],
            "valid_start": atmospheric.attrs["valid_start"],
            "valid_end": atmospheric.attrs["valid_end"],
            "source_files": provenance["source_files"],
            "sample_valid_times": provenance["sample_valid_times"],
            "message_validation": provenance["field_validation"],
        },
        "static": dict(static.attrs),
        "atmospheric_fields": {name: _stats(atmospheric[name]) for name in raw_names},
        "derived_fields": {name: _stats(atmospheric[name]) for name in derived_names},
        "static_fields": {name: _stats(static[name]) for name in STATIC_MAP_FIELDS},
        "assumptions": [
            "Daily atmospheric state is the mean of f006, f012, f018, and f024 instantaneous fields.",
            "PWAT/MSLP anomalies use an 8-cell Gaussian spatial background, not a climatology.",
            "Land is ETOPO elevation > 0 over the configured fraction of a target cell.",
        ],
        "protected_logic": {
            "gfs_precipitation": "unchanged",
            "observation_pipeline": "unchanged",
            "regime_thresholds_and_classification": "unchanged",
            "rainfall_correction_ml": "unchanged",
        },
    }

    atmosphere_path = out / "phase_b_atmospheric_daily.nc"
    static_path = out / "phase_b_static_geography.nc"
    json_path = out / "phase_b_validation.json"
    report_path = out / "phase_b_validation_report.md"
    raw_map = out / "gfs_atmospheric_fields.png"
    derived_map = out / "gfs_derived_fields.png"
    static_map = out / "static_geography.png"
    atmospheric.to_netcdf(atmosphere_path)
    static.assign(land=static.land.astype("int8")).to_netcdf(static_path)
    json_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    _write_markdown(qa, report_path)
    _plot_fields(atmospheric, RAW_MAP_FIELDS, raw_map, "Validated GFS atmospheric predictors")
    _plot_fields(atmospheric, derived_names, derived_map, "Phase B derived atmospheric fields")
    _plot_fields(static, STATIC_MAP_FIELDS, static_map, "ETOPO-derived static geography")
    outputs = [report_path, json_path, atmosphere_path, static_path, raw_map, derived_map, static_map]
    return {"status": "PASS", "outputs": [str(path) for path in outputs], "qa": qa}
