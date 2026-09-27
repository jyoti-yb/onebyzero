"""Run the seven-day smoke-test pipeline on local forecast and IMD files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.align import align_forecast_observation
from src.build_precip import build_precip_accumulation
from src.config import ConfigError, load_config
from src.download_gfs import GFSForecastAdapter
from src.inspect_grib import inspect_grib_metadata
from src.load_imd import load_imd_precip
from src.validate import validate_paired_dataset
from src.verify import verify_paired_dataset


def resolve_path(project_root: Path, value: str) -> Path:
    return (project_root / value).resolve()


def run(config_path: Path) -> dict:
    project_root = config_path.resolve().parents[1]
    config = load_config(config_path)
    paths = config["paths"]

    raw_gfs_dir = resolve_path(project_root, paths["raw_gfs_dir"])
    raw_imd_dir = resolve_path(project_root, paths["raw_imd_dir"])
    processed_gfs_dir = resolve_path(project_root, paths["processed_gfs_dir"])
    processed_imd_dir = resolve_path(project_root, paths["processed_imd_dir"])
    paired_dir = resolve_path(project_root, paths["paired_dir"])
    manifest_dir = resolve_path(project_root, paths["manifest_dir"])
    report_dir = resolve_path(project_root, paths["report_dir"])

    for directory in (processed_gfs_dir, processed_imd_dir, paired_dir, manifest_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    adapter = GFSForecastAdapter(
        source_dir=raw_gfs_dir,
        file_glob=config["gfs"]["local_file_glob"],
        expected_days=config["smoke_test"]["expected_days"],
        cycle=config["gfs"]["cycle"],
        model=config["gfs"]["model"],
        lead_day=config["gfs"]["lead_day"],
        domain=config["domain"],
        variables=config["variables"],
    )
    forecast_files = adapter.list_local_files()
    if not forecast_files:
        raise FileNotFoundError(
            f"No local forecast files found in {raw_gfs_dir}. "
            "Download logic is not implemented; place a 7-day smoke-test sample there."
        )

    observation_files = sorted(raw_imd_dir.glob(config["observation"]["local_file_glob"]))
    if not observation_files:
        raise FileNotFoundError(
            f"No local observation files found in {raw_imd_dir}. "
            "Place the matching 7-day IMD smoke-test sample there."
        )

    metadata = inspect_grib_metadata(
        forecast_files,
        config["variables"],
        report_path=report_dir / "02_grib_metadata.csv",
        apcp_aliases=config["gfs"]["precipitation_variable_aliases"],
        print_apcp=True,
    )
    gfs_precip_path = processed_gfs_dir / "gfs_daily_precip.nc"
    imd_precip_path = processed_imd_dir / "imd_daily_precip.nc"
    paired_path = paired_dir / "paired_daily_precip.nc"

    build_precip_accumulation(
        forecast_files=forecast_files,
        output_path=gfs_precip_path,
        variable_candidates=config["gfs"]["precipitation_variable_aliases"],
        domain=config["domain"],
        rainfall_window_start_utc=config["verification"]["rainfall_window_start_utc"],
        rainfall_window_hours=config["verification"]["rainfall_window_hours"],
    )
    load_imd_precip(
        input_files=observation_files,
        output_path=imd_precip_path,
        variable_candidates=config["observation"]["precipitation_variable_candidates"],
        domain=config["domain"],
    )
    align_forecast_observation(
        forecast_path=gfs_precip_path,
        observation_path=imd_precip_path,
        output_path=paired_path,
        alignment_config=config["alignment"],
    )

    validation = validate_paired_dataset(paired_path, config["quality"])
    verification = verify_paired_dataset(paired_path, config["verification"]["metrics"])

    manifest = {
        "config": str(config_path),
        "configuration": {
            "domain": config["domain"],
            "gfs": {
                "cycle": config["gfs"]["cycle"],
                "model": config["gfs"]["model"],
                "lead_day": config["gfs"]["lead_day"],
            },
            "verification": config["verification"],
            "variables": config["variables"],
            "quality": config["quality"],
        },
        "forecast_files": [str(path) for path in forecast_files],
        "observation_files": [str(path) for path in observation_files],
        "grib_metadata": metadata,
        "outputs": {
            "forecast_precip": str(gfs_precip_path),
            "observation_precip": str(imd_precip_path),
            "paired": str(paired_path),
        },
        "validation": validation,
        "verification": verification,
    }
    manifest_path = manifest_dir / "smoke_test_manifest.json"
    report_path = report_dir / "smoke_test_summary.json"
    for path in (manifest_path, report_path):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/smoke_test.yaml"))
    args = parser.parse_args()

    try:
        manifest = run(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "ok", "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
