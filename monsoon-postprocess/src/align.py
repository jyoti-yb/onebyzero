"""Exact temporal and spatial alignment for GFS and IMD daily rainfall."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


class AlignmentError(RuntimeError):
    """Raised when forecast and observation periods or grids do not match exactly."""


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_z(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise AlignmentError(f"Missing or invalid {name} metadata: {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlignmentError(f"Invalid {name} metadata: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _datetime64_to_utc(value: Any) -> datetime:
    seconds = np.datetime64(value, "s").astype(np.int64)
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def validate_temporal_pair(
    forecast_attrs: Mapping[str, Any],
    observation_date: Any,
    *,
    rainfall_window_start_utc: int = 3,
    rainfall_window_hours: int = 24,
) -> dict[str, Any]:
    """Validate one GFS window against one IMD labelled observation date."""

    initialization = _parse_iso_z(
        forecast_attrs.get("initialization_time"), "initialization_time"
    )
    valid_start = _parse_iso_z(forecast_attrs.get("valid_start"), "valid_start")
    valid_end = _parse_iso_z(forecast_attrs.get("valid_end"), "valid_end")
    observation_label = _datetime64_to_utc(observation_date).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    expected_start = initialization + timedelta(hours=rainfall_window_start_utc)
    expected_end = expected_start + timedelta(hours=rainfall_window_hours)
    observation_start = observation_label - timedelta(days=1) + timedelta(
        hours=rainfall_window_start_utc
    )
    observation_end = observation_label + timedelta(hours=rainfall_window_start_utc)
    checks = {
        "gfs_start_matches_initialization": valid_start == expected_start,
        "gfs_end_matches_window": valid_end == expected_end,
        "imd_start_matches_gfs": observation_start == valid_start,
        "imd_end_matches_gfs": observation_end == valid_end,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise AlignmentError(
            f"Temporal alignment failed for initialization {_iso_z(initialization)} "
            f"and IMD date {observation_label.date()}: {failed}"
        )
    return {
        "initialization_time": _iso_z(initialization),
        "gfs_valid_start": _iso_z(valid_start),
        "gfs_valid_end": _iso_z(valid_end),
        "imd_observation_date": observation_label.date().isoformat(),
        "imd_window_start": _iso_z(observation_start),
        "imd_window_end": _iso_z(observation_end),
        "period_match": True,
        "status": "PASS",
    }


def normalize_and_select_exact_grid(
    forecast: xr.DataArray,
    observation_latitude: np.ndarray,
    observation_longitude: np.ndarray,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Normalize longitude, sort latitude, and select the observation grid exactly."""

    if "latitude" not in forecast.coords or "longitude" not in forecast.coords:
        raise AlignmentError("GFS rainfall must have latitude and longitude coordinates.")
    source_latitude = np.asarray(forecast.latitude.values, dtype=np.float64)
    source_longitude = np.asarray(forecast.longitude.values, dtype=np.float64)
    target_latitude = np.asarray(observation_latitude, dtype=np.float64)
    target_longitude = np.asarray(observation_longitude, dtype=np.float64)
    normalized_longitude = ((source_longitude + 180.0) % 360.0) - 180.0
    normalized = forecast.assign_coords(longitude=normalized_longitude)
    normalized = normalized.sortby("latitude").sortby("longitude")

    normalized_latitude = np.asarray(normalized.latitude.values, dtype=np.float64)
    normalized_longitude = np.asarray(normalized.longitude.values, dtype=np.float64)
    missing_latitude = target_latitude[~np.isin(target_latitude, normalized_latitude)]
    missing_longitude = target_longitude[~np.isin(target_longitude, normalized_longitude)]
    if missing_latitude.size or missing_longitude.size:
        raise AlignmentError(
            "Exact grid matching failed; missing GFS coordinates: "
            f"latitude={missing_latitude.tolist()}, longitude={missing_longitude.tolist()}"
        )

    selected = normalized.sel(
        latitude=xr.DataArray(target_latitude, dims="latitude"),
        longitude=xr.DataArray(target_longitude, dims="longitude"),
    )
    selected_latitude = np.asarray(selected.latitude.values, dtype=np.float64)
    selected_longitude = np.asarray(selected.longitude.values, dtype=np.float64)
    latitude_difference = np.abs(selected_latitude - target_latitude)
    longitude_difference = np.abs(selected_longitude - target_longitude)
    if not np.array_equal(selected_latitude, target_latitude) or not np.array_equal(
        selected_longitude, target_longitude
    ):
        raise AlignmentError("Selected GFS coordinates are not exactly equal to IMD coordinates.")

    diagnostics = {
        "native_shape": list(forecast.shape),
        "aligned_shape": list(selected.shape),
        "native_latitude_first": float(source_latitude[0]),
        "native_latitude_last": float(source_latitude[-1]),
        "native_longitude_first": float(source_longitude[0]),
        "native_longitude_last": float(source_longitude[-1]),
        "longitude_normalization": "((longitude + 180) % 360) - 180",
        "latitude_sort": "ascending",
        "selection_method": "exact_coordinate_selection",
        "interpolation_performed": False,
        "maximum_latitude_difference": float(latitude_difference.max(initial=0.0)),
        "maximum_longitude_difference": float(longitude_difference.max(initial=0.0)),
    }
    return selected, diagnostics


def _write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> Path:
    if not rows:
        raise AlignmentError("Cannot write an empty time-alignment report.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_json(payload: Mapping[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def _save_aligned_field(
    rainfall: xr.DataArray,
    source_attrs: Mapping[str, Any],
    output_path: Path,
) -> Path:
    dataset = rainfall.astype("float32").to_dataset(name="rainfall")
    dataset.attrs = dict(source_attrs)
    dataset.attrs.update(
        {
            "spatial_alignment": "exact_coordinate_selection",
            "longitude_normalization": "((longitude + 180) % 360) - 180",
            "latitude_order": "ascending",
            "interpolation_performed": "false",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding={"rainfall": {"dtype": "float32", "zlib": True, "complevel": 4}},
    )
    return output_path


def align_gfs_imd_smoke_test(
    *,
    imd_path: Path,
    gfs_dir: Path,
    aligned_dir: Path,
    paired_path: Path,
    time_report_path: Path,
    grid_report_path: Path,
    cycle: str,
    expected_days: int,
    rainfall_window_start_utc: int,
    rainfall_window_hours: int,
) -> dict[str, Any]:
    """Align the seven processed GFS fields to IMD using exact coordinates."""

    if not imd_path.is_file():
        raise AlignmentError(f"IMD input does not exist: {imd_path}")
    with xr.open_dataset(imd_path) as imd_source:
        required = {"TIME", "LATITUDE", "LONGITUDE", "RAINFALL"}
        missing = required - set(imd_source.variables)
        if missing:
            raise AlignmentError(f"IMD input is missing variables: {sorted(missing)}")
        observation = imd_source["RAINFALL"].load().rename(
            {"TIME": "time", "LATITUDE": "latitude", "LONGITUDE": "longitude"}
        )
        observation_attrs = dict(imd_source.attrs)

    observation = observation.transpose("time", "latitude", "longitude")
    if observation.sizes["time"] != expected_days:
        raise AlignmentError(
            f"Expected {expected_days} IMD days, found {observation.sizes['time']}"
        )
    observation_times = np.asarray(observation.time.values)
    if not np.all(np.diff(observation_times) == np.timedelta64(1, "D")):
        raise AlignmentError("IMD dates are not consecutive daily records.")
    imd_latitude = np.asarray(observation.latitude.values, dtype=np.float64)
    imd_longitude = np.asarray(observation.longitude.values, dtype=np.float64)
    if not np.all(np.diff(imd_latitude) > 0) or not np.all(np.diff(imd_longitude) > 0):
        raise AlignmentError("IMD latitude and longitude must both be ascending.")

    temporal_rows: list[dict[str, Any]] = []
    grid_records: list[dict[str, Any]] = []
    aligned_fields: list[xr.DataArray] = []
    aligned_paths: list[str] = []
    native_grid_reference: dict[str, Any] | None = None
    for observation_time in observation_times:
        observation_label = _datetime64_to_utc(observation_time)
        initialization_date = observation_label - timedelta(days=1)
        date_token = initialization_date.strftime("%Y%m%d")
        gfs_path = gfs_dir / f"gfs_{date_token}_{cycle}_d01_03-03.nc"
        if not gfs_path.is_file():
            raise AlignmentError(f"Missing GFS field for IMD {observation_label.date()}: {gfs_path}")
        with xr.open_dataset(gfs_path) as gfs_source:
            if "rainfall" not in gfs_source:
                raise AlignmentError(f"GFS input has no rainfall variable: {gfs_path}")
            temporal = validate_temporal_pair(
                gfs_source.attrs,
                observation_time,
                rainfall_window_start_utc=rainfall_window_start_utc,
                rainfall_window_hours=rainfall_window_hours,
            )
            temporal["gfs_file"] = str(gfs_path)
            temporal_rows.append(temporal)
            native = gfs_source["rainfall"].load()
            source_attrs = dict(gfs_source.attrs)

        native_nan_count = int(np.isnan(native.values).sum())
        aligned, diagnostics = normalize_and_select_exact_grid(
            native, imd_latitude, imd_longitude
        )
        aligned_nan_count = int(np.isnan(aligned.values).sum())
        if aligned_nan_count != 0:
            raise AlignmentError(
                f"Unexpected GFS NaNs after exact selection for {date_token}: "
                f"{aligned_nan_count}"
            )
        if native_grid_reference is None:
            native_grid_reference = diagnostics
        elif any(
            diagnostics[key] != native_grid_reference[key]
            for key in (
                "native_shape",
                "native_latitude_first",
                "native_latitude_last",
                "native_longitude_first",
                "native_longitude_last",
            )
        ):
            raise AlignmentError(f"Native GFS grid changed for {date_token}")

        scalar_coords = [
            name
            for name in ("initialization_time", "valid_start", "valid_end")
            if name in aligned.coords
        ]
        if scalar_coords:
            aligned = aligned.drop_vars(scalar_coords)
        aligned.attrs.update(
            {
                "units": "mm",
                "spatial_alignment": "exact_coordinate_selection",
                "interpolation_performed": "false",
            }
        )
        aligned_path = aligned_dir / f"gfs_{date_token}_{cycle}_d01_03-03.nc"
        _save_aligned_field(aligned, source_attrs, aligned_path)
        aligned_paths.append(str(aligned_path))
        aligned_fields.append(aligned.expand_dims(time=[observation_time]))
        grid_records.append(
            {
                "initialization_time": temporal["initialization_time"],
                "imd_observation_date": temporal["imd_observation_date"],
                "native_gfs_nan_count": native_nan_count,
                "aligned_gfs_nan_count": aligned_nan_count,
                **diagnostics,
                "aligned_file": str(aligned_path),
                "status": "PASS",
            }
        )

    _write_csv(temporal_rows, time_report_path)
    forecast = xr.concat(aligned_fields, dim="time").rename("forecast_rainfall")
    forecast = forecast.transpose("time", "latitude", "longitude").astype("float32")
    observed = observation.rename("observed_rainfall").astype("float32")
    forecast, observed = xr.align(forecast, observed, join="exact", copy=False)
    valid_mask = (forecast.notnull() & observed.notnull()).astype("int8").rename("valid_mask")
    valid_mask.attrs = {
        "long_name": "Cells where both forecast and observation are valid",
        "flag_values": [0, 1],
        "flag_meanings": "invalid valid",
    }
    forecast.attrs.update({"units": "mm", "alignment": "exact; no interpolation"})
    observed.attrs.update({"units": "mm", "missing_values": "NaN preserved from IMD"})
    paired = xr.Dataset(
        {
            "forecast_rainfall": forecast,
            "observed_rainfall": observed,
            "valid_mask": valid_mask,
        }
    )
    paired.attrs = {
        "title": "Temporally and spatially aligned GFS and IMD daily rainfall",
        "temporal_alignment": "GFS 03Z-to-03Z window paired with IMD ending date",
        "spatial_alignment": "exact_coordinate_selection",
        "interpolation_performed": "false",
        "imd_temporal_convention_source": observation_attrs.get(
            "temporal_convention_source", ""
        ),
    }
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_netcdf(
        paired_path,
        engine="netcdf4",
        encoding={
            "forecast_rainfall": {"dtype": "float32", "zlib": True, "complevel": 4},
            "observed_rainfall": {"dtype": "float32", "zlib": True, "complevel": 4},
            "valid_mask": {"dtype": "int8", "zlib": True, "complevel": 4},
        },
    )

    observed_nan_source = np.isnan(observation.values)
    observed_nan_paired = np.isnan(paired.observed_rainfall.values)
    valid_counts = [int(value) for value in paired.valid_mask.sum(("latitude", "longitude")).values]
    checks = {
        "all_pairs_exist": len(temporal_rows) == expected_days,
        "same_shape": paired.forecast_rainfall.shape == paired.observed_rainfall.shape,
        "same_latitude_coordinates": np.array_equal(
            paired.forecast_rainfall.latitude.values,
            paired.observed_rainfall.latitude.values,
        ),
        "same_longitude_coordinates": np.array_equal(
            paired.forecast_rainfall.longitude.values,
            paired.observed_rainfall.longitude.values,
        ),
        "all_periods_match": all(row["period_match"] for row in temporal_rows),
        "imd_nan_mask_preserved": np.array_equal(
            observed_nan_source, observed_nan_paired
        ),
        "no_gfs_nans_introduced": int(np.isnan(paired.forecast_rainfall.values).sum())
        == 0,
        "no_interpolation": all(
            not record["interpolation_performed"] for record in grid_records
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise AlignmentError(f"Final paired-dataset QA failed: {failed}")

    grid_report = {
        "status": "PASS",
        "alignment_method": "exact_coordinate_selection",
        "interpolation_performed": False,
        "longitude_normalization": "((longitude + 180) % 360) - 180",
        "gfs_native_grid": native_grid_reference,
        "imd_grid": {
            "shape": [int(imd_latitude.size), int(imd_longitude.size)],
            "latitude_first": float(imd_latitude[0]),
            "latitude_last": float(imd_latitude[-1]),
            "longitude_first": float(imd_longitude[0]),
            "longitude_last": float(imd_longitude[-1]),
            "latitude_spacing_degrees": float(np.diff(imd_latitude)[0]),
            "longitude_spacing_degrees": float(np.diff(imd_longitude)[0]),
            "orientation": "latitude_ascending_longitude_ascending",
        },
        "aligned_grid": {
            "shape": [int(imd_latitude.size), int(imd_longitude.size)],
            "maximum_latitude_difference": max(
                record["maximum_latitude_difference"] for record in grid_records
            ),
            "maximum_longitude_difference": max(
                record["maximum_longitude_difference"] for record in grid_records
            ),
            "latitude_spacing_degrees": float(np.diff(imd_latitude)[0]),
            "longitude_spacing_degrees": float(np.diff(imd_longitude)[0]),
            "orientation": "latitude_ascending_longitude_ascending",
        },
        "records": grid_records,
        "paired_dataset": {
            "path": str(paired_path),
            "shape": list(paired.forecast_rainfall.shape),
            "valid_cell_counts": valid_counts,
            "total_valid_cells": int(sum(valid_counts)),
            "observed_nan_count": int(observed_nan_paired.sum()),
            "forecast_nan_count": int(np.isnan(paired.forecast_rainfall.values).sum()),
        },
        "aligned_files": aligned_paths,
        "checks": checks,
    }
    _write_json(grid_report, grid_report_path)
    return grid_report


def align_forecast_observation(
    forecast_path: Path,
    observation_path: Path,
    output_path: Path,
    alignment_config: dict,
) -> Path:
    """Align one forecast to one observation grid using exact coordinates only."""

    with xr.open_dataset(forecast_path) as forecast_source, xr.open_dataset(
        observation_path
    ) as observation_source:
        forecast_variable = next(iter(forecast_source.data_vars))
        observation_variable = next(iter(observation_source.data_vars))
        forecast = forecast_source[forecast_variable].load()
        observation = observation_source[observation_variable].load()
    rename = {}
    for name in alignment_config["latitude_names"]:
        if name in observation.coords:
            rename[name] = "latitude"
            break
    for name in alignment_config["longitude_names"]:
        if name in observation.coords:
            rename[name] = "longitude"
            break
    observation = observation.rename(rename)
    aligned, _ = normalize_and_select_exact_grid(
        forecast, observation.latitude.values, observation.longitude.values
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_dataset(name=forecast_variable).to_netcdf(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/smoke_test.yaml"))
    parser.add_argument(
        "--imd-file",
        type=Path,
        default=Path("data/processed/imd/imd_20240901_20240907.nc"),
    )
    parser.add_argument("--gfs-dir", type=Path, default=Path("data/processed/gfs"))
    parser.add_argument(
        "--aligned-dir", type=Path, default=Path("data/processed/gfs_aligned")
    )
    parser.add_argument(
        "--paired-file",
        type=Path,
        default=Path("data/processed/paired/gfs_imd_20240901_20240907.nc"),
    )
    args = parser.parse_args()

    from src.config import load_config

    config = load_config(args.config)
    report_dir = Path(config["paths"]["report_dir"])
    result = align_gfs_imd_smoke_test(
        imd_path=args.imd_file,
        gfs_dir=args.gfs_dir,
        aligned_dir=args.aligned_dir,
        paired_path=args.paired_file,
        time_report_path=report_dir / "03_time_alignment.csv",
        grid_report_path=report_dir / "04_grid_alignment.json",
        cycle=config["gfs"]["cycle"],
        expected_days=config["smoke_test"]["expected_days"],
        rainfall_window_start_utc=config["verification"]["rainfall_window_start_utc"],
        rainfall_window_hours=config["verification"]["rainfall_window_hours"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
