"""Inspect GRIB2 metadata without modifying the source files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_REPORT_PATH = Path("reports/smoke_test/02_grib_metadata.csv")
REPORT_COLUMNS = (
    "file",
    "group_index",
    "message_index",
    "variable",
    "variables",
    "dimensions",
    "coordinates",
    "variable_dimensions",
    "GRIB_shortName",
    "GRIB_units",
    "GRIB_stepType",
    "GRIB_stepRange",
    "GRIB_forecastTime",
    "GRIB_typeOfLevel",
    "GRIB_level",
    "forecast_reference_time",
    "valid_time",
    "forecast_hour",
    "pressure_level_hpa",
    "pressure_level_type",
)
PRESSURE_COORDINATES = (
    ("isobaricInhPa", 1.0),
    ("isobaricInPa", 0.01),
)


class GribInspectionError(RuntimeError):
    """Raised when a GRIB file cannot be decoded or inspected."""


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    value = getattr(value, "values", value)

    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) in {"M", "m"}:
        flattened = value.reshape(-1) if hasattr(value, "reshape") else [value]
        return [str(item) for item in flattened]

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _display_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if hasattr(value, "item"):
        try:
            item = value.item()
        except (TypeError, ValueError):
            item = value
        if item is not value:
            return _display_value(item)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serialize(values: Sequence[Any]) -> str:
    return json.dumps([_display_value(value) for value in values], ensure_ascii=True)


def _coordinate_values(dataset: Any, *names: str) -> tuple[str | None, list[Any]]:
    coordinates = getattr(dataset, "coords", {})
    for name in names:
        if name in coordinates:
            return name, _flatten(coordinates[name])
    return None, []


def _hours_from_value(value: Any) -> float | int | None:
    if hasattr(value, "item"):
        try:
            item = value.item()
        except (TypeError, ValueError):
            item = value
        if item is not value:
            return _hours_from_value(item)
    if isinstance(value, timedelta):
        hours = value.total_seconds() / 3600
        return int(hours) if hours.is_integer() else hours
    if hasattr(value, "total_seconds"):
        hours = value.total_seconds() / 3600
        return int(hours) if float(hours).is_integer() else float(hours)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value

    match = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*"
        r"(hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|milliseconds?|microseconds?|nanoseconds?)\s*",
        str(value),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("day"):
        hours = amount * 24
    elif unit.startswith(("hour", "hr")):
        hours = amount
    elif unit.startswith(("minute", "min")):
        hours = amount / 60
    elif unit.startswith(("second", "sec")):
        hours = amount / 3600
    elif unit.startswith("millisecond"):
        hours = amount / 3_600_000
    elif unit.startswith("microsecond"):
        hours = amount / 3_600_000_000
    else:
        hours = amount / 3_600_000_000_000
    return int(hours) if hours.is_integer() else hours


def _forecast_hours(dataset: Any, variable_attrs: Mapping[str, Any]) -> list[float | int]:
    _, steps = _coordinate_values(dataset, "step")
    hours = [hour for step in steps if (hour := _hours_from_value(step)) is not None]
    if hours:
        return hours
    forecast_time = variable_attrs.get("GRIB_forecastTime")
    if forecast_time is None:
        return []
    hour = _hours_from_value(forecast_time)
    return [] if hour is None else [hour]


def _pressure_metadata(
    dataset: Any,
    variable_attrs: Mapping[str, Any],
) -> tuple[list[float | int], str]:
    for coordinate_name, scale in PRESSURE_COORDINATES:
        _, values = _coordinate_values(dataset, coordinate_name)
        if values:
            levels = [float(value) * scale for value in values]
            normalized = [int(level) if level.is_integer() else level for level in levels]
            return normalized, coordinate_name

    level_type = str(variable_attrs.get("GRIB_typeOfLevel", ""))
    level = variable_attrs.get("GRIB_level")
    if level is None or level_type not in {"isobaricInhPa", "isobaricInPa"}:
        return [], level_type
    scale = 0.01 if level_type == "isobaricInPa" else 1.0
    pressure = float(level) * scale
    return [int(pressure) if pressure.is_integer() else pressure], level_type


def _metadata_rows_for_group(path: Path, group_index: int, dataset: Any) -> list[dict[str, Any]]:
    """Convert one xarray-like GRIB group into CSV-ready metadata rows."""

    dimensions = {
        str(name): int(size) for name, size in dict(getattr(dataset, "sizes", {})).items()
    }
    coordinates = [str(name) for name in getattr(dataset, "coords", {})]
    data_variables = getattr(dataset, "data_vars", {})
    variable_names = [str(name) for name in data_variables]
    dataset_attrs = getattr(dataset, "attrs", {})
    _, forecast_reference_times = _coordinate_values(dataset, "time", "forecast_reference_time")
    _, valid_times = _coordinate_values(dataset, "valid_time")

    rows: list[dict[str, Any]] = []
    for variable_name, variable in data_variables.items():
        attrs = dict(dataset_attrs)
        attrs.update(getattr(variable, "attrs", {}))
        pressure_levels, pressure_type = _pressure_metadata(dataset, attrs)
        rows.append(
            {
                "file": str(path),
                "group_index": group_index,
                "message_index": "",
                "variable": str(variable_name),
                "variables": json.dumps(variable_names),
                "dimensions": json.dumps(dimensions, sort_keys=True),
                "coordinates": json.dumps(coordinates),
                "variable_dimensions": json.dumps(
                    [str(name) for name in getattr(variable, "dims", ())]
                ),
                "GRIB_shortName": str(attrs.get("GRIB_shortName", variable_name)),
                "GRIB_units": str(attrs.get("GRIB_units", attrs.get("units", ""))),
                "GRIB_stepType": str(attrs.get("GRIB_stepType", "")),
                "GRIB_stepRange": str(attrs.get("GRIB_stepRange", "")),
                "GRIB_forecastTime": str(attrs.get("GRIB_forecastTime", "")),
                "GRIB_typeOfLevel": str(attrs.get("GRIB_typeOfLevel", "")),
                "GRIB_level": str(attrs.get("GRIB_level", "")),
                "forecast_reference_time": _serialize(forecast_reference_times),
                "valid_time": _serialize(valid_times),
                "forecast_hour": _serialize(_forecast_hours(dataset, attrs)),
                "pressure_level_hpa": _serialize(pressure_levels),
                "pressure_level_type": pressure_type,
            }
        )
    return rows


def _grib_datetime(date_value: Any, time_value: Any) -> datetime | None:
    try:
        date_text = f"{int(date_value):08d}"
        time_text = f"{int(time_value):04d}"
        return datetime.strptime(f"{date_text}{time_text}", "%Y%m%d%H%M")
    except (TypeError, ValueError):
        return None


def _read_message_metadata(path: Path) -> list[dict[str, Any]]:
    """Read every GRIB message header so duplicate APCP records are not collapsed."""

    try:
        from eccodes import (
            CodesInternalError,
            codes_get,
            codes_grib_new_from_file,
            codes_release,
        )
    except ImportError as exc:
        raise GribInspectionError("ecCodes is required to enumerate GRIB messages.") from exc

    def get_optional(handle: Any, key: str) -> Any:
        try:
            return codes_get(handle, key)
        except (CodesInternalError, KeyError):
            return ""

    messages: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            while True:
                grib_handle = codes_grib_new_from_file(handle)
                if grib_handle is None:
                    break
                try:
                    data_date = get_optional(grib_handle, "dataDate")
                    data_time = get_optional(grib_handle, "dataTime")
                    validity_date = get_optional(grib_handle, "validityDate")
                    validity_time = get_optional(grib_handle, "validityTime")
                    reference_time = _grib_datetime(data_date, data_time)
                    valid_time = _grib_datetime(validity_date, validity_time)
                    forecast_hours: list[float | int] = []
                    if reference_time is not None and valid_time is not None:
                        hours = (valid_time - reference_time).total_seconds() / 3600
                        forecast_hours = [int(hours) if hours.is_integer() else hours]

                    messages.append(
                        {
                            "message_index": len(messages) + 1,
                            "GRIB_shortName": str(get_optional(grib_handle, "shortName")),
                            "GRIB_units": str(get_optional(grib_handle, "units")),
                            "GRIB_stepType": str(get_optional(grib_handle, "stepType")),
                            "GRIB_stepRange": str(get_optional(grib_handle, "stepRange")),
                            "GRIB_forecastTime": str(
                                get_optional(grib_handle, "forecastTime")
                            ),
                            "GRIB_typeOfLevel": str(
                                get_optional(grib_handle, "typeOfLevel")
                            ),
                            "GRIB_level": str(get_optional(grib_handle, "level")),
                            "forecast_reference_time": _serialize(
                                [] if reference_time is None else [reference_time]
                            ),
                            "valid_time": _serialize(
                                [] if valid_time is None else [valid_time]
                            ),
                            "forecast_hour": _serialize(forecast_hours),
                        }
                    )
                finally:
                    codes_release(grib_handle)
    except OSError as exc:
        raise GribInspectionError(f"Could not read GRIB messages from {path}: {exc}") from exc
    return messages


def _merge_group_and_message_rows(
    path: Path,
    group_rows: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    matched_group_rows: set[int] = set()

    for message in messages:
        short_name = str(message["GRIB_shortName"]).casefold()
        candidates = [
            (index, row)
            for index, row in enumerate(group_rows)
            if short_name
            in {
                str(row["variable"]).casefold(),
                str(row["GRIB_shortName"]).casefold(),
            }
        ]
        message_level_type = str(message["GRIB_typeOfLevel"])
        exact_candidates = [
            candidate
            for candidate in candidates
            if str(candidate[1]["GRIB_typeOfLevel"]) == message_level_type
        ]
        if exact_candidates:
            candidates = exact_candidates

        if candidates:
            group_row_index, base_row = candidates[0]
            matched_group_rows.add(group_row_index)
            merged = dict(base_row)
        else:
            merged = {column: "" for column in REPORT_COLUMNS}
            merged.update(
                {
                    "file": str(path),
                    "group_index": -1,
                    "variable": str(message["GRIB_shortName"]),
                }
            )
        merged.update(message)
        merged_rows.append(merged)

    merged_rows.extend(
        dict(row) for index, row in enumerate(group_rows) if index not in matched_group_rows
    )
    return merged_rows


def _is_apcp(row: Mapping[str, Any], aliases: Sequence[str]) -> bool:
    expected = {alias.casefold() for alias in aliases}
    return (
        str(row["variable"]).casefold() in expected
        or str(row["GRIB_shortName"]).casefold() in expected
    )


def _write_csv_report(rows: Sequence[Mapping[str, Any]], report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, report_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return report_path


def print_apcp_metadata(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print the accumulation metadata that must be understood before processing APCP."""

    print("APCP metadata")
    for index, row in enumerate(rows, start=1):
        print(f"  APCP record: {index}")
        print(f"  GRIB message: {row.get('message_index', '')}")
        print(f"  file: {row['file']}")
        print(f"  group: {row['group_index']}")
        print(f"  variable: {row['variable']}")
        print(f"  GRIB_shortName: {row['GRIB_shortName']}")
        print(f"  stepType: {row['GRIB_stepType']}")
        print(f"  stepRange: {row['GRIB_stepRange']}")
        print(f"  units: {row['GRIB_units']}")
        print(f"  valid time: {row['valid_time']}")
        print(f"  forecast time: {row['forecast_reference_time']}")
        print(f"  forecast hour: {row['forecast_hour']}")
        print(f"  typeOfLevel: {row.get('GRIB_typeOfLevel', '')}")
        print(f"  level: {row.get('GRIB_level', '')}")


def inspect_grib_metadata(
    grib_paths: Sequence[Path],
    configured_variables: Mapping[str, Any] | None = None,
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
    apcp_aliases: Sequence[str] = ("APCP", "tp"),
    print_apcp: bool = False,
    require_apcp: bool = True,
) -> dict[str, Any]:
    """Inspect every cfgrib group, write a CSV report, and return JSON-safe metadata."""

    if not grib_paths:
        raise GribInspectionError("No GRIB files were provided for inspection.")
    try:
        import cfgrib
    except ImportError as exc:
        raise GribInspectionError(
            "cfgrib and ecCodes are required to inspect GRIB2 files."
        ) from exc

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for path_value in grib_paths:
        path = Path(path_value)
        if not path.is_file():
            raise GribInspectionError(f"GRIB file does not exist or is not a file: {path}")
        datasets: list[Any] = []
        try:
            datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
            if not datasets:
                raise GribInspectionError(f"No decodable GRIB groups found in {path}")
            for group_index, dataset in enumerate(datasets):
                group_rows = _metadata_rows_for_group(path, group_index, dataset)
                if not group_rows:
                    raise GribInspectionError(
                        f"Decoded GRIB group {group_index} in {path} contains no variables."
                    )
                rows.extend(group_rows)
                groups.append(
                    {
                        "path": str(path),
                        "group_index": group_index,
                        "variables": json.loads(group_rows[0]["variables"]),
                        "dimensions": json.loads(group_rows[0]["dimensions"]),
                        "coordinates": json.loads(group_rows[0]["coordinates"]),
                        "metadata": [],
                    }
                )
        except GribInspectionError:
            raise
        except Exception as exc:
            raise GribInspectionError(f"Could not decode GRIB file {path}: {exc}") from exc
        finally:
            for dataset in datasets:
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()

        file_group_rows = [row for row in rows if row["file"] == str(path)]
        file_rows = _merge_group_and_message_rows(
            path,
            file_group_rows,
            _read_message_metadata(path),
        )
        rows = [row for row in rows if row["file"] != str(path)] + file_rows
        for group in groups:
            if group["path"] == str(path):
                group["metadata"] = [
                    row
                    for row in file_rows
                    if row["group_index"] == group["group_index"]
                ]

    _write_csv_report(rows, report_path)
    apcp_rows = [row for row in rows if _is_apcp(row, apcp_aliases)]
    if require_apcp and not apcp_rows:
        raise GribInspectionError(
            f"APCP was not found in the decoded GRIB metadata. Report written to {report_path}"
        )
    if print_apcp and apcp_rows:
        print_apcp_metadata(apcp_rows)

    return {
        "configured_variables": dict(configured_variables or {}),
        "report_path": str(report_path),
        "groups": groups,
        "apcp": apcp_rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="GRIB2 files to inspect")
    parser.add_argument("--config", type=Path, default=Path("config/smoke_test.yaml"))
    parser.add_argument("--output", type=Path, help="Metadata CSV output path")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from src.config import ConfigError, load_config

    try:
        config = load_config(args.config)
        project_root = args.config.resolve().parents[1]
        raw_gfs_dir = project_root / config["paths"]["raw_gfs_dir"]
        grib_paths = args.files or sorted(raw_gfs_dir.glob(config["gfs"]["local_file_glob"]))
        report_path = args.output or (
            project_root / config["paths"]["report_dir"] / DEFAULT_REPORT_PATH.name
        )
        result = inspect_grib_metadata(
            grib_paths,
            config["variables"],
            report_path=report_path,
            apcp_aliases=config["gfs"]["precipitation_variable_aliases"],
            print_apcp=True,
        )
    except (ConfigError, GribInspectionError) as exc:
        parser.error(str(exc))

    print(f"Wrote {len(result['groups'])} GRIB groups to {result['report_path']}")


if __name__ == "__main__":
    main()
