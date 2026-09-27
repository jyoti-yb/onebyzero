"""Load and validate smoke-test configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the smoke-test configuration is missing or invalid."""


def _mapping(parent: Mapping[str, Any], key: str, path: str = "config") -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}.{key} must be a mapping.")
    return value


def _number(parent: Mapping[str, Any], key: str, path: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key} must be a number.")
    return float(value)


def _integer(parent: Mapping[str, Any], key: str, path: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key} must be an integer.")
    return value


def _string(parent: Mapping[str, Any], key: str, path: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key} must be a non-empty string.")
    return value


def _string_list(parent: Mapping[str, Any], key: str, path: str) -> list[str]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ConfigError(f"{path}.{key} must be a non-empty list of strings.")
    if len(value) != len(set(value)):
        raise ConfigError(f"{path}.{key} must not contain duplicate values.")
    return value


def validate_config(config: Any) -> dict[str, Any]:
    """Validate the complete smoke-test configuration and return it."""

    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping.")

    project = _mapping(config, "project")
    _string(project, "name", "project")
    if _string(project, "python", "project") != "3.11":
        raise ConfigError('project.python must be "3.11".')

    smoke_test = _mapping(config, "smoke_test")
    if _integer(smoke_test, "expected_days", "smoke_test") != 7:
        raise ConfigError("smoke_test.expected_days must be 7 for the initial smoke test.")

    domain = _mapping(config, "domain")
    south = _number(domain, "south", "domain")
    north = _number(domain, "north", "domain")
    west = _number(domain, "west", "domain")
    east = _number(domain, "east", "domain")
    if not (-90 <= south < north <= 90):
        raise ConfigError("domain must satisfy -90 <= south < north <= 90.")
    if not (-180 <= west < east <= 360):
        raise ConfigError("domain must satisfy -180 <= west < east <= 360.")

    gfs = _mapping(config, "gfs")
    cycle = _string(gfs, "cycle", "gfs")
    if cycle not in {"00", "06", "12", "18"}:
        raise ConfigError('gfs.cycle must be one of "00", "06", "12", or "18".')
    _string(gfs, "model", "gfs")
    if _integer(gfs, "lead_day", "gfs") < 0:
        raise ConfigError("gfs.lead_day must be zero or greater.")
    _string(gfs, "local_file_glob", "gfs")
    precipitation_variable = _string(gfs, "precipitation_variable", "gfs")
    precipitation_aliases = _string_list(gfs, "precipitation_variable_aliases", "gfs")

    verification = _mapping(config, "verification")
    window_start = _integer(verification, "rainfall_window_start_utc", "verification")
    window_hours = _integer(verification, "rainfall_window_hours", "verification")
    if not 0 <= window_start <= 23:
        raise ConfigError("verification.rainfall_window_start_utc must be between 0 and 23.")
    if window_hours <= 0:
        raise ConfigError("verification.rainfall_window_hours must be greater than zero.")
    metrics = _string_list(verification, "metrics", "verification")
    unsupported_metrics = set(metrics) - {"bias", "mae", "rmse", "correlation"}
    if unsupported_metrics:
        raise ConfigError(f"verification.metrics contains unsupported values: {sorted(unsupported_metrics)}")

    variables = _mapping(config, "variables")
    surface_variables = _string_list(variables, "surface", "variables")
    if precipitation_variable not in surface_variables:
        raise ConfigError("gfs.precipitation_variable must also appear in variables.surface.")
    if precipitation_variable not in precipitation_aliases:
        raise ConfigError("gfs.precipitation_variable must appear in gfs.precipitation_variable_aliases.")
    pressure = _mapping(variables, "pressure", "variables")
    if not pressure:
        raise ConfigError("variables.pressure must contain at least one pressure level.")
    for level, level_variables in pressure.items():
        try:
            numeric_level = int(level)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"variables.pressure level {level!r} must be an integer.") from exc
        if numeric_level <= 0:
            raise ConfigError(f"variables.pressure level {level!r} must be positive.")
        _string_list({"values": level_variables}, "values", f"variables.pressure.{level}")

    quality = _mapping(config, "quality")
    rainfall_min = _number(quality, "rainfall_min_mm", "quality")
    rainfall_max = _number(quality, "rainfall_max_reasonable_mm", "quality")
    rh_min = _number(quality, "rh_min", "quality")
    rh_max = _number(quality, "rh_max", "quality")
    valid_fraction = _number(quality, "min_valid_grid_fraction", "quality")
    if rainfall_min > rainfall_max:
        raise ConfigError("quality.rainfall_min_mm must not exceed rainfall_max_reasonable_mm.")
    if not 0 <= rh_min <= rh_max <= 100:
        raise ConfigError("quality RH bounds must satisfy 0 <= rh_min <= rh_max <= 100.")
    if not 0 < valid_fraction <= 1:
        raise ConfigError("quality.min_valid_grid_fraction must be in the interval (0, 1].")

    paths = _mapping(config, "paths")
    for key in (
        "raw_gfs_dir",
        "raw_imd_dir",
        "processed_gfs_dir",
        "processed_imd_dir",
        "paired_dir",
        "manifest_dir",
        "report_dir",
    ):
        _string(paths, key, "paths")

    observation = _mapping(config, "observation")
    _string(observation, "provider", "observation")
    _string(observation, "local_file_glob", "observation")
    _string_list(observation, "precipitation_variable_candidates", "observation")

    alignment = _mapping(config, "alignment")
    if _number(alignment, "time_tolerance_hours", "alignment") < 0:
        raise ConfigError("alignment.time_tolerance_hours must be zero or greater.")
    if _string(alignment, "regrid_method", "alignment") not in {"nearest", "linear"}:
        raise ConfigError('alignment.regrid_method must be "nearest" or "linear".')
    _string_list(alignment, "latitude_names", "alignment")
    _string_list(alignment, "longitude_names", "alignment")

    return config


def load_config(path: Path) -> dict[str, Any]:
    """Read YAML from *path* and fail early with an actionable error."""

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration file {path}: {exc}") from exc

    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    return validate_config(config)
