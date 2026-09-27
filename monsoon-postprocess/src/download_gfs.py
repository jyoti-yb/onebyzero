"""Download geographically subsetted NOAA GFS 0.25-degree GRIB2 files.

This module performs no network access at import time. Precipitation semantics
are deliberately outside its scope; it only downloads the requested fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

NOMADS_FILTER_ENDPOINT = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GFS_PRODUCT = "gfs_0p25"
MANIFEST_FILENAME = "gfs_download_manifest.json"
DEFAULT_SURFACE_LEVELS = (
    "surface",
    "mean_sea_level",
    "entire_atmosphere_(considered_as_a_single_layer)",
)
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


class DownloadError(RuntimeError):
    """Raised when NOMADS does not return a usable GRIB2 file."""


class ForecastAdapter(Protocol):
    """Interface downstream stages use for forecast file discovery."""

    source_name: str

    def list_local_files(self) -> list[Path]:
        """Return local forecast files for the smoke-test window."""

    def download(
        self,
        *,
        date: str,
        forecast_hour: int,
        manifest_dir: Path,
        timeout: float = 120.0,
        retries: int = 4,
        backoff_factor: float = 1.0,
    ) -> dict[str, Any]:
        """Download one forecast file and return its manifest entry."""


def _validate_date(date: str) -> str:
    if not isinstance(date, str) or re.fullmatch(r"\d{8}", date) is None:
        raise ValueError("date must use YYYYMMDD format.")
    try:
        parsed = datetime.strptime(date, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("date must be a valid calendar date in YYYYMMDD format.") from exc
    if parsed.strftime("%Y%m%d") != date:
        raise ValueError("date must use YYYYMMDD format.")
    return date


def _validate_cycle(cycle: str) -> str:
    if cycle not in {"00", "06", "12", "18"}:
        raise ValueError('cycle must be one of "00", "06", "12", or "18".')
    return cycle


def _validate_forecast_hour(forecast_hour: int) -> int:
    if isinstance(forecast_hour, bool) or not isinstance(forecast_hour, int):
        raise TypeError("forecast_hour must be an integer.")
    if not 0 <= forecast_hour <= 384:
        raise ValueError("forecast_hour must be between 0 and 384.")
    return forecast_hour


def _validate_domain(domain: Mapping[str, float]) -> dict[str, float]:
    required = ("south", "north", "west", "east")
    missing = [name for name in required if name not in domain]
    if missing:
        raise ValueError(f"domain is missing required bounds: {missing}")

    bounds: dict[str, float] = {}
    for name in required:
        value = domain[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"domain.{name} must be numeric.")
        bounds[name] = float(value)
        if not math.isfinite(bounds[name]):
            raise ValueError(f"domain.{name} must be finite.")

    if not -90 <= bounds["south"] < bounds["north"] <= 90:
        raise ValueError("domain must satisfy -90 <= south < north <= 90.")
    if not -180 <= bounds["west"] < bounds["east"] <= 360:
        raise ValueError("domain must satisfy -180 <= west < east <= 360.")
    if bounds["north"] - bounds["south"] >= 180 and bounds["east"] - bounds["west"] >= 360:
        raise ValueError("A global domain is not allowed; request a geographic subset.")
    return bounds


def _normalize_variables(variables: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for variable in variables:
        if not isinstance(variable, str) or re.fullmatch(r"[A-Za-z0-9-]+", variable) is None:
            raise ValueError(f"Invalid NOMADS variable name: {variable!r}")
        value = variable.upper()
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one requested variable is required.")
    return normalized


def _normalize_pressure_levels(pressure_levels: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    for level in pressure_levels:
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError(f"Pressure level must be a positive integer in mb: {level!r}")
        if level not in normalized:
            normalized.append(level)
    return normalized


def _normalize_additional_levels(levels: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for level in levels:
        if not isinstance(level, str) or not level.strip():
            raise ValueError("Additional NOMADS levels must be non-empty strings.")
        slug = re.sub(r"\s+", "_", level.strip())
        if re.fullmatch(r"[A-Za-z0-9_().+-]+", slug) is None:
            raise ValueError(f"Invalid NOMADS level name: {level!r}")
        if slug not in normalized:
            normalized.append(slug)
    return normalized


def _format_coordinate(value: float) -> str:
    return f"{value:g}"


def build_output_filename(date: str, cycle: str, forecast_hour: int) -> str:
    """Construct the stable local filename for one GFS forecast file."""

    _validate_date(date)
    _validate_cycle(cycle)
    _validate_forecast_hour(forecast_hour)
    return f"gfs_{date}_{cycle}_f{forecast_hour:03d}.grib2"


def build_gfs_url(
    *,
    date: str,
    cycle: str,
    forecast_hour: int,
    domain: Mapping[str, float],
    variables: Sequence[str],
    pressure_levels: Sequence[int] = (),
    additional_levels: Sequence[str] = (),
) -> str:
    """Construct a NOMADS filter URL for a subsetted GFS GRIB2 request."""

    date = _validate_date(date)
    cycle = _validate_cycle(cycle)
    forecast_hour = _validate_forecast_hour(forecast_hour)
    bounds = _validate_domain(domain)
    requested_variables = _normalize_variables(variables)
    requested_pressure_levels = _normalize_pressure_levels(pressure_levels)
    requested_additional_levels = _normalize_additional_levels(additional_levels)
    if not requested_pressure_levels and not requested_additional_levels:
        raise ValueError("At least one pressure or additional NOMADS level is required.")

    query: list[tuple[str, str]] = [
        ("file", f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"),
    ]
    query.extend((f"var_{variable}", "on") for variable in requested_variables)
    query.extend((f"lev_{level}_mb", "on") for level in requested_pressure_levels)
    query.extend((f"lev_{level}", "on") for level in requested_additional_levels)
    query.extend(
        [
            ("subregion", ""),
            ("leftlon", _format_coordinate(bounds["west"])),
            ("rightlon", _format_coordinate(bounds["east"])),
            ("toplat", _format_coordinate(bounds["north"])),
            ("bottomlat", _format_coordinate(bounds["south"])),
            ("dir", f"/gfs.{date}/{cycle}/atmos"),
        ]
    )
    return f"{NOMADS_FILTER_ENDPOINT}?{urlencode(query)}"


def _create_retrying_session(retries: int, backoff_factor: float) -> Any:
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for GFS downloads; install requirements.txt first."
        ) from exc

    if retries < 0:
        raise ValueError("retries must be zero or greater.")
    if backoff_factor < 0:
        raise ValueError("backoff_factor must be zero or greater.")
    retry_policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=RETRYABLE_STATUS_CODES,
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    return session


def _append_manifest_entry(manifest_dir: Path, entry: dict[str, Any]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / MANIFEST_FILENAME
    entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownloadError(f"Could not read existing manifest {manifest_path}: {exc}") from exc
        if not isinstance(existing, list):
            raise DownloadError(f"Existing manifest must contain a JSON list: {manifest_path}")
        entries = existing
    entries.append(entry)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_dir,
            prefix=f".{MANIFEST_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(entries, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest_path


def download_gfs_file(
    *,
    date: str,
    cycle: str,
    forecast_hour: int,
    domain: Mapping[str, float],
    variables: Sequence[str],
    pressure_levels: Sequence[int],
    output_dir: Path,
    manifest_dir: Path,
    additional_levels: Sequence[str] = (),
    timeout: float = 120.0,
    retries: int = 4,
    backoff_factor: float = 1.0,
    session: Any | None = None,
) -> dict[str, Any]:
    """Download one subsetted file atomically and record its manifest entry."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")
    url = build_gfs_url(
        date=date,
        cycle=cycle,
        forecast_hour=forecast_hour,
        domain=domain,
        variables=variables,
        pressure_levels=pressure_levels,
        additional_levels=additional_levels,
    )
    filename = build_output_filename(date, cycle, forecast_hour)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    owned_session = session is None
    active_session = session or _create_retrying_session(retries, backoff_factor)
    temporary_path: Path | None = None
    renamed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=f".{filename}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            with active_session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        if temporary_path.stat().st_size < 4:
            raise DownloadError("NOMADS returned an empty or truncated response.")
        with temporary_path.open("rb") as handle:
            if handle.read(4) != b"GRIB":
                raise DownloadError("NOMADS response is not a GRIB file.")

        os.replace(temporary_path, destination)
        renamed = True
        entry = {
            "source": "NOAA GFS",
            "product": GFS_PRODUCT,
            "date": date,
            "cycle": cycle,
            "forecast_hour": forecast_hour,
            "url": url,
            "local_path": str(destination.resolve()),
            "downloaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        _append_manifest_entry(manifest_dir, entry)
        return entry
    except Exception:
        if renamed:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if owned_session:
            active_session.close()


def _configured_variables(variable_config: Mapping[str, Any]) -> list[str]:
    variables = list(variable_config["surface"])
    for level_variables in variable_config["pressure"].values():
        variables.extend(level_variables)
    return list(dict.fromkeys(variables))


@dataclass(frozen=True)
class GFSForecastAdapter:
    """GFS adapter for local discovery and explicit one-file downloads."""

    source_dir: Path
    file_glob: str
    expected_days: int
    cycle: str
    model: str
    lead_day: int
    domain: dict[str, float]
    variables: dict[str, Any]
    source_name: str = "gfs"

    def list_local_files(self) -> list[Path]:
        return sorted(self.source_dir.glob(self.file_glob))

    def download(
        self,
        *,
        date: str,
        forecast_hour: int,
        manifest_dir: Path,
        timeout: float = 120.0,
        retries: int = 4,
        backoff_factor: float = 1.0,
    ) -> dict[str, Any]:
        return download_gfs_file(
            date=date,
            cycle=self.cycle,
            forecast_hour=forecast_hour,
            domain=self.domain,
            variables=_configured_variables(self.variables),
            pressure_levels=[int(level) for level in self.variables["pressure"]],
            additional_levels=DEFAULT_SURFACE_LEVELS,
            output_dir=self.source_dir,
            manifest_dir=manifest_dir,
            timeout=timeout,
            retries=retries,
            backoff_factor=backoff_factor,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="GFS initialization date in YYYYMMDD format")
    parser.add_argument("--forecast-hour", required=True, type=int, help="Forecast lead hour")
    parser.add_argument("--cycle", choices=("00", "06", "12", "18"))
    parser.add_argument("--config", type=Path, default=Path("config/smoke_test.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--south", type=float)
    parser.add_argument("--north", type=float)
    parser.add_argument("--west", type=float)
    parser.add_argument("--east", type=float)
    parser.add_argument("--variable", action="append", dest="variables")
    parser.add_argument("--pressure-level", action="append", type=int, dest="pressure_levels")
    parser.add_argument("--level", action="append", dest="additional_levels")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--backoff-factor", type=float, default=1.0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from src.config import load_config

    config = load_config(args.config)
    project_root = args.config.resolve().parents[1]

    domain = dict(config["domain"])
    for name in ("south", "north", "west", "east"):
        override = getattr(args, name)
        if override is not None:
            domain[name] = override

    use_config_variables = args.variables is None
    variables = args.variables or _configured_variables(config["variables"])
    if args.pressure_levels is not None:
        pressure_levels = args.pressure_levels
    elif use_config_variables:
        pressure_levels = [int(level) for level in config["variables"]["pressure"]]
    else:
        pressure_levels = []
    if args.additional_levels is not None:
        additional_levels = args.additional_levels
    elif use_config_variables:
        additional_levels = list(DEFAULT_SURFACE_LEVELS)
    else:
        additional_levels = []

    output_dir = args.output_dir or project_root / config["paths"]["raw_gfs_dir"]
    manifest_dir = args.manifest_dir or project_root / config["paths"]["manifest_dir"]
    entry = download_gfs_file(
        date=args.date,
        cycle=args.cycle or config["gfs"]["cycle"],
        forecast_hour=args.forecast_hour,
        domain=domain,
        variables=variables,
        pressure_levels=pressure_levels,
        additional_levels=additional_levels,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        timeout=args.timeout,
        retries=args.retries,
        backoff_factor=args.backoff_factor,
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
