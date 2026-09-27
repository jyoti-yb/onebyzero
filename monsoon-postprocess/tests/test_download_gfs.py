import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from src.download_gfs import (
    NOMADS_FILTER_ENDPOINT,
    build_gfs_url,
    build_output_filename,
    download_gfs_file,
)


DOMAIN = {"south": 6.5, "north": 38.5, "west": 66.5, "east": 100.0}


def test_build_output_filename_preserves_run_identity() -> None:
    assert build_output_filename("20260926", "00", 3) == "gfs_20260926_00_f003.grib2"
    assert build_output_filename("20260926", "18", 120) == "gfs_20260926_18_f120.grib2"


def test_build_gfs_url_contains_subset_variables_and_levels() -> None:
    url = build_gfs_url(
        date="20260926",
        cycle="00",
        forecast_hour=3,
        domain=DOMAIN,
        variables=["APCP", "UGRD", "VGRD", "UGRD"],
        pressure_levels=[850, 700, 850],
        additional_levels=["surface"],
    )

    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == NOMADS_FILTER_ENDPOINT
    assert query["file"] == ["gfs.t00z.pgrb2.0p25.f003"]
    assert query["dir"] == ["/gfs.20260926/00/atmos"]
    assert query["leftlon"] == ["66.5"]
    assert query["rightlon"] == ["100"]
    assert query["toplat"] == ["38.5"]
    assert query["bottomlat"] == ["6.5"]
    assert query["subregion"] == [""]
    assert query["var_APCP"] == ["on"]
    assert query["var_UGRD"] == ["on"]
    assert query["var_VGRD"] == ["on"]
    assert query["lev_850_mb"] == ["on"]
    assert query["lev_700_mb"] == ["on"]
    assert query["lev_surface"] == ["on"]
    assert "all" not in query


def test_build_gfs_url_is_deterministic() -> None:
    url = build_gfs_url(
        date="20260926",
        cycle="00",
        forecast_hour=3,
        domain=DOMAIN,
        variables=["APCP"],
        additional_levels=["surface"],
    )

    assert url == (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        "file=gfs.t00z.pgrb2.0p25.f003&var_APCP=on&lev_surface=on&subregion="
        "&leftlon=66.5&rightlon=100&toplat=38.5&bottomlat=6.5"
        "&dir=%2Fgfs.20260926%2F00%2Fatmos"
    )


@pytest.mark.parametrize(
    ("date", "cycle", "forecast_hour"),
    [
        ("2026-09-26", "00", 3),
        ("20260230", "00", 3),
        ("20260926", "01", 3),
        ("20260926", "00", -1),
        ("20260926", "00", 385),
    ],
)
def test_invalid_filename_components_are_rejected(
    date: str,
    cycle: str,
    forecast_hour: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_output_filename(date, cycle, forecast_hour)


def test_global_url_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="global domain"):
        build_gfs_url(
            date="20260926",
            cycle="00",
            forecast_hour=3,
            domain={"south": -90, "north": 90, "west": 0, "east": 360},
            variables=["APCP"],
            additional_levels=["surface"],
        )


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        assert chunk_size > 0
        yield b"GRIB"
        yield b"offline-test-payload"


class _FakeSession:
    def __init__(self) -> None:
        self.requested_url: str | None = None

    def get(self, url: str, *, stream: bool, timeout: float) -> _FakeResponse:
        assert stream is True
        assert timeout == 5.0
        self.requested_url = url
        return _FakeResponse()


def test_download_writes_file_and_manifest_without_internet(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    manifest_dir = tmp_path / "manifests"
    session = _FakeSession()

    entry = download_gfs_file(
        date="20260926",
        cycle="00",
        forecast_hour=3,
        domain=DOMAIN,
        variables=["APCP"],
        pressure_levels=[],
        additional_levels=["surface"],
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        timeout=5.0,
        session=session,
    )

    downloaded = output_dir / "gfs_20260926_00_f003.grib2"
    assert downloaded.read_bytes() == b"GRIBoffline-test-payload"
    assert entry["local_path"] == str(downloaded.resolve())
    assert entry["url"] == session.requested_url
    manifest = json.loads((manifest_dir / "gfs_download_manifest.json").read_text())
    assert manifest == [entry]
    assert list(output_dir.glob("*.part")) == []
