"""Strict GFS atmospheric metadata checks and the five allowed derivatives."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ec = pytest.importorskip("eccodes")

import monsoonpp.adapters.gfs_atmos as ga
from monsoonpp.adapters.gfs import GFSAdapter
from monsoonpp.adapters.gfs_atmos import (FIELD_SPECS, REGIME_FIELDS, GFSAtmosMetadataError,
                                          decode_gfs_atmos_file, derive_atmospheric_fields)
from monsoonpp.config import load_config


INIT = pd.Timestamp("2024-07-15 00:00")
FXX = 6
LAT = np.arange(10.0, 11.01, 0.25)
LON = np.arange(70.0, 71.26, 0.25)


def _value(name: str) -> float:
    if name == "mslp":
        return 100_000.0
    if name == "tcwv":
        return 48.0
    if name == "cape":
        return 1_200.0
    if name.startswith("u"):
        return 8.0
    if name.startswith("v"):
        return 4.0
    if name.startswith("rh"):
        return 65.0
    return {850: 1_500.0, 700: 3_100.0, 500: 5_600.0, 200: 12_000.0}[int(name[3:])]


def _message(name: str, fxx: int = FXX) -> bytes:
    spec = FIELD_SPECS[name]
    gid = ec.codes_grib_new_from_samples("GRIB2")
    ec.codes_set(gid, "centre", 7)
    for key, value in {
        "Ni": LON.size,
        "Nj": LAT.size,
        "latitudeOfFirstGridPointInDegrees": float(LAT[-1]),
        "latitudeOfLastGridPointInDegrees": float(LAT[0]),
        "longitudeOfFirstGridPointInDegrees": float(LON[0]),
        "longitudeOfLastGridPointInDegrees": float(LON[-1]),
        "iDirectionIncrementInDegrees": 0.25,
        "jDirectionIncrementInDegrees": 0.25,
    }.items():
        ec.codes_set(gid, key, value)
    ec.codes_set(gid, "productDefinitionTemplateNumber", 0)
    ec.codes_set(gid, "discipline", spec.param[0])
    ec.codes_set(gid, "parameterCategory", spec.param[1])
    ec.codes_set(gid, "parameterNumber", spec.param[2])
    if spec.type_of_level == "atmosphereSingleLayer":
        ec.codes_set(gid, "typeOfFirstFixedSurface", 200)
    else:
        ec.codes_set(gid, "typeOfLevel", spec.type_of_level)
        if spec.level:
            ec.codes_set(gid, "level", spec.level)
    ec.codes_set(gid, "dataDate", int(INIT.strftime("%Y%m%d")))
    ec.codes_set(gid, "dataTime", 0)
    ec.codes_set(gid, "stepUnits", 1)
    ec.codes_set(gid, "forecastTime", fxx)
    ec.codes_set(gid, "bitsPerValue", 16)
    yy, xx = np.meshgrid(LAT, LON, indexing="ij")
    values = _value(name) + 0.01 * yy + 0.02 * xx
    ec.codes_set_values(gid, np.ascontiguousarray(values[::-1]).ravel())
    message = ec.codes_get_message(gid)
    ec.codes_release(gid)
    return message


def _write(path: Path, names=tuple(FIELD_SPECS), duplicate=None) -> Path:
    messages = [_message(name) for name in names]
    if duplicate:
        messages.append(_message(duplicate))
    path.write_bytes(b"".join(messages))
    return path


@pytest.fixture(autouse=True)
def _small_grid(monkeypatch):
    monkeypatch.setattr(ga, "GFS_NI", LON.size)
    monkeypatch.setattr(ga, "GFS_NJ", LAT.size)
    monkeypatch.setattr(ga, "GFS_GRID_BOUNDS", (float(LAT[-1]), float(LAT[0]), float(LON[0]), float(LON[-1])))


def test_decode_all_required_fields_and_metadata(tmp_path):
    ds, qa = decode_gfs_atmos_file(_write(tmp_path / "atmos.grib2"), INIT, FXX)
    assert set(ds.data_vars) == set(FIELD_SPECS)
    assert qa["status"] == "PASS" and qa["missing_values"] == 0
    assert qa["valid_time"] == "2024-07-15 06:00:00"
    assert ds.mslp.attrs["source_units"] == "Pa" and ds.mslp.attrs["units"] == "hPa"
    assert ds.u850.attrs["source_level"] == "isobaricInhPa:850"
    assert float(ds.mslp.mean()) == pytest.approx(1000.0, abs=0.1)


def test_missing_and_duplicate_messages_fail(tmp_path):
    names = tuple(name for name in FIELD_SPECS if name != "hgt200")
    with pytest.raises(GFSAtmosMetadataError, match="missing.*hgt200"):
        decode_gfs_atmos_file(_write(tmp_path / "missing.grib2", names), INIT, FXX)
    with pytest.raises(GFSAtmosMetadataError, match="duplicate u850"):
        decode_gfs_atmos_file(_write(tmp_path / "duplicate.grib2", duplicate="u850"), INIT, FXX)


def test_regime_subset_uses_same_strict_checks_without_changing_full_default(tmp_path):
    path = _write(tmp_path / "regime.grib2", REGIME_FIELDS)
    ds, qa = decode_gfs_atmos_file(path, INIT, FXX, required_fields=REGIME_FIELDS)
    assert set(ds.data_vars) == set(REGIME_FIELDS)
    assert qa["required_fields"] == list(REGIME_FIELDS)
    with pytest.raises(GFSAtmosMetadataError, match="missing required atmospheric messages"):
        decode_gfs_atmos_file(path, INIT, FXX)


def test_regime_fetch_reuses_exact_cached_subset_without_network(tmp_path):
    cfg = load_config(None, raw_dir=str(tmp_path))
    adapter = GFSAdapter(cfg)
    day = tmp_path / "gfs" / "gfs" / "20240715"
    day.mkdir(parents=True)
    cached = day / "subset_abcd0921__gfs.t00z.pgrb2.0p25.f006"
    cached.write_bytes(b"cached")
    assert adapter._fetch_regime_atmos_file(INIT, FXX) == cached


def test_forecast_time_mismatch_fails(tmp_path):
    path = tmp_path / "wrong-time.grib2"
    path.write_bytes(_message("mslp", fxx=12) + b"".join(_message(name) for name in list(FIELD_SPECS)[1:]))
    with pytest.raises(GFSAtmosMetadataError, match="forecast hour"):
        decode_gfs_atmos_file(path, INIT, FXX)


def test_exactly_five_derived_fields(tmp_path):
    ds, _ = decode_gfs_atmos_file(_write(tmp_path / "atmos.grib2"), INIT, FXX)
    out = derive_atmospheric_fields(ds)
    assert set(out.data_vars) == {"wspd850", "vo850", "moisture_transport", "pwat_anom", "mslp_anom"}
    assert np.isfinite(out.to_array()).all()
    assert out.wspd850.attrs["units"] == "m s-1"
    assert out.vo850.attrs["units"] == "1e-5 s-1"
