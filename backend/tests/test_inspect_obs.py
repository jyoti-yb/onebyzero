"""Read-only observation-file inspector."""
import hashlib
import json

import netCDF4
import numpy as np
import pytest

from monsoonpp.grid import imd_canonical_coords
from monsoonpp.tools.inspect_obs import inspect, run


def make_nc(path, *, lat_desc=False, lon360=False, hour=0, global_attrs=None, time_attrs=None,
            rain_attrs=None, rain_name="RAINFALL", ndays=10, fmt="NETCDF4"):
    lat, lon = imd_canonical_coords()
    if lat_desc:
        lat = lat[::-1]
    if lon360:
        lon = lon.copy()                     # all positive already; mark via attr only
    rng = np.random.default_rng(0)
    data = rng.gamma(0.6, 12, (ndays, lat.size, lon.size)).astype("f4")
    data[:, :10, :10] = -999.0
    with netCDF4.Dataset(path, "w", format=fmt) as nc:
        nc.createDimension("TIME", None); nc.createDimension("LATITUDE", lat.size); nc.createDimension("LONGITUDE", lon.size)
        t = nc.createVariable("TIME", "f8", ("TIME",))
        t.units = "days since 2024-01-01 00:00:00"; t.calendar = "gregorian"
        for k, v in (time_attrs or {}).items():
            setattr(t, k, v)
        t[:] = np.arange(ndays) + 196 + hour / 24.0          # from 2024-07-15
        la = nc.createVariable("LATITUDE", "f4", ("LATITUDE",)); la[:] = lat; la.units = "degrees_north"
        lo = nc.createVariable("LONGITUDE", "f4", ("LONGITUDE",)); lo[:] = lon; lo.units = "degrees_east"
        r = nc.createVariable(rain_name, "f4", ("TIME", "LATITUDE", "LONGITUDE"), fill_value=-999.0)
        r.units = "mm"
        for k, v in (rain_attrs or {}).items():
            setattr(r, k, v)
        r[:] = data
        for k, v in (global_attrs or {}).items():
            setattr(nc, k, v)
    return path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_imd_like_file_full_report_no_evidence(tmp_path):
    p = make_nc(tmp_path / "RF25_ind2024_rfp25.nc", global_attrs={"title": "IMD gridded rainfall"})
    rep = inspect(p)
    assert rep["detected_format"] == "hdf5/netcdf4" and rep["source_unchanged"]
    assert rep["identified"] == {"rain_variable": "RAINFALL", "lat": "LATITUDE", "lon": "LONGITUDE", "time": "TIME"}
    assert rep["lat"]["count"] == 129 and rep["lon"]["count"] == 135
    assert rep["lat"]["matches_canonical_imd"] and rep["lon"]["matches_canonical_imd"]
    assert rep["lat"]["spacing"] == pytest.approx(0.25) and rep["lat"]["uniform_spacing"]
    assert rep["lat"]["ordering"] == "ascending"
    assert rep["time"]["decoded_first"].startswith("2024-07-15") and rep["time"]["time_of_day_values"] == ["00:00"]
    assert "date-only labels" in rep["time"]["time_of_day_meaning"]
    assert rep["rain"]["units"] == "mm" and rep["rain"]["declared_fill_values"] == [-999.0]
    assert rep["rain"]["fill_like_value_counts"]["-999.0"] == 10 * 100
    assert rep["rain"]["negative_valid_count"] == 0
    assert rep["time_convention_verdict"]["status"] == "NO_EVIDENCE_IN_FILE"
    assert rep["time_convention_verdict"]["obs_time_convention"] == "UNVERIFIED"


def test_descriptive_text_is_candidate_only_never_sets_convention(tmp_path):
    p = make_nc(tmp_path / "x.nc", global_attrs={"comment": "24 hour rainfall ending at 0830 hrs IST"})
    rep = inspect(p)
    v = rep["time_convention_verdict"]
    assert v["status"] == "CANDIDATE_TEXT_FOUND_REQUIRES_HUMAN_REVIEW"
    assert v["obs_time_convention"] == "UNVERIFIED"
    assert any("0830" in h["text"] for h in rep["convention_evidence_candidates"])


@pytest.mark.parametrize("fname", ["IMD_ENDING_03Z_2024.nc", "rain_STARTING_0830IST.nc"])
def test_filename_is_never_evidence(tmp_path, fname):
    rep = inspect(make_nc(tmp_path / fname))
    assert rep["convention_evidence_candidates"] == []
    assert rep["time_convention_verdict"]["status"] == "NO_EVIDENCE_IN_FILE"
    assert rep["source_filename"] == fname


def test_descending_lat_and_timestamps_with_hours(tmp_path):
    rep = inspect(make_nc(tmp_path / "d.nc", lat_desc=True, hour=3))
    assert rep["lat"]["ordering"] == "descending" and rep["lat"]["matches_canonical_imd"]
    assert rep["time"]["time_of_day_values"] == ["03:00"]
    assert "not evidence of a convention by itself" in rep["time"]["time_of_day_meaning"]
    assert rep["time_convention_verdict"]["obs_time_convention"] == "UNVERIFIED"


def test_netcdf3_and_odd_names(tmp_path):
    rep = inspect(make_nc(tmp_path / "c.nc", fmt="NETCDF3_CLASSIC", rain_name="rf"))
    assert rep["detected_format"] == "netcdf3" and rep["identified"]["rain_variable"] == "rf"


def test_imerg_grid_group_is_inspected(tmp_path):
    p = tmp_path / "imerg.HDF5"
    with netCDF4.Dataset(p, "w", format="NETCDF4") as nc:
        grid = nc.createGroup("Grid")
        grid.createDimension("time", 1)
        grid.createDimension("nv", 2)
        grid.createDimension("lon", 3)
        grid.createDimension("lat", 2)
        t = grid.createVariable("time", "i4", ("time",))
        t.units = "seconds since 1980-01-06 00:00:00 UTC"
        t.calendar = "julian"
        t.bounds = "time_bnds"
        t[:] = [1405047600]
        tb = grid.createVariable("time_bnds", "i4", ("time", "nv"))
        tb.units = t.units
        tb[:] = [[1405047600, 1405049400]]
        grid.createVariable("lon", "f4", ("lon",))[:] = [70.05, 70.15, 70.25]
        grid.createVariable("lat", "f4", ("lat",))[:] = [10.05, 10.15]
        rain = grid.createVariable("precipitation", "f4", ("time", "lon", "lat"), fill_value=-9999.9)
        rain.units = "mm/hr"
        rain[:] = np.ones((1, 3, 2), dtype="f4")

    rep = inspect(p)
    assert rep["root_groups"] == ["Grid"] and rep["inspected_group"] == "/Grid"
    assert rep["identified"] == {"rain_variable": "precipitation", "lat": "lat", "lon": "lon", "time": "time"}
    assert rep["rain"]["units"] == "mm/hr"
    assert rep["time"]["bounds_first"] == [1405047600, 1405049400]


def test_grd_binary_has_no_metadata(tmp_path):
    p = tmp_path / "2024.grd"
    a = np.full((366, 129, 135), -999.0, "<f4"); a[:, 50:60, 50:60] = 5.0
    a.tofile(p)
    rep = inspect(p)
    assert rep["layout"]["n_records"] == 366 and rep["layout"]["n_records_is_365_or_366"]
    assert rep["rain"]["cells_with_any_data"] == 100
    assert "no embedded names" in rep["note"]
    assert rep["time_convention_verdict"]["status"] == "NO_EVIDENCE_IN_FILE"


def test_read_only_and_outputs_elsewhere(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    p = make_nc(src / "RF25_ind2024_rfp25.nc")
    before, listing = sha(p), sorted(x.name for x in src.iterdir())
    rep = run(str(p), str(tmp_path / "out"))
    assert sha(p) == before and sorted(x.name for x in src.iterdir()) == listing
    j = json.loads((tmp_path / "out" / "RF25_ind2024_rfp25.nc.inspection.json").read_text())
    assert j["sha256"] == before
    md = (tmp_path / "out" / "RF25_ind2024_rfp25.nc.inspection.md").read_text()
    assert "remains **UNVERIFIED**" in md


def test_cli(tmp_path, capsys):
    from monsoonpp.cli import main
    p = make_nc(tmp_path / "f.nc")
    main(["inspect-obs", str(p), "--out", str(tmp_path / "o")])
    assert "NO_EVIDENCE_IN_FILE -> obs_time_convention UNVERIFIED" in capsys.readouterr().out
