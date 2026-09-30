"""Observation sources: roles, anti-masquerade, provenance round-trips, decode-vs-interpret separation."""
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

h5py = pytest.importorskip("h5py")

from monsoonpp.adapters import get_adapter
from monsoonpp.adapters.imerg import IMERGAdapter, decode_imerg_granule
from monsoonpp.adapters.obs_netcdf import ObsDecodeError
from monsoonpp.config import load_config
from monsoonpp.data.obs_source import (ATTR_PREFIX, ObsSourceError, ObsSourceMeta, ProductRole,
                                       read, require_daily_truth_capable)
from monsoonpp.data.obs_time import ObsTimeConventionError, interpret_obs_times

T = pd.Timestamp
LAT25 = np.round(np.arange(10.0, 12.001, 0.25), 4)
LON25 = np.round(np.arange(70.0, 72.501, 0.25), 4)
DATES = pd.date_range("2024-07-15", "2024-07-21")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ------------------------------------------------------------------ fixtures
def write_daily_nc(path, *, lat=LAT25, lon=LON25, dates=DATES, units="mm", fill=-999.0, desc_lat=True,
                   step_days=1, var="RAINFALL", gattrs=None):
    rng = np.random.default_rng(1)
    data = rng.gamma(0.7, 10, (len(dates), lat.size, lon.size)).astype("f4")
    data[:, 0, 0] = fill if fill is not None else data[:, 0, 0]
    la = lat[::-1] if desc_lat else lat
    d = data[:, ::-1, :] if desc_lat else data
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("TIME", None); nc.createDimension("LATITUDE", lat.size); nc.createDimension("LONGITUDE", lon.size)
        t = nc.createVariable("TIME", "f8", ("TIME",)); t.units = "days since 1900-01-01 00:00:00"; t.calendar = "gregorian"
        t[:] = (pd.DatetimeIndex(dates) - T("1900-01-01")).days.values * step_days
        nc.createVariable("LATITUDE", "f4", ("LATITUDE",))[:] = la
        nc.createVariable("LONGITUDE", "f4", ("LONGITUDE",))[:] = lon
        kw = {"fill_value": fill} if fill is not None else {}
        r = nc.createVariable(var, "f4", ("TIME", "LATITUDE", "LONGITUDE"), **kw)
        if units is not None:
            r.units = units
        r[:] = d
        for k, v in (gattrs or {}).items():
            setattr(nc, k, v)
    return data


IMERG_LAT = np.round(np.arange(5.05, 15.96, 0.1), 2)
IMERG_LON = np.round(np.arange(65.05, 80.96, 0.1), 2)


def write_imerg(path, start, *, minutes=30, units="mm/hr", with_bnds=True, time_offset_min=0,
                scales=True, lat=IMERG_LAT, lon=IMERG_LON, value=1.5):
    t0 = (T(start) - T("1970-01-01")).total_seconds()
    with h5py.File(path, "w") as f:
        f.attrs["FileHeader"] = b"AlgorithmID=3IMERGHH;\nProductVersion=V07B;\n"
        g = f.create_group("Grid")
        g.create_dataset("lat", data=lat); g.create_dataset("lon", data=lon)
        t = g.create_dataset("time", data=np.array([t0 + time_offset_min * 60], "i4"))
        t.attrs["units"] = b"seconds since 1970-01-01 00:00:00 UTC"; t.attrs["calendar"] = b"julian"
        if with_bnds:
            tb = g.create_dataset("time_bnds", data=np.array([[t0, t0 + minutes * 60]], "i4"))
            tb.attrs["units"] = b"seconds since 1970-01-01 00:00:00 UTC"
        arr = np.full((1, lon.size, lat.size), value, "f4")          # IMERG layout (time, lon, lat)
        arr[0, 0, 0] = -9999.9
        p = g.create_dataset("precipitation", data=arr)
        p.attrs["units"] = units.encode(); p.attrs["_FillValue"] = np.float32(-9999.9)
        p.attrs["CodeMissingValue"] = b"-9999.9"
        if scales:
            for k in ("time", "lon", "lat"):
                g[k].make_scale(k)
            p.dims[0].attach_scale(g["time"]); p.dims[1].attach_scale(g["lon"]); p.dims[2].attach_scale(g["lat"])
    return path


def cfg_for(tmp_path, source, files, **kw):
    cfg = load_config(None, obs_source=source, data_dir=str(tmp_path / "run"), raw_dir=str(tmp_path / "raw"),
                      adapter_options={"obs": {"files": [str(f) for f in files], "origin": "unit-test fixture"}}, **kw)
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max = 10.0, 12.0, 70.0, 72.5
    return cfg


def imerg_run(tmp_path, n=4, start="2024-07-15 03:00"):
    files = [write_imerg(tmp_path / f"granule_{i}.HDF5", T(start) + pd.Timedelta(minutes=30 * i)) for i in range(n)]
    cfg = cfg_for(tmp_path, "synthetic", files)
    return IMERGAdapter(cfg).load()


# -------------------------------------------------------- distinguishable
def test_each_source_is_distinguishable(tmp_path):
    write_daily_nc(tmp_path / "g.nc"); write_daily_nc(tmp_path / "m.nc")
    imd = get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "g.nc"])).load(DATES)
    mrg = get_adapter("obs", "imd_ncmrwf_merged", cfg_for(tmp_path, "imd_ncmrwf_merged", [tmp_path / "m.nc"])).load(DATES)
    imr = imerg_run(tmp_path)
    metas = {k: read(d) for k, d in (("imd", imd), ("merged", mrg), ("imerg", imr))}
    assert (metas["imd"].source_name, metas["imd"].product_role, metas["imd"].is_proxy) == ("IMD", ProductRole.FINAL_TRUTH, False)
    assert (metas["merged"].source_name, metas["merged"].product_role, metas["merged"].is_proxy) == \
        ("IMD-NCMRWF", ProductRole.OFFICIAL_ALTERNATIVE, False)
    assert (metas["imerg"].source_name, metas["imerg"].product_role, metas["imerg"].is_proxy) == \
        ("NASA-GPM-IMERG", ProductRole.SMOKE_TEST_PROXY, True)
    assert len({m.product_name for m in metas.values()}) == 3
    from monsoonpp.adapters.imd import imd_grd_meta
    assert imd_grd_meta(["2024.grd"]).product_role is ProductRole.FINAL_TRUTH


def test_all_provenance_fields_present(tmp_path):
    write_daily_nc(tmp_path / "g.nc")
    obs = get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "g.nc"])).load(DATES)
    m = read(obs)
    assert m.source_filename == "g.nc" and m.source_url_or_origin == "unit-test fixture"
    assert m.observation_resolution.startswith("0.25 deg, daily")
    assert "0.25 deg" in m.native_grid and "(9 x 11)" in m.native_grid
    assert m.units.startswith("mm (file attribute)") and m.time_convention == "UNVERIFIED"


# ----------------------------------------------------- IMERG cannot masquerade
@pytest.mark.parametrize("kw", [
    dict(source_name="IMD", product_role="SMOKE_TEST_PROXY", is_proxy=True),
    dict(source_name="NASA-GPM-IMERG", product_role="FINAL_TRUTH", is_proxy=True),
    dict(source_name="NASA-GPM-IMERG", product_role="OFFICIAL_ALTERNATIVE", is_proxy=True),
    dict(source_name="NASA-GPM-IMERG", product_role="SMOKE_TEST_PROXY", is_proxy=False),
    dict(source_name="NASA-GPM-IMERG", product_role="SMOKE_TEST_PROXY", is_proxy=True, product_name="IMD rainfall (IMERG)"),
    dict(source_name="IMD", product_role="FINAL_TRUTH", is_proxy=False, product_name="IMERG final run"),
    dict(source_name="GPM", product_role="SMOKE_TEST_PROXY", is_proxy=True),
    dict(source_name="IMD", product_role="TRUTH", is_proxy=False),
])
def test_invalid_role_combinations_rejected(kw):
    base = dict(source_name="NASA-GPM-IMERG", product_name="IMERG half-hourly", product_role="SMOKE_TEST_PROXY",
                source_filename="f", source_url_or_origin="o", observation_resolution="0.1 deg", native_grid="g",
                units="mm/hr", time_convention="x", is_proxy=True)
    with pytest.raises(ObsSourceError):
        ObsSourceMeta(**{**base, **kw})


def test_imerg_relabelled_as_imd_is_rejected_on_read(tmp_path):
    ds = imerg_run(tmp_path)
    for tamper in ({ATTR_PREFIX + "source_name": "IMD"},
                   {ATTR_PREFIX + "product_role": "FINAL_TRUTH"},
                   {ATTR_PREFIX + "is_proxy": "false"},
                   {ATTR_PREFIX + "product_name": "IMD 0.25 deg gauge"}):
        bad = ds.copy(); bad.attrs.update(tamper)
        with pytest.raises(ObsSourceError):
            read(bad)


def test_imerg_is_unreachable_from_daily_pairing_path(tmp_path):
    with pytest.raises(KeyError):
        get_adapter("obs", "imerg", load_config())                 # not a daily obs adapter
    ds = imerg_run(tmp_path)
    with pytest.raises(ObsSourceError, match="smoke-test proxy"):
        require_daily_truth_capable(ds, "build forecast/observation pairs")
    assert "Not IMD" in ds.attrs["WARNING"] and "IMD" not in read(ds).product_name
    cfg = load_config(None, obs_source="imerg", obs_time_convention="STARTING_03Z", data_dir=str(tmp_path),
                      years=[2021], leads=[1], season_start="06-01", season_end="06-03")
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max, cfg.grid.res = 6.5, 37.5, 66.5, 97.5, 1.0
    from monsoonpp.data.build import build_dataset
    with pytest.raises(KeyError, match="No obs adapter 'imerg'"):
        build_dataset(cfg, save=False)


# --------------------------------------------------- provenance serialization
def test_daily_provenance_survives_netcdf(tmp_path):
    write_daily_nc(tmp_path / "m.nc")
    obs = get_adapter("obs", "imd_ncmrwf_merged", cfg_for(tmp_path, "imd_ncmrwf_merged", [tmp_path / "m.nc"])).load(DATES)
    obs.to_netcdf(tmp_path / "o.nc")
    back = xr.open_dataset(tmp_path / "o.nc")
    assert read(back) == read(obs)
    assert list(back.source_filename.values) == ["m.nc"] * 7
    assert list(back.source_date_label.values) == list(DATES.strftime("%Y-%m-%d"))
    assert np.array_equal(back.source_time_raw.values, obs.source_time_raw.values)
    assert back.attrs["source_time_units"] == "days since 1900-01-01 00:00:00"


def test_imerg_provenance_survives_netcdf(tmp_path):
    ds = imerg_run(tmp_path)
    ds.to_netcdf(tmp_path / "i.nc")
    back = xr.open_dataset(tmp_path / "i.nc")
    assert read(back) == read(ds) and read(back).is_proxy is True
    assert (pd.DatetimeIndex(back.source_interval_start.values) == pd.DatetimeIndex(ds.source_interval_start.values)).all()
    assert back.attrs["WARNING"].startswith("SMOKE-TEST PROXY")


def test_provenance_survives_build_dataset(tmp_path):
    from monsoonpp.data.build import build_dataset, load_dataset
    cfg = load_config(None, name="prov", years=[2021], leads=[1], season_start="06-01", season_end="06-05",
                      data_dir=str(tmp_path), obs_time_convention="STARTING_03Z")
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max, cfg.grid.res = 6.5, 37.5, 66.5, 97.5, 1.0
    build_dataset(cfg)
    ds = load_dataset(cfg)
    m = ObsSourceMeta.from_attrs(ds.obs_rain.attrs)
    assert m.product_role is ProductRole.SYNTHETIC_TEST and m.time_convention == "STARTING_03Z"
    assert ds.attrs["obs_product_role"] == "SYNTHETIC_TEST" and ds.attrs["obs_is_proxy"] == "true"


# ------------------------------------------- decoding independent of timing
def test_decoding_independent_of_configured_convention(tmp_path):
    write_daily_nc(tmp_path / "g.nc")
    outs = []
    for conv in ("UNVERIFIED", "ENDING_03Z", "STARTING_03Z"):
        cfg = cfg_for(tmp_path, "imd_netcdf", [tmp_path / "g.nc"], obs_time_convention=conv)
        outs.append(get_adapter("obs", "imd_netcdf", cfg).load(DATES))
    for o in outs:
        assert o.attrs["time_convention"] == "UNVERIFIED" and pd.isna(o.valid_start.values).all()
        xr.testing.assert_identical(o, outs[0])
    e = interpret_obs_times(outs[0], "ENDING_03Z")
    assert T(e.valid_start.values[0]) == T("2024-07-14 03:00")
    np.testing.assert_array_equal(e.rain.values, outs[0].rain.values)       # only timing coords change
    assert list(e.source_date_label.values) == list(outs[0].source_date_label.values)


def test_imerg_timing_only_from_time_bnds(tmp_path):
    # filename claims 12:00Z; metadata says 03:00-03:30Z -> metadata wins, filename ignored
    p = write_imerg(tmp_path / "3B-HHR.MS.MRG.3IMERG.20240715-S120000-E122959.V07B.HDF5", "2024-07-15 03:00")
    ds, _ = decode_imerg_granule(p, bbox=None)
    assert T(ds.source_interval_start.values[0]) == T("2024-07-15 03:00")
    assert T(ds.source_interval_end.values[0]) == T("2024-07-15 03:30")
    assert "valid_start" not in ds.coords and "source_date_label" not in ds.coords
    assert ds.precipitation_rate.attrs["units"] == "mm/hr" and np.isnan(ds.precipitation_rate.values[0, 0, 0])
    assert ds.sizes == {"time": 1, "lat": IMERG_LAT.size, "lon": IMERG_LON.size}     # native, (time, lat, lon)


@pytest.mark.parametrize("kw,match", [
    (dict(with_bnds=False), "time_bnds missing"),
    (dict(minutes=1440), "!= 30 min"),
    (dict(time_offset_min=45), "outside its bounds"),
    (dict(units="mm/day"), "mm/hr rate"),
])
def test_imerg_ambiguous_timing_or_units_rejected(tmp_path, kw, match):
    p = write_imerg(tmp_path / "g.HDF5", "2024-07-15 03:00", **kw)
    with pytest.raises(ObsDecodeError, match=match):
        decode_imerg_granule(p)


def test_imerg_axis_order_without_scales_uses_unambiguous_lengths(tmp_path):
    p = write_imerg(tmp_path / "g.HDF5", "2024-07-15 03:00", scales=False)
    ds, _ = decode_imerg_granule(p)
    assert ds.sizes["lat"] == IMERG_LAT.size and ds.sizes["lon"] == IMERG_LON.size
    sq = np.round(np.arange(5.05, 6.0, 0.1), 2)
    p2 = write_imerg(tmp_path / "sq.HDF5", "2024-07-15 03:00", scales=False, lat=sq, lon=sq + 60)
    with pytest.raises(ObsDecodeError, match="axis order"):
        decode_imerg_granule(p2)


def test_imerg_native_grid_preserved_and_bbox_only_selects(tmp_path):
    files = [write_imerg(tmp_path / f"g{i}.HDF5", T("2024-07-15 03:00") + pd.Timedelta(minutes=30 * i)) for i in range(3)]
    ds = IMERGAdapter(cfg_for(tmp_path, "synthetic", files)).load()
    assert np.allclose(np.diff(ds.lat.values), 0.1) and np.allclose(np.diff(ds.lon.values), 0.1)
    assert set(np.round(ds.lat.values, 2)) <= set(IMERG_LAT) and set(np.round(ds.lon.values, 2)) <= set(IMERG_LON)
    assert ds.lat.values.min() >= 9.0 and ds.lat.values.max() <= 13.0          # runtime grid +1 deg box
    assert int(ds.attrs["n_granules"]) == 3 and int(ds.attrs["n_gaps"]) == 0


def test_imerg_selection_by_metadata_and_duplicates(tmp_path):
    files = [write_imerg(tmp_path / f"g{i}.HDF5", T("2024-07-15 03:00") + pd.Timedelta(minutes=30 * i)) for i in range(4)]
    ad = IMERGAdapter(cfg_for(tmp_path, "synthetic", files))
    ds = ad.load(T("2024-07-15 03:30"), T("2024-07-15 04:30"))
    assert list(pd.DatetimeIndex(ds.source_interval_start.values).strftime("%H:%M")) == ["03:30", "04:00"]
    dup = write_imerg(tmp_path / "dup.HDF5", "2024-07-15 03:00")
    with pytest.raises(ObsDecodeError, match="duplicate"):
        IMERGAdapter(cfg_for(tmp_path, "synthetic", files + [dup])).load()


# ------------------------------------------------- strict daily decoding
def test_daily_decoder_strictness(tmp_path):
    write_daily_nc(tmp_path / "nounits.nc", units=None)
    with pytest.raises(ObsDecodeError, match="no units attribute"):
        get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "nounits.nc"])).load(DATES)
    cfg = cfg_for(tmp_path, "imd_netcdf", [tmp_path / "nounits.nc"])
    cfg.adapter_options["obs"].update(units_declared="mm", units_evidence="IMD product page, retrieved 2024-..")
    obs = get_adapter("obs", "imd_netcdf", cfg).load(DATES)
    assert "declared by operator" in read(obs).units

    write_daily_nc(tmp_path / "nofill.nc", fill=None)
    ds = netCDF4.Dataset(tmp_path / "nofill.nc", "a"); ds["RAINFALL"][0, 0, 0] = -5.0; ds.close()
    with pytest.raises(ObsDecodeError, match="negative values"):
        get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "nofill.nc"])).load(DATES)

    write_daily_nc(tmp_path / "coarse.nc", lat=np.arange(10.0, 12.01, 0.5), lon=np.arange(70.0, 72.51, 0.5))
    with pytest.raises(ObsDecodeError, match="must be 0.25"):
        get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "coarse.nc"])).load(DATES)

    write_daily_nc(tmp_path / "offgrid.nc", lat=LAT25 + 0.125, lon=LON25 + 0.125)
    with pytest.raises(ObsDecodeError, match="never interpolated"):
        get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "offgrid.nc"])).load(DATES)

    write_daily_nc(tmp_path / "twoday.nc", step_days=2)
    with pytest.raises(ObsDecodeError, match="not exactly 1 day"):
        get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "twoday.nc"])).load(DATES)

    write_daily_nc(tmp_path / "wrongunits.nc", units="mm/hr")
    with pytest.raises(ObsDecodeError, match="daily millimetre depth"):
        get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "wrongunits.nc"])).load(DATES)


def test_daily_values_and_orientation(tmp_path):
    data = write_daily_nc(tmp_path / "g.nc", desc_lat=True)
    obs = get_adapter("obs", "imd_netcdf", cfg_for(tmp_path, "imd_netcdf", [tmp_path / "g.nc"])).load(DATES)
    assert np.all(np.diff(obs.lat.values) > 0)
    np.testing.assert_allclose(obs.rain.values[:, 1:, 1:], data[:, 1:, 1:])
    assert np.isnan(obs.rain.values[:, 0, 0]).all()


# -------------------------------------------------- IMD guard not weakened
@pytest.mark.parametrize("source", ["imd", "imd_netcdf", "imd_ncmrwf_merged"])
def test_unverified_official_sources_blocked(tmp_path, monkeypatch, source):
    import monsoonpp.data.build as build
    from monsoonpp.pipeline import run_train
    from monsoonpp.verify.report import write_report
    monkeypatch.setattr(build, "get_adapter", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no I/O")))
    cfg = cfg_for(tmp_path, source, [])
    assert cfg.obs_time_convention == "UNVERIFIED"
    with pytest.raises(ObsTimeConventionError, match="UNVERIFIED"):
        build.build_dataset(cfg, save=False)
    with pytest.raises(ObsTimeConventionError):
        run_train(cfg)
    with pytest.raises(ObsTimeConventionError):
        write_report(cfg)


@pytest.mark.parametrize("name", ["gfs_imd_era5.yaml", "ncum.yaml"])
def test_real_configs_still_unverified(name):
    assert load_config(CONFIGS / name).obs_time_convention == "UNVERIFIED"
