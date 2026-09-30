"""Observation time-convention model: UNVERIFIED blocks, explicit conventions map labels exactly."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsoonpp.config import load_config
from monsoonpp.data.obs_time import (PROVENANCE_ATTRS, PROVENANCE_COORDS, ObsTimeConvention,
                                     ObsTimeConventionError, decode_provenance, interpret_obs_times,
                                     label_window, parse_convention, require_verified)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
T = pd.Timestamp


def small_cfg(tmp_path, **kw):
    cfg = load_config(None, name="obs_time", years=[2021], leads=[1], season_start="06-01",
                      season_end="06-06", data_dir=str(tmp_path), raw_dir=str(tmp_path / "raw"), **kw)
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max, cfg.grid.res = 6.5, 37.5, 66.5, 97.5, 1.0
    return cfg


def imd_like(labels, lat=np.arange(10.0, 12.01, 0.25), lon=np.arange(70.0, 72.51, 0.25)):
    rng = np.random.default_rng(0)
    return xr.DataArray(rng.gamma(0.8, 10, (len(labels), lat.size, lon.size)).astype("float32"),
                        dims=("time", "lat", "lon"),
                        coords={"time": pd.DatetimeIndex(labels), "lat": lat, "lon": lon})


# ---------------------------------------------------------------- defaults
def test_default_convention_is_unverified():
    assert parse_convention(load_config().obs_time_convention) is ObsTimeConvention.UNVERIFIED


@pytest.mark.parametrize("name", ["gfs_imd_era5.yaml", "ncum.yaml"])
def test_real_imd_configs_are_unverified(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.obs_source == "imd"
    assert parse_convention(cfg.obs_time_convention) is ObsTimeConvention.UNVERIFIED
    assert "imd_window" not in cfg.adapter_options


def test_decoded_imd_is_unverified_with_no_valid_times():
    from monsoonpp.adapters.imd import decode_imd
    lat, lon = np.arange(10.0, 12.01, 0.25), np.arange(70.0, 72.51, 0.25)
    obs = decode_imd(imd_like(["2023-07-15", "2023-07-16"]), lat, lon, "rain/2023.grd", "imd_crop")
    assert obs.attrs["time_convention"] == "UNVERIFIED"
    assert pd.isna(obs.valid_start.values).all() and pd.isna(obs.valid_end.values).all()


# ------------------------------------------------------- blocked workflows
def test_real_pairing_blocked_before_any_io(tmp_path, monkeypatch):
    import monsoonpp.data.build as build
    def boom(*a, **k):
        raise AssertionError("adapters must not be called when the convention is unverified")
    monkeypatch.setattr(build, "get_adapter", boom)
    cfg = load_config(CONFIGS / "gfs_imd_era5.yaml", data_dir=str(tmp_path))
    with pytest.raises(ObsTimeConventionError, match="BLOCKED: 'build forecast/observation pairs'.*UNVERIFIED"):
        build.build_dataset(cfg, save=False)


def test_training_and_verification_blocked(tmp_path):
    from monsoonpp.pipeline import run_train
    from monsoonpp.verify.report import verification_tables, write_report
    cfg = load_config(CONFIGS / "gfs_imd_era5.yaml", data_dir=str(tmp_path))
    with pytest.raises(ObsTimeConventionError, match="'train model'"):
        run_train(cfg)
    with pytest.raises(ObsTimeConventionError, match="'run verification'"):
        write_report(cfg)
    with pytest.raises(ObsTimeConventionError, match="'run verification'"):
        verification_tables(cfg, pd.DataFrame())


def test_training_samples_blocked_for_unverified_or_untracked_dataset():
    from monsoonpp.features import build_table
    ds = xr.Dataset({"obs_rain": (("time", "lat", "lon"), np.zeros((1, 2, 2)))},
                    coords={"time": [T("2023-07-15")], "lat": [10.0, 10.25], "lon": [70.0, 70.25]})
    with pytest.raises(ObsTimeConventionError, match="no obs_time_convention record"):
        build_table(ds, bundle=None)
    ds.attrs["obs_time_convention"] = "UNVERIFIED"
    with pytest.raises(ObsTimeConventionError, match="'create training samples'"):
        build_table(ds, bundle=None)


def test_load_dataset_blocks_stale_or_mismatched(tmp_path):
    from monsoonpp.data.build import load_dataset
    cfg = small_cfg(tmp_path, obs_time_convention="STARTING_03Z")
    cfg.run_dir.mkdir(parents=True)
    ds = xr.Dataset({"obs_rain": (("time", "lat", "lon"), np.zeros((1, 2, 2), "float32")),
                     "land": (("lat", "lon"), np.ones((2, 2), "int8"))},
                    coords={"time": [T("2023-07-15")], "lat": [10.0, 10.25], "lon": [70.0, 70.25]})
    ds.to_netcdf(cfg.dataset_path)                                   # legacy: no convention attr
    with pytest.raises(ObsTimeConventionError, match="Rebuild"):
        load_dataset(cfg)
    ds.attrs["obs_time_convention"] = "ENDING_03Z"
    ds.to_netcdf(cfg.dataset_path, mode="w")
    with pytest.raises(ObsTimeConventionError, match="built with ENDING_03Z, config says STARTING_03Z"):
        load_dataset(cfg)


def test_legacy_imd_window_key_rejected(tmp_path):
    from monsoonpp.data.build import build_dataset
    cfg = small_cfg(tmp_path, obs_time_convention="STARTING_03Z", adapter_options={"imd_window": "starting"})
    with pytest.raises(ObsTimeConventionError, match="imd_window is no longer supported"):
        build_dataset(cfg, save=False)


def test_unverified_label_window_raises():
    with pytest.raises(ObsTimeConventionError, match="UNVERIFIED"):
        label_window("2023-07-15", "UNVERIFIED")
    with pytest.raises(ObsTimeConventionError):
        require_verified("UNVERIFIED", "anything")


# ------------------------------------------------------ explicit conventions
def test_ending_03z_timestamps():
    assert label_window("2023-07-15", "ENDING_03Z") == (T("2023-07-14 03:00"), T("2023-07-15 03:00"))


def test_starting_03z_timestamps():
    assert label_window("2023-07-15", "STARTING_03Z") == (T("2023-07-15 03:00"), T("2023-07-16 03:00"))


@pytest.mark.parametrize("label", ["2021-06-01", "2023-07-15", "2024-02-28", "2024-02-29", "2024-12-31"])
def test_conventions_differ_by_exactly_one_day_in_labelling(label):
    es, ee = label_window(label, "ENDING_03Z")
    ss, se = label_window(label, "STARTING_03Z")
    assert ss - es == pd.Timedelta(days=1) and se - ee == pd.Timedelta(days=1)
    assert ee - es == se - ss == pd.Timedelta(hours=24)
    # the same physical window carries label D under STARTING and D+1 under ENDING
    nxt = (T(label) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert label_window(nxt, "ENDING_03Z") == (ss, se)


def test_interpretation_keeps_source_metadata_separate():
    obs = decode_provenance(xr.Dataset({"rain": imd_like(["2023-07-15", "2023-07-16"])}), source="IMD",
                            product_name="p", source_filenames=["a.grd", "b.grd"], units="mm", grid_role="imd_crop")
    out = interpret_obs_times(obs, "ENDING_03Z")
    assert list(out.source_date_label.values) == ["2023-07-15", "2023-07-16"]      # unchanged source label
    assert list(pd.DatetimeIndex(out.time.values).strftime("%Y-%m-%d")) == ["2023-07-15", "2023-07-16"]
    assert T(out.valid_start.values[0]) == T("2023-07-14 03:00")
    assert T(out.valid_end.values[1]) == T("2023-07-16 03:00")
    assert list(out.source_filename.values) == ["a.grd", "b.grd"]
    assert out.attrs["time_convention"] == "ENDING_03Z" and obs.attrs["time_convention"] == "UNVERIFIED"


def test_provenance_fields_present():
    obs = decode_provenance(xr.Dataset({"rain": imd_like(["2023-07-15"])}), source="IMD", product_name="p",
                            source_filenames="rain/2023.grd", units="mm", grid_role="imd_crop")
    obs = interpret_obs_times(obs, "STARTING_03Z")
    for k in PROVENANCE_ATTRS:
        assert k in obs.attrs
    for k in PROVENANCE_COORDS:
        assert k in obs.coords and obs[k].dims == ("time",)
    assert set(PROVENANCE_ATTRS + PROVENANCE_COORDS) == {
        "source", "product_name", "source_filename", "source_date_label", "time_convention",
        "valid_start", "valid_end", "units", "grid_role"}


# --------------------------------------------------------- no silent inference
@pytest.mark.parametrize("fname", ["IMD_rain_ENDING_03Z_2023.grd", "rain_STARTING_03Z.grd",
                                   "imd_ending_0830IST_2023.nc", "starting/2023.grd"])
def test_filenames_never_establish_convention(fname):
    from monsoonpp.adapters.imd import decode_imd
    lat, lon = np.arange(10.0, 12.01, 0.25), np.arange(70.0, 72.51, 0.25)
    obs = decode_imd(imd_like(["2023-07-15"]), lat, lon, fname, "imd_crop")
    assert obs.attrs["time_convention"] == "UNVERIFIED"
    assert pd.isna(obs.valid_start.values).all()
    assert obs.source_filename.values[0] == fname                 # recorded, not interpreted


def test_imd_adapter_load_does_not_interpret(tmp_path, monkeypatch):
    from monsoonpp.adapters.imd import IMDAdapter
    cfg = load_config(None, obs_source="imd", raw_dir=str(tmp_path / "ENDING_03Z_mirror"))
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max = 10.0, 12.0, 70.0, 72.5
    ad = IMDAdapter(cfg)
    (ad.dir / "rain").mkdir(parents=True)
    (ad.dir / "rain" / "2023.grd").write_bytes(b"")
    monkeypatch.setattr(ad, "_open_source", lambda y0, y1: imd_like(["2023-07-15", "2023-07-16"]))
    obs = ad.load(pd.DatetimeIndex(["2023-07-15", "2023-07-16"]))
    assert obs.attrs["time_convention"] == "UNVERIFIED" and obs.attrs["source"] == "IMD"
    assert obs.attrs["grid_role"] == "imd_crop" and obs.attrs["units"] == "mm"
    assert pd.isna(obs.valid_start.values).all()
    assert Path(str(obs.source_filename.values[0])).as_posix().endswith("rain/2023.grd")


def test_convention_strings_are_strict():
    for bad in ["starting", "ending", "ending_03z", "Starting_03Z", "", None, "0830IST"]:
        with pytest.raises(ObsTimeConventionError):
            parse_convention(bad)


# -------------------------------------------------------------- synthetic path
def test_synthetic_usable_with_explicit_convention(tmp_path):
    from monsoonpp.data.build import build_dataset
    ds = build_dataset(small_cfg(tmp_path, obs_time_convention="STARTING_03Z"), save=True)
    assert ds.attrs["obs_time_convention"] == "STARTING_03Z"
    assert ds.obs_rain.attrs["time_convention"] == "STARTING_03Z" and ds.obs_rain.attrs["source"] == "synthetic"
    lab = pd.DatetimeIndex(ds.time.values)
    assert (pd.DatetimeIndex(ds.obs_valid_start.values) == lab + pd.Timedelta(hours=3)).all()
    assert (pd.DatetimeIndex(ds.obs_valid_end.values) == lab + pd.Timedelta(hours=27)).all()
    assert list(ds.obs_source_date_label.values) == list(lab.strftime("%Y-%m-%d"))
    from monsoonpp.data.build import load_dataset
    load_dataset(small_cfg(tmp_path, obs_time_convention="STARTING_03Z"))      # round-trips through netCDF


def test_synthetic_without_convention_is_blocked(tmp_path):
    from monsoonpp.data.build import build_dataset
    with pytest.raises(ObsTimeConventionError, match="UNVERIFIED"):
        build_dataset(small_cfg(tmp_path), save=False)


def test_ending_convention_pairs_by_window_not_label(tmp_path):
    """Under ENDING_03Z, forecast label D (03Z D -> 03Z D+1) must pair with observation label D+1.
    (Replaces the pre-pairing test that refused ENDING_03Z outright.)"""
    from monsoonpp.data.build import build_dataset
    ds = build_dataset(small_cfg(tmp_path, obs_time_convention="ENDING_03Z"), save=False)
    lab = pd.DatetimeIndex(ds.time.values)
    obs_lab = pd.DatetimeIndex(pd.to_datetime(ds.obs_source_date_label.values))
    assert ((obs_lab - lab) == pd.Timedelta(days=1)).all()
    assert (pd.DatetimeIndex(ds.obs_valid_start.values) == lab + pd.Timedelta(hours=3)).all()
    assert (pd.DatetimeIndex(ds.obs_valid_end.values) == lab + pd.Timedelta(hours=27)).all()
