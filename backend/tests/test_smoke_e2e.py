"""End-to-end smoke-test machinery on real-FORMAT fixtures (GRIB2 GFS, HDF5 IMERG, NetCDF IMD).
These prove the chain; they are not a real-data smoke test."""
import json
import shutil

import netCDF4
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("eccodes"); pytest.importorskip("h5py")

from test_gfs_precip import LAT, LAYOUT, LON, SHAPE, accum, grib_msg, hourly_rates      # noqa: E402
from test_obs_sources import write_imerg                                                 # noqa: E402

from monsoonpp.config import load_config
from monsoonpp.data.obs_time import ObsTimeConventionError
from monsoonpp.smoke import SmokeError, run_smoke

T = pd.Timestamp
LABELS = pd.date_range("2023-07-15", "2023-07-21")


def truth(d):                       # GFS window accumulation for forecast label d (seed per day)
    return accum(hourly_rates(seed=int(d.day)), 3, 27)


def write_gfs(root):
    for d in LABELS:
        r = hourly_rates(seed=int(d.day))
        run = root / f"gfs.{d:%Y%m%d}" / "00" / "atmos"
        run.mkdir(parents=True)
        for fxx, ivs in LAYOUT.items():
            (run / f"gfs.t00z.pgrb2.0p25.f{fxx:03d}").write_bytes(
                b"".join(grib_msg(accum(r, a, b), a, b, init=T(d)) for a, b in ivs))


def base_cfg(tmp_path, obs_source, files, conv="UNVERIFIED", **smoke):
    cfg = load_config(None, name="smoke_t", forecast_source="gfs", obs_source=obs_source, obs_time_convention=conv,
                      data_dir=str(tmp_path / "run"), raw_dir=str(tmp_path / "raw"),
                      adapter_options={"obs": {"files": [str(f) for f in files], "origin": "fixture"}},
                      smoke={"start": "2023-07-15", "end": "2023-07-21", "lead": 1,
                             "gfs_local_dir": str(tmp_path / "gfs"), **smoke})
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max = 10.0, 12.0, 70.0, 72.5
    return cfg


def write_imd_like(path, labels, values_by_label, missing=None):
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("TIME", None); nc.createDimension("LATITUDE", LAT.size); nc.createDimension("LONGITUDE", LON.size)
        t = nc.createVariable("TIME", "f8", ("TIME",)); t.units = "days since 1900-01-01"; t.calendar = "gregorian"
        t[:] = (pd.DatetimeIndex(labels) - T("1900-01-01")).days.values
        nc.createVariable("LATITUDE", "f4", ("LATITUDE",))[:] = LAT
        nc.createVariable("LONGITUDE", "f4", ("LONGITUDE",))[:] = LON
        r = nc.createVariable("RAINFALL", "f4", ("TIME", "LATITUDE", "LONGITUDE"), fill_value=-999.0); r.units = "mm"
        arr = np.stack([values_by_label[l] for l in labels]).astype("f4")
        if missing:
            arr[missing] = -999.0
        r[:] = arr


@pytest.fixture(scope="module")
def gfs_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("g")
    write_gfs(root / "gfs")
    return root


def mirror_gfs(source, destination):
    """Use links where permitted and a copy fallback on Windows CI."""
    for path in source.iterdir():
        target = destination / path.name
        try:
            target.symlink_to(path, target_is_directory=path.is_dir())
        except OSError:
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)


def test_imd_ending_pairs_by_window_and_is_correct(gfs_dir, tmp_path):
    """IMD-like file under ENDING_03Z: label D+1 holds window D. Correct pairing => ~zero error."""
    mirror_gfs(gfs_dir, tmp_path)
    obs_labels = LABELS + pd.Timedelta(days=1)
    write_imd_like(tmp_path / "imd.nc", obs_labels, {l: truth(l - pd.Timedelta(days=1)) for l in obs_labels},
                   missing=(2, 4, 5))                                   # one missing cell on one day
    s = run_smoke(base_cfg(tmp_path, "imd_netcdf", [tmp_path / "imd.nc"], conv="ENDING_03Z"))
    assert s["windows_paired"] == 7 and s["unpaired"] == []
    daily = s["metrics"]["daily"]
    assert [r["obs_label"] for r in daily] == list(obs_labels.strftime("%Y-%m-%d"))
    assert all(abs(r["bias"]) < 0.02 and r["rmse"] < 0.02 for r in daily)
    assert daily[2]["obs_missing_in_mask"] == 1 and daily[2]["cells_used"] == SHAPE[0] * SHAPE[1] - 1
    assert s["metrics"]["pooled"]["obs_product_role"] == "FINAL_TRUTH"
    assert s["observation_provenance"]["time_convention"] == "ENDING_03Z"
    for k in ("bias", "mae", "rmse", "corr", "pod_64.5", "far_64.5", "csi_64.5", "ets_64.5", "fss_64.5_s3"):
        assert k in s["metrics"]["pooled"] and k in daily[0]


def test_wrong_convention_does_not_silently_pair_equal_labels(gfs_dir, tmp_path):
    """Same file declared STARTING_03Z: only labels 16..21 exist, so forecast label 15 is unpaired
    and the others pair with the NEXT day's rain -> large errors. Labels are never matched by equality."""
    mirror_gfs(gfs_dir, tmp_path)
    obs_labels = LABELS + pd.Timedelta(days=1)
    write_imd_like(tmp_path / "imd.nc", obs_labels, {l: truth(l - pd.Timedelta(days=1)) for l in obs_labels})
    s = run_smoke(base_cfg(tmp_path, "imd_netcdf", [tmp_path / "imd.nc"], conv="STARTING_03Z"))
    assert s["windows_paired"] == 6 and s["unpaired"][0]["forecast_label"] == "2023-07-15"
    assert np.mean([r["rmse"] for r in s["metrics"]["daily"]]) > 1.0


def test_unverified_imd_blocked(gfs_dir, tmp_path):
    mirror_gfs(gfs_dir, tmp_path)
    write_imd_like(tmp_path / "imd.nc", LABELS, {l: truth(l) for l in LABELS})
    with pytest.raises(ObsTimeConventionError, match="BLOCKED"):
        run_smoke(base_cfg(tmp_path, "imd_netcdf", [tmp_path / "imd.nc"]))


IMERG_LAT = np.round(np.arange(9.05, 12.96, 0.1), 2)
IMERG_LON = np.round(np.arange(69.05, 73.46, 0.1), 2)


def test_imerg_proxy_full_chain(gfs_dir, tmp_path):
    mirror_gfs(gfs_dir, tmp_path)
    files = []
    for i, d in enumerate(LABELS):
        for k in range(48):
            if i == 3 and k == 20:
                continue                                                # one gap -> day 4 rejected
            t0 = T(d) + pd.Timedelta(hours=3) + pd.Timedelta(minutes=30 * k)
            files.append(write_imerg(tmp_path / f"g_{i}_{k:02d}.HDF5", t0, lat=IMERG_LAT, lon=IMERG_LON, value=0.25 * (i + 1)))
    cfg = base_cfg(tmp_path, "imerg", files, mask="obs_valid")
    s = run_smoke(cfg)
    assert s["windows_paired"] == 6
    assert s["unpaired"][0]["forecast_label"] == "2023-07-18" and "gap" in s["unpaired"][0]["reason"]
    assert s["imerg_rejected_windows"][0]["granules_found"] == 47
    ob = s["metrics"]["daily"][0]
    assert ob["obs_product_role"] == "SMOKE_TEST_PROXY" and ob["obs_is_proxy"] is True and ob["obs_label"] == "explicit interval"
    assert "NOT IMD" in s["banner"][0] and "not evidence of forecast skill" in s["banner"][1]
    assert s["observation_provenance"]["source_name"] == "NASA-GPM-IMERG"
    # obs uniform daily depth = 48 * 0.5 h * rate (conservative regrid keeps a constant constant)
    import xarray as xr
    p = xr.open_dataset(cfg.report_dir / "smoke" / "smoke_paired.nc")
    assert np.allclose(p.obs.isel(time=0).values, 24 * 0.25, atol=1e-4)
    assert p.attrs["obs_meta_product_role"] == "SMOKE_TEST_PROXY"
    md = (cfg.report_dir / "smoke" / "smoke_report.md").read_text()
    assert "SMOKE_TEST_PROXY" in md and "NOT IMD" in md and "Unpaired windows" in md
    assert len(s["maps"]) == 7 and all(m.endswith(".png") for m in s["maps"])
    j = json.loads((cfg.report_dir / "smoke" / "smoke_metrics.json").read_text())
    assert j["metrics"]["pooled"]["obs_product_role"] == "SMOKE_TEST_PROXY"


def test_smoke_rejects_non_real_sources(gfs_dir, tmp_path):
    cfg = base_cfg(tmp_path, "synthetic", [])
    with pytest.raises(SmokeError, match="real observation product"):
        run_smoke(cfg)
