import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsoonpp import schema
from monsoonpp.data.align import forecast_window, gfs_apcp_pieces, to_grid
from monsoonpp.regimes.engine import _run_filter
from monsoonpp.verify.metrics import contingency, fss, probabilistic


def test_contingency_known_values():
    f = np.array([70, 70, 0, 0, 70, 0])
    o = np.array([70, 0, 70, 0, 70, 0])
    c = contingency(f, o, 64.5)
    assert (c["hits"], c["false_alarms"], c["misses"], c["correct_neg"]) == (2, 1, 1, 2)
    assert c["pod"] == pytest.approx(2 / 3) and c["far"] == pytest.approx(1 / 3) and c["csi"] == pytest.approx(0.5)
    ar = 3 * 3 / 6
    assert c["ets"] == pytest.approx((2 - ar) / (4 - ar))


def test_fss_perfect_and_displaced():
    mask = np.ones((20, 20), bool)
    a = np.zeros((1, 20, 20)); a[0, 5, 5] = 100
    b = np.zeros((1, 20, 20)); b[0, 5, 7] = 100
    assert fss(a, a, mask, 64.5, 1) == pytest.approx(1.0)
    assert fss(a, b, mask, 64.5, 1) == pytest.approx(0.0)
    assert fss(a, b, mask, 64.5, 5) > 0.5          # neighbourhood forgives a 2-cell shift


def test_brier_perfect():
    y = np.array([0, 1, 0, 1], bool)
    assert probabilistic(y.astype(float), y)["brier"] == 0


def test_forecast_window_and_gfs_buckets():
    w = forecast_window(pd.Timestamp("2023-07-15"), 1, "starting")
    assert w.init == pd.Timestamp("2023-07-15") and (w.f_start, w.f_end) == (3, 27)
    w3 = forecast_window(pd.Timestamp("2023-07-15"), 3, "ending")
    assert w3.init == pd.Timestamp("2023-07-12") and (w3.f_start, w3.f_end) == (51, 75)
    pieces = gfs_apcp_pieces(w)
    # buckets must tile [3, 27] exactly: +[0,6] -[0,3] +[6,12] +[12,18] +[18,24] +[24,27]
    cover = np.zeros(30)
    for _, rng, sign in pieces:
        a, b = map(int, rng.split("-"))
        cover[a:b] += sign
    assert np.all(cover[3:27] == 1) and np.all(cover[:3] == 0) and np.all(cover[27:] == 0)


def test_to_grid_handles_0_360_and_descending_lat():
    lat_src = np.arange(40, 0, -0.25); lon_src = np.arange(60, 100, 0.25)
    da = xr.DataArray(np.add.outer(lat_src, lon_src * 0), dims=("latitude", "longitude"),
                      coords={"latitude": lat_src, "longitude": lon_src})
    lat, lon = np.arange(6.5, 10, 0.25), np.arange(70, 72, 0.25)
    out = to_grid(da, lat, lon)
    assert np.allclose(out.values[:, 0], lat)


def test_run_filter():
    s = np.array([1, 1, 0, 1, 1, 1, -1, -1, -1, -1, 0])
    assert _run_filter(s, 3).tolist() == [0, 0, 0, 1, 1, 1, -1, -1, -1, -1, 0]


def test_schema_rejects_bad_dims():
    ds = xr.Dataset({"rain": (("lat", "time", "lon"), np.zeros((2, 3, 2)))},
                    coords={"lat": [1, 2], "lon": [1, 2], "time": pd.date_range("2021-06-01", periods=3)})
    with pytest.raises(schema.SchemaError):
        schema.validate_obs(ds)


def test_netcdf_ncum_adapter_roundtrip(tmp_path):
    """The operational NCUM path: per-init NetCDF, accumulated precip since init."""
    from monsoonpp.adapters import get_adapter
    from monsoonpp.config import load_config
    cfg = load_config(None, forecast_source="netcdf",
                      adapter_options={"forecast_dir": str(tmp_path), "filename_pattern": "{init:%Y%m%d}.nc"})
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max, cfg.grid.res = 10, 12, 75, 77, 1.0
    lat, lon = np.arange(10, 12.01, 1.0), np.arange(75, 77.01, 1.0)
    init = pd.Timestamp("2023-07-15")
    times = init + pd.to_timedelta(np.arange(0, 49, 3), unit="h")
    hours = np.arange(0, 49, 3, dtype=float)
    shape = (len(times), len(lat), len(lon))
    ds = xr.Dataset({
        "tp_accum": (("time", "lat", "lon"), np.broadcast_to(hours[:, None, None] * 1.0, shape).copy()),  # 1 mm/h
        **{v: (("time", "lat", "lon"), np.full(shape, k)) for k, v in enumerate(["u850", "v850", "tcwv", "cape", "mslp"])},
    }, coords={"time": times, "lat": lat, "lon": lon})
    ds.to_netcdf(tmp_path / "20230715.nc")
    out = get_adapter("forecast", "netcdf", cfg).load(pd.DatetimeIndex([init]), [1])
    assert np.allclose(out.tp.values, 24.0)          # f027 - f003 = 24 h x 1 mm/h
    assert np.allclose(out.tcwv.values, 2.0)
    schema.validate_forecast(out)
