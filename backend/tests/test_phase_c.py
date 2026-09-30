"""Scientific capability gates and Regime x Error Atlas mechanics."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsoonpp.config import load_config
from monsoonpp.phase_c import _daily_regimes, build_atlas_table
from monsoonpp.regimes.engine import (RegimeCapabilityError, RegimeEngine, RegimeParams,
                                      SYNOPTIC)


def _static(lat, lon):
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    elev = np.maximum(0, (xx - lon.min()) * 200).astype("float32")
    coast = np.maximum(0, (xx - lon.min()) * 111).astype("float32")
    return xr.Dataset(
        {
            "elev": (("lat", "lon"), elev),
            "slope": (("lat", "lon"), np.zeros_like(elev)),
            "aspect": (("lat", "lon"), np.zeros_like(elev)),
            "land": (("lat", "lon"), np.ones_like(elev, dtype=bool)),
            "coast_dist": (("lat", "lon"), coast),
        }, coords={"lat": lat, "lon": lon},
    )


def _fields(lat, lon, times, mslp=None, u=None, v=None):
    shape = (len(times), len(lat), len(lon))
    def arr(value):
        data = np.broadcast_to(value, shape).astype("float32")
        return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": times, "lat": lat, "lon": lon})
    return {
        "mslp": arr(1005.0 if mslp is None else mslp),
        "u850": arr(5.0 if u is None else u),
        "v850": arr(1.0 if v is None else v),
        "cape": arr(1200.0),
        "tcwv": arr(50.0),
    }


def test_active_break_refuses_short_climatology():
    lat, lon = np.arange(18, 20.1, 1.0), np.arange(70, 73.1, 1.0)
    times = pd.date_range("2024-07-01", periods=7)
    rain = xr.DataArray(np.ones((7, len(lat), len(lon))), dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": lat, "lon": lon})
    with pytest.raises(RegimeCapabilityError, match="at least 20 years"):
        RegimeEngine().fit(rain, np.ones((len(lat), len(lon)), bool))


def test_proxy_outputs_are_explicit_and_spatial_residual_is_named():
    lat, lon = np.arange(15, 20.1, 0.5), np.arange(74, 80.1, 0.5)
    times = pd.date_range("2024-07-15", periods=2)
    engine = RegimeEngine()
    out = engine.transform_atmosphere(_fields(lat, lon, times), _static(lat, lon))
    assert "mslp_spatial_residual" in out and "mslp_anom" not in out
    assert out.synoptic.attrs["classification_kind"].startswith("objective NWP proxy")
    assert out.attrs["wd_proxy_available"] == "false"
    assert "wd" not in ",".join(SYNOPTIC)
    assert set(("oro_score", "coast_score", "conv_score")) <= set(out.data_vars)
    assert len(engine.last_detection_audit) == 2


def test_depression_proxy_requires_combined_evidence():
    lat, lon = np.arange(15.0, 20.01, 0.25), np.arange(75.0, 80.01, 0.25)
    times = pd.DatetimeIndex(["2024-07-15"])
    iy, ix = len(lat) // 2, len(lon) // 2
    y_cells, x_cells = np.meshgrid(np.arange(len(lat)) - iy, np.arange(len(lon)) - ix, indexing="ij")
    radius = np.hypot(y_cells, x_cells)
    mslp = 998.0 + np.minimum(radius, 8) * 1.2
    dy = y_cells * 0.25 * 111_000.0
    dx = x_cells * 0.25 * 111_000.0 * np.cos(np.deg2rad(lat[iy]))
    omega = 2e-5
    u, v = -omega * dy, omega * dx
    params = RegimeParams(background_sigma_cells=3, closed_ring_radius_deg=1.0,
                          depression_wind_ms=0.1, min_center_separation_deg=2.0)
    engine = RegimeEngine(params)
    out = engine.transform_atmosphere(_fields(lat, lon, times, mslp, u, v), _static(lat, lon))
    assert 2 in np.unique(out.synoptic)
    assert any(system["class"] == "depression_proxy" for system in engine.last_systems)
    centre = next(system for system in engine.last_systems if system["class"] == "depression_proxy")
    assert centre["closed_ring_depth_hpa"] >= params.depression_closed_depth_hpa
    assert centre["vorticity_1e5_s-1"] >= params.depression_vorticity_1e5


def test_atlas_has_required_metrics_and_low_sample_warnings():
    lat, lon = np.arange(18, 19.01, 0.25), np.arange(75, 76.01, 0.25)
    times = pd.date_range("2024-07-15", periods=2)
    shape = (2, len(lat), len(lon))
    fc = np.full(shape, 20.0, "float32")
    obs = np.full(shape, 10.0, "float32")
    paired = xr.Dataset({"fc": (("time", "lat", "lon"), fc), "obs": (("time", "lat", "lon"), obs)},
                        coords={"time": times, "lat": lat, "lon": lon})
    regimes = RegimeEngine().transform_atmosphere(_fields(lat, lon, times), _static(lat, lon))
    regimes["monsoon_state"] = (("time",), np.full(2, -1, "int8"))
    cfg = load_config(None)
    cfg.thresholds, cfg.fss_scales = [15.6], [1, 3]
    cfg.phase_c = {"min_group_cells": 1000, "min_group_days": 30, "min_events": 20,
                   "local_score_threshold": 0.5}
    table = build_atlas_table(cfg, paired, regimes, np.ones(shape[1:], bool), monsoon_available=False)
    required = {"axis", "regime", "region", "month", "n_cells", "n_days", "bias", "mae", "rmse",
                "pod", "far", "csi", "ets", "fss_s1", "fss_s3", "low_sample", "warning"}
    assert required <= set(table.columns)
    assert set(table.axis) == {"monsoon_state", "synoptic", "local_forcing"}
    assert table.low_sample.all()
    row = table[(table.axis == "synoptic") & (table.regime == "none") & (table.region == "all")].iloc[0]
    assert row.bias == pytest.approx(10.0) and row.n_days == 2


def test_snapshot_regimes_aggregate_to_daily_without_averaging_synoptic_class():
    lat, lon = np.array([18.0, 18.25]), np.array([75.0, 75.25])
    valid = pd.DatetimeIndex(["2024-07-15 06:00", "2024-07-15 12:00",
                              "2024-07-15 18:00", "2024-07-16 00:00"])
    shape = (4, 2, 2)
    ds = xr.Dataset(
        {
            "synoptic": (("time", "lat", "lon"), np.array([0, 0, 2, 0])[:, None, None] * np.ones(shape, "int8")),
            "dist_system_km": (("time", "lat", "lon"), np.arange(4)[:, None, None] * np.ones(shape)),
            "oro_score": (("time", "lat", "lon"), np.arange(4)[:, None, None] * np.ones(shape)),
            "coast_score": (("time", "lat", "lon"), np.zeros(shape)),
            "conv_score": (("time", "lat", "lon"), np.zeros(shape)),
            "mslp_spatial_residual": (("time", "lat", "lon"), np.zeros(shape)),
            "vo850": (("time", "lat", "lon"), np.zeros(shape)),
        },
        coords={"time": valid, "lat": lat, "lon": lon,
                "forecast_label": ("time", [pd.Timestamp("2024-07-15")] * 4)},
    )
    out = _daily_regimes(ds, pd.DatetimeIndex(["2024-07-15"]))
    assert int(out.synoptic.max()) == 2
    assert float(out.oro_score.mean()) == pytest.approx(1.5)
    assert float(out.dist_system_km.mean()) == 0.0
