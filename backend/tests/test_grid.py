"""Canonical IMD 0.25 deg grid: geometry, config defaults, crops, and mask-based handling."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsoonpp.adapters.imd import land_mask_from_obs
from monsoonpp.config import GridConfig, load_config
from monsoonpp.data.align import to_grid
from monsoonpp.grid import (IMD_CANONICAL, IMD_NLAT, IMD_NLON, build_static, check_imd_compatible,
                            grid_role, imd_canonical_coords, make_coords)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_canonical_counts():
    lat, lon = imd_canonical_coords()
    assert (lat.size, lon.size) == (129, 135) == (IMD_NLAT, IMD_NLON)


def test_canonical_ranges():
    lat, lon = imd_canonical_coords()
    assert lat[0] == 6.5 and lat[-1] == 38.5
    assert lon[0] == 66.5 and lon[-1] == 100.0


def test_canonical_spacing_uniform_quarter_degree():
    lat, lon = imd_canonical_coords()
    assert np.allclose(np.diff(lat), 0.25, atol=1e-9)
    assert np.allclose(np.diff(lon), 0.25, atol=1e-9)
    # every centre exactly on the 0.25 lattice (no float drift)
    assert np.allclose((lat - 6.5) / 0.25, np.round((lat - 6.5) / 0.25), atol=1e-9)
    assert np.allclose((lon - 66.5) / 0.25, np.round((lon - 66.5) / 0.25), atol=1e-9)


def test_default_gridconfig_is_canonical():
    assert GridConfig() == IMD_CANONICAL
    assert grid_role(GridConfig()) == "imd_canonical"
    lat, lon = make_coords(load_config().grid)
    assert (lat.size, lon.size) == (129, 135)


@pytest.mark.parametrize("name", ["gfs_imd_era5.yaml", "ncum.yaml"])
def test_real_data_configs_use_canonical_grid(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.obs_source == "imd"
    assert grid_role(cfg.grid) == "imd_canonical"
    lat, lon = make_coords(cfg.grid)
    assert (lat.size, lon.size) == (129, 135)


def test_synthetic_config_is_not_labelled_imd():
    cfg = load_config(CONFIGS / "synthetic.yaml")
    assert grid_role(cfg.grid) == "non_imd"


def test_no_config_calls_a_cropped_domain_native():
    for p in CONFIGS.glob("*.yaml"):
        cfg = load_config(p)
        text = p.read_text().lower()
        if "native" in text and "imd" in text:
            assert grid_role(cfg.grid) == "imd_canonical", f"{p.name} calls a non-canonical grid native IMD"


def test_aligned_crop_accepted_misaligned_rejected():
    crop = GridConfig(lat_min=8.0, lat_max=30.0, lon_min=68.0, lon_max=97.5, res=0.25)
    assert grid_role(crop) == "imd_crop" and check_imd_compatible(crop) == "imd_crop"
    lat, lon = make_coords(crop)
    clat, clon = imd_canonical_coords()
    assert np.isin(lat, clat).all() and np.isin(lon, clon).all()
    for bad in (GridConfig(6.6, 38.5, 66.5, 100.0, 0.25),     # off-lattice
                GridConfig(6.5, 38.5, 66.5, 100.0, 0.5),      # coarsened
                GridConfig(6.5, 39.0, 66.5, 100.0, 0.25)):    # outside canonical rectangle
        assert grid_role(bad) == "non_imd"
        with pytest.raises(ValueError):
            check_imd_compatible(bad)


def _fake_imd(nt=3):
    """IMD-like file: canonical grid, lat descending on disk, -999/NaN outside India."""
    lat, lon = imd_canonical_coords()
    rng = np.random.default_rng(0)
    data = rng.gamma(0.8, 10, (nt, lat.size, lon.size)).astype("float32")
    valid = build_static(IMD_CANONICAL)["land"].values
    data[:, ~valid] = np.nan
    da = xr.DataArray(data[:, ::-1, :], dims=("time", "lat", "lon"),
                      coords={"time": pd.date_range("2023-07-01", periods=nt), "lat": lat[::-1], "lon": lon})
    return da, data, valid


def test_imd_on_canonical_grid_is_selected_not_interpolated():
    da, data, _ = _fake_imd()
    lat, lon = imd_canonical_coords()
    out = to_grid(da, lat, lon)
    assert out.shape == (3, 129, 135)
    np.testing.assert_array_equal(out.values, data)        # bit-identical, NaNs preserved


def test_missing_cells_become_mask_not_smaller_grid():
    da, _, valid = _fake_imd()
    lat, lon = imd_canonical_coords()
    obs = xr.Dataset({"rain": to_grid(da, lat, lon)})
    land = land_mask_from_obs(obs)
    assert land.shape == (129, 135)                         # full rectangle kept
    assert 0 < land.sum() < 129 * 135                       # ocean cells masked, not dropped
    np.testing.assert_array_equal(land, valid)
    static = build_static(IMD_CANONICAL, land=land)
    assert static["land"].shape == (129, 135) and static["coast_dist"].shape == (129, 135)
    assert static["slope"].shape == (129, 135) and static["aspect"].shape == (129, 135)
