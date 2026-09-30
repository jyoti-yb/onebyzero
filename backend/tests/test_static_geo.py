"""Real static geography provenance, remapping, and synthetic-data guard."""
import numpy as np
import pytest
import xarray as xr

from monsoonpp.config import load_config
from monsoonpp.data.static_geo import StaticGeographyError, load_real_static, require_real_static
from monsoonpp.grid import build_static
from monsoonpp.schema import validate_static


def _source(path):
    lat = np.round(np.arange(9.75, 10.751, 0.025), 4)
    lon = np.round(np.arange(69.75, 70.751, 0.025), 4)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    z = (600 + 120 * (yy - 10) + 80 * (xx - 70)).astype("float32")
    z[:, :12] = -50.0
    ds = xr.Dataset(
        {"z": (("lat", "lon"), z, {"units": "m"})},
        coords={"lat": lat, "lon": lon},
        attrs={
            "source": "NOAA NCEI ETOPO 2022 test subset",
            "source_url": "https://example.invalid/etopo-test",
            "source_doi": "10.25921/fd45-gt74",
            "source_resolution": "90 arc-second test fixture",
        },
    )
    ds.to_netcdf(path)


def test_real_static_fields_and_provenance(tmp_path):
    path = tmp_path / "etopo.nc"
    _source(path)
    cfg = load_config(None, raw_dir=str(tmp_path))
    cfg.grid.lat_min, cfg.grid.lat_max = 10.0, 10.5
    cfg.grid.lon_min, cfg.grid.lon_max = 70.0, 70.5
    cfg.grid.res = 0.25
    cfg.adapter_options["static"] = {"path": str(path), "allow_download": False}
    ds = validate_static(load_real_static(cfg))
    assert set(ds.data_vars) == {"elev", "slope", "aspect", "land", "coast_dist"}
    assert ds.attrs["is_synthetic"] == "false"
    assert ds.attrs["static_source_doi"] == "10.25921/fd45-gt74"
    assert np.isfinite(ds.to_array()).all()
    assert float(ds.slope.min()) >= 0 and float(ds.aspect.max()) <= 360


def test_synthetic_or_unproven_static_is_blocked():
    with pytest.raises(StaticGeographyError, match="synthetic or unproven"):
        require_real_static(build_static(load_config().grid))
    bare = build_static(load_config().grid)
    bare.attrs["is_synthetic"] = "false"
    with pytest.raises(StaticGeographyError, match="provenance missing"):
        require_real_static(bare)
