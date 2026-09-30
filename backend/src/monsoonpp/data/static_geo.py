"""Real static geography from NOAA ETOPO 2022 for non-synthetic runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

from ..config import Config
from ..grid import coast_distance_km, make_coords, terrain_slope_aspect
from .regrid import overlap_matrix, regular_edges


ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
)
ETOPO_DOI = "10.25921/fd45-gt74"


class StaticGeographyError(RuntimeError):
    pass


def _conservative_static(da: xr.DataArray, dst_lat: np.ndarray, dst_lon: np.ndarray) -> xr.DataArray:
    """Conservative 2-D remap using sparse separable weights for high-resolution terrain."""
    src_lat, src_lon = np.asarray(da.lat), np.asarray(da.lon)
    src_lat_edges, src_lon_edges = regular_edges(src_lat), regular_edges(src_lon)
    dst_lat_edges, dst_lon_edges = regular_edges(dst_lat), regular_edges(dst_lon)
    wy = csr_matrix(overlap_matrix(np.sin(np.deg2rad(src_lat_edges)),
                                   np.sin(np.deg2rad(dst_lat_edges))))
    wx = csr_matrix(overlap_matrix(src_lon_edges, dst_lon_edges))
    area = np.outer(np.diff(np.sin(np.deg2rad(dst_lat_edges))), np.diff(dst_lon_edges))
    values = np.asarray(da.values, dtype="float64")
    valid = np.isfinite(values)

    def weighted(array):
        return (wx @ (wy @ array).T).T

    numerator = weighted(np.where(valid, values, 0.0))
    coverage = weighted(valid.astype("float64"))
    fraction = coverage / area
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(fraction >= 1.0 - 1e-9, numerator / coverage, np.nan)
    return xr.DataArray(
        result.astype("float32"), dims=("lat", "lon"), coords={"lat": dst_lat, "lon": dst_lon},
        attrs={**da.attrs, "regrid_method": "first-order conservative area mean (sparse separable weights)",
               "regrid_min_coverage": 1.0},
    )


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def require_real_static(ds: xr.Dataset) -> xr.Dataset:
    if _truthy(ds.attrs.get("is_synthetic", True)):
        raise StaticGeographyError("real-data run is blocked because static geography is synthetic or unproven")
    required = ("static_source", "static_source_url", "static_source_doi", "static_source_resolution")
    missing = [key for key in required if not ds.attrs.get(key)]
    if missing:
        raise StaticGeographyError(f"real static geography provenance missing: {missing}")
    return ds


def fetch_etopo_subset(path: str | Path, cfg: Config, source_url: str = ETOPO_URL,
                       margin_degrees: float = 1.0) -> Path:
    """Cache a bounded ETOPO subset; never download the 491 MB global file."""
    import netCDF4

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = cfg.grid
    with netCDF4.Dataset(source_url) as source:
        lat = np.asarray(source.variables["lat"][:], dtype="float64")
        lon = np.asarray(source.variables["lon"][:], dtype="float64")
        iy = np.nonzero((lat >= g.lat_min - margin_degrees) & (lat <= g.lat_max + margin_degrees))[0]
        ix = np.nonzero((lon >= g.lon_min - margin_degrees) & (lon <= g.lon_max + margin_degrees))[0]
        if not iy.size or not ix.size:
            raise StaticGeographyError("requested grid does not intersect the ETOPO source")
        zvar = source.variables["z"]
        units = str(getattr(zvar, "units", ""))
        if units != "meters":
            raise StaticGeographyError(f"ETOPO elevation units {units!r} != 'meters'")
        z = np.asarray(zvar[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], dtype="float32")
        fill = float(getattr(zvar, "_FillValue", -99999.0))
        z[np.isclose(z, fill)] = np.nan
    subset = xr.Dataset(
        {"z": (("lat", "lon"), z)},
        coords={"lat": lat[iy], "lon": lon[ix]},
        attrs={
            "source": "NOAA NCEI ETOPO 2022 60 arc-second bedrock elevation",
            "source_url": source_url,
            "source_doi": ETOPO_DOI,
            "source_resolution": "60 arc-second",
            "subset_bounds": f"{lat[iy[0]]}..{lat[iy[-1]]} N, {lon[ix[0]]}..{lon[ix[-1]]} E",
        },
    )
    subset.z.attrs.update(units="m", long_name="bedrock elevation relative to EGM2008")
    subset.to_netcdf(path, encoding={"z": {"zlib": True, "complevel": 3}})
    return path


def load_real_static(cfg: Config) -> xr.Dataset:
    opts = dict(cfg.adapter_options.get("static", {}))
    path = Path(opts.get("path", Path(cfg.raw_dir) / "static" / "etopo_2022_60s_india.nc"))
    source_url = str(opts.get("source_url", ETOPO_URL))
    if not path.exists():
        if not bool(opts.get("allow_download", True)):
            raise StaticGeographyError(f"real static geography file missing: {path}")
        fetch_etopo_subset(path, cfg, source_url=source_url,
                           margin_degrees=float(opts.get("margin_degrees", 1.0)))

    source = xr.open_dataset(path).load()
    if "z" not in source or tuple(source.z.dims) != ("lat", "lon"):
        raise StaticGeographyError(f"{path}: expected z(lat, lon)")
    if str(source.z.attrs.get("units", "")) not in ("m", "meters"):
        raise StaticGeographyError(f"{path}: elevation units are not metres")
    if not (np.all(np.diff(source.lat.values) > 0) and np.all(np.diff(source.lon.values) > 0)):
        raise StaticGeographyError(f"{path}: coordinates must be strictly ascending")
    if not np.isfinite(source.z.values).all():
        raise StaticGeographyError(f"{path}: elevation contains missing/non-finite values")
    source_range = (float(source.z.min()), float(source.z.max()))
    if source_range[0] < -12_000 or source_range[1] > 9_000:
        raise StaticGeographyError(f"{path}: elevation range {source_range} is physically implausible")
    lat_step = float(np.median(np.diff(source.lat.values)))
    lon_step = float(np.median(np.diff(source.lon.values)))
    if not (0 < lat_step <= 1 / 30 and 0 < lon_step <= 1 / 30):
        raise StaticGeographyError(f"{path}: source resolution {lat_step}, {lon_step} is too coarse")

    lat, lon = make_coords(cfg.grid)
    terrain = xr.where(source.z > 0, source.z, 0.0)
    land_fraction = _conservative_static((source.z > 0).astype("float32"), lat, lon)
    elevation = _conservative_static(terrain, lat, lon)
    threshold = float(opts.get("land_fraction_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise StaticGeographyError(f"land_fraction_threshold must be in [0, 1], got {threshold}")
    land = land_fraction.values >= threshold
    elev = np.where(land, elevation.values, 0.0).astype("float32")
    slope, aspect = terrain_slope_aspect(elev, lat, cfg.grid.res)
    slope = np.where(land, slope, 0.0).astype("float32")
    aspect = np.where(land, aspect, 0.0).astype("float32")
    out = xr.Dataset(
        {
            "elev": (("lat", "lon"), elev),
            "slope": (("lat", "lon"), slope),
            "aspect": (("lat", "lon"), aspect),
            "land": (("lat", "lon"), land),
            "coast_dist": (("lat", "lon"), coast_distance_km(land, cfg.grid.res)),
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "static_source": source.attrs.get("source", "NOAA NCEI ETOPO 2022"),
            "static_source_url": source.attrs.get("source_url", source_url),
            "static_source_doi": source.attrs.get("source_doi", ETOPO_DOI),
            "static_source_resolution": source.attrs.get("source_resolution", "60 arc-second"),
            "static_source_file": str(path),
            "static_regrid_method": "first-order conservative area-weighted mean",
            "land_definition": f"ETOPO elevation > 0 over at least {threshold:.2f} of target-cell area",
            "is_synthetic": "false",
        },
    )
    out.elev.attrs.update(units="m", long_name="mean land elevation; zero over ocean")
    out.slope.attrs.update(units="degree", long_name="terrain slope from remapped elevation")
    out.aspect.attrs.update(units="degree", long_name="downslope aspect clockwise from north")
    out.land.attrs.update(units="1", long_name="ETOPO-derived land mask")
    out.coast_dist.attrs.update(units="km", long_name="distance from land cell to nearest sea cell")
    checks = {
        "elev": (0.0, 9_000.0),
        "slope": (0.0, 90.0),
        "aspect": (0.0, 360.0),
        "coast_dist": (0.0, 20_000.0),
    }
    for name, (lo, hi) in checks.items():
        observed = (float(out[name].min()), float(out[name].max()))
        if not np.isfinite(out[name].values).all() or observed[0] < lo or observed[1] > hi:
            raise StaticGeographyError(f"{name} range {observed} outside [{lo}, {hi}]")
    return require_real_static(out)
