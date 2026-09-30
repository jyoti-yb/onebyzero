"""Spatial alignment onto the canonical 0.25 deg grid.

* same lattice (IMD products, GFS 0.25): exact cell SELECTION, never interpolation
  (obs_netcdf.select_exact / strict_select here).
* finer source (IMERG 0.1 deg): FIRST-ORDER CONSERVATIVE REMAPPING (area-weighted mean;
  Jones 1999, MWR 127). On a regular lat-lon grid a cell's area is
      A = R^2 * dlambda * d(sin phi),
  so overlap weights factorise into a longitude overlap (degrees) and a sin(latitude)
  overlap. For an intensive field (mm depth) the destination value is the overlap-area
  weighted mean of source cells, which conserves the area integral (water volume).
  0.1 deg cells do not nest in 0.25 deg cells (2.5 per side); partial overlaps are exact.

Missing data: source NaNs contribute no area. A destination cell is kept only if the
valid source area covers at least `min_coverage` of its footprint (default 1.0 = any
missing or out-of-extent area makes the destination missing — missing stays missing).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..data.align import standardise_latlon


class RegridError(RuntimeError):
    pass


def regular_edges(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype="float64")
    d = np.diff(c)
    spacing = float(np.median(d)) if d.size else float("nan")
    # IMERG coordinates are float32; a 0.1-degree lattice quantizes adjacent
    # differences by up to about 8e-6 degrees after conversion to float64.
    if c.size < 2 or not np.allclose(d, spacing, rtol=0, atol=1e-5) or spacing <= 0:
        raise RegridError("coordinates must be strictly ascending with uniform spacing")
    return np.concatenate([c - spacing / 2, [c[-1] + spacing / 2]])


def overlap_matrix(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """(n_dst, n_src) lengths of overlap between 1-D cells (in whatever coordinate is given)."""
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.clip(hi - lo, 0.0, None)


def conservative_regrid(da: xr.DataArray, dst_lat: np.ndarray, dst_lon: np.ndarray,
                        min_coverage: float = 1.0) -> xr.DataArray:
    da = standardise_latlon(da)
    s_lat, s_lon = da.lat.values, da.lon.values
    se_lat, se_lon = regular_edges(s_lat), regular_edges(s_lon)
    de_lat, de_lon = regular_edges(np.asarray(dst_lat)), regular_edges(np.asarray(dst_lon))
    if not (-90 - 1e-9 <= se_lat[0] and se_lat[-1] <= 90 + 1e-9):
        raise RegridError("source latitude edges outside [-90, 90]")
    wy = overlap_matrix(np.sin(np.deg2rad(se_lat)), np.sin(np.deg2rad(de_lat)))       # (ny_d, ny_s)
    wx = overlap_matrix(se_lon, de_lon)                                               # (nx_d, nx_s)
    area_d = np.outer(np.diff(np.sin(np.deg2rad(de_lat))), np.diff(de_lon))          # (ny_d, nx_d)

    other = [d for d in da.dims if d not in ("lat", "lon")]
    arr = da.transpose(*other, "lat", "lon").values.astype("float64")
    valid = np.isfinite(arr)
    num = np.einsum("ij,...jk,lk->...il", wy, np.where(valid, arr, 0.0), wx)
    cov = np.einsum("ij,...jk,lk->...il", wy, valid.astype("float64"), wx)
    frac = cov / area_d
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(frac >= min_coverage - 1e-9, num / cov, np.nan)
    coords = {d: da[d] for d in other}
    coords.update(lat=np.asarray(dst_lat), lon=np.asarray(dst_lon))
    res = xr.DataArray(out.astype("float32"), dims=(*other, "lat", "lon"), coords=coords, name=da.name,
                       attrs={**da.attrs, "regrid_method": "first-order conservative (area-weighted, Jones 1999)",
                              "regrid_min_coverage": float(min_coverage)})
    for c, v in da.coords.items():                    # keep per-time provenance coords
        if c not in res.coords and set(v.dims) <= set(other):
            res = res.assign_coords({c: v})
    return res


def strict_select(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray, tol: float = 1e-4) -> xr.DataArray:
    """Same-lattice alignment: pick cells exactly; raise if any target is not a source cell."""
    da = standardise_latlon(da)
    def idx(src, tgt, name):
        j = np.abs(src[:, None] - tgt[None, :]).argmin(0)
        if (np.abs(src[j] - tgt) > tol).any():
            raise RegridError(f"target {name} points are not on the source lattice; refusing to interpolate")
        return j
    return da.isel(lat=idx(da.lat.values, np.asarray(lat), "lat"),
                   lon=idx(da.lon.values, np.asarray(lon), "lon")).assign_coords(lat=lat, lon=lon)
