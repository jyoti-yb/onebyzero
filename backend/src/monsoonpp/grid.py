"""Grids, static geography (terrain, land mask, coast distance) and named regions.

Three distinct concepts. Do not conflate them:

1. CANONICAL IMD GRID (`IMD_CANONICAL`): the official IMD 0.25 deg gridded-rainfall
   rectangle, 6.5-38.5 N x 66.5-100.0 E, 129 lat x 135 lon cell centres. This is the
   default `GridConfig` and the only grid that may be called "the IMD grid".
2. VERIFICATION / LAND MASK (`land` static variable): which cells of the grid carry
   observations. Real runs derive it from IMD itself (cells with valid data); synthetic
   runs use `INDIA_OUTLINE`. Ocean / no-data cells stay in the grid and are excluded
   by this mask, never by shrinking the grid.
3. RUNTIME DOMAIN (`cfg.grid`): the grid a run actually uses. It is either the canonical
   grid, a CROP of it (same 0.25 deg lattice, smaller bounds), or a coarsened/other
   development grid (e.g. the synthetic 0.5 deg run). `grid_role()` reports which.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from shapely import contains_xy
from shapely.geometry import Polygon

from .config import GridConfig

# ---------------------------------------------------------------- canonical IMD grid
IMD_CANONICAL = GridConfig(lat_min=6.5, lat_max=38.5, lon_min=66.5, lon_max=100.0, res=0.25)
IMD_NLAT, IMD_NLON = 129, 135
_TOL = 1e-6


def imd_canonical_coords() -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates of the official IMD 0.25 deg grid (129 x 135)."""
    lat, lon = make_coords(IMD_CANONICAL)
    assert (lat.size, lon.size) == (IMD_NLAT, IMD_NLON), (lat.size, lon.size)
    return lat, lon


def _on_lattice(x: float, origin: float, res: float) -> bool:
    k = (x - origin) / res
    return abs(k - round(k)) < _TOL


def grid_role(g: GridConfig) -> str:
    """'imd_canonical' | 'imd_crop' (subset of the canonical 0.25 lattice) | 'non_imd'."""
    c = IMD_CANONICAL
    same_res = abs(g.res - c.res) < _TOL
    if same_res and all(abs(getattr(g, k) - getattr(c, k)) < _TOL for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
        return "imd_canonical"
    inside = (g.lat_min >= c.lat_min - _TOL and g.lat_max <= c.lat_max + _TOL
              and g.lon_min >= c.lon_min - _TOL and g.lon_max <= c.lon_max + _TOL)
    aligned = all(_on_lattice(getattr(g, k), getattr(c, k.replace("max", "min")), c.res)
                  for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
    if same_res and inside and aligned:
        return "imd_crop"
    return "non_imd"


def check_imd_compatible(g: GridConfig) -> str:
    """Raise if a runtime grid cannot hold IMD rainfall without interpolation.
    IMD observations must be selected on their own lattice (canonical or crop), never
    interpolated. Returns the grid role."""
    role = grid_role(g)
    if role == "non_imd":
        raise ValueError(
            f"grid {g} is not the IMD 0.25 deg grid nor an aligned crop of it "
            f"({IMD_CANONICAL.lat_min}-{IMD_CANONICAL.lat_max} N, {IMD_CANONICAL.lon_min}-{IMD_CANONICAL.lon_max} E, "
            f"step {IMD_CANONICAL.res}); IMD rainfall would have to be interpolated.")
    return role


# ------------------------------------------------------------ verification mask
# Rough India outline (lon, lat). Used ONLY to build the synthetic land/verification
# mask. Real runs take the mask from IMD observations (cells with valid data).
INDIA_OUTLINE = [
    (68.4, 23.6), (69.0, 22.4), (70.5, 20.8), (72.6, 21.2), (72.8, 19.0), (73.3, 16.5),
    (74.0, 14.8), (74.8, 12.8), (75.6, 11.2), (76.3, 9.5), (77.3, 8.1), (78.2, 8.9),
    (79.3, 10.3), (79.8, 11.6), (80.3, 13.3), (80.2, 15.5), (81.3, 16.4), (82.3, 17.0),
    (84.0, 18.5), (85.5, 19.7), (87.0, 21.3), (88.3, 21.6), (89.0, 21.9), (89.0, 24.0),
    (92.0, 24.0), (92.5, 23.0), (93.3, 23.0), (93.4, 24.1), (94.2, 25.0), (95.0, 26.8),
    (96.9, 27.6), (96.0, 28.3), (94.5, 29.2), (92.0, 27.8), (89.8, 26.8), (88.8, 27.3),
    (88.1, 27.9), (88.0, 26.6), (85.8, 26.8), (84.0, 27.5), (81.0, 28.8), (80.1, 30.0),
    (79.0, 31.0), (78.5, 32.5), (79.5, 34.5), (77.8, 35.5), (74.5, 34.8), (73.9, 33.5),
    (74.6, 32.5), (74.5, 31.0), (73.9, 30.0), (72.9, 29.0), (71.1, 27.9), (70.0, 27.2),
    (69.5, 26.5), (70.3, 25.7), (71.0, 24.4), (68.8, 24.3),
]

# Named verification regions (lat_min, lat_max, lon_min, lon_max). Used by the
# error atlas and the stratified verification. Replace with IMD meteorological
# subdivisions once a shapefile is plugged in.
REGIONS = {
    "west_coast_ghats": (8.0, 21.0, 72.5, 76.0),
    "central_india": (18.0, 26.0, 76.0, 86.0),
    "northeast": (22.0, 29.5, 89.5, 97.5),
    "northwest": (23.0, 32.0, 68.0, 78.0),
    "south_peninsula": (8.0, 18.0, 76.0, 84.5),
    "himalayan_foothills": (27.0, 36.0, 74.0, 89.5),
    "east_coast_bay": (17.0, 23.0, 84.5, 89.5),
}

# Core monsoon zone for active/break classification (after Rajeevan et al. 2010,
# approximated by a box; swap for the published CMZ polygon in real runs).
CMZ_BOX = (18.0, 28.0, 65.0, 88.0)


def make_coords(g: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres from min to max (inclusive when max lies on the lattice).
    Integer point counts avoid floating-point drift in np.arange."""
    def axis(lo, hi):
        n = int(np.floor((hi - lo) / g.res + _TOL)) + 1
        return np.round(lo + np.arange(n) * g.res, 4)
    return axis(g.lat_min, g.lat_max), axis(g.lon_min, g.lon_max)


def india_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lon2, lat2 = np.meshgrid(lon, lat)
    return contains_xy(Polygon(INDIA_OUTLINE), lon2, lat2)


def synthetic_terrain(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Terrain with the three features that matter for monsoon rain:
    Western Ghats ridge, Himalayan wall, NE (Khasi–Jaintia) hills."""
    lon2, lat2 = np.meshgrid(lon, lat)
    # Western Ghats: ridge longitude varies with latitude
    ridge_lon = np.interp(lat2, [8, 11, 13, 16, 19, 21.5], [77.2, 76.6, 75.2, 74.0, 73.7, 73.6])
    ghats = 1100 * np.exp(-((lon2 - ridge_lon) / 0.45) ** 2) * ((lat2 > 8) & (lat2 < 21.5))
    # Himalaya: foothill latitude as a function of longitude, then a steep wall
    foot = np.interp(lon2, [72, 74, 77, 80, 84, 88, 92, 97], [34.5, 33.0, 31.0, 29.5, 28.0, 27.2, 27.5, 28.2])
    himal = 4500 / (1 + np.exp(-(lat2 - foot - 0.8) / 0.35))
    # NE hills (Shillong plateau) + Arakan
    ne = 1500 * np.exp(-(((lat2 - 25.5) / 0.6) ** 2 + ((lon2 - 91.5) / 1.2) ** 2))
    ne += 1200 * np.exp(-(((lon2 - 93.8) / 0.6) ** 2)) * ((lat2 > 22) & (lat2 < 27))
    # Eastern Ghats (weak)
    eg = 500 * np.exp(-(((lat2 - 18.5) / 1.5) ** 2 + ((lon2 - 82.5) / 0.8) ** 2))
    return (ghats + himal + ne + eg).astype("float32")


def coast_distance_km(land: np.ndarray, res_deg: float) -> np.ndarray:
    # distance from each land cell to the nearest sea cell (0 over sea)
    return (distance_transform_edt(land) * res_deg * 111.0).astype("float32")


def terrain_slope_aspect(elev: np.ndarray, lat: np.ndarray, res_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope and downslope aspect (clockwise from north) on a regular lat-lon grid."""
    elevation = np.asarray(elev, dtype="float64")
    dy = res_deg * 111_000.0
    dx = dy * np.cos(np.deg2rad(lat))[:, None]
    dz_dy_grid, dz_dx_grid = np.gradient(elevation)
    dz_dx, dz_dy = dz_dx_grid / dx, dz_dy_grid / dy
    slope = np.rad2deg(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = (np.rad2deg(np.arctan2(-dz_dx, -dz_dy)) + 360.0) % 360.0
    aspect[np.hypot(dz_dx, dz_dy) < 1e-12] = 0.0
    return slope.astype("float32"), aspect.astype("float32")


def build_static(g: GridConfig, land: np.ndarray | None = None, elev: np.ndarray | None = None) -> xr.Dataset:
    lat, lon = make_coords(g)
    if land is None:
        land = india_mask(lat, lon)
    if elev is None:
        elev = synthetic_terrain(lat, lon)
    slope, aspect = terrain_slope_aspect(elev, lat, g.res)
    out = xr.Dataset(
        {
            "elev": (("lat", "lon"), elev.astype("float32")),
            "slope": (("lat", "lon"), slope),
            "aspect": (("lat", "lon"), aspect),
            "land": (("lat", "lon"), land.astype(bool)),
            "coast_dist": (("lat", "lon"), coast_distance_km(land, g.res)),
        },
        coords={"lat": lat, "lon": lon},
    )
    out.attrs.update(static_source="monsoonpp synthetic terrain fixture", is_synthetic="true")
    out.elev.attrs.update(units="m", long_name="synthetic terrain height")
    out.slope.attrs.update(units="degree", long_name="terrain slope")
    out.aspect.attrs.update(units="degree", long_name="downslope aspect clockwise from north")
    out.land.attrs.update(units="1", long_name="synthetic land mask")
    out.coast_dist.attrs.update(units="km", long_name="distance from land cell to nearest sea cell")
    return out


def region_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorised region label for arrays of lat/lon (first match wins)."""
    out = np.full(np.shape(lat), "other", dtype=object)
    for name, (a, b, c, d) in REGIONS.items():
        m = (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d) & (out == "other")
        out[m] = name
    return out


def box_mask(lat: np.ndarray, lon: np.ndarray, box) -> np.ndarray:
    a, b, c, d = box
    lon2, lat2 = np.meshgrid(lon, lat)
    return (lat2 >= a) & (lat2 <= b) & (lon2 >= c) & (lon2 <= d)
