"""District-level product: area-weighted aggregation of the corrected grid.

No per-district models: correct on the grid, aggregate with exact cell/polygon
overlap weights. Plug a real district GeoJSON via `district_geojson` in the config
(e.g. Survey of India / datameet / GADM level-2). Without it, demo tiles are used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box, shape

from ..grid import INDIA_OUTLINE

# IMD rainfall intensity categories (mm/day)
IMD_CATEGORIES = [(0.0, "no rain"), (0.1, "very light"), (2.5, "light"), (15.6, "moderate"),
                  (64.5, "heavy"), (115.6, "very heavy"), (204.5, "extremely heavy")]
NAME_KEYS = ("district", "DISTRICT", "dtname", "NAME_2", "district_name", "name")


def imd_category(mm: float) -> str:
    lab = IMD_CATEGORIES[0][1]
    for lo, name in IMD_CATEGORIES:
        if mm >= lo:
            lab = name
    return lab


def warning_colour(max_mm: float, p_heavy: float, p_vheavy: float) -> str:
    """Prototype rule, NOT IMD's impact-based warning matrix (needs impact data)."""
    if max_mm >= 204.5 or p_vheavy >= 0.5:
        return "red"
    if max_mm >= 115.6 or p_heavy >= 0.5 or p_vheavy >= 0.25:
        return "orange"
    if max_mm >= 64.5 or p_heavy >= 0.2:
        return "yellow"
    return "green"


@dataclass
class DistrictWeights:
    names: list[str]
    d_idx: np.ndarray
    iy: np.ndarray
    ix: np.ndarray
    w: np.ndarray      # normalised within each district


def load_districts(path: str | None, res_tile: float = 2.0) -> list[tuple[str, Polygon]]:
    if path:
        gj = json.loads(open(path).read())
        out = []
        for i, f in enumerate(gj["features"]):
            props = f.get("properties", {})
            name = next((str(props[k]) for k in NAME_KEYS if k in props), f"district_{i}")
            st = next((str(props[k]) for k in ("state", "STATE", "stname", "NAME_1") if k in props), None)
            out.append((f"{name} ({st})" if st else name, shape(f["geometry"])))
        return out
    india = Polygon(INDIA_OUTLINE)
    minx, miny, maxx, maxy = india.bounds
    out = []
    for la in np.arange(np.floor(miny), maxy, res_tile):
        for lo in np.arange(np.floor(minx), maxx, res_tile):
            g = box(lo, la, lo + res_tile, la + res_tile).intersection(india)
            if g.area > 0.3:
                out.append((f"DEMO_{la:.0f}N_{lo:.0f}E", g))
    return out


def build_weights(districts, lat: np.ndarray, lon: np.ndarray, valid: np.ndarray) -> DistrictWeights:
    res = float(lat[1] - lat[0])
    names, D, Y, X, W = [], [], [], [], []
    for d, (name, poly) in enumerate(districts):
        minx, miny, maxx, maxy = poly.bounds
        ys = np.nonzero((lat + res / 2 >= miny) & (lat - res / 2 <= maxy))[0]
        xs = np.nonzero((lon + res / 2 >= minx) & (lon - res / 2 <= maxx))[0]
        rows = []
        for iy in ys:
            for ix in xs:
                if not valid[iy, ix]:
                    continue
                a = poly.intersection(box(lon[ix] - res / 2, lat[iy] - res / 2, lon[ix] + res / 2, lat[iy] + res / 2)).area
                if a > 0:
                    rows.append((iy, ix, a))
        if not rows:   # district smaller than a cell and off-mask: nearest valid cell
            c = poly.representative_point()
            iy, ix = int(np.abs(lat - c.y).argmin()), int(np.abs(lon - c.x).argmin())
            rows = [(iy, ix, 1.0)]
        tot = sum(r[2] for r in rows)
        names.append(name)
        for iy, ix, a in rows:
            D.append(len(names) - 1); Y.append(iy); X.append(ix); W.append(a / tot)
    return DistrictWeights(names, np.array(D), np.array(Y), np.array(X), np.array(W))


def aggregate(fields: dict[str, np.ndarray], dw: DistrictWeights, rain_key: str,
              p_heavy_key: str | None, p_vheavy_key: str | None) -> pd.DataFrame:
    """fields: name -> 2D (lat, lon) grid. Returns one row per district."""
    df = pd.DataFrame({"d": dw.d_idx, "w": dw.w})
    for k, g in fields.items():
        df[k] = np.nan_to_num(g[dw.iy, dw.ix])
    rows = []
    for d, g in df.groupby("d"):
        r = {"district": dw.names[d], "n_cells": int(len(g))}
        r["rain_mean_mm"] = float(np.sum(g.w * g[rain_key]))
        r["rain_max_cell_mm"] = float(g[rain_key].max())
        if "raw" in g:
            r["raw_mean_mm"] = float(np.sum(g.w * g["raw"]))
        ph = float(g[p_heavy_key].max()) if p_heavy_key else 0.0
        pv = float(g[p_vheavy_key].max()) if p_vheavy_key else 0.0
        r["p_heavy_max"], r["p_very_heavy_max"] = ph, pv
        r["p_heavy_area_mean"] = float(np.sum(g.w * g[p_heavy_key])) if p_heavy_key else 0.0
        r["category"] = imd_category(r["rain_mean_mm"])
        r["warning"] = warning_colour(r["rain_max_cell_mm"], ph, pv)
        rows.append(r)
    order = {"red": 0, "orange": 1, "yellow": 2, "green": 3}
    out = pd.DataFrame(rows)
    return out.sort_values(["warning", "rain_mean_mm"], key=lambda s: s.map(order) if s.name == "warning" else -s).reset_index(drop=True)
