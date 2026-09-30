"""NASA GPM IMERG — SMOKE-TEST PROXY ONLY. Never IMD, never truth.

Supported: IMERG half-hourly granules in HDF5 (e.g. GPM_3IMERGHH V07 Final), group /Grid.
Decoding only:
* variable `precipitation` (configurable), a RATE; units must be stated in the file (mm/hr).
  No conversion to depth, no aggregation.
* NATIVE grid kept (0.1 deg). No regridding. An optional lat/lon box only SELECTS native
  cells (to keep memory sane over India); the selection is recorded.
* axis order from HDF5 dimension scales, else from unambiguous axis lengths; else error.
* timing ONLY from /Grid/time_bnds (explicit start/end of each granule). The granule must
  be exactly 30 minutes and `time` must lie inside its bounds. Filenames are never parsed.
  No IMD date convention is applied or inferred: coords are source_interval_start/end,
  never valid_start/valid_end or a date label.
* Daily IMERG products are rejected: their "data day" boundary is not stated in the file.

Config (adapter_options.obs):
  files: [paths or globs]    origin: "URL / how obtained"    variable: default "precipitation"
  bbox: [lat_min, lat_max, lon_min, lon_max]  (native-cell selection; default: runtime grid +1 deg)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..data.obs_source import ObsSourceMeta, ProductRole, attach
from .base import NativeObsAdapter, register
from .obs_netcdf import ObsDecodeError, _files

RATE_UNITS = {"mm/hr", "mm/h", "mm hr-1", "mm h-1", "mm hr^-1"}
GRANULE = pd.Timedelta(minutes=30)
PRODUCT_NAME = "NASA GPM IMERG half-hourly precipitation rate (HDF5)"


def _s(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode(errors="replace")
    if isinstance(x, np.ndarray) and x.size == 1:
        return _s(x.ravel()[0])
    return str(x)


def _dim_names(dset) -> list[str | None]:
    out = []
    for d in dset.dims:
        try:
            out.append(d[0].name.rsplit("/", 1)[-1] if len(d) else None)
        except Exception:
            out.append(None)
    return out


def decode_imerg_granule(path, variable: str = "precipitation", bbox=None) -> tuple[xr.Dataset, dict]:
    import cftime
    import h5py
    where = str(path)
    with h5py.File(path, "r") as f:
        if "Grid" not in f:
            raise ObsDecodeError(f"{where}: no /Grid group (not an IMERG L3 HDF5 granule)")
        g = f["Grid"]
        for k in (variable, "lat", "lon", "time"):
            if k not in g:
                raise ObsDecodeError(f"{where}: /Grid/{k} missing")
        if "time_bnds" not in g:
            raise ObsDecodeError(f"{where}: /Grid/time_bnds missing — granule interval is not explicit")
        v = g[variable]
        units = _s(v.attrs.get("units", "")).strip()
        if units.lower() not in RATE_UNITS:
            raise ObsDecodeError(f"{where}: {variable} units {units!r} not an explicit mm/hr rate")
        lat = np.asarray(g["lat"][:], dtype="float64")
        lon = np.asarray(g["lon"][:], dtype="float64")

        # axis order
        names = _dim_names(v)
        if v.ndim != 3:
            raise ObsDecodeError(f"{where}: {variable} is {v.ndim}-D, expected (time, lon, lat) in some order")
        if all(n in ("time", "lat", "lon") for n in names) and len(set(names)) == 3:
            order = names
        else:
            by_len = {}
            for i, n in enumerate(v.shape):
                by_len.setdefault(n, []).append(i)
            if lat.size == lon.size or any(len(ix) > 1 for ix in by_len.values()):
                raise ObsDecodeError(f"{where}: axis order not declared and not identifiable from lengths")
            order = [None] * 3
            order[by_len[lat.size][0]] = "lat"; order[by_len[lon.size][0]] = "lon"
            order[[i for i in range(3) if order[i] is None][0]] = "time"
        raw = np.asarray(v[...], dtype="float64")
        raw = np.transpose(raw, [order.index("time"), order.index("lat"), order.index("lon")])

        fills = []
        for k in ("_FillValue", "CodeMissingValue", "missing_value"):
            if k in v.attrs:
                try:
                    fills.append(float(_s(v.attrs[k])))
                except ValueError:
                    raise ObsDecodeError(f"{where}: unreadable {k}={v.attrs[k]!r}")
        mask = np.isnan(raw)
        for fv in fills:
            mask |= np.isclose(raw, fv, rtol=0, atol=1e-3)
        data = np.where(mask, np.nan, raw)
        if np.nansum(data < 0):
            raise ObsDecodeError(f"{where}: negative rates not covered by declared fill values {fills or 'none'}")

        # explicit timing
        t, tb = g["time"], g["time_bnds"]
        tunits = _s(t.attrs.get("units", ""))
        bunits = _s(tb.attrs.get("units", tunits)) or tunits
        cal = _s(t.attrs.get("calendar", "standard")) or "standard"
        if "since" not in tunits or "since" not in bunits:
            raise ObsDecodeError(f"{where}: time units not CF 'X since ...' ({tunits!r}, {bunits!r})")
        ts = cftime.num2date(np.asarray(t[:], "float64"), tunits, calendar=cal, only_use_cftime_datetimes=False)
        bb = cftime.num2date(np.asarray(tb[:], "float64"), bunits, calendar=cal, only_use_cftime_datetimes=False)
        ts = pd.DatetimeIndex([pd.Timestamp(str(x)) for x in np.ravel(ts)])
        bb = np.asarray([[pd.Timestamp(str(x)) for x in row] for row in np.atleast_2d(bb)])
        if bb.shape != (len(ts), 2):
            raise ObsDecodeError(f"{where}: time_bnds shape {bb.shape} does not match time {len(ts)}")
        starts, ends = pd.DatetimeIndex(bb[:, 0]), pd.DatetimeIndex(bb[:, 1])
        if not ((ends - starts) == GRANULE).all():
            raise ObsDecodeError(f"{where}: granule length {(ends - starts)[0]} != 30 min "
                                 f"(daily/monthly IMERG not supported: data-day boundary not stated in file)")
        if not ((ts >= starts) & (ts < ends)).all():
            raise ObsDecodeError(f"{where}: time {ts[0]} lies outside its bounds {starts[0]}..{ends[0]} — ambiguous")
        header = _s(f.attrs.get("FileHeader", "")) if "FileHeader" in f.attrs else ""

    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ObsDecodeError(f"{where}: lat/lon not strictly ascending")
    res_lat, res_lon = float(np.median(np.diff(lat))), float(np.median(np.diff(lon)))
    if not (np.allclose(np.diff(lat), res_lat, atol=1e-4) and np.allclose(np.diff(lon), res_lon, atol=1e-4)):
        raise ObsDecodeError(f"{where}: native grid spacing not uniform")
    native = (f"regular lat-lon {res_lat:.2f} deg, {lat[0]:.2f}..{lat[-1]:.2f} N x {lon[0]:.2f}..{lon[-1]:.2f} E "
              f"({lat.size} x {lon.size})")
    iy = np.arange(lat.size); ix = np.arange(lon.size)
    if bbox is not None:
        a, b, c, d = bbox
        iy = np.nonzero((lat >= a) & (lat <= b))[0]; ix = np.nonzero((lon >= c) & (lon <= d))[0]
        if not iy.size or not ix.size:
            raise ObsDecodeError(f"{where}: bbox {bbox} selects no native cells")
    ds = xr.Dataset(
        {"precipitation_rate": (("time", "lat", "lon"), data[:, iy][:, :, ix].astype("float32"))},
        coords={"time": starts, "lat": lat[iy], "lon": lon[ix],
                "source_interval_start": ("time", starts.values), "source_interval_end": ("time", ends.values),
                "source_time_raw_units": ((), tunits)})
    ds["precipitation_rate"].attrs["units"] = units
    return ds, {"units": units, "native_grid": native, "res": (res_lat, res_lon), "fills": fills, "file_header": header}


@register("obs_native", "imerg")
class IMERGAdapter(NativeObsAdapter):
    SOURCE_NAME = "NASA-GPM-IMERG"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.opts = dict(cfg.adapter_options.get("obs", {}))
        g = cfg.grid
        self.bbox = self.opts.get("bbox", [g.lat_min - 1, g.lat_max + 1, g.lon_min - 1, g.lon_max + 1])

    def load(self, start=None, end=None) -> xr.Dataset:
        parts, names, facts = [], [], None
        for p in _files(self.opts.get("files")):
            ds, f = decode_imerg_granule(p, self.opts.get("variable", "precipitation"), self.bbox)
            if facts and (f["units"] != facts["units"] or f["native_grid"] != facts["native_grid"]):
                raise ObsDecodeError(f"{p}: units or native grid differ from earlier granules")
            facts = facts or f
            s0 = pd.Timestamp(ds.source_interval_start.values[0])
            if (start is not None and s0 < pd.Timestamp(start)) or (end is not None and s0 >= pd.Timestamp(end)):
                continue                                   # selection by granule metadata, not filename
            parts.append(ds.assign_coords(source_filename=("time", [p.name] * ds.sizes["time"])))
            names.append(p.name)
        if not parts:
            raise ObsDecodeError("no IMERG granules inside the requested interval")
        out = xr.concat(parts, "time", data_vars="all", coords="different", compat="equals").sortby("time")
        st = pd.DatetimeIndex(out.source_interval_start.values)
        if st.has_duplicates:
            raise ObsDecodeError("duplicate IMERG granules for the same interval")
        en = pd.DatetimeIndex(out.source_interval_end.values)
        gaps = int(((st[1:] - en[:-1]) > pd.Timedelta(0)).sum()) if len(st) > 1 else 0
        overlaps = int(((st[1:] - en[:-1]) < pd.Timedelta(0)).sum()) if len(st) > 1 else 0
        if overlaps:
            raise ObsDecodeError(f"{overlaps} overlapping IMERG granules")
        meta = ObsSourceMeta(
            source_name=self.SOURCE_NAME, product_name=PRODUCT_NAME, product_role=ProductRole.SMOKE_TEST_PROXY,
            source_filename="; ".join(sorted(set(names))) if len(names) <= 20 else f"{len(names)} granules (see source_filename coord)",
            source_url_or_origin=str(self.opts.get("origin", "not provided")),
            observation_resolution="0.1 deg, 30-minute granules (precipitation RATE)",
            native_grid=facts["native_grid"], units=facts["units"],
            time_convention="EXPLICIT_SOURCE_INTERVALS (from /Grid/time_bnds; no date-label convention)",
            is_proxy=True)
        out = attach(out, meta)
        out.attrs.update(native_subset_bbox=str(self.bbox), n_granules=len(st), n_gaps=gaps,
                         interval_first=str(st[0]), interval_last_end=str(en[-1]),
                         WARNING="SMOKE-TEST PROXY: satellite estimate. Not IMD. Not observational truth.")
        return out
