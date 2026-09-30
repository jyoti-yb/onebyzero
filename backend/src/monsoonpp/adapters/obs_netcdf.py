"""Official daily gridded rainfall delivered as NetCDF files (decode only).

  imd_netcdf          IMD 0.25 deg gauge gridded rainfall       -> FINAL_TRUTH
  imd_ncmrwf_merged   IMD–NCMRWF 0.25 deg merged gauge+satellite -> OFFICIAL_ALTERNATIVE

Strict decoding, no undocumented assumptions:
* variable names: taken from config, else the unique CF-ish candidate; ambiguity raises.
* units: must be present in the file and be a millimetre depth, else the run config must
  supply `units_declared` WITH `units_evidence` (a documentation reference). Recorded.
* missing values: declared _FillValue / missing_value are masked. Undeclared negative
  values raise (unknown encoding).
* time: CF "X since ..." units required; strictly daily steps, no duplicates. The raw
  numeric time and its units are preserved; the date label is NOT interpreted here.
* grid: runtime grid cells are SELECTED exactly from the native grid; if any runtime cell
  is not a native cell the decode fails (observations are never interpolated).
* time convention: always UNVERIFIED at decode (data.obs_time). Interpretation happens
  later, only from the verified run config.

Config (adapter_options.obs):
  files: [paths or globs]      required
  origin: "URL or how obtained" recorded as source_url_or_origin
  variable / lat / lon / time: optional explicit variable names
  units_declared + units_evidence: only if the file lacks a units attribute
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..data.align import standardise_latlon
from ..data.obs_source import ObsSourceMeta, ProductRole, attach
from ..data.obs_time import decode_provenance
from ..grid import grid_role, make_coords
from .base import ObsAdapter, register

RAIN_NAMES = ("rainfall", "rain", "rf", "precip", "precipitation", "pr", "prcp")
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")
TIME_NAMES = ("time", "t")
DAILY_DEPTH_UNITS = {"mm", "mm/day", "mm day-1", "mm d-1", "mm/d", "millimeter", "millimetre",
                     "millimeters", "millimetres"}


class ObsDecodeError(RuntimeError):
    """The file cannot be decoded without an undocumented assumption."""


def _one(names: list[str], candidates, what: str, explicit: str | None, where: str) -> str:
    if explicit:
        if explicit not in names:
            raise ObsDecodeError(f"{where}: configured {what} variable {explicit!r} not in file {names}")
        return explicit
    low = {n.lower(): n for n in names}
    hits = [low[c] for c in candidates if c in low]
    if len(hits) != 1:
        raise ObsDecodeError(f"{where}: cannot identify the {what} variable unambiguously "
                             f"(candidates {hits or 'none'} in {names}); set adapter_options.obs.{what}")
    return hits[0]


def _files(spec) -> list[Path]:
    specs = [spec] if isinstance(spec, str) else list(spec or [])
    out = []
    for s in specs:
        m = sorted(glob.glob(str(s)))
        out += [Path(x) for x in (m or [s])]
    if not out:
        raise ObsDecodeError("adapter_options.obs.files is empty")
    for p in out:
        if not p.exists():
            raise ObsDecodeError(f"observation file not found: {p}")
    return out


def decode_daily_netcdf(path: Path, opts: dict) -> tuple[xr.DataArray, dict]:
    """One file -> (rain DataArray on NATIVE grid with raw time preserved, facts dict)."""
    import cftime
    import netCDF4
    where = str(path)
    with netCDF4.Dataset(path, "r") as nc:
        nc.set_auto_maskandscale(False)
        names = list(nc.variables)
        three = [n for n in names if nc.variables[n].ndim == 3]
        rv = _one(three, RAIN_NAMES, "variable", opts.get("variable"), where)
        la = _one(names, LAT_NAMES, "lat", opts.get("lat"), where)
        lo = _one(names, LON_NAMES, "lon", opts.get("lon"), where)
        tv = _one(names, TIME_NAMES, "time", opts.get("time"), where)
        v, t = nc.variables[rv], nc.variables[tv]
        if list(v.dimensions) != [nc.variables[tv].dimensions[0], nc.variables[la].dimensions[0],
                                  nc.variables[lo].dimensions[0]]:
            raise ObsDecodeError(f"{where}: {rv} dims {v.dimensions} are not (time, lat, lon)")
        attrs = {k: v.getncattr(k) for k in v.ncattrs()}
        tattrs = {k: t.getncattr(k) for k in t.ncattrs()}

        # units
        units = str(attrs.get("units", "")).strip()
        units_source = "file attribute"
        if not units:
            if not (opts.get("units_declared") and opts.get("units_evidence")):
                raise ObsDecodeError(f"{where}: {rv} has no units attribute; set adapter_options.obs.units_declared "
                                     f"AND units_evidence (documentation reference)")
            units, units_source = str(opts["units_declared"]), f"declared by operator: {opts['units_evidence']}"
        if units.lower() not in DAILY_DEPTH_UNITS:
            raise ObsDecodeError(f"{where}: units {units!r} are not a daily millimetre depth {sorted(DAILY_DEPTH_UNITS)}")

        # packing
        if "scale_factor" in attrs or "add_offset" in attrs:
            scale, off = float(attrs.get("scale_factor", 1.0)), float(attrs.get("add_offset", 0.0))
        else:
            scale, off = 1.0, 0.0
        raw = np.asarray(v[:], dtype="float64")
        fills = [float(np.ravel(attrs[k])[0]) for k in ("_FillValue", "missing_value") if k in attrs]
        mask = np.isnan(raw)
        for fv in fills:
            mask |= np.isclose(raw, fv, rtol=0, atol=1e-6 * max(1.0, abs(fv)))
        data = np.where(mask, np.nan, raw * scale + off)
        neg = np.nansum(data < 0)
        if neg:
            raise ObsDecodeError(f"{where}: {int(neg)} negative values not covered by a declared fill value "
                                 f"(declared {fills or 'none'}); missing-value encoding undocumented")

        # time (raw preserved; decoded only to calendar dates for labelling)
        tunits = str(tattrs.get("units", ""))
        if "since" not in tunits:
            raise ObsDecodeError(f"{where}: time units {tunits!r} are not CF 'X since ...'")
        traw = np.asarray(t[:], dtype="float64")
        dts = cftime.num2date(traw, tunits, calendar=tattrs.get("calendar", "standard"),
                              only_use_cftime_datetimes=False)
        times = pd.DatetimeIndex([pd.Timestamp(str(d)) for d in dts])
        if times.has_duplicates:
            raise ObsDecodeError(f"{where}: duplicate time stamps")
        if len(times) > 1 and not (np.diff(times.values) == np.timedelta64(1, "D")).all():
            raise ObsDecodeError(f"{where}: time steps are not exactly 1 day")

        lat = np.asarray(nc.variables[la][:], dtype="float64")
        lon = np.asarray(nc.variables[lo][:], dtype="float64")
        facts = {"units": units, "units_source": units_source, "declared_fill": fills, "time_units": tunits,
                 "calendar": tattrs.get("calendar"), "global_attrs": {k: str(nc.getncattr(k)) for k in nc.ncattrs()}}
    da = xr.DataArray(data.astype("float32"), dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": lat, "lon": lon,
                              "source_time_raw": ("time", traw)}, name="rain")
    return standardise_latlon(da), facts


def _native_grid(da: xr.DataArray) -> tuple[str, float]:
    lat, lon = da.lat.values, da.lon.values
    dl = np.diff(lat); dn = np.diff(lon)
    if not (np.allclose(dl, dl[0], atol=1e-5) and np.allclose(dn, dn[0], atol=1e-5) and abs(dl[0] - dn[0]) < 1e-5):
        raise ObsDecodeError("native grid is not a regular lat-lon grid with equal spacing")
    res = round(float(dl[0]), 6)
    return (f"regular lat-lon {res} deg, {lat[0]:.3f}..{lat[-1]:.3f} N x {lon[0]:.3f}..{lon[-1]:.3f} E "
            f"({lat.size} x {lon.size})"), res


def select_exact(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray, tol: float = 1e-4) -> xr.DataArray:
    """Runtime grid must be a subset of native cells. Never interpolates."""
    def idx(src, tgt, name):
        j = np.abs(src[:, None] - tgt[None, :]).argmin(0)
        bad = np.abs(src[j] - tgt) > tol
        if bad.any():
            raise ObsDecodeError(f"runtime {name} {tgt[bad][:3]}... are not native grid points; "
                                 f"observations are never interpolated")
        return j
    return da.isel(lat=idx(da.lat.values, lat, "lat"), lon=idx(da.lon.values, lon, "lon")).assign_coords(lat=lat, lon=lon)


class _DailyNetCDFObs(ObsAdapter):
    SOURCE_NAME: str
    PRODUCT_NAME: str
    ROLE: ProductRole
    REQUIRED_RES = 0.25

    def __init__(self, cfg):
        super().__init__(cfg)
        self.opts = dict(cfg.adapter_options.get("obs", {}))
        self.lat, self.lon = make_coords(cfg.grid)

    def decode(self) -> tuple[xr.DataArray, list[str], dict, str]:
        parts, files, facts = [], [], None
        for p in _files(self.opts.get("files")):
            da, f = decode_daily_netcdf(p, self.opts)
            if facts and (f["units"] != facts["units"] or f["time_units"].split("since")[0] != facts["time_units"].split("since")[0]):
                raise ObsDecodeError(f"{p}: units/time units differ from earlier files")
            facts = facts or f
            parts.append(da)
            files += [p.name] * da.sizes["time"]
        da = xr.concat(parts, "time")
        if pd.DatetimeIndex(da.time.values).has_duplicates:
            raise ObsDecodeError("the same date appears in more than one file")
        order = np.argsort(da.time.values)
        da, files = da.isel(time=order), [files[i] for i in order]
        native, res = _native_grid(da)
        if abs(res - self.REQUIRED_RES) > 1e-6:
            raise ObsDecodeError(f"{self.SOURCE_NAME} product must be {self.REQUIRED_RES} deg; file grid is {res} deg")
        return da, files, facts, native

    def meta(self, files, facts, native) -> ObsSourceMeta:
        uniq = sorted(set(files))
        return ObsSourceMeta(
            source_name=self.SOURCE_NAME, product_name=self.PRODUCT_NAME, product_role=self.ROLE,
            source_filename="; ".join(uniq), source_url_or_origin=str(self.opts.get("origin", "not provided")),
            observation_resolution=f"{self.REQUIRED_RES} deg, daily (24 h accumulation per date label)",
            native_grid=native, units=f"{facts['units']} ({facts['units_source']})",
            time_convention="UNVERIFIED", is_proxy=False)

    def load(self, dates: pd.DatetimeIndex) -> xr.Dataset:
        da, files, facts, native = self.decode()
        keep = pd.DatetimeIndex(da.time.values).normalize().isin(pd.DatetimeIndex(dates).normalize())
        da = da.isel(time=np.nonzero(keep)[0])
        files = [f for f, k in zip(files, keep) if k]
        da = select_exact(da, self.lat, self.lon)
        obs = xr.Dataset({"rain": da.drop_vars("source_time_raw")})
        obs = decode_provenance(obs, source=self.SOURCE_NAME, product_name=self.PRODUCT_NAME,
                                source_filenames=files, units="mm", grid_role=grid_role(self.cfg.grid))
        obs = obs.assign_coords(source_time_raw=("time", da.source_time_raw.values))
        obs.attrs.update(source_time_units=facts["time_units"], source_calendar=str(facts["calendar"]))
        return attach(obs, self.meta(files, facts, native))


@register("obs", "imd_netcdf")
class IMDNetCDFAdapter(_DailyNetCDFObs):
    SOURCE_NAME = "IMD"
    PRODUCT_NAME = "IMD 0.25 deg daily gridded gauge rainfall (NetCDF)"
    ROLE = ProductRole.FINAL_TRUTH


@register("obs", "imd_ncmrwf_merged")
class IMDNCMRWFMergedAdapter(_DailyNetCDFObs):
    SOURCE_NAME = "IMD-NCMRWF"
    PRODUCT_NAME = "IMD-NCMRWF 0.25 deg daily merged gauge + satellite rainfall (NetCDF)"
    ROLE = ProductRole.OFFICIAL_ALTERNATIVE
