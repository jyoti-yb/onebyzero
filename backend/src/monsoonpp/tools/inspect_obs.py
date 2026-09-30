"""Read-only inspection of an official observation rainfall file (IMD NetCDF / .grd).

Reports what the file itself says. It never:
  * modifies the source file (hash is checked before/after),
  * aligns anything with GFS,
  * sets or recommends a time convention (obs_time_convention stays UNVERIFIED unless a
    human confirms explicit evidence),
  * treats the filename as evidence (filename is recorded for provenance only).

Usage:  monsoonpp inspect-obs PATH [--out DIR]
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..grid import IMD_NLAT, IMD_NLON, imd_canonical_coords

RAIN_NAMES = ("rf", "rain", "rainfall", "precip", "precipitation", "pr", "prcp", "tp")
LAT_NAMES = ("lat", "latitude", "y", "lats")
LON_NAMES = ("lon", "longitude", "x", "lons")
TIME_NAMES = ("time", "t", "date", "day")
FILL_CANDIDATES = (-999.0, -99.9, -99.0, -9999.0, 9.96921e36, 1e20, -1e20)

# Text that MIGHT bear on which 24 h period a date label covers. Matches are reported
# verbatim for a human to judge; they never set the convention automatically.
EVIDENCE_PATTERNS = [
    r"08\s*[:.]?\s*30", r"\b0?3\s*[:.]?\s*00\s*(utc|gmt|z)\b", r"\b03\s*z\b", r"\b0300\b", r"\bist\b",
    r"\bending\b", r"\bending at\b", r"\bstarting\b", r"\bprevious day\b", r"\bnext day\b",
    r"\bpreceding\b", r"\b24[\s-]*h(ou)?r", r"\baccumulat", r"\bvalid\b", r"\bobservation time\b",
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _format(p: Path) -> str:
    head = p.read_bytes()[:8] if p.stat().st_size < 64 else open(p, "rb").read(8)
    if head.startswith(b"CDF\x01") or head.startswith(b"CDF\x02") or head.startswith(b"CDF\x05"):
        return "netcdf3"
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5/netcdf4"
    if head.startswith(b"GRIB"):
        return "grib"
    return "raw_binary"


def _jsonable(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


def _pick(names, candidates, ndim_ok=None):
    low = {n.lower(): n for n in names}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _axis_report(vals: np.ndarray, canonical: np.ndarray, name: str) -> dict:
    vals = np.asarray(vals, dtype="float64")
    d = np.diff(vals) if vals.size > 1 else np.array([])
    asc = bool(d.size and np.all(d > 0))
    desc = bool(d.size and np.all(d < 0))
    spacing = float(np.median(np.abs(d))) if d.size else None
    uniform = bool(d.size and np.allclose(np.abs(d), spacing, atol=1e-6))
    s = np.sort(vals)
    return {
        "name": name, "count": int(vals.size), "first": float(vals[0]), "last": float(vals[-1]),
        "min": float(vals.min()), "max": float(vals.max()),
        "ordering": "ascending" if asc else "descending" if desc else "non-monotonic",
        "spacing": spacing, "uniform_spacing": uniform,
        "matches_canonical_imd": bool(s.size == canonical.size and np.allclose(s, canonical, atol=1e-4)),
        "canonical": {"count": int(canonical.size), "first": float(canonical[0]), "last": float(canonical[-1])},
    }


def _scan_evidence(attr_sources: dict[str, dict]) -> list[dict]:
    hits = []
    for where, attrs in attr_sources.items():
        for k, v in attrs.items():
            text = str(v)
            for pat in EVIDENCE_PATTERNS:
                if re.search(pat, text, flags=re.I):
                    hits.append({"location": where, "attribute": k, "pattern": pat, "text": text[:500]})
                    break
    return hits


def _value_report(raw: np.ndarray, attrs: dict) -> dict:
    raw = np.asarray(raw, dtype="float64")
    declared = [attrs.get(k) for k in ("_FillValue", "missing_value") if k in attrs]
    declared = [float(np.ravel(x)[0]) for x in declared if x is not None]
    counts = {}
    for fv in sorted(set(declared) | set(FILL_CANDIDATES)):
        n = int(np.sum(np.isclose(raw, fv, rtol=1e-6, atol=1e-6)))
        if n:
            counts[str(fv)] = n
    nan_n = int(np.isnan(raw).sum())
    mask = np.isnan(raw)
    for fv in [float(x) for x in counts]:
        mask |= np.isclose(raw, fv, rtol=1e-6, atol=1e-6)
    valid = raw[~mask]
    out = {
        "raw_min": float(np.nanmin(raw)) if raw.size and not np.isnan(raw).all() else None,
        "raw_max": float(np.nanmax(raw)) if raw.size and not np.isnan(raw).all() else None,
        "declared_fill_values": declared, "fill_like_value_counts": counts, "nan_count": nan_n,
        "missing_fraction": float(mask.mean()) if raw.size else None,
        "valid_count": int(valid.size),
    }
    if valid.size:
        wet = valid[valid > 0]
        out.update(valid_min=float(valid.min()), valid_max=float(valid.max()),
                   negative_valid_count=int((valid < 0).sum()), wet_fraction=float((valid > 0).mean()),
                   wet_p50=float(np.percentile(wet, 50)) if wet.size else None,
                   wet_p99=float(np.percentile(wet, 99)) if wet.size else None)
    return out


def _cells_with_data(raw: np.ndarray, attrs: dict, time_axis: int) -> int | None:
    if raw.ndim != 3:
        return None
    r = np.asarray(raw, dtype="float64")
    bad = np.isnan(r)
    for fv in list(FILL_CANDIDATES) + [attrs.get("_FillValue"), attrs.get("missing_value")]:
        if fv is not None:
            bad |= np.isclose(r, float(np.ravel(fv)[0]), rtol=1e-6, atol=1e-6)
    return int((~bad).any(axis=time_axis).sum())


def _time_report(raw_vals, attrs: dict) -> dict:
    import cftime
    units, cal = attrs.get("units"), attrs.get("calendar", "standard")
    rep = {"raw_units": units, "calendar": attrs.get("calendar"), "count": int(np.size(raw_vals)),
           "raw_first": _jsonable(np.ravel(raw_vals)[0]) if np.size(raw_vals) else None,
           "raw_last": _jsonable(np.ravel(raw_vals)[-1]) if np.size(raw_vals) else None,
           "all_attrs": {k: _jsonable(v) for k, v in attrs.items()}}
    if not units or "since" not in str(units):
        rep["decoded"] = None
        rep["note"] = "time units missing or not CF 'X since ...'; labels cannot be decoded from the file"
        return rep
    dts = cftime.num2date(np.ravel(raw_vals), units, calendar=cal, only_use_cftime_datetimes=False)
    iso = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in dts]
    hours = sorted({(d.hour, d.minute) for d in dts})
    steps = np.diff(np.ravel(raw_vals).astype("float64"))
    rep.update(decoded_first=iso[0], decoded_last=iso[-1],
               time_of_day_values=[f"{h:02d}:{m:02d}" for h, m in hours],
               step_raw_unique=sorted({float(s) for s in steps})[:10] if steps.size else [],
               duplicates=int(len(iso) - len(set(iso))),
               years=sorted({d.year for d in dts}),
               has_feb29=any(d.month == 2 and d.day == 29 for d in dts),
               sample_first5=iso[:5], sample_last5=iso[-5:])
    if hours == [(0, 0)]:
        rep["time_of_day_meaning"] = ("all labels at 00:00 — date-only labels; the file does not say which "
                                      "24 h period each date covers")
    else:
        rep["time_of_day_meaning"] = ("non-midnight timestamps present; a timestamp can mark the start, end or "
                                      "centre of an accumulation — not evidence of a convention by itself")
    if "bounds" in attrs:
        rep["bounds_variable"] = attrs["bounds"]
    return rep


def inspect_netcdf(p: Path) -> dict:
    import netCDF4
    rep: dict = {}
    with netCDF4.Dataset(p, "r") as nc:
        nc.set_auto_maskandscale(False)
        scope = nc
        if not nc.variables and "Grid" in nc.groups:
            scope = nc.groups["Grid"]
            scope.set_auto_maskandscale(False)
            rep["inspected_group"] = "/Grid"
        rep["root_groups"] = list(nc.groups)
        rep["file_format_detail"] = nc.data_model
        rep["global_attributes"] = {k: _jsonable(nc.getncattr(k)) for k in nc.ncattrs()}
        rep["dimensions"] = {k: (len(d), "unlimited" if d.isunlimited() else "fixed") for k, d in scope.dimensions.items()}
        rep["variables"] = {n: {"dims": list(v.dimensions), "shape": list(v.shape), "dtype": str(v.dtype),
                                "attributes": {k: _jsonable(v.getncattr(k)) for k in v.ncattrs()}}
                            for n, v in scope.variables.items()}
        names = list(scope.variables)
        lat_n, lon_n, time_n = _pick(names, LAT_NAMES), _pick(names, LON_NAMES), _pick(names, TIME_NAMES)
        rain_n = _pick([n for n in names if scope.variables[n].ndim == 3], RAIN_NAMES)
        if rain_n is None:
            three = [n for n in names if scope.variables[n].ndim == 3]
            rain_n = three[0] if len(three) == 1 else None
        rep["identified"] = {"rain_variable": rain_n, "lat": lat_n, "lon": lon_n, "time": time_n}
        clat, clon = imd_canonical_coords()
        if lat_n:
            rep["lat"] = _axis_report(scope.variables[lat_n][:], clat, lat_n)
        if lon_n:
            rep["lon"] = _axis_report(scope.variables[lon_n][:], clon, lon_n)
            if rep["lon"]["max"] > 180:
                rep["lon"]["note"] = "0..360 longitudes"
        if time_n:
            tv = scope.variables[time_n]
            rep["time"] = _time_report(tv[:], {k: tv.getncattr(k) for k in tv.ncattrs()})
            if "bounds" in tv.ncattrs() and tv.getncattr("bounds") in scope.variables:
                b = scope.variables[tv.getncattr("bounds")][:]
                rep["time"]["bounds_first"] = _jsonable(b[0]); rep["time"]["bounds_last"] = _jsonable(b[-1])
        if rain_n:
            v = scope.variables[rain_n]
            attrs = {k: v.getncattr(k) for k in v.ncattrs()}
            raw = v[:]
            rep["rain"] = {"name": rain_n, "dims": list(v.dimensions), "shape": list(v.shape),
                           "dtype": str(v.dtype), "units": _jsonable(attrs.get("units")),
                           "scale_factor": _jsonable(attrs.get("scale_factor")),
                           "add_offset": _jsonable(attrs.get("add_offset")),
                           "attributes": {k: _jsonable(x) for k, x in attrs.items()},
                           **_value_report(raw, attrs)}
            if time_n in v.dimensions:
                rep["rain"]["cells_with_any_data"] = _cells_with_data(raw, attrs, list(v.dimensions).index(time_n))
            if "scale_factor" in attrs or "add_offset" in attrs:
                rep["rain"]["note"] = "raw (packed) values shown; apply scale_factor/add_offset for physical values"
        attr_sources = {"global": rep["global_attributes"]}
        for n in filter(None, [rain_n, time_n]):
            attr_sources[f"variable:{n}"] = rep["variables"][n]["attributes"]
        rep["convention_evidence_candidates"] = _scan_evidence(attr_sources)
    return rep


def inspect_grd(p: Path) -> dict:
    """IMD binary .grd has NO metadata. Only size-based facts are reported; layout claims are ASSUMED."""
    size = p.stat().st_size
    per_day = IMD_NLAT * IMD_NLON * 4
    rep = {"note": "raw binary: no embedded names, units, coordinates or time. Nothing about the time "
                   "convention can be established from this file."}
    if size % per_day:
        rep["layout"] = f"size {size} is not a multiple of {per_day} (129x135 float32); layout unknown"
        return rep
    n = size // per_day
    raw = np.fromfile(p, dtype="<f4").reshape(n, IMD_NLAT, IMD_NLON)
    rep["layout"] = {"ASSUMED": "little-endian float32, (day, 129 lat, 135 lon) as used by imdlib",
                     "n_records": int(n), "n_records_is_365_or_366": bool(n in (365, 366))}
    rep["rain"] = _value_report(raw, {"_FillValue": -999.0})
    rep["rain"]["cells_with_any_data"] = _cells_with_data(raw, {"_FillValue": -999.0}, 0)
    rep["convention_evidence_candidates"] = []
    return rep


def inspect(path: str | Path) -> dict:
    p = Path(path)
    before = _sha256(p)
    fmt = _format(p)
    rep = {
        "tool": "monsoonpp.tools.inspect_obs", "inspected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_filename": p.name, "source_path": str(p.resolve()), "size_bytes": p.stat().st_size,
        "sha256": before, "detected_format": fmt,
        "filename_policy": "filename recorded for provenance only; never used as evidence",
    }
    if fmt in ("netcdf3", "hdf5/netcdf4"):
        try:
            rep.update(inspect_netcdf(p))
        except OSError as e:
            rep["error"] = f"not readable as netCDF: {e}"
    elif fmt == "raw_binary" and p.suffix.lower() == ".grd":
        rep.update(inspect_grd(p))
    else:
        rep["error"] = f"unsupported format {fmt} for this inspector"
    rep["source_unchanged"] = _sha256(p) == before
    rep["time_convention_verdict"] = _verdict(rep)
    return rep


def _verdict(rep: dict) -> dict:
    hits = rep.get("convention_evidence_candidates", [])
    base = {"obs_time_convention": "UNVERIFIED",
            "rule": "This tool never changes the convention. A human must confirm explicit evidence "
                    "(file metadata or official documentation) and then edit the run config."}
    if hits:
        return {**base, "status": "CANDIDATE_TEXT_FOUND_REQUIRES_HUMAN_REVIEW",
                "reason": f"{len(hits)} attribute(s) contain time-window wording; read them verbatim."}
    return {**base, "status": "NO_EVIDENCE_IN_FILE",
            "reason": "No attribute describes the accumulation window; evidence must come from official documentation."}


def to_markdown(rep: dict) -> str:
    L = [f"# Observation file inspection: `{rep['source_filename']}`", "",
         f"- sha256 `{rep['sha256']}` · {rep['size_bytes']:,} bytes · format **{rep['detected_format']}** "
         f"({rep.get('file_format_detail', '-')}) · source unchanged: **{rep['source_unchanged']}**",
         f"- inspected {rep['inspected_utc']}", ""]
    if "error" in rep:
        L += [f"**Error:** {rep['error']}", ""]
    if "note" in rep:
        L += [f"> {rep['note']}", ""]
    if "identified" in rep:
        L += ["## Variables", "", f"identified: `{rep['identified']}`", "", "| variable | dims | shape | dtype | units |",
              "|---|---|---|---|---|"]
        for n, v in rep["variables"].items():
            L.append(f"| {n} | {', '.join(v['dims'])} | {v['shape']} | {v['dtype']} | {v['attributes'].get('units', '')} |")
        L.append("")
    for ax in ("lat", "lon"):
        if ax in rep:
            a = rep[ax]
            L.append(f"- **{ax}** `{a['name']}`: {a['count']} pts, {a['first']} → {a['last']}, {a['ordering']}, "
                     f"spacing {a['spacing']} (uniform {a['uniform_spacing']}), matches canonical IMD grid: "
                     f"**{a['matches_canonical_imd']}**{' — ' + a['note'] if 'note' in a else ''}")
    if "time" in rep:
        t = rep["time"]
        L += ["", "## Time", "", f"- units `{t.get('raw_units')}`, calendar `{t.get('calendar')}`, count {t['count']}",
              f"- first {t.get('decoded_first')}, last {t.get('decoded_last')}, times of day {t.get('time_of_day_values')}",
              f"- steps (raw) {t.get('step_raw_unique')}, duplicates {t.get('duplicates')}, Feb-29 present {t.get('has_feb29')}",
              f"- {t.get('time_of_day_meaning', t.get('note', ''))}"]
    if "layout" in rep:
        L += ["", f"- layout: `{rep['layout']}`"]
    if "rain" in rep:
        r = rep["rain"]
        L += ["", "## Values", ""] + [f"- {k}: `{r[k]}`" for k in
              ("name", "units", "scale_factor", "add_offset", "declared_fill_values", "fill_like_value_counts",
               "nan_count", "missing_fraction", "cells_with_any_data", "valid_min", "valid_max",
               "negative_valid_count", "wet_fraction", "wet_p50", "wet_p99") if k in r]
    if rep.get("global_attributes"):
        L += ["", "## Global attributes (verbatim)", ""] + [f"- **{k}**: {v}" for k, v in rep["global_attributes"].items()]
    L += ["", "## Time-convention evidence", ""]
    hits = rep.get("convention_evidence_candidates", [])
    L += [f"- `{h['location']}` / `{h['attribute']}`: “{h['text']}”" for h in hits] or ["- none found in file metadata"]
    v = rep["time_convention_verdict"]
    L += ["", f"**Verdict:** `{v['status']}` → obs_time_convention remains **{v['obs_time_convention']}**. {v['reason']}",
          "", f"_{v['rule']}_", ""]
    return "\n".join(L)


def run(path: str, out_dir: str | None = None) -> dict:
    rep = inspect(path)
    out = Path(out_dir or "data/inspections")
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).name
    (out / f"{stem}.inspection.json").write_text(
        json.dumps(rep, indent=1, default=_jsonable), encoding="utf-8"
    )
    (out / f"{stem}.inspection.md").write_text(to_markdown(rep), encoding="utf-8")
    rep["_written"] = [str(out / f"{stem}.inspection.json"), str(out / f"{stem}.inspection.md")]
    return rep
