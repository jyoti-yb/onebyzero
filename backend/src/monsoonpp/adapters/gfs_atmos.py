"""Strict decoding and validation of instantaneous GFS atmospheric fields.

The decoder reads GRIB2 metadata directly with ecCodes. Herbie inventory text and
filenames are used only to fetch a byte-range subset; they are never accepted as
evidence of variable identity, pressure level, units, or valid time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter


class GFSAtmosMetadataError(RuntimeError):
    pass


class GFSAtmosValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    short_name: str
    param: tuple[int, int, int]
    type_of_level: str
    level: int
    source_units: tuple[str, ...]
    plausible_source_range: tuple[float, float]
    output_units: str
    scale: float = 1.0


PRESSURE_LEVELS = (850, 700, 500, 200)
GFS_NI = 1440
GFS_NJ = 721
GFS_GRID_BOUNDS = (90.0, -90.0, 0.0, 359.75)
FIELD_SPECS: dict[str, FieldSpec] = {
    "mslp": FieldSpec("mslp", "prmsl", (0, 3, 1), "meanSea", 0, ("Pa",), (80_000, 110_000), "hPa", 0.01),
    "tcwv": FieldSpec("tcwv", "pwat", (0, 1, 3), "atmosphereSingleLayer", 0,
                      ("kg m**-2", "kg m-2"), (0, 100), "kg m-2"),
    "cape": FieldSpec("cape", "cape", (0, 7, 6), "surface", 0,
                      ("J kg**-1", "J kg-1"), (0, 20_000), "J kg-1"),
}
for _level in PRESSURE_LEVELS:
    FIELD_SPECS[f"u{_level}"] = FieldSpec(
        f"u{_level}", "u", (0, 2, 2), "isobaricInhPa", _level,
        ("m s**-1", "m s-1"), (-150, 150), "m s-1",
    )
    FIELD_SPECS[f"v{_level}"] = FieldSpec(
        f"v{_level}", "v", (0, 2, 3), "isobaricInhPa", _level,
        ("m s**-1", "m s-1"), (-150, 150), "m s-1",
    )
    FIELD_SPECS[f"rh{_level}"] = FieldSpec(
        f"rh{_level}", "r", (0, 1, 1), "isobaricInhPa", _level,
        ("%",), (0, 100), "%",
    )

_HGT_RANGES = {850: (-1_000, 3_000), 700: (1_000, 5_000), 500: (3_500, 7_000), 200: (9_000, 15_000)}
for _level in PRESSURE_LEVELS:
    FIELD_SPECS[f"hgt{_level}"] = FieldSpec(
        f"hgt{_level}", "gh", (0, 3, 5), "isobaricInhPa", _level,
        ("gpm",), _HGT_RANGES[_level], "gpm",
    )

HERBIE_SEARCH = (
    r":(?:PRMSL:mean sea level|PWAT:entire atmosphere .*|CAPE:surface|"
    r"(?:UGRD|VGRD|RH|HGT):(?:850|700|500|200) mb):"
)
REGIME_FIELDS = ("mslp", "tcwv", "cape", "u850", "v850")
REGIME_HERBIE_SEARCH = (
    r":(?:PRMSL:mean sea level|PWAT:entire atmosphere .*|CAPE:surface|"
    r"(?:UGRD|VGRD):850 mb):"
)

_TIME_UNIT_HOURS = {0: 1 / 60, 1: 1.0, 2: 24.0, 10: 3.0, 11: 6.0, 12: 12.0, 13: 1 / 3600}


def _need(ec, gid, key: str, where: str):
    try:
        return ec.codes_get(gid, key)
    except Exception as exc:
        raise GFSAtmosMetadataError(f"{where}: required GRIB key {key!r} missing ({exc})") from None


def _forecast_hours(value: int, unit_code: int, where: str) -> int:
    if unit_code not in _TIME_UNIT_HOURS:
        raise GFSAtmosMetadataError(f"{where}: unsupported forecast-time unit {unit_code}")
    hours = value * _TIME_UNIT_HOURS[unit_code]
    if abs(hours - round(hours)) > 1e-9:
        raise GFSAtmosMetadataError(f"{where}: forecast time is not a whole number of hours")
    return int(round(hours))


def _message_name(short_name: str, type_of_level: str, level: int) -> str | None:
    for name, spec in FIELD_SPECS.items():
        if (short_name, type_of_level, level) == (spec.short_name, spec.type_of_level, spec.level):
            return name
    return None


def decode_gfs_atmos_file(path: str | Path, init, fxx: int, bbox=None,
                          required_fields=None) -> tuple[xr.Dataset, dict]:
    """Decode one GFS file with strict metadata checks for every requested field.

    The default remains the complete Phase B field set. Phase C2 may explicitly request
    its five consumed fields without weakening any check applied to those messages.
    """
    import eccodes as ec

    path = Path(path)
    expected_init = pd.Timestamp(init)
    expected_valid = expected_init + pd.Timedelta(hours=int(fxx))
    required = tuple(FIELD_SPECS) if required_fields is None else tuple(required_fields)
    unknown = sorted(set(required) - set(FIELD_SPECS))
    if unknown:
        raise ValueError(f"unknown required atmospheric fields: {unknown}")
    fields: dict[str, tuple[np.ndarray, FieldSpec, dict]] = {}
    ref_lat = ref_lon = None
    with path.open("rb") as stream:
        message_index = 0
        while True:
            gid = ec.codes_grib_new_from_file(stream)
            if gid is None:
                break
            message_index += 1
            where = f"{path}#{message_index}"
            try:
                short_name = str(_need(ec, gid, "shortName", where))
                type_of_level = str(_need(ec, gid, "typeOfLevel", where))
                level = int(_need(ec, gid, "level", where))
                name = _message_name(short_name, type_of_level, level)
                if name is None or name not in required:
                    continue
                if name in fields:
                    raise GFSAtmosMetadataError(f"{where}: duplicate {name} message")
                spec = FIELD_SPECS[name]
                param = (
                    int(_need(ec, gid, "discipline", where)),
                    int(_need(ec, gid, "parameterCategory", where)),
                    int(_need(ec, gid, "parameterNumber", where)),
                )
                if param != spec.param:
                    raise GFSAtmosMetadataError(f"{where}: {name} parameter {param} != {spec.param}")
                units = str(_need(ec, gid, "units", where))
                if units not in spec.source_units:
                    raise GFSAtmosMetadataError(f"{where}: {name} units {units!r} not in {spec.source_units}")
                if _need(ec, gid, "stepType", where) != "instant":
                    raise GFSAtmosMetadataError(f"{where}: {name} is not instantaneous")
                if int(_need(ec, gid, "significanceOfReferenceTime", where)) != 1:
                    raise GFSAtmosMetadataError(f"{where}: reference time is not forecast initialization")
                message_fxx = _forecast_hours(
                    int(_need(ec, gid, "forecastTime", where)),
                    int(_need(ec, gid, "indicatorOfUnitOfTimeRange", where)),
                    where,
                )
                if message_fxx != int(fxx):
                    raise GFSAtmosMetadataError(f"{where}: forecast hour {message_fxx} != expected {fxx}")
                step_range = str(_need(ec, gid, "stepRange", where)).rstrip("h")
                if step_range != str(int(fxx)):
                    raise GFSAtmosMetadataError(f"{where}: stepRange {step_range!r} != {fxx}")

                date, clock = int(_need(ec, gid, "dataDate", where)), int(_need(ec, gid, "dataTime", where))
                actual_init = pd.Timestamp(f"{date:08d}") + pd.Timedelta(hours=clock // 100, minutes=clock % 100)
                vdate, vclock = int(_need(ec, gid, "validityDate", where)), int(_need(ec, gid, "validityTime", where))
                actual_valid = pd.Timestamp(f"{vdate:08d}") + pd.Timedelta(hours=vclock // 100, minutes=vclock % 100)
                if actual_init != expected_init or actual_valid != expected_valid:
                    raise GFSAtmosMetadataError(
                        f"{where}: init/valid {actual_init}/{actual_valid} != {expected_init}/{expected_valid}"
                    )

                if _need(ec, gid, "gridType", where) != "regular_ll":
                    raise GFSAtmosMetadataError(f"{where}: grid is not regular_ll")
                ni, nj = int(_need(ec, gid, "Ni", where)), int(_need(ec, gid, "Nj", where))
                if (ni, nj) != (GFS_NI, GFS_NJ):
                    raise GFSAtmosMetadataError(
                        f"{where}: grid {ni}x{nj} != GFS 0.25-degree {GFS_NI}x{GFS_NJ}"
                    )
                if (int(_need(ec, gid, "iScansNegatively", where))
                        or int(_need(ec, gid, "jScansPositively", where))
                        or int(_need(ec, gid, "jPointsAreConsecutive", where))):
                    raise GFSAtmosMetadataError(f"{where}: unsupported scanning mode")
                increments = (
                    float(_need(ec, gid, "iDirectionIncrementInDegrees", where)),
                    float(_need(ec, gid, "jDirectionIncrementInDegrees", where)),
                )
                if not np.allclose(increments, (0.25, 0.25), rtol=0, atol=1e-9):
                    raise GFSAtmosMetadataError(f"{where}: grid increments {increments} != (0.25, 0.25) degrees")
                endpoints = (
                    float(_need(ec, gid, "latitudeOfFirstGridPointInDegrees", where)),
                    float(_need(ec, gid, "latitudeOfLastGridPointInDegrees", where)),
                    float(_need(ec, gid, "longitudeOfFirstGridPointInDegrees", where)),
                    float(_need(ec, gid, "longitudeOfLastGridPointInDegrees", where)),
                )
                if not np.allclose(endpoints, GFS_GRID_BOUNDS, rtol=0, atol=1e-9):
                    raise GFSAtmosMetadataError(
                        f"{where}: global grid endpoints {endpoints} != {GFS_GRID_BOUNDS}"
                    )
                lat = np.linspace(
                    endpoints[0],
                    endpoints[1],
                    nj,
                )
                lon = np.linspace(
                    endpoints[2],
                    endpoints[3],
                    ni,
                )
                values = np.asarray(ec.codes_get_values(gid), dtype="float64")
                if values.size != ni * nj:
                    raise GFSAtmosMetadataError(f"{where}: {values.size} values for {nj}x{ni} grid")
                values = values.reshape(nj, ni)
                if int(_need(ec, gid, "bitmapPresent", where)):
                    missing_value = float(_need(ec, gid, "missingValue", where))
                    if np.any(values == missing_value):
                        raise GFSAtmosValidationError(f"{where}: {name} contains bitmap missing values")
                if not np.isfinite(values).all():
                    raise GFSAtmosValidationError(f"{where}: {name} contains non-finite values")
                observed = (float(values.min()), float(values.max()))
                lo, hi = spec.plausible_source_range
                if observed[0] < lo or observed[1] > hi:
                    raise GFSAtmosValidationError(
                        f"{where}: {name} range {observed} outside plausible source range {(lo, hi)}"
                    )
                if lat[0] > lat[-1]:
                    lat, values = lat[::-1], values[::-1]
                if ref_lat is None:
                    ref_lat, ref_lon = lat, lon
                elif not (np.array_equal(lat, ref_lat) and np.array_equal(lon, ref_lon)):
                    raise GFSAtmosMetadataError(f"{where}: {name} grid differs from earlier messages")
                fields[name] = (
                    values * spec.scale,
                    spec,
                    {
                        "message_index": message_index,
                        "short_name": short_name,
                        "param": list(param),
                        "type_of_level": type_of_level,
                        "level": level,
                        "source_units": units,
                        "output_units": spec.output_units,
                        "source_range": list(observed),
                        "step_type": "instant",
                        "forecast_hour": int(fxx),
                        "init_time": str(actual_init),
                        "valid_time": str(actual_valid),
                    },
                )
            finally:
                ec.codes_release(gid)

    missing = sorted(set(required) - set(fields))
    if missing:
        raise GFSAtmosMetadataError(f"{path}: missing required atmospheric messages {missing}")

    if bbox is None:
        iy, ix = np.arange(len(ref_lat)), np.arange(len(ref_lon))
    else:
        lat_min, lat_max, lon_min, lon_max = bbox
        iy = np.nonzero((ref_lat >= lat_min) & (ref_lat <= lat_max))[0]
        lon180 = ((ref_lon + 180) % 360) - 180
        ix = np.nonzero((lon180 >= lon_min) & (lon180 <= lon_max))[0]
        ix = ix[np.argsort(lon180[ix])]
        if not iy.size or not ix.size:
            raise GFSAtmosValidationError(f"{path}: bbox {bbox} selects no grid cells")
        ref_lon = lon180

    data_vars = {}
    qa_fields = {}
    for name, (values, spec, facts) in fields.items():
        da = xr.DataArray(
            values[np.ix_(iy, ix)].astype("float32"),
            dims=("lat", "lon"),
            coords={"lat": ref_lat[iy], "lon": ref_lon[ix]},
            attrs={
                "units": spec.output_units,
                "source_units": facts["source_units"],
                "source_short_name": spec.short_name,
                "source_parameter": str(spec.param),
                "source_level": f"{spec.type_of_level}:{spec.level}",
                "aggregation": "instantaneous source field",
            },
        )
        data_vars[name] = da
        qa_fields[name] = facts
    ds = xr.Dataset(data_vars)
    ds.attrs.update(
        source="NOAA GFS",
        product="pgrb2.0p25",
        init_time=str(expected_init),
        valid_time=str(expected_valid),
        forecast_hour=int(fxx),
        source_file=str(path),
        validation="direct GRIB2 metadata and values via ecCodes",
    )
    return ds, {
        "status": "PASS",
        "source_file": str(path),
        "init_time": str(expected_init),
        "valid_time": str(expected_valid),
        "forecast_hour": int(fxx),
        "grid": {"type": "regular_ll", "Ni": GFS_NI, "Nj": GFS_NJ, "resolution_degrees": 0.25,
                 "global_endpoints": list(GFS_GRID_BOUNDS), "scanning_mode": "eastward, north-to-south"},
        "missing_values": 0,
        "required_fields": list(required),
        "fields": qa_fields,
    }


def derive_atmospheric_fields(ds: xr.Dataset, background_sigma_cells: float = 8.0) -> xr.Dataset:
    """The five Phase B atmospheric derivatives; no additional fields are invented."""
    required = ("u850", "v850", "tcwv", "mslp")
    missing = [name for name in required if name not in ds]
    if missing:
        raise GFSAtmosValidationError(f"cannot derive atmospheric fields; missing {missing}")
    lat = np.asarray(ds.lat.values, dtype="float64")
    res = float(np.median(np.diff(lat)))
    dy = res * 111_000.0
    dx = dy * np.cos(np.deg2rad(lat))[:, None]
    u = ds.u850.values.astype("float64")
    v = ds.v850.values.astype("float64")
    tcwv = ds.tcwv.values.astype("float64")
    mslp = ds.mslp.values.astype("float64")
    du_dy = np.gradient(u, axis=-2) / dy
    dv_dx = np.gradient(v, axis=-1) / dx
    sigma = (0.0,) * (u.ndim - 2) + (float(background_sigma_cells), float(background_sigma_cells))
    dims = ds.u850.dims
    coords = ds.u850.coords
    out = xr.Dataset(
        {
            "wspd850": (dims, np.hypot(u, v).astype("float32")),
            "vo850": (dims, ((dv_dx - du_dy) * 1e5).astype("float32")),
            "moisture_transport": (dims, (np.hypot(u, v) * tcwv).astype("float32")),
            "pwat_anom": (dims, (tcwv - gaussian_filter(tcwv, sigma=sigma, mode="nearest")).astype("float32")),
            "mslp_anom": (dims, (mslp - gaussian_filter(mslp, sigma=sigma, mode="nearest")).astype("float32")),
        },
        coords=coords,
    )
    attrs = {
        "wspd850": ("m s-1", "hypot(u850, v850)"),
        "vo850": ("1e-5 s-1", "dv850/dx - du850/dy on the regular lat-lon grid"),
        "moisture_transport": ("kg m-1 s-1", "wspd850 * PWAT; magnitude proxy, not vertically integrated flux vector"),
        "pwat_anom": ("kg m-2", f"PWAT minus Gaussian background (sigma={background_sigma_cells:g} grid cells)"),
        "mslp_anom": ("hPa", f"MSLP minus Gaussian background (sigma={background_sigma_cells:g} grid cells)"),
    }
    for name, (units, derivation) in attrs.items():
        out[name].attrs.update(units=units, derivation=derivation)
    out.attrs["derived_field_policy"] = "only wspd850, vo850, moisture_transport, pwat_anom, mslp_anom"
    return out
