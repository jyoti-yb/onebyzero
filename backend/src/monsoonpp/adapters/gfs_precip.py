"""GFS daily (03Z -> next 03Z) rainfall from APCP, with GRIB-metadata proof of every interval.

PRIMARY (00Z run, D+1):   APCP(0-27) - APCP(0-3)                      cell by cell, native grid
QA      (independent):     (APCP(0-6) - APCP(0-3)) + APCP(6-12) + APCP(12-18)
                           + APCP(18-24) + APCP(24-27)
kg m**-2 is taken as mm of liquid-water equivalent (density of water = 1000 kg m**-3).

Rules enforced here:
* Accumulation intervals come ONLY from GRIB2 metadata (product definition template 4.8),
  never from file names, forecast-hour labels or .idx inventory text. `fxx` merely says
  which file to open; the message inside must prove its own interval.
* Each APCP message is checked for: parameter identity (discipline 0 / category 1 /
  number 8, i.e. NCEP APCP; ecCodes shortName 'tp'), stepType == 'accum',
  typeOfStatisticalProcessing == 1, a single time range, reference time = forecast start,
  initialization time, start/end hours from the raw PDT keys AND from ecCodes' stepRange
  (must agree), end-of-interval time AND validity time (must equal init + end), units,
  level (surface) and a regular lat/lon grid without missing values.
* Missing or ambiguous (duplicate, disagreeing) messages raise; nothing is approximated.
* Checks on the result: no meaningful negatives, cumulative monotonicity of all running
  totals seen, window exactly 03Z -> next-day 03Z, and primary-vs-QA difference statistics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from ..data.align import Window, gfs_apcp_primary, gfs_apcp_qa_pieces

log = logging.getLogger(__name__)

APCP_PARAM = (0, 1, 8)                      # discipline, parameterCategory, parameterNumber
ACCEPTED_SHORT_NAMES = {"tp", "APCP"}       # ecCodes names NCEP APCP 'tp'
ACCEPTED_SOURCE_UNITS = {"kg m**-2", "kg m-2", "kg m^-2", "kg/m^2", "kg/m2"}
OUTPUT_UNITS = "mm"
# GRIB2 code table 4.4 (indicator of unit of time range) -> hours
UNIT_HOURS = {0: 1 / 60, 1: 1.0, 2: 24.0, 10: 3.0, 11: 6.0, 12: 12.0, 13: 1 / 3600}
PRIMARY_METHOD = "APCP(0-{e}) - APCP(0-{s}); cell by cell on the native GFS grid"
QA_METHOD = ("(APCP({a0}-{b0}) - APCP({a1}-{b1})) + APCP({a2}-{b2}) + APCP({a3}-{b3}) "
             "+ APCP({a4}-{b4}) + APCP({a5}-{b5})")


class APCPMetadataError(RuntimeError):
    """A message claiming to be APCP has missing/inconsistent/unsupported metadata."""


class APCPIntervalError(APCPMetadataError):
    """The required accumulation interval is missing or ambiguous."""


class APCPValidationError(RuntimeError):
    """The constructed daily rainfall failed a physical/QA check."""


@dataclass(frozen=True)
class Tolerances:
    negative_mm: float = 0.25       # primary < -this  -> meaningful negative -> fail
    monotonic_mm: float = 0.25      # running total may drop by at most this (packing noise)
    qa_max_abs_mm: float = 0.5      # every cell: |primary - QA| < this
    qa_mean_abs_mm: float = 0.1     # domain mean |primary - QA| <= this
    qa_frac_below: float = 1.0      # fraction of cells with |diff| < qa_max_abs_mm required


@dataclass
class APCPMessage:
    path: str
    msg_index: int
    short_name: str
    param: tuple
    step_type: str
    start_h: int
    end_h: int
    step_range: str
    init: pd.Timestamp
    valid_end: pd.Timestamp
    units: str
    level: str
    lat: np.ndarray          # ascending
    lon: np.ndarray
    values: np.ndarray       # (lat, lon) float64, lat ascending
    packing_quantum: float | None = None

    @property
    def valid_start(self) -> pd.Timestamp:
        return self.init + pd.Timedelta(hours=self.start_h)

    @property
    def interval(self) -> str:
        return f"{self.start_h}-{self.end_h}"

    @property
    def grid_key(self) -> tuple:
        return (self.lat.size, self.lon.size, round(float(self.lat[0]), 6), round(float(self.lat[-1]), 6),
                round(float(self.lon[0]), 6), round(float(self.lon[-1]), 6))


# ----------------------------------------------------------------------------- reading
def _hours(value: int, unit_code: int, what: str) -> int:
    if unit_code not in UNIT_HOURS:
        raise APCPMetadataError(f"{what}: unsupported time-unit code {unit_code} (GRIB2 table 4.4)")
    h = value * UNIT_HOURS[unit_code]
    if abs(h - round(h)) > 1e-9:
        raise APCPMetadataError(f"{what}: {value} x unit {unit_code} is not a whole number of hours")
    return int(round(h))


def _parse_step_range(sr: str, step_units: int) -> tuple[int, int]:
    s = str(sr).strip().rstrip("h")
    if "-" not in s:
        raise APCPMetadataError(f"stepRange '{sr}' is not an accumulation range")
    a, b = s.split("-", 1)
    return _hours(int(a), step_units, "stepRange start"), _hours(int(b), step_units, "stepRange end")


def verify_units(units: str) -> str:
    if units not in ACCEPTED_SOURCE_UNITS:
        raise APCPMetadataError(f"APCP units '{units}' not in {sorted(ACCEPTED_SOURCE_UNITS)}")
    return units


def _decode(gid, path: str, idx: int) -> APCPMessage:
    import eccodes as ec
    g = lambda k: ec.codes_get(gid, k)

    def need(k):
        try:
            return ec.codes_get(gid, k)
        except Exception as e:  # key absent -> cannot prove the interval
            raise APCPMetadataError(f"{path}#{idx}: required GRIB key '{k}' missing ({e})") from None

    where = f"{path}#{idx}"
    short = need("shortName")
    if short not in ACCEPTED_SHORT_NAMES:
        raise APCPMetadataError(f"{where}: param {APCP_PARAM} but shortName '{short}'")
    if need("productDefinitionTemplateNumber") != 8:
        raise APCPMetadataError(f"{where}: PDT {g('productDefinitionTemplateNumber')} != 8 (deterministic accumulation)")
    step_type = need("stepType")
    if step_type != "accum":
        raise APCPMetadataError(f"{where}: stepType '{step_type}' != 'accum'")
    if need("typeOfStatisticalProcessing") != 1:
        raise APCPMetadataError(f"{where}: typeOfStatisticalProcessing {g('typeOfStatisticalProcessing')} != 1 (accumulation)")
    if need("numberOfTimeRange") != 1:
        raise APCPMetadataError(f"{where}: numberOfTimeRange {g('numberOfTimeRange')} != 1")
    if need("significanceOfReferenceTime") != 1:
        raise APCPMetadataError(f"{where}: reference time is not the start of forecast")

    # interval from the raw product definition
    start_h = _hours(need("forecastTime"), need("indicatorOfUnitOfTimeRange"), f"{where} forecastTime")
    length_h = _hours(need("lengthOfTimeRange"), need("indicatorOfUnitForTimeRange"), f"{where} lengthOfTimeRange")
    end_h = start_h + length_h
    if length_h <= 0:
        raise APCPMetadataError(f"{where}: non-positive accumulation length {length_h} h")
    # ... must agree with ecCodes' derived stepRange
    sr = need("stepRange")
    if _parse_step_range(sr, need("stepUnits")) != (start_h, end_h):
        raise APCPMetadataError(f"{where}: stepRange '{sr}' disagrees with PDT interval {start_h}-{end_h}")

    dd, dt = need("dataDate"), need("dataTime")
    init = pd.Timestamp(f"{dd:08d}") + pd.Timedelta(hours=dt // 100, minutes=dt % 100)
    end_iv = pd.Timestamp(year=need("yearOfEndOfOverallTimeInterval"), month=need("monthOfEndOfOverallTimeInterval"),
                          day=need("dayOfEndOfOverallTimeInterval"), hour=need("hourOfEndOfOverallTimeInterval"),
                          minute=need("minuteOfEndOfOverallTimeInterval"), second=need("secondOfEndOfOverallTimeInterval"))
    vd, vt = need("validityDate"), need("validityTime")
    validity = pd.Timestamp(f"{vd:08d}") + pd.Timedelta(hours=vt // 100, minutes=vt % 100)
    expected_end = init + pd.Timedelta(hours=end_h)
    if end_iv != expected_end or validity != expected_end:
        raise APCPMetadataError(f"{where}: end-of-interval {end_iv} / validity {validity} != init {init} + {end_h} h")

    units = verify_units(need("units"))
    level = need("typeOfLevel")
    if level != "surface":
        raise APCPMetadataError(f"{where}: typeOfLevel '{level}' != 'surface'")

    if need("gridType") != "regular_ll":
        raise APCPMetadataError(f"{where}: gridType '{g('gridType')}' unsupported (need regular_ll)")
    ni, nj = need("Ni"), need("Nj")
    lat0, lat1 = need("latitudeOfFirstGridPointInDegrees"), need("latitudeOfLastGridPointInDegrees")
    lon0, lon1 = need("longitudeOfFirstGridPointInDegrees"), need("longitudeOfLastGridPointInDegrees")
    lat = np.linspace(lat0, lat1, nj)
    lon = np.linspace(lon0, lon1, ni)
    if need("iScansNegatively") != 0 or need("jPointsAreConsecutive") != 0:
        raise APCPMetadataError(f"{where}: unsupported scanning mode")
    vals = np.asarray(ec.codes_get_values(gid), dtype="float64")
    if vals.size != ni * nj:
        raise APCPMetadataError(f"{where}: {vals.size} values for {nj}x{ni} grid")
    if ec.codes_get(gid, "bitmapPresent"):
        miss = vals == ec.codes_get(gid, "missingValue")
        if miss.any():
            raise APCPMetadataError(f"{where}: {int(miss.sum())} missing grid values in APCP")
    vals = vals.reshape(nj, ni)
    if lat[0] > lat[-1]:
        lat, vals = lat[::-1], vals[::-1, :]
    quantum = None
    try:
        quantum = float(2.0 ** g("binaryScaleFactor") * 10.0 ** (-g("decimalScaleFactor")))
    except Exception:
        pass
    return APCPMessage(path, idx, short, APCP_PARAM, step_type, start_h, end_h, str(sr), init, expected_end,
                       units, level, np.round(lat, 6), np.round(lon, 6), vals, quantum)


def read_apcp_messages(path: str | Path) -> list[APCPMessage]:
    """Every APCP message (identified by GRIB2 parameter 0/1/8) in a file, fully verified.
    Non-APCP messages are skipped; APCP messages with bad metadata raise."""
    import eccodes as ec
    out = []
    with open(path, "rb") as f:
        idx = 0
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            idx += 1
            try:
                if ec.codes_get(gid, "edition") != 2:
                    continue
                param = (ec.codes_get(gid, "discipline"), ec.codes_get(gid, "parameterCategory"),
                         ec.codes_get(gid, "parameterNumber"))
                if param != APCP_PARAM:
                    continue
                out.append(_decode(gid, str(path), idx))
            finally:
                ec.codes_release(gid)
    return out


def select_interval(msgs: list[APCPMessage], start_h: int, end_h: int, init: pd.Timestamp, where: str) -> APCPMessage:
    hits = [m for m in msgs if (m.start_h, m.end_h) == (start_h, end_h)]
    if not hits:
        have = sorted({m.interval for m in msgs})
        raise APCPIntervalError(f"{where}: no APCP {start_h}-{end_h} h message (file has {have or 'no APCP'})")
    if len(hits) > 1 and not all(np.array_equal(h.values, hits[0].values) for h in hits[1:]):
        raise APCPIntervalError(f"{where}: {len(hits)} disagreeing APCP {start_h}-{end_h} h messages (ambiguous)")
    m = hits[0]
    if m.init != init:
        raise APCPMetadataError(f"{where}: message init {m.init} != expected init {init}")
    return m


# ------------------------------------------------------------------------------ builder
@dataclass
class DailyPrecipResult:
    field: xr.DataArray          # native grid, lat ascending, mm, clipped >= 0 after checks
    qa: dict
    provenance: dict = field(default_factory=dict)


class GFSDailyPrecip:
    """fetch(init, fxx) -> path of a GRIB2 file for that run/forecast hour containing its APCP messages."""

    def __init__(self, fetch: Callable[[pd.Timestamp, int], str | Path], tol: Tolerances | None = None,
                 model: str = "GFS", product: str = "pgrb2.0p25", run_qa: bool = True):
        self.fetch, self.tol, self.model, self.product, self.run_qa = fetch, tol or Tolerances(), model, product, run_qa
        self._cache: dict = {}

    def _msgs(self, init, fxx) -> list[APCPMessage]:
        key = (init, fxx)
        if key not in self._cache:
            self._cache[key] = read_apcp_messages(self.fetch(init, fxx))
        return self._cache[key]

    def _get(self, init, fxx, a, b) -> APCPMessage:
        return select_interval(self._msgs(init, fxx), a, b, init, f"{self.model} {init:%Y%m%d%H}Z f{fxx:03d}")

    @staticmethod
    def check_window(w: Window, valid_start: pd.Timestamp, valid_end: pd.Timestamp):
        if w.init.hour != 0 or w.init.minute != 0:
            raise APCPValidationError(f"cycle {w.init:%H}Z (minute {w.init.minute}) unsupported: construction validated for 00Z runs only")
        if w.f_end - w.f_start != 24:
            raise APCPValidationError(f"window f{w.f_start:03d}-f{w.f_end:03d} is not 24 h")
        ok = (valid_start.hour == 3 and valid_start.minute == 0 and valid_start.second == 0
              and valid_end - valid_start == pd.Timedelta(hours=24)
              and valid_end.normalize() == valid_start.normalize() + pd.Timedelta(days=1))
        if not ok:
            raise APCPValidationError(f"window {valid_start} -> {valid_end} is not 03Z -> next-day 03Z")

    def build(self, w: Window, lead: int) -> DailyPrecipResult:
        tol = self.tol
        (fe, _, e, _), (fs, _, s, _) = gfs_apcp_primary(w)
        tot_e = self._get(w.init, fe, 0, e)
        tot_s = self._get(w.init, fs, 0, s)
        valid_start, valid_end = tot_s.valid_end, tot_e.valid_end     # from GRIB, not from fxx
        self.check_window(w, valid_start, valid_end)
        if tot_e.grid_key != tot_s.grid_key:
            raise APCPMetadataError("primary messages are on different grids; cell-by-cell difference impossible")

        primary = tot_e.values - tot_s.values                        # cell by cell, native grid
        qa: dict = {"tolerances": tol.__dict__}

        # cumulative monotonicity over every 0-N running total available in the files touched
        totals = {tot_s.end_h: tot_s, tot_e.end_h: tot_e}
        files = {fs, fe} | ({p[0] for p in gfs_apcp_qa_pieces(w)} if self.run_qa else set())
        for fxx in sorted(files):
            for m in self._msgs(w.init, fxx):
                if m.start_h == 0 and m.grid_key == tot_e.grid_key:
                    totals.setdefault(m.end_h, m)
        chain = [totals[k] for k in sorted(totals)]
        viol = np.zeros(primary.shape, bool)
        for a, b in zip(chain[:-1], chain[1:]):
            viol |= (b.values - a.values) < -tol.monotonic_mm
        qa["monotonic_chain"] = [m.interval for m in chain]
        qa["monotonic_violation_frac"] = float(viol.mean())
        if viol.any():
            raise APCPValidationError(f"cumulative APCP decreases by > {tol.monotonic_mm} mm in {int(viol.sum())} cells "
                                      f"(chain {qa['monotonic_chain']})")

        neg = primary < -tol.negative_mm
        qa["meaningful_negative_cells"] = int(neg.sum())
        qa["small_negative_cells_clipped"] = int(((primary < 0) & ~neg).sum())
        qa["min_raw_mm"] = float(primary.min())
        if neg.any():
            raise APCPValidationError(f"{int(neg.sum())} cells with daily rain < -{tol.negative_mm} mm")

        qa_sources = []
        if self.run_qa:
            pieces = gfs_apcp_qa_pieces(w)
            qa_field = np.zeros_like(primary)
            for fxx, a, b, sign in pieces:
                m = self._get(w.init, fxx, a, b)
                if m.grid_key != tot_e.grid_key:
                    raise APCPMetadataError(f"QA piece {m.interval} on a different grid")
                qa_field += sign * m.values
                qa_sources.append(("+" if sign > 0 else "-") + m.interval)
            d = np.abs(primary - qa_field)
            qa.update(qa_method=QA_METHOD.format(**{f"{k}{i}": v for i, (_, a, b, _) in enumerate(pieces)
                                                     for k, v in (("a", a), ("b", b))}),
                      qa_sources=qa_sources, qa_mean_abs_mm=float(d.mean()), qa_median_abs_mm=float(np.median(d)),
                      qa_p95_abs_mm=float(np.percentile(d, 95)), qa_max_abs_mm=float(d.max()),
                      qa_frac_below_max=float((d < tol.qa_max_abs_mm).mean()))
            if qa["qa_frac_below_max"] < tol.qa_frac_below or qa["qa_mean_abs_mm"] > tol.qa_mean_abs_mm:
                stats = {k: round(qa[k], 4) for k in ("qa_mean_abs_mm", "qa_p95_abs_mm", "qa_max_abs_mm", "qa_frac_below_max")}
                raise APCPValidationError(f"primary vs QA disagree beyond tolerance {tol}: {stats}")

        out = np.clip(primary, 0.0, None)
        prov = {
            "model": self.model, "product": self.product, "cycle": f"{w.init:%H}Z",
            "init_time": str(w.init), "forecast_lead_days": int(lead),
            "forecast_hours": f"f{s:03d}-f{e:03d}",
            "valid_start": str(valid_start), "valid_end": str(valid_end),
            "source_apcp_step_ranges": f"+{tot_e.interval} -{tot_s.interval}",
            "source_apcp_grib_stepRange": f"{tot_e.step_range}|{tot_s.step_range}",
            "source_files": f"{Path(tot_e.path).name}#{tot_e.msg_index}|{Path(tot_s.path).name}#{tot_s.msg_index}",
            "source_units": tot_e.units, "output_units": OUTPUT_UNITS,
            "unit_conversion": "kg m**-2 == mm liquid-water equivalent (x1)",
            "construction_method": PRIMARY_METHOD.format(e=e, s=s),
            "qa_method": qa.get("qa_method", "not run"),
            "packing_quantum_mm": tot_e.packing_quantum,
        }
        da = xr.DataArray(out, dims=("lat", "lon"), coords={"lat": tot_e.lat, "lon": tot_e.lon}, name="tp",
                          attrs={**{k: v for k, v in prov.items() if v is not None}, "units": OUTPUT_UNITS})
        log.info("GFS %s lead %d: %s -> %s  QA max|d|=%.4f mm", w.init, lead, valid_start, valid_end,
                 qa.get("qa_max_abs_mm", float("nan")))
        return DailyPrecipResult(da, qa, prov)
