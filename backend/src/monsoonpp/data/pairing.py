"""Forecast/observation pairing by EXACT accumulation window, never by equal date labels.

Every daily field carries (valid_start, valid_end):
  forecast  : from GRIB provenance (GFS tp_valid_start/end) or the forecast label rule
  obs, label-based products (IMD, IMD-NCMRWF): from the VERIFIED obs_time_convention
              (data.obs_time.interpret_obs_times). UNVERIFIED -> NaT -> pairing refused.
  obs, interval products (IMERG daily built here): from explicit source intervals.
A pair exists only when both ends of the window are identical. Under ENDING_03Z the
observation label D+1 pairs with forecast label D; under STARTING_03Z label D pairs with D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .align import forecast_label_window
from .obs_time import ObsTimeConvention, ObsTimeConventionError, label_window, parse_convention

DAY = pd.Timedelta(days=1)


class PairingError(RuntimeError):
    pass


def _ns(x) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(np.ravel(x)).values, dtype="datetime64[ns]")


def forecast_windows(fc: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Windows for each forecast label (lead-independent by the label rule). If the adapter
    recorded GRIB-derived windows (GFS tp_valid_start/end), they must agree exactly."""
    labels = pd.DatetimeIndex(fc.time.values)
    win = [forecast_label_window(t) for t in labels]
    starts, ends = _ns([w[0] for w in win]), _ns([w[1] for w in win])
    for c, ref in (("tp_valid_start", starts), ("tp_valid_end", ends)):
        if c in fc.coords:
            rec = np.atleast_2d(fc[c].values)
            for li in range(rec.shape[0]):
                for ti, v in enumerate(rec[li]):
                    if v not in ("", None) and not pd.isna(v) and pd.Timestamp(str(v)) != pd.Timestamp(ref[ti]):
                        raise PairingError(f"forecast {c}={v} disagrees with label-rule window {pd.Timestamp(ref[ti])}")
    return starts, ends


def obs_windows(obs: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "valid_start" not in obs.coords or "valid_end" not in obs.coords:
        raise PairingError("observations carry no valid_start/valid_end")
    s, e = _ns(obs.valid_start.values), _ns(obs.valid_end.values)
    if np.isnat(s).any() or np.isnat(e).any():
        raise ObsTimeConventionError(
            f"BLOCKED: observation windows are unknown (time_convention={obs.attrs.get('time_convention')}); "
            "a verified convention or explicit source intervals are required for pairing")
    return s, e


def obs_labels_for_windows(starts, ends, convention) -> pd.DatetimeIndex:
    """Which observation DATE LABELS cover the given windows under a VERIFIED convention.
    Candidates are the start date and the next date; the one whose convention window equals
    the requested window exactly is chosen, otherwise it is an error."""
    conv = parse_convention(convention)
    if conv is ObsTimeConvention.UNVERIFIED:
        label_window("2000-01-01", conv)          # raises the frozen UNVERIFIED error
    out = []
    for s, e in zip(_ns(starts), _ns(ends)):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        hits = [d for d in (s.normalize(), s.normalize() + DAY) if label_window(d, conv) == (s, e)]
        if len(hits) != 1:
            raise PairingError(f"window {s} -> {e} is not representable by one {conv.value} date label")
        out.append(hits[0])
    return pd.DatetimeIndex(out)


def match_windows(t_start, t_end, c_start, c_end) -> np.ndarray:
    """Index into candidates for each target window (-1 if none). Duplicates are an error."""
    key = {}
    for i, (s, e) in enumerate(zip(_ns(c_start), _ns(c_end))):
        k = (s, e)
        if k in key:
            raise PairingError(f"two observation records cover the identical window {pd.Timestamp(s)} -> {pd.Timestamp(e)}")
        key[k] = i
    return np.array([key.get((s, e), -1) for s, e in zip(_ns(t_start), _ns(t_end))], dtype=int)


def align_to_windows(src: xr.Dataset, src_start, src_end, t_start, t_end, target_time) -> tuple[xr.Dataset, np.ndarray]:
    """Re-express `src` (dim time) on the target windows. Unmatched targets become NaN / empty.
    Per-time coords of src are carried along (so the observation's own date label survives)."""
    idx = match_windows(t_start, t_end, src_start, src_end)
    take = np.where(idx >= 0, idx, 0)
    out = src.isel(time=take)
    miss = idx < 0
    for v in out.data_vars:
        if "time" in out[v].dims and miss.any():
            out[v] = out[v].where(xr.DataArray(~miss, dims="time"))
    new = {}
    for c in list(out.coords):
        if c != "time" and out[c].dims == ("time",):
            vals = out[c].values.copy()
            if miss.any():
                if np.issubdtype(vals.dtype, np.datetime64):
                    vals[miss] = np.datetime64("NaT")
                elif vals.dtype.kind in "fc":
                    vals[miss] = np.nan
                else:
                    vals = vals.astype(object); vals[miss] = ""; vals = vals.astype(str)
            new[c] = ("time", vals)
    out = out.assign_coords(time=pd.DatetimeIndex(target_time), **new)
    return out, ~miss
