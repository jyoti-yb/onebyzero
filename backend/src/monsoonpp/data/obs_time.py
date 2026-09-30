"""Observation time-convention model: what 24-hour period does an obs DATE LABEL represent?

File decoding and temporal interpretation are deliberately separate:

  decode      : adapters read values + the source's own date label, and record provenance.
                They never decide which 24 h period a label covers.
  interpret   : `interpret_obs_times()` maps labels -> valid_start / valid_end, and ONLY
                via an explicitly configured `ObsTimeConvention`.

The convention is never inferred from filenames, file years, other products, README
text or NWP timing. Until someone verifies it against the official product
documentation and sets `obs_time_convention` in the run config, it is UNVERIFIED and
every workflow that pairs observations with forecasts is blocked.

States
------
UNVERIFIED     : unknown. Reading allowed; pairing / training samples / training /
                 verification raise ObsTimeConventionError. Default for every run.
ENDING_03Z     : label D covers D-1 03:00 UTC -> D 03:00 UTC
STARTING_03Z   : label D covers D   03:00 UTC -> D+1 03:00 UTC
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr


class ObsTimeConvention(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    ENDING_03Z = "ENDING_03Z"
    STARTING_03Z = "STARTING_03Z"


DAY_BOUNDARY_UTC = pd.Timedelta(hours=3)        # both named conventions use 03 UTC (08:30 IST)
DAY = pd.Timedelta(days=1)

# Observation provenance fields carried on every obs dataset.
PROVENANCE_ATTRS = ["source", "product_name", "time_convention", "units", "grid_role"]
PROVENANCE_COORDS = ["source_filename", "source_date_label", "valid_start", "valid_end"]   # per time step


class ObsTimeConventionError(RuntimeError):
    """Fatal: a workflow needs to know what an observation date label means, and nobody has verified it."""


def parse_convention(value) -> ObsTimeConvention:
    """Strict: only the exact state names are accepted. No aliases, no guessing
    (legacy 'starting'/'ending' strings are rejected so old configs cannot carry
    an unverified assumption forward silently)."""
    if isinstance(value, ObsTimeConvention):
        return value
    try:
        return ObsTimeConvention(value)
    except ValueError:
        raise ObsTimeConventionError(
            f"obs_time_convention {value!r} is not one of {[c.value for c in ObsTimeConvention]}. "
            "Set it explicitly after verifying the product documentation.") from None


def label_window(label, convention) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(valid_start, valid_end) in UTC for one date label under a VERIFIED convention."""
    conv = parse_convention(convention)
    d = pd.Timestamp(label).normalize()
    if conv is ObsTimeConvention.ENDING_03Z:
        return d - DAY + DAY_BOUNDARY_UTC, d + DAY_BOUNDARY_UTC
    if conv is ObsTimeConvention.STARTING_03Z:
        return d + DAY_BOUNDARY_UTC, d + DAY + DAY_BOUNDARY_UTC
    raise ObsTimeConventionError(
        f"cannot convert observation date label {d.date()} to a valid window: convention is UNVERIFIED")


def require_verified(convention, workflow: str, source: str = "") -> ObsTimeConvention:
    """Fatal guard for any workflow that pairs observations with forecasts."""
    conv = parse_convention(convention)
    if conv is ObsTimeConvention.UNVERIFIED:
        raise ObsTimeConventionError(
            f"BLOCKED: '{workflow}' needs the observation time convention of source "
            f"'{source or 'obs'}', which is UNVERIFIED. Verify from the official product documentation "
            f"which 24 h period a date label covers, then set obs_time_convention to one of "
            f"{[c.value for c in ObsTimeConvention if c is not ObsTimeConvention.UNVERIFIED]} in the run config.")
    return conv


def reject_legacy_keys(adapter_options: dict):
    if "imd_window" in (adapter_options or {}):
        raise ObsTimeConventionError(
            "adapter_options.imd_window is no longer supported (it let a run assume an IMD date convention). "
            "Remove it and set top-level obs_time_convention explicitly once verified.")


# ----------------------------------------------------------------------- decoding
def decode_provenance(obs: xr.Dataset, *, source: str, product_name: str, source_filenames,
                      units: str, grid_role: str) -> xr.Dataset:
    """Attach source metadata WITHOUT interpreting time. `time` stays the source date label;
    valid_start/valid_end are NaT and time_convention is UNVERIFIED until interpret_obs_times()."""
    n = obs.sizes["time"]
    labels = pd.DatetimeIndex(obs.time.values)
    fn = np.asarray(list(source_filenames) if not isinstance(source_filenames, str) else [source_filenames] * n,
                    dtype=object)
    if fn.size != n:
        raise ValueError(f"{fn.size} source filenames for {n} time steps")
    obs = obs.assign_coords(
        source_filename=("time", fn.astype(str)),
        source_date_label=("time", np.array([t.strftime("%Y-%m-%d") for t in labels])),
        valid_start=("time", np.full(n, np.datetime64("NaT", "ns"))),
        valid_end=("time", np.full(n, np.datetime64("NaT", "ns"))),
    )
    obs.attrs.update(source=source, product_name=product_name, units=units, grid_role=grid_role,
                     time_convention=ObsTimeConvention.UNVERIFIED.value)
    for v in obs.data_vars:
        obs[v].attrs.setdefault("units", units)
    return obs


def interpret_obs_times(obs: xr.Dataset, convention) -> xr.Dataset:
    """Set valid_start/valid_end from the configured convention. UNVERIFIED leaves them NaT
    (allowed for inspection); any later pairing will be blocked."""
    conv = parse_convention(convention)
    obs = obs.copy()
    obs.attrs["time_convention"] = conv.value
    if conv is ObsTimeConvention.UNVERIFIED:
        return obs
    labels = pd.DatetimeIndex(pd.to_datetime(obs["source_date_label"].values))
    win = [label_window(t, conv) for t in labels]
    obs = obs.assign_coords(valid_start=("time", pd.DatetimeIndex([w[0] for w in win]).values),
                            valid_end=("time", pd.DatetimeIndex([w[1] for w in win]).values))
    return obs
