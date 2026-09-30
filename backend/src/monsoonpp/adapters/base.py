"""Adapter interfaces. The ML system only ever talks to these.

Development : GFSAdapter      + IMDAdapter + ERA5Adapter
Operational : NCUMAdapter     + IMDAdapter + (NCMRWF analysis or ERA5)
CI / demo   : Synthetic*Adapter (no network)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr

from ..config import Config

# "obs"        : daily date-labelled rain on the runtime grid -> the only kind build_dataset reads.
# "obs_native" : observations kept on their NATIVE grid and NATIVE time step with explicit source
#                intervals (e.g. IMERG half-hourly 0.1 deg). They do not satisfy the "obs" contract
#                and are deliberately unreachable from build_dataset / pairing.
_REGISTRY: dict[str, dict[str, type]] = {"forecast": {}, "obs": {}, "obs_native": {}, "analysis": {}}


def register(kind: str, name: str):
    def deco(cls):
        _REGISTRY[kind][name] = cls
        return cls
    return deco


def get_adapter(kind: str, name: str, cfg: Config):
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"No {kind} adapter '{name}'. Available: {sorted(_REGISTRY[kind])}")
    return cls(cfg)


class ForecastAdapter(ABC):
    """Returns canonical forecast Dataset: vars tp,u850,v850,tcwv,cape,mslp on (lead,time,lat,lon).

    `time` is a forecast label D meaning the window 03 UTC D -> 03 UTC D+1
    (data/align.py FORECAST_LABEL_RULE); `tp` is accumulated over that window. This is a
    forecast-side rule only: what an observation date label means is set separately
    and explicitly via obs_time_convention (data/obs_time.py).
    """

    source_name: str = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def load(self, dates: pd.DatetimeIndex, leads: list[int]) -> xr.Dataset: ...

    @property
    def model_version(self) -> str:
        return self.cfg.nwp_model_version


class ObsAdapter(ABC):
    """Returns canonical obs Dataset: var rain on (time,lat,lon), NaN where no data.
    `time` is the SOURCE date label, undecoded in meaning; attach provenance with
    data.obs_time.decode_provenance. Adapters must never set valid_start/valid_end or
    choose a time convention — that is interpret_obs_times()' job, from config only."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def load(self, dates: pd.DatetimeIndex) -> xr.Dataset: ...


class AnalysisAdapter(ABC):
    """Returns canonical analysis Dataset: u850,v850,tcwv,cape,mslp on (time,lat,lon).
    Used for regime LABELS (ground truth of the atmospheric situation), never as model input."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def load(self, dates: pd.DatetimeIndex) -> xr.Dataset: ...


class NativeObsAdapter(ABC):
    """Observation product decoded on its NATIVE grid and time step, no regridding, no
    aggregation, no date-label convention. Timing comes only from the source's own
    interval metadata (coords source_interval_start / source_interval_end)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def load(self, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset: ...
