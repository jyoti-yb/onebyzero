"""Run configuration. One YAML file drives the whole pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridConfig:
    """Runtime analysis grid (cell centres, inclusive bounds).

    The default is the CANONICAL IMD 0.25 deg rectangular grid (Pai et al. 2014):
    6.5-38.5 N, 66.5-100.0 E -> 129 lat x 135 lon. Cells outside India / without
    observations are NOT removed from this grid; they are excluded by the `land`
    verification mask (see grid.py). A config may override these values to run on a
    cropped and/or coarsened runtime domain; grid.grid_role() reports which one a run uses,
    and only the canonical grid may be described as "the IMD grid".
    """
    lat_min: float = 6.5
    lat_max: float = 38.5
    lon_min: float = 66.5
    lon_max: float = 100.0
    res: float = 0.25


@dataclass
class SplitConfig:
    train_years: list[int] = field(default_factory=lambda: [2021, 2022])
    val_years: list[int] = field(default_factory=lambda: [2023])
    test_years: list[int] = field(default_factory=lambda: [2024])


@dataclass
class ModelConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    tweedie_variance_power: float = 1.3
    moe_shrinkage_k: int = 20000      # expert weight = n / (n + k)
    min_expert_samples: int = 5000
    max_train_rows: int | None = None  # subsample training rows for speed (None = all)


@dataclass
class Config:
    name: str = "synthetic"
    forecast_source: str = "synthetic"      # synthetic | gfs | ncum
    obs_source: str = "synthetic"           # synthetic | imd | imd_netcdf | imd_ncmrwf_merged (IMERG: obs_native only)
    analysis_source: str = "synthetic"      # synthetic | era5
    # What 24 h period an OBSERVATION date label covers (data/obs_time.py).
    # UNVERIFIED | ENDING_03Z | STARTING_03Z. Default UNVERIFIED blocks pairing/training/
    # verification; set it only after verifying the official product documentation.
    obs_time_convention: str = "UNVERIFIED"
    nwp_model_version: str = "synthetic-v1"
    years: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024])
    season_start: str = "06-01"
    season_end: str = "09-30"
    leads: list[int] = field(default_factory=lambda: [1, 3])   # days
    grid: GridConfig = field(default_factory=GridConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # IMD operational thresholds (mm/day): heavy, very heavy, extremely heavy
    thresholds: list[float] = field(default_factory=lambda: [2.5, 15.6, 64.5, 115.6, 204.5])
    prob_thresholds: list[float] = field(default_factory=lambda: [64.5, 115.6])
    fss_scales: list[int] = field(default_factory=lambda: [1, 3, 5, 9])   # grid cells (odd)
    seed: int = 42
    data_dir: str = "data"
    raw_dir: str = "data/raw"             # where real-source downloads are cached
    district_geojson: str | None = None   # real district polygons; None -> demo tiles
    adapter_options: dict[str, Any] = field(default_factory=dict)
    smoke: dict[str, Any] = field(default_factory=dict)       # seven-day smoke test (smoke.py)
    phase_b: dict[str, Any] = field(default_factory=dict)     # real GFS/static validation (phase_b.py)
    phase_c: dict[str, Any] = field(default_factory=dict)     # regime audit + pre-ML real-data atlas
    phase_c2: dict[str, Any] = field(default_factory=dict)    # expanded atlas + statistical evidence audit
    phase_c3: dict[str, Any] = field(default_factory=dict)    # JJAS exact-pair + independent-system evidence audit

    # ---- derived paths ----
    @property
    def run_dir(self) -> Path:
        return Path(self.data_dir) / self.name

    @property
    def dataset_path(self) -> Path:
        return self.run_dir / "dataset.nc"

    @property
    def model_dir(self) -> Path:
        return self.run_dir / "models"

    @property
    def product_dir(self) -> Path:
        return self.run_dir / "products"

    @property
    def report_dir(self) -> Path:
        return self.run_dir / "reports"

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(dc, values: dict):
    for k, v in values.items():
        if not hasattr(dc, k):
            raise KeyError(f"Unknown config key: {k}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None = None, **overrides) -> Config:
    cfg = Config()
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    _merge(cfg, overrides)
    return cfg
