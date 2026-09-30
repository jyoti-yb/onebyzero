"""Observation SOURCE provenance: which product this is, and what it may be used for.

Independent of (and layered on top of) the time-convention model in obs_time.py.

Roles
-----
FINAL_TRUTH           IMD 0.25 deg gauge gridded rainfall — the intended scientific truth.
OFFICIAL_ALTERNATIVE  IMD–NCMRWF 0.25 deg merged (gauge + satellite) rainfall — acceptable official stand-in.
SMOKE_TEST_PROXY      NASA GPM IMERG — engineering smoke tests only; never truth, never "IMD".
SYNTHETIC_TEST        generated in-memory world for CI; not an observation at all.

Every combination of source_name / role / is_proxy is validated on construction AND on
deserialisation, so an IMERG dataset cannot be relabelled as IMD or as final truth.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum

import xarray as xr


class ProductRole(str, Enum):
    FINAL_TRUTH = "FINAL_TRUTH"
    OFFICIAL_ALTERNATIVE = "OFFICIAL_ALTERNATIVE"
    SMOKE_TEST_PROXY = "SMOKE_TEST_PROXY"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"


class ObsSourceError(ValueError):
    """Invalid, inconsistent or masquerading observation-source provenance."""


# source_name -> (the only role it may carry, is_proxy)
SOURCE_RULES: dict[str, tuple[ProductRole, bool]] = {
    "IMD": (ProductRole.FINAL_TRUTH, False),
    "IMD-NCMRWF": (ProductRole.OFFICIAL_ALTERNATIVE, False),
    "NASA-GPM-IMERG": (ProductRole.SMOKE_TEST_PROXY, True),
    "synthetic": (ProductRole.SYNTHETIC_TEST, True),
}
_IMD_TOKEN = re.compile(r"\bIMD\b", re.I)
_IMERG_TOKEN = re.compile(r"IMERG", re.I)

FIELDS = ["source_name", "product_name", "product_role", "source_filename", "source_url_or_origin",
          "observation_resolution", "native_grid", "units", "time_convention", "is_proxy"]
ATTR_PREFIX = "obs_meta_"          # namespaced so it never collides with other attrs


@dataclass(frozen=True)
class ObsSourceMeta:
    source_name: str
    product_name: str
    product_role: ProductRole
    source_filename: str
    source_url_or_origin: str
    observation_resolution: str
    native_grid: str
    units: str
    time_convention: str
    is_proxy: bool

    def __post_init__(self):
        role = self.product_role
        if not isinstance(role, ProductRole):
            try:
                object.__setattr__(self, "product_role", ProductRole(role))
            except ValueError:
                raise ObsSourceError(f"product_role {role!r} not in {[r.value for r in ProductRole]}") from None
        if not isinstance(self.is_proxy, bool):
            raise ObsSourceError(f"is_proxy must be bool, got {self.is_proxy!r}")
        if self.source_name not in SOURCE_RULES:
            raise ObsSourceError(f"unknown observation source {self.source_name!r}; known: {sorted(SOURCE_RULES)}")
        want_role, want_proxy = SOURCE_RULES[self.source_name]
        if self.product_role is not want_role:
            raise ObsSourceError(f"{self.source_name} may only carry role {want_role.value}, not {self.product_role.value}")
        if self.is_proxy is not want_proxy:
            raise ObsSourceError(f"{self.source_name} must have is_proxy={want_proxy}")
        # anti-masquerade on free-text fields
        if self.source_name == "NASA-GPM-IMERG":
            if _IMD_TOKEN.search(self.product_name) or not _IMERG_TOKEN.search(self.product_name):
                raise ObsSourceError(f"IMERG product_name must name IMERG and must not name IMD: {self.product_name!r}")
        if self.source_name == "IMD" and _IMERG_TOKEN.search(self.product_name):
            raise ObsSourceError(f"IMD gauge product_name must not name IMERG: {self.product_name!r}")
        for f in ("product_name", "source_filename", "observation_resolution", "native_grid", "units", "time_convention"):
            if not str(getattr(self, f)).strip():
                raise ObsSourceError(f"{f} must not be empty")

    # ---------------------------------------------------------------- serialisation
    def to_attrs(self) -> dict[str, str]:
        d = asdict(self)
        d["product_role"] = self.product_role.value
        d["is_proxy"] = "true" if self.is_proxy else "false"      # netCDF has no bool attrs
        return {ATTR_PREFIX + k: str(v) for k, v in d.items()}

    @classmethod
    def from_attrs(cls, attrs: dict) -> "ObsSourceMeta":
        missing = [f for f in FIELDS if ATTR_PREFIX + f not in attrs]
        if missing:
            raise ObsSourceError(f"observation source provenance missing: {missing}")
        d = {f: str(attrs[ATTR_PREFIX + f]) for f in FIELDS}
        if d["is_proxy"] not in ("true", "false"):
            raise ObsSourceError(f"is_proxy attr must be 'true' or 'false', got {d['is_proxy']!r}")
        d["is_proxy"] = d["is_proxy"] == "true"
        return cls(**d)

    def describe(self) -> str:
        tag = " [PROXY — NOT IMD, NOT TRUTH]" if self.is_proxy and self.source_name != "synthetic" else ""
        return f"{self.source_name}: {self.product_name} ({self.product_role.value}){tag}"


def attach(ds: xr.Dataset, meta: ObsSourceMeta) -> xr.Dataset:
    ds = ds.copy()
    ds.attrs.update(meta.to_attrs())
    return ds


def read(ds: xr.Dataset) -> ObsSourceMeta:
    return ObsSourceMeta.from_attrs(ds.attrs)


def require_daily_truth_capable(obs: xr.Dataset, workflow: str) -> ObsSourceMeta:
    """Gate for the daily obs path (build_dataset). Proxies cannot enter it in this build."""
    meta = read(obs)
    if meta.product_role is ProductRole.SMOKE_TEST_PROXY:
        raise ObsSourceError(f"BLOCKED: '{workflow}': {meta.describe()} is a smoke-test proxy; it cannot enter "
                             f"the forecast/observation pairing path.")
    return meta
