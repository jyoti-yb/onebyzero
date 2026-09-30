"""NOAA GFS historical forecasts via Herbie (AWS open data, noaa-gfs-bdp-pds).

Coverage of the 0.25 deg pgrb2 archive on AWS starts ~2021-02, which is why the
development period is JJAS 2021-2024. Earlier years: NCEI GFS archive (0.5 deg).

pip install herbie-data cfgrib eccodes
Daily rainfall: see gfs_precip.py (primary APCP(0-27)-APCP(0-3), bucket QA, GRIB-metadata checks).

NOT exercised in the build sandbox (no network). First thing to run on a real
machine:  pytest -m network   or   python -m monsoonpp.adapters.gfs 2023-07-15
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..data.align import FORECAST_LABEL_RULE, forecast_window, to_grid
from ..grid import make_coords
from .base import ForecastAdapter, register
from .gfs_atmos import (FIELD_SPECS, HERBIE_SEARCH, REGIME_HERBIE_SEARCH,
                        GFSAtmosMetadataError, GFSAtmosValidationError,
                        decode_gfs_atmos_file, derive_atmospheric_fields)
from .gfs_precip import APCPMetadataError, APCPValidationError, GFSDailyPrecip, Tolerances

# per-(lead, time) provenance carried as coordinates on the forecast dataset
TP_PROVENANCE_KEYS = ["init_time", "valid_start", "valid_end", "source_apcp_step_ranges",
                      "source_apcp_grib_stepRange", "source_files", "qa_mean_abs_mm", "qa_max_abs_mm",
                      "qa_p95_abs_mm", "qa_frac_below_max", "atmos_source_files",
                      "atmos_sample_valid_times"]

log = logging.getLogger(__name__)

# canonical name -> (Herbie search regex, scale, offset)
GFS_FIELDS = {
    "u850": (r":UGRD:850 mb:", 1.0, 0.0),
    "v850": (r":VGRD:850 mb:", 1.0, 0.0),
    "tcwv": (r":PWAT:entire atmosphere", 1.0, 0.0),
    "cape": (r":CAPE:surface:", 1.0, 0.0),
    "mslp": (r":PRMSL:mean sea level:", 0.01, 0.0),   # Pa -> hPa
}

DERIVED_ATMOS_FIELDS = ("wspd850", "vo850", "moisture_transport", "pwat_anom", "mslp_anom")


def _first_var(ds) -> xr.DataArray:
    if isinstance(ds, list):
        ds = ds[0]
    name = [v for v in ds.data_vars if v not in ("gribfile_projection",)][0]
    return ds[name]


@register("forecast", "gfs")
class GFSAdapter(ForecastAdapter):
    source_name = "gfs"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lat, self.lon = make_coords(cfg.grid)
        self.label_rule = FORECAST_LABEL_RULE   # forecast label = window-start date; never an IMD assumption
        self.product = cfg.adapter_options.get("gfs_product", "pgrb2.0p25")
        self.strict = bool(cfg.adapter_options.get("gfs_strict", True))
        self.cache = Path(cfg.raw_dir) / "gfs"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _herbie(self, init, fxx):
        from herbie import Herbie   # lazy: optional dependency
        return Herbie(
            init, model="gfs", product=self.product, fxx=fxx, save_dir=str(self.cache), verbose=False,
            priority=self.cfg.adapter_options.get("gfs_priority"),
        )

    def _read(self, init, fxx, search) -> xr.DataArray:
        da = _first_var(self._herbie(init, fxx).xarray(search, remove_grib=False))
        return to_grid(da, self.lat, self.lon)

    def _fetch_apcp_file(self, init, fxx) -> Path:
        """Download the APCP messages of one GFS file. The .idx search only limits the
        bytes fetched; which accumulation each message holds is proven from GRIB metadata
        in gfs_precip.read_apcp_messages, never from this search string or the file name."""
        H = self._herbie(init, fxx)
        out = H.download(r":APCP:", verbose=False)
        path = Path(out) if out else Path(H.get_localFilePath(r":APCP:"))
        if not path.exists():
            raise FileNotFoundError(f"GFS APCP subset not found for {init} f{fxx:03d}: {path}")
        return path

    def _fetch_atmos_file(self, init, fxx) -> Path:
        """Fetch only Phase B fields; identity and timing are verified from GRIB2."""
        H = self._herbie(init, fxx)
        out = H.download(HERBIE_SEARCH, verbose=False)
        path = Path(out) if out else Path(H.get_localFilePath(HERBIE_SEARCH))
        if not path.exists():
            raise FileNotFoundError(f"GFS atmospheric subset not found for {init} f{fxx:03d}: {path}")
        return path

    def _fetch_regime_atmos_file(self, init, fxx) -> Path:
        """Fetch only the five strictly validated fields consumed by the regime engine."""
        day = self.cache / "gfs" / pd.Timestamp(init).strftime("%Y%m%d")
        cached = list(day.glob(f"subset_*0921__gfs.t00z.{self.product}.f{int(fxx):03d}"))
        if len(cached) == 1:
            return cached[0]
        if len(cached) > 1:
            raise RuntimeError(f"ambiguous cached GFS regime subsets for {init} f{fxx:03d}: {cached}")
        H = self._herbie(init, fxx)
        out = H.download(REGIME_HERBIE_SEARCH, verbose=False)
        path = Path(out) if out else Path(H.get_localFilePath(REGIME_HERBIE_SEARCH))
        if not path.exists():
            raise FileNotFoundError(f"GFS regime subset not found for {init} f{fxx:03d}: {path}")
        return path

    def _load_atmos_day(self, w):
        samples, qa, paths = [], [], []
        bbox = [float(self.lat[0]), float(self.lat[-1]), float(self.lon[0]), float(self.lon[-1])]
        for fxx in w.mean_hours:
            path = self._fetch_atmos_file(w.init, fxx)
            sample, facts = decode_gfs_atmos_file(path, w.init, fxx, bbox=bbox)
            sample = sample.sel(lat=self.lat, lon=self.lon)
            samples.append(sample)
            qa.append(facts)
            paths.append(str(path))
        daily = xr.concat(samples, dim="sample_time").mean("sample_time").astype("float32")
        for name in FIELD_SPECS:
            daily[name].attrs = {
                **samples[0][name].attrs,
                "aggregation": f"mean of instantaneous f{','.join(f'{h:03d}' for h in w.mean_hours)} fields",
                "sample_valid_times": ";".join(item["valid_time"] for item in qa),
            }
        derived = derive_atmospheric_fields(daily)
        for name in DERIVED_ATMOS_FIELDS:
            daily[name] = derived[name]
        return daily, {
            "source_files": paths,
            "sample_valid_times": [item["valid_time"] for item in qa],
            "sample_forecast_hours": list(w.mean_hours),
            "field_validation": qa,
        }

    def load_atmospheric(self, valid_date, lead: int = 1):
        """Load one validated daily atmospheric state without invoking precipitation."""
        window = forecast_window(pd.Timestamp(valid_date), int(lead), self.label_rule)
        daily, provenance = self._load_atmos_day(window)
        daily.attrs.update(
            forecast_label=str(pd.Timestamp(valid_date).normalize()),
            lead_days=int(lead),
            init_time=str(window.init),
            valid_start=str(window.init + pd.Timedelta(hours=window.f_start)),
            valid_end=str(window.init + pd.Timedelta(hours=window.f_end)),
        )
        return daily, provenance

    @property
    def precip(self) -> GFSDailyPrecip:
        if getattr(self, "_precip", None) is None:
            tol = Tolerances(**self.cfg.adapter_options.get("gfs_precip_tolerances", {}))
            self._precip = GFSDailyPrecip(self._fetch_apcp_file, tol, model="GFS", product=self.product)
        return self._precip

    def _tp(self, w, lead: int = 1):
        """Primary APCP(0-e) - APCP(0-s) on the native grid (verified + QA'd), then put on cfg.grid."""
        res = self.precip.build(w, lead)
        prov = {**res.provenance, **{k: v for k, v in res.qa.items() if k.startswith("qa_") and not isinstance(v, (list, dict))}}
        return to_grid(res.field, self.lat, self.lon).values, prov

    def _one(self, valid, lead) -> tuple[dict[str, np.ndarray], dict]:
        w = forecast_window(valid, lead, self.label_rule)
        tp, prov = self._tp(w, lead)
        atmos, atmos_prov = self._load_atmos_day(w)
        out = {"tp": tp, **{name: atmos[name].values for name in atmos.data_vars}}
        prov["atmos_source_files"] = ";".join(atmos_prov["source_files"])
        prov["atmos_sample_valid_times"] = ";".join(atmos_prov["sample_valid_times"])
        self.last_atmos_qa = atmos_prov["field_validation"]
        return out, prov

    def load(self, dates, leads):
        atmos_names = [*FIELD_SPECS, *DERIVED_ATMOS_FIELDS]
        arrs = {k: np.full((len(leads), len(dates), len(self.lat), len(self.lon)), np.nan, "float32")
                for k in ["tp", *atmos_names]}
        prov = {k: np.full((len(leads), len(dates)), "", dtype=object) for k in TP_PROVENANCE_KEYS}
        static_prov = {}
        for li, L in enumerate(leads):
            for ti, d in enumerate(dates):
                try:
                    fields, p = self._one(d, L)
                except (APCPMetadataError, APCPValidationError, GFSAtmosMetadataError, GFSAtmosValidationError):
                    if self.strict:          # default: metadata/interval/QA problems stop the build
                        raise
                    log.error("GFS %s lead %s REJECTED (gfs_strict=false)", d.date(), L, exc_info=True)
                    continue
                except Exception as e:   # download/IO failure: day stays NaN, dropped downstream
                    log.error("GFS %s lead %s unavailable: %s: %s", d.date(), L, type(e).__name__, e)
                    continue
                for k, v in fields.items():
                    arrs[k][li, ti] = v
                for k in TP_PROVENANCE_KEYS:
                    prov[k][li, ti] = str(p.get(k, ""))
                static_prov = {k: p[k] for k in ("model", "product", "cycle", "source_units", "output_units",
                                                 "unit_conversion", "qa_method") if k in p}
                static_prov["construction_method"] = "APCP(0-f_end) - APCP(0-f_start); cell by cell on native grid"
        ds = xr.Dataset({k: (("lead", "time", "lat", "lon"), v) for k, v in arrs.items()},
                        coords={"lead": leads, "time": dates, "lat": self.lat, "lon": self.lon})
        for k, v in prov.items():
            ds.coords[f"tp_{k}"] = (("lead", "time"), v.astype(str))
        ds["tp"].attrs.update(static_prov, units="mm", tolerances=str(self.precip.tol.__dict__))
        for name in FIELD_SPECS:
            spec = FIELD_SPECS[name]
            ds[name].attrs.update(
                units=spec.output_units,
                source="NOAA GFS pgrb2.0p25",
                source_short_name=spec.short_name,
                source_parameter=str(spec.param),
                source_level=f"{spec.type_of_level}:{spec.level}",
                aggregation="mean of four instantaneous fields within the 03Z-03Z window",
                validation="direct GRIB2 metadata and values via ecCodes",
            )
        derived_attrs = derive_atmospheric_fields(ds[["u850", "v850", "tcwv", "mslp"]]).data_vars
        for name in DERIVED_ATMOS_FIELDS:
            ds[name].attrs.update(derived_attrs[name].attrs)
        ds.attrs.update(source="gfs", product=self.product, model_version=self.model_version)
        return ds


if __name__ == "__main__":   # smoke test on a real machine
    import sys
    from ..config import load_config
    cfg = load_config(None, forecast_source="gfs")
    cfg.grid.res = 0.25
    ds = GFSAdapter(cfg).load(pd.DatetimeIndex([sys.argv[1] if len(sys.argv) > 1 else "2023-07-15"]), [1])
    print(ds, float(ds.tp.mean()))
