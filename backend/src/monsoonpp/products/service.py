"""Operational forecast service: raw NWP in -> corrected grid / districts / explanations out.

Observations and analysis are stripped from the dataset before inference, so the
serving path is provably leak-free (it runs exactly as it would on a live NCUM/GFS run).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Config
from ..data.build import load_dataset
from ..pipeline import MODEL_ORDER, Predictor
from ..regimes.classifier import PROB_COLS
from ..regimes.engine import MONSOON_STATES, PRIMARY, SYNOPTIC
from .district import aggregate, build_weights, load_districts
from .explain import Explainer

PRODUCT_MODEL = "L6_regime_qm"


class ForecastService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pred = Predictor.load(cfg.model_dir)
        ds = load_dataset(cfg)
        keep = [v for v in ds.data_vars if v.startswith("fc_")] + ["elev", "land", "coast_dist"]
        self.ds_fc = ds[keep]                    # forecast + static only
        self._obs = ds["obs_rain"]               # only for the optional "verify this day" view
        self.lat, self.lon = ds.lat.values, ds.lon.values
        self.dates = pd.DatetimeIndex(ds.time.values)
        self.leads = [int(x) for x in ds.lead.values]
        self.dw = build_weights(load_districts(cfg.district_geojson), self.lat, self.lon, ds.land.values)
        self._explainer = None
        self.run = lru_cache(maxsize=64)(self._run)

    @property
    def explainer(self) -> Explainer:
        if self._explainer is None:
            arch = pd.read_parquet(self.cfg.product_dir / "train_archive.parquet")
            self._explainer = Explainer(self.pred, arch, seed=self.cfg.seed)
        return self._explainer

    def _check(self, date, lead) -> pd.Timestamp:
        d = pd.Timestamp(date)
        if d not in self.dates:
            raise KeyError(f"date {d.date()} not available ({self.dates[0].date()}..{self.dates[-1].date()})")
        if int(lead) not in self.leads:
            raise KeyError(f"lead {lead} not available {self.leads}")
        return d

    def _run(self, date: str, lead: int) -> pd.DataFrame:
        d = self._check(date, lead)
        return self.pred.forecast_day(self.ds_fc.sel(time=[d]), int(lead))

    def _grid(self, tab: pd.DataFrame, col: str) -> np.ndarray:
        g = np.full((len(self.lat), len(self.lon)), np.nan, "float32")
        g[tab["iy"].values, tab["ix"].values] = tab[col].values
        return g

    # ------------------------------------------------------------ products
    def grid(self, date, lead, model: str = PRODUCT_MODEL) -> dict:
        tab = self.run(str(pd.Timestamp(date).date()), int(lead))
        if model not in tab:
            raise KeyError(f"model {model} unknown; choose from {[m for m in MODEL_ORDER if m in tab]}")
        out = {"date": str(pd.Timestamp(date).date()), "lead_days": int(lead), "model": model,
               "lat": self.lat.round(3).tolist(), "lon": self.lon.round(3).tolist(), "units": "mm/day"}
        def enc(a):
            return [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in a]
        out["rain"] = enc(self._grid(tab, model))
        out["raw"] = enc(self._grid(tab, "L0_raw"))
        for h in self.pred.heads:
            out[h.name] = enc(self._grid(tab, h.name))
        out["regime"] = enc(self._grid(tab, "regime_pred"))
        out["regime_classes"] = PRIMARY
        return out

    def districts(self, date, lead, model: str = PRODUCT_MODEL) -> pd.DataFrame:
        tab = self.run(str(pd.Timestamp(date).date()), int(lead))
        heads = [h.name for h in self.pred.heads]
        fields = {"rain": self._grid(tab, model), "raw": self._grid(tab, "L0_raw")}
        for h in heads:
            fields[h] = self._grid(tab, h)
        return aggregate(fields, self.dw, "rain", heads[0] if heads else None, heads[1] if len(heads) > 1 else None)

    def point(self, lat: float, lon: float, date, lead) -> dict:
        tab = self.run(str(pd.Timestamp(date).date()), int(lead))
        iy, ix = int(np.abs(self.lat - lat).argmin()), int(np.abs(self.lon - lon).argmin())
        row = tab[(tab.iy == iy) & (tab.ix == ix)]
        if row.empty:
            raise KeyError("point is outside the verification (land) mask")
        r = row.iloc[0]
        out = {"grid_lat": float(self.lat[iy]), "grid_lon": float(self.lon[ix]),
               "models_mm": {m: round(float(r[m]), 2) for m in MODEL_ORDER if m in r},
               "probabilities": {h.name: round(float(r[h.name]), 3) for h in self.pred.heads}}
        out["explanation"] = self.explainer.explain(row, float(r[PRODUCT_MODEL]), float(r["L0_raw"]))
        d = pd.Timestamp(date)
        obs = float(self._obs.sel(time=d).values[iy, ix])
        out["observed_mm_for_demo_only"] = None if np.isnan(obs) else round(obs, 2)
        return out

    def regimes(self, date, lead) -> dict:
        d = self._check(date, lead)
        eng = self.pred.bundle
        fr = eng.forecast(self.ds_fc.sel(time=[d]), int(lead))
        tab = self.run(str(d.date()), int(lead))
        counts = pd.Series(tab["regime_pred"]).map(dict(enumerate(PRIMARY))).value_counts(normalize=True)
        systems = [{"lat": float(s[1]), "lon": float(s[2]), "deficit_hPa": round(s[3], 2), "class": s[4]}
                   for s in eng.fc_engines[int(lead)].last_systems]
        return {"date": str(d.date()), "lead_days": int(lead),
                "monsoon_state": MONSOON_STATES[int(fr.monsoon_state.values[0])],
                "monsoon_index_z": round(float(fr.monsoon_z.values[0]), 2),
                "systems": systems,
                "land_fraction_by_regime": {k: round(float(v), 3) for k, v in counts.items()},
                "mean_regime_confidence": round(float(tab["regime_conf"].mean()), 3)}
