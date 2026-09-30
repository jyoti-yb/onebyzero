"""Forecaster-facing explanations for a corrected value.

1. Drivers: TreeSHAP contributions of the regime-aware model (log-link space),
   grouped into meteorological themes.
2. Analogues: the k most similar historical forecast situations in the same
   predicted regime, with what the raw NWP said and what was observed
   ("meteorological retrieval-augmented forecasting").
3. Confidence: regime-classifier confidence x how typical the situation is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..regimes.classifier import PROB_COLS
from ..regimes.engine import PRIMARY

ANALOGUE_FEATURES = ["tp", "tp_n5_mean", "tcwv", "u850", "upslope", "oro_score", "monsoon_z",
                     "dist_system_km", "elev", "coast_dist", "lead"]

THEMES = {
    "raw NWP rainfall": ["tp", "tp_log", "tp_n3_mean", "tp_n5_mean", "tp_n5_max", "tp_n9_mean", "tp_n9_max", "tp_grad"],
    "orographic forcing": ["upslope", "oro_score", "elev", "p_orographic"],
    "coastal / onshore flow": ["onshore", "coast_score", "coast_dist", "p_coastal"],
    "monsoon state (active/break)": ["monsoon_z", "cmz_tp", "p_active", "p_break", "p_normal"],
    "synoptic system (objective proxy)": ["dist_system_km", "synoptic", "mslp_spatial_residual", "vo850",
                                             "p_low_depression_proxy"],
    "moisture & instability": ["tcwv", "cape", "moist_flux", "conv_score"],
    "low-level wind": ["u850", "v850", "wspd"],
    "location & season": ["lat", "lon", "doy_sin", "doy_cos", "lead", "fc_primary"],
}


class Explainer:
    def __init__(self, predictor, archive: pd.DataFrame, k: int = 20, max_per_regime: int = 60_000, seed: int = 0):
        self.pred, self.k = predictor, k
        self.index: dict[int, tuple] = {}
        reg = archive[PROB_COLS].values.argmax(1)
        self.mu = archive[ANALOGUE_FEATURES].mean()
        self.sd = archive[ANALOGUE_FEATURES].std().replace(0, 1)
        for r in np.unique(reg):
            a = archive[reg == r]
            if len(a) > max_per_regime:
                a = a.sample(max_per_regime, random_state=seed)
            Z = ((a[ANALOGUE_FEATURES] - self.mu) / self.sd).values
            nn = NearestNeighbors(n_neighbors=min(k, len(a))).fit(Z)
            d_ref = nn.kneighbors(Z[:: max(1, len(Z) // 2000)])[0].mean(1)
            self.index[int(r)] = (nn, a.reset_index(drop=True), np.percentile(d_ref, [50, 95]))

    def drivers(self, row: pd.DataFrame, top: int = 5) -> list[dict]:
        c = self.pred.regime.contributions(row).iloc[0]
        themes = []
        for t, feats in THEMES.items():
            v = float(sum(c.get(f, 0.0) for f in feats))
            themes.append({"driver": t, "contribution_log": round(v, 3),
                           "direction": "increases" if v > 0 else "decreases"})
        return sorted(themes, key=lambda x: -abs(x["contribution_log"]))[:top]

    def analogues(self, row: pd.DataFrame, regime: int) -> dict:
        if regime not in self.index:
            return {"n": 0}
        nn, a, (d50, d95) = self.index[regime]
        z = ((row[ANALOGUE_FEATURES] - self.mu) / self.sd).values
        d, i = nn.kneighbors(z)
        sel = a.iloc[i[0]]
        dist = float(d[0].mean())
        return {
            "n": int(len(sel)), "regime": PRIMARY[regime],
            "raw_nwp_median_mm": float(sel["tp"].median()),
            "observed_median_mm": float(sel["obs"].median()),
            "observed_p90_mm": float(sel["obs"].quantile(0.9)),
            "historical_bias_median_mm": float((sel["obs"] - sel["tp"]).median()),
            "frac_obs_ge_64.5": float((sel["obs"] >= 64.5).mean()),
            "typicality": "typical" if dist <= d50 else ("unusual" if dist <= d95 else "rare"),
            "mean_distance": round(dist, 3),
            "dates": sorted({str(pd.Timestamp(t).date()) for t in sel["time"].head(5)}),
        }

    def explain(self, row: pd.DataFrame, corrected: float, raw: float) -> dict:
        probs = row[PROB_COLS].values[0]
        regime = int(probs.argmax())
        ana = self.analogues(row, regime)
        conf = float(probs.max())
        typ = ana.get("typicality", "rare")
        level = "HIGH" if conf >= 0.6 and typ == "typical" else ("LOW" if conf < 0.4 or typ == "rare" else "MEDIUM")
        return {
            "raw_mm": round(raw, 2), "corrected_mm": round(corrected, 2), "correction_mm": round(corrected - raw, 2),
            "regime": PRIMARY[regime], "regime_probs": {PRIMARY[i]: round(float(p), 3) for i, p in enumerate(probs)},
            "drivers": self.drivers(row), "analogues": ana,
            "confidence": level,
            "confidence_note": "Post-processing cannot create a system the NWP run missed; LOW means the "
                               "situation is rare in the training archive or the regime is ambiguous.",
        }
