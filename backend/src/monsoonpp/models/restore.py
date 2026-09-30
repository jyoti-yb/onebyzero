"""L6: regime-conditioned distribution restoration of the ML output.

Tweedie/MSE-type regressors return a conditional MEAN, which is smoother than
reality: heavy-rain frequency bias << 1 and poor CSI at 64.5/115.6 mm. This step
quantile-maps the ML output onto the observed distribution, separately per
lead x predicted regime (fitted on the validation year, never on test), restoring
the tail while keeping the ML model's ranking/placement skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Q = np.unique(np.concatenate([np.linspace(0, 0.9, 46), np.linspace(0.9, 0.99, 19), np.linspace(0.99, 1, 11)]))


def _qmap(x, qf, qo):
    qf_u, keep = np.unique(qf, return_index=True)
    y = np.interp(x, qf_u, qo[keep])
    top = x > qf[-1]
    y[top] = x[top] + (qo[-1] - qf[-1])
    return np.clip(y, 0, None)


class RegimeQuantileRestore:
    name = "L6_regime_qm"

    def __init__(self, base_col: str = "L4_gbm_regime", min_n: int = 2000):
        self.base_col, self.min_n = base_col, min_n
        self.maps: dict = {}

    def fit(self, base_pred: np.ndarray, obs: np.ndarray, lead: np.ndarray, regime: np.ndarray):
        df = pd.DataFrame({"p": base_pred, "o": obs, "L": lead, "r": regime})
        for L, g in df.groupby("L"):
            self.maps[(L, None)] = (np.quantile(g.p, Q), np.quantile(g.o, Q))
            for r, gg in g.groupby("r"):
                if len(gg) >= self.min_n:
                    self.maps[(L, int(r))] = (np.quantile(gg.p, Q), np.quantile(gg.o, Q))
        return self

    def predict(self, base_pred: np.ndarray, lead: np.ndarray, regime: np.ndarray) -> np.ndarray:
        out = np.asarray(base_pred, dtype="float64").copy()
        df = pd.DataFrame({"L": lead, "r": regime})
        for (L, r), idx in df.groupby(["L", "r"]).indices.items():
            m = self.maps.get((L, int(r))) or self.maps.get((L, None))
            if m is not None:
                out[idx] = _qmap(out[idx], *m)
        return out.astype("float32")
