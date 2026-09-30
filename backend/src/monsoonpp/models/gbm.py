"""Gradient-boosted rainfall correctors (levels 3-5).

Tweedie objective: rainfall is zero-inflated and right-skewed; tweedie handles the
point mass at 0 plus a heavy tail far better than MSE on raw mm (which smooths
extremes) or MSE on log1p (which biases the mean low).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore", category=lgb.basic.LGBMDeprecationWarning if hasattr(lgb.basic, "LGBMDeprecationWarning") else DeprecationWarning)
import pandas as pd

from ..config import ModelConfig
from ..regimes.classifier import PROB_COLS
from ..regimes.engine import PRIMARY


def _reg_params(mc: ModelConfig, seed: int) -> dict:
    return dict(objective="tweedie", tweedie_variance_power=mc.tweedie_variance_power,
                n_estimators=mc.n_estimators, learning_rate=mc.learning_rate, num_leaves=mc.num_leaves,
                min_child_samples=mc.min_child_samples, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1)


class RainGBM:
    def __init__(self, name: str, features: list[str], mc: ModelConfig, seed: int = 0):
        self.name, self.features, self.mc, self.seed = name, features, mc, seed
        self.model: lgb.LGBMRegressor | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, weight=None):
        self.model = lgb.LGBMRegressor(**_reg_params(self.mc, self.seed))
        kw = {}
        if val is not None and len(val):
            kw = dict(eval_set=[(val[self.features], val["obs"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        self.model.fit(train[self.features], train["obs"], sample_weight=weight, **kw)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[self.features]), 0, None).astype("float32")

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions (TreeSHAP, in the model's log-link space) + bias column."""
        c = self.model.predict(df[self.features], pred_contrib=True)
        return pd.DataFrame(c, columns=[*self.features, "_bias"], index=df.index)

    def importance(self) -> dict:
        imp = self.model.booster_.feature_importance("gain")
        return dict(sorted(zip(self.features, (imp / imp.sum()).round(4).tolist()), key=lambda x: -x[1]))


class MixtureOfExperts:
    """One expert per regime, trained on rows whose ANALYSIS regime is that class.
    Inference gates softly with the regime classifier's probabilities and shrinks each
    expert toward the global regime-aware model by n_c / (n_c + k) (data-poor regimes
    fall back to the global model instead of overfitting)."""

    name = "L5_moe"

    def __init__(self, features: list[str], mc: ModelConfig, global_model: RainGBM, seed: int = 0):
        self.features, self.mc, self.glob, self.seed = features, mc, global_model, seed
        self.experts: dict[int, RainGBM] = {}
        self.weights: dict[int, float] = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        for c, cname in enumerate(PRIMARY):
            tr = train[train["label_primary"] == c]
            n = len(tr)
            if n < self.mc.min_expert_samples:
                self.weights[c] = 0.0
                continue
            va = val[val["label_primary"] == c] if val is not None else None
            self.experts[c] = RainGBM(f"expert_{cname}", self.features, self.mc, self.seed).fit(tr, va)
            self.weights[c] = n / (n + self.mc.moe_shrinkage_k)
        return self

    def predict(self, df: pd.DataFrame, return_parts: bool = False):
        g = self.glob.predict(df).astype("float64")
        P = df[PROB_COLS].values
        out = np.zeros(len(df))
        parts = {}
        for c in range(len(PRIMARY)):
            w = self.weights.get(c, 0.0)
            f = self.experts[c].predict(df) if c in self.experts else g
            blend = w * f + (1 - w) * g
            parts[PRIMARY[c]] = blend
            out += P[:, c] * blend
        out = np.clip(out, 0, None).astype("float32")
        return (out, parts) if return_parts else out
