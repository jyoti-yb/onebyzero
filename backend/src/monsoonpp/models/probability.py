"""Heavy-rainfall exceedance probabilities P(R > thr) for IMD thresholds.

Separate binary heads rather than thresholding the deterministic forecast: the
regression mean is smoothed and systematically under-hits extremes. Each head is
isotonically calibrated on the validation year so probabilities are reliable.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore", category=lgb.basic.LGBMDeprecationWarning if hasattr(lgb.basic, "LGBMDeprecationWarning") else DeprecationWarning)
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class ExceedanceHead:
    def __init__(self, threshold: float, features: list[str], seed: int = 0, n_estimators: int = 300):
        self.thr, self.features = threshold, features
        self.model = lgb.LGBMClassifier(objective="binary", n_estimators=n_estimators, learning_rate=0.05,
                                        num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1)
        self.cal: IsotonicRegression | None = None

    @property
    def name(self):
        return f"p_gt_{self.thr:g}"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        y = (train["obs"].values > self.thr).astype(int)
        if y.sum() < 20:
            raise ValueError(f"too few events > {self.thr} mm in training ({y.sum()})")
        yv = (val["obs"].values > self.thr).astype(int)
        self.model.fit(train[self.features], y, eval_set=[(val[self.features], yv)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        raw = self.model.predict_proba(val[self.features])[:, 1]
        self.cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, yv)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(df[self.features])[:, 1]
        return self.cal.predict(raw).astype("float32")
