"""ML weather-regime classifier: forecast predictors -> analysis-derived regime label.

Why both a rule engine and a classifier?
  * rules on forecast fields are interpretable and always available;
  * the classifier learns how the NWP model's own depiction maps to what actually
    happened (e.g. a displaced low), and gives probabilities for soft MoE gating.
Training-set probabilities are cross-fitted by year so downstream models never see
in-sample (overconfident) regime probabilities.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..features import ALL_FEATURES
from .engine import PRIMARY

PROB_COLS = [f"p_{c}" for c in PRIMARY]


class RegimeClassifier:
    def __init__(self, n_estimators=150, max_rows=300_000, seed=0):
        self.params = dict(objective="multiclass", num_class=len(PRIMARY), n_estimators=n_estimators,
                           learning_rate=0.08, num_leaves=63, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1)
        self.max_rows = max_rows
        self.seed = seed
        self.model: lgb.LGBMClassifier | None = None

    def _fit(self, df):
        if len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=self.seed)
        m = lgb.LGBMClassifier(**self.params)
        m.fit(df[ALL_FEATURES], df["label_primary"].astype(int))
        return m

    def _proba(self, m, X):
        p = np.zeros((len(X), len(PRIMARY)), "float32")
        p[:, m.classes_] = m.predict_proba(X[ALL_FEATURES])
        return p

    def fit_crossfit(self, train: pd.DataFrame) -> np.ndarray:
        """Fit final model on all training rows; return out-of-year probabilities for them."""
        oof = np.zeros((len(train), len(PRIMARY)), "float32")
        years = sorted(train["year"].unique())
        if len(years) > 1:
            for y in years:
                te = (train["year"] == y).values
                oof[te] = self._proba(self._fit(train[~te]), train[te])
        self.model = self._fit(train)
        if len(years) == 1:
            oof = self._proba(self.model, train)
        return oof

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._proba(self.model, df)

    @staticmethod
    def evaluate(y_true, proba) -> dict:
        y_pred = proba.argmax(1)
        labels = list(range(len(PRIMARY)))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "per_class_f1": dict(zip(PRIMARY, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).round(3).tolist())),
            "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classes": PRIMARY,
        }
