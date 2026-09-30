"""Correction ladder levels 0-2 (the bars any AI model must clear).

L0 raw NWP
L1 climatological multiplicative bias, per lead x grid cell
L2 empirical quantile mapping, per lead x grid cell
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RawModel:
    name = "L0_raw"

    def fit(self, df):
        return self

    def predict(self, df):
        return df["tp"].values.astype("float32")


class ClimBias:
    name = "L1_climbias"

    def __init__(self, pseudo_mm: float = 50.0, clip=(0.2, 5.0)):
        self.a, self.clip = pseudo_mm, clip
        self.table: pd.Series | None = None

    def fit(self, df):
        g = df.groupby(["lead", "iy", "ix"])
        r = (g["obs"].sum() + self.a) / (g["tp"].sum() + self.a)
        self.table = r.clip(*self.clip).rename("ratio")
        return self

    def predict(self, df):
        key = pd.MultiIndex.from_arrays([df["lead"], df["iy"], df["ix"]])
        r = self.table.reindex(key).fillna(1.0).values
        return (df["tp"].values * r).astype("float32")


class QuantileMapping:
    name = "L2_qmap"
    Q = np.linspace(0, 1, 101)

    def fit(self, df):
        self.maps = {}
        for (L, iy, ix), g in df.groupby(["lead", "iy", "ix"]):
            self.maps[(L, iy, ix)] = (np.quantile(g["tp"].values, self.Q), np.quantile(g["obs"].values, self.Q))
        return self

    def predict(self, df):
        out = df["tp"].values.astype("float64").copy()
        for (L, iy, ix), idx in df.groupby(["lead", "iy", "ix"]).indices.items():
            m = self.maps.get((L, iy, ix))
            if m is None:
                continue
            qf, qo = m
            x = out[idx]
            # interpolate on unique forecast quantiles; linear extrapolation above top via additive shift
            qf_u, keep = np.unique(qf, return_index=True)
            y = np.interp(x, qf_u, qo[keep])
            top = x > qf[-1]
            y[top] = x[top] + (qo[-1] - qf[-1])
            out[idx] = np.clip(y, 0, None)
        return out.astype("float32")
