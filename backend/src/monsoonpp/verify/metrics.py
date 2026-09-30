"""Forecast verification metrics (WMO/JWGFVR definitions).

Continuous : bias, MAE, RMSE, correlation
Categorical: POD, FAR, CSI, ETS (Gilbert skill score), frequency bias — per threshold
Spatial    : FSS (Roberts & Lean 2008) at several neighbourhood scales
Probability: Brier score, Brier skill score vs sample climatology, reliability, ROC AUC
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import roc_auc_score


def continuous(f: np.ndarray, o: np.ndarray) -> dict:
    e = f - o
    return {
        "n": int(len(o)),
        "bias": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "corr": float(np.corrcoef(f, o)[0, 1]) if len(o) > 2 and f.std() > 0 and o.std() > 0 else float("nan"),
        "mean_fc": float(f.mean()),
        "mean_obs": float(o.mean()),
    }


def contingency(f: np.ndarray, o: np.ndarray, thr: float) -> dict:
    fe, oe = f >= thr, o >= thr
    a = int(np.sum(fe & oe))      # hits
    b = int(np.sum(fe & ~oe))     # false alarms
    c = int(np.sum(~fe & oe))     # misses
    d = int(np.sum(~fe & ~oe))    # correct negatives
    n = a + b + c + d
    nan = float("nan")
    ar = (a + b) * (a + c) / n if n else nan
    return {
        "threshold": thr, "hits": a, "false_alarms": b, "misses": c, "correct_neg": d,
        "pod": a / (a + c) if (a + c) else nan,
        "far": b / (a + b) if (a + b) else nan,
        "csi": a / (a + b + c) if (a + b + c) else nan,
        "ets": (a - ar) / (a + b + c - ar) if (a + b + c - ar) else nan,
        "freq_bias": (a + b) / (a + c) if (a + c) else nan,
        "base_rate": (a + c) / n if n else nan,
    }


def fss(fc: np.ndarray, ob: np.ndarray, mask: np.ndarray, thr: float, scale: int) -> float:
    """FSS aggregated over all fields. fc, ob: (n, lat, lon). mask: (lat, lon) valid cells.
    Fractions are computed over valid cells only (masked normalisation)."""
    m = mask.astype("float64")
    msum = uniform_filter(m, scale, mode="constant")
    num = den = 0.0
    for t in range(fc.shape[0]):
        bf = ((fc[t] >= thr) & mask).astype("float64")
        bo = ((np.nan_to_num(ob[t]) >= thr) & mask).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            pf = np.where(msum > 0, uniform_filter(bf, scale, mode="constant") / msum, 0)[mask]
            po = np.where(msum > 0, uniform_filter(bo, scale, mode="constant") / msum, 0)[mask]
        num += np.sum((pf - po) ** 2)
        den += np.sum(pf ** 2) + np.sum(po ** 2)
    return float(1 - num / den) if den > 0 else float("nan")


def fss_useful(base_rate: float) -> float:
    """FSS value regarded as 'useful skill': 0.5 + f0/2."""
    return 0.5 + base_rate / 2


def probabilistic(p: np.ndarray, o_event: np.ndarray, n_bins: int = 10) -> dict:
    y = o_event.astype(float)
    bs = float(np.mean((p - y) ** 2))
    clim = y.mean()
    bs_ref = float(np.mean((clim - y) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rel = []
    for k in range(n_bins):
        s = idx == k
        if s.sum():
            rel.append({"bin_mid": float((bins[k] + bins[k + 1]) / 2), "n": int(s.sum()),
                        "mean_p": float(p[s].mean()), "obs_freq": float(y[s].mean())})
    try:
        auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    except ValueError:
        auc = float("nan")
    return {"brier": bs, "brier_clim": bs_ref, "bss": 1 - bs / bs_ref if bs_ref > 0 else float("nan"),
            "roc_auc": auc, "base_rate": float(clim), "reliability": rel}
