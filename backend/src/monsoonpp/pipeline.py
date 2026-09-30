"""End-to-end orchestration: build -> train -> predict -> verify -> products."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from .config import Config
from .data.build import build_dataset, load_dataset
from .data.obs_time import require_verified
from .features import ALL_FEATURES, BASE_FEATURES, RegimeBundle, build_table
from .models.baselines import ClimBias, QuantileMapping, RawModel
from .models.gbm import MixtureOfExperts, RainGBM
from .models.probability import ExceedanceHead
from .models.restore import RegimeQuantileRestore
from .regimes.classifier import PROB_COLS, RegimeClassifier
from .regimes.engine import PRIMARY

log = logging.getLogger(__name__)

MODEL_ORDER = ["L0_raw", "L1_climbias", "L2_qmap", "L3_gbm_generic", "L4_gbm_regime", "L5_moe", "L6_regime_qm"]
REGIME_AWARE = ALL_FEATURES + PROB_COLS


class Predictor:
    """Everything needed at inference, versioned together (the model registry unit)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bundle: RegimeBundle | None = None
        self.clf: RegimeClassifier | None = None
        self.baselines = [RawModel(), ClimBias(), QuantileMapping()]
        self.generic: RainGBM | None = None
        self.regime: RainGBM | None = None
        self.moe: MixtureOfExperts | None = None
        self.heads: list[ExceedanceHead] = []
        self.restore: RegimeQuantileRestore | None = None
        self.meta: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, train_mask: np.ndarray, val_mask: np.ndarray):
        cfg, mc = self.cfg, self.cfg.model
        t0 = time.time()
        tr, va = df[train_mask].copy(), df[val_mask].copy()
        self.clf = RegimeClassifier(seed=cfg.seed)
        tr[PROB_COLS] = self.clf.fit_crossfit(tr)
        va[PROB_COLS] = self.clf.predict_proba(va)
        log.info("regime classifier done (%.0fs)", time.time() - t0)

        fit_tr = tr.sample(mc.max_train_rows, random_state=cfg.seed) if mc.max_train_rows and len(tr) > mc.max_train_rows else tr
        for b in self.baselines:
            b.fit(tr)
        self.generic = RainGBM("L3_gbm_generic", BASE_FEATURES, mc, cfg.seed).fit(fit_tr, va)
        log.info("L3 done (%.0fs)", time.time() - t0)
        self.regime = RainGBM("L4_gbm_regime", REGIME_AWARE, mc, cfg.seed).fit(fit_tr, va)
        log.info("L4 done (%.0fs)", time.time() - t0)
        self.moe = MixtureOfExperts(REGIME_AWARE, mc, self.regime, cfg.seed).fit(fit_tr, va)
        log.info("L5 done (%.0fs)", time.time() - t0)
        self.heads = []
        for thr in cfg.prob_thresholds:
            try:
                self.heads.append(ExceedanceHead(thr, REGIME_AWARE, cfg.seed).fit(fit_tr, va))
            except ValueError as e:
                log.warning("skip head %s: %s", thr, e)
        log.info("probability heads done (%.0fs)", time.time() - t0)
        self.restore = RegimeQuantileRestore("L4_gbm_regime").fit(
            self.regime.predict(va), va["obs"].values, va["lead"].values, va[PROB_COLS].values.argmax(1))
        self.meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "forecast_source": cfg.forecast_source, "nwp_model_version": cfg.nwp_model_version,
            "obs_source": cfg.obs_source, "train_years": cfg.split.train_years, "val_years": cfg.split.val_years,
            "leads": cfg.leads, "n_train_rows": int(len(tr)), "features_regime_aware": REGIME_AWARE,
            "moe_expert_weights": {PRIMARY[k]: round(v, 3) for k, v in self.moe.weights.items()},
            "train_seconds": round(time.time() - t0, 1),
        }
        return tr, va

    # -------------------------------------------------------------- predict
    def predict_table(self, df: pd.DataFrame, have_probs: bool = False) -> pd.DataFrame:
        df = df.copy()
        if not have_probs:
            df[PROB_COLS] = self.clf.predict_proba(df)
        out = pd.DataFrame(index=df.index)
        for b in self.baselines:
            out[b.name] = b.predict(df)
        out["L3_gbm_generic"] = self.generic.predict(df)
        out["L4_gbm_regime"] = self.regime.predict(df)
        out["L5_moe"] = self.moe.predict(df)
        regime_pred = df[PROB_COLS].values.argmax(1)
        if self.restore is not None:
            out["L6_regime_qm"] = self.restore.predict(out[self.restore.base_col].values, df["lead"].values, regime_pred)
        for h in self.heads:
            out[h.name] = h.predict(df)
        out["regime_pred"] = regime_pred.astype("int8")
        out["regime_conf"] = df[PROB_COLS].values.max(1).astype("float32")
        for c in PROB_COLS:
            out[c] = df[c].values
        return out

    def forecast_day(self, ds_fc: xr.Dataset, lead: int) -> pd.DataFrame:
        """Operational path: forecast-only dataset (no obs) -> corrected table."""
        sub = ds_fc.sel(lead=[lead])
        df = build_table(sub, self.bundle, labels=None)
        return pd.concat([df, self.predict_table(df)], axis=1)

    # ------------------------------------------------------------- persist
    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "predictor.joblib", compress=3)
        (path / "model_card.json").write_text(json.dumps(self.meta, indent=2))

    @staticmethod
    def load(path) -> "Predictor":
        return joblib.load(path / "predictor.joblib")


def split_masks(df: pd.DataFrame, cfg: Config):
    y = df["year"].values
    return (np.isin(y, cfg.split.train_years), np.isin(y, cfg.split.val_years), np.isin(y, cfg.split.test_years))


def run_train(cfg: Config, rebuild: bool = False) -> dict:
    require_verified(cfg.obs_time_convention, "train model", cfg.obs_source)
    if rebuild or not cfg.dataset_path.exists():
        build_dataset(cfg)
    ds = load_dataset(cfg, "train model")
    years = pd.DatetimeIndex(ds.time.values).year
    train_times = ds.time.values[np.isin(years, cfg.split.train_years)]

    bundle = RegimeBundle().fit(ds, train_times)
    labels = bundle.labels(ds)
    df = build_table(ds, bundle, labels)
    log.info("feature table: %d rows x %d cols", *df.shape)
    trm, vam, tem = split_masks(df, cfg)

    pred = Predictor(cfg)
    pred.bundle = bundle
    tr, va = pred.fit(df, trm, vam)

    # predictions for val + test (train rows use cross-fitted regime probs)
    te = df[tem]
    out_va = pd.concat([va, pred.predict_table(va, have_probs=True).drop(columns=PROB_COLS)], axis=1)
    out_te = pd.concat([te, pred.predict_table(te)], axis=1)
    out = pd.concat([out_va.assign(split="val"), out_te.assign(split="test")], ignore_index=True)

    cfg.product_dir.mkdir(parents=True, exist_ok=True)
    keep = ["lead", "time", "iy", "ix", "lat", "lon", "year", "region", "obs", "tp", "label_primary",
            "label_state", "fc_primary", "split", *MODEL_ORDER, *[h.name for h in pred.heads],
            "regime_pred", "regime_conf", *PROB_COLS, "tcwv", "cape", "u850", "oro_score", "monsoon_z"]
    out[keep].to_parquet(cfg.product_dir / "predictions.parquet", index=False)
    labels.to_netcdf(cfg.product_dir / "regime_labels.nc")

    clf_eval = {"val": RegimeClassifier.evaluate(va["label_primary"].values, va[PROB_COLS].values),
                "test": RegimeClassifier.evaluate(te["label_primary"].values, out_te[PROB_COLS].values)}
    pred.meta["regime_classifier"] = {k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in clf_eval.items()}
    pred.meta["importance_L4"] = dict(list(pred.regime.importance().items())[:15])
    pred.save(cfg.model_dir)
    (cfg.product_dir / "regime_classifier_eval.json").write_text(json.dumps(clf_eval, indent=2))
    # keep a training archive slice for analogue retrieval (explanations)
    arch_cols = [*REGIME_AWARE, "obs", "label_primary", "time", "iy", "ix", "year"]
    tr[arch_cols].to_parquet(cfg.product_dir / "train_archive.parquet", index=False)
    return pred.meta
