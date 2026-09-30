"""End-to-end on a tiny synthetic run: build -> train -> verify -> serve."""
import numpy as np
import pandas as pd
import pytest

from monsoonpp.features import ALL_FEATURES, BASE_FEATURES
from monsoonpp.pipeline import MODEL_ORDER


def test_no_obs_or_analysis_in_predictors():
    for f in ALL_FEATURES + BASE_FEATURES:
        assert not f.startswith(("obs", "an_", "label"))


def test_pipeline_outputs(trained):
    cfg, res = trained
    pred = pd.read_parquet(cfg.product_dir / "predictions.parquet")
    assert set(pred.split) == {"val", "test"}
    assert set(pred[pred.split == "test"].year) == set(cfg.split.test_years)
    for m in MODEL_ORDER:
        assert m in pred and (pred[m] >= 0).all()
    assert {"p_gt_64.5", "p_gt_115.6"} <= set(pred.columns)
    assert pred["p_gt_64.5"].between(0, 1).all()
    assert (cfg.report_dir / "report_test.md").exists()
    assert (cfg.report_dir / "error_atlas.nc").exists()
    assert res["fss"] and res["atlas"]["table"]


def test_ml_beats_raw_rmse(trained):
    cfg, res = trained
    ov = pd.DataFrame(res["overall"])
    for L, g in ov.groupby("lead"):
        g = g.set_index("model")
        assert g.loc["L4_gbm_regime", "rmse"] < g.loc["L0_raw", "rmse"]


def test_operational_path_matches_offline(trained):
    """Serving from forecast-only data must reproduce the offline test predictions."""
    from monsoonpp.products.service import ForecastService
    cfg, _ = trained
    svc = ForecastService(cfg)
    pred = pd.read_parquet(cfg.product_dir / "predictions.parquet")
    t = pred[pred.split == "test"]
    day = pd.Timestamp(t.time.iloc[0])
    lead = int(t.lead.iloc[0])
    online = svc.run(str(day.date()), lead).set_index(["iy", "ix"])
    off = t[(t.time == day) & (t.lead == lead)].set_index(["iy", "ix"])
    j = off.join(online[["L6_regime_qm", "p_gt_64.5"]], rsuffix="_on")
    assert np.allclose(j["L6_regime_qm"], j["L6_regime_qm_on"], atol=1e-3)
    assert np.allclose(j["p_gt_64.5"], j["p_gt_64.5_on"], atol=1e-4)


def test_api(trained):
    from fastapi.testclient import TestClient
    from monsoonpp.api import app as appmod
    from monsoonpp.export_demo import export_demo
    cfg, _ = trained
    bundle = cfg.run_dir / "demo-bundle"
    export_demo(cfg, bundle)
    client = TestClient(appmod.create_app(bundle))
    health = client.get("/health").json()
    assert health["status"] == "ok" and not health["network_required"]
    status = client.get("/api/v1/status").json()
    assert status["bundle_mode"] == "trained_demo" and not status["raw_data_in_bundle"]
    g = client.get("/api/v1/cycles/latest").json()
    assert len(g["rain"]) == len(g["lat"])
    assert client.get("/api/v1/regimes/latest").status_code == 200
    verification = client.get("/api/v1/verification/latest").json()
    assert verification["summary"]["atlas"]["table"]
