"""REST API.  Run:  MONSOONPP_CONFIG=configs/synthetic.yaml uvicorn monsoonpp.api.app:app --port 8000"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import load_config
from ..pipeline import MODEL_ORDER
from ..products.service import PRODUCT_MODEL, ForecastService


def _clean(o):
    """NaN/inf are not valid JSON (e.g. FAR with no forecast events) -> null."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):          # numpy scalar
        return _clean(o.item())
    return o


class SafeJSON(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_clean(content), separators=(",", ":")).encode()


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path or os.environ.get("MONSOONPP_CONFIG"))
    app = FastAPI(title="Regime-aware monsoon rainfall post-processing", version="0.1.0",
                  description="Raw NWP -> regime -> corrected rainfall, heavy-rain probabilities, districts, verification.",
                  default_response_class=SafeJSON)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    @lru_cache(maxsize=1)
    def svc() -> ForecastService:
        return ForecastService(cfg)

    def guard(fn, *a, **k):
        try:
            return fn(*a, **k)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))

    @app.get("/health")
    def health():
        return {"status": "ok", "run": cfg.name, "forecast_source": cfg.forecast_source,
                "models_ready": (cfg.model_dir / "predictor.joblib").exists()}

    @app.get("/model-card")
    def model_card():
        return json.loads((cfg.model_dir / "model_card.json").read_text())

    @app.get("/available")
    def available():
        s = svc()
        return {"leads": s.leads, "first_date": str(s.dates[0].date()), "last_date": str(s.dates[-1].date()),
                "n_dates": len(s.dates), "models": MODEL_ORDER, "product_model": PRODUCT_MODEL,
                "test_years": cfg.split.test_years}

    @app.get("/forecast/grid")
    def grid(date: str, lead: int = 1, model: str = PRODUCT_MODEL):
        return guard(svc().grid, date, lead, model)

    @app.get("/forecast/districts")
    def districts(date: str, lead: int = 1, model: str = PRODUCT_MODEL,
                  warning: str | None = Query(None, pattern="^(red|orange|yellow|green)$")):
        df = guard(svc().districts, date, lead, model)
        if warning:
            df = df[df.warning == warning]
        return {"date": date, "lead_days": lead, "model": model, "n": len(df), "districts": df.round(3).to_dict("records")}

    @app.get("/forecast/point")
    def point(lat: float, lon: float, date: str, lead: int = 1):
        return guard(svc().point, lat, lon, date, lead)

    @app.get("/regimes")
    def regimes(date: str, lead: int = 1):
        return guard(svc().regimes, date, lead)

    @app.get("/verification")
    def verification(split: str = "test", section: str = "headline"):
        p = cfg.report_dir / f"verification_{split}.json"
        if not p.exists():
            raise HTTPException(404, "run `monsoonpp evaluate` first")
        res = json.loads(p.read_text())
        if section == "headline":
            from ..verify.report import headline
            return headline(res).round(4).to_dict("records")
        if section not in res:
            raise HTTPException(404, f"sections: headline, {', '.join(res)}")
        return res[section]

    @app.get("/atlas")
    def atlas(lead: int | None = None, regime: str | None = None):
        p = cfg.report_dir / "verification_test.json"
        if not p.exists():
            raise HTTPException(404, "run `monsoonpp evaluate` first")
        rows = json.loads(p.read_text())["atlas"]["table"]
        rows = [r for r in rows if (lead is None or r["lead"] == lead) and (regime is None or r["regime"] == regime)]
        return {"n": len(rows), "rows": rows}

    return app


app = create_app() if os.environ.get("MONSOONPP_CONFIG") else None
