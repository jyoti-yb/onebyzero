"""Bundle-only production/demo API with no rebuild-path imports."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..serving.bundle import BundleError, DemoBundle


def create_app(bundle_path: str | os.PathLike) -> FastAPI:
    bundle = DemoBundle(bundle_path)
    app = FastAPI(
        title="MonsoonPP serving bundle API",
        version="1.0.0",
        description="Read-only precomputed products. No NOAA/NASA access or rebuild pipeline is used.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
    app.state.bundle = bundle

    @app.get("/health")
    def health():
        return {
            "status": "ok", "bundle_id": bundle.bundle_id,
            "schema_version": bundle.manifest["schema_version"], "network_required": False,
        }

    @app.get("/api/v1/status")
    def status():
        return bundle.get("status")

    @app.get("/api/v1/cycles/latest")
    def latest_cycle():
        return bundle.get("cycle")

    @app.get("/api/v1/regimes/latest")
    def latest_regimes():
        return bundle.get("regimes")

    @app.get("/api/v1/verification/latest")
    def latest_verification():
        return bundle.get("verification")

    return app


def _missing_bundle_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        raise BundleError(
            "MONSOONPP_BUNDLE is required and must point to a complete bundle created by `monsoonpp export-demo`"
        )
        yield

    return FastAPI(title="MonsoonPP bundle missing", lifespan=lifespan)


app = create_app(os.environ["MONSOONPP_BUNDLE"]) if os.environ.get("MONSOONPP_BUNDLE") else _missing_bundle_app()
