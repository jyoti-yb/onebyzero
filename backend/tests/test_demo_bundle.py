"""Serving bundles are complete, tamper-evident, and runtime-only."""
import json

import pytest
from fastapi.testclient import TestClient

from monsoonpp.api.app import create_app
from monsoonpp.serving.bundle import BundleError, DemoBundle, write_bundle


def _payloads():
    return {
        "status": {"status": "ready", "network_required_at_runtime": False},
        "cycle": {"forecast_label": "2024-07-31", "rain": [[1.0]]},
        "regimes": {"forecast_label": "2024-07-31", "synoptic_classes": ["none"]},
        "verification": {"decision": "DO_NOT_PROCEED_TO_ML"},
    }


def test_bundle_api_exposes_only_precomputed_payloads(tmp_path):
    path = write_bundle(tmp_path / "bundle", _payloads(), {"source_run": "fixture"})
    client = TestClient(create_app(path))
    assert client.get("/health").json()["network_required"] is False
    assert client.get("/api/v1/status").json()["status"] == "ready"
    assert client.get("/api/v1/cycles/latest").json()["rain"] == [[1.0]]
    assert client.get("/api/v1/regimes/latest").json()["synoptic_classes"] == ["none"]
    assert client.get("/api/v1/verification/latest").json()["decision"] == "DO_NOT_PROCEED_TO_ML"
    assert client.get("/forecast/grid").status_code == 404


def test_missing_or_tampered_bundle_fails_at_startup(tmp_path):
    with pytest.raises(BundleError, match="directory is missing"):
        DemoBundle(tmp_path / "absent")
    path = write_bundle(tmp_path / "bundle", _payloads(), {"source_run": "fixture"})
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    cycle = path / manifest["payloads"]["cycle"]
    cycle.write_bytes(cycle.read_bytes() + b"tamper")
    with pytest.raises(BundleError, match="checksum validation"):
        create_app(path)


def test_bundle_export_refuses_partial_payload_set(tmp_path):
    payloads = _payloads()
    payloads.pop("verification")
    with pytest.raises(BundleError, match=r"missing=\['verification'\]"):
        write_bundle(tmp_path / "bundle", payloads, {})
