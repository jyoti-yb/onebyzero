"""Versioned, checksummed demo bundles with no scientific-runtime imports."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1
REQUIRED_PAYLOADS = ("status", "cycle", "regimes", "verification")


class BundleError(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(destination: str | Path, payloads: dict[str, Any], metadata: dict[str, Any]) -> Path:
    """Atomically write a complete immutable bundle directory."""
    missing = sorted(set(REQUIRED_PAYLOADS) - set(payloads))
    extra = sorted(set(payloads) - set(REQUIRED_PAYLOADS))
    if missing or extra:
        raise BundleError(f"bundle payload mismatch; missing={missing}, extra={extra}")
    destination = Path(destination)
    if destination.exists():
        raise BundleError(f"bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    try:
        (temporary / "payloads").mkdir(parents=True)
        files = {}
        payload_index = {}
        for name in REQUIRED_PAYLOADS:
            relative = Path("payloads") / f"{name}.json.gz"
            path = temporary / relative
            path.write_bytes(gzip.compress(_json_bytes(payloads[name]), compresslevel=9, mtime=0))
            payload_index[name] = relative.as_posix()
            files[relative.as_posix()] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        identity = hashlib.sha256("".join(files[path]["sha256"] for path in sorted(files)).encode()).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "bundle_id": identity[:16],
            "payloads": payload_index,
            "files": files,
            "metadata": metadata,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


class DemoBundle:
    """Eagerly validate and load a serving bundle at process startup."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise BundleError(f"demo bundle directory is missing: {self.root}")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise BundleError(f"required bundle file is missing: {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError(f"cannot read bundle manifest {manifest_path}: {exc}") from exc
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(
                f"unsupported bundle schema {self.manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        payload_index = self.manifest.get("payloads", {})
        if set(payload_index) != set(REQUIRED_PAYLOADS):
            raise BundleError(f"manifest must define payloads {list(REQUIRED_PAYLOADS)}")
        declared_files = self.manifest.get("files", {})
        self.payloads = {}
        for name in REQUIRED_PAYLOADS:
            relative = Path(payload_index[name])
            path = (self.root / relative).resolve()
            if relative.is_absolute() or self.root not in path.parents:
                raise BundleError(f"bundle payload path escapes bundle root: {relative}")
            if not path.is_file():
                raise BundleError(f"required bundle file is missing: {path}")
            facts = declared_files.get(relative.as_posix())
            if not facts:
                raise BundleError(f"bundle manifest has no checksum for {relative.as_posix()}")
            if path.stat().st_size != int(facts.get("bytes", -1)) or _sha256(path) != facts.get("sha256"):
                raise BundleError(f"bundle file failed size/checksum validation: {path}")
            try:
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    value = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise BundleError(f"cannot decode bundle payload {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise BundleError(f"bundle payload {name!r} must be a JSON object")
            self.payloads[name] = value

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    def get(self, name: str) -> dict:
        return self.payloads[name]
