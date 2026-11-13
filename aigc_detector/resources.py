"""Inspect and verify local assets without importing inference dependencies."""

import hashlib
from importlib.resources import files
import json

from .assets import assets_installed
from .catalog import ASSET_CATALOG, CATALOG, MODEL_GROUPS, STATISTICAL_BACKENDS


def manifest(root):
    path = root / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")).get("models", {}) if path.is_file() else {}


def missing(root, selected):
    entries = manifest(root)
    return [mid for mid in dict.fromkeys(selected)
            if not assets_installed(root / mid, entries.get(mid, {}))]


def describe(selected):
    bundled = json.loads(files("aigc_detector").joinpath("resource_manifest.json").read_text(encoding="utf-8"))["models"]
    return [(mid, sum(item["bytes"] for item in bundled[mid]["files"].values())) for mid in selected]


def selected_assets(*, models=None, language=None, annotation=None, backends=None):
    if models:
        unknown = set(models) - ASSET_CATALOG.keys()
        if unknown:
            raise ValueError(f"Unknown model IDs: {sorted(unknown)}")
        return list(dict.fromkeys(models))
    if backends:
        return list(dict.fromkeys(mid for backend in backends for mid in STATISTICAL_BACKENDS[backend]))
    if annotation:
        from .fusion import load_profile, resolve_policy
        profile, _ = load_profile(resolve_policy(None, annotation))
        return list(profile["revisions"])
    return list(MODEL_GROUPS[language or "zh"])


def inventory(root):
    entries = manifest(root)
    models = [{"id": key, **spec, "revision": entries.get(key, {}).get("revision"),
               "installed": assets_installed(root / key, entries.get(key, {})), "loaded": False,
               "max_tokens": spec.get("max_tokens", 512)} for key, spec in CATALOG.items()]
    backends = [{"id": key, "assets": list(ids),
                 "installed": not missing(root, ids),
                 "score_direction": "lower_is_more_ai_like" if key.startswith("binoculars") else "higher_is_more_ai_like",
                 "calibrated": False} for key, ids in STATISTICAL_BACKENDS.items()]
    return models, backends


def verify(root, selected):
    entries = manifest(root)
    for mid in dict.fromkeys(selected):
        entry = entries.get(mid, {})
        verify_entry(root, mid, entry)
        print(f"PASS {mid}: {len(entry['files'])} files, {entry['revision']}")


def verify_entry(root, mid, entry):
    if not assets_installed(root / mid, entry):
        raise ValueError(f"Missing or incomplete asset: {mid}")
    for filename, expected in entry["files"].items():
        path = root / mid / filename
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
