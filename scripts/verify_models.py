"""Verify every installed asset against the recorded SHA-256 manifest offline."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.catalog import CATALOG, MODEL_ROOT


def main():
    manifest = json.loads((MODEL_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for model_id in CATALOG:
        entry = manifest["models"][model_id]
        assert entry["installed"], model_id
        for filename, expected in entry["files"].items():
            path = MODEL_ROOT / model_id / filename
            assert path.stat().st_size == expected["bytes"], path
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == expected["sha256"], path
        print(f"PASS {model_id}: {len(entry['files'])} files, {entry['revision']}")


if __name__ == "__main__":
    main()
