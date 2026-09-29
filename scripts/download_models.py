"""Fetch pinned public model snapshots. No model repository code is executed."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.catalog import CATALOG, MODEL_ROOT
from scripts.network import EVENTS, download, get_json

TOKENIZER_FILES = {"vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "sentencepiece.bpe.model", "spiece.model"}
ALLOW = TOKENIZER_FILES | {"README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "config.json",
                           "pytorch_model.bin", "model.safetensors"}


def model_assets(metadata, subfolder=""):
    """Flatten only the selected directory; never fetch other checkpoints."""
    prefix = subfolder.rstrip("/") + "/" if subfolder else ""
    assets = []
    for item in metadata["siblings"]:
        path = item["rfilename"]
        if path.startswith(prefix) and path[len(prefix):] in ALLOW:
            assets.append((path[len(prefix):], item))
    if any(name == "model.safetensors" for name, _ in assets):
        assets = [(name, item) for name, item in assets if name != "pytorch_model.bin"]
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(CATALOG), help="Only these IDs; default: all")
    args = parser.parse_args()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = MODEL_ROOT / "manifest.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"models": {}}
    old_events = lock.get("network_events", [])
    metadata_cache = {}

    def metadata(repo, revision):
        key = (repo, revision)
        if key not in metadata_cache:
            metadata_cache[key] = get_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
        return metadata_cache[key]

    for model_id in dict.fromkeys(args.models or CATALOG):
        spec = CATALOG[model_id]
        previous = lock["models"].get(model_id, {})
        meta = metadata(spec["repo_id"], previous.get("revision", spec.get("revision", "main")))
        revision = meta["sha"]
        entry = {**spec, "revision": revision, "downloaded_at": datetime.now(timezone.utc).isoformat(), "files": {}}
        assets = [(name, item, spec["repo_id"], revision)
                  for name, item in model_assets(meta, spec.get("subfolder", ""))]
        if spec.get("tokenizer_repo_id"):
            repo = spec["tokenizer_repo_id"]
            tok_meta = metadata(repo, previous.get("tokenizer_revision", spec["tokenizer_revision"]))
            entry["tokenizer_revision"] = tok_meta["sha"]
            for name, item in model_assets(tok_meta):
                if name in TOKENIZER_FILES:
                    assets.append((name, item, repo, tok_meta["sha"]))
                elif name == "README.md":
                    assets.append(("tokenizer-source-README.md", item, repo, tok_meta["sha"]))
        names = {name for name, *_ in assets}
        if "config.json" not in names or not names.intersection({"pytorch_model.bin", "model.safetensors"}):
            raise RuntimeError(f"Missing config or supported weights for {model_id}")
        if not names.intersection({"tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model"}):
            raise RuntimeError(f"Missing tokenizer for {model_id}")
        print(f"Downloading {model_id} at {revision}", flush=True)
        for name, item, repo, source_revision in assets:
            assert PurePosixPath(name).name == name
            source_path = item["rfilename"]
            url = f"https://huggingface.co/{repo}/resolve/{source_revision}/{source_path}"
            result = download(url, MODEL_ROOT / model_id / name, item.get("lfs", {}).get("sha256"), item.get("size"))
            entry["files"][name] = {**result, "source": {"repo_id": repo, "revision": source_revision, "path": source_path}}
        entry["installed"] = True
        lock["models"][model_id] = entry
        lock["network_events"] = old_events + EVENTS
        temporary = lock_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(lock_path)
    print("Selected models downloaded and hashes verified.", flush=True)


if __name__ == "__main__":
    main()
