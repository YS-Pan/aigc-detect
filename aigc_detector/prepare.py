"""Fetch pinned public model snapshots. No model repository code is executed."""

import json
import re
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import PurePosixPath

from .catalog import ASSET_CATALOG
from .assets import assets_installed
from .network import EVENTS, download, get_json
from .resources import verify_entry

TOKENIZER_FILES = {"vocab.txt", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "sentencepiece.bpe.model", "spiece.model", "spm.model"}
ALLOW = TOKENIZER_FILES | {"README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "config.json",
                           "pytorch_model.bin", "model.safetensors", "model.safetensors.index.json",
                           "pytorch_model.bin.index.json"}
SHARD_NAME = re.compile(r"(?:model-\d{5}-of-\d{5}\.safetensors|pytorch_model-\d{5}-of-\d{5}\.bin)\Z")


def model_assets(metadata, subfolder="", *, weight_file=None):
    """Flatten only the selected directory; never fetch other checkpoints."""
    prefix = subfolder.rstrip("/") + "/" if subfolder else ""
    assets = []
    for item in metadata["siblings"]:
        path = item["rfilename"]
        name = path[len(prefix):]
        if path.startswith(prefix) and (name in ALLOW or SHARD_NAME.fullmatch(name) or name == weight_file):
            assets.append(("pytorch_model.bin" if weight_file and name == weight_file else name, item))
    names = {name for name, _ in assets}
    if "model.safetensors" in names:
        assets = [(name, item) for name, item in assets if not name.endswith(".bin")
                  and name not in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
                  and not SHARD_NAME.fullmatch(name)]
    elif "model.safetensors.index.json" in names:
        assets = [(name, item) for name, item in assets if not name.endswith(".bin")
                  and name != "pytorch_model.bin.index.json"]
    elif "pytorch_model.bin" in names:
        assets = [(name, item) for name, item in assets if not SHARD_NAME.fullmatch(name)
                  and not name.endswith(".index.json")]
    return assets


def prepare(model_root, selected, *, download_workers=1):
    """Prepare only requested missing assets in a user-selected directory."""
    model_root.mkdir(parents=True, exist_ok=True)
    lock_path = model_root / "manifest.json"
    bundled = json.loads(files("aigc_detector").joinpath("resource_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"models": {}}
    old_events = lock.get("network_events", [])
    metadata_cache = {}

    def metadata(repo, revision, repo_type="model"):
        key = (repo_type, repo, revision)
        if key not in metadata_cache:
            plural = {"model": "models", "space": "spaces"}[repo_type]
            metadata_cache[key] = get_json(f"https://huggingface.co/api/{plural}/{repo}/revision/{revision}?blobs=true")
        return metadata_cache[key]

    for model_id in dict.fromkeys(selected):
        spec = ASSET_CATALOG[model_id]
        previous = lock["models"].get(model_id, bundled["models"].get(model_id, {}))
        if assets_installed(model_root / model_id, previous):
            verify_entry(model_root, model_id, previous)
            if model_id not in lock["models"]:
                lock["models"][model_id] = previous
                lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Already installed: {model_id}", flush=True)
            continue
        repo_type = spec.get("repo_type", "model")
        meta = metadata(spec["repo_id"], previous.get("revision", spec.get("revision", "main")), repo_type)
        revision = meta["sha"]
        entry = {**spec, "revision": revision, "downloaded_at": datetime.now(timezone.utc).isoformat(), "files": {}}
        assets = [(name, item, spec["repo_id"], revision, repo_type)
                  for name, item in model_assets(meta, spec.get("subfolder", ""), weight_file=spec.get("weight_file"))]
        if spec.get("subfolder"):
            for name, item in model_assets(meta):
                if name in {"README.md", "LICENSE", "LICENSE.md", "LICENSE.txt"}:
                    assets.append(("repository-" + name, item, spec["repo_id"], revision, repo_type))
        if spec.get("tokenizer_repo_id"):
            repo = spec["tokenizer_repo_id"]
            tok_meta = metadata(repo, previous.get("tokenizer_revision", spec["tokenizer_revision"]))
            entry["tokenizer_revision"] = tok_meta["sha"]
            for name, item in model_assets(tok_meta):
                if name in TOKENIZER_FILES or (name == "config.json" and spec.get("config_from_tokenizer_repo")):
                    assets.append((name, item, repo, tok_meta["sha"], "model"))
                elif name == "README.md":
                    assets.append(("tokenizer-source-README.md", item, repo, tok_meta["sha"], "model"))
        names = {name for name, *_ in assets}
        if "config.json" not in names or not names.intersection({"pytorch_model.bin", "model.safetensors",
                                                               "model.safetensors.index.json", "pytorch_model.bin.index.json"}):
            raise RuntimeError(f"Missing config or supported weights for {model_id}")
        if not names.intersection({"tokenizer.json", "vocab.txt", "vocab.json", "sentencepiece.bpe.model", "spiece.model", "spm.model"}):
            raise RuntimeError(f"Missing tokenizer for {model_id}")
        print(f"Downloading {model_id} at {revision}", flush=True)
        for name, item, repo, source_revision, source_type in assets:
            assert PurePosixPath(name).name == name
            source_path = item["rfilename"]
            prefix = "spaces/" if source_type == "space" else ""
            url = f"https://huggingface.co/{prefix}{repo}/resolve/{source_revision}/{source_path}"
            known = previous.get("files", {}).get(name, {})
            result = download(url, model_root / model_id / name,
                              known.get("sha256") or item.get("lfs", {}).get("sha256"),
                              known.get("bytes", item.get("size")), workers=download_workers)
            entry["files"][name] = {**result, "source": {"repo_id": repo, "revision": source_revision, "path": source_path,
                                                       **({"repo_type": source_type} if source_type != "model" else {})}}
        entry["installed"] = True
        if not assets_installed(model_root / model_id, entry):
            raise RuntimeError(f"Incomplete or invalid weight shard index for {model_id}")
        lock["models"][model_id] = entry
        lock["network_events"] = old_events + EVENTS
        temporary = lock_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(lock_path)
    print("Selected models downloaded and hashes verified.", flush=True)
