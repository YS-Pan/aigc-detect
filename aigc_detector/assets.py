"""Offline existence/size and complete shard-index checks (SHA verification is separate)."""

import json
from pathlib import Path, PurePosixPath


def assets_installed(directory, entry):
    directory = Path(directory)
    files = entry.get("files", {})
    if not entry.get("installed") or "config.json" not in files:
        return False
    if not all(PurePosixPath(name).name == name and "\\" not in name
               and (directory / name).is_file() and (directory / name).stat().st_size == meta["bytes"]
               for name, meta in files.items()):
        return False
    if any(name in files for name in ("pytorch_model.bin", "model.safetensors")):
        return True
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if index in files:
            try:
                mapping = json.loads((directory / index).read_text(encoding="utf-8"))["weight_map"]
                shards = set(mapping.values())
            except (ValueError, OSError, KeyError, TypeError):
                return False
            return bool(mapping and shards) and all(isinstance(name, str) and PurePosixPath(name).name == name
                                                    and "\\" not in name and name in files for name in shards)
    return False
