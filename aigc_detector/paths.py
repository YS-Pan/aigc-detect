"""User-selected model storage and writable state, independent of the package."""

import json
import os
from pathlib import Path


class PreparationRequired(RuntimeError):
    """Required local resources have not been prepared."""


def state_dir():
    configured = os.environ.get("AIGC_STATE_DIR")
    return (Path(configured).expanduser() if configured else Path.home() / ".local" / "share" / "aigc-detector").resolve()


def configured_model_dir():
    value = os.environ.get("AIGC_MODEL_DIR")
    if value:
        return Path(value).expanduser().resolve()
    config = state_dir() / "config.json"
    if config.is_file():
        return Path(json.loads(config.read_text(encoding="utf-8"))["model_dir"]).expanduser().resolve()
    return None


def model_root():
    # This unconfigured location is only inspected; it never authorizes downloads.
    return configured_model_dir() or state_dir() / "unconfigured-models"


def save_model_dir(path):
    path = Path(path).expanduser().resolve()
    config = state_dir() / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    temp = config.with_suffix(".json.tmp")
    temp.write_text(json.dumps({"model_dir": str(path)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(config)
    return path
