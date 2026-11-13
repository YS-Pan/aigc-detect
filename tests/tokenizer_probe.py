"""Compare locally installed tokenizer assets between Python environments.

No model weights are loaded. From the source tree in PowerShell, run in each
environment after setting $env:PYTHONPATH = (Get-Location).Path:
  python tests/tokenizer_probe.py <model-directory> <output-json>
"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, logging
from aigc_detector.catalog import CATALOG
from aigc_detector.engine import prepare_window, window_ranges

logging.set_verbosity_error()
root, output = map(Path, sys.argv[1:])
result = {}
texts = ["Hello world, 学术论文。", ("中英文 text with whitespace. " * 30)]
for model_id in CATALOG:
    try:
        tokenizer = AutoTokenizer.from_pretrained(root / model_id, local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        cases = []
        for text in texts:
            encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True, verbose=False)
            ids = encoded["input_ids"]
            ranges = list(window_ranges(len(ids), 32, 8))
            cases.append({"ids": ids, "offsets": encoded["offset_mapping"], "ranges": ranges,
                          "features": [dict(prepare_window(tokenizer, ids[a:b], model_id)) for a, b in ranges]})
        result[model_id] = {"class": type(tokenizer).__name__, "cases": cases}
    except Exception as exc:
        result[model_id] = {"error": f"{type(exc).__name__}: {exc}"}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {len(result)} tokenizer results to {output}")
