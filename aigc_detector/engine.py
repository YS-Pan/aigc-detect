"""Offline inference and token-window coverage for Chinese detector models."""

import hashlib
import json
import math
import os
import threading
import time
from itertools import islice
from pathlib import Path

# Set before importing transformers/huggingface_hub. Loading also explicitly uses
# local_files_only, so an inherited environment override cannot cause downloads.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .catalog import CATALOG, MODEL_ROOT

MAX_TEXT_CHARS = 200_000
MAX_BATCH_CHARS = 500_000


def select_models(models):
    if models is None:
        return list(CATALOG)
    if not isinstance(models, (list, tuple)) or not models or not all(isinstance(item, str) for item in models):
        raise ValueError("models must be a non-empty list of model IDs, or null for all")
    selected = list(dict.fromkeys(models))
    unknown = set(selected) - CATALOG.keys()
    if unknown:
        raise ValueError(f"Unknown model IDs: {', '.join(sorted(unknown))}")
    return selected


class ModelUnavailable(RuntimeError):
    pass


def window_ranges(total, capacity, overlap):
    """Yield half-open token intervals with complete coverage and no empty tail."""
    if total < 1:
        raise ValueError("Text must contain at least one token")
    if capacity < 1 or not 0 <= overlap < capacity:
        raise ValueError("overlap_tokens must be smaller than the usable window")
    start = 0
    while start < total:
        end = min(start + capacity, total)
        yield start, end
        if end == total:
            break
        start = end - overlap


def coverage_weights(ranges, total):
    """Each source token contributes exactly one unit, shared among its windows."""
    difference = [0] * (total + 1)
    for start, end in ranges:
        difference[start] += 1
        difference[end] -= 1
    prefix = [0.0]
    active = 0
    for index in range(total):
        active += difference[index]
        if active == 0:
            raise ValueError("Uncovered token in chunk layout")
        prefix.append(prefix[-1] + 1.0 / active)
    return [prefix[end] - prefix[start] for start, end in ranges]


class Detector:
    """Lazy model cache; serializes inference to bound GPU memory across callers."""

    def __init__(self, device=None, model_dir=None, batch_size=8):
        requested = device or os.environ.get("AIGC_DEVICE", "auto")
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
        self.model_dir = Path(model_dir or MODEL_ROOT).resolve()
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be between 1 and 32")
        self.batch_size = batch_size
        self._loaded = {}
        self._lock = threading.RLock()
        # Avoid excessive CPU thread overhead on a desktop when CPU is selected.
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    def _manifest(self):
        path = self.model_dir / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8")).get("models", {}) if path.exists() else {}

    def _installed(self, model_id, manifest):
        entry = manifest.get(model_id, {})
        directory = self.model_dir / model_id
        files = entry.get("files", {})
        has_weights = any(name in files for name in ("pytorch_model.bin", "model.safetensors"))
        return bool(entry.get("installed") and has_weights and "config.json" in files and all(
            (directory / name).is_file() and (directory / name).stat().st_size == meta["bytes"]
            for name, meta in files.items()
        ))

    def list_models(self):
        manifest = self._manifest()
        with self._lock:
            return [{"id": key, **spec, "revision": manifest.get(key, {}).get("revision"),
                     "installed": self._installed(key, manifest), "loaded": key in self._loaded,
                     "max_tokens": 512} for key, spec in CATALOG.items()]

    def _load(self, model_id):
        if model_id in self._loaded:
            return self._loaded[model_id]
        if not self._installed(model_id, self._manifest()):
            raise ModelUnavailable(f"Model {model_id} is missing or incomplete; run scripts/download_models.py")
        directory = self.model_dir / model_id
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=True)
        model_class = AutoModelForSequenceClassification
        if CATALOG[model_id].get("loader") == "bert_mlp":
            from .modeling import BertMlpClassifier
            model_class = BertMlpClassifier
        model, loading = model_class.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, weights_only=True,
            output_loading_info=True,
        )
        problems = {key: loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
        if problems:
            raise ModelUnavailable(f"Checkpoint does not match the supported architecture for {model_id}: {problems}")
        model = model.to(self.device).eval()
        if model.config.num_labels != 2:
            raise ModelUnavailable(f"Unexpected number of output labels for {model_id}")
        limit = min(512, model.config.max_position_embeddings, tokenizer.model_max_length)
        self._loaded[model_id] = (tokenizer, model, int(limit))
        return self._loaded[model_id]

    def detect(self, text, models=None, *, overlap_tokens=64, include_chunks=True):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters; submit smaller sections")
        selected = select_models(models)
        if not isinstance(overlap_tokens, int) or isinstance(overlap_tokens, bool) or not 0 <= overlap_tokens <= 256:
            raise ValueError("overlap_tokens must be an integer between 0 and 256")
        started = time.perf_counter()
        warnings = []
        if len(text.strip()) < 100:
            warnings.append("Short text: scores may be unstable; no decision threshold is applied.")
        han_count = sum('\u3400' <= char <= '\u9fff' for char in text)
        if han_count / max(1, len(text.strip())) < 0.2:
            warnings.append("These checkpoints target Chinese; the input contains little Chinese text.")
        with self._lock:
            results = [self._score(text, model_id, overlap_tokens, include_chunks) for model_id in selected]
        return {
            "schema_version": "1.0", "input": {"characters": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
            "device": self.device, "results": results, "warnings": warnings,
            "interpretation": "Scores are uncalibrated model outputs, not AI-written percentages. No threshold or cross-model verdict is applied.",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }

    def _score(self, text, model_id, overlap, include_chunks):
        return self._score_texts([text], model_id, overlap, include_chunks)[0]

    def _probabilities(self, features, tokenizer, model, model_id):
        """Batch a stream of windows, including windows from different passages."""
        iterator = iter(features)
        with torch.inference_mode():
            while batch_features := list(islice(iterator, self.batch_size)):
                batch = tokenizer.pad(batch_features, padding=True, return_tensors="pt", verbose=False).to(self.device)
                values = model(**batch).logits.float().softmax(dim=-1)[:, CATALOG[model_id]["ai_label_index"]].cpu().tolist()
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                    raise RuntimeError(f"Non-finite probability returned by {model_id}")
                yield from values

    def _score_texts(self, texts, model_id, overlap, include_chunks, *, allow_empty=False):
        """Shared full-coverage inference, batched across bounded input passages."""
        started = time.perf_counter()
        tokenizer, model, limit = self._load(model_id)
        capacity = limit - tokenizer.num_special_tokens_to_add(pair=False)
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
        plans = []
        for ids, offsets in zip(encoded["input_ids"], encoded["offset_mapping"]):
            if not ids and allow_empty:
                plans.append((ids, offsets, [], []))
                continue
            ranges = list(window_ranges(len(ids), capacity, overlap))
            plans.append((ids, offsets, ranges, coverage_weights(ranges, len(ids))))
        features = (tokenizer.prepare_for_model(ids[start:end], add_special_tokens=True,
                    return_attention_mask=True, truncation=False, verbose=False)
                    for ids, _, ranges, _ in plans for start, end in ranges)
        score_iterator = self._probabilities(features, tokenizer, model, model_id)
        results = []
        revision = self._manifest()[model_id]["revision"]
        for ids, offsets, ranges, weights in plans:
            scores = list(islice(score_iterator, len(ranges)))
            result = {
                "model_id": model_id, "repo_id": CATALOG[model_id]["repo_id"], "revision": revision,
                "ai_score": sum(score * weight for score, weight in zip(scores, weights)) / sum(weights) if weights else None,
                "score_min": min(scores) if scores else None, "score_max": max(scores) if scores else None,
                "aggregation": "token_coverage_weighted_mean", "token_count": len(ids),
                "chunk_count": len(ranges), "max_tokens": limit, "overlap_tokens": overlap,
                "truncated": False,
            }
            if include_chunks:
                result["chunks"] = [{"index": index, "token_start": start, "token_end": end,
                                      "char_start": offsets[start][0], "char_end": offsets[end - 1][1],
                                      "token_count": end - start, "aggregation_weight": weight, "ai_score": score}
                                     for index, ((start, end), weight, score) in enumerate(zip(ranges, weights, scores))]
            results.append(result)
        elapsed = round(time.perf_counter() - started, 4)
        for result in results:
            result["elapsed_seconds"] = elapsed  # Shared batch elapsed when texts has multiple entries.
        return results

    def annotate(self, text, models=None, *, segment_chars=400, min_segment_chars=120,
                 high_threshold=None, possible_threshold=None, include_text=True, policy="thesis-cnki-20260929-v1"):
        from .annotation import annotate_text
        return annotate_text(self, text, models, segment_chars=segment_chars, min_segment_chars=min_segment_chars,
                             high_threshold=high_threshold, possible_threshold=possible_threshold, include_text=include_text, policy=policy)

    def detect_batch(self, texts, models=None, **options):
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= 32:
            raise ValueError("texts must contain 1 to 32 strings")
        if any(not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS for text in texts):
            raise ValueError("Each text must be non-empty and within the character limit")
        if sum(map(len, texts)) > MAX_BATCH_CHARS:
            raise ValueError(f"Batch exceeds {MAX_BATCH_CHARS} characters")
        return [self.detect(text, models, **options) for text in texts]
