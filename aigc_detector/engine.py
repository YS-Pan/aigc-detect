"""Offline inference and token-window coverage for Chinese and English models."""

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

from .catalog import CATALOG, MODEL_GROUPS
from .assets import assets_installed
from .paths import model_root

MAX_TEXT_CHARS = 200_000
MAX_BATCH_CHARS = 500_000


def select_models(models, language="zh"):
    if not isinstance(language, str) or language not in MODEL_GROUPS:
        raise ValueError("language must be zh or en")
    if models is None:
        return list(MODEL_GROUPS[language])
    if not isinstance(models, (list, tuple)) or not models or not all(isinstance(item, str) for item in models):
        raise ValueError("models must be a non-empty list of model IDs, or null for the language group")
    selected = list(dict.fromkeys(models))
    unknown = set(selected) - CATALOG.keys()
    if unknown:
        raise ValueError(f"Unknown model IDs: {', '.join(sorted(unknown))}")
    return selected


def language_warnings(text, selected):
    """Warn only about language-specific checkpoints actually selected."""
    little_han = sum('\u3400' <= char <= '\u9fff' for char in text) / max(1, len(text.strip())) < 0.2
    warnings = []
    if little_han and any(CATALOG[m]["language"] == "zh" for m in selected):
        warnings.append("little_chinese_text")
    if not little_han and any(CATALOG[m]["language"] == "en" for m in selected):
        warnings.append("english_models_on_chinese_text")
    return warnings


def score_warnings(selected):
    return [CATALOG[mid]["score_warning"] for mid in selected if CATALOG[mid].get("score_warning")]


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


def prepare_window(tokenizer, ids, model_id):
    if CATALOG[model_id].get("loader") == "cocodet":
        # Generic PreTrainedTokenizerFast.prepare_for_model does not run its
        # Rust TemplateProcessing postprocessor. ModernBERT requires CLS/SEP,
        # and CoCoDet classifies position zero, so omitting CLS changes scores.
        if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise ModelUnavailable("CoCoDet requires the pinned CLS/SEP tokenizer template")
        tokens = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
    if hasattr(tokenizer, "prepare_for_model"):
        return tokenizer.prepare_for_model(ids, add_special_tokens=True,
                                           return_attention_mask=True, truncation=False, verbose=False)
    # Transformers 5 fast tokenizers no longer provide prepare_for_model. The
    # registered classifier tokenizers use a single CLS ... SEP sequence, as
    # did the 4.x per-tokenizer implementations. Do not re-encode the tokens:
    # that would change the original offsets and overlapping window boundaries.
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ModelUnavailable(f"Unsupported single-sequence tokenizer template for {model_id}")
    tokens = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
    features = {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
    if "token_type_ids" in tokenizer.model_input_names:
        features["token_type_ids"] = [0] * len(tokens)
    return features


class Detector:
    """Lazy model cache; serializes inference to bound GPU memory across callers."""

    def __init__(self, device=None, model_dir=None, batch_size=8):
        requested = device or os.environ.get("AIGC_DEVICE", "auto")
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
        self.model_dir = Path(model_dir or model_root()).resolve()
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
        return assets_installed(directory, entry)

    def list_models(self):
        manifest = self._manifest()
        with self._lock:
            return [{"id": key, **spec, "revision": manifest.get(key, {}).get("revision"),
                     "installed": self._installed(key, manifest), "loaded": key in self._loaded,
                     "max_tokens": spec.get("max_tokens", 512)} for key, spec in CATALOG.items()]

    def _load(self, model_id):
        if model_id in self._loaded:
            loaded = self._loaded.pop(model_id)
            self._loaded[model_id] = loaded
            return loaded
        if not self._installed(model_id, self._manifest()):
            raise ModelUnavailable(f"Model {model_id} is missing or incomplete; run python -m aigc_detector prepare --models {model_id} --assets-dir <model-directory>")
        directory = self.model_dir / model_id
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=True)
        model_class = AutoModelForSequenceClassification
        if CATALOG[model_id].get("loader") == "bert_mlp":
            from .modeling import BertMlpClassifier
            model_class = BertMlpClassifier
        elif CATALOG[model_id].get("loader") == "desklib_mean_pool":
            from .modeling import DesklibAIDetectionModel
            model_class = DesklibAIDetectionModel
        elif CATALOG[model_id].get("loader") == "legacy_roberta_pooler":
            from .modeling import LegacyRobertaPoolerClassifier
            model_class = LegacyRobertaPoolerClassifier
        if CATALOG[model_id].get("loader") == "cocodet":
            from .modeling import CoCoDetModel
            try:
                model, loading = CoCoDetModel.from_local(directory), {}
            except (ValueError, RuntimeError, OSError) as exc:
                raise ModelUnavailable(f"Checkpoint does not match CoCoDet architecture: {exc}") from exc
        else:
            model, loading = model_class.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False, weights_only=True,
                output_loading_info=True,
            )
        problems = {key: loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
        if problems:
            raise ModelUnavailable(f"Checkpoint does not match the supported architecture for {model_id}: {problems}")
        if self.device == "cuda":
            weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            weight_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
            reserve = (4 if CATALOG[model_id].get("max_tokens", 512) > 512 else 2) * 2**30
            self._ensure_cuda_room(weight_bytes + reserve)
        model = model.to(self.device).eval()
        expected_labels = CATALOG[model_id].get("num_labels", 1 if CATALOG[model_id].get("score_activation") == "sigmoid" else 2)
        if model.config.num_labels != expected_labels:
            raise ModelUnavailable(f"Unexpected number of output labels for {model_id}")
        limit = min(CATALOG[model_id].get("max_tokens", 512), model.config.max_position_embeddings, tokenizer.model_max_length)
        self._loaded[model_id] = (tokenizer, model, int(limit))
        return self._loaded[model_id]

    def _ensure_cuda_room(self, required_bytes):
        """Release least-recently-used classifier snapshots under GPU pressure."""
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        while free < required_bytes and self._loaded:
            del self._loaded[next(iter(self._loaded))]
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
        if free < required_bytes:
            raise ModelUnavailable("Insufficient free CUDA memory for model and inference workspace; use device=cpu or free other GPU jobs")

    def detect(self, text, models=None, *, language="zh", overlap_tokens=64, include_chunks=True):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters; submit smaller sections")
        selected = select_models(models, language)
        if not isinstance(overlap_tokens, int) or isinstance(overlap_tokens, bool) or not 0 <= overlap_tokens <= 256:
            raise ValueError("overlap_tokens must be an integer between 0 and 256")
        started = time.perf_counter()
        warnings = []
        if len(text.strip()) < 100:
            warnings.append("Short text: scores may be unstable; no decision threshold is applied.")
        warnings.extend(language_warnings(text, selected))
        warnings.extend(score_warnings(selected))
        with self._lock:
            results = [self._score(text, model_id, overlap_tokens, include_chunks) for model_id in selected]
        return {
            "schema_version": "1.0", "input": {"characters": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
            "model_selection": {"language": language, "explicit_models": models is not None},
            "device": self.device, "results": results, "warnings": warnings,
            "interpretation": "Scores are uncalibrated model outputs, not AI-written percentages. No threshold or cross-model verdict is applied.",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }

    def _score(self, text, model_id, overlap, include_chunks):
        return self._score_texts([text], model_id, overlap, include_chunks)[0]

    def _probabilities(self, features, tokenizer, model, model_id, *, include_classes=False):
        """Batch a stream of windows, including windows from different passages."""
        iterator = iter(features)
        with torch.inference_mode():
            batch_size = min(self.batch_size, CATALOG[model_id].get("inference_batch_size", self.batch_size))
            while batch_features := list(islice(iterator, batch_size)):
                batch = tokenizer.pad(batch_features, padding=True, return_tensors="pt", verbose=False).to(self.device)
                logits = model(**batch).logits.float()
                spec = CATALOG[model_id]
                sigmoid = spec.get("score_activation") == "sigmoid"
                expected_labels = spec.get("num_labels", 1 if sigmoid else 2)
                if logits.ndim != 2 or logits.shape != (len(batch_features), expected_labels):
                    raise ModelUnavailable(f"Unexpected output shape for {model_id}: {tuple(logits.shape)}")
                probabilities = logits.sigmoid() if sigmoid else logits.softmax(dim=-1)
                if not torch.isfinite(probabilities).all():
                    raise RuntimeError(f"Non-finite class probability returned by {model_id}")
                indices = spec["ai_label_indices"] if "ai_label_indices" in spec else [spec["ai_label_index"]]
                # Sum in float64 to avoid a float32 sum slightly exceeding 1.
                values = probabilities[:, indices].double().sum(-1).clamp(0, 1).cpu().tolist()
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                    raise RuntimeError(f"Non-finite probability returned by {model_id}")
                if include_classes:
                    for value, row in zip(values, probabilities.cpu().tolist()):
                        yield {"ai_score": value, "class_scores": dict(zip(spec["class_labels"], row))}
                else:
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
        features = (prepare_window(tokenizer, ids[start:end], model_id)
                    for ids, _, ranges, _ in plans for start, end in ranges)
        spec = CATALOG[model_id]
        multiclass = "class_labels" in spec
        score_iterator = self._probabilities(features, tokenizer, model, model_id, include_classes=True) if multiclass else self._probabilities(features, tokenizer, model, model_id)
        results = []
        revision = self._manifest()[model_id]["revision"]
        for ids, offsets, ranges, weights in plans:
            rows = list(islice(score_iterator, len(ranges)))
            scores = [row["ai_score"] for row in rows] if multiclass else rows
            result = {
                "model_id": model_id, "repo_id": CATALOG[model_id]["repo_id"], "revision": revision,
                "ai_score": sum(score * weight for score, weight in zip(scores, weights)) / sum(weights) if weights else None,
                "score_min": min(scores) if scores else None, "score_max": max(scores) if scores else None,
                "aggregation": "token_coverage_weighted_mean", "token_count": len(ids),
                "chunk_count": len(ranges), "max_tokens": limit, "overlap_tokens": overlap,
                "truncated": False,
            }
            if multiclass:
                result.update({"score_semantics": spec["score_semantics"],
                               "ai_score_classes": [spec["class_labels"][i] for i in spec["ai_label_indices"]],
                               "class_scores": {label: sum(row["class_scores"][label] * weight for row, weight in zip(rows, weights)) / sum(weights) if weights else None
                                                for label in spec["class_labels"]}})
            if include_chunks:
                result["chunks"] = [{"index": index, "token_start": start, "token_end": end,
                                      "char_start": offsets[start][0], "char_end": offsets[end - 1][1],
                                      "token_count": end - start, "aggregation_weight": weight, "ai_score": score}
                                     for index, ((start, end), weight, score) in enumerate(zip(ranges, weights, scores))]
                if multiclass:
                    for chunk, row in zip(result["chunks"], rows):
                        chunk["class_scores"] = row["class_scores"]
            results.append(result)
        elapsed = round(time.perf_counter() - started, 4)
        for result in results:
            result["elapsed_seconds"] = elapsed  # Shared batch elapsed when texts has multiple entries.
        return results

    def annotate(self, text, models=None, *, language="zh", segment_chars=None, min_segment_chars=None,
                 high_threshold=None, possible_threshold=None, include_text=True, policy=None):
        from .annotation import annotate_text
        return annotate_text(self, text, models, language=language, segment_chars=segment_chars, min_segment_chars=min_segment_chars,
                             high_threshold=high_threshold, possible_threshold=possible_threshold, include_text=include_text, policy=policy)

    def detect_batch(self, texts, models=None, **options):
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= 32:
            raise ValueError("texts must contain 1 to 32 strings")
        if any(not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS for text in texts):
            raise ValueError("Each text must be non-empty and within the character limit")
        if sum(map(len, texts)) > MAX_BATCH_CHARS:
            raise ValueError(f"Batch exceeds {MAX_BATCH_CHARS} characters")
        return [self.detect(text, models, **options) for text in texts]
