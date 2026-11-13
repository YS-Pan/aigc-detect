"""Local Fast-DetectGPT/Binoculars statistics with sequential, bounded LM loading.

Formula references (read as evidence, never imported):
Fast-DetectGPT, Bao et al., MIT:
https://github.com/baoguangsheng/fast-detect-gpt/blob/971b05202bac2bb504d60c0ac0812fea7a8f7c82/scripts/fast_detect_gpt.py
Binoculars, Hans et al., BSD-3-Clause:
https://github.com/ahans30/Binoculars/blob/c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8/binoculars/metrics.py

Raw statistics are not probabilities. Full-coverage multi-window aggregation is
a project adaptation; no upstream or fitted decision threshold is applied.
"""

import gc
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .assets import assets_installed
from .catalog import CAUSAL_ASSETS, STATISTICAL_BACKENDS
from .engine import MAX_BATCH_CHARS, MAX_TEXT_CHARS, ModelUnavailable, window_ranges
from .paths import model_root, state_dir


def inverse_coverage(ranges, total):
    difference = [0] * (total + 1)
    for start, end in ranges:
        difference[start] += 1
        difference[end] -= 1
    counts, active = [], 0
    for index in range(total):
        active += difference[index]
        if active < 1:
            raise ValueError("Uncovered statistical token")
        counts.append(1.0 / active)
    return torch.tensor(counts, dtype=torch.float64)


def distribution_components(reference_logits, scoring_logits, ids, *, block_tokens=32):
    """Compute token statistics on CPU, without several full float32 vocab tensors."""
    if reference_logits.shape != scoring_logits.shape or scoring_logits.ndim != 2:
        raise ValueError("Statistical models must have identical sequence and vocabulary shapes")
    if len(ids) != len(scoring_logits) or len(ids) < 2:
        raise ValueError("At least two input tokens are required")
    observed, means, variances, cross_entropies = [], [], [], []
    for start in range(0, len(ids), block_tokens):
        end = min(start + block_tokens, len(ids))
        log_probs = scoring_logits[start:end].float().log_softmax(dim=-1)
        probs = reference_logits[start:end].float().softmax(dim=-1)
        expected = (probs * log_probs).sum(dim=-1)
        variance = (probs * log_probs.square()).sum(dim=-1) - expected.square()
        count = min(end, len(ids) - 1) - start
        if count > 0:
            labels = torch.tensor(ids[start + 1:start + 1 + count], dtype=torch.long)
            observed.append(log_probs[:count].gather(-1, labels[:, None]).squeeze(-1).double())
            means.append(expected[:count].double())
            variances.append(variance[:count].clamp(min=0).double())
        cross_entropies.append(-expected.double())
    result = {"observed": torch.cat(observed), "mean": torch.cat(means),
              "variance": torch.cat(variances), "cross_entropy": torch.cat(cross_entropies)}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Non-finite statistical model output")
    return result


def checkpoint_problems(asset, model, loading):
    """Retain strict checkpoint checks, except GPT-Neo's obsolete masks.

    Transformers 4 excluded these registered non-persistent buffers from its
    unexpected-keys report; Transformers 5 regenerates the causal mask and no
    longer registers masked_bias. The checkpoint may still contain both names.
    """
    unexpected = set(loading.get("unexpected_keys", ()))
    ignored = set()
    if asset == "gpt-neo-2.7b" and model.config.model_type == "gpt_neo":
        legacy = {f"transformer.h.{index}.attn.attention.{name}"
                  for index in range(model.config.num_layers)
                  for name in ("bias", "masked_bias")}
        ignored = unexpected & legacy
        unexpected -= ignored
    problems = {key: loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if unexpected:
        problems["unexpected_keys"] = sorted(unexpected)
    return problems, sorted(ignored)


def raw_statistic(method, components, prediction_weights, token_weights, ids, pad_token_id):
    if method.startswith("fast-detect-gpt"):
        variance = float((components["variance"] * prediction_weights).sum())
        if variance <= 0:
            raise RuntimeError("Zero conditional variance; statistic is undefined")
        return float(((components["observed"] - components["mean"]) * prediction_weights).sum()) / math.sqrt(variance)
    if method != "binoculars-falcon":
        raise ValueError("Unknown statistical backend")
    ce = float((-components["observed"] * prediction_weights).sum() / prediction_weights.sum())
    mask = torch.tensor([token != pad_token_id for token in ids], dtype=torch.float64)
    weights = token_weights * mask
    if not float(weights.sum()):
        raise ValueError("No non-padding tokens for Binoculars")
    cross_ce = float((components["cross_entropy"] * weights).sum() / weights.sum())
    if cross_ce <= 0:
        raise RuntimeError("Zero cross entropy; statistic is undefined")
    return ce / cross_ce


class StatisticalDetector:
    """Batch size one; only one causal LM resides in GPU/CPU memory at a time."""

    def __init__(self, device="auto", model_dir=None, *, gpu_memory_gib=10, cpu_memory_gib=20):
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be auto, cuda or cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        for memory in (gpu_memory_gib, cpu_memory_gib):
            if isinstance(memory, bool) or not isinstance(memory, int) or not 1 <= memory <= 1024:
                raise ValueError("Memory budgets must be positive integer GiB values")
        self.gpu_memory_gib, self.cpu_memory_gib = gpu_memory_gib, cpu_memory_gib
        self.model_dir = Path(model_dir or model_root()).resolve()
        self.last_loads = []
        self._lock = threading.RLock()
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    def _manifest(self):
        path = self.model_dir / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8")).get("models", {}) if path.exists() else {}

    def list_backends(self):
        manifest = self._manifest()
        return [{"id": key, "assets": list(assets),
                 "installed": all(assets_installed(self.model_dir / asset, manifest.get(asset, {})) for asset in assets),
                 "score_direction": "lower_is_more_ai_like" if key.startswith("binoculars") else "higher_is_more_ai_like",
                 "calibrated": False} for key, assets in STATISTICAL_BACKENDS.items()]

    def _tokenizer(self, asset):
        if not assets_installed(self.model_dir / asset, self._manifest().get(asset, {})):
            raise ModelUnavailable(f"Causal LM {asset} missing/incomplete; run python -m aigc_detector prepare --models {asset} --assets-dir <model-directory>")
        return AutoTokenizer.from_pretrained(self.model_dir / asset, local_files_only=True,
                                              trust_remote_code=False, use_fast=True)

    def _load(self, asset):
        import psutil
        memory = {"cpu": min(self.cpu_memory_gib * 2**30, int(psutil.virtual_memory().available * .8))}
        if self.device == "cuda":
            free, _ = torch.cuda.mem_get_info()
            memory[0] = min(self.gpu_memory_gib * 2**30, max(1, free - 3 * 2**30))
        model, loading = AutoModelForCausalLM.from_pretrained(
            self.model_dir / asset, local_files_only=True, trust_remote_code=False,
            weights_only=True, output_loading_info=True, dtype=torch.bfloat16,
            device_map="auto",
            max_memory=memory, attn_implementation="eager",
        )
        problems, ignored_buffers = checkpoint_problems(asset, model, loading)
        if problems:
            raise ModelUnavailable(f"Causal checkpoint architecture mismatch for {asset}: {problems}")
        devices = {str(value) for value in model.hf_device_map.values()}
        if "disk" in devices:
            raise ModelUnavailable("Configured GPU/CPU budgets require disk offload; increase available memory")
        self.last_loads.append({"asset": asset, "parameters": sum(p.numel() for p in model.parameters()),
                                "dtype": "bfloat16", "device_map": {key: str(value) for key, value in model.hf_device_map.items()},
                                "max_memory_bytes": {str(key): value for key, value in memory.items()},
                                "loading_problems": problems, "ignored_legacy_buffer_keys": ignored_buffers,
                                "cpu_rss_after_load_bytes": psutil.Process().memory_info().rss})
        return model.eval()

    def _cache_logits(self, asset, windows, temporary, *, progress=None, consume=None):
        import psutil
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = self._load(asset)
        peak_rss = psutil.Process().memory_info().rss
        # Accelerate routes CPU-offloaded layers through the main execution GPU.
        execution = "cuda:0" if self.device == "cuda" and any(str(d).startswith(("0", "cuda"))
                    for d in model.hf_device_map.values()) else "cpu"
        self.last_loads[-1]["execution_device"] = execution
        paths = []
        try:
            with torch.inference_mode():
                for index, ids in enumerate(windows):
                    inputs = torch.tensor([ids], dtype=torch.long, device=execution)
                    outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
                    logits = outputs.logits[0].detach().to("cpu")
                    if consume is not None:
                        consume(index, logits)
                    else:
                        path = temporary / f"{asset}-{index}.pt"
                        torch.save(logits, path)
                        paths.append(path)
                    del outputs, inputs, logits
                    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                    if progress and (index % 16 == 0 or index + 1 == len(windows)):
                        progress(f"{asset}: {index+1}/{len(windows)} windows")
        finally:
            self.last_loads[-1]["cpu_peak_observed_rss_bytes"] = peak_rss
            self.last_loads[-1]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return paths

    def detect(self, text, methods=None, *, max_tokens=512, overlap_tokens=64):
        with self._lock:
            return self._detect(text, methods, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def _detect(self, text, methods, *, max_tokens, overlap_tokens):
        return self._detect_batch([text], methods, max_tokens=max_tokens, overlap_tokens=overlap_tokens)[0]

    def detect_batch(self, texts, methods=None, *, max_tokens=512, overlap_tokens=64, progress=None, allow_empty=False):
        """Reuse each LM load across up to 1024 independent bounded inputs.

        Scores and window aggregation are identical to separate calls. The LM
        forward batch size remains one; inputs never share a token context.
        """
        with self._lock:
            return self._detect_batch(texts, methods, max_tokens=max_tokens,
                                      overlap_tokens=overlap_tokens, progress=progress, allow_empty=allow_empty)

    def _detect_batch(self, texts, methods, *, max_tokens, overlap_tokens, progress=None, allow_empty=False):
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= 1024:
            raise ValueError("texts must contain 1 to 1024 strings")
        if not isinstance(allow_empty, bool):
            raise ValueError("allow_empty must be boolean")
        if any(not isinstance(text, str) or not allow_empty and not text.strip() or len(text) > MAX_TEXT_CHARS for text in texts):
            raise ValueError(f"Each text must be nonempty and at most {MAX_TEXT_CHARS} characters")
        if sum(map(len, texts)) > MAX_BATCH_CHARS:
            raise ValueError(f"Statistical batch exceeds {MAX_BATCH_CHARS} characters")
        if methods is None:
            methods = ["fast-detect-gpt-neo"]
        if not isinstance(methods, (list, tuple)) or not methods or not all(isinstance(method, str) for method in methods):
            raise ValueError("methods must be a nonempty list of backend IDs")
        methods = list(dict.fromkeys(methods))
        if not methods or any(method not in STATISTICAL_BACKENDS for method in methods):
            raise ValueError("Unknown statistical backend")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 32 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be an integer between 32 and 2048")
        if isinstance(overlap_tokens, bool) or not isinstance(overlap_tokens, int) or not 1 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be an integer between 1 and max_tokens - 1")
        started = time.perf_counter()
        self.last_loads = []
        results = [[] for _ in texts]
        families = {}
        for method in methods:
            families.setdefault(STATISTICAL_BACKENDS[method], []).append(method)
        scratch = state_dir() / "statistical-logits"
        scratch.mkdir(parents=True, exist_ok=True)
        for assets, family_methods in families.items():
            tokenizers = [self._tokenizer(asset) for asset in assets]
            tokenizer = tokenizers[0]
            for other in tokenizers[1:]:
                if tokenizer.get_vocab() != other.get_vocab() or tokenizer.all_special_ids != other.all_special_ids:
                    raise ModelUnavailable("Statistical LM tokenizers do not have identical vocabularies/special IDs")
            encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False,
                                return_offsets_mapping=True, return_token_type_ids=False)
            plans, windows = [], []
            for ids, offsets in zip(encoded["input_ids"], encoded["offset_mapping"]):
                if len(ids) < 2 and not allow_empty:
                    raise ValueError("Statistical detection needs at least two tokens per input")
                ranges = list(window_ranges(len(ids), max_tokens, overlap_tokens)) if len(ids) >= 2 else []
                plans.append((ids, offsets, ranges, len(windows)))
                windows.extend(ids[start:end] for start, end in ranges)
            states, owners = [], []
            for input_index, (ids, offsets, ranges, first) in enumerate(plans):
                if not ranges:
                    states.append(None)
                    continue
                states.append({"token_weights": inverse_coverage(ranges, len(ids)),
                    "prediction_weights": inverse_coverage([(start, end-1) for start, end in ranges], len(ids)-1),
                    "components": {key: torch.zeros(len(ids)-(key != "cross_entropy"), dtype=torch.float64)
                                   for key in ("observed", "mean", "variance", "cross_entropy")},
                    "window_results": {method: [] for method in family_methods}})
                owners.extend((input_index, index, start, end) for index, (start, end) in enumerate(ranges))

            def accumulate(global_index, reference, scoring):
                input_index, index, start, end = owners[global_index]
                ids, offsets, _, _ = plans[input_index]
                state = states[input_index]
                window = ids[start:end]
                values = distribution_components(reference, scoring, window)
                for key, value in values.items():
                    stop = end if key == "cross_entropy" else end-1
                    weights = state["token_weights"][start:stop] if key == "cross_entropy" else state["prediction_weights"][start:stop]
                    state["components"][key][start:stop] += value * weights
                for method in family_methods:
                    raw = raw_statistic(method, values, torch.ones(end-start-1), torch.ones(end-start),
                                        window, tokenizer.pad_token_id or tokenizer.eos_token_id)
                    state["window_results"][method].append({"index": index, "token_start": start, "token_end": end,
                        "char_start": offsets[start][0], "char_end": offsets[end-1][1], "raw_score": raw})

            with tempfile.TemporaryDirectory(prefix="lm-", dir=scratch) as temp:
                temporary = Path(temp)
                if not temporary.resolve().is_relative_to(scratch.resolve()):
                    raise RuntimeError("Statistical scratch directory escaped its configured root")
                if windows and len(assets) == 1:
                    self._cache_logits(assets[0], windows, temporary, progress=progress,
                                       consume=lambda index, logits: accumulate(index, logits, logits))
                elif windows:
                    reference_paths = self._cache_logits(assets[0], windows, temporary, progress=progress)
                    def consume_scoring(index, logits):
                        reference = torch.load(reference_paths[index], map_location="cpu", weights_only=True)
                        accumulate(index, reference, logits)
                    self._cache_logits(assets[1], windows, temporary, progress=progress, consume=consume_scoring)
                for input_index, (ids, offsets, ranges, first) in enumerate(plans):
                    if not ranges:
                        for method in family_methods:
                            results[input_index].append({"backend_id": method, "raw_score": None, "token_count": len(ids),
                                "predicted_tokens": 0, "unscored_initial_tokens": len(ids), "chunk_count": 0, "chunks": [],
                                "max_tokens": max_tokens, "overlap_tokens": overlap_tokens, "coverage_complete": True,
                                "truncated": False, "calibrated": False, "threshold_applied": False,
                                "assets": [{"id": a, "repo_id": CAUSAL_ASSETS[a]["repo_id"], "revision": self._manifest()[a]["revision"]} for a in assets],
                                "score_direction": "lower_is_more_ai_like" if method.startswith("binoculars") else "higher_is_more_ai_like",
                                "aggregation": "inverse_coverage_token_statistics_then_formula", "evaluation_status": "no_scorable_tokens"})
                        continue
                    components = states[input_index]["components"]
                    window_results = states[input_index]["window_results"]
                    for method in family_methods:
                        raw = raw_statistic(method, components, torch.ones(len(ids)-1), torch.ones(len(ids)), ids,
                                            tokenizer.pad_token_id or tokenizer.eos_token_id)
                        if not math.isfinite(raw):
                            raise RuntimeError("Non-finite aggregate statistical score")
                        results[input_index].append({"backend_id": method, "raw_score": raw,
                            "score_direction": "lower_is_more_ai_like" if method.startswith("binoculars") else "higher_is_more_ai_like",
                            "assets": [{"id": asset, "repo_id": CAUSAL_ASSETS[asset]["repo_id"],
                                        "revision": self._manifest()[asset]["revision"]} for asset in assets],
                            "token_count": len(ids), "predicted_tokens": len(ids)-1, "unscored_initial_tokens": 1,
                            "chunk_count": len(ranges), "max_tokens": max_tokens, "overlap_tokens": overlap_tokens,
                            "coverage_complete": True, "truncated": False, "chunks": window_results[method],
                            "aggregation": "inverse_coverage_token_statistics_then_formula",
                            "calibrated": False, "threshold_applied": False})
                    if progress and (input_index % 16 == 0 or input_index+1 == len(plans)):
                        progress(f"{'/'.join(family_methods)}: {input_index+1}/{len(plans)} inputs")
        return [{"schema_version": "aigc-statistical/1.0", "status": "complete",
                "input": {"characters": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
                "device": "cuda" if any(row["execution_device"].startswith("cuda") for row in self.last_loads) else "cpu",
                "device_preference": self.device, "dtype": "bfloat16", "statistics_dtype": "float32",
                "execution": "sequential_lms_streamed_cpu_statistics", "loads": self.last_loads,
                "gpu_memory_gib": self.gpu_memory_gib, "cpu_memory_gib": self.cpu_memory_gib,
                "results": sorted(rows, key=lambda value: methods.index(value["backend_id"])),
                "batch_inputs": len(texts), "batch_index": index,
                "elapsed_seconds": round(time.perf_counter()-started, 4),
                "interpretation": "Raw statistics, not probabilities or authorship proof. No threshold/fusion is applied. Full-coverage aggregation and float32 statistic arithmetic require separate validation; upstream thresholds do not establish local false-positive rates."}
                for index, (text, rows) in enumerate(zip(texts, results))]
