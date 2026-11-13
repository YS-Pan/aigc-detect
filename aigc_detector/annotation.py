"""Passage-level screening labels with exact, exhaustive source spans.

Thresholds are configurable engineering rules, not calibrated probabilities.
"""

import bisect
import hashlib
import math
import re
import time
from datetime import datetime, timezone

from . import __version__
from .catalog import CATALOG
from .fusion import (ENGLISH_POLICY, ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY, ENGLISH_PROFILES, load_profile, policy_settings,
                     decide_fusion, resolve_policy, signal_revision, segment_raw_scores)

LABELS = {"strong_signal": "明显 AIGC 特征", "possible_signal": "较可能 AIGC", "other": "其他"}
BOUNDARIES = re.compile(r"[。！？!?；;]+[”’\"'」』]*|(?:\r\n|\r|\n)+")
EN_BOUNDARIES = re.compile(r"[!?;]+[\"'’”)]*|\.+[\"'’”)]*(?=\s|$)|(?:\r\n|\r|\n)+")


def validate_policy(high_threshold, possible_threshold, segment_chars, min_segment_chars, *, independent_thresholds=False):
    for value in (high_threshold, possible_threshold):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Thresholds must be finite numbers")
    if independent_thresholds:
        if not 0 <= possible_threshold <= 1 or not 0 <= high_threshold <= 1:
            raise ValueError("Each head threshold must be in [0, 1]")
    elif not 0 <= possible_threshold < high_threshold <= 1:
        raise ValueError("Require 0 <= possible_threshold < high_threshold <= 1")
    for value in (segment_chars, min_segment_chars):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Segment lengths must be integers")
    if not 128 <= segment_chars <= 2000 or not 1 <= min_segment_chars <= segment_chars:
        raise ValueError("Require 128 <= segment_chars <= 2000 and 1 <= min_segment_chars <= segment_chars")


def segment_spans(text, max_chars=400, min_chars=120, *, language="zh"):
    """Partition every source character once; prefer natural boundaries near the cap."""
    boundaries = EN_BOUNDARIES if language == "en" else BOUNDARIES
    cuts = [match.end() for match in boundaries.finditer(text)]
    word_cuts = [match.end() for match in re.finditer(r"\s+", text)] if language == "en" else []
    spans = []
    start = 0
    while start < len(text):
        cap = min(start + max_chars, len(text))
        if cap == len(text):
            end = cap
        else:
            candidates = cuts[bisect.bisect_left(cuts, start + min_chars):bisect.bisect_right(cuts, cap)]
            if not candidates and word_cuts:
                candidates = word_cuts[bisect.bisect_left(word_cuts, start + min_chars):bisect.bisect_right(word_cuts, cap)]
            end = candidates[-1] if candidates else cap
            # Keep the two code points of a CRLF newline together.
            if end < len(text) and text[end - 1:end + 1] == "\r\n":
                end -= 1
        spans.append((start, end))
        start = end
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_chars:
        left, right = spans[-2][0], spans[-1][1]
        if right - left <= max_chars:
            spans[-2:] = [(left, right)]
        else:
            # Avoid a tiny final fragment where the previous passage can share it.
            cut = right - min_chars
            if cut < right and text[cut - 1:cut + 1] == "\r\n":
                cut -= 1
            if left < cut and right - cut <= max_chars:
                spans[-2:] = [(left, cut), (cut, right)]
    return spans


def score_category(score, high_threshold, possible_threshold):
    if score is None:
        return "other"
    if score >= high_threshold:
        return "strong_signal"
    if score >= possible_threshold:
        return "possible_signal"
    return "other"


def decide(scores, high_threshold, possible_threshold):
    """The user's selected sensitive rule: any model may flag a passage."""
    usable = {key: value for key, value in scores.items() if value is not None}
    high = [key for key, value in usable.items() if value >= high_threshold]
    possible = [key for key, value in usable.items() if value >= possible_threshold]
    category = "strong_signal" if high else "possible_signal" if possible else "other"
    categories = {score_category(value, high_threshold, possible_threshold) for value in usable.values()}
    return {"category": category, "label": LABELS[category],
            "triggered_models": high or possible, "high_models": high, "possible_or_high_models": possible,
            "max_ai_score": max(usable.values()) if usable else None,
            "model_disagreement": len(categories) > 1,
            "evaluation_status": "scored" if usable else "no_scorable_tokens"}


def summarize(segments):
    summary = {key: {"segments": 0, "characters": 0} for key in LABELS}
    for segment in segments:
        bucket = summary[segment["decision"]["category"]]
        bucket["segments"] += 1
        bucket["characters"] += segment["char_end"] - segment["char_start"]
    return summary


def statistical_passage_scores(detector, passages, profile):
    """Load LMs sequentially and preserve raw statistics, without probabilities."""
    import gc
    import torch
    from .engine import ModelUnavailable
    from .statistical import StatisticalDetector
    methods = list(profile["statistical_assets"])
    detector._loaded.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    statistical = StatisticalDetector(device=detector.device, model_dir=detector.model_dir,
                                       gpu_memory_gib=13, cpu_memory_gib=20)
    inference = profile["statistical_inference"]
    result = {m: [] for m in methods}
    loads = []
    for start in range(0, len(passages), 1024):
        batch = statistical.detect_batch(passages[start:start+1024], methods,
                    max_tokens=inference["max_tokens"], overlap_tokens=inference["overlap_tokens"], allow_empty=True)
        loads.extend(statistical.last_loads)
        for output in batch:
            for row in output["results"]:
                mid = row["backend_id"]
                if {a["id"]: a["revision"] for a in row["assets"]} != profile["statistical_assets"][mid]:
                    raise ModelUnavailable("Statistical model revisions differ from the fusion profile")
                result[mid].append({"raw_score": row["raw_score"], "ai_score": None, "kind": "statistical",
                                   "revision": signal_revision(profile, mid), "token_count": row["token_count"],
                                   "chunk_count": row["chunk_count"], "chunks": row["chunks"],
                                   "score_direction": row["score_direction"], "assets": row["assets"],
                                   "aggregation": row["aggregation"]})
    return result, loads


def annotate_text(detector, text, models=None, *, language="zh", segment_chars=None, min_segment_chars=None,
                  high_threshold=None, possible_threshold=None, include_text=True, policy=None):
    from .engine import MAX_TEXT_CHARS, language_warnings, score_warnings, select_models
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must be non-empty and at most {MAX_TEXT_CHARS} characters")
    if not isinstance(include_text, bool):
        raise ValueError("include_text must be a boolean")
    policy = resolve_policy(policy, language)
    selected = select_models(None if policy in (ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY) else models, language)
    defaults = load_profile(policy)[0] if policy in ENGLISH_PROFILES else {"segment_chars": 400, "min_segment_chars": 120}
    if policy in (ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY):
        if models is None:
            selected = list(defaults["models"])
        elif not isinstance(models, (list, tuple)) or not models or not all(isinstance(m, str) for m in models):
            raise ValueError("models must be a nonempty list of signal IDs")
        else:
            selected = list(dict.fromkeys(models))
    segment_chars = defaults["segment_chars"] if segment_chars is None else segment_chars
    min_segment_chars = defaults["min_segment_chars"] if min_segment_chars is None else min_segment_chars
    profile, settings = policy_settings(policy, selected, segment_chars, min_segment_chars, high_threshold, possible_threshold)
    high_threshold, possible_threshold = settings["high_threshold"], settings["possible_threshold"]
    raw_high, raw_possible = settings["raw_score_high_threshold"], settings["raw_score_possible_threshold"]
    validate_policy(high_threshold, possible_threshold, segment_chars, min_segment_chars,
                    independent_thresholds=settings.get("threshold_mode") == "separate_heads")
    started = time.perf_counter()
    spans = segment_spans(text, segment_chars, min_segment_chars, language=language)
    passages = [text[start:end] for start, end in spans]
    line_starts = [0] + [match.end() for match in re.finditer(r"\r\n|\r|\n", text)]
    statistical = set(profile.get("statistical_assets", {})) if profile else set()
    statistical_loads = []
    with detector._lock:
        all_scores = {model_id: detector._score_texts(passages, model_id, 64, True, allow_empty=True)
                      for model_id in selected if model_id not in statistical}
        if profile and any(rows[0]["revision"] != signal_revision(profile, mid) for mid, rows in all_scores.items()):
            from .engine import ModelUnavailable
            raise ModelUnavailable("Model revision differs from the fusion reference profile; refit or select the legacy policy explicitly")
        if statistical:
            values, statistical_loads = statistical_passage_scores(detector, passages, profile)
            all_scores.update(values)
    if profile and any(all_scores[m][0]["revision"] != signal_revision(profile, m) for m in selected):
        from .engine import ModelUnavailable
        raise ModelUnavailable("Model revision differs from the fusion reference profile; refit or select the legacy policy explicitly")
    segments = []
    for index, ((start, end), passage) in enumerate(zip(spans, passages)):
        model_scores = {}
        for model_id in selected:
            item = all_scores[model_id][index]
            model_scores[model_id] = {"ai_score": item["ai_score"],
                                      "category": score_category(item["ai_score"], raw_high, raw_possible),
                                      "token_count": item["token_count"], "chunk_count": item["chunk_count"],
                                      "windows": [{**chunk, "char_start": chunk["char_start"] + start,
                                                   "char_end": chunk["char_end"] + start} for chunk in item["chunks"]]}
            if model_id in statistical:
                model_scores[model_id].update({key: item[key] for key in ("raw_score", "kind", "score_direction", "assets", "aggregation")})
                model_scores[model_id]["category"] = "other"
                model_scores[model_id]["raw_threshold_applied"] = False
            if "class_scores" in item:
                model_scores[model_id].update({key: item[key] for key in ("class_scores", "score_semantics", "ai_score_classes")})
        warnings = []
        if len(passage.strip()) < min_segment_chars:
            warnings.append("short_passage")
        warnings.extend(language_warnings(passage, [m for m in selected if m not in statistical]))
        scores = segment_raw_scores({"model_scores": model_scores})
        decision = decide_fusion(scores, profile, high_threshold, possible_threshold) if profile else decide(scores, high_threshold, possible_threshold)
        if decision["evaluation_status"] != "scored":
            warnings.append(decision["evaluation_status"])
        segments.append({"id": f"s{index + 1:06}", "index": index, "char_start": start, "char_end": end,
                         "line_start": bisect.bisect_right(line_starts, start),
                         "line_end": bisect.bisect_right(line_starts, end - 1),
                         "sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
                         "decision": decision, "model_scores": model_scores, "warnings": warnings,
                         **({"text": passage} if include_text else {})})
    return {"schema_version": "aigc-annotation/1.2", "software_version": __version__, "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {"characters": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                      "offset_unit": "python_unicode_code_points", "offset_convention": "zero_based_half_open", "line_base": 1},
            "device": detector.device,
            "model_selection": {"language": language, "explicit_models": models is not None},
            "models": [{"id": model_id, "repo_id": CATALOG[model_id]["repo_id"] if model_id not in statistical else None,
                        "revision": all_scores[model_id][0]["revision"],
                        **({"kind": "statistical", "assets": all_scores[model_id][0]["assets"]} if model_id in statistical else {})}
                       for model_id in selected],
            **({"statistical_execution": {"inference": profile["statistical_inference"], "loads": statistical_loads}}
               if statistical else {}),
            "policy": {**settings, "labels": LABELS},
            "segmentation": {"id": "english-natural-boundaries-v1" if language == "en" else "natural-boundaries-v1", "max_characters": segment_chars,
                             "minimum_target_characters": min_segment_chars, "independent_passages": True,
                             "coverage_complete": True, "characters_covered": len(text), "segment_count": len(spans)},
            "summary": summarize(segments), "segments": segments,
            "interpretation": "Uncalibrated screening labels, not proof of authorship. Other does not certify human writing. Character counts are flagged spans, not AI-written percentages." +
                              (" " + settings["validation_warning"] if "validation_warning" in settings else "") +
                              "".join(" " + warning for warning in score_warnings([m for m in selected if m not in statistical])),
            "elapsed_seconds": round(time.perf_counter() - started, 4)}
