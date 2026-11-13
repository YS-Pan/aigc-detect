"""Expanded English score-only campaign; prior test papers are repeat evaluation.

Never writes source text into evaluations/. Raw inference caches, mappings and
reports live in reports/. The v1 campaign and frozen profile remain untouched.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.annotation import segment_spans
from aigc_detector.catalog import ROOT, ENGLISH_MODELS, EXTRA_ENGLISH_MODELS, STATISTICAL_BACKENDS
from scripts.english_academic_samples import DEFAULT_DIR, digest, read_json, verify, write_json
from scripts.calibrate_english import binary_metrics
from scripts.fit_fusion_v2 import fit_head, feature_terms, sigmoid

POLICY = "english-academic-20261003-v2"
EVIDENCE = ROOT / "evaluations" / POLICY
REPORTS = ROOT / "reports" / "english-recalibration-20261003-v2"
CLASSIFIERS = list(ENGLISH_MODELS + EXTRA_ENGLISH_MODELS)
STATISTICS = list(STATISTICAL_BACKENDS)
SIGNALS = CLASSIFIERS + STATISTICS


def prepare():
    if (EVIDENCE / "campaign.json").exists():
        raise FileExistsError("Preserve the declared campaign; do not overwrite it")
    _, collection, protocol = verify(DEFAULT_DIR)
    manifest = read_json(ROOT / "models/manifest.json")["models"]
    campaign = {
        "schema_version": "english-expanded-campaign/1.0", "id": POLICY,
        "sample_collection_sha256": collection["samples_sha256"], "groups": protocol["groups"],
        "segmentations": [[400, 120], [800, 240], [1600, 400]],
        "classifiers": CLASSIFIERS, "statistical_backends": STATISTICS,
        "revisions": {m: manifest[m]["revision"] for m in CLASSIFIERS},
        "asset_revisions": {a: manifest[a]["revision"] for ms in STATISTICAL_BACKENDS.values() for a in ms},
        "statistical_inference": {"max_tokens": 512, "overlap_tokens": 64, "dtype": "bfloat16",
                                  "aggregation": "inverse_coverage_token_statistics_then_formula"},
        "feature_rosters": [list(ENGLISH_MODELS), list(EXTRA_ENGLISH_MODELS), CLASSIFIERS,
                            [m for m in CLASSIFIERS if m != "bibbbu-xlm-roberta"],
                            CLASSIFIERS + ["fast-detect-gpt-neo"],
                            CLASSIFIERS + ["fast-detect-gpt-falcon", "binoculars-falcon"], SIGNALS, STATISTICS],
        "families": ["linear", "squares", "quadratic"], "regularizations": [.1, 1., 10., 100.],
        "target_modes": ["ai_character_fraction", "any_generated_character"],
        "human_weights": [1., 5., 20.], "threshold_logit_margins": {"review": .01, "strong": .1},
        "selection": "Fit-paper leave-one-paper-out: minimize human flagged characters, then maximize generated characters; ties prefer fewer signals/terms, stronger regularization, shorter context.",
        "thresholds": "Maximum human logit in fit+calibration plus fixed margins; calibration AI does not choose thresholds or candidates.",
        "repeat_evaluation": {"papers": protocol["groups"]["test"], "previous_test_metrics_known": True,
                              "used_for_v2_parameter_or_candidate_selection": False,
                              "is_new_independent_validation": False},
        "runtime_features": "Raw classifier scores and raw LM statistics only; no text/length/identity features.",
        "sample_use": "All six fit originals, eighteen mixed documents and fifty-one generated paragraphs; calibration controls also include pure-human windows of its six mixed documents. Deduplicate identical passages within each paper; exclude conflicting labels. Test papers never enter fitting or thresholds.",
        "contains_source_text": False,
        "previous_profile_sha256": digest((ROOT / "aigc_detector/profiles/english-academic-20261003-v1.json").read_bytes()),
    }
    write_json(EVIDENCE / "campaign.json", campaign)
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"campaign": str(EVIDENCE / "campaign.json"), "signals": len(SIGNALS)}), flush=True)


def make_records(length, minimum, splits, kinds):
    samples, collection, _ = verify(DEFAULT_DIR)
    items, passages = [], {}
    for sample in samples:
        if sample["split"] not in splits or sample["kind"] not in kinds:
            continue
        blocks = read_json(DEFAULT_DIR / sample["mapping_file"])["blocks"] if "mapping_file" in sample else []
        ai_spans = [(b["char_start"], b["char_end"]) for b in blocks if b["source_label"] == "ai"]
        for n, (start, end) in enumerate(segment_spans(sample["text"], length, minimum, language="en")):
            text = sample["text"][start:end]
            key = digest(text.encode("utf-8"))
            passages.setdefault(key, text)
            items.append({"id": f"{sample['id']}:s{n+1:04}", "sample_id": sample["id"],
                          "paper_id": sample["paper_id"], "split": sample["split"], "kind": sample["kind"],
                          "generator": sample.get("generator"), "paragraph_id": sample.get("paragraph_id"),
                          "sha256": key, "char_start": start, "char_end": end, "chars": end-start,
                          "ai_chars": len(text) if sample["kind"] == "ai_paragraph" else
                          sum(max(0, min(end, b)-max(start, a)) for a, b in ai_spans)})
    owners = {}
    for r in items:
        owners.setdefault(r["sha256"], set()).add(r["paper_id"])
    for r in items:
        r["cross_paper_duplicate"] = len(owners[r["sha256"]]) > 1
    return items, passages, collection


def bounded_batches(keys, passages):
    batch, characters = [], 0
    for key in keys:
        size = len(passages[key])
        if batch and (len(batch) >= 1024 or characters + size > 500_000):
            yield batch
            batch, characters = [], 0
        batch.append(key)
        characters += size
    if batch:
        yield batch


def score(length, minimum, *, splits=("fit", "calibration"), kinds=("human_document", "ai_paragraph", "mixed_document"),
          signals=None, output=None):
    from aigc_detector.engine import Detector
    from aigc_detector.statistical import StatisticalDetector
    import torch
    campaign = read_json(EVIDENCE / "campaign.json")
    if [length, minimum] not in campaign["segmentations"]:
        raise ValueError("Segmentation was not declared in campaign")
    frozen = None
    if "test" in splits:
        if list(splits) != ["test"]:
            raise ValueError("Repeat evaluation must be a separate test-only run")
        frozen = read_json(EVIDENCE / "FREEZE.json")["profile_sha256"]
        profile = read_json(ROOT / "aigc_detector/profiles" / f"{POLICY}.json")
        if digest((ROOT / "aigc_detector/profiles" / f"{POLICY}.json").read_bytes()) != frozen:
            raise ValueError("Repeat evaluation requires unchanged frozen profile")
        if [length, minimum] != [profile["segment_chars"], profile["min_segment_chars"]]:
            raise ValueError("Repeat evaluation requires frozen segmentation")
    elif (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Preserve frozen training scores; use a new campaign for new fitting inference")
    items, passages, collection = make_records(length, minimum, splits, kinds)
    if collection["samples_sha256"] != campaign["sample_collection_sha256"]:
        raise ValueError("Source collection changed")
    signals = list(signals or SIGNALS)
    if not set(signals) <= set(SIGNALS):
        raise ValueError("Unknown signal")
    run_dir = REPORTS / f"context{length}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = {k: {} for k in passages}
    # Reuse only matched classifier revision, segmentation, text hash and overlap.
    for path in (ROOT / "evaluations/english-academic-20261003-v1").rglob("*scores.json"):
        old = read_json(path)
        for r in old.get("records", []):
            if r["sha256"] not in scores or r["split"] not in splits:
                continue
            for mid in ENGLISH_MODELS:
                if old["config"]["revisions"].get(mid) == campaign["revisions"][mid]:
                    scores[r["sha256"]][mid] = r["scores"][mid]
    def blocked(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")
    detector = Detector(batch_size=8)
    manifest = detector._manifest()
    for mid, revision in {**campaign["revisions"], **campaign["asset_revisions"]}.items():
        if manifest[mid]["revision"] != revision:
            raise ValueError("Model revision changed since declaration")
    print(json.dumps({"context": length, "records": len(items), "unique_passages": len(passages), "signals": signals}), flush=True)
    started = time.perf_counter()
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for mid in [s for s in signals if s in CLASSIFIERS]:
            cache_path = run_dir / f"{mid}-cache.json"
            cache = read_json(cache_path) if cache_path.exists() else {"revision": campaign["revisions"][mid], "items": {}}
            if cache["revision"] != campaign["revisions"][mid]:
                raise ValueError("Classifier cache revision mismatch")
            for key, value in cache["items"].items():
                if key in scores:
                    scores[key][mid] = value["ai_score"]
            missing = [k for k in passages if mid not in scores[k]]
            for offset in range(0, len(missing), 64):
                keys = missing[offset:offset+64]
                values = detector._score_texts([passages[k] for k in keys], mid, 64, True, allow_empty=True)
                for key, value in zip(keys, values):
                    if value["revision"] != campaign["revisions"][mid]:
                        raise ValueError("Classifier revision changed")
                    scores[key][mid] = value["ai_score"]
                    cache["items"][key] = value
                write_json(cache_path, cache)
                print(f"context{length} {mid}: {min(offset+64,len(missing))}/{len(missing)}", flush=True)
            detector._loaded.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        methods = [s for s in signals if s in STATISTICS]
        if methods:
            stat = StatisticalDetector(gpu_memory_gib=13, cpu_memory_gib=20)
            cache_path = run_dir / "statistical-cache.json"
            cache = read_json(cache_path) if cache_path.exists() else {
                "asset_revisions": campaign["asset_revisions"], "inference": campaign["statistical_inference"], "items": {}}
            if cache["asset_revisions"] != campaign["asset_revisions"] or cache["inference"] != campaign["statistical_inference"]:
                raise ValueError("Statistical cache contract mismatch")
            for key, rows in cache["items"].items():
                if key in scores:
                    scores[key].update({row["backend_id"]: row["raw_score"] for row in rows})
            missing = [k for k in passages if any(m not in scores[k] for m in methods)]
            completed = 0
            for keys in bounded_batches(missing, passages):
                outputs = stat.detect_batch([passages[k] for k in keys], methods,
                            progress=lambda msg: print(f"context{length} {msg}", flush=True))
                for key, value in zip(keys, outputs):
                    scores[key].update({row["backend_id"]: row["raw_score"] for row in value["results"]})
                    prior = {row["backend_id"]: row for row in cache["items"].get(key, [])}
                    prior.update({row["backend_id"]: row for row in value["results"]})
                    cache["items"][key] = list(prior.values())
                write_json(cache_path, cache)
                write_json(run_dir / f"statistical-loads-{completed}.json", stat.last_loads)
                completed += len(keys)
                print(f"context{length} statistical: {completed}/{len(missing)}", flush=True)
    for r in items:
        r["scores"] = {s: scores[r["sha256"]].get(s) for s in signals}
    dataset = {"schema_version": "english-expanded-scores/1.0", "contains_source_text": False,
               "config": {"signals": signals, "segment_chars": length, "min_segment_chars": minimum,
                          "revisions": campaign["revisions"], "asset_revisions": campaign["asset_revisions"],
                          "scored_splits": list(splits), "kinds": list(kinds), "frozen_profile_sha256": frozen,
                          "campaign_sha256": digest((EVIDENCE / "campaign.json").read_bytes()),
                          "network_blocked_during_inference": True, "elapsed_seconds": time.perf_counter()-started},
               "records": items}
    path = output or (EVIDENCE / "repeat-scores.json" if "test" in splits else EVIDENCE / f"context{length}" / "train-scores.json")
    write_json(path, dataset)
    print(json.dumps({"complete": str(path), "seconds": dataset["config"]["elapsed_seconds"]}), flush=True)


def transformed(records, signals):
    raw = np.array([[r["scores"][s] for s in signals] for r in records], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Incomplete or non-finite raw signals")
    for i, s in enumerate(signals):
        if s in CLASSIFIERS:
            p = np.clip(raw[:, i], 1e-5, 1-1e-5)
            raw[:, i] = np.log(p / (1-p))
        elif s == "binoculars-falcon":
            raw[:, i] *= -1
    return raw


def matrix(records, model):
    z = np.clip((transformed(records, model["signals"])-np.array(model["center"])) / np.array(model["scale"]), -5., 5.)
    return np.column_stack([np.ones(len(records))] + [np.prod(z[:, term], axis=1) for term in model["terms"]])


def fit_model(records, signals, family, reg, hw, target_mode="ai_character_fraction"):
    totals = {}
    for r in records:
        bucket = (r["paper_id"], r["ai_chars"] == 0)
        totals[bucket] = totals.get(bucket, 0) + r["chars"]
    w = np.array([r["chars"] / totals[r["paper_id"], r["ai_chars"] == 0]
                  * (hw if r["ai_chars"] == 0 else 1) for r in records])
    raw = transformed(records, signals)
    center = np.average(raw, axis=0, weights=w)
    floor = np.array([.01 if s == "binoculars-falcon" else .5 for s in signals])
    scale = np.maximum(np.sqrt(np.average((raw-center)**2, axis=0, weights=w)), floor)
    model = {"signals": signals, "center": center.tolist(), "scale": scale.tolist(), "terms": feature_terms(len(signals), family)}
    target = np.array([r["ai_chars"] / r["chars"] if target_mode == "ai_character_fraction" else float(r["ai_chars"] > 0)
                       for r in records])
    parameters, info = fit_head(matrix(records, model), target, w, model["terms"], reg)
    return {**model, "parameters": parameters.tolist(), "optimization": info}


def control_threshold(records, values, margin):
    return max(float(v) for r, v in zip(records, values) if r["ai_chars"] == 0) + margin


def eligible(data, split, *, deduplicate=True):
    rows = [r for r in data["records"] if r["split"] == split and not r["cross_paper_duplicate"]]
    grouped = {}
    for r in rows:
        grouped.setdefault((r["paper_id"], r["sha256"]), []).append(r)
    # Identical passage labels must agree; do not infer a label from scores.
    return [r for values in grouped.values() if len({r["ai_chars"] for r in values}) == 1
            for r in (values[:1] if deduplicate else values)]


def screening_metrics(records, flags):
    flags = np.asarray(flags, dtype=bool)
    pure_human = np.array([r["ai_chars"] == 0 for r in records])
    ai = np.array([r["ai_chars"] for r in records])
    chars = np.array([r["chars"] for r in records])
    return {"human_segments": int(pure_human.sum()), "human_flagged_segments": int((pure_human & flags).sum()),
            "human_flagged_characters": int(chars[pure_human & flags].sum()),
            "ai_characters": int(ai.sum()), "ai_flagged_characters": int(ai[flags].sum()),
            "ai_character_recall": float(ai[flags].sum()/ai.sum()) if ai.sum() else None,
            "human_characters_in_mixed_flagged_segments": int((chars[flags & ~pure_human]-ai[flags & ~pure_human]).sum())}


def search_context(length, minimum):
    campaign = read_json(EVIDENCE / "campaign.json")
    if [length, minimum] not in campaign["segmentations"]:
        raise ValueError("Undeclared context")
    path = EVIDENCE / f"context{length}" / "train-scores.json"
    cache_path = EVIDENCE / f"context{length}" / "candidates.json"
    identity = {"dataset_sha256": digest(path.read_bytes()), "campaign_sha256": digest((EVIDENCE / "campaign.json").read_bytes()),
                "fitter_sha256": digest(Path(__file__).read_bytes()), "optimizer_sha256": digest((ROOT / "scripts/fit_fusion_v2.py").read_bytes())
                if (ROOT / "scripts/fit_fusion_v2.py").exists() else "fixture"}
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached["identity"] == identity:
            return cached["candidates"]
    data = read_json(path)
    records = eligible(data, "fit")
    validation_rows = eligible(data, "fit", deduplicate=False)
    candidates = []
    for signals in campaign["feature_rosters"]:
      if not set(signals) <= set(data["config"]["signals"]):
        continue
      for family in campaign["families"]:
        for reg in campaign["regularizations"]:
          for hw, target_mode in [(w, t) for w in campaign["human_weights"] for t in campaign["target_modes"]]:
            folds = []
            for paper in campaign["groups"]["fit"]:
                train = [r for r in records if r["paper_id"] != paper]
                held = [r for r in validation_rows if r["paper_id"] == paper]
                model = fit_model(train, signals, family, reg, hw, target_mode)
                threshold = control_threshold(train, matrix(train, model) @ model["parameters"], .01)
                folds.append({"paper": paper, "threshold_logit": threshold,
                              "metrics": screening_metrics(held, matrix(held, model) @ model["parameters"] >= threshold)})
            candidates.append({"segment_chars": length, "min_segment_chars": minimum,
                               "signals": signals, "family": family, "regularization": reg, "human_weight": hw, "target_mode": target_mode,
                               "human_flagged_characters": sum(f["metrics"]["human_flagged_characters"] for f in folds),
                               "ai_flagged_characters": sum(f["metrics"]["ai_flagged_characters"] for f in folds), "folds": folds})
        print(json.dumps({"context": length, "signals": signals, "family": family, "candidates": len(candidates)}), flush=True)
    write_json(cache_path, {"identity": identity, "candidates": candidates, "contains_source_text": False})
    best = min(candidates, key=lambda c: (c["human_flagged_characters"], -c["ai_flagged_characters"]))
    print(json.dumps({"context_diagnostic_best": {k: v for k, v in best.items() if k != "folds"}}), flush=True)
    return candidates


def search():
    if (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Frozen campaign cannot be reselected")
    campaign = read_json(EVIDENCE / "campaign.json")
    candidates = []
    for length, minimum in campaign["segmentations"]:
        data = read_json(EVIDENCE / f"context{length}" / "train-scores.json")
        required = {s for roster in campaign["feature_rosters"] for s in roster}
        if not required <= set(data["config"]["signals"]):
            raise ValueError("Final selection requires every declared signal in every context")
        candidates.extend(search_context(length, minimum))
        write_json(REPORTS / "search-progress.json", {"completed_context": length, "candidates": candidates})
        print(json.dumps({"searched_context": length, "candidates": len(candidates)}), flush=True)
    expected = math.prod(len(campaign[k]) for k in ("segmentations", "feature_rosters", "families",
                                                  "regularizations", "human_weights", "target_modes"))
    if len(candidates) != expected:
        raise ValueError("Final selection requires the complete declared candidate comparison")
    def objective(c):
        return (c["human_flagged_characters"], -c["ai_flagged_characters"], len(c["signals"]),
                len(feature_terms(len(c["signals"]), c["family"])), -c["regularization"], c["segment_chars"], -c["human_weight"])
    chosen = min(candidates, key=objective)
    path = EVIDENCE / f"context{chosen['segment_chars']}" / "train-scores.json"
    data = read_json(path)
    train, calibration = eligible(data, "fit"), eligible(data, "calibration")
    model = fit_model(train, chosen["signals"], chosen["family"], chosen["regularization"], chosen["human_weight"], chosen["target_mode"])
    controls = train + calibration
    values = matrix(controls, model) @ model["parameters"]
    review, strong = [control_threshold(controls, values, campaign["threshold_logit_margins"][head]) for head in ("review", "strong")]
    head = {"intercept": model["parameters"][0], "coefficients": model["parameters"][1:]}
    profile = {"schema_version": "aigc-score-fusion/3.0", "id": POLICY, "language": "en",
               "segmentation_id": "english-natural-boundaries-v1", "models": chosen["signals"],
               "revisions": {s: campaign["revisions"][s] for s in chosen["signals"] if s in CLASSIFIERS},
               "statistical_assets": {s: {a: campaign["asset_revisions"][a] for a in STATISTICAL_BACKENDS[s]}
                                      for s in chosen["signals"] if s in STATISTICS},
               "statistical_inference": campaign["statistical_inference"],
               "score_transforms": {s: "logit" if s in CLASSIFIERS else "negative_identity" if s == "binoculars-falcon" else "identity" for s in chosen["signals"]},
               "segment_chars": chosen["segment_chars"], "min_segment_chars": chosen["min_segment_chars"],
               "center": model["center"], "scale": model["scale"], "feature_terms": model["terms"],
               "clip_probability": 1e-5, "clip_standardized": 5., "heads": {"strong": head, "review": head},
               "high_threshold": float(sigmoid(strong)), "possible_threshold": float(sigmoid(review)),
               "fit": {"dataset_sha256": digest(path.read_bytes()), "campaign_sha256": digest((EVIDENCE / "campaign.json").read_bytes()),
                       "groups": campaign["groups"], "family": chosen["family"], "regularization": chosen["regularization"],
                       "human_weight": chosen["human_weight"], "optimization": model["optimization"],
                       "target_mode": chosen["target_mode"], "training_passages_after_deduplication": len(train),
                       "head_relationship": "Identical fitted score function in both heads; distinct conservative thresholds.",
                       "calibration_passages_after_deduplication": len(calibration),
                       "test_used_for_fitting_or_selection": False, "prior_test_metrics_known": True,
                       "new_independent_validation": False, "threshold_controls": "Human fit+calibration only",
                       "threshold_logit_margins": campaign["threshold_logit_margins"], "cv_zero_human_flags_feasible": chosen["human_flagged_characters"] == 0},
               "limitations": ["Same ten antenna papers and three related generators; prior v1 test outcomes already known.",
                               "Repeat evaluation is not a fresh independent test; no broad reliability claim.",
                               "Human zero-control flags do not establish a zero population false-positive rate.",
                               "Screening scores are not authorship probabilities; PDF conversion uncertainty remains."]}
    if not 0 <= profile["possible_threshold"] < profile["high_threshold"] < 1:
        raise ValueError("Threshold saturation leaves no usable conservative operating point")
    target = ROOT / "aigc_detector/profiles" / f"{POLICY}.json"
    summary = {}
    for split, rows in [("fit", train), ("calibration", calibration)]:
        ls = matrix(rows, model) @ model["parameters"]
        summary[split] = {"strong": screening_metrics(rows, ls >= strong), "any_flag": screening_metrics(rows, ls >= review)}
    write_json(target, profile)
    write_json(EVIDENCE / "selection.json", {"selected": chosen, "candidates": candidates, "fit_and_calibration": summary,
                                            "test_used": False, "prior_test_outcomes_known": True})
    write_json(EVIDENCE / "FREEZE.json", {"profile_sha256": digest(target.read_bytes()), "train_dataset_sha256": digest(path.read_bytes()),
                                         "campaign_sha256": digest((EVIDENCE / "campaign.json").read_bytes()),
                                         "selection_sha256": digest((EVIDENCE / "selection.json").read_bytes()),
                                         "test_rule": "Post-freeze repeat evaluation only; preserve parameters after viewing results."})
    print(json.dumps({"selected": {k: v for k, v in chosen.items() if k != "folds"}, "fit_and_calibration": summary}), flush=True)


def spatial_metrics(rows, flags, samples):
    flags = [bool(f) for f in flags]
    ai_total = sum(r["ai_chars"] for r in rows)
    human_total = sum(r["chars"]-r["ai_chars"] for r in rows)
    blocks_total, blocks_hit, documents_hit = 0, 0, 0
    for sid in sorted({r["sample_id"] for r in rows}):
        mapping = read_json(DEFAULT_DIR / samples[sid]["mapping_file"])["blocks"]
        flagged = [(r["char_start"], r["char_end"]) for r, f in zip(rows, flags) if f and r["sample_id"] == sid]
        documents_hit += any(b["source_label"] == "ai" and
                             any(min(end, b["char_end"]) > max(start, b["char_start"]) for start, end in flagged)
                             for b in mapping)
        for block in mapping:
            if block["source_label"] == "ai":
                blocks_total += 1
                blocks_hit += any(min(b, block["char_end"]) > max(a, block["char_start"]) for a, b in flagged)
    return {"mixed_documents": len({r["sample_id"] for r in rows}), "segments": len(rows),
            "mixed_documents_with_any_flag": len({r["sample_id"] for r, f in zip(rows, flags) if f}),
            "mixed_documents_with_generated_position_flagged": documents_hit,
            "generated_blocks": blocks_total, "generated_blocks_with_any_flag": blocks_hit,
            "ai_characters": ai_total, "ai_flagged_characters": sum(r["ai_chars"] for r, f in zip(rows, flags) if f),
            "ai_character_recall": sum(r["ai_chars"] for r, f in zip(rows, flags) if f)/ai_total if ai_total else None,
            "human_characters": human_total,
            "human_characters_within_flagged_segments": sum(r["chars"]-r["ai_chars"] for r, f in zip(rows, flags) if f),
            "pure_human_segments": sum(r["ai_chars"] == 0 for r in rows),
            "pure_human_flagged_segments": sum(r["ai_chars"] == 0 and f for r, f in zip(rows, flags)),
            "note": "Whole mixed spans are marked; adjacent human characters are spillover, not estimated AI authorship."}


def evaluate():
    from aigc_detector.fusion import fused_score, validate_profile
    target = ROOT / "aigc_detector/profiles" / f"{POLICY}.json"
    profile = read_json(target)
    validate_profile(profile)
    before = digest(target.read_bytes())
    freeze = read_json(EVIDENCE / "FREEZE.json")
    if before != freeze["profile_sha256"]:
        raise ValueError("Frozen profile changed")
    train_path = EVIDENCE / f"context{profile['segment_chars']}" / "train-scores.json"
    if digest(train_path.read_bytes()) != freeze["train_dataset_sha256"] or profile["fit"]["dataset_sha256"] != freeze["train_dataset_sha256"]:
        raise ValueError("Frozen training dataset changed")
    train, repeat = read_json(train_path), read_json(EVIDENCE / "repeat-scores.json")
    if repeat["config"]["frozen_profile_sha256"] != before or repeat["config"]["scored_splits"] != ["test"]:
        raise ValueError("Repeat scores must be computed after freezing")
    if (repeat["config"]["segment_chars"], repeat["config"]["min_segment_chars"]) != (profile["segment_chars"], profile["min_segment_chars"]):
        raise ValueError("Repeat segmentation mismatch")
    samples = {s["id"]: s for s in verify(DEFAULT_DIR)[0]}
    rows = train["records"] + repeat["records"]
    owners = {}
    for r in rows:
        owners.setdefault(r["sha256"], set()).add(r["paper_id"])
    for r in rows:
        r["cross_paper_duplicate"] = len(owners[r["sha256"]]) > 1
    groups = {}
    for split in ("fit", "calibration", "test"):
        selected = eligible({"records": rows}, split, deduplicate=False)
        values = [fused_score(r["scores"], profile) for r in selected]
        strong = np.array([s["score"] is not None and s["score"] >= profile["high_threshold"] for s in values])
        any_flag = strong | np.array([s["review_score"] is not None and s["review_score"] >= profile["possible_threshold"] for s in values])
        pure_mask = np.array([r["kind"] != "mixed_document" for r in selected])
        mixed_mask = ~pure_mask
        pure = [r for r in selected if r["kind"] != "mixed_document"]
        mixed = [r for r in selected if r["kind"] == "mixed_document"]
        group = {"source_inputs": {k: len({r["sample_id"] for r in selected if r["kind"] == k})
                                  for k in ("human_document", "mixed_document", "ai_paragraph")},
                 "segments": len(selected), "pure_inputs": {}, "mixed_documents": {}, "original_documents": {}}
        for name, flags in (("strong", strong), ("any_flag", any_flag)):
            group["pure_inputs"][name] = binary_metrics(pure, flags[pure_mask])
            group["mixed_documents"][name] = spatial_metrics(mixed, flags[mixed_mask], samples)
            originals = {r["sample_id"] for r in selected if r["kind"] == "human_document"}
            flagged = {r["sample_id"] for r, f in zip(selected, flags) if f and r["kind"] == "human_document"}
            group["original_documents"][name] = {"documents": len(originals), "documents_with_any_flag": len(flagged)}
        groups[split] = group
    old = read_json(ROOT / "evaluations/english-academic-20261003-v1/evaluation.json")
    result = {"status": "complete", "profile_sha256": before, "train_dataset_sha256": freeze["train_dataset_sha256"],
              "repeat_dataset_sha256": digest((EVIDENCE / "repeat-scores.json").read_bytes()), "groups": groups,
              "full_collection": {"original_documents": sum(g["source_inputs"]["human_document"] for g in groups.values()),
                                  "mixed_documents": sum(g["source_inputs"]["mixed_document"] for g in groups.values()),
                                  "isolated_generated_paragraphs": sum(g["source_inputs"]["ai_paragraph"] for g in groups.values()),
                                  "scored_segments": sum(g["segments"] for g in groups.values())},
              "old_v1_fixed_1600_baseline": {"profile_sha256": old["profile_sha256"], "test": old["groups"]["test"]["new_any_flag"],
                                             "mixed_test": old["groups"]["test"]["mixed_documents"]["new_any_flag"]},
              "prior_v1_test_results_known": True, "repeat_scores_used_for_parameter_or_candidate_selection": False,
              "is_new_independent_validation": False,
              "interpretation": "All ten originals, thirty mixed documents and seventy-five isolated generated paragraphs are accounted for. Paper groups are the split unit. Fit/CV and calibration are reference evidence; P07/P08 are repeated evaluation, not a fresh independent test. Character spillover and paragraph hits are reported separately.",
              "contains_source_text": False, "cross_paper_duplicate_segments_excluded": sum(r["cross_paper_duplicate"] for r in rows)}
    if digest(target.read_bytes()) != before:
        raise ValueError("Profile changed during evaluation")
    write_json(EVIDENCE / "evaluation.json", result)
    print(json.dumps({"full_collection": result["full_collection"], "repeat_test": groups["test"], "profile_sha256": before}), flush=True)


def reproduce():
    """Rebuild parameters/thresholds using only the frozen numeric evidence."""
    target = ROOT / "aigc_detector/profiles" / f"{POLICY}.json"
    profile = read_json(target)
    freeze = read_json(EVIDENCE / "FREEZE.json")
    before = digest(target.read_bytes())
    if before != freeze["profile_sha256"]:
        raise ValueError("Frozen profile changed")
    path = EVIDENCE / f"context{profile['segment_chars']}" / "train-scores.json"
    if digest(path.read_bytes()) != freeze["train_dataset_sha256"]:
        raise ValueError("Frozen score dataset changed")
    data = read_json(path)
    train, calibration = eligible(data, "fit"), eligible(data, "calibration")
    fit = profile["fit"]
    model = fit_model(train, profile["models"], fit["family"], fit["regularization"], fit["human_weight"], fit["target_mode"])
    assert model["center"] == profile["center"] and model["scale"] == profile["scale"]
    assert model["terms"] == profile["feature_terms"]
    for head in profile["heads"].values():
        assert model["parameters"][0] == head["intercept"] and model["parameters"][1:] == head["coefficients"]
    controls = train + calibration
    values = matrix(controls, model) @ model["parameters"]
    for name, threshold in [("strong", "high_threshold"), ("review", "possible_threshold")]:
        cutoff = control_threshold(controls, values, fit["threshold_logit_margins"][name])
        assert float(sigmoid(cutoff)) == profile[threshold]
    assert digest(target.read_bytes()) == before
    result = {"status": "complete", "profile_sha256": before, "parameters_and_thresholds_reproduce_exactly": True,
              "source_text_read": False, "model_inference_run": False, "test_used": False}
    write_json(REPORTS / "reproduction.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score", "diagnose", "search", "evaluate", "reproduce"])
    parser.add_argument("--segment-chars", type=int, default=1600)
    parser.add_argument("--min-segment-chars", type=int, default=400)
    parser.add_argument("--splits", nargs="+", choices=["fit", "calibration", "test"], default=["fit", "calibration"])
    parser.add_argument("--kinds", nargs="+", choices=["human_document", "ai_paragraph", "mixed_document"], default=["human_document", "ai_paragraph", "mixed_document"])
    parser.add_argument("--signals", nargs="+", choices=SIGNALS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "search":
        search()
    elif args.command == "diagnose":
        search_context(args.segment_chars, args.min_segment_chars)
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "reproduce":
        reproduce()
    else:
        score(args.segment_chars, args.min_segment_chars, splits=args.splits, kinds=args.kinds, signals=args.signals, output=args.output)


if __name__ == "__main__":
    main()
