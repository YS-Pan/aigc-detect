"""Offline score-only calibration with paper-grouped fitting and frozen testing."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import socket
import shutil
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.annotation import segment_spans
from aigc_detector.catalog import ENGLISH_MODELS, ROOT
from scripts.english_academic_samples import DEFAULT_DIR, digest, read_json, verify, write_json
from scripts.fit_fusion_v2 import fit_head, sigmoid, standardizer, design, feature_terms

POLICY = "english-academic-20261003-v1"
EVIDENCE = ROOT / "evaluations" / POLICY


def score(samples, output, report_dir, segment_chars=400, min_segment_chars=120,
          splits=("fit", "calibration"), profile_path=None):
    from aigc_detector.engine import Detector
    import torch
    frozen_profile_hash = None
    if "test" in splits:
        profile_path = profile_path or ROOT / "aigc_detector/profiles" / f"{POLICY}.json"
        freeze = read_json(EVIDENCE / "FREEZE.json")
        frozen_profile_hash = digest(profile_path.read_bytes())
        if freeze["profile_sha256"] != frozen_profile_hash:
            raise ValueError("Test inference requires an unchanged frozen profile")
        profile = read_json(profile_path)
        if [segment_chars, min_segment_chars] != [profile["segment_chars"], profile["min_segment_chars"]]:
            raise ValueError("Test inference must use frozen segmentation")
    records, collection, protocol = verify(samples)
    records = [r for r in records if r["split"] in splits]
    protocol = {**protocol, "segment_chars": segment_chars, "min_segment_chars": min_segment_chars,
                "candidate_feature_families": ["linear", "squares", "quadratic"],
                "threshold_logit_margins": {"review": .01, "strong": .1},
                "protocol_amendment": "After fit-only diagnosis of weak 400-character separation, compare longer contexts and polynomial score terms. Test metrics remain unread."}
    report_dir.mkdir(parents=True, exist_ok=False)
    items, passages = [], {}
    for sample in records:
        blocks = read_json(samples / sample["mapping_file"])["blocks"] if "mapping_file" in sample else []
        ai_spans = [(b["char_start"], b["char_end"]) for b in blocks if b["source_label"] == "ai"]
        for n, (start, end) in enumerate(segment_spans(sample["text"], segment_chars, min_segment_chars, language="en")):
            text = sample["text"][start:end]
            key = digest(text.encode("utf-8"))
            passages.setdefault(key, text)
            ai_chars = len(text) if sample["kind"] == "ai_paragraph" else sum(max(0, min(end, b)-max(start, a)) for a, b in ai_spans)
            items.append({"id": f"{sample['id']}:s{n+1:04}", "sample_id": sample["id"], "paper_id": sample["paper_id"],
                          "split": sample["split"], "kind": sample["kind"], "generator": sample.get("generator"),
                          "paragraph_id": sample.get("paragraph_id"), "sha256": key,
                          "char_start": start, "char_end": end, "chars": end-start, "ai_chars": ai_chars})
    # A repeated passage from a different source paper is excluded from all fit
    # and test metrics, retaining its raw scores for audit.
    owners = {}
    for item in items:
        owners.setdefault(item["sha256"], set()).add(item["paper_id"])
    for item in items:
        item["cross_paper_duplicate"] = len(owners[item["sha256"]]) > 1
    detector = Detector()
    revisions = {m["id"]: m["revision"] for m in detector.list_models() if m["id"] in ENGLISH_MODELS}
    config = {"device": detector.device, "models": list(ENGLISH_MODELS), "revisions": revisions,
              "scored_splits": list(splits),
              "collection_sha256": collection["samples_sha256"], "protocol_sha256": collection["protocol_sha256"],
              "groups": protocol["groups"], "manifest_sha256": digest((ROOT / "models/manifest.json").read_bytes()),
              "network_blocked_during_inference": True, "text_used_in_fusion_features": False,
              "source_code_sha256": {f: digest((ROOT / f).read_bytes()) for f in
                                     ["scripts/english_academic_samples.py", "scripts/calibrate_english.py", "aigc_detector/annotation.py", "aigc_detector/engine.py"]}}
    if frozen_profile_hash:
        config["frozen_profile_sha256"] = frozen_profile_hash
    write_json(report_dir / "run-config.json", config)
    keys = list(passages)
    scores = {key: {} for key in keys}
    def blocked(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")
    print(json.dumps({"segments": len(items), "unique_passages": len(keys), "device": detector.device}), flush=True)
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for mid in ENGLISH_MODELS:
            cache = []
            for offset in range(0, len(keys), 64):
                batch = keys[offset:offset+64]
                values = detector._score_texts([passages[k] for k in batch], mid, 64, True, allow_empty=True)
                for key, value in zip(batch, values):
                    if value["revision"] != revisions[mid]:
                        raise ValueError("Model revision changed during inference")
                    scores[key][mid] = value["ai_score"]
                    cache.append({"sha256": key, **value})
                if offset % 256 == 0 or offset+64 >= len(keys):
                    print(f"{mid}: {min(offset+64,len(keys))}/{len(keys)}", flush=True)
            write_json(report_dir / f"{mid}-raw.json", cache)
            detector._loaded.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    for item in items:
        item["scores"] = scores[item["sha256"]]
    dataset = {"schema_version": "english-academic-scores/1.0", "config": config,
               "protocol": protocol, "records": items, "contains_source_text": False}
    if frozen_profile_hash and digest(profile_path.read_bytes()) != frozen_profile_hash:
        raise ValueError("Profile changed during test inference")
    write_json(output, dataset)
    write_json(report_dir / "COMPLETE.json", {"status": "complete", "dataset_sha256": digest(output.read_bytes()),
                                              "segments": len(items), "unique_passages": len(keys)})
    print(json.dumps({"status": "complete", "dataset": str(output)}), flush=True)


def eligible(dataset, splits):
    return [r for r in dataset["records"] if r["split"] in splits and r["kind"] in ("human_document", "ai_paragraph")
            and not r["cross_paper_duplicate"] and all(r["scores"][m] is not None for m in ENGLISH_MODELS)]


def raw_array(records):
    return np.array([[r["scores"][m] for m in ENGLISH_MODELS] for r in records])


def weights_for(records, human_weight):
    totals = {}
    for r in records:
        key = (r["paper_id"], r["kind"])
        totals[key] = totals.get(key, 0) + r["chars"]
    return np.array([r["chars"] / totals[r["paper_id"], r["kind"]]
                     * (human_weight if r["kind"] == "human_document" else 1) for r in records])


def fit_records(records, regularization, human_weight, family="linear"):
    raw = raw_array(records)
    weights = weights_for(records, human_weight)
    center, scale = standardizer(raw, weights)
    terms = feature_terms(len(ENGLISH_MODELS), family)
    x = design(raw, center, scale, terms)
    target = np.array([float(r["kind"] == "ai_paragraph") for r in records])
    parameters, info = fit_head(x, target, weights, terms, regularization)
    return {"center": center.tolist(), "scale": scale.tolist(), "parameters": parameters.tolist(), "terms": terms, "optimization": info}


def logits(records, model):
    x = design(raw_array(records), np.array(model["center"]), np.array(model["scale"]), model["terms"])
    return x @ np.array(model["parameters"])


def cutpoint(records, values, margin=.5):
    human = [float(v) for r, v in zip(records, values) if r["kind"] == "human_document"]
    if not human:
        raise ValueError("Human threshold controls are required")
    return max(human) + margin


def binary_metrics(records, flags):
    human = np.array([r["kind"] == "human_document" for r in records])
    ai = ~human
    chars = np.array([r["chars"] for r in records])
    ai_samples = {r["sample_id"] for r in records if r["kind"] == "ai_paragraph"}
    detected = {r["sample_id"] for r, flag in zip(records, flags) if flag and r["kind"] == "ai_paragraph"}
    by_generator = {}
    for generator in sorted({r["generator"] for r in records if r["generator"]}):
        mask = np.array([r["generator"] == generator for r in records])
        by_generator[generator] = {"segments": int(mask.sum()), "flagged": int((mask & flags).sum()),
                                   "character_recall": float(chars[mask & flags].sum()/chars[mask].sum())}
    return {"human_segments": int(human.sum()), "human_flagged_segments": int((human & flags).sum()),
            "human_characters": int(chars[human].sum()), "human_flagged_characters": int(chars[human & flags].sum()),
            "human_segment_fpr": float((human & flags).sum()/human.sum()) if human.any() else None,
            "human_papers": len({r["paper_id"] for r in records if r["kind"] == "human_document"}),
            "ai_segments": int(ai.sum()), "ai_flagged_segments": int((ai & flags).sum()),
            "ai_characters": int(chars[ai].sum()), "ai_flagged_characters": int(chars[ai & flags].sum()),
            "ai_character_recall": float(chars[ai & flags].sum()/chars[ai].sum()) if ai.any() else None,
            "ai_paragraphs": len(ai_samples), "ai_paragraphs_with_any_flag": len(detected), "by_generator": by_generator}


def fit(dataset_path, profile_path, diagnostics_path):
    dataset = read_json(dataset_path)
    records = eligible(dataset, ["fit"])
    calibration = eligible(dataset, ["calibration"])
    protocol = dataset["protocol"]
    candidates = []
    for family in protocol.get("candidate_feature_families", ["linear", "squares", "quadratic"]):
      for reg in protocol["candidate_regularization"]:
        for hw in protocol["candidate_human_weights"]:
            folds = []
            for paper in protocol["groups"]["fit"]:
                train = [r for r in records if r["paper_id"] != paper]
                held = [r for r in records if r["paper_id"] == paper]
                model = fit_records(train, reg, hw, family)
                threshold = cutpoint(train, logits(train, model), .01)
                folds.append({"paper": paper, "threshold_logit": threshold,
                              "metrics": binary_metrics(held, logits(held, model) >= threshold)})
            fp_chars = sum(f["metrics"]["human_flagged_characters"] for f in folds)
            tp_chars = sum(f["metrics"]["ai_flagged_characters"] for f in folds)
            candidates.append({"family": family, "regularization": reg, "human_weight": hw, "folds": folds,
                               "human_flagged_characters": fp_chars, "ai_flagged_characters": tp_chars})
    # Hard priority for false positives; infeasible zero-flag CV is reported.
    chosen = min(candidates, key=lambda c: (c["human_flagged_characters"], -c["ai_flagged_characters"], -c["regularization"],
                                          len(feature_terms(4,c["family"])), -c["human_weight"]))
    model = fit_records(records, chosen["regularization"], chosen["human_weight"], chosen["family"])
    controls = records + calibration
    values = logits(controls, model)
    review_logit = cutpoint(controls, values, .01)
    strong_logit = cutpoint(controls, values, .1)
    head = {"intercept": model["parameters"][0], "coefficients": model["parameters"][1:]}
    profile = {"schema_version": "aigc-score-fusion/2.0", "id": POLICY, "language": "en",
               "segmentation_id": "english-natural-boundaries-v1", "models": list(ENGLISH_MODELS),
               "revisions": dataset["config"]["revisions"], "segment_chars": protocol["segment_chars"], "min_segment_chars": protocol["min_segment_chars"],
               "clip_probability": 1e-5, "clip_standardized": 5.0,
               "center": model["center"], "scale": model["scale"], "feature_terms": model["terms"],
               "heads": {"strong": head, "review": head}, "high_threshold": float(sigmoid(strong_logit)),
               "possible_threshold": float(sigmoid(review_logit)),
               "fit": {"dataset_sha256": digest(dataset_path.read_bytes()), "protocol_sha256": dataset["config"]["protocol_sha256"],
                       "groups": protocol["groups"], "family": chosen["family"], "regularization": chosen["regularization"], "human_weight": chosen["human_weight"],
                       "optimization": model["optimization"], "fit_records": len(records), "calibration_records": len(calibration),
                       "threshold_controls": "Human segments from fit and calibration only; no test score or AI calibration score influences thresholds.",
                       "threshold_logit_margins": {"strong": .1, "review": .01},
                       "head_relationship": "Identical score polynomial in both heads; distinct conservative thresholds for strong and review tiers.",
                       "cv_zero_human_flags_feasible": chosen["human_flagged_characters"] == 0,
                       "test_used_for_fitting_or_selection": False},
               "limitations": ["Single-subject academic corpus of ten source papers; three related generation models.",
                               "Source-derived human text includes PDF-to-LaTeX conversion uncertainty.",
                               "Zero observed control flags does not establish a zero population false-positive rate.",
                               "Sigmoid scores are screening scores, not authorship probabilities.",
                               "Different preprocessing, subjects, lengths or revisions require new verification."]}
    if not 0 <= profile["possible_threshold"] < profile["high_threshold"] <= 1:
        raise ValueError("Thresholds saturated; no usable conservative operating point")
    write_json(profile_path, profile)
    summary = {}
    for group, selected in [("fit", records), ("calibration", calibration)]:
        ls = logits(selected, model)
        summary[group] = {"strong": binary_metrics(selected, ls >= strong_logit), "any_flag": binary_metrics(selected, ls >= review_logit)}
    diagnostics = {"profile_sha256": digest(profile_path.read_bytes()), "selected": chosen,
                   "candidates": candidates, "fit_and_calibration": summary, "test_evaluated": False,
                   "protocol_amendment": {"families": ["linear", "squares", "quadratic"], "logit_margins": {"review": .01, "strong": .1},
                                          "reason": "Fit-only extension documented by campaign.json; test metrics excluded."},
                   "selection_priority": "False-positive characters first, AI true-positive characters second; no test data used."}
    write_json(diagnostics_path, diagnostics)
    print(json.dumps({"profile": str(profile_path), "selected": {k: chosen[k] for k in ["regularization", "human_weight", "human_flagged_characters"]},
                      "fit_and_calibration": summary}, ensure_ascii=False), flush=True)


def select(profile_path):
    if (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Existing frozen evidence must be preserved; use a new versioned campaign for reselection")
    campaign = read_json(EVIDENCE / "campaign.json")
    candidates = []
    for length, minimum in campaign["segmentation_candidates"]:
        directory = EVIDENCE / f"context{length}"
        diagnostics = read_json(directory / "fit.json")
        selected = diagnostics["selected"]
        candidates.append({"segment_chars": length, "min_segment_chars": minimum,
                           "family": selected["family"], "human_flagged_characters": selected["human_flagged_characters"],
                           "ai_flagged_characters": selected["ai_flagged_characters"],
                           "profile_sha256": digest((directory / "profile.json").read_bytes()),
                           "diagnostics_sha256": digest((directory / "fit.json").read_bytes()),
                           "directory": str(directory.relative_to(ROOT))})
    chosen = min(candidates, key=lambda c: (c["human_flagged_characters"], -c["ai_flagged_characters"], c["segment_chars"]))
    directory = ROOT / chosen["directory"]
    selected_profile = read_json(directory / "profile.json")
    selected_profile["fit"]["campaign_sha256"] = digest((EVIDENCE / "campaign.json").read_bytes())
    selected_profile["fit"]["context_selection"] = "Fit-paper CV: false-positive characters first, generated-character recall second; test not evaluated."
    write_json(profile_path, selected_profile)
    shutil.copy2(directory / "scores.json", EVIDENCE / "train-scores.json")
    write_json(EVIDENCE / "selection.json", {"candidates": candidates, "selected": chosen, "test_used": False})
    write_json(EVIDENCE / "FREEZE.json", {"profile_sha256": digest(profile_path.read_bytes()),
                                          "train_dataset_sha256": digest((EVIDENCE / "train-scores.json").read_bytes()),
                                          "selection_sha256": digest((EVIDENCE / "selection.json").read_bytes()),
                                          "test_metrics_inspected": False,
                                          "test_rule": "Run new test-only inference at frozen segmentation; no parameter/threshold changes after evaluation."})
    print(json.dumps({"selected": chosen, "profile_sha256": digest(profile_path.read_bytes())}), flush=True)


def evaluate(dataset_path, profile_path, output, test_dataset_path=None):
    from aigc_detector.fusion import fused_score, validate_profile
    dataset, profile = read_json(dataset_path), read_json(profile_path)
    validate_profile(profile)
    if profile["fit"]["dataset_sha256"] != digest(dataset_path.read_bytes()):
        raise ValueError("Profile fitted on a different score dataset")
    digest_before = digest(profile_path.read_bytes())
    if test_dataset_path:
        freeze = read_json(EVIDENCE / "FREEZE.json")
        test = read_json(test_dataset_path)
        if freeze["profile_sha256"] != digest_before or test["config"].get("frozen_profile_sha256") != digest_before:
            raise ValueError("Test data must come from unchanged post-freeze inference")
        if test["config"]["scored_splits"] != ["test"] or test["config"]["revisions"] != profile["revisions"]:
            raise ValueError("Invalid held-out test dataset")
        dataset["records"] = [r for r in dataset["records"] if r["split"] != "test"] + test["records"]
    owners = {}
    for r in dataset["records"]:
        owners.setdefault(r["sha256"], set()).add(r["paper_id"])
    for r in dataset["records"]:
        r["cross_paper_duplicate"] = len(owners[r["sha256"]]) > 1
    groups = {}
    for group in ["fit", "calibration", "test"]:
        records = eligible(dataset, [group])
        scores = np.array([fused_score(r["scores"], profile)["score"] for r in records])
        old = np.array([max(r["scores"].values()) for r in records])
        groups[group] = {"new_strong": binary_metrics(records, scores >= profile["high_threshold"]),
                         "new_any_flag": binary_metrics(records, scores >= profile["possible_threshold"]),
                         "old_strong": binary_metrics(records, old >= .9), "old_any_flag": binary_metrics(records, old >= .7)}
        mixed = [r for r in dataset["records"] if r["split"] == group and r["kind"] == "mixed_document" and not r["cross_paper_duplicate"]]
        sc = [fused_score(r["scores"], profile)["score"] for r in mixed]
        spatial = {}
        for name, flags in [("new_strong", [s is not None and s >= profile["high_threshold"] for s in sc]),
                            ("new_any_flag", [s is not None and s >= profile["possible_threshold"] for s in sc]),
                            ("old_any_flag", [max(v for v in r["scores"].values() if v is not None) >= .7 for r in mixed])]:
            ai_total = sum(r["ai_chars"] for r in mixed)
            human_total = sum(r["chars"]-r["ai_chars"] for r in mixed)
            spatial[name] = {"ai_characters": ai_total, "ai_flagged_characters": sum(r["ai_chars"] for r, f in zip(mixed, flags) if f),
                             "ai_character_recall": sum(r["ai_chars"] for r, f in zip(mixed, flags) if f)/ai_total if ai_total else None,
                             "human_characters": human_total,
                             "human_characters_within_flagged_segments": sum(r["chars"]-r["ai_chars"] for r, f in zip(mixed, flags) if f),
                             "pure_human_segments": sum(r["ai_chars"] == 0 for r in mixed),
                             "pure_human_flagged_segments": sum(r["ai_chars"] == 0 and f for r, f in zip(mixed, flags)),
                             "note": "Mixed-boundary segments flag whole spans; human characters sharing an AI segment are counted as spillover."}
        groups[group]["mixed_documents"] = spatial
    excluded = sum(r["cross_paper_duplicate"] for r in dataset["records"])
    result = {"status": "complete", "profile_sha256": digest_before, "dataset_sha256": digest(dataset_path.read_bytes()),
              "test_dataset_sha256": digest(test_dataset_path.read_bytes()) if test_dataset_path else None,
              "groups": groups, "cross_paper_duplicate_segments_excluded": excluded,
              "test_independent_of_fitting_and_selection": True,
              "uncertainty": "Test is two entire papers from the same subject. Segments within a paper are correlated; segment counts are not independent trials.",
              "zero_fp_paper_level_95pct_upper_bound_if_both_test_papers_unflagged": 1-math.pow(.05, 1/2),
              "contains_source_text": False}
    if digest_before != digest(profile_path.read_bytes()):
        raise ValueError("Profile changed during test evaluation")
    write_json(output, result)
    print(json.dumps({"test": groups["test"], "profile_sha256": digest_before}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["score", "fit", "select", "evaluate"])
    parser.add_argument("--samples", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dataset", type=Path, default=EVIDENCE / "train-scores.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "aigc_detector/profiles" / f"{POLICY}.json")
    parser.add_argument("--diagnostics", type=Path, default=EVIDENCE / "fit.json")
    parser.add_argument("--evaluation", type=Path, default=EVIDENCE / "evaluation.json")
    parser.add_argument("--test-dataset", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--segment-chars", type=int, default=400)
    parser.add_argument("--min-segment-chars", type=int, default=120)
    parser.add_argument("--splits", nargs="+", choices=["fit", "calibration", "test"], default=["fit", "calibration"])
    args = parser.parse_args()
    if args.command == "score":
        if args.report_dir is None:
            parser.error("score requires --report-dir")
        score(args.samples, args.dataset, args.report_dir, args.segment_chars, args.min_segment_chars, args.splits, args.profile)
    elif args.command == "fit":
        fit(args.dataset, args.profile, args.diagnostics)
    elif args.command == "select":
        select(args.profile)
    else:
        evaluate(args.dataset, args.profile, args.evaluation, args.test_dataset)


if __name__ == "__main__":
    main()
