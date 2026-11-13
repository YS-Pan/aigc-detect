"""Declared nonlinear score-only campaign, paper-held-out selection and repeats."""
import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

for setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(setting, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts import calibrate_english_v2 as previous
from scripts.calibrate_english_v2 import ROOT, CLASSIFIERS, STATISTICS, SIGNALS, eligible, control_threshold, screening_metrics
from scripts.english_academic_samples import read_json, write_json, digest, DEFAULT_DIR, verify
from scripts.fit_fusion_v2 import sigmoid
from scripts import nonlinear_english_fit as fitter
from scripts.calibrate_english import binary_metrics

POLICY = "english-academic-20261004-v3"
EVIDENCE = ROOT / "evaluations" / POLICY
REPORTS = ROOT / "reports/english-nonlinear-20261004-v3"
PARENT = ROOT / "evaluations/english-academic-20261003-v2"
PARENT_REPORTS = ROOT / "reports/english-recalibration-20261003-v2"
PROFILE = ROOT / "aigc_detector/profiles" / f"{POLICY}.json"


def prepare():
    if (EVIDENCE / "campaign.json").exists():
        raise FileExistsError("Preserve declared nonlinear campaign")
    parent = read_json(PARENT / "campaign.json")
    sources = {}
    for length in (400, 800, 1600):
        path = PARENT / f"context{length}/train-scores.json"
        data = read_json(path)
        if data["config"]["scored_splits"] != ["fit", "calibration"] or set(data["config"]["signals"]) != set(SIGNALS):
            raise ValueError("Complete fit/calibration score evidence required")
        sources[str(length)] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes())}
    campaign = {"schema_version": "english-nonlinear-campaign/1.0", "id": POLICY,
                "groups": parent["groups"], "sample_collection_sha256": parent["sample_collection_sha256"],
                "segmentations": parent["segmentations"], "revisions": parent["revisions"],
                "asset_revisions": parent["asset_revisions"], "statistical_inference": parent["statistical_inference"],
                "raw_score_sources": sources, "feature_rosters": [parent["feature_rosters"][1], CLASSIFIERS, SIGNALS],
                "rbf": {"centers": [16, 48], "widths": [.5, 1., 2.], "regularizations": [1., 10.],
                        "human_weights": [1., 20.], "target_modes": ["fraction", "presence"]},
                "mlp": {"hidden": [4, 12], "regularizations": [1., 10.], "human_weights": [1., 20.],
                        "target_modes": ["fraction", "presence", "ranking_presence"]},
                "optimizer": {"seeds": [20261004, 20261005], "adam_steps": 160, "adam_learning_rate": .03,
                              "lbfgs_steps": 60, "ranking_margin": .1, "ranking_weight": 1.},
                "threshold_logit_margins": {"review": .01, "strong": .1},
                "selection": "Fit-only leave-one-paper-out: minimize pure-human flagged characters, maximize mixed generated characters, then all generated characters; ties prefer fewer parameters and stronger regularization.",
                "thresholds": "Fit+calibration pure-human maximum logit + declared margins; calibration generated scores unused.",
                "candidate_count": 648, "prior_test_outcomes_known": True, "new_independent_validation": False,
                "test_used_for_fitting_or_selection": False,
                "promotion_rule": "Zero fit-CV human flags; repeat originals and pure-human mixed flags zero; repeat isolated paragraph hits at least v2, mixed generated character coverage strictly exceeds v2. Promotion is a repeat-based operational decision, not independent validation.",
                "runtime_features": "Pinned raw classifier/LM outputs only; no text, author, length or document identity.",
                "contains_source_text": False,
                "previous_profile_sha256": digest((ROOT / "aigc_detector/profiles/english-academic-20261003-v2.json").read_bytes())}
    write_json(EVIDENCE / "campaign.json", campaign)
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"campaign": str(EVIDENCE / "campaign.json"), "candidates": 648}), flush=True)


def configurations(campaign):
    for length, minimum in campaign["segmentations"]:
      for signals in campaign["feature_rosters"]:
        for centers, width, reg, hw, target in itertools.product(*[campaign["rbf"][k] for k in
                                            ("centers", "widths", "regularizations", "human_weights", "target_modes")]):
            yield {"family": "rbf", "segment_chars": length, "min_segment_chars": minimum, "signals": signals,
                   "centers": centers, "width": width, "regularization": reg, "human_weight": hw, "target_mode": target}
        for hidden, reg, hw, target in itertools.product(*[campaign["mlp"][k] for k in
                                              ("hidden", "regularizations", "human_weights", "target_modes")]):
            yield {"family": "tanh_mlp", "segment_chars": length, "min_segment_chars": minimum, "signals": signals,
                   "hidden": hidden, "regularization": reg, "human_weight": hw, "target_mode": target}


def metrics(records, flags):
    result = screening_metrics(records, flags)
    mixed = [r for r in records if r["kind"] == "mixed_document"]
    mask = np.array([r["kind"] == "mixed_document" for r in records])
    m = screening_metrics(mixed, np.asarray(flags)[mask])
    result.update({"mixed_ai_characters": m["ai_characters"], "mixed_ai_flagged_characters": m["ai_flagged_characters"],
                   "mixed_human_spillover": m["human_characters_in_mixed_flagged_segments"]})
    return result


def source_data(campaign, length):
    source = campaign["raw_score_sources"][str(length)]
    path = ROOT / source["path"]
    if digest(path.read_bytes()) != source["sha256"]:
        raise ValueError("Declared source scores changed")
    return read_json(path)


def search():
    if (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Frozen nonlinear campaign cannot be reselected")
    campaign = read_json(EVIDENCE / "campaign.json")
    code_hashes = {f: digest((ROOT / f).read_bytes()) for f in ["scripts/calibrate_english_nonlinear.py", "scripts/nonlinear_english_fit.py"]}
    cases = list(configurations(campaign))
    assert len(cases) == campaign["candidate_count"]
    data = {length: source_data(campaign, length) for length, _ in campaign["segmentations"]}
    fit = {length: eligible(d, "fit") for length, d in data.items()}
    held = {length: eligible(d, "fit", deduplicate=False) for length, d in data.items()}
    prepared, candidates = {}, []
    for number, config in enumerate(cases, 1):
        token = digest(json.dumps(config, sort_keys=True).encode())
        path = REPORTS / "candidate-cache" / f"{token}.json"
        if path.exists():
            result = read_json(path)
            if result["campaign_sha256"] != digest((EVIDENCE / "campaign.json").read_bytes()) or result["fitter_code_sha256"] != code_hashes:
                raise ValueError("Candidate cache does not match declared run")
            candidates.append(result["candidate"])
            continue
        folds = []
        length, signals = config["segment_chars"], config["signals"]
        for paper in campaign["groups"]["fit"]:
            train = [r for r in fit[length] if r["paper_id"] != paper]
            validation = [r for r in held[length] if r["paper_id"] == paper]
            key = (length, tuple(signals), config["human_weight"], paper)
            if key not in prepared:
                prepared[key] = fitter.standardize(train, signals, config["human_weight"])
            model = fitter.fit(train, signals, config, campaign["optimizer"], prepared=prepared[key])
            cutoff = control_threshold(train, fitter.predict(train, model), campaign["threshold_logit_margins"]["review"])
            diagnostic = metrics(validation, fitter.predict(validation, model) >= cutoff)
            folds.append({"paper": paper, "threshold_logit": cutoff, "metrics": diagnostic,
                          "optimization": {k: v for k, v in model["optimization"].items() if k != "starts" and k != "trace"}})
        candidate = {"config": config, "folds": folds, "parameter_count": model["optimization"]["parameter_count"],
                     **{k: sum(f["metrics"][k] for f in folds) for k in
                        ("human_flagged_characters", "ai_flagged_characters", "mixed_ai_flagged_characters", "ai_characters", "mixed_ai_characters")}}
        write_json(path, {"campaign_sha256": digest((EVIDENCE / "campaign.json").read_bytes()),
                          "fitter_code_sha256": code_hashes, "candidate": candidate})
        candidates.append(candidate)
        print(json.dumps({"candidate": number, "total": len(cases), "context": length, "signals": len(signals),
                          "family": config["family"], "human_false_characters": candidate["human_flagged_characters"],
                          "mixed_generated_hit_characters": candidate["mixed_ai_flagged_characters"]}), flush=True)
        write_json(REPORTS / "search-progress.json", {"candidates_complete": number, "total": len(cases),
                   "best_so_far": min(candidates, key=objective)})
    if any(digest((ROOT / f).read_bytes()) != value for f, value in code_hashes.items()):
        raise ValueError("Fitter code changed during declared optimization")
    chosen = min(candidates, key=objective)
    config = chosen["config"]
    rows, calibration = fit[config["segment_chars"]], eligible(data[config["segment_chars"]], "calibration")
    model = fitter.fit(rows, config["signals"], config, campaign["optimizer"])
    controls = rows+calibration
    values = fitter.predict(controls, model)
    review, strong = [control_threshold(controls, values, campaign["threshold_logit_margins"][head]) for head in ("review", "strong")]
    signals = config["signals"]
    parent_path = ROOT / campaign["raw_score_sources"][str(config["segment_chars"])]["path"]
    train_path = EVIDENCE / "train-scores.json"
    train_path.write_bytes(parent_path.read_bytes())
    profile = {"id": POLICY, "schema_version": "aigc-score-fusion/4.0", "language": "en",
               "segmentation_id": "english-natural-boundaries-v1", "models": signals,
               "revisions": {s: campaign["revisions"][s] for s in signals if s in CLASSIFIERS},
               "statistical_assets": {s: {a: campaign["asset_revisions"][a] for a in previous.STATISTICAL_BACKENDS[s]}
                                      for s in signals if s in STATISTICS},
               "statistical_inference": campaign["statistical_inference"],
               "score_transforms": {s: "logit" if s in CLASSIFIERS else "negative_identity" if s == "binoculars-falcon" else "identity" for s in signals},
               "segment_chars": config["segment_chars"], "min_segment_chars": config["min_segment_chars"],
               "center": model["center"], "scale": model["scale"], "clip_probability": 1e-5, "clip_standardized": 5.,
               "heads": {"strong": model["head"], "review": copy.deepcopy(model["head"])},
               "high_threshold": float(sigmoid(strong)), "possible_threshold": float(sigmoid(review)),
               "fit": {"configuration": config, "groups": campaign["groups"], "optimizer": campaign["optimizer"],
                       "optimization": model["optimization"], "training_passages_after_deduplication": len(rows),
                       "calibration_passages_after_deduplication": len(calibration), "dataset_sha256": digest(train_path.read_bytes()),
                       "campaign_sha256": digest((EVIDENCE / "campaign.json").read_bytes()), "fitter_code_sha256": code_hashes,
                       "threshold_logit_margins": campaign["threshold_logit_margins"], "prior_test_outcomes_known": True,
                       "test_used_for_fitting_or_selection": False, "new_independent_validation": False,
                       "head_relationship": "Same optimized nonlinear function, different control-derived thresholds."},
               "limitations": ["Previously evaluated ten antenna papers; this is reference fitting and repeat evaluation.",
                               "No independent generalization or population zero-false-positive claim.",
                               "Nonlinear allocations are exact accounting, not causal importance; unflagged cannot exclude AIGC."]}
    if not 0 < profile["possible_threshold"] < profile["high_threshold"] < 1:
        raise ValueError("No finite conservative operating point")
    write_json(PROFILE, profile)
    write_json(EVIDENCE / "selection.json", {"selected": chosen, "candidates": candidates, "contains_source_text": False,
                                            "test_used_for_selection": False, "prior_test_outcomes_known": True})
    write_json(EVIDENCE / "FREEZE.json", {"profile_sha256": digest(PROFILE.read_bytes()),
               "train_dataset_sha256": digest(train_path.read_bytes()), "campaign_sha256": digest((EVIDENCE / "campaign.json").read_bytes()),
               "selection_sha256": digest((EVIDENCE / "selection.json").read_bytes()), "fitter_code_sha256": code_hashes,
               "parent_score_sources": campaign["raw_score_sources"], "test_rule": "Repeat only after freeze; no parameter changes afterward."})
    print(json.dumps({"frozen": str(PROFILE), "selected": {k: v for k, v in chosen.items() if k != "folds"}}), flush=True)


def objective(candidate):
    c = candidate["config"]
    return (candidate["human_flagged_characters"], -candidate["mixed_ai_flagged_characters"], -candidate["ai_flagged_characters"],
            candidate["parameter_count"], -c["regularization"], c["segment_chars"], json.dumps(c, sort_keys=True))


def repeat_score():
    freeze = read_json(EVIDENCE / "FREEZE.json")
    if digest(PROFILE.read_bytes()) != freeze["profile_sha256"]:
        raise ValueError("Frozen profile changed")
    profile = read_json(PROFILE)
    destination = REPORTS / f"context{profile['segment_chars']}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in [f"{m}-cache.json" for m in profile["models"] if m in CLASSIFIERS]+["statistical-cache.json"]:
        original, target = PARENT_REPORTS / destination.name / name, destination / name
        if original.exists() and not target.exists():
            shutil.copyfile(original, target)
    with patch.object(previous, "EVIDENCE", EVIDENCE), patch.object(previous, "REPORTS", REPORTS), patch.object(previous, "POLICY", POLICY):
        previous.score(profile["segment_chars"], profile["min_segment_chars"], splits=["test"], signals=profile["models"])


def evaluate():
    from aigc_detector.fusion import fused_score, validate_profile
    profile, freeze = read_json(PROFILE), read_json(EVIDENCE / "FREEZE.json")
    validate_profile(profile)
    assert digest(PROFILE.read_bytes()) == freeze["profile_sha256"]
    assert digest((EVIDENCE / "train-scores.json").read_bytes()) == freeze["train_dataset_sha256"]
    train, repeat = read_json(EVIDENCE / "train-scores.json"), read_json(EVIDENCE / "repeat-scores.json")
    assert repeat["config"]["frozen_profile_sha256"] == freeze["profile_sha256"]
    assert repeat["config"]["scored_splits"] == ["test"]
    assert digest((EVIDENCE / "campaign.json").read_bytes()) == freeze["campaign_sha256"]
    assert digest((EVIDENCE / "selection.json").read_bytes()) == freeze["selection_sha256"]
    samples = {s["id"]: s for s in verify(DEFAULT_DIR)[0]}
    rows = train["records"]+repeat["records"]
    owners = {}
    for r in rows:
        owners.setdefault(r["sha256"], set()).add(r["paper_id"])
    for r in rows:
        r["cross_paper_duplicate"] = len(owners[r["sha256"]]) > 1
    groups = {}
    for split in ("fit", "calibration", "test"):
        records = eligible({"records": rows}, split, deduplicate=False)
        logits = fitter.predict(records, {"signals": profile["models"], "center": np.array(profile["center"]),
                    "scale": np.array(profile["scale"]), "head": profile["heads"]["strong"]})
        runtime = [fused_score(r["scores"], profile) for r in records]
        assert max(abs(float(v)-r["logit"]) for v, r in zip(logits, runtime)) < 1e-9
        pure_mask = np.array([r["kind"] != "mixed_document" for r in records])
        pure, mixed = [r for r in records if r["kind"] != "mixed_document"], [r for r in records if r["kind"] == "mixed_document"]
        group = {"source_inputs": {k: len({r["sample_id"] for r in records if r["kind"] == k}) for k in
                                   ("human_document", "mixed_document", "ai_paragraph")}, "segments": len(records),
                 "pure_inputs": {}, "mixed_documents": {}, "original_documents": {}}
        for name, threshold in [("strong", profile["high_threshold"]), ("any_flag", profile["possible_threshold"])]:
            flags = sigmoid(logits) >= threshold
            group["pure_inputs"][name] = binary_metrics(pure, flags[pure_mask])
            group["mixed_documents"][name] = previous.spatial_metrics(mixed, flags[~pure_mask], samples)
            originals = {r["sample_id"] for r in records if r["kind"] == "human_document"}
            flagged = {r["sample_id"] for r, f in zip(records, flags) if f and r["kind"] == "human_document"}
            group["original_documents"][name] = {"documents": len(originals), "documents_with_any_flag": len(flagged)}
        groups[split] = group
    baseline = read_json(PARENT / "evaluation.json")
    a, b = groups["test"], baseline["groups"]["test"]
    chosen = read_json(EVIDENCE / "selection.json")["selected"]
    promotion = chosen["human_flagged_characters"] == 0 and a["pure_inputs"]["any_flag"]["human_flagged_segments"] == 0 \
                and a["mixed_documents"]["any_flag"]["pure_human_flagged_segments"] == 0 \
                and a["pure_inputs"]["any_flag"]["ai_paragraphs_with_any_flag"] >= b["pure_inputs"]["any_flag"]["ai_paragraphs_with_any_flag"] \
                and a["mixed_documents"]["any_flag"]["ai_character_recall"] > b["mixed_documents"]["any_flag"]["ai_character_recall"]
    result = {"status": "complete", "profile_sha256": freeze["profile_sha256"], "groups": groups,
              "full_collection": {"original_documents": sum(g["source_inputs"]["human_document"] for g in groups.values()),
                                  "mixed_documents": sum(g["source_inputs"]["mixed_document"] for g in groups.values()),
                                  "isolated_generated_paragraphs": sum(g["source_inputs"]["ai_paragraph"] for g in groups.values()),
                                  "scored_segments": sum(g["segments"] for g in groups.values())},
              "v2_baseline": baseline["groups"], "v2_baseline_segmentation": [1600, 400],
              "repeat_dataset_sha256": digest((EVIDENCE / "repeat-scores.json").read_bytes()),
              "promotion_rule_satisfied": bool(promotion), "prior_test_outcomes_known": True, "is_new_independent_validation": False,
              "contains_source_text": False, "parameters_unchanged_after_repeat": digest(PROFILE.read_bytes()) == freeze["profile_sha256"]}
    write_json(EVIDENCE / "evaluation.json", result)
    print(json.dumps({"full_collection": result["full_collection"], "repeat": a, "promotion": promotion}), flush=True)


def reproduce():
    profile, freeze, campaign = read_json(PROFILE), read_json(EVIDENCE / "FREEZE.json"), read_json(EVIDENCE / "campaign.json")
    assert digest(PROFILE.read_bytes()) == freeze["profile_sha256"]
    data = read_json(EVIDENCE / "train-scores.json")
    assert digest((EVIDENCE / "train-scores.json").read_bytes()) == freeze["train_dataset_sha256"]
    rows, cal = eligible(data, "fit"), eligible(data, "calibration")
    rebuilt = fitter.fit(rows, profile["models"], profile["fit"]["configuration"], campaign["optimizer"])
    for field in ("center", "scale", "head"):
        expected = profile["heads"]["strong"] if field == "head" else profile[field]
        assert json.dumps(rebuilt[field], sort_keys=True) == json.dumps(expected, sort_keys=True), field
    controls = rows+cal
    values = fitter.predict(controls, rebuilt)
    for head, field in [("strong", "high_threshold"), ("review", "possible_threshold")]:
        assert float(sigmoid(control_threshold(controls, values, campaign["threshold_logit_margins"][head]))) == profile[field]
    write_json(REPORTS / "reproduction.json", {"status": "complete", "parameters_and_thresholds_reproduce_exactly": True,
               "profile_sha256": freeze["profile_sha256"], "source_text_read": False, "model_inference_run": False, "test_used": False})
    print("Nonlinear parameters and thresholds reproduce exactly from numeric evidence.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "search", "repeat-score", "evaluate", "reproduce"])
    args = parser.parse_args()
    {"prepare": prepare, "search": search, "repeat-score": repeat_score, "evaluate": evaluate, "reproduce": reproduce}[args.command]()


if __name__ == "__main__":
    main()
