"""Pinned local experiment for four specialized classifiers; no runtime default edits.

Text is only used by the local base classifiers. The fusion consumes numeric
outputs; source identity and spatial labels serve fitting/evaluation only.
All ten papers are historical development material, not independent tests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aigc_detector.catalog import CATALOG, ENGLISH_MODELS, EXTRA_ENGLISH_MODELS, SPECIALIZED_ENGLISH_MODELS
from scripts import english_hard_negative as hn
from scripts.calibrate_english_v2 import make_records
from scripts.english_academic_samples import verify as verify_samples

EVIDENCE = ROOT / "evaluations/english-specialized-fusion-20261004"
REPORTS = ROOT / "reports/english-specialized-fusion-20261004"
PARENT = hn.PARENT
SAMPLES = hn.SAMPLES
PROFILE = hn.PROFILE
NEW = list(SPECIALIZED_ENGLISH_MODELS)
OLD = list(ENGLISH_MODELS + EXTRA_ENGLISH_MODELS)
CLASS_SIGNALS = [f"{m}::{label}" for m in NEW for label in CATALOG[m].get("class_labels", [])[1:]]
ALL_SIGNALS = OLD + NEW + CLASS_SIGNALS
read, write, sha = hn.read, hn.write, hn.sha


def configurations():
    detailed = NEW[:2] + CLASS_SIGNALS
    rosters = {"new_aggregate": NEW, "new_classes": detailed,
               "v2_plus_new": list(EXTRA_ENGLISH_MODELS) + NEW,
               "v2_plus_classes": list(EXTRA_ENGLISH_MODELS) + detailed,
               "all_plus_classes": OLD + detailed}
    configs = []
    for name, signals in rosters.items():
        for method in ("presence", "hard_human_bag_ranking"):
            for reg in (.01, .1, 1.):
                configs.append({"id": f"{name}-{method}-{reg:g}", "family": f"{name}-{method}",
                                "signals": signals, "method": method, "regularization": reg})
    for reg in (.01, .1, 1.):
        configs.append({"id": f"v2_refit-presence-{reg:g}", "family": "v2_refit-presence",
                        "signals": list(EXTRA_ENGLISH_MODELS), "method": "presence", "regularization": reg})
    for signal in NEW + CLASS_SIGNALS:
        configs.append({"id": "single-" + signal, "family": "single-" + signal,
                        "signals": [signal], "method": "single", "regularization": 0.})
    return configs


def prepare():
    if (EVIDENCE / "campaign.json").exists():
        raise FileExistsError("Preserve the declared campaign")
    _, collection, protocol = verify_samples(SAMPLES)
    manifest = read(ROOT / "models/manifest.json")["models"]
    parent = read(PARENT / "campaign.json")
    for m in OLD:
        assert manifest[m]["revision"] == parent["revisions"][m]
    paths = [Path(__file__), ROOT / "scripts/english_hard_negative.py",
             ROOT / "scripts/calibrate_english_v2.py", ROOT / "scripts/english_academic_samples.py",
             ROOT / "aigc_detector/engine.py", ROOT / "aigc_detector/modeling.py",
             ROOT / "aigc_detector/catalog.py", ROOT / "aigc_detector/annotation.py",
             ROOT / "models/manifest.json", PROFILE, SAMPLES / "samples.jsonl",
             PARENT / "campaign.json", PARENT / "context1600/train-scores.json", PARENT / "repeat-scores.json"]
    sample_meta = [json.loads(s) for s in (SAMPLES / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    paths += [SAMPLES / s["mapping_file"] for s in sample_meta if "mapping_file" in s]
    campaign = {
        "schema_version": "english-specialized-fusion-experiment/1.0",
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "groups": protocol["groups"], "sample_collection_sha256": collection["samples_sha256"],
        "models": OLD + NEW, "signals": ALL_SIGNALS,
        "revisions": {m: manifest[m]["revision"] for m in OLD + NEW},
        "class_signal_mapping": {s: {"model": s.split("::")[0], "class": s.split("::")[1]} for s in CLASS_SIGNALS},
        "segmentation": [1600, 400], "inference": {"overlap_tokens": 64, "dtype": "float32", "device": "cuda", "batch_size": 8},
        "configurations": configurations(), "candidate_count": len(configurations()),
        "bag_temperature": .1, "ranking_margin": .1, "threshold_margin": .01,
        "optimizer": {"algorithm": "CPU float64 L-BFGS strong Wolfe", "max_iter": 250,
                      "max_eval": 400, "gradient_tolerance": 1e-8, "change_tolerance": 1e-12},
        "fit_scope": "Original/mixed windows from fit papers only; same-paper text hashes deduplicated with bag memberships retained; conflicting and cross-paper duplicates excluded. Isolated generated paragraphs are evaluation-only.",
        "standardization": "Fit-only logit clip 1e-5; equal paper/class weights; SD floor .5; standardized clip 5. Single-signal controls use unclipped standardized logit range (clip 5 avoided with scale 3).",
        "selection": "Fit-only six-paper LOPO; minimize pure-human false windows, maximize mixed block hits, minimize false originals, maximize mixed AI characters; ties fewer signals, stronger L2, ID. No calibration positives or repeat scores.",
        "nested": "Outer six-paper LOPO repeats complete candidate selection in inner five-paper LOPO. Every train/scale/cutoff excludes outer paper.",
        "threshold": "Max pure-human score in fit plus calibration papers + .01; calibration positives excluded.",
        "freeze": "Freeze overall choice and each family champion before new-model P07/P08 inference; no retuning after repeat.",
        "success": "Selected pipeline: zero pure-human false windows and >=50% mixed-block hits both in nested diagnostic and frozen repeat. Historical material only; no automatic default replacement.",
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "new_independent_validation": False, "historical_results_known": True,
        "automatic_deployment": False, "contains_source_text": False,
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "python": sys.version},
    }
    write(EVIDENCE / "campaign.json", campaign)
    print(json.dumps({"prepared": str(EVIDENCE), "candidates": len(configurations())}), flush=True)


def verify_campaign():
    campaign = read(EVIDENCE / "campaign.json")
    for name, expected in campaign["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Declared source changed: {name}")
    return campaign


def new_scores(mid, value):
    result = {mid: value["ai_score"]}
    if "class_labels" in CATALOG[mid]:
        classes = value["class_scores"]
        assert set(classes) == set(CATALOG[mid]["class_labels"])
        assert abs(sum(classes.values()) - 1) < 2e-5
        indices = CATALOG[mid]["ai_label_indices"]
        assert abs(sum(classes[CATALOG[mid]["class_labels"][i]] for i in indices) - value["ai_score"]) < 2e-6
        result.update({f"{mid}::{label}": classes[label] for label in CATALOG[mid]["class_labels"][1:]})
    if not all(v is not None and np.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError("Invalid model output")
    return result


def score(repeat=False):
    campaign = verify_freeze() if repeat else verify_campaign()
    if not repeat and (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Cannot replace frozen development scores")
    name = "repeat" if repeat else "development"
    target = EVIDENCE / f"{name}-data.json"
    if target.exists():
        raise FileExistsError(target)
    splits = ("test",) if repeat else ("fit", "calibration")
    rows, passages, collection = make_records(1600, 400, splits, ("human_document", "ai_paragraph", "mixed_document"))
    assert collection["samples_sha256"] == campaign["sample_collection_sha256"]
    prior_path = PARENT / ("repeat-scores.json" if repeat else "context1600/train-scores.json")
    prior = read(prior_path)
    for m in OLD:
        assert prior["config"]["revisions"][m] == campaign["revisions"][m]
    by_id = {r["id"]: r for r in prior["records"]}
    assert set(by_id) == {r["id"] for r in rows}
    for r in rows:
        assert all(by_id[r["id"]][k] == v for k, v in r.items()), r["id"]
        r["scores"] = {m: by_id[r["id"]]["scores"][m] for m in OLD}
    from aigc_detector.engine import Detector
    detector = Detector(device="cuda", batch_size=8)
    scores = {k: {} for k in passages}
    cache_hashes = {}
    started = time.perf_counter()
    def blocked(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")
    print(json.dumps({"stage": name, "rows": len(rows), "unique_passages": len(passages)}), flush=True)
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for mid in NEW:
            path = REPORTS / name / f"{mid}-cache.json"
            contract = {"revision": campaign["revisions"][mid], "inference": campaign["inference"],
                        "campaign_sha256": sha(EVIDENCE / "campaign.json")}
            cache = read(path) if path.exists() else {**contract, "items": {}}
            assert all(cache[k] == v for k, v in contract.items())
            missing = [k for k in passages if k not in cache["items"]]
            for offset in range(0, len(missing), 32):
                keys = missing[offset:offset+32]
                values = detector._score_texts([passages[k] for k in keys], mid, 64, True, allow_empty=True)
                assert len(keys) == len(values)
                for key, value in zip(keys, values):
                    assert value["revision"] == campaign["revisions"][mid]
                    new_scores(mid, value)
                    cache["items"][key] = value
                write(path, cache)
                print(f"{name} {mid}: {min(offset+32, len(missing))}/{len(missing)}", flush=True)
            for key in passages:
                scores[key].update(new_scores(mid, cache["items"][key]))
            cache_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            detector._loaded.clear()
            gc.collect()
            torch.cuda.empty_cache()
    for r in rows:
        r["scores"].update(scores[r["sha256"]])
        assert set(r["scores"]) == set(ALL_SIGNALS)
    samples = {s["id"]: s for s in map(json.loads, (SAMPLES / "samples.jsonl").read_text(encoding="utf-8").splitlines())}
    dataset = hn.build_dataset({"records": rows}, samples, ALL_SIGNALS)
    dataset["config"] = {"campaign_sha256": sha(EVIDENCE / "campaign.json"), "cache_hashes": cache_hashes,
                         "prior_scores_sha256": sha(prior_path), "network_blocked_during_inference": True,
                         "seconds": time.perf_counter()-started, "unique_passages": len(passages),
                         "revisions": campaign["revisions"], "splits": list(splits)}
    if repeat:
        dataset["config"]["freeze_sha256"] = sha(EVIDENCE / "FREEZE.json")
    write(target, dataset)
    print(json.dumps({"scored": name, "rows": len(rows), "seconds": dataset["config"]["seconds"]}), flush=True)


def fit_model(data, papers, config, campaign):
    if config["method"] == "single":
        # hn.predict clips standardized values at 5; logits clipped at 1e-5
        # lie within +/-11.52, so scaling by 3 and coefficient 3 preserves them.
        return {"signals": config["signals"], "center": [0.], "scale": [3.], "coefficients": [3.],
                "intercept": 0., "configuration": config, "fit_papers": sorted(papers),
                "optimization": {"stationary_1e_5": True, "gradient_inf": 0.}}
    return hn.fit_model(data, papers, config, {**campaign, "signals": config["signals"]})


def objective(candidate):
    m, c = candidate["metrics"], candidate["configuration"]
    return (m["pure_human_false_windows"], -m["mixed_block_hits"], m["original_false_documents"],
            -m["mixed_ai_hit_chars"], len(c["signals"]), -c["regularization"], c["id"])


def fit():
    if (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Frozen experiment cannot be reselected")
    campaign, data = verify_campaign(), read(EVIDENCE / "development-data.json")
    identity, data_hash = sha(EVIDENCE / "campaign.json"), sha(EVIDENCE / "development-data.json")
    papers, models = sorted(campaign["groups"]["fit"]), {}
    def model_for(pool, config):
        key = (tuple(sorted(pool)), config["id"])
        if key not in models:
            path = REPORTS / "fit-cache" / ("-".join(key[0]) + "--" + key[1].replace("::", "__") + ".json")
            if path.exists():
                cached = read(path)
                assert cached["campaign_sha256"] == identity and cached["data_sha256"] == data_hash
                models[key] = cached["model"]
            else:
                models[key] = fit_model(data, list(key[0]), config, campaign)
                write(path, {"campaign_sha256": identity, "data_sha256": data_hash, "model": models[key]})
        return models[key]
    def compare(pool):
        candidates = []
        for config in campaign["configurations"]:
            folds = []
            for held in pool:
                train = [p for p in pool if p != held]
                model = model_for(train, config)
                controls = [r for r in data["records"] if r["paper_id"] in train]
                threshold = hn.cutoff(controls, model, campaign["threshold_margin"])
                folds.append({"held_paper": held, "fit_papers": train, "threshold": threshold,
                              "metrics": hn.metrics(data, {held}, model, threshold)})
            candidates.append({"configuration": config, "folds": folds, "metrics": hn.summed(f["metrics"] for f in folds)})
        return candidates
    candidates = compare(papers)
    chosen = min(candidates, key=objective)
    print(json.dumps({"selection": chosen["configuration"], "metrics": chosen["metrics"]}), flush=True)
    nested = []
    for held in papers:
        train = [p for p in papers if p != held]
        inner = compare(train)
        selected = min(inner, key=objective)
        model = model_for(train, selected["configuration"])
        threshold = hn.cutoff([r for r in data["records"] if r["paper_id"] in train], model, campaign["threshold_margin"])
        fold = {"held_paper": held, "inner_candidates": inner, "selected": selected["configuration"],
                "model": model, "threshold": threshold, "metrics": hn.metrics(data, {held}, model, threshold)}
        nested.append(fold)
        write(REPORTS / "nested-progress.json", {"folds": nested})
        print(json.dumps({"nested_held": held, "selected": selected["configuration"]["id"], "metrics": fold["metrics"]}), flush=True)
    champions, frozen = {}, {}
    for family in sorted({c["configuration"]["family"] for c in candidates}):
        candidate = min((c for c in candidates if c["configuration"]["family"] == family), key=objective)
        champions[family] = candidate
        model = model_for(papers, candidate["configuration"])
        controls = [r for r in data["records"] if r["paper_id"] in papers + campaign["groups"]["calibration"]]
        frozen[family] = {"model": model, "threshold": hn.cutoff(controls, model, campaign["threshold_margin"]),
                          "cv_zero_false_feasible": candidate["metrics"]["pure_human_false_windows"] == 0}
    verify_campaign()
    selection = {"candidates": candidates, "selected": chosen, "family_champions": champions,
                 "nested": nested, "nested_totals": hn.summed(f["metrics"] for f in nested),
                 "unique_fits_and_controls": len(models),
                 "optimization": {"nonstationary_fits": sum(not m["optimization"]["stationary_1e_5"] for m in models.values()),
                                  "max_gradient_inf": max(m["optimization"]["gradient_inf"] for m in models.values())},
                 "repeat_scores_used": False, "new_independent_validation": False}
    write(EVIDENCE / "selection.json", selection)
    write(EVIDENCE / "frozen-models.json", {"models": frozen, "overall_family": chosen["configuration"]["family"], "contains_source_text": False})
    write(EVIDENCE / "FREEZE.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
          "hashes": {p: sha(EVIDENCE / p) for p in ("campaign.json", "development-data.json", "selection.json", "frozen-models.json")}})
    print(json.dumps({"frozen": True, "selected": chosen["configuration"]["id"], "nested": selection["nested_totals"], "optimization": selection["optimization"]}), flush=True)


def verify_freeze():
    campaign = verify_campaign()
    for name, expected in read(EVIDENCE / "FREEZE.json")["hashes"].items():
        assert sha(EVIDENCE / name) == expected, name
    return campaign


def baseline():
    p = read(PROFILE)
    model = {"signals": p["models"], "center": p["center"], "scale": p["scale"],
             "coefficients": p["heads"]["review"]["coefficients"], "intercept": p["heads"]["review"]["intercept"]}
    return {"model": model, "threshold": float(np.log(p["possible_threshold"] / (1-p["possible_threshold"])))}


def evaluate():
    campaign = verify_freeze()
    if (EVIDENCE / "evaluation.json").exists():
        raise FileExistsError("Preserve evaluation")
    dev, repeat = read(EVIDENCE / "development-data.json"), read(EVIDENCE / "repeat-data.json")
    assert repeat["config"]["freeze_sha256"] == sha(EVIDENCE / "FREEZE.json")
    assert not {r["sha256"] for r in dev["records"]} & {r["sha256"] for r in repeat["records"]}
    data = {k: dev[k] + repeat[k] for k in ("records", "blocks")}
    assert len(data["records"]) == 678 and len(data["blocks"]) == 75
    frozen, selection = read(EVIDENCE / "frozen-models.json"), read(EVIDENCE / "selection.json")
    results = {}
    for name, item in {"frozen_v2": baseline(), **frozen["models"]}.items():
        model, threshold = item["model"], item["threshold"]
        papers = sorted({r["paper_id"] for r in data["records"]})
        results[name] = {"threshold": threshold,
                        "by_split": {s: hn.metrics(data, set(ps), model, threshold) for s, ps in campaign["groups"].items()},
                        "by_paper": {p: hn.metrics(data, {p}, model, threshold) for p in papers},
                        "all": hn.metrics(data, set(papers), model, threshold)}
    selected = results[frozen["overall_family"]]
    checks = [selection["nested_totals"], selected["by_split"]["test"]]
    outcome = {"results": results, "selected_family": frozen["overall_family"], "nested_totals": selection["nested_totals"],
               "meets_declared_target": all(m["pure_human_false_windows"] == 0 and m["mixed_block_hits"] >= .5*m["mixed_blocks"] for m in checks),
               "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": sha(EVIDENCE / "FREEZE.json"),
               "repeat_data_sha256": sha(EVIDENCE / "repeat-data.json"), "default_profile_unchanged": True,
               "new_base_models_scored": NEW, "reused_classifiers": OLD, "new_independent_validation": False,
               "contains_source_text": False}
    write(EVIDENCE / "evaluation.json", outcome)
    print(json.dumps({"selected": frozen["overall_family"], "nested": selection["nested_totals"], "selected_results": selected,
                      "meets_declared_target": outcome["meets_declared_target"]}), flush=True)


def verify_result():
    campaign = verify_freeze()
    data, frozen = read(EVIDENCE / "development-data.json"), read(EVIDENCE / "frozen-models.json")
    errors = {}
    for name, item in frozen["models"].items():
        rebuilt = fit_model(data, sorted(campaign["groups"]["fit"]), item["model"]["configuration"], campaign)
        errors[name] = float(np.max(np.abs(hn.predict(data["records"], rebuilt) - hn.predict(data["records"], item["model"]))))
        assert errors[name] < 1e-10
        controls = [r for r in data["records"] if r["paper_id"] in campaign["groups"]["fit"] + campaign["groups"]["calibration"]]
        assert abs(hn.cutoff(controls, rebuilt, campaign["threshold_margin"]) - item["threshold"]) < 1e-10
    result = {"champion_max_logit_errors": errors, "freeze_verified": True}
    write(REPORTS / "reproduction.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score-development", "fit", "score-repeat", "evaluate", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    actions = {"prepare": prepare, "score-development": score, "fit": fit, "score-repeat": lambda: score(True),
               "evaluate": evaluate, "verify": verify_result}
    actions[args.action]()


if __name__ == "__main__":
    main()
