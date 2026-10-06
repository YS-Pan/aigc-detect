"""Small, leakage-controlled frozen-model experiment on all 177 full documents.

No base network, classifier head or neural fusion is trained. Exactly two changes
to the existing five-score workflow are allowed: human-only threshold calibration
and one fixed convex linear score-fusion refit for window presence. Acceptance
inference is refused until corpus, implementation, parameters and cutoffs freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import gc
import html
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.english_diverse_corpus import (SOURCE, audit_duplicates, conversion_audit, digest, load_corpus,
    overlap, previously_seen, read, union, write)
from aigc_detector.annotation import segment_spans
from aigc_detector.fusion import decide_fusion, load_profile

EVIDENCE = ROOT / "evaluations/english-diverse-20261006"
REPORTS = ROOT / "reports/english-diverse-20261006"
POLICY = "english-academic-20261003-v2"
PROFILE = ROOT / "aigc_detector/profiles" / (POLICY + ".json")
SEED = 20261006
REFIT_EPS = 1e-12
REGULARIZATION = 10.0
HUMAN_WEIGHT = 20.0
MARGIN = 0.1
PROCESS_FILES = [Path(__file__), ROOT / "scripts/english_diverse_corpus.py", ROOT / "scripts/fit_fusion_v2.py",
    ROOT / "aigc_detector/annotation.py", ROOT / "aigc_detector/engine.py", ROOT / "aigc_detector/fusion.py",
    ROOT / "aigc_detector/modeling.py", ROOT / "aigc_detector/catalog.py", ROOT / "aigc_detector/assets.py",
    ROOT / "models/manifest.json", PROFILE]


def sha(path):
    return digest(Path(path).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def source_locks():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in PROCESS_FILES}


def split_groups(groups, seen, seed=SEED):
    old = [g for g in groups if set(g) & set(seen)]
    fresh = [g for g in groups if not set(g) & set(seen)]
    random.Random(seed).shuffle(fresh)
    if len(fresh) < 12:
        raise ValueError("Too few unexposed author/source groups")
    # 30% acceptance; separate 20% human calibration and 20% development.
    n_accept, n_cal, n_dev = round(len(fresh) * .3), round(len(fresh) * .2), round(len(fresh) * .2)
    assigned = {"acceptance": fresh[:n_accept], "calibration": fresh[n_accept:n_accept+n_cal],
                "development": fresh[n_accept+n_cal:n_accept+n_cal+n_dev], "fit": old + fresh[n_accept+n_cal+n_dev:]}
    return {k: sorted(p for g in gs for p in g) for k, gs in assigned.items()}


def declare(source=SOURCE):
    if (EVIDENCE / "campaign.json").exists() or (REPORTS / "corpus.json").exists():
        raise FileExistsError("Preserve this campaign; use a new named campaign for changed rules")
    papers, documents, audit = load_corpus(source)
    duplicates, seen = audit_duplicates(papers, documents), previously_seen(papers, documents)
    groups = split_groups(duplicates["groups"], seen["seen_papers"])
    roles = {pid: role for role, ids in groups.items() for pid in ids}
    profile, profile_sha = load_profile(POLICY)
    manifest = read(ROOT / "models/manifest.json")["models"]
    campaign = {"schema": "english-diverse-campaign/1.0", "declared_at_utc": now(), "seed": SEED,
        "groups": groups, "author_source_groups": duplicates["groups"], "previously_seen": seen,
        "split_before_scores": True, "source": str(Path(source).resolve()), "source_files": audit["source_files"],
        "process_hashes": source_locks(), "baseline_policy": POLICY, "baseline_profile_sha256": profile_sha,
        "models": profile["models"], "revisions": {m: manifest[m]["revision"] for m in profile["models"]},
        "segmentation": {"id": "english-natural-boundaries-v1", "segment_chars": 1600, "minimum_chars": 400},
        "inference": {"device": "cuda", "dtype": "float32", "batch_size": 8, "overlap_tokens": 64, "network_blocked": True},
        "primary_metrics": "Any positive-length generated intersection makes a window positive. Pure-human windows in originals AND mixed documents count as negatives. A narrative unit hits if any of its exact generated characters intersects a flagged window.",
        "section_metric": "Report resource-excluded narrative-slot recall (247 labelled slots) and whole-section recall (27) separately; 1112 split spans are not paragraphs.",
        "candidates": ["v2-human-calibration", "presence-linear-refit"],
        "candidate_scope": "Same five pinned raw scores, same full input and 1600/400 segmentation; no local scan, new signals, neural training or classifier heads.",
        "refit": {"features": "Five raw score logits only", "clip_probability": REFIT_EPS, "clip_standardized": 5.0,
                  "regularization": REGULARIZATION, "human_weight": HUMAN_WEIGHT, "target": "any generated character",
                  "weights": "Deduplicate (paper, input hash, window label); equal total per author group and class bucket; retain contradictions.",
                  "search": "none; one fixed convex fit, plus leave-author-group-out diagnostics"},
        "thresholds": "All pure-human scan opportunities in fit/development/calibration; maximum score logit plus fixed 0.1 margin. V2 calibration may only raise existing thresholds.",
        "selection": "Development only: minimize false windows, maximize narrative-unit hits, prefer threshold-only change on ties. Calibration generated results are masked until final evaluation.",
        "acceptance": "Infer only after FREEZE.json; no threshold/candidate changes after acceptance. Six fresh full-paper groups, every derived version bound.",
        "gate": "Zero observed pure-human-window false positives and >=50% generated narrative-unit recall, plus reliable development improvement before default promotion.",
        "no_bottom_network_training": True, "contains_source_text": False}
    windows = []
    passages = {}
    for d in documents:
        d["role"] = roles[d["paper_id"]]
        for i, (a, b) in enumerate(segment_spans(d["text"], 1600, 400, language="en")):
            value = d["text"][a:b]; h = digest(value); passages[h] = value
            windows.append({"id": d["id"] + f":s{i:04}", "document_id": d["id"], "paper_id": d["paper_id"],
                "role": d["role"], "sample_type": d["sample_type"], "generator": d["generator"], "topic": d["topic"],
                "start": a, "end": b, "sha256": h, "ai_characters": overlap(d["ai_spans"], a, b), "characters": b-a})
    owners, labels = defaultdict(set), defaultdict(set)
    for r in windows:
        owners[r["sha256"]].add(r["paper_id"]); labels[r["sha256"]].add(r["ai_characters"] > 0)
    crosses = [{"sha256": h, "papers": sorted(ps)} for h, ps in owners.items() if len(ps) > 1]
    if crosses:
        raise ValueError("Cross-paper window duplicates require binding before declaration")
    public_docs = [{k: v for k, v in d.items() if k not in ["text", "origins"]} for d in documents]
    inputs = {"documents": public_docs, "windows": windows, "duplicate_audit": duplicates,
              "window_unique_hashes": len(owners), "window_label_conflicts": sorted(h for h, ls in labels.items() if len(ls) > 1),
              "counts": {k: v for k, v in audit.items() if k not in ["source_files", "read_only_source"]}}
    write(REPORTS / "corpus.json", {"papers": papers, "documents": documents, "passages": passages})
    write(EVIDENCE / "inputs.json", inputs)
    write(EVIDENCE / "conversion-audit.json", conversion_audit(source, papers, documents))
    write(EVIDENCE / "campaign.json", campaign)
    write(EVIDENCE / "input-lock.json", {"campaign_sha256": sha(EVIDENCE / "campaign.json"),
        "inputs_sha256": sha(EVIDENCE / "inputs.json"), "corpus_sha256": sha(REPORTS / "corpus.json"),
        "conversion_audit_sha256": sha(EVIDENCE / "conversion-audit.json")})
    status = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--porcelain"], cwd=ROOT).decode("utf-8")
    write(REPORTS / "initial-git-state.json", {"captured_at_utc": now(), "status": status,
        "note": "Pre-existing changes are preserved; task files will be committed separately."})
    print({"declared": True, "groups": groups, "counts": inputs["counts"], "windows": len(windows), "unique_windows": len(owners)}, flush=True)


def verify_inputs(check_upstream=False):
    c, lock = read(EVIDENCE / "campaign.json"), read(EVIDENCE / "input-lock.json")
    for p, key in [(EVIDENCE / "campaign.json", "campaign_sha256"), (EVIDENCE / "inputs.json", "inputs_sha256"),
                   (REPORTS / "corpus.json", "corpus_sha256"), (EVIDENCE / "conversion-audit.json", "conversion_audit_sha256")]:
        if sha(p) != lock[key]:
            raise ValueError("Frozen input changed: " + str(p))
    for p, h in c["process_hashes"].items():
        if sha(ROOT / p) != h:
            raise ValueError("Declared implementation changed: " + p)
    if check_upstream:
        for p, entry in c["source_files"].items():
            if sha(Path(c["source"]) / p) != entry["sha256"]:
                raise ValueError("Upstream corpus changed: " + p)
    return c


def verify_freeze():
    c = verify_inputs()
    f = read(EVIDENCE / "FREEZE.json")
    for p, h in f["files"].items():
        if sha(ROOT / p) != h:
            raise ValueError("Final frozen pipeline changed: " + p)
    return c, f


def score(stage):
    c = verify_inputs(check_upstream=True)
    if stage == "acceptance":
        verify_freeze()
        roles = {"acceptance"}
    else:
        if (EVIDENCE / "FREEZE.json").exists():
            raise FileExistsError("Preserve frozen development scores")
        roles = {"fit", "development", "calibration"}
    data, local = read(EVIDENCE / "inputs.json"), read(REPORTS / "corpus.json")
    hashes = sorted({r["sha256"] for r in data["windows"] if r["role"] in roles})
    from aigc_detector.engine import Detector
    import torch
    torch.set_num_threads(1)
    detector = Detector(device="cuda", batch_size=8)
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden during local inference")
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for mid in c["models"]:
            target = REPORTS / stage / (mid + ".json")
            contract = {"campaign_sha256": sha(EVIDENCE / "campaign.json"), "input_lock_sha256": sha(EVIDENCE / "input-lock.json"),
                        "revision": c["revisions"][mid], "roles": sorted(roles), "overlap_tokens": 64, "float32": True}
            cache = read(target) if target.exists() else {**contract, "items": {}}
            if any(cache.get(k) != v for k, v in contract.items()):
                raise ValueError("Cache contract mismatch")
            _, model, _ = detector._load(mid)
            model.requires_grad_(False)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise AssertionError("Unfrozen base network/head")
            missing = sorted(set(hashes) - set(cache["items"]))
            started = time.perf_counter()
            for offset in range(0, len(missing), 64):
                batch = missing[offset:offset+64]
                result = detector._score_texts([local["passages"][h] for h in batch], mid, 64, True)
                for h, v in zip(batch, result):
                    if v["revision"] != c["revisions"][mid] or v["ai_score"] is None:
                        raise ValueError("Invalid raw score/revision")
                    cache["items"][h] = v
                cache["seconds_this_invocation"] = time.perf_counter() - started
                write(target, cache)
                print(f"{stage} {mid}: {min(offset+64, len(missing))}/{len(missing)}", flush=True)
            detector._loaded.clear(); del model; gc.collect(); torch.cuda.empty_cache()
    caches = {m: read(REPORTS / stage / (m + ".json"))["items"] for m in c["models"]}
    raw = {h: {m: caches[m][h]["ai_score"] for m in c["models"]} for h in hashes}
    write(EVIDENCE / (stage + "-scores.json"), {"scores": raw, "roles": sorted(roles), "models": c["models"],
        "created_at_utc": now(), "cache_hashes": {m: sha(REPORTS / stage / (m + ".json")) for m in c["models"]},
        "campaign_sha256": sha(EVIDENCE / "campaign.json"),
        **({"freeze_sha256": sha(EVIDENCE / "FREEZE.json")} if stage == "acceptance" else {})})
    print({"stage": stage, "unique_windows_scored": len(raw)}, flush=True)


def logit(p):
    p = min(1 - REFIT_EPS, max(REFIT_EPS, float(p)))
    return math.log(p) - math.log1p(-p)


@lru_cache(maxsize=8192)
def baseline_logit(values):
    profile = read(PROFILE)
    raw = dict(zip(profile["models"], values))
    return decide_fusion(raw, profile, profile["high_threshold"], profile["possible_threshold"])["review_logit"]


def value(raw, configuration):
    if configuration["id"] in ["baseline-v2", "v2-human-calibration"]:
        return baseline_logit(tuple(raw[m] for m in configuration["models"]))
    z = [max(-5., min(5., (logit(raw[m]) - a) / b)) for m, a, b in
         zip(configuration["models"], configuration["center"], configuration["scale"])]
    return configuration["intercept"] + sum(a*b for a, b in zip(z, configuration["coefficients"]))


def baseline_config():
    p = read(PROFILE)
    return {"id": "baseline-v2", "models": p["models"], "cutoff": logit(p["possible_threshold"])}


def metrics(data, raw, configuration, papers=None, documents=None):
    docs = [d for d in data["documents"] if (papers is None or d["paper_id"] in papers) and
            (documents is None or d["id"] in documents)]
    ids = {d["id"] for d in docs}
    rows = [r for r in data["windows"] if r["document_id"] in ids]
    flags = [r for r in rows if value(raw[r["sha256"]], configuration) >= configuration["cutoff"]]
    false = [r for r in flags if r["ai_characters"] == 0]
    bydoc = defaultdict(list)
    for r in flags:
        bydoc[r["document_id"]].append((r["start"], r["end"]))
    hit, section_hit, cover, ai_total = [], [], 0, 0
    for d in docs:
        ranges = bydoc[d["id"]]
        cover += sum(overlap(d["ai_spans"], a, b) for a, b in union(ranges))
        ai_total += sum(b-a for a, b in union(d["ai_spans"]))
        for u in d["units"]:
            if any(overlap(u["spans"], a, b) for a, b in ranges):
                hit.append(d["id"] + ":" + u["id"])
        for u in d["section_units"]:
            if any(overlap(u["spans"], a, b) for a, b in ranges):
                section_hit.append(d["id"] + ":" + u["id"])
    positive_flags = [r for r in flags if r["ai_characters"] > 0]
    ai_flagged = sum(r["ai_characters"] for r in positive_flags)
    human_flagged = sum(r["characters"] - r["ai_characters"] for r in positive_flags)
    units = sum(len(d["units"]) for d in docs)
    return {"documents": len(docs), "papers": len({d["paper_id"] for d in docs}), "windows": len(rows),
        "pure_human_windows": sum(r["ai_characters"] == 0 for r in rows), "false_windows": len(false),
        "false_unique_hashes": len({r["sha256"] for r in false}), "false_documents": len({r["document_id"] for r in false}),
        "original_documents": sum(d["sample_type"] == "original" for d in docs),
        "original_false_windows": sum(r["sample_type"] == "original" for r in false),
        "original_false_documents": len({r["document_id"] for r in false if r["sample_type"] == "original"}),
        "mixed_false_windows": sum(r["sample_type"] != "original" for r in false),
        "generated_windows": sum(r["ai_characters"] > 0 for r in rows), "flagged_generated_windows": len(positive_flags),
        "narrative_units": units, "hit_units": len(hit), "unit_recall": len(hit)/units if units else None,
        "section_units": sum(len(d["section_units"]) for d in docs), "hit_sections": len(section_hit),
        "generated_characters": ai_total, "covered_generated_characters": cover, "generated_character_coverage": cover/ai_total if ai_total else None,
        "positive_window_generated_characters": ai_flagged, "positive_window_human_characters": human_flagged,
        "positive_window_human_fraction": human_flagged/(human_flagged+ai_flagged) if human_flagged+ai_flagged else None,
        "hit_ids": hit, "section_hit_ids": section_hit, "false_ids": [r["id"] for r in false], "flag_ids": [r["id"] for r in flags]}


def calibrate(data, raw, configuration, papers):
    result = dict(configuration)
    negatives = [r for r in data["windows"] if r["paper_id"] in papers and r["ai_characters"] == 0]
    if not negatives:
        raise ValueError("Empty human calibration pool")
    maximum = max(value(raw[r["sha256"]], result) for r in negatives)
    result["cutoff"] = max(configuration.get("cutoff", -math.inf), maximum + MARGIN)
    result["human_control_maximum"] = maximum
    result["human_control_unique_windows"] = len({r["sha256"] for r in negatives})
    result["calibration_papers"] = sorted(papers)
    result["margin"] = MARGIN
    return result


def fit_linear(data, raw, papers, groups):
    import numpy as np
    from scripts.fit_fusion_v2 import fit_head
    models = read(PROFILE)["models"]
    unique = {(r["paper_id"], r["sha256"], r["ai_characters"] > 0): r for r in data["windows"] if r["paper_id"] in papers}
    rows = list(unique.values())
    gid = {p: i for i, ps in enumerate(groups) for p in ps}
    counts = Counter((gid[r["paper_id"]], r["ai_characters"] > 0) for r in rows)
    weights = np.array([(1. if r["ai_characters"] else HUMAN_WEIGHT) / counts[gid[r["paper_id"]], r["ai_characters"] > 0] for r in rows])
    features = np.array([[logit(raw[r["sha256"]][m]) for m in models] for r in rows])
    center = np.average(features, axis=0, weights=weights)
    scale = np.maximum(.5, np.sqrt(np.average((features-center)**2, axis=0, weights=weights)))
    x = np.column_stack([np.ones(len(rows)), np.clip((features-center)/scale, -5., 5.)])
    params, optimization = fit_head(x, np.array([r["ai_characters"] > 0 for r in rows], dtype=float), weights,
                                   [[i] for i in range(len(models))], REGULARIZATION)
    return {"id": "presence-linear-refit", "models": models, "center": center.tolist(), "scale": scale.tolist(),
        "intercept": float(params[0]), "coefficients": params[1:].tolist(), "clip_probability": REFIT_EPS,
        "fit_papers": sorted(papers), "fit_unique_labelled_windows": len(rows), "optimization": optimization,
        "regularization": REGULARIZATION, "human_weight": HUMAN_WEIGHT,
        "runtime_features": "Pinned raw classifier scores only; no text, length, source or author features."}


def diagnose():
    c = verify_inputs()
    if (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Do not reselect after freezing")
    data, raw = read(EVIDENCE / "inputs.json"), read(EVIDENCE / "development-scores.json")["scores"]
    config = baseline_config()
    out = {"at_utc": now(), "baseline": {role: metrics(data, raw, config, ids)
        for role, ids in c["groups"].items() if role in ["fit", "development"]}, "signals": {},
        "calibration_generated_scores_not_used": True, "acceptance_scores_not_available": True}
    fitdev = set(c["groups"]["fit"] + c["groups"]["development"])
    rows = [r for r in data["windows"] if r["paper_id"] in fitdev]
    # Descriptive development diagnostics, not candidate searches or performance
    # estimates. The detector score may saturate on human/AI alike.
    for m in c["models"]:
        human = [raw[r["sha256"]][m] for r in rows if r["ai_characters"] == 0]
        positive = [raw[r["sha256"]][m] for r in rows if r["ai_characters"] > 0]
        out["signals"][m] = {"human_max": max(human), "generated_window_max": max(positive),
            "generated_windows_above_human_max": sum(v > max(human) for v in positive),
            "human_saturated_at_one": sum(v == 1 for v in human), "generated_saturated_at_one": sum(v == 1 for v in positive)}
    write(EVIDENCE / "baseline-diagnosis.json", out)
    print({"baseline": {k: {x: v[x] for x in ["false_windows", "pure_human_windows", "hit_units", "narrative_units"]} for k, v in out["baseline"].items()}, "signals": out["signals"]}, flush=True)


def freeze():
    c = verify_inputs(check_upstream=True)
    if (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Already frozen; no post-acceptance selection")
    diagnosis = read(EVIDENCE / "baseline-diagnosis.json")
    data, raw = read(EVIDENCE / "inputs.json"), read(EVIDENCE / "development-scores.json")["scores"]
    fit, dev, cal = (set(c["groups"][r]) for r in ["fit", "development", "calibration"])
    # Exactly the two predeclared paths. The fixed refit addresses current mixed
    # presence labels and domain shift; it does not expand old nonlinear searches.
    old = baseline_config(); old["id"] = "v2-human-calibration"
    candidates = [calibrate(data, raw, old, fit), calibrate(data, raw, fit_linear(data, raw, fit, c["author_source_groups"]), fit)]
    entries = [{"configuration": x, "development": metrics(data, raw, x, dev)} for x in candidates]
    winner = min(entries, key=lambda e: (e["development"]["false_windows"], -e["development"]["hit_units"],
                 e["configuration"]["id"] != "v2-human-calibration"))
    nested = []
    # One fixed formula only; no inner hyperparameter search. Every holdout group
    # is excluded from standardization, fitting and the complete human cutoff.
    for g in c["author_source_groups"]:
        held = set(g) & fit
        if not held:
            continue
        remaining = fit - held
        cfg = calibrate(data, raw, fit_linear(data, raw, remaining, c["author_source_groups"]), remaining)
        nested.append({"held": sorted(held), "training": sorted(remaining), "configuration": cfg,
                       "metrics": metrics(data, raw, cfg, held)})
    final = calibrate(data, raw, winner["configuration"], fit | dev | cal)
    all_configs = {e["configuration"]["id"]: calibrate(data, raw, e["configuration"], fit | dev | cal) for e in entries}
    write(EVIDENCE / "selection.json", {"at_utc": now(), "entries": entries, "selected_id": final["id"],
        "leave_author_group_out_fixed_refit": nested, "diagnosis_sha256": sha(EVIDENCE / "baseline-diagnosis.json"),
        "calibration_positive_scores_used": False, "acceptance_used": False, "candidate_count": 2})
    write(EVIDENCE / "frozen-configurations.json", {"selected": final, "candidates": all_configs, "baseline": baseline_config(),
        "note": "Experimental score-only parameters; not a registered deployment profile."})
    files = [EVIDENCE / p for p in ["campaign.json", "input-lock.json", "inputs.json", "conversion-audit.json", "development-scores.json",
             "baseline-diagnosis.json", "selection.json", "frozen-configurations.json"]] + PROCESS_FILES
    write(EVIDENCE / "FREEZE.json", {"frozen_at_utc": now(), "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        "selected_id": final["id"], "acceptance_papers": c["groups"]["acceptance"], "no_training_base_or_head": True,
        "default_changed": False})
    print({"frozen": True, "selected": final["id"], "development": {x: winner["development"][x] for x in
        ["false_windows", "pure_human_windows", "hit_units", "narrative_units"]}, "acceptance": c["groups"]["acceptance"]}, flush=True)


def evaluate():
    c, f = verify_freeze()
    verify_inputs(check_upstream=True)
    data = read(EVIDENCE / "inputs.json")
    dev, accept = read(EVIDENCE / "development-scores.json"), read(EVIDENCE / "acceptance-scores.json")
    if accept["freeze_sha256"] != sha(EVIDENCE / "FREEZE.json") or set(dev["scores"]) & set(accept["scores"]):
        raise ValueError("Acceptance lock or separation failure")
    raw = {**dev["scores"], **accept["scores"]}
    configs = read(EVIDENCE / "frozen-configurations.json")
    # Candidate comparison on acceptance is descriptive only; selected ID is
    # already locked, so a retrospectively better alternative never replaces it.
    methods = {"baseline-v2": configs["baseline"], **configs["candidates"]}
    out = {"evaluated_at_utc": now(), "selected_id": f["selected_id"], "freeze_sha256": sha(EVIDENCE / "FREEZE.json"),
        "all_scores_sha256": {s: sha(EVIDENCE / (s + "-scores.json")) for s in ["development", "acceptance"]}, "methods": {}}
    for name, cfg in methods.items():
        byrole = {role: metrics(data, raw, cfg, ids) for role, ids in c["groups"].items()}
        breakdown = {}
        for field in ["sample_type", "generator", "topic", "paper_id"]:
            breakdown[field] = {}
            for v in sorted({d[field] for d in data["documents"]}, key=str):
                ids = {d["id"] for d in data["documents"] if d[field] == v}
                breakdown[field][str(v)] = {"all": metrics(data, raw, cfg, documents=ids),
                    "acceptance": metrics(data, raw, cfg, set(c["groups"]["acceptance"]), ids)}
        out["methods"][name] = {"all": metrics(data, raw, cfg), "by_role": byrole, "breakdown": breakdown}
    result = out["methods"][f["selected_id"]]["by_role"]["acceptance"]
    out["acceptance_target_met"] = result["false_windows"] == 0 and result["unit_recall"] >= .5
    out["default_changed"] = False
    out["interpretation"] = "Six new source/author groups; zero observations do not establish zero population error. Full source-converted originals and all mixed versions were evaluated. Section slot recall is reported separately from paragraph recall. No acceptance retuning or new neural training."
    write(EVIDENCE / "evaluation.json", out)
    print({"selected": f["selected_id"], "target_met": out["acceptance_target_met"], "acceptance":
        {name: {k: v["by_role"]["acceptance"][k] for k in ["false_windows", "pure_human_windows", "hit_units", "narrative_units", "generated_character_coverage"]} for name, v in out["methods"].items()}}, flush=True)


def report():
    c, f = verify_freeze()
    evaluation, data, local = read(EVIDENCE / "evaluation.json"), read(EVIDENCE / "inputs.json"), read(REPORTS / "corpus.json")
    raw = {**read(EVIDENCE / "development-scores.json")["scores"], **read(EVIDENCE / "acceptance-scores.json")["scores"]}
    configs = read(EVIDENCE / "frozen-configurations.json")
    target = REPORTS / "report"; target.mkdir(parents=True, exist_ok=True)
    css = "body{max-width:1150px;margin:35px auto;padding:0 25px;font:16px/1.6 system-ui}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px;text-align:left}pre{white-space:pre-wrap;font:15px/1.6 Georgia}mark{background:#ffe69b}.known{border-bottom:3px solid #537be2}.false{background:#ffcccc}a{color:#245fbd}small{color:#555}"
    def page(title, body):
        return '<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>'+css+'</style><h1>'+html.escape(title)+'</h1>'+body+'</html>'
    def table(rows, keys):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(r[k])+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
    summary = []
    for name, v in evaluation["methods"].items():
        m = v["by_role"]["acceptance"]
        summary.append({"配置": html.escape(name), "人工窗口误报": f"{m['false_windows']}/{m['pure_human_windows']}",
            "叙述单位命中": f"{m['hit_units']}/{m['narrative_units']}", "生成字符覆盖": f"{100*m['generated_character_coverage']:.2f}%",
            "正例窗口人工字符": m['positive_window_human_characters']})
    intro = f"<p>冻结选中：{html.escape(f['selected_id'])}。默认仍为英文 v2。验收来源：{', '.join(c['groups']['acceptance'])}。</p>"
    intro += "<p>目标为完全人工窗口零观察误报且生成叙述单位召回至少50%。短跨度、叙述槽位和整节分别核算；247个整节叙述槽位不等于1112个跨度，也不是独立论文。下列原版和混合正文均为完整抽取输入，未据分数删除难例。蓝色下划线为已知生成字符；黄色为判正窗口，红色为纯人工误报。</p>"
    intro += table(summary, ["配置", "人工窗口误报", "叙述单位命中", "生成字符覆盖", "正例窗口人工字符"])
    links = []
    for d in local["documents"]:
        rows = [r for r in data["windows"] if r["document_id"] == d["id"]]
        parts = ['<p><a href="index.html">返回总表</a></p><p>'+html.escape(d["role"]+" / "+d["topic"])+"</p>"]
        for name, cfg in [("baseline-v2", configs["baseline"]), (f["selected_id"], configs["selected"])]:
            flags = [r for r in rows if value(raw[r["sha256"]], cfg) >= cfg["cutoff"]]
            cuts = sorted({0, len(d["text"]), *(i for r in flags for i in [r["start"], r["end"]]), *(i for s in d["ai_spans"] for i in s)})
            pieces = []
            for a, b in zip(cuts, cuts[1:]):
                generated = bool(overlap(d["ai_spans"], a, b)); flagged = next((r for r in flags if r["start"] <= a < r["end"]), None)
                classes = (["known"] if generated else []) + (["false" if not flagged["ai_characters"] else "flag"] if flagged else [])
                tag = "mark" if flagged else "span"
                pieces.append(f'<{tag} class="'+" ".join(classes)+'">'+html.escape(d["text"][a:b])+f'</{tag}>')
            m = metrics(data, raw, cfg, documents={d["id"]})
            parts.append('<h2>'+html.escape(name)+f"</h2><p>纯人工窗口误报 {m['false_windows']}/{m['pure_human_windows']}；叙述单位 {m['hit_units']}/{m['narrative_units']}；生成字符覆盖 {m['covered_generated_characters']}/{m['generated_characters']}。</p><pre>"+"".join(pieces)+"</pre>")
        filename = d["id"] + ".html"
        (target / filename).write_text(page(d["id"], "".join(parts)), encoding="utf-8")
        links.append({"论文/版本": '<a href="'+filename+'">'+html.escape(d["id"])+"</a>", "角色": d["role"], "主题": d["topic"]})
    intro += '<p><a href="../corpus.json">精确输入与源位置映射</a> · <a href="../../../evaluations/english-diverse-20261006/evaluation.json">全部数值及逐主题/生成方式/生成器结果</a></p>'
    intro += table(links, ["论文/版本", "角色", "主题"])
    intro += "<p>转换审计是机械文字对照，未作全部页的视觉校对；人工标签依据来源，分数不证明作者身份。拟合/开发材料不称为独立测试，六篇验收不足以证明总体零误报。</p>"
    (target / "index.html").write_text(page("英文多来源 AIGC 完整正文验收", intro), encoding="utf-8")
    write(REPORTS / "COMPLETE.json", {"status": "complete", "created_at_utc": now(), "documents": len(local["documents"]),
        "report_index": str(target / "index.html"), "evaluation_sha256": sha(EVIDENCE / "evaluation.json"),
        "freeze_sha256": sha(EVIDENCE / "FREEZE.json"), "index_sha256": sha(target / "index.html")})
    print({"report": str(target / "index.html"), "documents": len(local["documents"])}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["declare", "score-development", "diagnose", "freeze", "score-acceptance", "evaluate", "report", "verify"])
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    if args.command == "declare":
        declare(args.source)
    elif args.command.startswith("score-"):
        score(args.command.removeprefix("score-"))
    elif args.command == "verify":
        verify_inputs(check_upstream=True); verify_freeze(); print({"verified": True}, flush=True)
    else:
        {"diagnose": diagnose, "freeze": freeze, "evaluate": evaluate, "report": report}[args.command]()


if __name__ == "__main__":
    main()
