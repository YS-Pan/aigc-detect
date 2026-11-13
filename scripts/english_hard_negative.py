"""Small, predeclared score-only experiment; never installs a runtime policy.

Source labels/positions define training bags and evaluation only. Prediction
accepts raw classifier scores, and no text, length, author or document features.
All papers are previously seen development material, not independent tests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "evaluations/english-hard-negative-20261004"
REPORTS = ROOT / "reports/english-hard-negative-20261004"
PARENT = ROOT / "evaluations/english-academic-20261003-v2"
SAMPLES = ROOT / "samples/english-academic-20261003"
PROFILE = ROOT / "aigc_detector/profiles/english-academic-20261003-v2.json"
METHODS = ("presence", "mean_human_bag_ranking", "hard_human_bag_ranking")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def overlap(a, b):
    return max(0, min(a["char_end"], b["char_end"]) - max(a["char_start"], b["char_start"]))


def build_dataset(source, samples, signals):
    """Whitelist numeric/provenance fields, validate complete source mappings."""
    fields = ("id", "sample_id", "paper_id", "split", "kind", "generator", "paragraph_id",
              "sha256", "char_start", "char_end", "chars", "ai_chars", "cross_paper_duplicate")
    rows = [{**{k: r[k] for k in fields}, "scores": {s: r["scores"][s] for s in signals}}
            for r in source["records"]]
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample_id"]].append(r)
        assert 0 <= r["ai_chars"] <= r["chars"] == r["char_end"] - r["char_start"]
    blocks = []
    for sid, group in by_sample.items():
        sample = samples[sid]
        assert {r["paper_id"] for r in group} == {sample["paper_id"]}
        assert {r["split"] for r in group} == {sample["split"]}
        ordered = sorted(group, key=lambda r: r["char_start"])
        assert ordered[0]["char_start"] == 0 and ordered[-1]["char_end"] == sample["characters"]
        assert all(a["char_end"] == b["char_start"] for a, b in zip(ordered, ordered[1:]))
        if sample["kind"] == "ai_paragraph":
            assert all(r["ai_chars"] == r["chars"] for r in group)
            continue
        mapping = read(SAMPLES / sample["mapping_file"])
        assert mapping["input_sha256"] == sample["sha256"]
        ai = [b for b in mapping["blocks"] if b["source_label"] == "ai"]
        for r in group:
            assert r["ai_chars"] == sum(overlap(r, b) for b in ai)
        if sample["kind"] == "mixed_document":
            for b in ai:
                hit = [r["id"] for r in group if overlap(r, b)]
                assert hit and sum(overlap(r, b) for r in group) == b["char_end"] - b["char_start"]
                blocks.append({"id": sid + ":" + b["block_id"], "sample_id": sid,
                               "paper_id": sample["paper_id"], "generator": sample.get("generator"),
                               "char_start": b["char_start"], "char_end": b["char_end"], "window_ids": hit})
    assert len({r["id"] for r in rows}) == len(rows)
    return {"records": rows, "blocks": blocks, "contains_source_text": False}


def prepare():
    if (EVIDENCE / "campaign.json").exists():
        raise FileExistsError("Preserve the declared campaign")
    parent, profile = read(PARENT / "campaign.json"), read(PROFILE)
    samples = {s["id"]: s for s in map(json.loads, (SAMPLES / "samples.jsonl").read_text(encoding="utf-8").splitlines())}
    paths = [PARENT / "campaign.json", PARENT / "context1600/train-scores.json", PARENT / "repeat-scores.json",
             PROFILE, SAMPLES / "samples.jsonl", Path(__file__)]
    paths += [SAMPLES / s["mapping_file"] for s in samples.values() if "mapping_file" in s]
    dataset = build_dataset(read(PARENT / "context1600/train-scores.json"), samples, profile["models"])
    assert {r["split"] for r in dataset["records"]} == {"fit", "calibration"}
    configs = [{"id": f"{method}-l2-{reg:g}", "method": method, "regularization": reg}
               for method in METHODS for reg in (.01, .1, 1.)]
    campaign = {
        "schema_version": "english-hard-negative-experiment/1.0", "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "groups": parent["groups"], "signals": profile["models"], "revisions": profile["revisions"],
        "segmentation": [1600, 400], "configurations": configs, "candidate_count": len(configs),
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "optimizer": {"algorithm": "CPU float64 L-BFGS strong Wolfe", "max_iter": 250,
                      "max_eval": 400, "gradient_tolerance": 1e-8, "change_tolerance": 1e-12,
                      "starts": "zero and matching presence optimum; choose training loss only"},
        "bag_temperature": .1, "ranking_margin": .1, "threshold_margin": .01,
        "losses": {
            "presence": "Equal-paper/equal-class weighted window BCE + lambda/2 ||w||^2",
            "mean_human_bag_ranking": "Equal-paper bags: human-weighted mean softplus(m+s_h-B_b) + lambda/2 ||w||^2",
            "hard_human_bag_ranking": "Equal-paper bags: softplus(m+H-B_b) + lambda/2 ||w||^2",
            "H": "tau*logsumexp(s_h/tau), an upper bound on the maximum pure-human score",
            "B_b": "tau*logmeanexp(s_w/tau) over windows intersecting generated block b, a lower bound on their maximum",
        },
        "fit_scope": "Original and mixed windows only; isolated generated paragraphs are evaluation-only. Same-paper hashes deduplicated with all bag memberships retained. Cross-paper duplicate training windows and conflicting labels excluded.",
        "standardization": "Fit-only classifier logit clip=1e-5, equal paper/class window weights, SD floor=.5, standardized clip=5",
        "ranking_intercept": "Fixed zero: a common intercept cancels in score differences and control thresholds",
        "selection": "Six fit-paper LOPO; lexicographic human false windows, negative mixed block hits, false originals, negative mixed character hits, stronger L2, config id. Zero-false feasibility reported separately; infeasible winner is diagnostic only.",
        "nested_diagnostic": "Outer six-paper LOPO. Inner five-paper LOPO reselects config globally and separately per objective. Every fit, standardization, bag and threshold excludes its held paper. No calibration/test data used.",
        "final_threshold": "Max pure-human score in six fit plus two calibration papers + fixed margin. Calibration positives never enter fitting, selection, standardization or cutoff.",
        "repeat_rule": "Freeze all three family champions and overall winner before loading P07/P08 cached scores. No retuning after repeat.",
        "success_rule": "Hard-ranking family must have zero pure-human false windows and >=50% mixed-block hits both in nested six-paper diagnostic and frozen P07/P08 repeat. Compare controls and per-paper outcomes; otherwise stop this campaign.",
        "runtime_features": "Five pinned raw classifier scores only", "new_independent_validation": False,
        "historical_results_known": True, "automatic_deployment": False, "contains_source_text": False,
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "python": sys.version},
    }
    write(EVIDENCE / "development-data.json", dataset)
    campaign["development_data_sha256"] = sha(EVIDENCE / "development-data.json")
    write(EVIDENCE / "campaign.json", campaign)
    print(json.dumps({"prepared": str(EVIDENCE), "candidates": len(configs), "records": len(dataset["records"]), "blocks": len(dataset["blocks"])}), flush=True)


def verify_campaign():
    campaign = read(EVIDENCE / "campaign.json")
    for path, expected in campaign["source_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Declared source changed: {path}")
    assert sha(EVIDENCE / "development-data.json") == campaign["development_data_sha256"]
    return campaign


def training_view(data, papers):
    grouped = defaultdict(list)
    for r in data["records"]:
        if r["paper_id"] in papers and r["kind"] in ("human_document", "mixed_document") and not r["cross_paper_duplicate"]:
            grouped[r["paper_id"], r["sha256"]].append(r)
    rows, lookup, excluded = [], {}, []
    for key, group in sorted(grouped.items()):
        if len({r["ai_chars"] for r in group}) != 1:
            excluded.extend(r["id"] for r in group)
            continue
        assert all(r["scores"] == group[0]["scores"] for r in group)
        for r in group:
            lookup[r["id"]] = len(rows)
        rows.append(group[0])
    bags, bag_papers = [], []
    for b in data["blocks"]:
        if b["paper_id"] in papers:
            indices = sorted({lookup[i] for i in b["window_ids"] if i in lookup})
            if not indices:
                raise ValueError("No eligible window for a training bag; do not silently drop it")
            assert all(rows[i]["ai_chars"] > 0 for i in indices)
            bags.append(indices)
            bag_papers.append(b["paper_id"])
    assert rows and bags and all(r["paper_id"] in papers for r in rows)
    return rows, bags, bag_papers, excluded


def raw_matrix(rows, signals):
    p = np.array([[r["scores"][s] for s in signals] for r in rows], dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid raw classifier scores")
    p = np.clip(p, 1e-5, 1-1e-5)
    return np.log(p / (1-p))


def balanced_weights(rows):
    counts = Counter((r["paper_id"], r["ai_chars"] > 0) for r in rows)
    weights = np.array([1 / counts[r["paper_id"], r["ai_chars"] > 0] for r in rows])
    return weights / weights.sum()


def make_loss(z, rows, bags, bag_papers, config, campaign):
    x = torch.tensor(z, dtype=torch.float64)
    y = torch.tensor([float(r["ai_chars"] > 0) for r in rows], dtype=torch.float64)
    weights = torch.tensor(balanced_weights(rows), dtype=torch.float64)
    human = y == 0
    hweights = weights[human] / weights[human].sum()
    membership = torch.zeros((len(bags), len(rows)), dtype=torch.bool)
    for i, bag in enumerate(bags):
        membership[i, bag] = True
    sizes = membership.sum(dim=1).to(torch.float64)
    counts = Counter(bag_papers)
    bw = torch.tensor([1 / counts[p] for p in bag_papers], dtype=torch.float64)
    bw /= bw.sum()
    tau, margin = campaign["bag_temperature"], campaign["ranking_margin"]

    def loss(theta):
        scores = x @ theta[:x.shape[1]]
        if config["method"] == "presence":
            base = (F.binary_cross_entropy_with_logits(scores + theta[-1], y, reduction="none") * weights).sum()
        else:
            bs = tau * (torch.logsumexp((scores[None, :] / tau).masked_fill(~membership, -torch.inf), dim=1) - torch.log(sizes))
            if config["method"] == "hard_human_bag_ranking":
                hs = tau * torch.logsumexp(scores[human] / tau, dim=0)
                per_bag = F.softplus(margin + hs - bs)
            elif config["method"] == "mean_human_bag_ranking":
                per_bag = (F.softplus(margin + scores[human, None] - bs[None, :]) * hweights[:, None]).sum(dim=0)
            else:
                raise ValueError(config["method"])
            base = (per_bag * bw).sum()
        return base + .5 * config["regularization"] * theta[:x.shape[1]].square().sum()
    return loss


def optimize(loss, start, campaign):
    theta = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    cfg = campaign["optimizer"]
    optimizer = torch.optim.LBFGS([theta], max_iter=cfg["max_iter"], max_eval=cfg["max_eval"],
                                  tolerance_grad=cfg["gradient_tolerance"], tolerance_change=cfg["change_tolerance"],
                                  line_search_fn="strong_wolfe", history_size=30)
    initial = float(loss(theta).detach())
    def closure():
        optimizer.zero_grad()
        value = loss(theta)
        value.backward()
        return value
    optimizer.step(closure)
    final = closure()
    gradient = float(theta.grad.abs().max())
    state = optimizer.state[theta]
    assert torch.isfinite(theta).all() and float(final.detach()) <= initial + 1e-10
    return theta.detach().numpy(), {"initial_loss": initial, "final_loss": float(final.detach()),
                                   "gradient_inf": gradient, "iterations": state["n_iter"],
                                   "evaluations": state["func_evals"], "stationary_1e_5": gradient < 1e-5}


def fit_model(data, papers, config, campaign):
    rows, bags, bag_papers, excluded = training_view(data, set(papers))
    raw = raw_matrix(rows, campaign["signals"])
    weights = balanced_weights(rows)
    center = np.average(raw, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((raw-center)**2, axis=0, weights=weights)), .5)
    z = np.clip((raw-center)/scale, -5, 5)
    n = z.shape[1]
    presence = {**config, "method": "presence"}
    theta0, info0 = optimize(make_loss(z, rows, bags, bag_papers, presence, campaign), np.zeros(n+1), campaign)
    if config["method"] == "presence":
        theta, info, starts = theta0, info0, [info0]
    else:
        loss = make_loss(z, rows, bags, bag_papers, config, campaign)
        fitted = [optimize(loss, start, campaign) for start in (np.zeros(n), theta0[:n])]
        theta, info = min(fitted, key=lambda p: p[1]["final_loss"])
        starts = [item[1] for item in fitted]
    model = {"signals": campaign["signals"], "center": center.tolist(), "scale": scale.tolist(),
             "coefficients": theta[:n].tolist(), "intercept": float(theta[-1]) if config["method"] == "presence" else 0.,
             "configuration": config, "fit_papers": sorted(papers), "fit_unique_windows": len(rows), "fit_bags": len(bags),
             "excluded_conflicting_window_ids": excluded, "optimization": {**info, "starts": starts}}
    return model


def predict(rows, model):
    # Only scores are read by this runtime function.
    z = np.clip((raw_matrix(rows, model["signals"])-model["center"])/model["scale"], -5, 5)
    return z @ np.asarray(model["coefficients"]) + model["intercept"]


def cutoff(rows, model, margin):
    human = [r for r in rows if r["ai_chars"] == 0]
    if not human:
        raise ValueError("No pure-human controls")
    return float(predict(human, model).max()) + margin


def metrics(data, papers, model, threshold):
    rows = [r for r in data["records"] if r["paper_id"] in papers]
    values = predict(rows, model)
    flags = {r["id"]: bool(v >= threshold) for r, v in zip(rows, values)}
    blocks = [b for b in data["blocks"] if b["paper_id"] in papers]
    human = [r for r in rows if r["ai_chars"] == 0]
    original = [r for r in rows if r["kind"] == "human_document"]
    mixed = [r for r in rows if r["kind"] == "mixed_document"]
    isolated = {r["sample_id"] for r in rows if r["kind"] == "ai_paragraph"}
    result = {
        "pure_human_windows": len(human), "pure_human_false_windows": sum(flags[r["id"]] for r in human),
        "original_windows": len(original), "original_false_windows": sum(flags[r["id"]] for r in original),
        "original_documents": len({r["sample_id"] for r in original}),
        "original_false_documents": len({r["sample_id"] for r in original if flags[r["id"]]}),
        "mixed_blocks": len(blocks), "mixed_block_hits": sum(any(flags[i] for i in b["window_ids"]) for b in blocks),
        "mixed_ai_chars": sum(r["ai_chars"] for r in mixed),
        "mixed_ai_hit_chars": sum(r["ai_chars"] for r in mixed if flags[r["id"]]),
        "mixed_non_ai_chars_in_positive_windows": sum(r["chars"]-r["ai_chars"] for r in mixed if r["ai_chars"] > 0 and flags[r["id"]]),
        "isolated_blocks": len(isolated),
        "isolated_block_hits": sum(any(flags[r["id"]] for r in rows if r["sample_id"] == sid) for sid in isolated),
    }
    return result


def summed(items):
    items = list(items)
    return {k: sum(item[k] for item in items) for k in items[0]}


def objective(candidate):
    m, c = candidate["metrics"], candidate["configuration"]
    return (m["pure_human_false_windows"], -m["mixed_block_hits"], m["original_false_documents"],
            -m["mixed_ai_hit_chars"], -c["regularization"], c["id"])


def fit_and_freeze():
    if (EVIDENCE / "FREEZE.json").exists():
        raise FileExistsError("Frozen experiment cannot be reselected")
    campaign = verify_campaign()
    data = read(EVIDENCE / "development-data.json")
    papers = sorted(campaign["groups"]["fit"])
    identity = sha(EVIDENCE / "campaign.json")
    model_cache = {}
    def model_for(train_papers, config):
        key = (tuple(sorted(train_papers)), config["id"])
        if key not in model_cache:
            path = REPORTS / "fit-cache" / ("-".join(key[0]) + "--" + key[1] + ".json")
            if path.exists():
                cached = read(path)
                assert cached["campaign_sha256"] == identity
                model_cache[key] = cached["model"]
            else:
                model_cache[key] = fit_model(data, train_papers, config, campaign)
                write(path, {"campaign_sha256": identity, "model": model_cache[key]})
        return model_cache[key]

    def compare(pool):
        candidates = []
        for config in campaign["configurations"]:
            folds = []
            for held in pool:
                train = [p for p in pool if p != held]
                model = model_for(train, config)
                controls = [r for r in data["records"] if r["paper_id"] in train]
                threshold = cutoff(controls, model, campaign["threshold_margin"])
                folds.append({"held_paper": held, "fit_papers": train, "threshold": threshold,
                              "metrics": metrics(data, {held}, model, threshold)})
            candidates.append({"configuration": config, "folds": folds, "metrics": summed(f["metrics"] for f in folds)})
        return candidates

    candidates = compare(papers)
    write(REPORTS / "selection-progress.json", {"candidates": candidates})
    print(json.dumps({"selection_comparison": [{"id": c["configuration"]["id"], **c["metrics"]} for c in candidates]}), flush=True)
    nested = []
    for held in papers:
        train = [p for p in papers if p != held]
        inner = compare(train)
        choices = {method: min((c for c in inner if c["configuration"]["method"] == method), key=objective) for method in METHODS}
        choices["overall"] = min(inner, key=objective)
        fold = {"held_paper": held, "inner_candidates": inner, "methods": {}}
        for name, chosen in choices.items():
            model = model_for(train, chosen["configuration"])
            controls = [r for r in data["records"] if r["paper_id"] in train]
            threshold = cutoff(controls, model, campaign["threshold_margin"])
            fold["methods"][name] = {"selected": chosen["configuration"], "inner_zero_false_feasible": chosen["metrics"]["pure_human_false_windows"] == 0,
                                      "model": model, "threshold": threshold, "metrics": metrics(data, {held}, model, threshold)}
        nested.append(fold)
        write(REPORTS / "nested-progress.json", {"folds": nested})
        print(json.dumps({"nested_held": held, "methods": {k: v["metrics"] for k, v in fold["methods"].items()}}), flush=True)

    choices = {method: min((c for c in candidates if c["configuration"]["method"] == method), key=objective) for method in METHODS}
    chosen = min(candidates, key=objective)
    models = {}
    for method, candidate in choices.items():
        model = model_for(papers, candidate["configuration"])
        controls = [r for r in data["records"] if r["paper_id"] in papers + campaign["groups"]["calibration"]]
        threshold = cutoff(controls, model, campaign["threshold_margin"])
        models[method] = {"model": model, "threshold": threshold,
                          "cv_zero_false_feasible": candidate["metrics"]["pure_human_false_windows"] == 0}
    verify_campaign()
    selection = {"candidates": candidates, "selected": chosen, "family_champions": choices,
                 "nested": nested, "nested_totals": {name: summed(f["methods"][name]["metrics"] for f in nested) for name in (*METHODS, "overall")},
                 "unique_fits": len(model_cache),
                 "optimization": {"nonstationary_fits": sum(not m["optimization"]["stationary_1e_5"] for m in model_cache.values()),
                                  "max_gradient_inf": max(m["optimization"]["gradient_inf"] for m in model_cache.values())},
                 "new_independent_validation": False, "repeat_scores_used": False}
    write(EVIDENCE / "selection.json", selection)
    write(EVIDENCE / "frozen-models.json", {"models": models, "overall_method": chosen["configuration"]["method"], "contains_source_text": False})
    write(EVIDENCE / "FREEZE.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "campaign_sha256": identity,
                                      "selection_sha256": sha(EVIDENCE / "selection.json"),
                                      "models_sha256": sha(EVIDENCE / "frozen-models.json"),
                                      "development_data_sha256": sha(EVIDENCE / "development-data.json"),
                                      "repeat_rule": campaign["repeat_rule"]})
    print(json.dumps({"frozen": True, "selected": chosen["configuration"], "nested_totals": selection["nested_totals"], "unique_fits": len(model_cache)}), flush=True)


def verify_freeze():
    campaign = verify_campaign()
    freeze = read(EVIDENCE / "FREEZE.json")
    for field, file in [("campaign_sha256", "campaign.json"), ("selection_sha256", "selection.json"), ("models_sha256", "frozen-models.json"), ("development_data_sha256", "development-data.json")]:
        assert sha(EVIDENCE / file) == freeze[field], file
    return campaign


def evaluate():
    campaign = verify_freeze()
    if (EVIDENCE / "evaluation.json").exists():
        raise FileExistsError("Preserve evaluation; use verify for reproduction")
    samples = {s["id"]: s for s in map(json.loads, (SAMPLES / "samples.jsonl").read_text(encoding="utf-8").splitlines())}
    repeated = build_dataset(read(PARENT / "repeat-scores.json"), samples, campaign["signals"])
    dev = read(EVIDENCE / "development-data.json")
    data = {k: dev[k] + repeated[k] for k in ("records", "blocks")}
    assert len(data["records"]) == 678 and len(data["blocks"]) == 75
    assert not {r["sha256"] for r in dev["records"]} & {r["sha256"] for r in repeated["records"]}, "Cross-split duplicate leakage"
    models = read(EVIDENCE / "frozen-models.json")
    profile = read(PROFILE)
    baseline = {"signals": profile["models"], "center": profile["center"], "scale": profile["scale"],
                "coefficients": profile["heads"]["review"]["coefficients"], "intercept": profile["heads"]["review"]["intercept"]}
    basethreshold = float(np.log(profile["possible_threshold"]/(1-profile["possible_threshold"])))
    from aigc_detector.fusion import fused_score
    expected = predict(data["records"], baseline)
    production = [fused_score(r["scores"], profile)["review_logit"] for r in data["records"]]
    error = float(np.max(np.abs(expected - production)))
    assert error < 1e-10
    all_models = {"frozen_v2": {"model": baseline, "threshold": basethreshold}, **models["models"]}
    results = {}
    for name, item in all_models.items():
        m, t = item["model"], item["threshold"]
        results[name] = {"threshold": t, "by_split": {split: metrics(data, set(papers), m, t) for split, papers in campaign["groups"].items()},
                         "all": metrics(data, {r["paper_id"] for r in data["records"]}, m, t),
                         "by_paper": {p: metrics(data, {p}, m, t) for p in sorted({r["paper_id"] for r in data["records"]})}}
    selection = read(EVIDENCE / "selection.json")
    hard_nested = selection["nested_totals"]["hard_human_bag_ranking"]
    hard_repeat = results["hard_human_bag_ranking"]["by_split"]["test"]
    passed = all(m["pure_human_false_windows"] == 0 and m["mixed_block_hits"] >= .5*m["mixed_blocks"] for m in (hard_nested, hard_repeat))
    evidence = {"results": results, "nested_totals": selection["nested_totals"], "hard_ranking_meets_declared_target": passed,
                "baseline_production_max_abs_logit_error": error, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "freeze_sha256": sha(EVIDENCE / "FREEZE.json"), "new_independent_validation": False,
                "no_new_base_model_inference": True, "default_profile_unchanged": sha(PROFILE) == campaign["source_hashes"][PROFILE.relative_to(ROOT).as_posix()],
                "contains_source_text": False}
    write(EVIDENCE / "repeat-data.json", repeated)
    write(EVIDENCE / "evaluation.json", evidence)
    print(json.dumps(evidence), flush=True)


def verify():
    campaign = verify_freeze()
    data = read(EVIDENCE / "development-data.json")
    frozen = read(EVIDENCE / "frozen-models.json")
    errors = {}
    for name, item in frozen["models"].items():
        rebuilt = fit_model(data, campaign["groups"]["fit"], item["model"]["configuration"], campaign)
        errors[name] = float(np.max(np.abs(predict(data["records"], rebuilt) - predict(data["records"], item["model"]))))
        assert errors[name] < 1e-10
        controls = [r for r in data["records"] if r["paper_id"] in campaign["groups"]["fit"] + campaign["groups"]["calibration"]]
        assert abs(cutoff(controls, rebuilt, campaign["threshold_margin"]) - item["threshold"]) < 1e-10
    result = {"reproduced_champions": errors, "freeze_verified": True}
    if (EVIDENCE / "evaluation.json").exists():
        evaluation = read(EVIDENCE / "evaluation.json")
        assert evaluation["freeze_sha256"] == sha(EVIDENCE / "FREEZE.json")
        samples = {s["id"]: s for s in map(json.loads, (SAMPLES / "samples.jsonl").read_text(encoding="utf-8").splitlines())}
        repeated = build_dataset(read(PARENT / "repeat-scores.json"), samples, campaign["signals"])
        assert repeated == read(EVIDENCE / "repeat-data.json")
        all_data = {k: data[k] + repeated[k] for k in ("records", "blocks")}
        for name, item in frozen["models"].items():
            actual = evaluation["results"][name]
            for split, papers in campaign["groups"].items():
                assert actual["by_split"][split] == metrics(all_data, set(papers), item["model"], item["threshold"])
            assert actual["all"] == metrics(all_data, {r["paper_id"] for r in all_data["records"]}, item["model"], item["threshold"])
        result["repeat_metrics_reproduced"] = True
    print(json.dumps(result), flush=True)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit", "evaluate", "verify"))
    command = parser.parse_args().command
    {"prepare": prepare, "fit": fit_and_freeze, "evaluate": evaluate, "verify": verify}[command]()


if __name__ == "__main__":
    main()
