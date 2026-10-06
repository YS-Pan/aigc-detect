"""Independent count, source, frozen-parameter and CLI checks for this campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import english_diverse_validation as e
from scripts.english_diverse_corpus import SOURCE, digest, read, write


def check_source():
    c = e.verify_inputs(check_upstream=True)
    upstream = Path(c["source"])
    paths = ["experiments/manifest.json", "experiments/batches/20261005-diverse20/config.json",
             "experiments/batches/20261005-additional-types/manifest.json", "qa/validation.json",
             "qa/batches/20261005-diverse20/delivery-audit.json", "qa/batches/20261005-additional-types/mechanical-validation.json",
             "qa/batches/20261005-additional-types/delivery-audit.json"]
    files = {p: digest((upstream/p).read_bytes()) for p in paths}
    old = ROOT / "samples/english-academic-20261003"
    previous = [json.loads(s) for s in (old/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    docs = read(e.REPORTS/"corpus.json")["documents"]
    mapping = {(r["paper_id"], r.get("generator"), r["kind"]): r["sha256"] for r in previous if r["kind"] != "ai_paragraph"}
    comparisons = []
    for d in docs:
        if int(d["paper_id"][1:]) <= 10 and d["sample_type"] in ["original", "paragraph_compress_restore"]:
            h = mapping[d["paper_id"], d["generator"], "human_document" if d["sample_type"] == "original" else "mixed_document"]
            assert h == d["sha256"], "Legacy full input differs"
            comparisons.append(d["id"])
    manifest = read(ROOT/"models/manifest.json")["models"]
    model_files = []
    for m in c["models"]:
        for path, info in manifest[m]["files"].items():
            file = ROOT/"models"/m/path
            assert file.stat().st_size == info["bytes"] and digest(file.read_bytes()) == info["sha256"]
            model_files.append({"model": m, "path": path, "sha256": info["sha256"]})
    write(e.EVIDENCE/"source-and-assets-audit.json", {"at_utc": e.now(), "manifest_qa_hashes": files,
        "legacy_identical_full_inputs": comparisons, "model_files_verified": model_files,
        "upstream_source_hashes_verified": len(c["source_files"]), "no_source_writes": True,
        "scope": "Current corpus hash/span/reversal checks and upstream QA documents; no claim of complete actual generator tool-read transcripts or full visual proofreading."})
    print({"source_files_verified": len(c["source_files"]), "model_files_verified": len(model_files),
           "identical_legacy_full_inputs": len(comparisons)}, flush=True)


def independent_logit(raw, config, baseline):
    # Separate implementation of both deployed v2 and experimental linear math.
    if config["id"] in ["baseline-v2", "v2-human-calibration"]:
        eps = baseline["clip_probability"]
        center, scale = baseline["center"], baseline["scale"]
        params = baseline["heads"]["review"]
        coefficients, intercept = params["coefficients"], params["intercept"]
        models = baseline["models"]
    else:
        eps = config["clip_probability"]
        center, scale = config["center"], config["scale"]
        coefficients, intercept, models = config["coefficients"], config["intercept"], config["models"]
    features = []
    for m, a, b in zip(models, center, scale):
        p = min(1-eps, max(eps, raw[m]))
        features.append(min(5., max(-5., (math.log(p/(1-p))-a)/b)))
    return intercept + sum(x*y for x, y in zip(features, coefficients))


def independent_counts(data, scores, configuration, docs):
    expected = {d["id"] for d in docs}
    profile = read(e.PROFILE)
    windows = [r for r in data["windows"] if r["document_id"] in expected]
    predicted = {r["id"] for r in windows if independent_logit(scores[r["sha256"]], configuration, profile) >= configuration["cutoff"]}
    totals = defaultdict(int)
    for d in docs:
        ai, flagged = bytearray(d["characters"]), bytearray(d["characters"])
        for a, b in d["ai_spans"]:
            ai[a:b] = b"\x01"*(b-a)
        totals["generated_characters"] += sum(ai)
        totals["narrative_units"] += len(d["units"])
        for r in [r for r in windows if r["document_id"] == d["id"]]:
            n = sum(ai[r["start"]:r["end"]])
            assert n == r["ai_characters"], "Independent label count mismatch"
            if n == 0:
                totals["pure_human_windows"] += 1
            if r["id"] in predicted:
                flagged[r["start"]:r["end"]] = b"\x01"*r["characters"]
                if n:
                    totals["positive_window_human_characters"] += r["characters"]-n
                    totals["positive_window_generated_characters"] += n
                else:
                    totals["false_windows"] += 1
        totals["covered_generated_characters"] += sum(a*b for a, b in zip(ai, flagged))
        totals["hit_units"] += sum(any(any(flagged[a:b]) for a, b in u["spans"]) for u in d["units"])
        totals["hit_sections"] += sum(any(any(flagged[a:b]) for a, b in u["spans"]) for u in d["section_units"])
    return totals, predicted


def final_audit():
    c, freeze = e.verify_freeze()
    e.verify_inputs(check_upstream=True)
    data = read(e.EVIDENCE/"inputs.json")
    scores = {**read(e.EVIDENCE/"development-scores.json")["scores"], **read(e.EVIDENCE/"acceptance-scores.json")["scores"]}
    configurations = read(e.EVIDENCE/"frozen-configurations.json")
    results = read(e.EVIDENCE/"evaluation.json")
    methods = {"baseline-v2": configurations["baseline"], **configurations["candidates"]}
    checked = []
    for name, config in methods.items():
        for role, ids in {"all": None, **c["groups"]}.items():
            docs = [d for d in data["documents"] if ids is None or d["paper_id"] in ids]
            totals, predicted = independent_counts(data, scores, config, docs)
            reference = results["methods"][name]["all"] if role == "all" else results["methods"][name]["by_role"][role]
            for k in ["generated_characters", "narrative_units", "pure_human_windows", "positive_window_human_characters",
                      "positive_window_generated_characters", "false_windows", "covered_generated_characters", "hit_units", "hit_sections"]:
                assert totals[k] == reference[k], (name, role, k)
            assert predicted == set(reference["flag_ids"])
            checked.append(name+"::"+role)
    selection = read(e.EVIDENCE/"selection.json")
    maximum_difference = 0.
    for fold in selection["leave_author_group_out_fixed_refit"]:
        train, held = set(fold["training"]), set(fold["held"])
        assert not train & held and train | held == set(c["groups"]["fit"])
        assert all(not (train & set(g) and held & set(g)) for g in c["author_source_groups"])
    # Refit from frozen NUMERIC data only; no source text or GPU access.
    reproduced = e.calibrate(data, scores, e.fit_linear(data, scores, set(c["groups"]["fit"]), c["author_source_groups"]),
                             set(c["groups"]["fit"]+c["groups"]["development"]+c["groups"]["calibration"]))
    reference = configurations["candidates"]["presence-linear-refit"]
    for k in ["center", "scale", "coefficients"]:
        maximum_difference = max(maximum_difference, max(abs(x-y) for x, y in zip(reproduced[k], reference[k])))
    for k in ["intercept", "cutoff"]:
        maximum_difference = max(maximum_difference, abs(reproduced[k]-reference[k]))
    assert maximum_difference < 1e-12
    # Paper-group bootstrap: six groups are the independent resampling units,
    # every generated version remains bound. These intervals are descriptive.
    import random
    rng = random.Random(e.SEED)
    uncertainty = {}
    for name, cfg in methods.items():
        outcomes = [e.metrics(data, scores, cfg, {p}) for p in c["groups"]["acceptance"]]
        boot = []
        for _ in range(2000):
            sample = rng.choices(outcomes, k=len(outcomes))
            boot.append(sum(v["hit_units"] for v in sample)/sum(v["narrative_units"] for v in sample))
        boot.sort()
        uncertainty[name] = {"paper_cluster_bootstrap_95_interval": [boot[50], boot[1949]], "source_groups": len(outcomes),
                             "scope": "Descriptive source-cluster uncertainty with only six groups; zero observed FP does not establish population zero."}
    write(e.EVIDENCE/"independent-audit.json", {"at_utc": e.now(), "checked_method_roles": checked,
        "formula_and_character_mask_agreement": True, "author_holdout_folds_verified": len(selection["leave_author_group_out_fixed_refit"]),
        "numeric_refit_max_absolute_difference": maximum_difference, "uncertainty": uncertainty,
        "baseline_profile_sha256_unchanged": e.sha(e.PROFILE) == c["baseline_profile_sha256"],
        "source_files_unchanged": len(c["source_files"]), "freeze_sha256": e.sha(e.EVIDENCE/"FREEZE.json")})
    print({"independent_checks": len(checked), "numeric_refit_difference": maximum_difference, "source_files_unchanged": len(c["source_files"])}, flush=True)


def compatibility():
    c, freeze = e.verify_freeze()
    local = read(e.REPORTS/"corpus.json")
    data = read(e.EVIDENCE/"inputs.json")
    scores = read(e.EVIDENCE/"development-scores.json")["scores"]
    # Predetermined new development paper P14, full original and Astra mixed
    # input. Each tests the installed CLI from outside the project cwd.
    checks = []
    for kind in ["original", "paragraph_compress_restore"]:
        d = next(d for d in local["documents"] if d["paper_id"] == "P14" and d["sample_type"] == kind
                 and d["generator"] in [None, "gpt-6-astra"])
        source = e.REPORTS/"cli-inputs"/(d["id"]+".txt"); source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(d["text"], encoding="utf-8", newline="\n")
        target = e.REPORTS/"cli"/d["id"]
        if target.exists():
            raise FileExistsError("Preserve the previous CLI check")
        command = [str(ROOT/".venv/Scripts/python.exe"), "-X", "utf8", "-B", "-m", "aigc_detector", "annotate",
                   "--language", "en", "--file", str(source), "--report-dir", str(target)]
        result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, encoding="utf-8", errors="replace")
        (e.REPORTS/"cli-inputs"/(d["id"]+".log")).write_text(result.stdout+result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stderr[-1000:]
        report = read(target/"report.json")
        complete = read(target/"COMPLETE.json")
        assert report["input"]["sha256"] == d["sha256"] == complete["input_sha256"]
        assert report["policy"]["id"] == e.POLICY and report["policy"]["profile_sha256"] == c["baseline_profile_sha256"]
        rows = [r for r in data["windows"] if r["document_id"] == d["id"]]
        differences = []
        for r, s in zip(rows, report["segments"]):
            assert [r["start"], r["end"]] == [s["char_start"], s["char_end"]]
            assert r["sha256"] == s["sha256"]
            for mid in c["models"]:
                differences.append(abs(scores[r["sha256"]][mid]-s["model_scores"][mid]["ai_score"]))
            from aigc_detector.fusion import decide_fusion
            p = read(e.PROFILE)
            expected = decide_fusion(scores[r["sha256"]], p, p["high_threshold"], p["possible_threshold"])
            assert expected["category"] == s["decision"]["category"]
        assert len(rows) == len(report["segments"]) and max(differences) < 2e-5
        assert complete["status"] == "complete"
        for path in complete["files"]:
            assert (target/path).is_file()
        checks.append({"id": d["id"], "segments": len(rows), "max_raw_score_absolute_difference": max(differences),
                       "report_sha256": e.sha(target/"report.json"), "complete_sha256": e.sha(target/"COMPLETE.json")})
    # Skill says raw detect=old four, annotate=frozen five; parser remains usable.
    from aigc_detector.engine import select_models
    from aigc_detector.fusion import resolve_policy
    assert len(select_models(None, language="en")) == 4 and resolve_policy(None, "en") == e.POLICY
    help_run = subprocess.run([str(ROOT/".venv/Scripts/python.exe"), "-B", "-m", "aigc_detector", "recombine", "--help"],
                              cwd=ROOT.parent, capture_output=True, encoding="utf-8")
    assert help_run.returncode == 0
    original = e.REPORTS/"cli"/checks[0]["id"]
    recombined = e.REPORTS/"cli"/"recombined"
    assert "--input-dir" in help_run.stdout and "--report-dir" in help_run.stdout
    command = [str(ROOT/".venv/Scripts/python.exe"), "-X", "utf8", "-B", "-m", "aigc_detector", "recombine",
               "--input-dir", str(original), "--report-dir", str(recombined)]
    recombination = subprocess.run(command, cwd=ROOT.parent, capture_output=True, encoding="utf-8")
    (e.REPORTS/"cli-inputs"/"recombine.log").write_text(recombination.stdout+recombination.stderr, encoding="utf-8")
    assert recombination.returncode == 0, recombination.stderr
    assert read(recombined/"report.json")["policy"]["id"] == e.POLICY
    write(e.EVIDENCE/"compatibility.json", {"at_utc": e.now(), "cross_cwd": str(ROOT.parent), "full_document_cli_checks": checks,
        "raw_detect_default_four_models": True, "annotate_default_five_models": True, "cache_recombine": True,
        "skill_sha256": e.sha(ROOT/".agents/skills/aigc-detect/SKILL.md"), "contract_changed": False})
    print({"cli_full_documents": len(checks), "recombined": True}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["source", "final", "compatibility"])
    args = parser.parse_args()
    {"source": check_source, "final": final_audit, "compatibility": compatibility}[args.command]()


if __name__ == "__main__":
    main()
