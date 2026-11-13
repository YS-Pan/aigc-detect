"""Fit a score-only, regularized fusion profile to an explicit reference dataset.

Input records contain model scores and reference span fractions, not text. This
is project adaptation, not a claim of detector accuracy. Threshold selection
is separate from the classifier fit. All requested target constraints and
unsatisfied constraints are recorded; the script never fabricates a rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.fusion import LINEAR_POLICY as REFERENCE_POLICY


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def transform(raw, center, scale):
    clipped = np.clip(raw, 1e-5, 1 - 1e-5)
    return np.clip((np.log(clipped / (1 - clipped)) - center) / scale, -5, 5)


def fit(raw, targets, weights, regularization=.1):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be positive and finite")
    clipped = np.clip(raw, 1e-5, 1 - 1e-5)
    logits = np.log(clipped / (1 - clipped))
    center = np.average(logits, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((logits-center)**2, axis=0, weights=weights)), .5)
    x = np.column_stack([np.ones(len(raw)), transform(raw, center, scale)])
    parameters = np.zeros(x.shape[1]); parameters[0] = -4
    penalty = np.eye(len(parameters)) * regularization; penalty[0, 0] = 1e-6
    weights = weights / weights.mean()
    for _ in range(200):
        predicted = sigmoid(x @ parameters)
        gradient = x.T @ (weights * (predicted-targets)) + penalty @ parameters
        hessian = x.T @ ((weights * predicted * (1-predicted))[:, None] * x) + penalty
        step = np.linalg.solve(hessian, gradient)
        parameters -= .8 * step
        if np.max(np.abs(step)) < 1e-7:
            break
    if not np.isfinite(parameters).all():
        raise ValueError("Non-finite fitted parameters")
    return center, scale, parameters


def predict(raw, model):
    center, scale, parameters = model
    return sigmoid(parameters[0] + transform(raw, center, scale) @ parameters[1:])


def metrics(records, scores, high, possible):
    result = {}
    for name in sorted({r["document"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["document"] == name]
        chars = np.array([records[i]["chars"] for i in indices])
        red = np.array([records[i]["strong"] for i in indices])
        selected = scores[indices]
        strong_chars = int(chars[selected >= high].sum())
        review_chars = int(chars[(selected >= possible) & (selected < high)].sum())
        total = int(chars.sum()); overlap = float((chars*red*(selected >= high)).sum())
        red_total = float((chars*red).sum())
        result[name] = {"characters": total, "segments": len(indices),
                        "strong_characters": strong_chars, "possible_characters": review_chars,
                        "strong_percent": 100*strong_chars/total, "possible_percent": 100*review_chars/total,
                        "any_flag_percent": 100*(strong_chars+review_chars)/total,
                        "cnki_strong_overlap_characters": round(overlap, 4),
                        "cnki_strong_recall": overlap/red_total if red_total else None,
                        "cnki_strong_precision": overlap/strong_chars if strong_chars and red_total else None}
    return result


def choose_threshold(records, scores, *, target_extra_pp, human_limit_pp=0):
    own = np.array([r["document"] == "thesis" for r in records])
    chars = np.array([r["chars"] for r in records]); red = np.array([r["strong"] for r in records])
    baseline = float((chars[own]*red[own]).sum()/chars[own].sum()*100)
    target = baseline + target_extra_pp
    human_names = sorted({r["document"] for r in records if r["document"] != "thesis"})
    candidates = sorted(set([1.0] + [float(x) for x in scores]))
    best = None
    for threshold in candidates:
        above = scores >= threshold
        rates = []
        for name in human_names:
            mask = np.array([r["document"] == name for r in records])
            rates.append(float(chars[mask & above].sum()/chars[mask].sum()*100))
        if max(rates, default=0) > human_limit_pp + 1e-9:
            continue
        own_rate = float(chars[own & above].sum()/chars[own].sum()*100)
        if own_rate < baseline:
            continue
        overlap = float((chars[own & above]*red[own & above]).sum())
        # Primary: honor the requested modest excess, then prefer fewer human
        # flags and greater spatial overlap with the commercial report.
        objective = (abs(own_rate-target), sum(rates), -overlap)
        if best is None or objective < best[0]:
            best = objective, threshold, own_rate, rates
    if best is None:
        raise ValueError("No threshold exceeds the CNKI prose baseline under the specified human-control cap")
    # Move to the midpoint between adjacent scores. The decision then does not
    # depend on equality at a reference sample or last-bit CPU/GPU differences.
    low = max((float(x) for x in scores if x < best[1]), default=0)
    selected_threshold = (low + best[1])/2
    return selected_threshold, {"baseline_percent": baseline, "requested_extra_percentage_points": target_extra_pp,
        "target_percent": target, "achieved_percent": best[2], "human_control_limit_percent": human_limit_pp,
        "human_control_percentages": dict(zip(human_names, best[3])),
        "selection": "Closest character-weighted thesis rate above the CNKI prose baseline under the human-control cap; midpoint threshold.",
        "is_reference_fit_evidence": True, "is_independent_validation": False}


def blocked_checks(records, raw, target, weights, regularization, human_limit_pp=0):
    checks = []
    # Hold out each complete human document: checks transfer to a different
    # writer. Hold out both CNKI-positive chapters separately: checks whether
    # one positive region predicts the other without passage-level leakage.
    groups = [("human_document", name) for name in sorted({r["document"] for r in records if r["document"] != "thesis"})]
    groups += [("cnki_chapter", unit) for unit in sorted({r["unit"] for r in records if r["document"] == "thesis" and r["strong"] > 0 and r["unit"] is not None})]
    for kind, name in groups:
        hold = np.array([r["document"] == name if kind == "human_document" else r["document"] == "thesis" and r["unit"] == name for r in records])
        train = ~hold
        if not hold.any() or target[train].sum() == 0:
            continue
        fitted = fit(raw[train], target[train], weights[train], regularization)
        training_records = [r for r, ok in zip(records, train) if ok]
        train_scores = predict(raw[train], fitted)
        try:
            threshold, selection = choose_threshold(training_records, train_scores, target_extra_pp=.4, human_limit_pp=human_limit_pp)
        except ValueError as exc:
            checks.append({"kind": kind, "held_out": name, "training_records": int(train.sum()),
                           "test_records": int(hold.sum()), "status": "training_constraints_not_met",
                           "reason": str(exc)})
            continue
        test_records = [r for r, ok in zip(records, hold) if ok]
        test_scores = predict(raw[hold], fitted)
        checks.append({"kind": kind, "held_out": name, "status": "evaluated", "training_records": int(train.sum()), "test_records": int(hold.sum()),
                       "threshold_from_training_only": threshold,
                       "results": metrics(test_records, test_scores, threshold, threshold),
                       "warning": "Limited transfer diagnostic: standardization, coefficients and threshold exclude the held-out group, but hyperparameters were selected on these three references. Not an untouched test set."})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--regularization", type=float, default=.1)
    parser.add_argument("--human-control-weight", type=float, default=4)
    parser.add_argument("--human-limit-percent", type=float, default=0)
    args = parser.parse_args()
    raw_data = args.dataset.read_bytes(); dataset = json.loads(raw_data.decode("utf-8"))
    records = dataset["records"]
    if {r["document"] for r in records} != {"thesis", "wang-zhan", "zhu-yilong"}:
        raise ValueError("Final project profile requires both requested human theses and the current thesis")
    raw = np.array([r["raw"] for r in records]); targets = np.array([r["strong"] for r in records]); weights = np.array([r["chars"] for r in records], dtype=float)
    if raw.shape != (len(records), len(dataset["models"])) or not np.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all():
        raise ValueError("Raw model scores must be finite probabilities")
    if not np.isfinite(targets).all() or not ((targets >= 0) & (targets <= 1)).all() or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Reference fractions must be in [0,1] and character weights positive")
    if not np.isfinite(args.human_control_weight) or args.human_control_weight <= 0 or not 0 <= args.human_limit_percent <= 100:
        raise ValueError("Human control weight must be positive; rate cap must be in [0,100]")
    # The user-confirmed human labels receive more weight than the absence of
    # a significant mark in the commercial comparator. Runtime has no author
    # or document identity feature: this weighting applies only during fitting.
    weights *= np.array([1 if r["document"] == "thesis" else args.human_control_weight for r in records])
    fitted = fit(raw, targets, weights, args.regularization); scores = predict(raw, fitted)
    high, high_choice = choose_threshold(records, scores, target_extra_pp=.4, human_limit_pp=args.human_limit_percent)
    possible, possible_choice = choose_threshold(records, scores, target_extra_pp=1.1, human_limit_pp=args.human_limit_percent)
    if possible >= high:
        # Never quietly collapse the two tiers or relax the human cap.
        raise ValueError("The requested review tier is not separable under the human-control cap; inspect diagnostics")
    center, scale, parameters = fitted
    profile = {"schema_version": "aigc-score-fusion/1.0", "id": REFERENCE_POLICY,
        "models": dataset["models"], "revisions": dataset["revisions"],
        "center": center.tolist(), "scale": scale.tolist(), "coefficients": parameters[1:].tolist(), "intercept": float(parameters[0]),
        "clip_probability": 1e-5, "clip_standardized": 5, "segment_chars": 400, "min_segment_chars": 120,
        "high_threshold": high, "possible_threshold": possible,
        "fit": {"algorithm": "L2 logistic stacking on standardized clipped model logits", "regularization": args.regularization,
                "features": "only the ten raw model scores", "target": "CNKI significant span fraction per source passage; user-confirmed human controls have target 0",
                "character_weighted": True, "human_control_weight_multiplier": args.human_control_weight,
                "hyperparameter_selection": "Reference-set adaptation. Defaults selected after comparing L2 values 0.1/1 and human weights 1/2/4/8; prefer zero control flags, two distinct tiers and greater CNKI spatial overlap. No untouched test set.",
                "dataset_sha256": hashlib.sha256(raw_data).hexdigest(),
                "reference_documents": dataset["sources"], "threshold_selection": {"strong": high_choice, "possible_or_strong": possible_choice},
                "probability_calibrated": False, "independent_generalization_established": False},
        "scope": "Chinese thesis continuous prose extracted with the documented rules; reference-fitted project profile, not a CNKI implementation.",
        "limitations": ["Only two verified-human theses and one commercial comparator report.",
                        "These documents contributed to fitting; their rates are not out-of-sample accuracy.",
                        "CNKI labels are a comparator, not authorship ground truth.",
                        "Different subjects, model versions, segmentation, PDF line wrapping or preprocessing require new validation."]}
    diagnostics = {"schema_version": "aigc-fusion-evaluation/1.0", "dataset_sha256": profile["fit"]["dataset_sha256"],
        "reference_fit": metrics(records, scores, high, possible),
        "strong_threshold_selection": high_choice, "possible_threshold_selection": possible_choice,
        "blocked_transfer_checks": blocked_checks(records, raw, targets, weights, args.regularization, args.human_limit_percent),
        "score_records": [{"document": r["document"], "id": r["id"], "ensemble_score": float(score), "cnki_strong_fraction": r["strong"]} for r, score in zip(records, scores)]}
    args.profile.parent.mkdir(parents=True, exist_ok=True); args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    args.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"reference_fit": diagnostics["reference_fit"], "blocked_transfer_checks": diagnostics["blocked_transfer_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
