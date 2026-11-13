"""Fit and audit two score-only fusion heads, with reproducible nonlinear search.

This program reads only the numeric reference dataset. Character labels are
commercial-report comparators, not authorship truth. Reference adaptation and
blocked refits are deliberately reported separately from independent testing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aigc_detector.catalog import CHINESE_MODELS


POLICY = "thesis-cnki-20260929-v2"
EPS = 1e-5
LIMIT = 5.0
STRONG_REGULARIZATION = 0.1
STRONG_HUMAN_WEIGHT = 4.0
STRONG_EXTRA_PP = 0.4
REVIEW_EXTRA_PP = 0.4


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700, 700)))


def feature_terms(count, family):
    terms = [[i] for i in range(count)]
    if family in ("squares", "quadratic"):
        terms += [[i, i] for i in range(count)]
    if family == "quadratic":
        terms += [[i, j] for i in range(count) for j in range(i + 1, count)]
    if family not in ("linear", "squares", "quadratic"):
        raise ValueError("Unknown feature family")
    return terms


def standardizer(raw, weights):
    clipped = np.clip(raw, EPS, 1 - EPS)
    logits = np.log(clipped / (1 - clipped))
    center = np.average(logits, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((logits - center) ** 2, axis=0, weights=weights)), .5)
    return center, scale


def design(raw, center, scale, terms):
    clipped = np.clip(raw, EPS, 1 - EPS)
    z = np.clip((np.log(clipped / (1 - clipped)) - center) / scale, -LIMIT, LIMIT)
    return np.column_stack([np.ones(len(raw))] + [np.prod(z[:, term], axis=1) for term in terms])


def sample_weights(records, human_weight):
    return np.array([r["chars"] * (1 if r["document"] == "thesis" else human_weight) for r in records], dtype=float)


def fit_head(x, targets, weights, terms, regularization, nonlinear_penalty=10.0):
    """Convex logistic objective in a fixed nonlinear score feature basis.

    A damped Newton step uses the exact Hessian; Armijo backtracking guarantees
    objective decrease. No SciPy, randomness, text features, or model calls.
    """
    if regularization <= 0 or nonlinear_penalty < 1:
        raise ValueError("Invalid regularization")
    weights = weights / weights.mean()
    penalty = np.array([1e-6] + [regularization * (nonlinear_penalty if len(term) == 2 else 1) for term in terms])
    parameters = np.zeros(x.shape[1])
    mean = np.clip(np.average(targets, weights=weights), 1e-5, 1 - 1e-5)
    parameters[0] = np.log(mean / (1 - mean))

    def objective(values):
        logits = x @ values
        return float(np.sum(weights * (np.logaddexp(0, logits) - targets * logits)) + .5 * np.dot(penalty * values, values))

    trace = []
    converged = False
    for iteration in range(100):
        predicted = sigmoid(x @ parameters)
        gradient = x.T @ (weights * (predicted - targets)) + penalty * parameters
        max_gradient = float(np.max(np.abs(gradient)))
        before = objective(parameters)
        if max_gradient < 1e-7:
            converged = True
            trace.append({"iteration": iteration, "objective": before, "gradient_inf_norm": max_gradient, "step_size": 0.0})
            break
        hessian = x.T @ ((weights * predicted * (1 - predicted))[:, None] * x) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        rate = 1.0
        directional = float(gradient @ step)
        for _ in range(40):
            candidate = parameters - rate * step
            after = objective(candidate)
            if after <= before - 1e-4 * rate * directional + 1e-12:
                break
            rate *= .5
        else:
            raise ValueError("Optimizer line search failed")
        parameters = candidate
        trace.append({"iteration": iteration, "objective": after, "gradient_inf_norm_before_step": max_gradient, "step_size": rate})
    final_gradient = x.T @ (weights * (sigmoid(x @ parameters) - targets)) + penalty * parameters
    final_norm = float(np.max(np.abs(final_gradient)))
    converged = converged or final_norm < 1e-6
    if not converged or not np.isfinite(parameters).all():
        raise ValueError(f"Optimizer did not converge: gradient infinity norm {final_norm}")
    return parameters, {"method": "damped Newton with exact Hessian and Armijo backtracking", "converged": converged,
                        "iterations": len(trace), "final_objective": objective(parameters), "final_gradient_inf_norm": final_norm,
                        "parameter_l2_norm": float(np.linalg.norm(parameters[1:])), "trace": trace}


def threshold_midpoint(scores, threshold):
    lower = max((float(value) for value in scores if value < threshold), default=0.0)
    return float((lower + threshold) / 2)


def choose_threshold(records, scores, *, tier, strong_flags=None, extra_pp=0.4):
    """Apply a zero-control constraint and fit the requested exclusive tier rate."""
    own = np.array([r["document"] == "thesis" for r in records])
    if not own.any():
        raise ValueError("No comparator document in training")
    chars = np.array([r["chars"] for r in records], dtype=float)
    labels = np.array([r["strong" if tier == "strong" else "possible"] for r in records])
    baseline = float((chars[own] * labels[own]).sum() / chars[own].sum() * 100)
    if baseline <= 0:
        raise ValueError(f"No {tier} comparator labels in training")
    target = baseline + extra_pp
    strong_flags = np.zeros(len(records), dtype=bool) if strong_flags is None else np.asarray(strong_flags)
    if np.any(strong_flags & ~own):
        raise ValueError("Strong head already violates the zero-human-control constraint")
    best = None
    for threshold in sorted(set([1.0] + scores.tolist())):
        flags = scores >= threshold
        if np.any(flags & ~own):
            continue
        selected = flags if tier == "strong" else flags & ~strong_flags
        rate = float(chars[own & selected].sum() / chars[own].sum() * 100)
        if rate + 1e-9 < baseline:
            continue
        overlap = float((chars[own & selected] * labels[own & selected]).sum())
        objective = (abs(rate - target), -overlap)
        if best is None or objective < best[0]:
            best = objective, float(threshold), rate
    if best is None:
        raise ValueError(f"No {tier} threshold reaches the CNKI same-scope rate with zero human-control flags")
    threshold = threshold_midpoint(scores, best[1])
    humans = {r["document"] for r in records if r["document"] != "thesis"}
    return threshold, {"tier": tier, "baseline_percent": baseline, "requested_extra_percentage_points": extra_pp,
                       "target_percent": target, "achieved_percent": best[2], "absolute_target_error_pp": best[0][0],
                       "human_control_limit_percent": 0, "human_control_percentages": {name: 0.0 for name in sorted(humans)},
                       "selection": "Closest exclusive-tier character rate at or above the same-scope comparator baseline, with zero control flags; midpoint between adjacent training scores.",
                       "is_reference_fit_evidence": True, "is_independent_validation": False}


def metrics(records, strong_scores, review_scores, high, possible):
    result = {}
    for name in sorted({r["document"] for r in records}):
        indices = np.array([i for i, r in enumerate(records) if r["document"] == name])
        chars = np.array([records[i]["chars"] for i in indices], dtype=float)
        strong_labels = np.array([records[i]["strong"] for i in indices])
        possible_labels = np.array([records[i]["possible"] for i in indices])
        strong = strong_scores[indices] >= high
        review = (review_scores[indices] >= possible) & ~strong
        any_flag = strong | review
        strong_chars = int(chars[strong].sum())
        review_chars = int(chars[review].sum())
        total = int(chars.sum())
        row = {"characters": total, "segments": len(indices), "strong_characters": strong_chars,
               "possible_characters": review_chars, "strong_percent": strong_chars / total * 100,
               "possible_percent": review_chars / total * 100, "any_flag_percent": (strong_chars + review_chars) / total * 100}
        for label_name, label in (("strong", strong_labels), ("possible", possible_labels), ("any", strong_labels + possible_labels)):
            reference_chars = float((chars * label).sum())
            row[f"cnki_{label_name}_characters"] = round(reference_chars, 6)
            for flag_name, flag in (("strong", strong), ("possible", review), ("any", any_flag)):
                overlap = float((chars * label * flag).sum())
                row[f"cnki_{label_name}_overlap_in_{flag_name}_characters"] = round(overlap, 6)
                row[f"cnki_{label_name}_recall_in_{flag_name}"] = overlap / reference_chars if reference_chars else None
        row["cnki_any_precision_in_any"] = row["cnki_any_overlap_in_any_characters"] / (strong_chars + review_chars) if strong_chars + review_chars else None
        result[name] = row
    return result


def fit_strong(records, raw):
    weights = sample_weights(records, STRONG_HUMAN_WEIGHT)
    center, scale = standardizer(raw, weights)
    terms = feature_terms(raw.shape[1], "linear")
    x = design(raw, center, scale, terms)
    parameters, optimizer = fit_head(x, np.array([r["strong"] for r in records]), weights, terms, STRONG_REGULARIZATION)
    scores = sigmoid(x @ parameters)
    threshold, selection = choose_threshold(records, scores, tier="strong", extra_pp=STRONG_EXTRA_PP)
    return {"center": center, "scale": scale, "parameters": parameters, "scores": scores,
            "threshold": threshold, "selection": selection, "optimizer": optimizer}


def fit_review(records, raw, strong, settings):
    terms = feature_terms(raw.shape[1], settings["family"])
    x = design(raw, strong["center"], strong["scale"], terms)
    targets = np.array([r["strong"] + r["possible"] for r in records])
    parameters, optimizer = fit_head(x, targets, sample_weights(records, settings["human_weight"]), terms,
                                     settings["regularization"], settings["nonlinear_penalty"])
    scores = sigmoid(x @ parameters)
    result = {"terms": terms, "parameters": parameters, "scores": scores, "optimizer": optimizer}
    try:
        threshold, selection = choose_threshold(records, scores, tier="possible", strong_flags=strong["scores"] >= strong["threshold"], extra_pp=REVIEW_EXTRA_PP)
        result.update(threshold=threshold, selection=selection)
    except ValueError as exc:
        result["constraint_error"] = str(exc)
    return result


def constraint_frontier(records, scores, strong):
    own = np.array([r["document"] == "thesis" for r in records])
    chars = np.array([r["chars"] for r in records])
    control_max = float(np.max(scores[~own]))
    flags = (scores > control_max) & (strong["scores"] < strong["threshold"])
    return {"largest_control_score": control_max,
            "maximum_possible_only_percent_with_zero_controls": float(chars[own & flags].sum() / chars[own].sum() * 100),
            "required_possible_only_baseline_percent": float(sum(r["chars"] * r["possible"] for r in records if r["document"] == "thesis") / chars[own].sum() * 100)}


def holdout_groups(records):
    groups = [("human_document", name) for name in sorted({r["document"] for r in records if r["document"] != "thesis"})]
    groups += [("cnki_chapter", name) for name in sorted({name for r in records if r["document"] == "thesis" and r["strong"] + r["possible"] > 0 for name in record_units(r)})]
    return groups


def prepare_folds(records, raw):
    folds = []
    for kind, name in holdout_groups(records):
        # A segment can cross a chapter boundary. Its midpoint unit alone is
        # not sufficient for a blocked split: purge every mixed segment that
        # touches the held-out chapter, and evaluate only unmixed segments.
        memberships = [record_units(r) for r in records]
        excluded = np.array([r["document"] == name if kind == "human_document" else r["document"] == "thesis" and name in units for r, units in zip(records, memberships)])
        hold = excluded & np.array([kind == "human_document" or len(units) == 1 for units in memberships])
        train = ~excluded
        purged = excluded & ~hold
        train_records = [r for r, keep in zip(records, train) if keep]
        test_records = [r for r, keep in zip(records, hold) if keep]
        row = {"kind": kind, "held_out": name, "training_records": len(train_records), "test_records": len(test_records),
               "purged_mixed_segments": [{"document": r["document"], "id": r["id"], "units": units, "chars": r["chars"],
                                           "cnki_strong_characters": r["chars"] * r["strong"], "cnki_possible_characters": r["chars"] * r["possible"]}
                                          for r, units, purge in zip(records, memberships, purged) if purge]}
        try:
            strong = fit_strong(train_records, raw[train])
            folds.append((row, train, hold, train_records, test_records, strong))
        except ValueError as exc:
            row.update(status="training_strong_constraints_not_met", reason=str(exc))
            folds.append((row, train, hold, train_records, test_records, None))
    return folds


def record_units(record):
    memberships = record.get("unit_memberships")
    if memberships is None:
        return [str(record["unit"])]
    if isinstance(memberships, dict):
        return sorted(memberships)
    return sorted({str(item["unit"] if isinstance(item, dict) else item) for item in memberships})


def blocked_checks(raw, folds, settings, include_parameters=False):
    checks = []
    for base_row, train, hold, train_records, test_records, strong in folds:
        row = dict(base_row)
        if strong is None:
            checks.append(row)
            continue
        try:
            review = fit_review(train_records, raw[train], strong, settings)
            if "constraint_error" in review:
                raise ValueError(review["constraint_error"])
        except ValueError as exc:
            row.update(status="training_review_constraints_not_met", reason=str(exc))
            checks.append(row)
            continue
        test_strong = sigmoid(design(raw[hold], strong["center"], strong["scale"], feature_terms(raw.shape[1], "linear")) @ strong["parameters"])
        test_review = sigmoid(design(raw[hold], strong["center"], strong["scale"], review["terms"]) @ review["parameters"])
        row.update(status="evaluated", thresholds_from_training_only={"strong": strong["threshold"], "review": review["threshold"]},
                   training_threshold_selection={"strong": strong["selection"], "review": review["selection"]},
                   results=metrics(test_records, test_strong, test_review, strong["threshold"], review["threshold"]))
        if include_parameters:
            row["training_fit"] = {"center": strong["center"].tolist(), "scale": strong["scale"].tolist(), "feature_terms": review["terms"],
                                   "strong_parameters": strong["parameters"].tolist(), "review_parameters": review["parameters"].tolist(),
                                   "strong_optimizer": strong["optimizer"], "review_optimizer": review["optimizer"]}
        row["warning"] = "All transformations, head coefficients, and both thresholds exclude this held-out group. The hyperparameter search used these references and these diagnostics; this is not independent validation."
        checks.append(row)
    return checks


def summarize_checks(checks):
    human_flagged = human_total = overlap = reference = 0.0
    failed = 0
    for check in checks:
        if check["status"] != "evaluated":
            failed += 1
            continue
        for name, result in check["results"].items():
            if name == "thesis":
                overlap += result["cnki_possible_overlap_in_any_characters"]
                reference += result["cnki_possible_characters"]
            else:
                human_flagged += result["strong_characters"] + result["possible_characters"]
                human_total += result["characters"]
    return {"failed_folds": failed, "held_out_human_flag_percent": human_flagged / human_total * 100 if human_total else None,
            "held_out_possible_recall_in_any": overlap / reference if reference else None,
            "held_out_possible_reference_characters_evaluated": reference}


def operating_point_diagnostics(records, raw, strong, review):
    """Frozen-model threshold and input sensitivity; no refitting or claims of accuracy."""
    thresholds = {"strong": strong["threshold"], "review": review["threshold"]}
    head_scores = {"strong": strong["scores"], "review": review["scores"]}
    margins = {}
    for head, scores in head_scores.items():
        threshold = thresholds[head]
        lower = scores[scores < threshold]
        upper = scores[scores >= threshold]
        controls = {}
        for name in sorted({r["document"] for r in records if r["document"] != "thesis"}):
            mask = np.array([r["document"] == name for r in records])
            maximum = float(scores[mask].max())
            controls[name] = {"maximum_score": maximum, "threshold_margin": threshold - maximum}
        margins[head] = {"threshold": threshold, "largest_score_below": float(lower.max()) if len(lower) else None,
                         "smallest_score_above": float(upper.min()) if len(upper) else None, "controls": controls}
    unique_scores = sorted(set(review["scores"].tolist()))
    center_index = int(np.searchsorted(unique_scores, review["threshold"]))
    neighbors = []
    for index in range(max(0, center_index - 3), min(len(unique_scores), center_index + 4)):
        threshold = threshold_midpoint(review["scores"], unique_scores[index])
        neighbors.append({"threshold": threshold, "results": metrics(records, strong["scores"], review["scores"], strong["threshold"], threshold)})
    linear_terms = feature_terms(raw.shape[1], "linear")
    base_categories = np.where(strong["scores"] >= strong["threshold"], 2, np.where(review["scores"] >= review["threshold"], 1, 0))
    input_sensitivity = []
    for index in range(raw.shape[1]):
        for delta in (-1e-6, 1e-6):
            modified = raw.copy()
            modified[:, index] = np.clip(modified[:, index] + delta, 0, 1)
            high_scores = sigmoid(design(modified, strong["center"], strong["scale"], linear_terms) @ strong["parameters"])
            low_scores = sigmoid(design(modified, strong["center"], strong["scale"], review["terms"]) @ review["parameters"])
            categories = np.where(high_scores >= strong["threshold"], 2, np.where(low_scores >= review["threshold"], 1, 0))
            input_sensitivity.append({"model_index": index, "raw_score_additive_delta": delta, "changed_segment_categories": int((categories != base_categories).sum()),
                                      "results": metrics(records, high_scores, low_scores, strong["threshold"], review["threshold"])})
    ablations = []
    for index in range(raw.shape[1]):
        modified = raw.copy()
        modified[:, index] = sigmoid(strong["center"][index])
        high_scores = sigmoid(design(modified, strong["center"], strong["scale"], linear_terms) @ strong["parameters"])
        low_scores = sigmoid(design(modified, strong["center"], strong["scale"], review["terms"]) @ review["parameters"])
        ablations.append({"model_index": index, "replacement": "raw score whose logit equals the fitted center (standardized value zero)",
                          "results": metrics(records, high_scores, low_scores, strong["threshold"], review["threshold"])})
    return {"threshold_margins": margins, "review_threshold_neighbors": neighbors, "input_score_sensitivity": input_sensitivity,
            "model_score_ablations": ablations,
            "warning": "Frozen-reference numerical diagnostics only. Synthetic score perturbations and ablation are not independent documents or accuracy tests."}


def stability_comparison(records, raw, strong, folds):
    """Common random perturbations compare the high-coverage and steadier fits."""
    clipped = np.clip(raw, EPS, 1 - EPS)
    logits = np.log(clipped / (1 - clipped))
    linear_terms = feature_terms(raw.shape[1], "linear")
    own = np.array([r["document"] == "thesis" for r in records])
    chars = np.array([r["chars"] for r in records])
    rows = []
    for regularization in (1e-5, 1e-4):
        for human_weight in (1, 4):
            settings = {"family": "quadratic", "regularization": regularization, "human_weight": human_weight, "nonlinear_penalty": 10}
            review = fit_review(records, raw, strong, settings)
            if "constraint_error" in review:
                rows.append({"settings": settings, "status": "infeasible", "reason": review["constraint_error"]})
                continue
            row = {"settings": settings, "status": "evaluated", "reference": metrics(records, strong["scores"], review["scores"], strong["threshold"], review["threshold"]),
                   "parameter_norm": review["optimizer"]["parameter_l2_norm"], "blocked": summarize_checks(blocked_checks(raw, folds, settings)), "sensitivity": []}
            base_categories = np.where(strong["scores"] >= strong["threshold"], 2, np.where(review["scores"] >= review["threshold"], 1, 0))
            for sigma in (.01, .05, .1):
                rng = np.random.default_rng(20260929)
                human_rates, possible_rates, recalls, changes = [], [], [], []
                for _ in range(100):
                    altered = sigmoid(logits + rng.normal(0, sigma, size=logits.shape))
                    a = sigmoid(design(altered, strong["center"], strong["scale"], linear_terms) @ strong["parameters"])
                    b = sigmoid(design(altered, strong["center"], strong["scale"], review["terms"]) @ review["parameters"])
                    categories = np.where(a >= strong["threshold"], 2, np.where(b >= review["threshold"], 1, 0))
                    measured = metrics(records, a, b, strong["threshold"], review["threshold"])
                    human_rates.append(float(chars[(categories > 0) & ~own].sum() / chars[~own].sum() * 100))
                    possible_rates.append(measured["thesis"]["possible_percent"])
                    recalls.append(measured["thesis"]["cnki_possible_recall_in_any"])
                    changes.append(int((categories != base_categories).sum()))

                def summary(values):
                    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)), "p95": float(np.percentile(values, 95))}

                row["sensitivity"].append({"logit_noise_sigma": sigma, "control_flag_percent": summary(human_rates), "thesis_possible_percent": summary(possible_rates),
                                           "possible_recall": summary(recalls), "category_change_count": summary(changes)})
            row["review_control_margin"] = float(review["threshold"] - review["scores"][~own].max())
            rows.append(row)
    return {"seed": 20260929, "replicates": 100, "noise": "Independent Gaussian additions to each clipped raw model logit, followed by the same score-only runtime transform; frozen parameters and thresholds",
            "is_independent_validation": False, "candidates": rows}


def span_coverage(dataset, strong_scores, review_scores, high, possible):
    results = []
    for span in dataset["cnki_reference"]["aligned_spans"]:
        row = {key: span[key] for key in ("cnki_id", "category", "char_start", "char_end")}
        row.update(strong_overlap_characters=0, possible_overlap_characters=0)
        for record, strong, review in zip(dataset["records"], strong_scores, review_scores):
            if record["document"] != "thesis":
                continue
            overlap = max(0, min(record["char_end"], span["char_end"]) - max(record["char_start"], span["char_start"]))
            if strong >= high:
                row["strong_overlap_characters"] += overlap
            elif review >= possible:
                row["possible_overlap_characters"] += overlap
        row["any_overlap_characters"] = row["strong_overlap_characters"] + row["possible_overlap_characters"]
        row["any_hit"] = row["any_overlap_characters"] > 0
        results.append(row)
    return results


def old_baseline(dataset, raw):
    path = Path(__file__).resolve().parents[1] / "aigc_detector/profiles/thesis-cnki-20260929-v1.json"
    data = path.read_bytes()
    profile = json.loads(data.decode("utf-8"))
    if profile["models"] != dataset["models"] or profile["revisions"] != dataset["revisions"]:
        raise ValueError("The v1 baseline must use identical model order and revisions")
    parameters = np.array([profile["intercept"]] + profile["coefficients"])
    scores = sigmoid(design(raw, np.array(profile["center"]), np.array(profile["scale"]), feature_terms(10, "linear")) @ parameters)
    result = {"policy": profile["id"], "profile_sha256": hashlib.sha256(data).hexdigest(),
              "reference_fit": metrics(dataset["records"], scores, scores, profile["high_threshold"], profile["possible_threshold"]),
              "span_coverage": span_coverage(dataset, scores, scores, profile["high_threshold"], profile["possible_threshold"])}
    strong = {"scores": scores, "threshold": profile["high_threshold"]}
    result["threshold_only_review_frontier"] = constraint_frontier(dataset["records"], scores, strong)
    return result


def validate_dataset(dataset):
    if dataset.get("schema_version") != "aigc-reference-scores/2.0":
        raise ValueError("The audited v2 dataset with chapter memberships is required")
    if dataset.get("segmentation") != {"max_characters": 400, "minimum_target_characters": 120}:
        raise ValueError("The fitted profile requires 400/120 segmentation")
    records = dataset["records"]
    if {r["document"] for r in records} != {"thesis", "wang-zhan", "zhu-yilong"}:
        raise ValueError("The current comparator and both requested human controls are required")
    models = dataset["models"]
    if len(models) != 10 or len(set(models)) != 10 or set(models) != set(CHINESE_MODELS) or set(dataset["revisions"]) != set(models):
        raise ValueError("Exactly ten unique revision-pinned models are required")
    if any(not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision) for revision in dataset["revisions"].values()):
        raise ValueError("Every revision must be a complete lowercase commit hash")
    if any(any(isinstance(value, bool) for value in r["raw"]) for r in records):
        raise ValueError("Boolean model scores are invalid")
    raw = np.array([r["raw"] for r in records], dtype=float)
    if raw.shape != (len(records), 10) or not np.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all():
        raise ValueError("Raw scores must be a finite n-by-ten matrix in [0,1]")
    seen = set()
    ends = {name: 0 for name in ("thesis", "wang-zhan", "zhu-yilong")}
    for record in records:
        key = record["document"], record["id"]
        if key in seen:
            raise ValueError("Duplicate document/segment identifier")
        seen.add(key)
        if isinstance(record["chars"], bool) or not isinstance(record["chars"], int) or record["chars"] <= 0:
            raise ValueError("Positive integer character counts are required")
        start, end = record["char_start"], record["char_end"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) or start != ends[record["document"]] or end - start != record["chars"]:
            raise ValueError("Document segment offsets must be contiguous and match character counts")
        ends[record["document"]] = end
        a, b = record["strong"], record["possible"]
        if isinstance(a, bool) or isinstance(b, bool) or not np.isfinite([a, b]).all() or min(a, b) < 0 or a + b > 1 + 1e-9:
            raise ValueError("Strong and possible must be disjoint fractions in [0,1]")
        if record["document"] != "thesis" and a + b != 0:
            raise ValueError("Human-control labels must be zero")
        memberships = record.get("unit_memberships")
        if not isinstance(memberships, list) or not memberships or any(not isinstance(item, dict) or not isinstance(item.get("unit"), str) or not item["unit"] or isinstance(item.get("characters"), bool) or not isinstance(item.get("characters"), int) or item["characters"] <= 0 for item in memberships):
            raise ValueError("Audited chapter memberships with positive overlap character counts are required")
        if len({item["unit"] for item in memberships}) != len(memberships) or sum(item["characters"] for item in memberships) > record["chars"]:
            raise ValueError("Chapter memberships must be unique and cannot exceed segment characters")
        for label in ("strong", "possible"):
            if abs(record[f"{label}_characters"] - record["chars"] * record[label]) > 1e-6:
                raise ValueError("Reference character counts disagree with their fractions")
    if set(dataset["sources"]) != set(ends) or any(dataset["sources"][name]["input_characters"] != end for name, end in ends.items()):
        raise ValueError("Document character totals disagree with source manifests")
    return records, raw


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=["linear", "squares", "quadratic"], default=["linear", "squares", "quadratic"])
    parser.add_argument("--regularizations", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, .001, .01, .1, 1, 10, 100])
    parser.add_argument("--human-weights", nargs="+", type=float, default=[1, 4, 8, 16, 32])
    parser.add_argument("--nonlinear-penalty", type=float, default=10)
    parser.add_argument("--minimum-selected-regularization", type=float, default=1e-4,
                        help="Reference-diagnosis choice: exclude weaker fits after observing their larger sensitivity; still record their full trials.")
    parser.add_argument("--stability-comparison", type=Path,
                        help="Write the reproducible fixed-seed nonlinear candidate perturbation comparison.")
    parser.add_argument("--skip-grid-blocked", action="store_true", help="Exploration only; final selected settings still receive blocked refits.")
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.regularizations + args.human_weights + [args.minimum_selected_regularization]) or not np.isfinite(args.nonlinear_penalty) or args.nonlinear_penalty < 1:
        raise ValueError("Search regularizations and weights must be finite and positive; nonlinear penalty must be >=1")
    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    records, raw = validate_dataset(dataset)
    strong = fit_strong(records, raw)
    folds = prepare_folds(records, raw)
    trials = []
    candidates = []
    for family in args.families:
        for regularization in args.regularizations:
            for human_weight in args.human_weights:
                settings = {"family": family, "regularization": regularization, "human_weight": human_weight, "nonlinear_penalty": args.nonlinear_penalty}
                row = {"settings": settings}
                try:
                    review = fit_review(records, raw, strong, settings)
                    row["optimizer"] = {k: v for k, v in review["optimizer"].items() if k != "trace"}
                    row["control_constraint_frontier"] = constraint_frontier(records, review["scores"], strong)
                    if "constraint_error" in review:
                        raise ValueError(review["constraint_error"])
                    row.update(status="feasible", reference_fit=metrics(records, strong["scores"], review["scores"], strong["threshold"], review["threshold"]),
                               threshold_selection=review["selection"], optimizer={k: v for k, v in review["optimizer"].items() if k != "trace"})
                    if not args.skip_grid_blocked:
                        row["blocked_checks"] = blocked_checks(raw, folds, settings)
                        row["blocked_summary"] = summarize_checks(row["blocked_checks"])
                    # Spatial overlap is required alongside rate alignment. The
                    # reference comparison drives adaptation; grouped refits
                    # break ties and expose transfer limitations explicitly.
                    own = row["reference_fit"]["thesis"]
                    diagnostic = row.get("blocked_summary", {})
                    key = (-own["cnki_possible_recall_in_any"],
                           -own["cnki_any_precision_in_any"],
                           review["optimizer"]["parameter_l2_norm"],
                           diagnostic.get("failed_folds", 0),
                           diagnostic.get("held_out_human_flag_percent") or 0,
                           -(diagnostic.get("held_out_possible_recall_in_any") or 0),
                           review["selection"]["absolute_target_error_pp"], len(review["terms"]), -regularization, human_weight)
                    row["selection_key"] = list(key)
                    row["selection_eligible"] = regularization >= args.minimum_selected_regularization
                    if row["selection_eligible"]:
                        candidates.append((key, settings, review, row))
                    else:
                        row["selection_exclusion_reason"] = "Below the reference-diagnosis minimum regularization, selected after the fixed-seed sensitivity comparison; not a preregistered validation rule."
                except ValueError as exc:
                    row.update(status="infeasible", reason=str(exc))
                trials.append(row)
                print(json.dumps({"settings": settings, "status": row["status"], "possible_percent": row.get("reference_fit", {}).get("thesis", {}).get("possible_percent"),
                                  "possible_recall": row.get("reference_fit", {}).get("thesis", {}).get("cnki_possible_recall_in_any"), "blocked": row.get("blocked_summary")}), flush=True)
    if not candidates:
        write_json(args.search, {"trials": trials, "status": "no_feasible_candidate"})
        raise ValueError("No feasible review fit; no profile has been written")
    _, settings, review, chosen_row = min(candidates, key=lambda candidate: candidate[0])
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    # A shared feature basis makes runtime simple; the preserved linear strong
    # head gets explicit zero coefficients for every nonlinear review term.
    strong_coefficients = strong["parameters"][1:].tolist() + [0.0] * (len(review["terms"]) - raw.shape[1])
    profile = {"schema_version": "aigc-score-fusion/2.0", "id": POLICY,
               "models": dataset["models"], "revisions": dataset["revisions"],
               "center": strong["center"].tolist(), "scale": strong["scale"].tolist(), "feature_terms": review["terms"],
               "heads": {"strong": {"intercept": float(strong["parameters"][0]), "coefficients": strong_coefficients},
                         "review": {"intercept": float(review["parameters"][0]), "coefficients": review["parameters"][1:].tolist()}},
               "clip_probability": EPS, "clip_standardized": LIMIT, "segment_chars": 400, "min_segment_chars": 120,
               "high_threshold": strong["threshold"], "possible_threshold": review["threshold"],
               "decision_rule": "strong if strong_score >= high_threshold; otherwise possible if review_score >= possible_threshold; otherwise other",
               "fit": {"algorithm": "Two L2-regularized logistic heads on standardized clipped score logits and optional quadratic terms; exact-Hessian damped Newton optimization",
                       "features": "only the ten raw model scores; no text, author, document identity, or metadata at runtime",
                       "strong_target": "CNKI significant span fraction per passage",
                       "review_target": "Union of CNKI significant and possible span fractions per passage",
                       "character_weighted": True, "standardizer_human_control_weight_multiplier": STRONG_HUMAN_WEIGHT,
                       "strong_settings": {"family": "linear", "regularization": STRONG_REGULARIZATION, "human_weight": STRONG_HUMAN_WEIGHT},
                       "review_settings": settings, "intercept_regularization": 1e-6,
                       "stability_selection": {"minimum_selected_regularization": args.minimum_selected_regularization,
                                               "basis": "Comparing fixed-seed Gaussian clipped-logit perturbations showed the higher-coverage 1e-5 fit had larger control flag rates and output drift than 1e-4. The floor was set after examining these reference diagnostics, not before an independent test."},
                       "weight_normalization": "Each head's character weights, including the control multiplier, are divided by their arithmetic mean.",
                       "nonlinear_penalty": "Degree-two coefficients receive review_settings.nonlinear_penalty times the linear L2 penalty.",
                       "dataset_sha256": digest, "reference_documents": dataset["sources"],
                       "fit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "threshold_selection": {"strong": strong["selection"], "possible_only": review["selection"]},
                       "hyperparameter_selection": "Deterministic fixed grid. Require zero reference-control flags and rates at or above same-scope CNKI. Exclude regularization below the declared stability-selection floor, which was set after examining reference perturbations. Maximize reference possible-span coverage; tie-break by reference marked-span precision, smaller coefficient norm, failed grouped fits, held-out human flags, grouped possible recall, rate target error, fewer terms, stronger regularization. All selection evidence uses these references, not independent test documents.",
                       "probability_calibrated": False, "independent_generalization_established": False},
               "scope": "Chinese thesis continuous prose, all ten pinned model revisions, 400/120 character segmentation; project reference adaptation.",
               "limitations": ["Three reference documents, only one with commercial comparator span labels.",
                               "Fitted rates and blocked-refit diagnostics do not establish independent accuracy or generalization.",
                               "Matching a document-wide flag rate does not imply matching marked locations.",
                               "The two screening heads are not calibrated probabilities and their thresholds are not interchangeable.",
                               "CNKI marks are a comparator, not authorship ground truth.",
                               "Preprocessing, model revision, subject, and document changes require new validation."]}
    diagnostics = {"schema_version": "aigc-fusion-evaluation/2.0", "dataset_sha256": digest, "selected_review_settings": settings,
                   "reference_fit": chosen_row["reference_fit"], "strong_threshold_selection": strong["selection"], "possible_threshold_selection": review["selection"],
                   "optimization": {"strong": strong["optimizer"], "review": review["optimizer"]},
                   "blocked_transfer_checks": blocked_checks(raw, folds, settings, include_parameters=True),
                   "operating_point_diagnostics": operating_point_diagnostics(records, raw, strong, review),
                   "v1_baseline": old_baseline(dataset, raw),
                   "span_coverage": span_coverage(dataset, strong["scores"], review["scores"], strong["threshold"], review["threshold"]),
                   "is_independent_validation": False,
                   "score_records": [{"document": r["document"], "id": r["id"], "strong_score": float(a), "review_score": float(b),
                                      "cnki_strong_fraction": r["strong"], "cnki_possible_fraction": r["possible"]} for r, a, b in zip(records, strong["scores"], review["scores"])]}
    diagnostics["blocked_summary"] = summarize_checks(diagnostics["blocked_transfer_checks"])
    search = {"schema_version": "aigc-fusion-search/2.0", "dataset_sha256": digest, "deterministic": True, "random_seed_required": False,
              "grid": {"families": args.families, "regularizations": args.regularizations, "human_control_weights": args.human_weights, "nonlinear_penalty": args.nonlinear_penalty},
              "stability_selection": profile["fit"]["stability_selection"],
              "strong_settings": profile["fit"]["strong_settings"], "selection_rule": profile["fit"]["hyperparameter_selection"],
              "grid_blocked_refits_computed": not args.skip_grid_blocked, "selected_review_settings": settings, "trial_count": len(trials), "trials": trials,
              "is_reference_adaptation": True, "is_independent_validation": False}
    write_json(args.profile, profile)
    diagnostics["profile_sha256"] = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    search["profile_sha256"] = diagnostics["profile_sha256"]
    write_json(args.diagnostics, diagnostics)
    write_json(args.search, search)
    if args.stability_comparison:
        stability = stability_comparison(records, raw, strong, folds)
        stability["dataset_sha256"] = digest
        stability["profile_sha256"] = diagnostics["profile_sha256"]
        write_json(args.stability_comparison, stability)
    print(json.dumps({"selected": settings, "reference_fit": diagnostics["reference_fit"], "blocked_summary": diagnostics["blocked_summary"], "profile_sha256": diagnostics["profile_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
