"""Versioned, score-only reference fusion. No text, filename or author features.

The sigmoid is a screening score fitted to a small project reference set, not
an estimated probability of AI authorship. Every raw classifier score remains
unchanged in annotation output.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

LEGACY_POLICY = "sensitive-any-model-v1"
REFERENCE_POLICY = "thesis-cnki-20260929-v1"
POLICIES = (REFERENCE_POLICY, LEGACY_POLICY)
PROFILE_PATH = Path(__file__).with_name("profiles") / f"{REFERENCE_POLICY}.json"


def load_profile():
    raw = PROFILE_PATH.read_bytes()
    profile = json.loads(raw.decode("utf-8"))
    validate_profile(profile)
    return profile, hashlib.sha256(raw).hexdigest()


def validate_profile(profile):
    if profile.get("schema_version") != "aigc-score-fusion/1.0" or profile.get("id") != REFERENCE_POLICY:
        raise ValueError("Unsupported fusion profile")
    models = profile.get("models", [])
    if not models or any(not isinstance(m, str) for m in models) or len(set(models)) != len(models):
        raise ValueError("Fusion models must be unique and non-empty")
    for key in ("center", "scale", "coefficients"):
        values = profile.get(key, [])
        if len(values) != len(models) or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError(f"Invalid fusion {key}")
    if any(x <= 0 for x in profile["scale"]):
        raise ValueError("Fusion scales must be positive")
    for key in ("intercept", "clip_standardized", "clip_probability"):
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid fusion {key}")
    if not 0 < profile["clip_probability"] < .5 or profile["clip_standardized"] <= 0:
        raise ValueError("Invalid fusion clipping bounds")
    if set(profile.get("revisions", {})) != set(models):
        raise ValueError("Fusion profile must pin each model revision")
    from .annotation import validate_policy
    validate_policy(profile.get("high_threshold"), profile.get("possible_threshold"),
                    profile.get("segment_chars"), profile.get("min_segment_chars"))


def policy_settings(policy, selected, segment_chars, min_segment_chars, high, possible):
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy. Choose one of {POLICIES}")
    if policy == LEGACY_POLICY:
        return None, {"id": policy, "high_threshold": .90 if high is None else high,
                      "possible_threshold": .70 if possible is None else possible,
                      "raw_score_high_threshold": .90 if high is None else high,
                      "raw_score_possible_threshold": .70 if possible is None else possible,
                      "combination": "any_model", "calibrated": False, "reference_fitted": False,
                      "description": "Any selected model may flag a passage."}
    profile, digest = load_profile()
    if set(selected) != set(profile["models"]):
        raise ValueError(f"{policy} requires its complete {len(profile['models'])}-model roster. Use {LEGACY_POLICY} explicitly for a subset.")
    if (segment_chars, min_segment_chars) != (profile["segment_chars"], profile["min_segment_chars"]):
        raise ValueError(f"{policy} requires segment_chars={profile['segment_chars']} and min_segment_chars={profile['min_segment_chars']}; different segmentation needs separate validation.")
    settings = {"id": policy, "high_threshold": profile["high_threshold"] if high is None else high,
                "possible_threshold": profile["possible_threshold"] if possible is None else possible,
                "raw_score_high_threshold": .90, "raw_score_possible_threshold": .70,
                "combination": "regularized_logit_stacking", "calibrated": False,
                "reference_fitted": True, "profile_sha256": digest,
                "profile": copy.deepcopy(profile), "thresholds_overridden": high is not None or possible is not None,
                "description": "Joint score from all ten classifiers, fitted to project human controls and CNKI marked spans; not an authorship probability."}
    return profile, settings


def fused_score(scores, profile):
    """Return the joint score and auditable signed logit contributions."""
    missing = [m for m in profile["models"] if scores.get(m) is None]
    if missing:
        return {"score": None, "logit": None, "contributions": {}, "missing_models": missing}
    contributions = {}; total = profile["intercept"]
    eps, limit = profile["clip_probability"], profile["clip_standardized"]
    for model, center, scale, coefficient in zip(profile["models"], profile["center"], profile["scale"], profile["coefficients"]):
        value = scores[model]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid raw score for {model}")
        clipped = min(1-eps, max(eps, value))
        standardized = max(-limit, min(limit, (math.log(clipped/(1-clipped)) - center)/scale))
        contribution = coefficient * standardized
        contributions[model] = {"raw_ai_score": value, "standardized_logit": standardized,
                                "coefficient": coefficient, "logit_contribution": contribution}
        total += contribution
    score = 1 / (1 + math.exp(-max(-700, min(700, total))))
    return {"score": score, "logit": total, "contributions": contributions, "missing_models": []}


def decide_fusion(scores, profile, high, possible):
    from .annotation import LABELS, decide, score_category
    result = fused_score(scores, profile)
    raw = decide(scores, .9, .7)
    category = score_category(result["score"], high, possible)
    status = "scored" if result["score"] is not None else "no_scorable_tokens" if not any(s is not None for s in scores.values()) else "incomplete_model_scores"
    return {**raw, "category": category, "label": LABELS[category],
            "triggered_models": list(profile["models"]) if category != "other" else [],
            "ensemble_score": result["score"], "ensemble_logit": result["logit"],
            "contributions": result["contributions"], "missing_models": result["missing_models"],
            "evaluation_status": status,
            "decision_basis": "joint_model_scores_only"}


def reapply_policy(report, *, policy=REFERENCE_POLICY, high_threshold=None, possible_threshold=None):
    """Recombine cached model scores without inference or source-dependent rules."""
    from datetime import datetime, timezone
    from . import __version__
    from .annotation import LABELS, decide, score_category, summarize, validate_policy
    if report.get("status") != "complete" or not report.get("segments"):
        raise ValueError("A complete annotation report with segments is required")
    selected = [m["id"] for m in report["models"]]
    segmentation = report["segmentation"]
    profile, settings = policy_settings(policy, selected, segmentation["max_characters"],
                                       segmentation["minimum_target_characters"], high_threshold, possible_threshold)
    high, possible = settings["high_threshold"], settings["possible_threshold"]
    validate_policy(high, possible, segmentation["max_characters"], segmentation["minimum_target_characters"])
    if profile and any(m["revision"] != profile["revisions"][m["id"]] for m in report["models"]):
        raise ValueError("Cached model revisions differ from the fusion profile")
    result = copy.deepcopy(report)
    result["schema_version"] = "aigc-annotation/1.1"
    result["software_version"] = __version__
    result["recombined_at"] = datetime.now(timezone.utc).isoformat()
    result["recombined_from"] = {"policy": report["policy"]["id"], "input_sha256": report["input"]["sha256"]}
    result["policy"] = {**settings, "labels": LABELS}
    for segment in result["segments"]:
        if set(segment["model_scores"]) != set(selected):
            raise ValueError("Segment model scores do not match report roster")
        scores = {m: segment["model_scores"][m]["ai_score"] for m in selected}
        segment["decision"] = decide_fusion(scores, profile, high, possible) if profile else decide(scores, high, possible)
        for model in segment["model_scores"].values():
            model["category"] = score_category(model["ai_score"], settings["raw_score_high_threshold"], settings["raw_score_possible_threshold"])
        if segment["decision"]["evaluation_status"] != "scored" and segment["decision"]["evaluation_status"] not in segment["warnings"]:
            segment["warnings"].append(segment["decision"]["evaluation_status"])
    result["summary"] = summarize(result["segments"])
    return result
