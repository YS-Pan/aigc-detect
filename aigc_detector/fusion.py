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
LINEAR_POLICY = "thesis-cnki-20260929-v1"
REFERENCE_POLICY = "thesis-cnki-20260929-v2"
ENGLISH_POLICY = "english-academic-20261003-v1"
ENGLISH_EXPANDED_POLICY = "english-academic-20261003-v2"
ENGLISH_NONLINEAR_POLICY = "english-academic-20261004-v3"
ENGLISH_PROFILES = (ENGLISH_POLICY, ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY)
POLICIES = (REFERENCE_POLICY, LINEAR_POLICY, *ENGLISH_PROFILES, LEGACY_POLICY)
PROFILE_PATH = Path(__file__).with_name("profiles") / f"{REFERENCE_POLICY}.json"


def load_profile(policy=REFERENCE_POLICY):
    if policy not in (REFERENCE_POLICY, LINEAR_POLICY, *ENGLISH_PROFILES):
        raise ValueError("This policy has no fusion profile")
    path = PROFILE_PATH if policy == REFERENCE_POLICY else PROFILE_PATH.with_name(f"{policy}.json")
    raw = path.read_bytes()
    profile = json.loads(raw.decode("utf-8"))
    validate_profile(profile)
    if profile["id"] != policy:
        raise ValueError("Fusion profile ID differs from the requested policy")
    return profile, hashlib.sha256(raw).hexdigest()


def resolve_policy(policy, language):
    """Choose the fitted policy by explicit language, retaining legacy opt-in."""
    selected = policy if policy is not None else ENGLISH_EXPANDED_POLICY if language == "en" else REFERENCE_POLICY
    if selected in ENGLISH_PROFILES and language != "en":
        raise ValueError("English fusion requires language=en")
    if selected in (REFERENCE_POLICY, LINEAR_POLICY) and language != "zh":
        raise ValueError("Chinese fusion requires language=zh; use the English profile for English annotation")
    return selected


def profile_segmentation(profile):
    return "english-natural-boundaries-v1" if profile.get("language") == "en" else "natural-boundaries-v1"


def dual_head(profile):
    return profile["schema_version"] in ("aigc-score-fusion/2.0", "aigc-score-fusion/3.0", "aigc-score-fusion/4.0")


def expanded_profile(profile):
    return profile["schema_version"] in ("aigc-score-fusion/3.0", "aigc-score-fusion/4.0")


def nonlinear_profile(profile):
    return profile["schema_version"] == "aigc-score-fusion/4.0"


def signal_revision(profile, signal):
    """Pin statistical scores to their assets and arithmetic/window contract."""
    if signal in profile["revisions"]:
        return profile["revisions"][signal]
    contract = {"backend_id": signal, "assets": profile["statistical_assets"][signal],
                "inference": profile["statistical_inference"]}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def segment_raw_scores(segment):
    return {m: v["raw_score"] if "raw_score" in v else v["ai_score"] for m, v in segment["model_scores"].items()}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_vector(values, length, name):
    if not isinstance(values, list) or len(values) != length or not all(_finite(x) for x in values):
        raise ValueError(f"Invalid fusion {name}")


def validate_profile(profile):
    schemas = {LINEAR_POLICY: "aigc-score-fusion/1.0", REFERENCE_POLICY: "aigc-score-fusion/2.0",
               ENGLISH_POLICY: "aigc-score-fusion/2.0", ENGLISH_EXPANDED_POLICY: "aigc-score-fusion/3.0",
               ENGLISH_NONLINEAR_POLICY: "aigc-score-fusion/4.0"}
    if not isinstance(profile, dict) or profile.get("id") not in schemas or profile.get("schema_version") != schemas[profile["id"]]:
        raise ValueError("Unsupported fusion profile")
    from .catalog import CHINESE_MODELS, ENGLISH_MODELS, EXTRA_ENGLISH_MODELS, STATISTICAL_BACKENDS
    english = profile["id"] in ENGLISH_PROFILES
    expanded = expanded_profile(profile)
    if english and (profile.get("language") != "en" or profile.get("segmentation_id") != "english-natural-boundaries-v1"):
        raise ValueError("English profile must pin its language and segmentation")
    roster = ENGLISH_MODELS if english else CHINESE_MODELS
    models = profile.get("models", [])
    if expanded:
        allowed = set(ENGLISH_MODELS + EXTRA_ENGLISH_MODELS) | set(STATISTICAL_BACKENDS)
        if (not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models)
                or len(set(models)) != len(models) or not set(models) <= allowed):
            raise ValueError("Expanded English fusion requires a pinned unique roster of installed English signals")
        classifiers = set(models) - STATISTICAL_BACKENDS.keys()
        statistical = set(models) & STATISTICAL_BACKENDS.keys()
        assets = profile.get("statistical_assets")
        if not isinstance(assets, dict) or set(assets) != statistical:
            raise ValueError("Expanded fusion must pin statistical asset revisions")
        for mid in statistical:
            if (not isinstance(assets[mid], dict) or set(assets[mid]) != set(STATISTICAL_BACKENDS[mid])
                    or any(not isinstance(r, str) or not r for r in assets[mid].values())):
                raise ValueError("Statistical backend asset contract mismatch")
        expected = {m: "logit" if m in classifiers else "negative_identity" if m == "binoculars-falcon" else "identity" for m in models}
        if profile.get("score_transforms") != expected:
            raise ValueError("Raw score transforms must match signal type and direction")
        if profile.get("statistical_inference") != {"max_tokens": 512, "overlap_tokens": 64, "dtype": "bfloat16",
                                                   "aggregation": "inverse_coverage_token_statistics_then_formula"}:
            raise ValueError("Unsupported statistical inference contract")
    elif not isinstance(models, list) or any(not isinstance(m, str) for m in models) or len(models) != len(roster) or set(models) != set(roster):
        raise ValueError("Fusion models must contain the complete four-model English roster once" if english else
                         "Fusion models must contain the complete ten-model roster once")
    for key in ("center", "scale"):
        _validate_vector(profile.get(key), len(models), key)
    if any(x <= 0 for x in profile["scale"]):
        raise ValueError("Fusion scales must be positive")
    for key in ("clip_standardized", "clip_probability"):
        if not _finite(profile.get(key)):
            raise ValueError(f"Invalid fusion {key}")
    if not 0 < profile["clip_probability"] < .5 or profile["clip_standardized"] <= 0:
        raise ValueError("Invalid fusion clipping bounds")
    revisions = profile.get("revisions")
    if not isinstance(revisions, dict) or set(revisions) != (classifiers if expanded else set(models)) or any(not isinstance(x, str) or not x for x in revisions.values()):
        raise ValueError("Fusion profile must pin each model revision")
    dual = dual_head(profile)
    if nonlinear_profile(profile):
        from .nonlinear import validate_head
        heads = profile.get("heads")
        if not isinstance(heads, dict) or set(heads) != {"strong", "review"}:
            raise ValueError("Fusion requires strong and review heads")
        for head in heads.values():
            validate_head(head, len(models))
        if heads["strong"] != heads["review"]:
            raise ValueError("This nonlinear profile requires identical functions with separate thresholds")
    elif dual:
        terms = profile.get("feature_terms")
        if not isinstance(terms, list) or not terms or len(terms) > len(models)*(len(models)+3)//2:
            raise ValueError("Invalid fusion feature terms")
        seen = set()
        for term in terms:
            if (not isinstance(term, list) or len(term) not in (1, 2)
                    or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(models) for i in term)
                    or term != sorted(term) or tuple(term) in seen):
                raise ValueError("Invalid or duplicate fusion feature term")
            seen.add(tuple(term))
        if not all((i,) in seen for i in range(len(models))):
            raise ValueError("Fusion feature terms must include every model's linear term")
        heads = profile.get("heads")
        if not isinstance(heads, dict) or set(heads) != {"strong", "review"}:
            raise ValueError("Fusion requires strong and review heads")
        for name, head in heads.items():
            if not isinstance(head, dict) or not _finite(head.get("intercept")):
                raise ValueError(f"Invalid fusion {name} intercept")
            _validate_vector(head.get("coefficients"), len(terms), f"{name} coefficients")
    else:
        if not _finite(profile.get("intercept")):
            raise ValueError("Invalid fusion intercept")
        _validate_vector(profile.get("coefficients"), len(models), "coefficients")
    from .annotation import validate_policy
    validate_policy(profile.get("high_threshold"), profile.get("possible_threshold"),
                    profile.get("segment_chars"), profile.get("min_segment_chars"), independent_thresholds=dual)
    if not english and (profile["segment_chars"], profile["min_segment_chars"]) != (400, 120):
        raise ValueError("These reference profiles require 400/120 character segmentation")


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
    profile, digest = load_profile(policy)
    if len(selected) != len(profile["models"]) or len(set(selected)) != len(selected) or set(selected) != set(profile["models"]):
        raise ValueError(f"{policy} requires its complete {len(profile['models'])}-model roster. Use {LEGACY_POLICY} explicitly for a subset.")
    if (segment_chars, min_segment_chars) != (profile["segment_chars"], profile["min_segment_chars"]):
        raise ValueError(f"{policy} requires segment_chars={profile['segment_chars']} and min_segment_chars={profile['min_segment_chars']}; different segmentation needs separate validation.")
    dual = dual_head(profile)
    settings = {"id": policy, "high_threshold": profile["high_threshold"] if high is None else high,
                "possible_threshold": profile["possible_threshold"] if possible is None else possible,
                "raw_score_high_threshold": .90, "raw_score_possible_threshold": .70,
                "combination": "dual_head_polynomial_logit_stacking" if dual else "regularized_logit_stacking",
                "threshold_mode": "separate_heads" if dual else "single_score", "calibrated": False,
                "reference_fitted": True, "profile_sha256": digest,
                "profile": copy.deepcopy(profile), "thresholds_overridden": high is not None or possible is not None,
                "description": ("Four raw English classifier scores; paper-grouped academic reference fitting and conservative human-control thresholds. Not authorship probabilities."
                                if policy == ENGLISH_POLICY else
                                "Scores from all ten classifiers, fitted to project human controls and CNKI marked spans; strong and review heads use separate thresholds in v2. Not authorship probabilities.")}
    if policy == ENGLISH_POLICY:
        settings["validation_warning"] = "Low sensitivity: none of 12 held-out generated paragraphs was detected. An unflagged result cannot exclude AIGC."
    elif policy == ENGLISH_EXPANDED_POLICY:
        settings["description"] = "Expanded English raw scores and optional LM statistics; complete pinned roster, score transforms and paper-grouped reference fitting. Screening scores are not authorship probabilities."
        settings["validation_warning"] = "Low sensitivity: repeat evaluation detected 3 of 12 isolated generated paragraphs and covered 20.38% of generated characters in mixed documents. Previously evaluated papers were reused; this is not new independent validation. An unflagged result cannot exclude AIGC."
    elif policy == ENGLISH_NONLINEAR_POLICY:
        kind = profile["heads"]["strong"]["kind"]
        settings["combination"] = f"nonlinear_{kind}_score_stacking"
        settings["description"] = "Optimized nonlinear fusion of pinned raw English scores; paper-grouped reference fitting and conservative human-control thresholds. Both tiers share a function. Screening scores are not authorship probabilities."
        settings["contribution_accounting"] = ("Analytic integrated gradients from standardized zero; exact allocation, not causal importance."
                                                if kind == "tanh_mlp" else
                                                "Centered kernel changes allocated by coordinate log-kernel changes; exact allocation, not causal importance.")
        settings["validation_warning"] = "Low sensitivity: repeat evaluation detected 1 of 12 isolated generated paragraphs and covered no generated characters in mixed documents, below linear v2. This experimental profile is explicit opt-in. Previously evaluated papers were reused; this is not new independent validation. An unflagged result cannot exclude AIGC."
    return profile, settings


def fused_score(scores, profile):
    """Evaluate score-only functions, exposing both heads and component sums.

    A pairwise term is split equally between its two models for the per-model
    accounting table. This is an exact sum, not a causal feature importance.
    """
    for model in profile["models"]:
        value = scores.get(model)
        probabilistic = not expanded_profile(profile) or profile["score_transforms"][model] == "logit"
        if value is not None and (not _finite(value) or probabilistic and not 0 <= value <= 1):
            raise ValueError(f"Invalid raw score for {model}")
    dual = dual_head(profile)
    missing = [m for m in profile["models"] if scores.get(m) is None]
    if missing:
        return {"score": None, "logit": None, "contributions": {}, "missing_models": missing,
                **({"review_score": None, "review_logit": None, "review_contributions": {}, "feature_contributions": []} if dual else {})}
    standardized_values = []
    eps, limit = profile["clip_probability"], profile["clip_standardized"]
    for model, center, scale in zip(profile["models"], profile["center"], profile["scale"]):
        transform = profile.get("score_transforms", {}).get(model, "logit")
        if transform == "logit":
            clipped = min(1-eps, max(eps, scores[model]))
            value = math.log(clipped/(1-clipped))
        else:
            value = -scores[model] if transform == "negative_identity" else scores[model]
        standardized_values.append(max(-limit, min(limit, (value - center)/scale)))
    if nonlinear_profile(profile):
        from .nonlinear import evaluate_head
        head = profile["heads"]["strong"]
        total, allocations, components = evaluate_head(standardized_values, head)
        contributions = {m: {"raw_ai_score": scores[m] if profile["score_transforms"][m] == "logit" else None,
                             "raw_score": scores[m], "score_transform": profile["score_transforms"][m],
                             "standardized_logit": standardized_values[i] if profile["score_transforms"][m] == "logit" else None,
                             "standardized_value": standardized_values[i], "logit_contribution": allocations[i]}
                         for i, m in enumerate(profile["models"])}
        if head["kind"] == "rbf":
            for i, model in enumerate(profile["models"]):
                contributions[model]["linear_coefficient"] = head["linear_coefficients"][i]
        features = [{"name": c["name"], "models": [profile["models"][i] for i in c["indices"]],
                     "value": c["value"], "coefficients": {n: c["coefficient"] for n in ("strong", "review")},
                     "logit_contributions": {n: c["contribution"] for n in ("strong", "review")}} for c in components]
        score = 1 / (1 + math.exp(-max(-700, min(700, total))))
        return {"score": score, "logit": total, "contributions": contributions, "missing_models": [],
                "review_score": score, "review_logit": total, "review_contributions": copy.deepcopy(contributions),
                "feature_contributions": features}
    terms = profile["feature_terms"] if dual else [[i] for i in range(len(profile["models"]))]
    heads = profile["heads"] if dual else {"strong": {"intercept": profile["intercept"], "coefficients": profile["coefficients"]}}
    head_results = {}
    features = [{"models": [profile["models"][i] for i in term],
                 "value": math.prod(standardized_values[i] for i in term),
                 "coefficients": {}, "logit_contributions": {}} for term in terms]
    for name, head in heads.items():
        contributions = {m: {"raw_ai_score": scores[m] if profile.get("score_transforms", {}).get(m, "logit") == "logit" else None,
                             **({"raw_score": scores[m], "score_transform": profile["score_transforms"][m]}
                                if expanded_profile(profile) else {}),
                             "standardized_logit": standardized_values[i] if profile.get("score_transforms", {}).get(m, "logit") == "logit" else None,
                             **({"standardized_value": standardized_values[i]} if expanded_profile(profile) else {}),
                             "logit_contribution": 0.0}
                         for i, m in enumerate(profile["models"])}
        values = []
        for term, feature, coefficient in zip(terms, features, head["coefficients"]):
            value = coefficient * feature["value"]
            values.append(value)
            feature["coefficients"][name] = coefficient
            feature["logit_contributions"][name] = value
            for i in term:
                contributions[profile["models"][i]]["logit_contribution"] += value / len(term)
            if len(term) == 1:
                contributions[profile["models"][term[0]]]["linear_coefficient" if dual else "coefficient"] = coefficient
        total = head["intercept"] + math.fsum(values)
        score = 1 / (1 + math.exp(-max(-700, min(700, total))))
        head_results[name] = {"score": score, "logit": total, "contributions": contributions}
    result = {**head_results["strong"], "missing_models": []}
    if dual:
        result.update({"review_score": head_results["review"]["score"], "review_logit": head_results["review"]["logit"],
                       "review_contributions": head_results["review"]["contributions"], "feature_contributions": features})
    return result


def decide_fusion(scores, profile, high, possible):
    from .annotation import LABELS, decide, score_category
    result = fused_score(scores, profile)
    raw = decide({m: scores.get(m) for m in profile["models"]
                  if not expanded_profile(profile) or profile["score_transforms"][m] == "logit"}, .9, .7)
    dual = dual_head(profile)
    if dual:
        category = ("strong_signal" if result["score"] is not None and result["score"] >= high else
                    "possible_signal" if result["review_score"] is not None and result["review_score"] >= possible else "other")
    else:
        category = score_category(result["score"], high, possible)
    status = "scored" if result["score"] is not None else "no_scorable_tokens" if not any(s is not None for s in scores.values()) else "incomplete_model_scores"
    return {**raw, "category": category, "label": LABELS[category],
            "triggered_models": list(profile["models"]) if category != "other" else [],
            "ensemble_score": result["score"], "ensemble_logit": result["logit"],
            "contributions": result["contributions"], "missing_models": result["missing_models"],
            **({key: result[key] for key in ("review_score", "review_logit", "review_contributions", "feature_contributions")} if dual else {}),
            "evaluation_status": status,
            "decision_basis": "joint_model_scores_only"}


def reapply_policy(report, *, policy=None, high_threshold=None, possible_threshold=None):
    """Recombine cached model scores without inference or source-dependent rules."""
    from datetime import datetime, timezone
    from . import __version__
    from .annotation import LABELS, decide, score_category, summarize, validate_policy
    if report.get("status") != "complete" or not report.get("segments"):
        raise ValueError("A complete annotation report with segments is required")
    selected = [m["id"] for m in report["models"]]
    segmentation = report["segmentation"]
    language = "en" if segmentation.get("id") == "english-natural-boundaries-v1" else "zh"
    if policy is None and language == "en" and report.get("policy", {}).get("id") in ENGLISH_PROFILES:
        policy = report["policy"]["id"]
    if policy in (REFERENCE_POLICY, LINEAR_POLICY) and language == "en":
        raise ValueError("Chinese fusion requires its original Chinese segmentation")
    policy = resolve_policy(policy, language)
    profile, settings = policy_settings(policy, selected, segmentation["max_characters"],
                                       segmentation["minimum_target_characters"], high_threshold, possible_threshold)
    if profile and segmentation.get("id") != profile_segmentation(profile):
        raise ValueError("Fusion requires its original language-specific segmentation")
    high, possible = settings["high_threshold"], settings["possible_threshold"]
    validate_policy(high, possible, segmentation["max_characters"], segmentation["minimum_target_characters"],
                    independent_thresholds=settings.get("threshold_mode") == "separate_heads")
    if profile and any(m["revision"] != signal_revision(profile, m["id"]) for m in report["models"]):
        raise ValueError("Cached model revisions differ from the fusion profile")
    if profile and profile.get("statistical_assets"):
        if report.get("statistical_execution", {}).get("inference") != profile["statistical_inference"]:
            raise ValueError("Cached statistical inference differs from the fusion profile")
        for m in report["models"]:
            if m["id"] in profile["statistical_assets"] and {a["id"]: a["revision"] for a in m.get("assets", [])} != profile["statistical_assets"][m["id"]]:
                raise ValueError("Cached statistical asset revisions differ from the fusion profile")
    result = copy.deepcopy(report)
    result["schema_version"] = "aigc-annotation/1.2"
    result["software_version"] = __version__
    result["recombined_at"] = datetime.now(timezone.utc).isoformat()
    result["recombined_from"] = {"policy": report["policy"]["id"], "input_sha256": report["input"]["sha256"]}
    result["policy"] = {**settings, "labels": LABELS}
    base_interpretation = "Uncalibrated screening labels, not proof of authorship. Other does not certify human writing. Character counts are flagged spans, not AI-written percentages."
    result["interpretation"] = base_interpretation + (" " + settings["validation_warning"] if "validation_warning" in settings else "")
    from .catalog import CATALOG
    result["interpretation"] += "".join(" " + CATALOG[mid]["score_warning"] for mid in selected
                                       if CATALOG.get(mid, {}).get("score_warning"))
    for segment in result["segments"]:
        if set(segment["model_scores"]) != set(selected):
            raise ValueError("Segment model scores do not match report roster")
        scores = segment_raw_scores(segment)
        segment["decision"] = decide_fusion(scores, profile, high, possible) if profile else decide(scores, high, possible)
        for model in segment["model_scores"].values():
            if "raw_score" not in model:
                model["category"] = score_category(model["ai_score"], settings["raw_score_high_threshold"], settings["raw_score_possible_threshold"])
        if segment["decision"]["evaluation_status"] != "scored" and segment["decision"]["evaluation_status"] not in segment["warnings"]:
            segment["warnings"].append(segment["decision"]["evaluation_status"])
    result["summary"] = summarize(result["segments"])
    return result
