import copy
import hashlib
import math
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aigc_detector.api import app
from aigc_detector.catalog import CHINESE_MODELS
from aigc_detector.engine import Detector, ModelUnavailable
from aigc_detector.fusion import (
    LEGACY_POLICY,
    LINEAR_POLICY,
    REFERENCE_POLICY,
    decide_fusion,
    fused_score,
    load_profile,
    reapply_policy,
    validate_profile,
)


@pytest.mark.parametrize("policy", [LINEAR_POLICY, REFERENCE_POLICY])
def test_profile_uses_all_pinned_models_and_finite_score_contributions(policy):
    profile, digest = load_profile(policy=policy)
    assert set(profile["models"]) == set(CHINESE_MODELS)
    assert len(digest) == 64
    if policy == LINEAR_POLICY:
        assert all(c != 0 for c in profile["coefficients"])
    for value in (0, .5, 1):
        scores = dict.fromkeys(CHINESE_MODELS, value)
        result = fused_score(scores, profile)
        assert 0 <= result["score"] <= 1
        assert math.isfinite(result["logit"])
        if policy == REFERENCE_POLICY:
            assert 0 <= result["review_score"] <= 1
            assert math.isfinite(result["review_logit"])
            assert profile["heads"]["strong"]["intercept"] + math.fsum(
                c["logit_contribution"] for c in result["contributions"].values()
            ) == pytest.approx(result["logit"])
            assert profile["heads"]["review"]["intercept"] + math.fsum(
                c["logit_contribution"] for c in result["review_contributions"].values()
            ) == pytest.approx(result["review_logit"])
            assert len(result["feature_contributions"]) == len(profile["feature_terms"])
        else:
            assert profile["intercept"] + math.fsum(
                c["logit_contribution"] for c in result["contributions"].values()
            ) == pytest.approx(result["logit"])
            assert "review_score" not in result
        assert scores == dict.fromkeys(CHINESE_MODELS, value)


def _synthetic_dual_profile():
    """Build a small, hand-computable v2 profile using the pinned ten-model roster."""
    profile, _ = load_profile(policy=LINEAR_POLICY)
    models = profile["models"]
    profile["id"] = REFERENCE_POLICY
    profile["schema_version"] = "aigc-score-fusion/2.0"
    profile["center"] = [0.0] * len(models)
    profile["scale"] = [1.0] * len(models)
    profile["clip_probability"] = 1e-6
    profile["clip_standardized"] = 5.0
    profile["high_threshold"] = .7
    profile["possible_threshold"] = .85
    profile["feature_terms"] = [[i] for i in range(len(models))] + [[0, 0], [0, 1]]
    strong = [0.0] * len(models) + [-.2, .3]
    strong[0] = .4
    review = [0.0] * len(models) + [-.1, .2]
    review[0] = .6
    profile["heads"] = {
        "strong": {"intercept": .1, "coefficients": strong},
        "review": {"intercept": -.2, "coefficients": review},
    }
    profile.pop("coefficients", None)
    profile.pop("intercept", None)
    validate_profile(profile)
    return profile


def _sigmoid(value):
    return 1 / (1 + math.exp(-value))


def test_dual_head_fusion_matches_hand_calculated_square_and_interaction_terms():
    profile = _synthetic_dual_profile()
    models = profile["models"]
    standardized = [1.0, 2.0] + [0.0] * (len(models) - 2)
    scores = {model: _sigmoid(value) for model, value in zip(models, standardized)}

    result = fused_score(scores, profile)
    expected_feature_values = standardized + [1.0, 2.0]
    expected_contributions = {
        head: dict.fromkeys(models, 0.0) for head in ("strong", "review")
    }
    expected_logits = {}
    for head_name, head in profile["heads"].items():
        term_total = 0.0
        for term, coefficient, feature_value in zip(
            profile["feature_terms"], head["coefficients"], expected_feature_values
        ):
            term_value = coefficient * feature_value
            term_total += term_value
            for index in term:
                expected_contributions[head_name][models[index]] += term_value / len(term)
        expected_logits[head_name] = head["intercept"] + term_total

    assert result["logit"] == pytest.approx(expected_logits["strong"])
    assert result["score"] == pytest.approx(_sigmoid(expected_logits["strong"]))
    assert result["review_logit"] == pytest.approx(expected_logits["review"])
    assert result["review_score"] == pytest.approx(_sigmoid(expected_logits["review"]))
    for model in models:
        assert result["contributions"][model]["logit_contribution"] == pytest.approx(
            expected_contributions["strong"][model]
        )
        assert result["review_contributions"][model]["logit_contribution"] == pytest.approx(
            expected_contributions["review"][model]
        )
    for feature, term, value in zip(
        result["feature_contributions"], profile["feature_terms"], expected_feature_values
    ):
        assert feature["models"] == [models[index] for index in term]
        assert feature["value"] == pytest.approx(value)
        for head_name, head in profile["heads"].items():
            index = profile["feature_terms"].index(term)
            coefficient = head["coefficients"][index]
            assert feature["coefficients"][head_name] == pytest.approx(coefficient)
            assert feature["logit_contributions"][head_name] == pytest.approx(coefficient * value)


def test_independent_heads_use_strong_priority_then_review_fallback():
    profile = _synthetic_dual_profile()
    standardized = [1.0, 2.0] + [0.0] * (len(profile["models"]) - 2)
    scores = {model: _sigmoid(value) for model, value in zip(profile["models"], standardized)}

    # Strong crosses its threshold while review stays below its separate threshold.
    strong_first = decide_fusion(scores, profile, high=.7, possible=.85)
    assert strong_first["ensemble_score"] >= .7
    assert strong_first["review_score"] < .85
    assert strong_first["category"] == "strong_signal"

    # When strong is below its threshold, a passing review head yields possible.
    review_only_profile = copy.deepcopy(profile)
    review_only_profile["heads"]["strong"]["intercept"] = -5
    review_only_profile["heads"]["review"]["intercept"] = 2
    review_only = decide_fusion(scores, review_only_profile, high=.7, possible=.85)
    assert review_only["ensemble_score"] < .7
    assert review_only["review_score"] >= .85
    assert review_only["category"] == "possible_signal"


@pytest.mark.parametrize("policy", [LINEAR_POLICY, REFERENCE_POLICY])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -.1, 1.1, True, "0.5"])
def test_invalid_model_output_is_never_turned_into_a_label(policy, invalid):
    profile, _ = load_profile(policy=policy)
    scores = dict.fromkeys(CHINESE_MODELS, .5)
    scores[next(iter(CHINESE_MODELS))] = invalid
    with pytest.raises(ValueError, match="Invalid raw score"):
        fused_score(scores, profile)


@pytest.mark.parametrize("policy", [LINEAR_POLICY, REFERENCE_POLICY])
def test_missing_model_scores_make_both_v2_heads_unavailable(policy):
    profile, _ = load_profile(policy=policy)
    scores = dict.fromkeys(CHINESE_MODELS, .5)
    scores[next(iter(CHINESE_MODELS))] = None
    result = decide_fusion(scores, profile, .9, .7)
    assert result["ensemble_score"] is None
    assert result["evaluation_status"] == "incomplete_model_scores"
    if policy == REFERENCE_POLICY:
        assert result["review_score"] is None
        assert result["ensemble_logit"] is None
        assert result["review_logit"] is None
        assert result["contributions"] == {}
        assert result["review_contributions"] == {}
    else:
        assert "review_score" not in result
    result = decide_fusion(dict.fromkeys(CHINESE_MODELS), profile, .9, .7)
    assert result["evaluation_status"] == "no_scorable_tokens"
    assert result["category"] == "other"


def test_new_policy_rejects_subset_segment_change_and_unknown_policy_before_loading(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        for extra in ({"models": ["hc3-zh"]}, {"segment_chars": 500}, {"min_segment_chars": 100}, {"policy": "unknown"}):
            response = client.post("/v1/annotate", json={"text": "本地中文正文。", **extra})
            assert response.status_code == 422, response.text
        assert not detector._loaded


class EvidenceDetector:
    device = "test"

    def __init__(self, profile, scores, bad_revision=False):
        import threading
        self._lock = threading.RLock()
        self.profile, self.scores, self.bad_revision = profile, scores, bad_revision

    def _score_texts(self, passages, model_id, *args, **kwargs):
        return [{"ai_score": self.scores[model_id], "token_count": len(text), "chunk_count": 1,
                 "chunks": [], "revision": "changed" if self.bad_revision else self.profile["revisions"][model_id]} for text in passages]


def test_live_and_cached_fusion_agree_for_v1_v2_and_legacy_policies():
    from aigc_detector.annotation import annotate_text
    profile, _ = load_profile(policy=REFERENCE_POLICY)
    text = "恒定阻抗传输结构与电流相位。" * 60
    scores = {model: .12 + i * .07 for i, model in enumerate(CHINESE_MODELS)}
    detector = EvidenceDetector(profile, scores)

    for policy in (LINEAR_POLICY, REFERENCE_POLICY, LEGACY_POLICY):
        report = annotate_text(detector, text, policy=policy)
        original = copy.deepcopy(report)
        recombined = reapply_policy(report, policy=policy)
        live = annotate_text(detector, text, include_text=False, policy=policy)
        assert report == original
        assert recombined["summary"] == live["summary"]
        for cached, actual in zip(recombined["segments"], live["segments"]):
            assert cached["decision"] == actual["decision"]
            assert cached["model_scores"] == actual["model_scores"]
            assert "text" not in actual
            if policy == REFERENCE_POLICY:
                assert cached["decision"]["review_score"] == actual["decision"]["review_score"]
        assert live["input"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert live["schema_version"] == "aigc-annotation/1.2"


def test_revision_drift_is_rejected_in_live_and_cached_fusion():
    from aigc_detector.annotation import annotate_text
    profile, _ = load_profile(policy=REFERENCE_POLICY)
    scores = dict.fromkeys(CHINESE_MODELS, .5)
    for policy in (REFERENCE_POLICY, LINEAR_POLICY):
        with pytest.raises(ModelUnavailable, match="revision"):
            annotate_text(EvidenceDetector(profile, scores, True), "原始正文。" * 40, policy=policy)
        report = annotate_text(EvidenceDetector(profile, scores), "原始正文。" * 40, policy=policy)
        report["models"][0]["revision"] = "changed"
        with pytest.raises(ValueError, match="revisions"):
            reapply_policy(report, policy=policy)


def test_broken_v1_profile_cannot_produce_plausible_scores():
    profile, _ = load_profile(policy=LINEAR_POLICY)
    for field, value in [("scale", [0] * len(CHINESE_MODELS)), ("coefficients", [1]), ("clip_probability", .5)]:
        broken = copy.deepcopy(profile)
        broken[field] = value
        with pytest.raises(ValueError):
            validate_profile(broken)


def test_v2_profile_rejects_invalid_features_and_incomplete_heads():
    profile = _synthetic_dual_profile()
    invalid_profiles = []

    broken_term = copy.deepcopy(profile)
    broken_term["feature_terms"].append([len(CHINESE_MODELS)])
    invalid_profiles.append(broken_term)

    unsorted_term = copy.deepcopy(profile)
    unsorted_term["feature_terms"][-1] = [1, 0]
    invalid_profiles.append(unsorted_term)

    incomplete_head = copy.deepcopy(profile)
    incomplete_head["heads"].pop("review")
    invalid_profiles.append(incomplete_head)

    short_head = copy.deepcopy(profile)
    short_head["heads"]["strong"]["coefficients"].pop()
    invalid_profiles.append(short_head)

    for broken in invalid_profiles:
        with pytest.raises(ValueError):
            validate_profile(broken)



@pytest.mark.parametrize("policy", [LINEAR_POLICY, REFERENCE_POLICY])
def test_cached_fusion_rejects_duplicate_model_records(policy):
    from aigc_detector.annotation import annotate_text

    profile, _ = load_profile(policy=policy)
    scores = {model: .5 for model in profile["models"]}
    report = annotate_text(
        EvidenceDetector(profile, scores), "恒定阻抗传输结构与电流相位。" * 40, policy=policy
    )
    report["models"].append(copy.deepcopy(report["models"][0]))

    with pytest.raises(ValueError):
        reapply_policy(report, policy=policy)
