import copy
import hashlib
import math
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aigc_detector.api import app
from aigc_detector.catalog import CATALOG
from aigc_detector.engine import Detector, ModelUnavailable
from aigc_detector.fusion import REFERENCE_POLICY, load_profile, fused_score, decide_fusion, reapply_policy, validate_profile


def test_profile_uses_all_pinned_models_and_finite_score_contributions():
    profile, digest = load_profile()
    assert set(profile["models"]) == set(CATALOG)
    assert len(digest) == 64
    assert all(c != 0 for c in profile["coefficients"])
    for value in (0, .5, 1):
        scores = dict.fromkeys(CATALOG, value)
        result = fused_score(scores, profile)
        assert 0 <= result["score"] <= 1
        assert math.isfinite(result["logit"])
        assert profile["intercept"] + sum(c["logit_contribution"] for c in result["contributions"].values()) == pytest.approx(result["logit"])
        assert scores == dict.fromkeys(CATALOG, value)


@pytest.mark.parametrize("invalid", [float('nan'), float('inf'), -.1, 1.1, True, '0.5'])
def test_invalid_model_output_is_never_turned_into_a_label(invalid):
    profile, _ = load_profile()
    scores = dict.fromkeys(CATALOG, .5); scores[next(iter(CATALOG))] = invalid
    with pytest.raises(ValueError, match="Invalid raw score"):
        fused_score(scores, profile)


def test_missing_model_does_not_silently_change_the_ensemble():
    profile, _ = load_profile()
    scores = dict.fromkeys(CATALOG, .5); scores[next(iter(CATALOG))] = None
    result = decide_fusion(scores, profile, .9, .7)
    assert result["ensemble_score"] is None
    assert result["evaluation_status"] == "incomplete_model_scores"
    result = decide_fusion(dict.fromkeys(CATALOG), profile, .9, .7)
    assert result["evaluation_status"] == "no_scorable_tokens"


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


def test_live_and_cached_fusion_agree_and_never_rewrite_raw_scores():
    from aigc_detector.annotation import annotate_text
    profile, _ = load_profile()
    text = "恒定阻抗传输结构与电流相位。" * 60
    scores = {m: .12 + i*.07 for i, m in enumerate(CATALOG)}
    detector = EvidenceDetector(profile, scores)
    legacy = annotate_text(detector, text, policy="sensitive-any-model-v1")
    original = copy.deepcopy(legacy)
    recombined = reapply_policy(legacy)
    live = annotate_text(detector, text, include_text=False)
    assert legacy == original
    assert recombined["summary"] == live["summary"]
    for cached, actual in zip(recombined["segments"], live["segments"]):
        assert cached["decision"] == actual["decision"]
        assert cached["model_scores"] == actual["model_scores"]
        assert "text" not in actual
    assert live["input"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_revision_drift_is_rejected_in_live_and_cached_fusion():
    from aigc_detector.annotation import annotate_text
    profile, _ = load_profile(); scores = dict.fromkeys(CATALOG, .5)
    with pytest.raises(ModelUnavailable, match="revision"):
        annotate_text(EvidenceDetector(profile, scores, True), "原始正文。"*40)
    legacy = annotate_text(EvidenceDetector(profile, scores), "原始正文。"*40, policy="sensitive-any-model-v1")
    legacy["models"][0]["revision"] = "changed"
    with pytest.raises(ValueError, match="revisions"):
        reapply_policy(legacy)


def test_broken_profile_cannot_produce_plausible_scores():
    profile, _ = load_profile()
    for field, value in [("scale", [0]*len(CATALOG)), ("coefficients", [1]), ("clip_probability", .5)]:
        broken = copy.deepcopy(profile); broken[field] = value
        with pytest.raises(ValueError):
            validate_profile(broken)
