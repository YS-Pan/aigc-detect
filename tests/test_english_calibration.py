"""Language/profile contracts and evidence isolation for English calibration."""
import copy
import json
import math
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from aigc_detector.api import app
from aigc_detector.catalog import ENGLISH_MODELS
from aigc_detector.engine import Detector
from aigc_detector.fusion import ENGLISH_POLICY, REFERENCE_POLICY, fused_score, load_profile, reapply_policy, validate_profile
from scripts.calibrate_english import fit, score, write_json
from scripts.english_academic_samples import plain_text


def fake_scores(profile):
    def score(passages, mid, *args, **kwargs):
        return [{"ai_score": .2, "token_count": len(p), "chunk_count": 1, "chunks": [],
                 "revision": profile["revisions"][mid]} for p in passages]
    return score


def test_original_english_profile_api_and_cache_recombination(tmp_path):
    profile, digest = load_profile(ENGLISH_POLICY)
    assert set(profile["models"]) == set(ENGLISH_MODELS)
    text = "English academic sentences are preserved exactly. " * 80
    detector = Detector(device="cpu", model_dir=tmp_path)
    with patch.object(detector, "_score_texts", side_effect=fake_scores(profile)), \
            patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        response = client.post("/v1/annotate", json={"text": text, "language": "en", "policy": ENGLISH_POLICY})
        assert response.status_code == 200
        report = response.json()
    assert report["policy"]["id"] == ENGLISH_POLICY
    assert report["policy"]["profile_sha256"] == digest
    assert report["segmentation"]["max_characters"] == profile["segment_chars"]
    assert report["segmentation"]["minimum_target_characters"] == profile["min_segment_chars"]
    assert "".join(s["text"] for s in report["segments"]) == text
    assert not report["policy"]["calibrated"]
    assert "Low sensitivity" in report["interpretation"]
    assert "12" in report["policy"]["validation_warning"]
    cached = reapply_policy(report)
    assert cached["summary"] == report["summary"]
    assert [s["decision"] for s in cached["segments"]] == [s["decision"] for s in report["segments"]]
    altered = copy.deepcopy(report)
    altered["models"][0]["revision"] = "wrong"
    with pytest.raises(ValueError, match="revisions"):
        reapply_policy(altered)


def test_language_roster_and_segmentation_rejected_before_model_loading(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    profile, _ = load_profile(ENGLISH_POLICY)
    with patch.object(detector, "_score_texts") as score:
        with pytest.raises(ValueError, match="English"):
            detector.annotate("中文正文", policy=ENGLISH_POLICY)
        with pytest.raises(ValueError, match="Chinese"):
            detector.annotate("English text.", language="en", policy=REFERENCE_POLICY)
        with pytest.raises(ValueError, match="complete"):
            detector.annotate("English text.", language="en", models=["hc3-en"], policy=ENGLISH_POLICY)
        with pytest.raises(ValueError, match="segment_chars"):
            detector.annotate("English text.", language="en", segment_chars=profile["segment_chars"]+1, policy=ENGLISH_POLICY)
        score.assert_not_called()


def test_polynomial_english_profile_exact_contribution_accounting_and_missing_scores():
    profile, _ = load_profile(ENGLISH_POLICY)
    raw = dict.fromkeys(profile["models"], .25)
    result = fused_score(raw, profile)
    for name, prefix in [("strong", ""), ("review", "review_")]:
        total = profile["heads"][name]["intercept"] + math.fsum(v["logit_contribution"] for v in result[prefix+"contributions"].values())
        assert total == pytest.approx(result[prefix+"logit"])
    assert raw == dict.fromkeys(profile["models"], .25)
    raw[profile["models"][0]] = None
    result = fused_score(raw, profile)
    assert result["score"] is None and result["review_score"] is None
    bad = copy.deepcopy(profile)
    bad["models"].pop()
    with pytest.raises(ValueError, match="four-model"):
        validate_profile(bad)


def test_test_scores_cannot_influence_parameters_candidate_selection_or_thresholds(tmp_path):
    records = []
    for paper, split in [("P01", "fit"), ("P02", "fit"), ("P03", "calibration"), ("P04", "test")]:
        for kind, value in [("human_document", .15), ("ai_paragraph", .85)]:
            for n in range(3):
                records.append({"paper_id": paper, "split": split, "kind": kind, "chars": 300,
                                "sample_id": f"{paper}-{kind}-{n}", "generator": "generator" if kind == "ai_paragraph" else None,
                                "cross_paper_duplicate": False, "scores": dict.fromkeys(ENGLISH_MODELS, value+n*.01)})
    data = {"records": records, "config": {"protocol_sha256": "fixture", "revisions": dict.fromkeys(ENGLISH_MODELS, "fixture")},
            "protocol": {"candidate_regularization": [1], "candidate_human_weights": [5], "candidate_feature_families": ["linear"],
                         "groups": {"fit": ["P01", "P02"], "calibration": ["P03"], "test": ["P04"]},
                         "segment_chars": 800, "min_segment_chars": 240}}
    source = tmp_path / "scores.json"
    profiles = []
    for run in range(2):
        if run:
            for r in data["records"]:
                if r["split"] == "test":
                    r["scores"] = dict.fromkeys(ENGLISH_MODELS, 1 if r["kind"] == "human_document" else 0)
        write_json(source, data)
        profile_path = tmp_path / f"profile{run}.json"
        fit(source, profile_path, tmp_path / f"fit{run}.json")
        profiles.append(json.loads(profile_path.read_text(encoding="utf-8")))
    for key in ["center", "scale", "feature_terms", "heads", "high_threshold", "possible_threshold"]:
        assert profiles[0][key] == profiles[1][key]
    assert profiles[0]["fit"]["family"] == profiles[1]["fit"]["family"]
    assert profiles[0]["fit"]["test_used_for_fitting_or_selection"] is False


def test_consistent_mechanical_text_extraction():
    text = r"\begin{abstract}Human and generated $x^2$ text [1] share \emph{the same} 10\% extraction.\end{abstract}"
    assert plain_text(text) == "Human and generated text share the same 10% extraction."


def test_test_inference_cannot_read_samples_before_freeze_and_segmentation_checks(tmp_path):
    profile, _ = load_profile(ENGLISH_POLICY)
    altered = tmp_path / "unfrozen.json"
    write_json(altered, profile)
    # Re-serialization changes bytes; explicitly add an invalidated fit marker.
    profile["fit"]["test_used_for_fitting_or_selection"] = True
    write_json(altered, profile)
    with patch("scripts.calibrate_english.verify") as verify_samples, \
            patch("aigc_detector.engine.Detector") as detector:
        with pytest.raises(ValueError, match="unchanged frozen"):
            score(tmp_path, tmp_path/"scores.json", tmp_path/"reports", splits=("test",), profile_path=altered)
        with pytest.raises(ValueError, match="frozen segmentation"):
            score(tmp_path, tmp_path/"scores.json", tmp_path/"reports", segment_chars=129,
                  min_segment_chars=120, splits=("test",))
        verify_samples.assert_not_called()
        detector.assert_not_called()
