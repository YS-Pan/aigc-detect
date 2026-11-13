"""Mixed-score contracts, complete sample use and paper-level isolation."""
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aigc_detector.api import app
from aigc_detector.catalog import ENGLISH_MODELS
from aigc_detector.engine import Detector
from aigc_detector.fusion import (ENGLISH_EXPANDED_POLICY, fused_score, validate_profile,
                                 reapply_policy, signal_revision, load_profile, resolve_policy)
from scripts import calibrate_english_v2 as calibration


def profile_fixture():
    models = ["hc3-en", "fast-detect-gpt-neo", "binoculars-falcon"]
    head = {"intercept": .1, "coefficients": [.3, -.2, .4]}
    return {"id": ENGLISH_EXPANDED_POLICY, "schema_version": "aigc-score-fusion/3.0", "language": "en",
            "segmentation_id": "english-natural-boundaries-v1", "models": models,
            "revisions": {"hc3-en": "fixture"},
            "statistical_assets": {"fast-detect-gpt-neo": {"gpt-neo-2.7b": "fixture"},
                                   "binoculars-falcon": {"falcon-7b": "fixture", "falcon-7b-instruct": "fixture"}},
            "statistical_inference": {"max_tokens": 512, "overlap_tokens": 64, "dtype": "bfloat16",
                                      "aggregation": "inverse_coverage_token_statistics_then_formula"},
            "score_transforms": dict(zip(models, ["logit", "identity", "negative_identity"])),
            "center": [0., 1., -1.], "scale": [1., 2., .2], "clip_probability": 1e-5,
            "clip_standardized": 5., "feature_terms": [[0], [1], [2]], "heads": {"strong": head, "review": head},
            "segment_chars": 800, "min_segment_chars": 240, "high_threshold": .8, "possible_threshold": .7}


def test_raw_statistics_keep_their_scale_direction_and_exact_contribution_sum():
    profile = profile_fixture()
    validate_profile(profile)
    scores = {"hc3-en": .2, "fast-detect-gpt-neo": -3., "binoculars-falcon": 1.1}
    original = copy.deepcopy(scores)
    result = fused_score(scores, profile)
    z = [math.log(.2/.8), (-3.-1.)/2., (-1.1+1.)/.2]
    expected = .1 + np.dot(z, [.3, -.2, .4])
    assert result["logit"] == pytest.approx(expected, abs=1e-12)
    assert .1 + math.fsum(c["logit_contribution"] for c in result["contributions"].values()) == pytest.approx(expected)
    assert result["contributions"]["fast-detect-gpt-neo"]["raw_ai_score"] is None
    assert scores == original
    scores["binoculars-falcon"] = None
    assert fused_score(scores, profile)["score"] is None
    scores["hc3-en"] = -1
    with pytest.raises(ValueError, match="raw score"):
        fused_score(scores, profile)
    changed = copy.deepcopy(profile)
    changed["score_transforms"]["binoculars-falcon"] = "logit"
    with pytest.raises(ValueError, match="transforms"):
        validate_profile(changed)


def test_default_expanded_api_roster_readiness_and_cache(tmp_path):
    profile, digest = load_profile(ENGLISH_EXPANDED_POLICY)
    detector = Detector(device="cpu", model_dir=tmp_path)
    seen = []
    def classify(passages, mid, *args, **kwargs):
        seen.append(mid)
        return [{"ai_score": .2, "revision": profile["revisions"][mid], "token_count": 50,
                 "chunk_count": 1, "chunks": []} for _ in passages]
    manifest = {m: {"revision": r} for m, r in profile["revisions"].items()}
    with patch.object(detector, "_score_texts", side_effect=classify), \
            patch.object(detector, "_manifest", return_value=manifest), \
            patch.object(detector, "_installed", return_value=True), \
            patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        response = client.post("/v1/annotate", json={"text": "An academic sentence. " * 100, "language": "en"})
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["policy"]["id"] == resolve_policy(None, "en") == ENGLISH_EXPANDED_POLICY
        assert report["policy"]["profile_sha256"] == digest
        assert seen == profile["models"]
        assert reapply_policy(report)["summary"] == report["summary"]
        health = client.get("/health").json()
        assert health["annotation_profiles"]["en"]["ready"]
        assert health["annotation_profiles"]["en"]["models"] == profile["models"]
        assert health["model_groups"]["en"]["models"] == list(ENGLISH_MODELS)
        manifest[profile["models"][0]]["revision"] = "changed"
        assert not client.get("/health").json()["annotation_profiles"]["en"]["ready"]


def test_spatial_metrics_count_generated_positions_and_serialize_numpy_flags():
    rows = [{"sample_id": "mixed", "char_start": 0, "char_end": 10, "chars": 10, "ai_chars": 0},
            {"sample_id": "mixed", "char_start": 10, "char_end": 20, "chars": 10, "ai_chars": 6}]
    mapping = {"blocks": [{"source_label": "human", "char_start": 0, "char_end": 12},
                          {"source_label": "ai", "char_start": 12, "char_end": 18},
                          {"source_label": "human", "char_start": 18, "char_end": 20}]}
    with patch.object(calibration, "read_json", return_value=mapping):
        metrics = calibration.spatial_metrics(rows, np.array([False, True]), {"mixed": {"mapping_file": "fixture"}})
    assert metrics["mixed_documents_with_generated_position_flagged"] == 1
    assert metrics["generated_blocks_with_any_flag"] == 1
    assert metrics["ai_flagged_characters"] == 6
    assert metrics["human_characters_within_flagged_segments"] == 4
    assert metrics["pure_human_flagged_segments"] == 0
    assert json.loads(json.dumps(metrics)) == metrics


def test_final_selection_rejects_incomplete_declared_signals_before_freezing(tmp_path):
    calibration.write_json(tmp_path / "campaign.json", {"segmentations": [[400, 120]],
                                                        "feature_rosters": [["hc3-en", "mage-en"]]})
    calibration.write_json(tmp_path / "context400/train-scores.json", {"config": {"signals": ["hc3-en"]}})
    with patch.object(calibration, "EVIDENCE", tmp_path), patch.object(calibration, "search_context") as search:
        with pytest.raises(ValueError, match="every declared signal"):
            calibration.search()
        search.assert_not_called()
    assert not (tmp_path / "FREEZE.json").exists()


def test_mixed_training_deduplicates_whole_passages_and_excludes_label_conflicts():
    rows = [{"paper_id": "P01", "split": "fit", "sha256": "a", "ai_chars": 0, "cross_paper_duplicate": False},
            {"paper_id": "P01", "split": "fit", "sha256": "a", "ai_chars": 0, "cross_paper_duplicate": False},
            {"paper_id": "P01", "split": "fit", "sha256": "b", "ai_chars": 10, "cross_paper_duplicate": False},
            {"paper_id": "P01", "split": "fit", "sha256": "b", "ai_chars": 0, "cross_paper_duplicate": False},
            {"paper_id": "P02", "split": "test", "sha256": "c", "ai_chars": 10, "cross_paper_duplicate": False}]
    assert len(calibration.eligible({"records": rows}, "fit")) == 1
    assert len(calibration.eligible({"records": rows}, "fit", deduplicate=False)) == 2


def test_expanded_annotation_and_recombination_pin_statistical_assets_and_windows(tmp_path):
    profile = profile_fixture()
    detector = Detector(device="cpu", model_dir=tmp_path)
    text = "An academic sentence for an isolated input context. " * 20

    def classify(passages, mid, *args, **kwargs):
        return [{"ai_score": .2, "revision": "fixture", "token_count": 50, "chunk_count": 1, "chunks": []} for _ in passages]

    def statistical(detector, passages, selected):
        rows = {}
        for mid, assets in profile["statistical_assets"].items():
            rows[mid] = [{"raw_score": 1.1, "ai_score": None, "kind": "statistical", "revision": signal_revision(profile, mid),
                          "token_count": 50, "chunk_count": 1, "chunks": [], "score_direction": "fixture",
                          "assets": [{"id": a, "revision": r} for a, r in assets.items()], "aggregation": "fixture"} for _ in passages]
        return rows, []

    with patch("aigc_detector.fusion.load_profile", return_value=(profile, "digest")), \
            patch("aigc_detector.annotation.load_profile", return_value=(profile, "digest")), \
            patch.object(detector, "_score_texts", side_effect=classify), \
            patch("aigc_detector.annotation.statistical_passage_scores", side_effect=statistical):
        report = detector.annotate(text, language="en", policy=ENGLISH_EXPANDED_POLICY)
        assert "".join(s["text"] for s in report["segments"]) == text
        assert report["models"][1]["revision"] == signal_revision(profile, "fast-detect-gpt-neo")
        cached = reapply_policy(report, policy=ENGLISH_EXPANDED_POLICY)
        assert cached["summary"] == report["summary"]
        assert [s["decision"] for s in cached["segments"]] == [s["decision"] for s in report["segments"]]
        changed = copy.deepcopy(report)
        changed["statistical_execution"]["inference"]["max_tokens"] = 1024
        with pytest.raises(ValueError, match="statistical inference"):
            reapply_policy(changed, policy=ENGLISH_EXPANDED_POLICY)
        changed = copy.deepcopy(report)
        changed["models"][1]["assets"][0]["revision"] = "wrong"
        with pytest.raises(ValueError, match="asset revisions"):
            reapply_policy(changed, policy=ENGLISH_EXPANDED_POLICY)


def test_prior_test_and_generated_calibration_scores_cannot_change_selection_or_thresholds(tmp_path):
    evidence = tmp_path / "evaluations"
    campaign = {"groups": {"fit": ["P01", "P02"], "calibration": ["P03"], "test": ["P04"]},
                "segmentations": [[800, 240]], "feature_rosters": [["hc3-en"]], "families": ["linear"],
                "regularizations": [1.], "human_weights": [5.], "target_modes": ["ai_character_fraction"],
                "threshold_logit_margins": {"strong": .1, "review": .01}, "revisions": {"hc3-en": "fixture"},
                "asset_revisions": {}, "statistical_inference": profile_fixture()["statistical_inference"]}
    records = []
    for paper, split in [("P01", "fit"), ("P02", "fit"), ("P03", "calibration"), ("P04", "test")]:
        for kind, ai, value in [("human_document", 0, .1), ("ai_paragraph", 300, .9), ("mixed_document", 150, .6)]:
            records.append({"paper_id": paper, "split": split, "kind": kind, "ai_chars": ai, "chars": 300,
                            "sample_id": f"{paper}-{kind}", "sha256": f"{paper}-{kind}", "cross_paper_duplicate": False,
                            "scores": {"hc3-en": value}, "generator": "fixture" if ai else None})
    calibration.write_json(evidence / "campaign.json", campaign)
    path = evidence / "context800/train-scores.json"
    profiles, selections = [], []
    with patch.object(calibration, "ROOT", tmp_path), patch.object(calibration, "EVIDENCE", evidence), \
            patch.object(calibration, "REPORTS", tmp_path / "reports"):
        for n in range(2):
            if n:
                (evidence / "FREEZE.json").unlink()
                for r in records:
                    if r["split"] == "test" or r["split"] == "calibration" and r["ai_chars"] > 0:
                        r["scores"] = {"hc3-en": 1.-r["scores"]["hc3-en"]}
            calibration.write_json(path, {"config": {"signals": ["hc3-en"]}, "records": records})
            calibration.search()
            profiles.append(calibration.read_json(tmp_path / "aigc_detector/profiles" / f"{calibration.POLICY}.json"))
            selections.append(calibration.read_json(evidence / "selection.json")["selected"])
        with pytest.raises(FileExistsError, match="Frozen"):
            calibration.search()
    for key in ["models", "center", "scale", "feature_terms", "heads", "high_threshold", "possible_threshold"]:
        assert profiles[0][key] == profiles[1][key]
    assert selections[0] == selections[1]
