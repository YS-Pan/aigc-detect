"""Independent arithmetic, cancellation and deterministic optimization checks."""
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aigc_detector.nonlinear import evaluate_head, validate_head
from scripts.nonlinear_english_fit import fit_mlp, fit_rbf, predict_z


def test_profile_contract_and_runtime_raw_score_validation():
    from aigc_detector.fusion import ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY, fused_score, load_profile, validate_profile
    profile, _ = load_profile(ENGLISH_EXPANDED_POLICY)
    profile.update(id=ENGLISH_NONLINEAR_POLICY, schema_version="aigc-score-fusion/4.0")
    profile.pop("feature_terms")
    head = {"kind": "tanh_mlp", "intercept": -.4, "input_weights": [[.1]*len(profile["models"])],
            "hidden_bias": [.2], "output_weights": [.5]}
    profile["heads"] = {"strong": head, "review": copy.deepcopy(head)}
    validate_profile(profile)
    scores = {m: .3 for m in profile["models"]}
    original = copy.deepcopy(scores)
    value = fused_score(scores, profile)
    assert value["score"] == value["review_score"]
    assert head["intercept"]+sum(c["logit_contribution"] for c in value["contributions"].values()) == pytest.approx(value["logit"])
    assert scores == original
    scores[profile["models"][0]] = None
    assert fused_score(scores, profile)["missing_models"] == [profile["models"][0]]
    scores[profile["models"][0]] = 1.1
    with pytest.raises(ValueError, match="raw score"):
        fused_score(scores, profile)
    profile["heads"]["review"]["intercept"] = 1.
    with pytest.raises(ValueError, match="identical"):
        validate_profile(profile)


@pytest.mark.parametrize("head", [
    {"kind": "rbf", "intercept": -.4, "linear_coefficients": [.3, -.2],
     "centers": [[.2, -.3], [1., 1.]], "bandwidth": .8, "basis_coefficients": [.7, -.9]},
    {"kind": "tanh_mlp", "intercept": -.4, "input_weights": [[.3, -.2], [1., -1.]],
     "hidden_bias": [.1, -.3], "output_weights": [.7, -.9]},
])
def test_runtime_agrees_with_independent_formula_and_allocations(head):
    validate_head(head, 2)
    original = copy.deepcopy(head)
    for z in ([0., 0.], [1., 1.], [-5., 5.], [.2, -.3], [1e-14, -1e-14]):
        if head["kind"] == "rbf":
            expected = head["intercept"] + sum(a*b for a, b in zip(z, head["linear_coefficients"]))
            for c, a in zip(head["centers"], head["basis_coefficients"]):
                expected += a * (math.exp(-sum((x-y)**2 for x, y in zip(z, c))/(2*.8**2))
                                 - math.exp(-sum(x*x for x in c)/(2*.8**2)))
        else:
            expected = head["intercept"] + sum(a*(math.tanh(b+sum(x*y for x, y in zip(z, w)))-math.tanh(b))
                          for w, b, a in zip(head["input_weights"], head["hidden_bias"], head["output_weights"]))
        value, allocations, components = evaluate_head(z, head)
        assert value == pytest.approx(expected, abs=1e-12)
        assert head["intercept"] + math.fsum(allocations) == pytest.approx(value, abs=1e-12)
        assert head["intercept"] + math.fsum(c["contribution"] for c in components) == pytest.approx(value, abs=1e-12)
        assert predict_z(np.array([z]), head)[0] == pytest.approx(value, abs=1e-12)
    assert head == original


@pytest.mark.parametrize("head", [
    {"kind": "rbf", "intercept": 0., "linear_coefficients": [0.], "centers": [[0.]],
     "bandwidth": 0., "basis_coefficients": [1.]},
    {"kind": "tanh_mlp", "intercept": 0., "input_weights": [[float("nan")]],
     "hidden_bias": [0.], "output_weights": [1.]},
])
def test_invalid_nonlinear_parameters_are_rejected(head):
    with pytest.raises(ValueError):
        validate_head(head, 1)


def test_both_optimizers_improve_objective_and_reproduce_parameters():
    z = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.], [-.8, -.8], [.8, .8]])
    y = np.array([1., 0., 0., 1., 1., 1.])
    weights = np.ones(len(z))
    config = {"hidden": 4, "regularization": .01, "target_mode": "ranking_presence", "width": .5}
    optimizer = {"seeds": [20, 21], "adam_steps": 80, "adam_learning_rate": .03,
                 "lbfgs_steps": 30, "ranking_margin": .1, "ranking_weight": 1.}
    head, info = fit_mlp(z, y, weights, config, optimizer)
    again, _ = fit_mlp(z, y, weights, config, optimizer)
    assert head == again
    assert all(s["final_objective"] < s["initial_objective"] for s in info["starts"])
    assert predict_z(z, head)[y == 1].min() > predict_z(z, head)[y == 0].max()
    rbf, evidence = fit_rbf(z, y, weights, config, z[:4])
    assert evidence["converged"]
    assert evidence["final_gradient_inf_norm"] < 1e-6
    assert predict_z(z, rbf)[y == 1].min() > predict_z(z, rbf)[y == 0].max()


def test_frozen_campaign_has_complete_candidates_and_no_repeat_fitting():
    from scripts.calibrate_english_nonlinear import EVIDENCE, configurations
    from scripts.english_academic_samples import read_json, digest
    from aigc_detector.fusion import ENGLISH_NONLINEAR_POLICY, load_profile
    campaign, freeze = read_json(EVIDENCE / "campaign.json"), read_json(EVIDENCE / "FREEZE.json")
    selection = read_json(EVIDENCE / "selection.json")
    profile, profile_hash = load_profile(ENGLISH_NONLINEAR_POLICY)
    assert len(list(configurations(campaign))) == len(selection["candidates"]) == 648
    assert freeze["profile_sha256"] == profile_hash
    assert freeze["selection_sha256"] == digest((EVIDENCE / "selection.json").read_bytes())
    assert not profile["fit"]["test_used_for_fitting_or_selection"]
    assert not selection["test_used_for_selection"]
    assert set(campaign["groups"]["test"]).isdisjoint(campaign["groups"]["fit"]+campaign["groups"]["calibration"])
    assert {r["split"] for r in read_json(EVIDENCE / "train-scores.json")["records"]} == {"fit", "calibration"}
    assert campaign["prior_test_outcomes_known"] and not campaign["new_independent_validation"]


def test_explicit_nonlinear_api_and_cached_policy_keep_default_linear():
    from fastapi.testclient import TestClient
    from aigc_detector.api import app
    from aigc_detector.engine import Detector
    from aigc_detector.fusion import ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY, load_profile, reapply_policy, resolve_policy
    profile, _ = load_profile(ENGLISH_NONLINEAR_POLICY)
    detector = Detector(device="cpu")
    def classify(texts, mid, *args, **kwargs):
        return [{"ai_score": .2, "revision": profile["revisions"][mid], "token_count": 30, "chunk_count": 1, "chunks": []} for _ in texts]
    with patch.object(detector, "_score_texts", side_effect=classify), patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        text = "An academic sentence containing a measured observation. "*40
        response = client.post("/v1/annotate", json={"text": text, "language": "en", "policy": ENGLISH_NONLINEAR_POLICY})
        assert response.status_code == 200, response.text
        report = response.json()
        assert reapply_policy(report)["policy"]["id"] == ENGLISH_NONLINEAR_POLICY
        assert reapply_policy(report)["summary"] == report["summary"]
        assert resolve_policy(None, "en") == ENGLISH_EXPANDED_POLICY
        for changed in [{"language": "zh"}, {"segment_chars": 800}, {"models": profile["models"][:1]}]:
            result = client.post("/v1/annotate", json={"text": text, "language": "en", "policy": ENGLISH_NONLINEAR_POLICY, **changed})
            assert result.status_code == 422
