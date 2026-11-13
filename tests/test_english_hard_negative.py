"""Numerical and leakage tests for the bounded hard-negative experiment."""
import copy

import numpy as np
import pytest
import torch

from scripts import english_hard_negative as hn


def row(name, paper, score, ai=0, kind="mixed_document", start=0, end=10, sha=None):
    return {"id": name, "sample_id": paper + "-" + kind, "paper_id": paper, "scores": {"s": score},
            "ai_chars": ai, "kind": kind, "char_start": start, "char_end": end, "chars": end-start,
            "sha256": sha or name, "cross_paper_duplicate": False}


def fixture():
    rows, blocks = [], []
    for p in ("a", "b", "c"):
        rows += [row(p+"h", p, .15, kind="human_document"), row(p+"m", p, .7, ai=2)]
        blocks.append({"id": p+"block", "sample_id": p+"-mixed_document", "paper_id": p,
                       "char_start": 2, "char_end": 4, "window_ids": [p+"m"]})
    campaign = {"signals": ["s"], "bag_temperature": .1, "ranking_margin": .1,
                "optimizer": {"max_iter": 250, "max_eval": 400, "gradient_tolerance": 1e-8, "change_tolerance": 1e-12}}
    return {"records": rows, "blocks": blocks}, campaign


@pytest.mark.parametrize("method", hn.METHODS)
def test_autograd_matches_finite_difference_and_optimization(method):
    data, campaign = fixture()
    rows, bags, papers, _ = hn.training_view(data, {"a", "b"})
    z = np.array([[-.7], [1.2], [-.4], [.8]])
    config = {"method": method, "regularization": .1}
    loss = hn.make_loss(z, rows, bags, papers, config, campaign)
    point = torch.tensor([.31, -.21] if method == "presence" else [.31], dtype=torch.float64, requires_grad=True)
    analytic = torch.autograd.grad(loss(point), point)[0].numpy()
    numeric = []
    for i in range(len(point)):
        delta = torch.zeros_like(point)
        delta[i] = 1e-6
        numeric.append(float(((loss(point+delta)-loss(point-delta))/(2e-6)).detach()))
    assert np.allclose(analytic, numeric, atol=1e-8, rtol=1e-6)
    model = hn.fit_model(data, ["a", "b"], config, campaign)
    assert model["optimization"]["final_loss"] < model["optimization"]["initial_loss"]
    threshold = hn.cutoff(data["records"][:4], model, .01)
    result = hn.metrics(data, {"c"}, model, threshold)
    assert result["pure_human_false_windows"] == 0
    assert result["mixed_block_hits"] == 1


def test_fit_and_prediction_cannot_use_held_metadata_or_isolated_labels():
    data, campaign = fixture()
    config = {"method": "hard_human_bag_ranking", "regularization": .1}
    baseline = hn.fit_model(data, ["a", "b"], config, campaign)
    changed = copy.deepcopy(data)
    for r in changed["records"]:
        if r["paper_id"] == "c":
            r["scores"]["s"] = 1.
            r["ai_chars"] = 9
    changed["records"].append(row("isolated", "a", .999, ai=10, kind="ai_paragraph"))
    assert hn.fit_model(changed, ["a", "b"], config, campaign) == baseline
    scored = [{"scores": r["scores"]} for r in data["records"]]
    assert np.array_equal(hn.predict(data["records"], baseline), hn.predict(scored, baseline))


def test_dedup_keeps_all_bag_memberships():
    data, _ = fixture()
    duplicate = {**data["records"][1], "id": "duplicate"}
    data["records"].append(duplicate)
    data["blocks"].append({**data["blocks"][0], "id": "extra-block", "window_ids": ["duplicate"]})
    rows, bags, _, _ = hn.training_view(data, {"a"})
    assert len(rows) == 2 and len(bags) == 2 and bags[0] == bags[1]


def test_generated_intersection_is_positive_but_human_only_flag_is_false():
    data, _ = fixture()
    model = {"signals": ["s"], "center": [0.], "scale": [1.], "coefficients": [1.], "intercept": 0.}
    # Generated content is only two characters of a ten-character window.
    hit = hn.metrics(data, {"a"}, model, 0.)
    assert hit["mixed_block_hits"] == 1 and hit["pure_human_false_windows"] == 0
    assert hit["mixed_non_ai_chars_in_positive_windows"] == 8
    false = hn.metrics(data, {"a"}, model, -10.)
    assert false["pure_human_false_windows"] == 1 and false["original_false_documents"] == 1
    assert hn.overlap({"char_start": 0, "char_end": 2}, {"char_start": 2, "char_end": 4}) == 0


def test_calibration_positives_never_affect_cutoff():
    data, _ = fixture()
    model = {"signals": ["s"], "center": [0.], "scale": [1.], "coefficients": [1.], "intercept": 0.}
    before = hn.cutoff(data["records"], model, .01)
    for r in data["records"]:
        if r["ai_chars"]:
            r["scores"]["s"] = 1.
    assert hn.cutoff(data["records"], model, .01) == before
    human_values = hn.predict([r for r in data["records"] if not r["ai_chars"]], model)
    assert not (human_values >= before).any()


def test_smooth_hard_loss_targets_worst_negative():
    # One extreme negative, many easy negatives: hard ranking must penalize it.
    rows = [row("hard", "a", .9)] + [row(f"easy{i}", "a", .1) for i in range(20)] + [row("ai", "a", .8, ai=1)]
    z = np.array([[3.]] + [[-3.]]*20 + [[1.]])
    _, campaign = fixture()
    point = torch.tensor([1.], dtype=torch.float64)
    def value(method):
        return float(hn.make_loss(z, rows, [[21]], ["a"], {"method": method, "regularization": .1}, campaign)(point))
    assert value("hard_human_bag_ranking") > 10 * value("mean_human_bag_ranking")
