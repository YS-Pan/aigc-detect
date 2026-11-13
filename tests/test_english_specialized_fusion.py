"""Coverage for class semantics, locked split boundaries and new controls."""
import copy

import numpy as np
import pytest

from scripts import english_specialized_fusion as sf
from scripts import english_hard_negative as hn


@pytest.mark.parametrize("mid,classes", [
    ("llm-detectaive-en", {"human": .4, "human_machine_polished": .3, "machine": .2, "machine_machine_humanized": .1}),
    ("cocodet-en", {"human": .4, "mix": .35, "ai": .25}),
])
def test_class_signals_preserve_semantics_and_reject_inconsistent_sum(mid, classes):
    values = sf.new_scores(mid, {"ai_score": .6, "class_scores": classes})
    for label, value in classes.items():
        if label != "human":
            assert values[mid + "::" + label] == value
    with pytest.raises(AssertionError):
        sf.new_scores(mid, {"ai_score": .7, "class_scores": classes})


def test_single_signal_preserves_extreme_logits_and_uses_no_metadata():
    config = {"method": "single", "signals": ["x"]}
    model = sf.fit_model({}, ["a"], config, {})
    probabilities = np.array([0., .0001, .1, .5, .9, .9999, 1.])
    rows = [{"scores": {"x": p}} for p in probabilities]
    bounded = np.clip(probabilities, 1e-5, 1-1e-5)
    assert np.allclose(hn.predict(rows, model), np.log(bounded/(1-bounded)), atol=1e-14)


def test_expanded_class_fit_excludes_held_and_isolated_data():
    rows, blocks = [], []
    for p in ("a", "b", "held"):
        for positive in (False, True):
            rid = p + str(positive)
            rows.append({"id": rid, "sample_id": rid, "paper_id": p, "kind": "mixed_document",
                         "sha256": rid, "cross_paper_duplicate": False, "ai_chars": int(positive),
                         "scores": {"m::mix": .8 if positive else .1, "m::ai": .1 if positive else .4}})
        blocks.append({"paper_id": p, "window_ids": [p + "True"]})
    data = {"records": rows, "blocks": blocks}
    config = {"method": "presence", "signals": ["m::mix", "m::ai"], "regularization": .1}
    campaign = {"bag_temperature": .1, "ranking_margin": .1,
                "optimizer": {"max_iter": 250, "max_eval": 400, "gradient_tolerance": 1e-8, "change_tolerance": 1e-12}}
    before = sf.fit_model(data, ["a", "b"], config, campaign)
    changed = copy.deepcopy(data)
    for row in changed["records"]:
        if row["paper_id"] == "held":
            row["scores"] = {"m::mix": 1., "m::ai": 1.}
            row["ai_chars"] = 0
    changed["records"].append({**rows[0], "kind": "ai_paragraph", "ai_chars": 100, "scores": {"m::mix": 1., "m::ai": 1.}})
    assert sf.fit_model(changed, ["a", "b"], config, campaign) == before
    assert np.array_equal(hn.predict(rows, before), hn.predict([{"scores": r["scores"]} for r in rows], before))


def test_no_false_positive_candidate_wins_even_with_lower_recall():
    def candidate(false, hits):
        return {"metrics": {"pure_human_false_windows": false, "mixed_block_hits": hits,
                            "original_false_documents": 0, "mixed_ai_hit_chars": hits * 10},
                "configuration": {"signals": ["x"], "regularization": 1., "id": "x"}}
    assert sf.objective(candidate(0, 1)) < sf.objective(candidate(1, 100))


def test_frozen_repeat_requires_unchanged_data_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "EVIDENCE", tmp_path)
    monkeypatch.setattr(sf, "verify_campaign", lambda: {"verified": True})
    path = tmp_path / "development-data.json"
    sf.write(path, {"a": 1})
    sf.write(tmp_path / "FREEZE.json", {"hashes": {path.name: sf.sha(path)}})
    assert sf.verify_freeze() == {"verified": True}
    sf.write(path, {"a": 2})
    with pytest.raises(AssertionError):
        sf.verify_freeze()
