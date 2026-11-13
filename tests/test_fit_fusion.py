"""Fitting invariants that protect the evidence, separate from runtime tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import fit_fusion_v2 as fitting


def test_logistic_optimizer_converges_with_decreasing_objective():
    x = np.column_stack([np.ones(6), [-2, -1, -.5, .5, 1, 2]])
    targets = np.array([0, 0, .2, .8, 1, 1])
    parameters, evidence = fitting.fit_head(x, targets, np.ones(6), [[0]], .1)
    assert evidence["converged"]
    assert evidence["final_gradient_inf_norm"] < 1e-6
    assert abs(parameters[0]) < 1e-8  # The balanced symmetric problem has zero intercept.
    predicted = fitting.sigmoid(x @ parameters)
    assert np.all(np.diff(predicted) > 0)
    assert np.allclose(predicted + predicted[::-1], 1)
    objectives = [row["objective"] for row in evidence["trace"]]
    assert all(b <= a + 1e-10 for a, b in zip(objectives, objectives[1:]))


def test_chapter_refits_purge_mixed_segments_regardless_of_midpoint_unit(monkeypatch):
    def record(name, identifier, units, target=0):
        return {"document": name, "id": identifier, "unit": units[-1], "chars": 100,
                "unit_memberships": [{"unit": unit, "characters": 100 // len(units)} for unit in units],
                "strong": 0, "possible": target}

    records = [record("thesis", "a", ["chapter-a"], .4),
               record("thesis", "mixed", ["chapter-a", "chapter-b"], .3),
               record("thesis", "b", ["chapter-b"], .5),
               record("wang-zhan", "human-a", ["one"]), record("zhu-yilong", "human-b", ["one"])]
    # Inspect the split before fitting; any cross-boundary label must not enter
    # either side of the chapter diagnostic.
    monkeypatch.setattr(fitting, "fit_strong", lambda rows, raw: {"training_ids": [r["id"] for r in rows]})
    folds = fitting.prepare_folds(records, np.zeros((len(records), 10)))
    chapter_folds = [fold for fold in folds if fold[0]["kind"] == "cnki_chapter"]
    assert len(chapter_folds) == 2
    for summary, train, test, train_rows, test_rows, _ in chapter_folds:
        held = summary["held_out"]
        assert not train[1] and not test[1]
        assert [r["id"] for r in summary["purged_mixed_segments"]] == ["mixed"]
        assert all(held not in fitting.record_units(r) for r in train_rows)
        assert all(fitting.record_units(r) == [held] for r in test_rows)
        assert len(test_rows) == 1


@pytest.fixture
def audited_dataset():
    path = Path(__file__).resolve().parents[1] / "evaluations/thesis-cnki-20260929-v2/score-dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_fitter_rejects_duplicated_records_before_weighting(audited_dataset):
    audited_dataset["records"].append(copy.deepcopy(audited_dataset["records"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        fitting.validate_dataset(audited_dataset)


def test_fitter_requires_chapter_memberships_and_contiguous_source_offsets(audited_dataset):
    no_memberships = copy.deepcopy(audited_dataset)
    no_memberships["records"][0].pop("unit_memberships")
    with pytest.raises(ValueError, match="memberships"):
        fitting.validate_dataset(no_memberships)
    audited_dataset["records"][0]["char_end"] += 1
    with pytest.raises(ValueError, match="offsets"):
        fitting.validate_dataset(audited_dataset)


def test_impossible_review_constraints_are_reported_instead_of_relaxed():
    records = [{"document": "thesis", "chars": 100, "strong": 0, "possible": .5},
               {"document": "wang-zhan", "chars": 100, "strong": 0, "possible": 0}]
    with pytest.raises(ValueError, match="zero human-control"):
        fitting.choose_threshold(records, np.array([.4, .5]), tier="possible")
