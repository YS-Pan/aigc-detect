import copy
import numpy as np
import pytest

from scripts import english_specialized_precision as sp


def test_tiny_probabilities_and_endpoint_values_are_retained():
    values = [0., 1e-25, 1e-13, 2e-6, 4e-6, .1, .9, 1.]
    rows = [{"scores": {"m": p}} for p in values]
    actual = sp.raw_matrix(rows, ["m"], "native_logit")[:, 0]
    assert np.isfinite(actual).all() and np.all(np.diff(actual) > 0)
    assert np.array_equal(sp.raw_matrix(rows, ["m"], "raw_probability")[:, 0], values)
    config = {"method": "single", "signals": ["m"], "transform": "native_logit"}
    model = sp.fit_model({}, [], config, {})
    assert np.array_equal(sp.predict(rows, model), actual)


@pytest.mark.parametrize("value", [np.nan, np.inf, -.1, 1.01])
def test_invalid_probability_is_rejected(value):
    with pytest.raises(ValueError):
        sp.raw_matrix([{"scores": {"m": value}}], ["m"], "native_logit")


def test_prediction_transform_is_pinned_to_the_fitted_model():
    model = {"transform": "native_logit", "signals": ["m"], "center": [0.], "scale": [1.],
             "clip_standardized": 70., "coefficients": [1.], "intercept": 0.}
    rows = [{"scores": {"m": 2e-6}}]
    before = sp.predict(rows, model)
    model["transform"] = "raw_probability"
    assert sp.predict(rows, model)[0] == 2e-6
    assert before[0] < -10


@pytest.mark.parametrize("transform", ["native_logit", "raw_probability"])
def test_precision_fitting_preserves_tiny_signal_and_excludes_held_rows(transform):
    rows, blocks = [], []
    for p in ("a", "b", "held"):
        for positive in (False, True):
            rid = p + str(positive)
            rows.append({"id": rid, "sample_id": rid, "paper_id": p, "kind": "mixed_document",
                         "sha256": rid, "cross_paper_duplicate": False, "ai_chars": int(positive),
                         "scores": {"m": 4e-6 if positive else 2e-6}})
        blocks.append({"paper_id": p, "window_ids": [p + "True"]})
    data = {"records": rows, "blocks": blocks}
    campaign = {"bag_temperature": .1, "ranking_margin": .1,
                "optimizer": {"max_iter": 250, "max_eval": 400, "gradient_tolerance": 1e-8, "change_tolerance": 1e-12}}
    config = {"method": "presence", "signals": ["m"], "regularization": .1, "transform": transform}
    before = sp.fit_model(data, ["a", "b"], config, campaign)
    assert sp.predict([rows[1]], before)[0] > sp.predict([rows[0]], before)[0]
    changed = copy.deepcopy(data)
    for row in changed["records"]:
        if row["paper_id"] == "held":
            row["scores"]["m"] = 1.
    assert sp.fit_model(changed, ["a", "b"], config, campaign) == before
