"""Independent input contexts and bounded model reuse for statistical batches."""
from unittest.mock import patch

import pytest
import torch

from aigc_detector.catalog import CAUSAL_ASSETS
from aigc_detector.statistical import StatisticalDetector, distribution_components, raw_statistic


class Tokenizer:
    pad_token_id = 99
    eos_token_id = 98
    all_special_ids = [98, 99]

    def get_vocab(self):
        return {str(i): i for i in range(7)}

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(c) % 7 for c in text] for text in texts],
                "offset_mapping": [[(i, i+1) for i in range(len(text))] for text in texts]}


def fake_logits(asset, ids):
    base = torch.arange(len(ids)*7, dtype=torch.float32).reshape(len(ids), 7)
    return torch.sin(base*.13 + torch.tensor(ids)[:, None]*.3 + (asset == "falcon-7b-instruct")*.7)


def test_batch_reuses_three_loads_without_sharing_input_contexts_or_components(tmp_path):
    detector = StatisticalDetector(device="cpu")
    manifest = {a: {"revision": "fixture"} for a in CAUSAL_ASSETS}
    calls = []

    def cache(asset, windows, temporary, **kwargs):
        calls.append(asset)
        detector.last_loads.append({"asset": asset, "execution_device": "cpu"})
        paths = []
        for i, ids in enumerate(windows):
            if kwargs.get("consume") is not None:
                kwargs["consume"](i, fake_logits(asset, ids))
            else:
                path = temporary / f"{asset}-{i}.pt"
                torch.save(fake_logits(asset, ids), path)
                paths.append(path)
        return paths

    methods = ["fast-detect-gpt-neo", "fast-detect-gpt-falcon", "binoculars-falcon"]
    texts = ["abcdefghij" * 7, "uvwxyzabcdef"]
    with patch("aigc_detector.statistical.state_dir", return_value=tmp_path), \
            patch.object(detector, "_tokenizer", return_value=Tokenizer()), \
            patch.object(detector, "_manifest", return_value=manifest), \
            patch.object(detector, "_cache_logits", side_effect=cache):
        batch = detector.detect_batch(texts, methods, max_tokens=32, overlap_tokens=8)
        assert calls == ["gpt-neo-2.7b", "falcon-7b", "falcon-7b-instruct"]
        single = [detector.detect(t, methods, max_tokens=32, overlap_tokens=8) for t in texts]
    for actual, expected in zip(batch, single):
        assert actual["input"] == expected["input"]
        assert actual["results"] == expected["results"]
        assert all(r["coverage_complete"] and not r["truncated"] for r in actual["results"])
    # The short input has one window; compare all three raw values directly
    # to the tensor formula, outside of any batch aggregation implementation.
    ids = [ord(c) % 7 for c in texts[1]]
    for row in batch[1]["results"]:
        reference = "gpt-neo-2.7b" if row["backend_id"].endswith("neo") else "falcon-7b"
        scoring = reference if reference.startswith("gpt") else "falcon-7b-instruct"
        values = distribution_components(fake_logits(reference, ids), fake_logits(scoring, ids), ids)
        direct = raw_statistic(row["backend_id"], values, torch.ones(len(ids)-1), torch.ones(len(ids)), ids, 99)
        assert row["raw_score"] == pytest.approx(direct, abs=1e-10)
    assert not list((tmp_path / "statistical-logits").iterdir())


@pytest.mark.parametrize("texts", [[], "text", [""], [None], ["x"]*1025, ["x"*199999]*3])
def test_invalid_batch_is_rejected_before_loading_any_weight(texts):
    detector = StatisticalDetector(device="cpu")
    with patch.object(detector, "_load") as load, pytest.raises(ValueError):
        detector.detect_batch(texts)
    load.assert_not_called()
