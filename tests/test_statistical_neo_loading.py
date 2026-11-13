"""GPT-Neo's old mask buffers are the only allowed unexpected keys."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aigc_detector.engine import ModelUnavailable
from aigc_detector.statistical import StatisticalDetector, checkpoint_problems


def model(model_type="gpt_neo"):
    value = SimpleNamespace(config=SimpleNamespace(model_type=model_type, num_layers=2), hf_device_map={"": "cpu"})
    value.parameters = lambda: iter(())
    value.eval = lambda: value
    return value


def test_only_exact_neo_mask_buffer_names_are_filtered():
    masks = [f"transformer.h.{index}.attn.attention.{name}" for index in range(2)
             for name in ("bias", "masked_bias")]
    learned = "transformer.h.0.attn.attention.out_proj.bias"
    outside = "transformer.h.2.attn.attention.bias"
    extra = "transformer.h.0.attn.attention.bias.extra"
    loading = {"unexpected_keys": [*masks, learned, outside, extra], "missing_keys": ["some.weight"],
               "mismatched_keys": [("other.weight", (1,), (2,))], "error_msgs": ["failure"]}
    problems, ignored = checkpoint_problems("gpt-neo-2.7b", model(), loading)
    assert ignored == sorted(masks)
    assert problems["unexpected_keys"] == sorted([learned, outside, extra])
    assert problems["missing_keys"] == ["some.weight"]
    assert problems["mismatched_keys"] and problems["error_msgs"]
    for asset, kind in (("falcon-7b", "gpt_neo"), ("gpt-neo-2.7b", "falcon")):
        result, ignored = checkpoint_problems(asset, model(kind), {"unexpected_keys": masks})
        assert not ignored
        assert result["unexpected_keys"] == sorted(masks)


def test_neo_loading_records_ignored_masks_and_rejects_learned_keys(tmp_path):
    detector = StatisticalDetector(device="cpu", model_dir=tmp_path)
    old_mask = "transformer.h.0.attn.attention.bias"
    loading = {"unexpected_keys": [old_mask], "missing_keys": [], "mismatched_keys": [], "error_msgs": []}
    with patch("aigc_detector.statistical.AutoModelForCausalLM.from_pretrained", return_value=(model(), loading)):
        detector._load("gpt-neo-2.7b")
    assert detector.last_loads[0]["ignored_legacy_buffer_keys"] == [old_mask]
    assert detector.last_loads[0]["loading_problems"] == {}
    loading["unexpected_keys"].append("transformer.h.0.attn.attention.out_proj.bias")
    with patch("aigc_detector.statistical.AutoModelForCausalLM.from_pretrained", return_value=(model(), loading)), \
            pytest.raises(ModelUnavailable, match="out_proj.bias"):
        detector._load("gpt-neo-2.7b")
