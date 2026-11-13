"""Protect coauthoring label semantics and strict, local-only model adaptation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import ModernBertConfig

from aigc_detector.api import app
from aigc_detector.catalog import CATALOG, ENGLISH_MODELS, SPECIALIZED_ENGLISH_MODELS
from aigc_detector.engine import Detector, ModelUnavailable, prepare_window, select_models
from aigc_detector.fusion import ENGLISH_EXPANDED_POLICY, load_profile
from aigc_detector.modeling import CoCoDetModel
from scripts.download_models import model_assets


class Batch(dict):
    def to(self, device):
        return self


@pytest.mark.parametrize("mid,probabilities,expected", [
    ("cocodet-en", [[.1, .8, .1], [.7, .2, .1]], [.9, .3]),
    ("llm-detectaive-en", [[.1, .7, .1, .1], [.6, .1, .2, .1]], [.9, .4]),
])
def test_mixed_and_polished_classes_are_not_silently_discarded(tmp_path, mid, probabilities, expected):
    detector = Detector(device="cpu", model_dir=tmp_path)
    tokenizer = SimpleNamespace(pad=lambda features, **kwargs: Batch())
    model = lambda **kwargs: SimpleNamespace(logits=torch.tensor(probabilities).log())
    with patch.dict(CATALOG[mid], {"inference_batch_size": 2}):
        rows = list(detector._probabilities([{}, {}], tokenizer, model, mid, include_classes=True))
    assert [r["ai_score"] for r in rows] == pytest.approx(expected)
    for row, wanted in zip(rows, probabilities):
        assert list(row["class_scores"].values()) == pytest.approx(wanted)
    with pytest.raises(ModelUnavailable, match="shape"):
        list(detector._probabilities([{}], tokenizer, model, mid))
    invalid = lambda **kwargs: SimpleNamespace(logits=torch.full_like(torch.tensor(probabilities), float("nan")))
    with patch.dict(CATALOG[mid], {"inference_batch_size": 2}), pytest.raises(RuntimeError, match="Non-finite"):
        list(detector._probabilities([{}, {}], tokenizer, invalid, mid))


def test_multiclass_windows_and_aggregation_preserve_every_class(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    spec = CATALOG["cocodet-en"]
    class Tokenizer:
        cls_token_id, sep_token_id = 30, 31
        def num_special_tokens_to_add(self, **kwargs):
            return 2
        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(range(len(t))) for t in texts],
                    "offset_mapping": [[(i, i + 1) for i in range(len(t))] for t in texts]}
        def prepare_for_model(self, ids, **kwargs):
            return {"input_ids": ids}
        def pad(self, features, **kwargs):
            return Batch()
    probabilities = torch.tensor([[.1, .8, .1], [.6, .1, .3]])
    model = lambda **kwargs: SimpleNamespace(logits=probabilities.log())
    # The two overlapping windows [0,4),[2,6) each have coverage weight 3.
    with patch.object(detector, "_load", return_value=(Tokenizer(), model, 6)), \
            patch.object(detector, "_manifest", return_value={"cocodet-en": {"revision": "fixture"}}), \
            patch.dict(spec, {"inference_batch_size": 2}):
        result = detector._score_texts(["abcdef"], "cocodet-en", 2, True)[0]
        empty = detector._score_texts([""], "cocodet-en", 2, False, allow_empty=True)[0]
    assert result["ai_score"] == pytest.approx(.65)
    assert result["class_scores"] == pytest.approx({"human": .35, "mix": .45, "ai": .2})
    assert result["chunks"][1]["class_scores"]["human"] == pytest.approx(.6)
    assert result["ai_score_classes"] == ["mix", "ai"]
    assert empty["class_scores"] == {"human": None, "mix": None, "ai": None}


def tiny_config():
    return ModernBertConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                            num_attention_heads=2, num_hidden_layers=1,
                            max_position_embeddings=64, pad_token_id=0,
                            cls_token_id=1, sep_token_id=2, global_attn_every_n_layers=1)


def test_cocodet_cosine_scale_and_strict_checkpoint(tmp_path):
    model = CoCoDetModel(tiny_config()).eval()
    model.config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    restored = CoCoDetModel.from_local(tmp_path).eval()
    tokens = {"input_ids": torch.tensor([[1, 5, 2]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with torch.inference_mode():
        assert torch.allclose(model(**tokens).logits, restored(**tokens).logits)
    fake_hidden = torch.zeros(1, 3, 8)
    fake_hidden[0, 0, :2] = torch.tensor([3., 4.])
    with torch.no_grad():
        model.main_classifier_weight.zero_()
        model.main_classifier_weight[0, 0] = 2.
        model.main_classifier_weight[1, 1] = 9.
        model.main_classifier_weight[2, :2] = torch.tensor([-3., -4.])
    with patch.object(model.bert, "forward", return_value=SimpleNamespace(last_hidden_state=fake_hidden)):
        assert model(**tokens).logits.tolist()[0] == pytest.approx([18., 24., -30.])
    state = restored.state_dict()
    del state["style_classifier.bias"]
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="Missing key"):
        CoCoDetModel.from_local(tmp_path)


def test_checkpoint_selection_alias_and_no_repository_code():
    selected = model_assets({"siblings": [{"rfilename": n} for n in
                            ["README.md", "CoCoDet.bin", "evil.py", "optimizer.pt"]]}, weight_file="CoCoDet.bin")
    assert dict(selected)["pytorch_model.bin"]["rfilename"] == "CoCoDet.bin"
    assert set(dict(selected)) == {"README.md", "pytorch_model.bin"}
    selected = model_assets({"siblings": [{"rfilename": n} for n in
                            ["app.py", "deberta-base/config.json", "deberta-base/model.safetensors",
                             "other/model.safetensors"]]}, "deberta-base")
    assert set(dict(selected)) == {"config.json", "model.safetensors"}


def test_generic_modernbert_tokenizer_template_is_applied_to_each_window():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "text": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", unk_token="[UNK]")
    ids = tokenizer("text text", add_special_tokens=False)["input_ids"]
    prepared = prepare_window(tokenizer, ids, "cocodet-en")
    assert prepared["input_ids"] == tokenizer("text text")["input_ids"] == [1, 4, 4, 2]
    assert prepared["attention_mask"] == [1, 1, 1, 1]


def test_new_roster_is_explicit_and_defaults_stay_frozen(tmp_path):
    assert not set(SPECIALIZED_ENGLISH_MODELS) & set(select_models(None, "en"))
    assert select_models(None, "en") == list(ENGLISH_MODELS)
    profile, _ = load_profile(ENGLISH_EXPANDED_POLICY)
    assert not set(profile["models"]) & set(SPECIALIZED_ENGLISH_MODELS)
    detector = Detector(device="cpu", model_dir=tmp_path)
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        listed = client.get("/v1/models").json()
        assert listed["specialized_english_models"] == list(SPECIALIZED_ENGLISH_MODELS)
        assert len(listed["models"]) == 22
        response = client.post("/v1/detect", json={"text": "English test", "language": "en", "models": ["cocodet-en"]})
        assert response.status_code == 503
