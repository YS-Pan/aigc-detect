from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import DebertaV2Config

from aigc_detector.annotation import segment_spans
from aigc_detector.api import app
from aigc_detector.catalog import CATALOG, CHINESE_MODELS, ENGLISH_MODELS
from aigc_detector.engine import Detector, ModelUnavailable, language_warnings, select_models
from aigc_detector.fusion import LEGACY_POLICY, LINEAR_POLICY, REFERENCE_POLICY, load_profile, reapply_policy
from aigc_detector.modeling import DesklibAIDetectionModel
from scripts.download_models import model_assets


def test_default_chinese_roster_and_reference_profiles_stay_at_ten():
    assert len(CATALOG) == 22 and len(CHINESE_MODELS) == 10
    assert select_models(None) == list(CHINESE_MODELS)
    assert select_models(None, "en") == list(ENGLISH_MODELS)
    assert select_models(["fakespot-en", "hc3-en", "fakespot-en"]) == ["fakespot-en", "hc3-en"]
    for policy in (LINEAR_POLICY, REFERENCE_POLICY):
        profile, _ = load_profile(policy)
        assert set(profile["models"]) == set(CHINESE_MODELS)


@pytest.mark.parametrize("language", ["auto", "fr", "EN", None, ["en"]])
def test_unsupported_language_is_rejected_even_with_explicit_models(language):
    with pytest.raises(ValueError, match="language"):
        select_models(["hc3-en"], language)


def test_language_warning_uses_selected_models():
    assert language_warnings("English prose with ordinary words.", ENGLISH_MODELS) == []
    assert language_warnings("English prose.", CHINESE_MODELS) == ["little_chinese_text"]
    assert language_warnings("这是一段中文正文。", ENGLISH_MODELS) == ["english_models_on_chinese_text"]
    assert language_warnings("English prose.", ["bibbbu-xlm-roberta"]) == []


@pytest.mark.parametrize("text", [
    ("An English sentence ending here. " * 200) + "Final sentence.",
    ("word " * 301) + "last",
    "🙂e\u0301 text!\r\n" * 150,
    "x" * 1700,
])
def test_english_segmentation_preserves_full_source(text):
    spans = segment_spans(text, language="en")
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(0 < b - a <= 400 for a, b in spans)
    assert all(text[b - 1:b + 1] != "\r\n" for _, b in spans)
    assert "".join(text[a:b] for a, b in spans) == text


def test_english_sentence_end_is_preferred_without_changing_chinese_splitter():
    text = "A" * 250 + ". " + "B" * 500
    assert segment_spans(text, language="en")[0][1] == 251
    assert segment_spans(text)[0][1] == 400


def test_http_language_and_batch_routes_select_only_english(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    with patch.object(detector, "_score", side_effect=lambda text, mid, *args: {"model_id": mid}), \
            patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        response = client.post("/v1/detect", json={"text": "An English input.", "language": "en"})
        assert response.status_code == 200
        assert [r["model_id"] for r in response.json()["results"]] == list(ENGLISH_MODELS)
        assert "little_chinese_text" not in response.json()["warnings"]
        batch = client.post("/v1/detect-batch", json={"texts": ["First input.", "Second input."], "language": "en"})
        assert batch.status_code == 200
        assert all([r["model_id"] for r in item["results"]] == list(ENGLISH_MODELS) for item in batch.json()["items"])
        assert client.post("/v1/detect", json={"text": "Input.", "language": "fr"}).status_code == 422
        assert client.post("/v1/annotate", json={"text": "English input.", "language": "en"}).status_code == 503
        assert not detector._loaded


def test_missing_optional_english_models_do_not_break_chinese_health(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    listed = [{"id": m, "installed": m in CHINESE_MODELS} for m in CATALOG]
    with patch.object(detector, "list_models", return_value=listed), \
            patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_groups"]["zh"]["ready"]
        assert not health["model_groups"]["en"]["ready"]


def test_english_annotation_requires_explicit_uncalibrated_policy(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    text = "A sentence of English text. " * 80
    def score(passages, mid, *args, **kwargs):
        return [{"ai_score": .2, "token_count": len(p), "chunk_count": 1, "chunks": [], "revision": "fixture"} for p in passages]
    with patch.object(detector, "_score_texts", side_effect=score):
        report = detector.annotate(text, language="en", policy=LEGACY_POLICY)
    assert [m["id"] for m in report["models"]] == list(ENGLISH_MODELS)
    assert report["segmentation"]["id"] == "english-natural-boundaries-v1"
    assert "".join(s["text"] for s in report["segments"]) == text
    assert not report["policy"]["reference_fitted"]
    assert all("little_chinese_text" not in s["warnings"] for s in report["segments"])


@pytest.mark.parametrize("policy", [LINEAR_POLICY, REFERENCE_POLICY])
def test_chinese_fusion_cannot_recombine_english_boundaries_even_with_chinese_roster(policy):
    profile, _ = load_profile(policy)
    report = {"status": "complete", "segments": [{"model_scores": {}}],
              "models": [{"id": m, "revision": profile["revisions"][m]} for m in CHINESE_MODELS],
              "segmentation": {"id": "english-natural-boundaries-v1", "max_characters": 400,
                               "minimum_target_characters": 120}}
    with pytest.raises(ValueError, match="segmentation"):
        reapply_policy(report, policy=policy)


def test_desklib_masked_pooling_and_local_checkpoint_roundtrip(tmp_path):
    config = DebertaV2Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                            num_attention_heads=2, num_hidden_layers=1,
                            max_position_embeddings=512, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    model = DesklibAIDetectionModel(config).eval()
    inputs = {"input_ids": torch.tensor([[1, 5, 2, 0], [1, 6, 7, 2]]),
              "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
    with torch.inference_mode():
        hidden = model.model(**inputs).last_hidden_state
        pooled = torch.stack([hidden[0, :3].mean(0), hidden[1].mean(0)])
        expected = torch.nn.functional.linear(pooled, model.classifier.weight, model.classifier.bias)
        assert torch.allclose(model(**inputs).logits, expected, atol=1e-7)
    model.save_pretrained(tmp_path)
    restored, loading = DesklibAIDetectionModel.from_pretrained(tmp_path, local_files_only=True,
                                                              weights_only=True, output_loading_info=True)
    assert not any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert torch.allclose(restored(**inputs).logits, expected, atol=1e-7)


def test_sigmoid_score_and_invalid_output_shape(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    class Batch(dict):
        def to(self, device):
            return self
    tokenizer = SimpleNamespace(pad=lambda features, **kwargs: Batch())
    model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[2.0], [-2.0]]))
    values = list(detector._probabilities([{}, {}], tokenizer, model, "desklib-en"))
    assert values == pytest.approx([.88079708, .11920292])
    with pytest.raises(ModelUnavailable, match="shape"):
        list(detector._probabilities([{}], tokenizer, model, "desklib-en"))


def test_download_selects_local_tokenizer_assets_without_repository_code():
    names = ["config.json", "pytorch_model.bin", "model.safetensors", "spm.model", "vocab.json", "merges.txt", "utils.py"]
    selected = {name for name, _ in model_assets({"siblings": [{"rfilename": n} for n in names]})}
    assert {"config.json", "model.safetensors", "spm.model", "vocab.json", "merges.txt"} <= selected
    assert "pytorch_model.bin" not in selected and "utils.py" not in selected
