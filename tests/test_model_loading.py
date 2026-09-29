import json

import pytest
from safetensors.torch import load_file, save_file
from transformers import BertConfig, BertTokenizerFast

from aigc_detector.engine import Detector, ModelUnavailable
from aigc_detector.modeling import BertMlpClassifier


def test_incomplete_classifier_is_rejected_instead_of_randomly_initialized(tmp_path):
    directory = tmp_path / "ensemjudge-bert"
    directory.mkdir()
    vocab = directory / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n中\n文\n。\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(directory)
    config = BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32)
    config.classifier_hidden_size = 4
    config.classifier_hidden_dropout = 0.1
    BertMlpClassifier(config).save_pretrained(directory)
    weights = directory / "model.safetensors"
    state = load_file(weights)
    del state["classifier.3.bias"]
    save_file(state, weights, metadata={"format": "pt"})
    manifest = {"models": {"ensemjudge-bert": {"installed": True, "revision": "test",
                "files": {path.name: {"bytes": path.stat().st_size} for path in directory.iterdir()}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    detector = Detector(device="cpu", model_dir=tmp_path)
    with pytest.raises(ModelUnavailable, match="classifier.3.bias"):
        detector._load("ensemjudge-bert")
    assert detector._loaded == {}
