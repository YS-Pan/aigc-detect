import hashlib
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aigc_detector.annotation import decide, segment_spans, validate_policy
from aigc_detector.api import app
from aigc_detector.engine import Detector
from aigc_detector.reporting import save_annotation


@pytest.mark.parametrize("text", [
    "第一段。\r\n第二段！" * 110,
    "字" * 100000,
    "🙂e\u0301中文？\r\n" * 100 + "末尾",
    "\r\n" + "a" * 127 + "\r\n" + "末" * 500,
    "短文。",
], ids=["crlf_paragraphs", "100k_unbroken", "unicode_combining", "newline_edge", "short_text"])
def test_segmentation_preserves_every_character_and_newline(text):
    spans = segment_spans(text, 400, 120)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert all(0 < end - start <= 400 for start, end in spans)
    assert "".join(text[start:end] for start, end in spans) == text
    assert all(text[end - 1:end + 1] != "\r\n" for _, end in spans)


@pytest.mark.parametrize("scores,expected,triggered", [
    ({"mpu": .99, "hc3": .01}, "strong_signal", ["mpu"]),
    ({"mpu": .20, "hc3": .90}, "strong_signal", ["hc3"]),
    ({"mpu": .70, "hc3": .20}, "possible_signal", ["mpu"]),
    ({"mpu": .699, "hc3": .05}, "other", []),
    ({"mpu": None, "hc3": None}, "other", []),
])
def test_sensitive_rule_and_boundary_values(scores, expected, triggered):
    result = decide(scores, .9, .7)
    assert result["category"] == expected
    assert result["triggered_models"] == triggered
    if scores.get("mpu") == .99:
        assert result["model_disagreement"]


@pytest.mark.parametrize("high,possible", [(.7, .7), (.6, .7), (float("nan"), .5), (1.1, .5), (.9, -1), (True, .5)])
def test_invalid_policy_is_rejected(high, possible):
    with pytest.raises(ValueError):
        validate_policy(high, possible, 400, 120)


def minimal_report(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    decision = decide({"hc3-zh": .5}, .9, .7)
    return {"schema_version": "aigc-annotation/1.0", "status": "complete",
            "input": {"characters": len(text), "sha256": digest},
            "policy": {"labels": {"strong_signal": "明显 AIGC 特征", "possible_signal": "较可能 AIGC", "other": "其他"}},
            "segments": [{"index": 0, "id": "s000001", "char_start": 0, "char_end": len(text),
                          "line_start": 1, "line_end": 2, "sha256": digest, "decision": decision,
                          "model_scores": {"hc3-zh": {"ai_score": .5, "category": "other"}}, "warnings": []}]}


def test_saved_bundle_is_lossless_and_html_treats_input_as_data(tmp_path):
    text = "中文🙂\r\n</script><script>alert('source')</script>&末尾"
    report = minimal_report(text)
    directory = tmp_path / "bundle"
    save_annotation(report, text, directory)
    assert (directory / "source.txt").read_bytes() == text.encode("utf-8")
    restored = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert "".join(s["text"] for s in restored["segments"]) == text
    lines = [json.loads(line) for line in (directory / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["text"] == text
    html = (directory / "report.html").read_text(encoding="utf-8")
    assert "<script>alert('source')" not in html
    assert "\\u003c/script\\u003e" in html
    assert json.loads((directory / "COMPLETE.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        save_annotation(report, text, directory)
    with pytest.raises(ValueError):
        save_annotation(report, text + "改", tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()


def test_annotation_contract_rejects_bad_options_before_loading(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        for payload in [
            {"text": " "}, {"text": "文本", "models": []}, {"text": "文本", "models": ["unknown"]},
            {"text": "文本", "high_threshold": .6, "possible_threshold": .7},
            {"text": "文本", "segment_chars": 200, "min_segment_chars": 201},
            {"text": "文本", "segment_chars": 100}, {"text": "文本", "high_threshold": True},
            {"text": "文本", "output_dir": "arbitrary"}, {"text": "x" * 200001},
        ]:
            response = client.post("/v1/annotate", json=payload)
            assert response.status_code == 422, response.text
        assert not detector._loaded
        assert client.post("/v1/annotate", json={"text": "缺少模型", "models": ["hc3-zh"], "policy": "sensitive-any-model-v1"}).status_code == 503
