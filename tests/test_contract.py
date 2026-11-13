import pytest
from fastapi.testclient import TestClient

from aigc_detector.api import app
from aigc_detector.engine import Detector, coverage_weights, window_ranges


@pytest.mark.parametrize("total,overlap", [(1, 0), (510, 64), (511, 64), (1020, 0), (3001, 256)])
def test_complete_coverage_without_double_weighting(total, overlap):
    ranges = list(window_ranges(total, 510, overlap))
    covered = {index for start, end in ranges for index in range(start, end)}
    assert covered == set(range(total))
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(0 < end - start <= 510 for start, end in ranges)
    assert sum(coverage_weights(ranges, total)) == pytest.approx(total)


def test_api_rejects_invalid_input_before_loading_models(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    app.dependency_overrides.clear()
    # get_detector is intentionally a simple singleton rather than a DI dependency.
    from unittest.mock import patch
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        invalid = [{"text": " "}, {"text": "中文", "models": []}, {"text": "中文", "models": ["unknown"]},
                   {"text": "中文", "overlap_tokens": -1}, {"text": "中文", "overlap_tokens": 999},
                   {"text": "中文", "unexpected": True}, {"text": "a" * 200001}]
        for payload in invalid:
            assert client.post("/v1/detect", json=payload).status_code == 422
        assert client.post("/v1/detect-batch", json={"texts": []}).status_code == 422
        assert client.post("/v1/detect-batch", json={"texts": ["  "]}).status_code == 422
        assert client.post("/v1/detect-batch", json={"texts": ["x" * 180000] * 3}).status_code == 422
        assert detector._loaded == {}
        assert client.get("/health").json()["status"] == "models_missing"
        assert client.post("/v1/detect", json={"text": "测试缺失模型", "models": ["hc3-zh"]}).status_code == 503
        from aigc_detector.catalog import CATALOG
        # Expanded lists must reach the loader, instead of the old four-ID cap.
        for route in ("/v1/detect", "/v1/annotate"):
            options = {"policy": "sensitive-any-model-v1"} if route.endswith("annotate") else {}
            assert client.post(route, json={"text": "测试新增模型列表", "models": list(CATALOG), **options}).status_code == 503


def test_python_contract_rejects_string_model_list(tmp_path):
    detector = Detector(device="cpu", model_dir=tmp_path)
    with pytest.raises(ValueError, match="models"):
        detector.detect("输入中文", models="hc3-zh")
    with pytest.raises(ValueError, match="Unknown"):
        detector.detect("输入中文", models=["bogus"])
    with pytest.raises(ValueError, match="overlap"):
        detector.detect("输入中文", overlap_tokens=True)
