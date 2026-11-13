"""Real-model annotation validation on a 100,000-character derived fixture."""

import json
import math
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aigc_detector.catalog import ROOT
from aigc_detector.engine import Detector
from aigc_detector.reporting import save_annotation


def main():
    import torch
    from fastapi.testclient import TestClient
    from aigc_detector import api
    samples = [json.loads(line) for line in (ROOT / "samples/smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    answers = [s for s in samples if s["kind"] == "official_dataset"]
    unit = "\r\n\r\n".join(s["text"] for s in answers) + "\r\n\r\n"
    text = (unit * (100000 // len(unit) + 1))[:100000]
    detector = Detector()

    def no_network(*args, **kwargs):
        raise AssertionError("Annotation inference attempted a network connection")

    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "create_connection", no_network):
        started = time.perf_counter()
        report = detector.annotate(text)
        elapsed = time.perf_counter() - started
    assert report["status"] == "complete"
    assert len(text) == 100000 and "".join(s["text"] for s in report["segments"]) == text
    assert sum(s["char_end"] - s["char_start"] for s in report["segments"]) == len(text)
    assert sum(v["characters"] for v in report["summary"].values()) == len(text)
    for segment in report["segments"]:
        assert segment["char_end"] > segment["char_start"]
        for value in segment["model_scores"].values():
            assert value["ai_score"] is not None and math.isfinite(value["ai_score"])
            assert 0 <= value["ai_score"] <= 1
            assert value["windows"][-1]["token_end"] == value["token_count"]
            assert all(segment["char_start"] <= window["char_start"] < window["char_end"] <= segment["char_end"] for window in value["windows"])
        decision = segment["decision"]
        joint = decision["ensemble_score"]
        review = decision["review_score"]
        assert joint is not None and math.isfinite(joint) and 0 <= joint <= 1
        assert review is not None and math.isfinite(review) and 0 <= review <= 1
        expected = ("strong_signal" if joint >= report["policy"]["high_threshold"] else
                    "possible_signal" if review >= report["policy"]["possible_threshold"] else "other")
        assert decision["category"] == expected
    # Independent calls should agree with cross-passage GPU batching.
    max_difference = 0.0
    for segment in report["segments"][:3]:
        independent = detector.detect(segment["text"])
        for model in independent["results"]:
            difference = abs(model["ai_score"] - segment["model_scores"][model["model_id"]]["ai_score"])
            max_difference = max(max_difference, difference)
            assert difference < 1e-4
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(api.app) as client:
        response = client.post("/v1/annotate", json={"text": answers[0]["text"], "models": ["hc3-zh"], "include_text": False,
                                                   "policy": "sensitive-any-model-v1"})
        assert response.status_code == 200, response.text
        assert "text" not in response.json()["segments"][0]
    bundle = ROOT / "reports" / ("annotation-100k-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    paths = save_annotation(report, text, bundle)
    provenance = {"purpose": "engineering_validation_only_not_accuracy_or_threshold_calibration",
                  "transformation": "Join six official HC3 answers with CRLF blank lines, repeat, then take the first 100000 Unicode code points.",
                  "source_samples": [s["id"] for s in answers], "source_urls": sorted({s["source"] for s in answers}),
                  "license": "CC-BY-SA-4.0", "attribution": "Biyang Guo et al., HC3 (2023)"}
    (bundle / "fixture-provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
              "characters": len(text), "segments": len(report["segments"]), "models": len(report["models"]),
              "model_segment_runs": len(report["segments"]) * len(report["models"]),
              "elapsed_seconds_cold": round(elapsed, 3), "device": detector.device,
              "max_batch_vs_independent_difference": max_difference,
              "offline_check": "passed", "full_coverage": "passed", "http_in_process": "passed",
              "bundle": paths, "summary": report["summary"]}
    if detector.device == "cuda":
        result["gpu"] = torch.cuda.get_device_name(0)
        result["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    (ROOT / "reports/annotation-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
