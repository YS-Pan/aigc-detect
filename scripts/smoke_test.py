"""Real-weight smoke checks. This deliberately calculates no accuracy/threshold."""

import argparse
import json
import math
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.catalog import CATALOG, ROOT
from aigc_detector.engine import Detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/smoke-test.json")
    args = parser.parse_args()
    import torch
    import transformers
    from fastapi.testclient import TestClient
    from aigc_detector import api

    samples = [json.loads(line) for line in (ROOT / "samples/smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {"date": datetime.now(timezone.utc).isoformat(), "purpose": "smoke_test_only_no_threshold_or_accuracy",
              "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
              "cuda_available": torch.cuda.is_available(), "checks": {}, "runs": []}
    detector = Detector()

    def no_network(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")

    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "create_connection", no_network):
        for sample in samples:
            result = detector.detect(sample["text"])
            for item in result["results"]:
                assert math.isfinite(item["ai_score"]) and 0 <= item["ai_score"] <= 1
                chunks = item["chunks"]
                assert chunks[0]["token_start"] == 0
                assert chunks[-1]["token_end"] == item["token_count"]
                assert all(left["token_end"] >= right["token_start"] for left, right in zip(chunks, chunks[1:]))
                assert all(0 <= chunk["char_start"] < chunk["char_end"] <= len(sample["text"]) for chunk in chunks)
                assert not item["truncated"]
                if sample["kind"] == "derived_integration_fixture":
                    assert item["chunk_count"] > 1
            report["runs"].append({"sample_id": sample["id"], "source_label": sample["source_label"],
                                   "source": sample["source"], "result": result})
            print(f"PASS {sample['id']}: {len(result['results'])} models", flush=True)

        # Independently repeat the authors' single-window inference formula.
        parity = {}
        mlp_parity = {}
        tokenizer_parity = {}
        for model_id in CATALOG:
            tokenizer, model, limit = detector._load(model_id)
            inputs = tokenizer(samples[0]["text"], return_tensors="pt", max_length=limit, truncation=True).to(detector.device)
            with torch.inference_mode():
                official_score = model(**inputs).logits[0].softmax(0)[CATALOG[model_id]["ai_label_index"]].item()
            wrapper_score = report["runs"][0]["result"]["results"][list(CATALOG).index(model_id)]["ai_score"]
            difference = abs(official_score - wrapper_score)
            assert difference < 1e-6, (model_id, difference)
            parity[model_id] = {"max_absolute_difference": difference, "passed": True}
            if CATALOG[model_id].get("loader") == "bert_mlp":
                # Check the checkpoint tensors' pooler -> Linear/ReLU/Linear
                # formula separately from our classifier module's forward.
                import torch.nn.functional as F
                with torch.inference_mode():
                    pooled = model.bert(**inputs).pooler_output
                    state = model.state_dict()
                    hidden = F.relu(F.linear(pooled, state["classifier.0.weight"], state["classifier.0.bias"]))
                    expected = F.linear(hidden, state["classifier.3.weight"], state["classifier.3.bias"])
                    delta = (expected - model(**inputs).logits).abs().max().item()
                    assert delta < 1e-6, (model_id, delta)
                mlp_parity[model_id] = {"max_logit_difference": delta, "passed": True}
                from transformers import BertTokenizer
                slow = BertTokenizer.from_pretrained(detector.model_dir / model_id, local_files_only=True)
                for text in [sample["text"] for sample in samples] + ["中文 English CAFÉ café，标点🙂测试。"]:
                    assert tokenizer(text)["input_ids"] == slow(text)["input_ids"], model_id
                tokenizer_parity[model_id] = "fast_matches_upstream_BertTokenizer"
        report["checks"]["author_single_window_formula"] = parity
        report["checks"]["ensemjudge_checkpoint_formula"] = mlp_parity
        report["checks"]["ensemjudge_tokenizer_parity"] = tokenizer_parity

        report["checks"]["offline_network_blocked"] = "passed"

        if detector.device == "cuda":
            cpu = Detector(device="cpu")
            cpu_result = cpu.detect(samples[0]["text"], include_chunks=False)
            cpu_diffs = {}
            for expected, actual in zip(report["runs"][0]["result"]["results"], cpu_result["results"]):
                difference = abs(expected["ai_score"] - actual["ai_score"])
                assert difference < 1e-4, (actual["model_id"], difference)
                cpu_diffs[actual["model_id"]] = difference
            report["checks"]["cpu_cuda_score_difference"] = cpu_diffs
    # Run the ASGI transport separately: Windows event-loop wakeup sockets are
    # infrastructure, and must not be mistaken for model network access.
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.get("/v1/models").json()["models"]) == len(CATALOG)
        response = client.post("/v1/detect", json={"text": samples[0]["text"], "models": list(CATALOG)})
        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == len(CATALOG)
        response = client.post("/v1/detect-batch", json={"texts": [samples[0]["text"], "中文🙂测试。"], "models": ["hc3-zh"], "include_chunks": False})
        assert response.status_code == 200, response.text
        assert len(response.json()["items"]) == 2
        assert "chunks" not in response.json()["items"][0]["results"][0]
        response = client.post("/v1/annotate", json={"text": samples[1]["text"], "models": list(CATALOG)})
        assert response.status_code == 200, response.text
        annotation = response.json()
        assert len(annotation["models"]) == len(CATALOG)
        assert annotation["segmentation"]["characters_covered"] == len(samples[1]["text"])
        assert set(annotation["segments"][0]["model_scores"]) == set(CATALOG)
    report["checks"]["api_in_process"] = "passed"
    report["status"] = "passed"
    report["model_sample_runs"] = len(samples) * len(CATALOG)
    report["device"] = detector.device
    if detector.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
        report["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All checks passed: {output}", flush=True)


if __name__ == "__main__":
    main()
