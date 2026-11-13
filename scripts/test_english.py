"""Real English weights and source examples; no fitting or accuracy claims."""

import argparse
import csv
import gc
import hashlib
import json
import math
import socket
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector import __version__
from aigc_detector.catalog import CATALOG, ENGLISH_MODELS, ROOT
from aigc_detector.engine import Detector
from aigc_detector.fusion import LEGACY_POLICY
from aigc_detector.reporting import save_annotation
from scripts.english_samples import DEFAULT_DIR, verify, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    records, collection = verify(args.samples)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    import torch
    import torch.nn.functional as F
    import transformers

    detector = Detector()
    config = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "software_version": __version__, "torch": torch.__version__, "transformers": transformers.__version__,
              "device": detector.device, "models": list(ENGLISH_MODELS),
              "model_revisions": {m["id"]: m["revision"] for m in detector.list_models() if m["id"] in ENGLISH_MODELS},
              "manifest_sha256": hashlib.sha256((ROOT / "models/manifest.json").read_bytes()).hexdigest(),
              "sample_collection_sha256": collection["samples_sha256"],
              "policy": LEGACY_POLICY, "high_threshold": .90, "possible_threshold": .70,
              "segment_chars": 400, "min_segment_chars": 120,
              "labels_used_as_model_inputs": False, "thresholds_fitted_on_these_samples": False,
              "source_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                     ("aigc_detector/catalog.py", "aigc_detector/engine.py", "aigc_detector/modeling.py",
                                      "aigc_detector/annotation.py", "scripts/english_samples.py", "scripts/test_english.py")}}
    write_json(args.report_dir / "run-config.json", config)
    raw_runs, rows, parity, checks = [], [], {}, {}

    def blocked(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")

    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for number, sample in enumerate(records, 1):
            text = sample["text"]
            raw = detector.detect(text, language="en")
            annotation = detector.annotate(text, language="en", policy=LEGACY_POLICY)
            assert raw["input"]["sha256"] == sample["sha256"]
            assert [r["model_id"] for r in raw["results"]] == list(ENGLISH_MODELS)
            assert not any("chinese" in warning for warning in raw["warnings"])
            for item in raw["results"]:
                assert math.isfinite(item["ai_score"]) and 0 <= item["ai_score"] <= 1
                chunks = item["chunks"]
                assert chunks[0]["token_start"] == 0 and chunks[-1]["token_end"] == item["token_count"]
                assert all(left["token_end"] >= right["token_start"] for left, right in zip(chunks, chunks[1:]))
                assert abs(sum(c["aggregation_weight"] for c in chunks) - item["token_count"]) < 1e-6
                assert all(0 <= c["char_start"] <= c["char_end"] <= len(text) for c in chunks)
                assert not item["truncated"]
            cursor = 0
            for segment in annotation["segments"]:
                assert segment["char_start"] == cursor
                assert set(segment["model_scores"]) == set(ENGLISH_MODELS)
                assert segment["text"] == text[segment["char_start"]:segment["char_end"]]
                cursor = segment["char_end"]
            assert cursor == len(text)
            save_annotation(annotation, text, args.report_dir / sample["id"], source_path=args.samples / sample["text_file"])
            counts = Counter(s["decision"]["category"] for s in annotation["segments"])
            rows.append({"sample_id": sample["id"], "source_label": sample["source_label"],
                         "source_group": sample["source_group"], "characters": len(text),
                         **{r["model_id"]: r["ai_score"] for r in raw["results"]},
                         "strong_segments": counts["strong_signal"], "possible_segments": counts["possible_signal"],
                         "other_segments": counts["other"], "html": sample["id"] + "/report.html"})
            raw_runs.append({"sample_id": sample["id"], "source_label": sample["source_label"], "result": raw})
            print(f"PASS {number}/{len(records)} {sample['id']}: " + ", ".join(f"{r['model_id']}={r['ai_score']:.4f}" for r in raw["results"]), flush=True)

        # A bounded substring is a separate engineering fixture, not a corpus row.
        parity_text = next(r["text"] for r in records if r["source_label"] == "ai")[:250]
        wrapped = detector.detect(parity_text, language="en", include_chunks=True)
        for item in wrapped["results"]:
            mid = item["model_id"]
            tokenizer, model, limit = detector._load(mid)
            assert item["chunk_count"] == 1
            inputs = tokenizer(parity_text, return_tensors="pt", padding="max_length", max_length=limit, truncation=True).to(detector.device)
            with torch.inference_mode():
                if mid == "desklib-en":
                    # Compute the publisher's masked mean + Linear + sigmoid
                    # independently of our custom forward.
                    hidden = model.model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    logits = F.linear(pooled, model.classifier.weight, model.classifier.bias)
                    expected = logits.sigmoid()[0, 0].item()
                else:
                    expected = model(**inputs).logits.float().softmax(-1)[0, CATALOG[mid]["ai_label_index"]].item()
            difference = abs(expected - item["ai_score"])
            assert difference < 1e-6, (mid, difference)
            parity[mid] = {"absolute_difference": difference, "passed": True}

        long_text = next(r["text"] for r in records if r["source_group"] == "gutenberg") * 3
        long_result = detector.detect(long_text, language="en")
        assert all(r["chunk_count"] > 1 for r in long_result["results"])
        assert all(r["chunks"][-1]["token_end"] == r["token_count"] for r in long_result["results"])
        write_json(args.report_dir / "long-window-fixture.json", long_result)
        checks["long_token_windows_complete"] = True
        if detector.device == "cuda":
            cpu = Detector(device="cpu")
            cpu_result = cpu.detect(parity_text, language="en", include_chunks=False)
            differences = {b["model_id"]: abs(a["ai_score"] - b["ai_score"]) for a, b in zip(wrapped["results"], cpu_result["results"])}
            assert max(differences.values()) < 1e-4, differences
            checks["cpu_cuda_score_differences"] = differences
            del cpu
            gc.collect()
        checks["offline_network_blocked"] = True
    checks["author_formula_parity"] = parity
    checks["all_sample_token_and_annotation_coverage_complete"] = True

    # Windows ASGI event-loop wakeup sockets are infrastructure, so transport
    # checks run outside the model-network blocking scope above.
    from fastapi.testclient import TestClient
    from aigc_detector.api import app
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client:
        assert client.get("/health").json()["model_groups"]["en"]["ready"]
        response = client.post("/v1/detect", json={"text": parity_text, "language": "en"})
        assert response.status_code == 200, response.text
        assert [r["model_id"] for r in response.json()["results"]] == list(ENGLISH_MODELS)
        response = client.post("/v1/detect-batch", json={"texts": [parity_text, records[0]["text"]], "language": "en"})
        assert response.status_code == 200 and len(response.json()["items"]) == 2
        response = client.post("/v1/annotate", json={"text": parity_text, "language": "en", "policy": LEGACY_POLICY})
        assert response.status_code == 200 and response.json()["segmentation"]["id"] == "english-natural-boundaries-v1"
        assert client.post("/v1/annotate", json={"text": parity_text, "language": "en"}).status_code == 422
    checks["english_http_routes"] = True
    write_json(args.report_dir / "raw-detection.json", {"status": "complete", "runs": raw_runs})
    with (args.report_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    descriptive = {}
    for mid in ENGLISH_MODELS:
        descriptive[mid] = {}
        for label in ("human", "ai"):
            values = [row[mid] for row in rows if row["source_label"] == label]
            descriptive[mid][label] = {"samples": len(values), "min": min(values), "max": max(values),
                                      "mean": sum(values) / len(values), "raw_score_ge_0_90": sum(v >= .9 for v in values),
                                      "raw_score_ge_0_70": sum(v >= .7 for v in values)}
    write_json(args.report_dir / "summary.json", {"config": config, "checks": checks, "rows": rows,
               "descriptive_raw_scores": descriptive, "sample_count": len(records), "raw_model_sample_runs": len(records) * 4,
               "interpretation": "Descriptive results on selected examples; HC3 samples overlap hc3-en training. Neither accuracy nor independent generalization is established. Whole-input scores and passage flags are different computations. English thresholds were not fitted."})
    write_json(args.report_dir / "COMPLETE.json", {"status": "complete", "samples": len(records), "checks_passed": True,
               "sample_collection_sha256": collection["samples_sha256"]})
    print(f"English checks passed: {args.report_dir}", flush=True)


if __name__ == "__main__":
    main()
