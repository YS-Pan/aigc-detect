"""Reproducible public-fixture collection and offline installation validation.

No threshold selection, calibration, or independent accuracy claim is made.
All full fixtures and runtime reports remain in Git-ignored reports/.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigc_detector.catalog import CATALOG, ROOT, SPECIALIZED_ENGLISH_MODELS
from aigc_detector.paths import model_root

MODEL_ROOT = model_root()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(directory):
    import yaml
    from scripts.network import download, open_public
    destination = directory / "fixtures.json"
    if destination.exists():
        raise FileExistsError(destination)
    fixtures = []
    card = (MODEL_ROOT / "roberta-academic-en/README.md").read_text(encoding="utf-8")
    widgets = yaml.safe_load(card.split("---", 2)[1])["widget"]
    for index, widget in enumerate(widgets):
        fixtures.append({"id": f"academic-card-{index}", "text": widget["text"],
                         "source_label": widget["example_title"],
                         "source": CATALOG["roberta-academic-en"]["label_source"].rsplit("/", 1)[0] + "/README.md"})
    spec = CATALOG["llm-detectaive-en"]
    url = f'https://huggingface.co/spaces/{spec["repo_id"]}/resolve/{spec["revision"]}/samples.json'
    download(url, directory / "detectaive-samples.json")
    rows = json.loads((directory / "detectaive-samples.json").read_text(encoding="utf-8"))
    seen = set()
    for index, (text, label) in enumerate(rows):
        if label not in seen:
            fixtures.append({"id": f"detectaive-{index}", "text": text, "source_label": label, "source": url})
            seen.add(label)
    dataset_revision = "ac418fb3dd2c2bbe0430683b0ab753a515de8315"
    url = f"https://huggingface.co/datasets/khaaaaaan/CoCoNUTS/resolve/{dataset_revision}/test.jsonl"
    with open_public(url) as response:
        record = json.loads(response.readline())
    fixtures.append({"id": "coconuts-line-1", "text": record["review"], "source_label": record["class"], "source": url})
    for row in fixtures:
        row["sha256"] = digest(row["text"])
    write_json(destination, {"purpose": "official examples for installation smoke only", "records": fixtures})
    print(f"Collected {len(fixtures)} official examples.", flush=True)


def validate(directory):
    import torch
    from fastapi.testclient import TestClient
    from aigc_detector import __version__
    from aigc_detector.api import app
    from aigc_detector.engine import Detector
    from aigc_detector.fusion import LEGACY_POLICY, reapply_policy
    from aigc_detector.reporting import save_annotation

    target = directory / "validation.json"
    if target.exists():
        raise FileExistsError(target)
    records = json.loads((directory / "fixtures.json").read_text(encoding="utf-8"))["records"]
    for record in records:
        assert digest(record["text"]) == record["sha256"]
    def blocked(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")
    def cleanup(detector):
        detector._loaded.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gpu, cpu = Detector(), Detector(device="cpu")
    checks, raw = [], []
    # Initialize Windows asyncio socketpair before blocking network access.
    with TestClient(app) as client, patch.object(socket.socket, "connect", blocked), \
            patch.object(socket, "create_connection", blocked):
        for mid in SPECIALIZED_ENGLISH_MODELS:
            print(f"Offline real-weight checks: {mid}", flush=True)
            for record in records:
                result = gpu.detect(record["text"], [mid], language="en")
                raw.append({"sample_id": record["id"], "source_label": record["source_label"], "result": result})
            text = next(r["text"] for r in records if r["id"].startswith("academic-card-2"))[:800]
            actual = gpu.detect(text, [mid], language="en")["results"][0]
            reference = cpu.detect(text, [mid], language="en")["results"][0]
            difference = abs(actual["ai_score"] - reference["ai_score"])
            assert difference < 1e-4, (mid, difference)
            tokenizer, model, limit = gpu._load(mid)
            inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(gpu.device)
            with torch.inference_mode():
                logits = model(**inputs).logits.float()
                if mid == "cocodet-en":
                    hidden = model.bert(**inputs).last_hidden_state[:, 0]
                    # Independent dot/(norm*norm) reference for the author's cosine head.
                    weights = model.main_classifier_weight
                    direct_logits = (hidden @ weights.T) / (hidden.norm(dim=1, keepdim=True) * weights.norm(dim=1)) * 30
                    assert torch.allclose(logits, direct_logits, atol=2e-5)
                probabilities = logits.softmax(-1)[0].cpu().tolist()
            spec = CATALOG[mid]
            indices = spec["ai_label_indices"] if "ai_label_indices" in spec else [spec["ai_label_index"]]
            formula_difference = abs(actual["ai_score"] - sum(probabilities[i] for i in indices))
            assert formula_difference < 2e-6, (mid, formula_difference)
            if "class_labels" in spec:
                for label, value in zip(spec["class_labels"], probabilities):
                    assert abs(actual["class_scores"][label] - value) < 2e-6
            # Force multiple windows and a real tail for every new architecture.
            phrase = "The antenna radiation pattern was measured across the frequency band. "
            long_text = phrase * (limit // 8 + 35)
            long_result = gpu.detect(long_text, [mid], language="en")["results"][0]
            assert long_result["token_count"] > limit and long_result["chunk_count"] > 1
            assert long_result["chunks"][0]["token_start"] == 0
            assert long_result["chunks"][-1]["token_end"] == long_result["token_count"]
            assert abs(sum(c["aggregation_weight"] for c in long_result["chunks"]) - long_result["token_count"]) < 1e-6
            write_json(directory / f"long-{mid}.json", long_result)
            checks.append({"model_id": mid, "revision": actual["revision"], "max_tokens": limit,
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "cpu_cuda_difference": difference, "formula_difference": formula_difference,
                           "long_token_count": long_result["token_count"], "long_chunk_count": long_result["chunk_count"]})
            cleanup(cpu)

        with patch("aigc_detector.api.get_detector", return_value=gpu):
            listing = client.get("/v1/models").json()
            assert len(listing["models"]) == len(CATALOG)
            assert listing["specialized_english_models"] == list(SPECIALIZED_ENGLISH_MODELS)
            text = records[2]["text"][:1300]
            payload = {"text": text, "models": list(SPECIALIZED_ENGLISH_MODELS), "language": "en"}
            response = client.post("/v1/detect", json=payload)
            assert response.status_code == 200, response.text
            write_json(directory / "http-detect.json", response.json())
            response = client.post("/v1/detect-batch", json={"texts": [text, text + " Further measurements are needed."],
                                                            "models": ["cocodet-en", "llm-detectaive-en"], "language": "en"})
            assert response.status_code == 200 and len(response.json()["items"]) == 2
            response = client.post("/v1/annotate", json={**payload, "policy": LEGACY_POLICY,
                                                        "segment_chars": 1600, "min_segment_chars": 400})
            assert response.status_code == 200, response.text
            report = response.json()
            for mid in ("cocodet-en", "llm-detectaive-en"):
                assert "class_scores" in report["segments"][0]["model_scores"][mid]
            save_annotation(report, text, directory / "annotation")
            recombined = reapply_policy(report, policy=LEGACY_POLICY)
            assert recombined["interpretation"] == report["interpretation"]
            assert recombined["segments"][0]["model_scores"] == report["segments"][0]["model_scores"]
            save_annotation(recombined, text, directory / "recombined")
        cleanup(gpu)

    write_json(directory / "raw-smoke.json", raw)
    summary = {"status": "complete", "software_version": __version__, "device": gpu.device,
               "offline_socket_guard_passed": True, "official_examples": len(records),
               "sample_model_runs": len(raw), "models": checks, "http_and_annotation_checks_passed": True,
               "interpretation": "Installation checks on official examples, not an independent accuracy test. No thresholds fitted and no default profile changed."}
    write_json(target, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def deployment(directory):
    import subprocess
    import urllib.request
    from aigc_detector import __version__
    from aigc_detector.fusion import load_profile, resolve_policy
    target = directory / "deployment.json"
    if target.exists():
        raise FileExistsError(target)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request("http://127.0.0.1:8765" + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        with opener.open(req, timeout=600) as response:
            return json.load(response)
    health, listing = request("/health"), request("/v1/models")
    assert health["service"] == "local-aigc-detectors" and health["version"] == __version__
    assert len(listing["models"]) == len(CATALOG)
    assert listing["specialized_english_models"] == list(SPECIALIZED_ENGLISH_MODELS)
    profiles = {}
    for language in ("zh", "en"):
        policy = resolve_policy(None, language)
        _, hash_value = load_profile(policy)
        assert health["annotation_profiles"][language]["ready"]
        assert health["annotation_profiles"][language]["profile_sha256"] == hash_value
        profiles[language] = {"policy": policy, "sha256": hash_value}
    text = json.loads((directory / "fixtures.json").read_text(encoding="utf-8"))["records"][2]["text"][:1300]
    source, output = (directory / "cli-input.txt").resolve(), (directory / "cli-detect.json").resolve()
    source.write_text(text, encoding="utf-8")
    http = request("/v1/detect", {"text": text, "language": "en", "models": list(SPECIALIZED_ENGLISH_MODELS)})
    write_json(directory / "deployed-http.json", http)
    command = [sys.executable, "-X", "utf8", "-B", "-m", "aigc_detector", "detect", "--language", "en",
               "--file", str(source), "--models", *SPECIALIZED_ENGLISH_MODELS, "--output", str(output)]
    completed = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8", timeout=600)
    (directory / "cli-crosscwd.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    cli = json.loads(output.read_text(encoding="utf-8"))
    assert http["input"]["sha256"] == cli["input"]["sha256"] == digest(text)
    differences = {}
    for actual, expected in zip(http["results"], cli["results"]):
        mid = actual["model_id"]
        assert mid == expected["model_id"] and actual["revision"] == expected["revision"]
        differences[mid] = abs(actual["ai_score"] - expected["ai_score"])
        assert differences[mid] < 1e-5
        for label, value in actual.get("class_scores", {}).items():
            assert abs(value - expected["class_scores"][label]) < 1e-5
    result = {"status": "complete", "software_version": __version__, "api_process_id": health["process_id"],
              "api_url": "http://127.0.0.1:8765", "catalog_count": len(listing["models"]),
              "input_sha256": digest(text), "models": list(SPECIALIZED_ENGLISH_MODELS),
              "cross_cwd": str(ROOT.parent), "http_cli_differences": differences, "default_profiles": profiles}
    write_json(target, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "validate", "deployment"))
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    {"collect": collect, "validate": validate, "deployment": deployment}[args.action](args.report_dir)


if __name__ == "__main__":
    main()
