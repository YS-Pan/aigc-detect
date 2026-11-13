"""Exercise the running service and CLI from the caller's working directory."""

import argparse
import json
import subprocess
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from aigc_detector.catalog import CHINESE_MODELS, ENGLISH_MODELS, ROOT
from aigc_detector.client import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/deployment-test.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--english", action="store_true", help="Also exercise optional English HTTP and CLI routes")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = Client(base_url=args.base_url)
    report = {"date": datetime.now(timezone.utc).isoformat(), "cwd": str(Path.cwd()), "checks": {}}
    health = client.health()
    assert health["status"] == "ok", health
    report["health"] = health
    samples = [json.loads(line) for line in (ROOT / "samples/smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    result = client.detect(samples[0]["text"], models=list(CHINESE_MODELS))
    assert {item["model_id"] for item in result["results"]} == set(CHINESE_MODELS)
    report["http_result"] = result
    report["checks"]["http_all_models"] = "passed"
    annotation = client.annotate(samples[1]["text"], models=list(CHINESE_MODELS))
    assert {item["id"] for item in annotation["models"]} == set(CHINESE_MODELS)
    assert annotation["segmentation"]["characters_covered"] == len(samples[1]["text"])
    report["checks"]["http_annotate_all_models"] = "passed"
    batch = client.detect_batch([samples[0]["text"], samples[1]["text"]], models=["hc3-zh"], include_chunks=False)
    assert len(batch["items"]) == 2
    assert "chunks" not in batch["items"][0]["results"][0]
    report["checks"]["http_batch_subset"] = "passed"
    try:
        client.detect(samples[0]["text"], models=["not-a-model"])
    except urllib.error.HTTPError as exc:
        assert exc.code == 422, exc.code
        report["checks"]["http_invalid_model_422"] = "passed"
    else:
        raise AssertionError("Invalid model did not produce an HTTP error")
    listed = client.models()["models"]
    assert all(item["loaded"] for item in listed if item["id"] in CHINESE_MODELS)
    report["checks"]["http_cached_models"] = "passed"
    schema = client._request("/openapi.json")
    for path in ("/health", "/v1/models", "/v1/detect", "/v1/detect-batch"):
        assert path in schema["paths"]
    args.output.with_name("openapi.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["checks"]["openapi"] = "passed"
    command = [sys.executable, "-m", "aigc_detector", "detect", "--file", str(ROOT / "samples/demo-zh.txt"),
               "--models", "ensemjudge-bert", "bibbbu-xlm-roberta", "--device", "cpu", "--no-chunks"]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert process.returncode == 0, process.stderr
    cli_result = json.loads(process.stdout)
    assert [item["model_id"] for item in cli_result["results"]] == ["ensemjudge-bert", "bibbbu-xlm-roberta"]
    report["checks"]["cli_from_external_cwd"] = "passed"
    report["cli_result"] = cli_result
    if args.english:
        english = json.loads((ROOT / "samples/english-public-20261002/samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
        result = client.detect(english["text"], language="en")
        assert [item["model_id"] for item in result["results"]] == list(ENGLISH_MODELS)
        report["http_english_result"] = result
        batch = client.detect_batch([english["text"][:250], english["text"][-250:]], language="en", include_chunks=False)
        assert len(batch["items"]) == 2 and all(len(item["results"]) == 4 for item in batch["items"])
        annotation = client.annotate(english["text"], language="en", policy="sensitive-any-model-v1")
        assert [item["id"] for item in annotation["models"]] == list(ENGLISH_MODELS)
        assert annotation["segmentation"]["id"] == "english-natural-boundaries-v1"
        assert "".join(s["text"] for s in annotation["segments"]) == english["text"]
        report["checks"]["http_english_group_batch_annotation"] = "passed"
        command = [sys.executable, "-m", "aigc_detector", "detect", "--language", "en", "--file",
                   str(ROOT / "samples/english-public-20261002" / english["text_file"]), "--no-chunks"]
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=180, cwd=ROOT / ".runtime")
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        assert [item["model_id"] for item in result["results"]] == list(ENGLISH_MODELS)
        report["checks"]["cli_english_from_other_cwd"] = "passed"
        report["cli_english_result"] = result
    report["status"] = "passed"
    path = args.output
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": report["checks"], "report": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
