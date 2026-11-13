"""Real local-weight installation checks, not an accuracy benchmark or fitting job."""

import argparse
import gc
import hashlib
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from fastapi.testclient import TestClient

from aigc_detector.api import app
from aigc_detector.catalog import CATALOG, EXTRA_ENGLISH_MODELS, ROOT
from aigc_detector.engine import Detector
from aigc_detector.statistical import StatisticalDetector


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def offline():
    def blocked(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        yield


def cleanup(detector):
    detector._loaded.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classifiers(records, report_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu, cpu = Detector(device=device), Detector(device="cpu")
    rows, checks = [], []
    # Windows' asyncio socketpair is created before blocking inference sockets.
    with TestClient(app) as client, offline():
        for model_id in EXTRA_ENGLISH_MODELS:
            print(f"Offline classifier checks: {model_id}", flush=True)
            for record in records:
                raw = gpu.detect(record["text"], [model_id], language="en")
                result = raw["results"][0]
                assert not result["truncated"]
                assert result["chunks"][0]["token_start"] == 0
                assert result["chunks"][-1]["token_end"] == result["token_count"]
                assert abs(sum(c["aggregation_weight"] for c in result["chunks"])-result["token_count"]) < 1e-6
                rows.append({"id":record["id"],"source_label":record["source_label"],
                             "input_sha256":raw["input"]["sha256"], **result})
            text = records[0]["text"][:500]
            reference = cpu.detect(text, [model_id], language="en")["results"][0]
            actual = gpu.detect(text, [model_id], language="en")["results"][0]
            difference = abs(reference["ai_score"]-actual["ai_score"])
            assert difference < 1e-4
            tokenizer, model, limit = gpu._load(model_id)
            encoding = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
            with torch.inference_mode():
                expected = float(model(**encoding).logits.float().softmax(-1)[0,0])
            formula_difference = abs(expected-actual["ai_score"])
            assert formula_difference < 1e-6
            checks.append({"model_id":model_id,"parameters":sum(p.numel() for p in model.parameters()),
                           "max_tokens":limit,"cpu_cuda_difference":difference,
                           "softmax_class_zero_difference":formula_difference})
            cleanup(cpu)
        long_text = records[0]["text"] * 6
        long_result = gpu.detect(long_text, ["mage-en"], language="en")["results"][0]
        assert long_result["max_tokens"] == 4096 and long_result["token_count"] > 4096
        assert long_result["chunk_count"] >= 2 and long_result["chunks"][-1]["token_end"] == long_result["token_count"]
        write_json(report_dir/"mage-long-coverage.json", long_result)
        with patch("aigc_detector.api.get_detector", return_value=gpu):
            listing = client.get("/v1/models").json()
            assert len(listing["models"]) == len(CATALOG)
            assert listing["extra_english_models"] == list(EXTRA_ENGLISH_MODELS)
            payload={"text":records[0]["text"][:500],"models":list(EXTRA_ENGLISH_MODELS),"language":"en"}
            response=client.post("/v1/detect",json=payload)
            assert response.status_code==200 and len(response.json()["results"])==5
            payload.update(policy="sensitive-any-model-v1",segment_chars=1600,min_segment_chars=400)
            response=client.post("/v1/annotate",json=payload)
            assert response.status_code==200 and response.json()["status"]=="complete"
            assert response.json()["policy"]["reference_fitted"] is False
        cleanup(gpu)
    summary={model:{label:{"count":len(values:=[r["ai_score"] for r in rows if r["model_id"]==model and r["source_label"]==label]),
                           "minimum":min(values),"maximum":max(values)} for label in ("human","ai")}
             for model in EXTRA_ENGLISH_MODELS}
    write_json(report_dir/"classifiers.json",{"status":"complete","models":checks,"sample_model_runs":len(rows),
               "raw_scores":rows,"descriptive_ranges":summary,"api_checks_passed":True,
               "offline_socket_guard_passed":True,"interpretation":"Installation smoke on existing public examples; training overlap is not audited, no thresholds were fitted, and this is not independent accuracy/generalization evidence."})
    print(f"Five classifiers: {len(rows)} public sample-model runs passed.",flush=True)


def statistical(records, report_dir):
    detector=StatisticalDetector()
    chosen=[next(record for record in records if record["source_label"]==label) for label in ("human","ai")]
    with offline():
        for record in chosen:
            print(f"Offline statistical checks: {record['id']}",flush=True)
            # Full existing excerpt, including more than one causal window.
            result=detector.detect(record["text"],["fast-detect-gpt-neo","fast-detect-gpt-falcon","binoculars-falcon"])
            assert result["status"]=="complete" and len(result["loads"])==3
            assert all(r["coverage_complete"] and not r["truncated"] and not r["threshold_applied"] for r in result["results"])
            assert all(not row["loading_problems"] for row in result["loads"])
            write_json(report_dir/(record["id"]+"-statistical.json"),result)
        cpu=StatisticalDetector(device="cpu")
        result=cpu.detect(chosen[0]["text"][:300],["fast-detect-gpt-neo"])
        write_json(report_dir/"neo-cpu.json",result)
    write_json(report_dir/"statistical-COMPLETE.json",{"status":"complete","methods":3,"public_inputs":len(chosen),
               "cpu_neo_passed":True,"offline_socket_guard_passed":True,
               "interpretation":"Raw statistics are installation checks only; no fitted/upstream decision thresholds or existing fusion changes."})
    print('Statistical offline checks passed.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--part',choices=('classifiers','statistical'),required=True)
    args=parser.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    target=args.report_dir/('classifiers.json' if args.part=='classifiers' else 'statistical-COMPLETE.json')
    if target.exists():
        raise FileExistsError(f'Preserve completed evidence; choose a new report directory: {target}')
    records=[json.loads(line) for line in (ROOT/'samples/english-public-20261002/samples.jsonl').read_text(encoding='utf-8').splitlines()]
    for record in records:
        assert hashlib.sha256(record['text'].encode('utf-8')).hexdigest()==record['sha256']
    (classifiers if args.part=='classifiers' else statistical)(records,args.report_dir)


if __name__=='__main__':
    main()
