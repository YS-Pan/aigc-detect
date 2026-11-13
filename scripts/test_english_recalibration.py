"""Real local annotation/API/cache checks for the expanded frozen profile."""
import argparse
import gc
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from aigc_detector.api import app
from aigc_detector.engine import Detector
from aigc_detector.fusion import (ENGLISH_EXPANDED_POLICY, fused_score, load_profile,
                                 reapply_policy, segment_raw_scores, resolve_policy)
from aigc_detector.reporting import save_annotation
from scripts.calibrate_english_v2 import EVIDENCE, REPORTS, ROOT
from scripts.english_academic_samples import DEFAULT_DIR, digest, read_json, verify, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORTS / "runtime")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    profile, profile_hash = load_profile(ENGLISH_EXPANDED_POLICY)
    if read_json(EVIDENCE / "FREEZE.json")["profile_sha256"] != profile_hash:
        raise ValueError("Frozen expanded profile changed")
    train = read_json(EVIDENCE / f"context{profile['segment_chars']}" / "train-scores.json")
    repeat = read_json(EVIDENCE / "repeat-scores.json")
    samples = {r["id"]: r for r in verify(DEFAULT_DIR)[0]}
    human = next(r for r in repeat["records"] if r["kind"] == "human_document")
    positive = next((r for r in train["records"] if r["split"] == "fit" and r["kind"] == "ai_paragraph" and
                     fused_score(r["scores"], profile)["score"] >= profile["possible_threshold"]), None)
    mixed = next(r for r in repeat["records"] if r["kind"] == "mixed_document" and r["ai_chars"] > 0)
    cases = [("repeat-human-window", human), ("repeat-mixed-window", mixed)]
    if positive:
        cases.append(("fit-generated-window", positive))
    # A statistical profile uses one real API input to avoid repeating cold LM
    # loads for an engineering check. Its three backend primitives were already
    # compared byte-for-byte to the earlier cached implementation.
    if profile["statistical_assets"]:
        cases = cases[:1]
    detector = Detector()
    checks = []
    def blocked(*args, **kwargs):
        raise AssertionError("Runtime verification attempted network access")
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client, \
            patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for name, record in cases:
            sample = samples[record["sample_id"]]
            text = sample["text"][record["char_start"]:record["char_end"]]
            inputs = args.report_dir / "inputs"
            inputs.mkdir(exist_ok=True)
            source = inputs / f"{name}.txt"
            source.write_bytes(text.encode("utf-8"))
            response = client.post("/v1/annotate", json={"text": text, "language": "en", "policy": ENGLISH_EXPANDED_POLICY})
            assert response.status_code == 200, response.text
            report = response.json()
            assert len(report["segments"]) == 1 and report["segments"][0]["sha256"] == record["sha256"]
            assert report["policy"]["profile_sha256"] == profile_hash
            assert report["segmentation"]["max_characters"] == profile["segment_chars"]
            raw = segment_raw_scores(report["segments"][0])
            differences = {m: abs(raw[m]-record["scores"][m]) for m in profile["models"]}
            assert max(differences.values()) < 2e-5, differences
            expected = fused_score(record["scores"], profile)
            assert abs(report["segments"][0]["decision"]["ensemble_score"]-expected["score"]) < 2e-5
            save_annotation(report, text, args.report_dir / name, source_path=source)
            rebuilt = reapply_policy(report, policy=ENGLISH_EXPANDED_POLICY)
            assert rebuilt["summary"] == report["summary"]
            assert [s["decision"] for s in rebuilt["segments"]] == [s["decision"] for s in report["segments"]]
            if name == "repeat-human-window":
                default = client.post("/v1/annotate", json={"text": text, "language": "en"})
                assert default.status_code == 200
                assert default.json()["policy"]["profile_sha256"] == profile_hash
                assert default.json()["summary"] == report["summary"]
            save_annotation(rebuilt, text, args.report_dir / f"{name}-recombined")
            checks.append({"case": name, "sample_id": record["sample_id"], "source_char_start": record["char_start"],
                           "source_char_end": record["char_end"], "input_sha256": record["sha256"],
                           "raw_absolute_differences": differences, "summary": report["summary"],
                           "cache_matches_runtime": True, "fit_or_repeat_evidence_only": True})
        health = client.get("/health").json()
        assert health["annotation_profiles"]["en"]["ready"]
        assert health["annotation_profiles"]["en"]["policy"] == ENGLISH_EXPANDED_POLICY
    target = ROOT / "aigc_detector/profiles" / f"{ENGLISH_EXPANDED_POLICY}.json"
    assert digest(target.read_bytes()) == profile_hash
    detector._loaded.clear()
    gc.collect()
    result = {"status": "complete", "profile_sha256": profile_hash, "checks": checks,
              "device": detector.device, "network_blocked": True, "profile_unchanged": True,
              "default_english_policy": resolve_policy(None, "en"),
              "code_sha256": {f: digest((ROOT / f).read_bytes()) for f in ["aigc_detector/fusion.py", "aigc_detector/statistical.py",
                               "aigc_detector/annotation.py", "aigc_detector/api.py", "scripts/calibrate_english_v2.py"]},
              "contains_source_text": False, "interpretation": "Engineering path checks, not independent effect validation."}
    write_json(args.report_dir / "validation.json", result)
    write_json(args.report_dir / "COMPLETE.json", {"status": "complete", "profile_sha256": profile_hash})
    print(result, flush=True)


if __name__ == "__main__":
    main()
