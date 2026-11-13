"""Real offline API and recombination checks for the frozen nonlinear profile."""
import gc
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from aigc_detector.api import app
from aigc_detector.engine import Detector
from aigc_detector.fusion import fused_score, load_profile, reapply_policy, resolve_policy, segment_raw_scores
from aigc_detector.reporting import save_annotation
from scripts.calibrate_english_nonlinear import EVIDENCE, REPORTS, ROOT, POLICY
from scripts.english_academic_samples import DEFAULT_DIR, digest, read_json, verify, write_json


def main():
    target = REPORTS / "runtime"
    target.mkdir(exist_ok=False)
    profile, profile_hash = load_profile(POLICY)
    assert read_json(EVIDENCE / "FREEZE.json")["profile_sha256"] == profile_hash
    train, repeat = read_json(EVIDENCE / "train-scores.json"), read_json(EVIDENCE / "repeat-scores.json")
    samples = {s["id"]: s for s in verify(DEFAULT_DIR)[0]}
    human = next(r for r in repeat["records"] if r["kind"] == "human_document")
    mixed = next(r for r in repeat["records"] if r["kind"] == "mixed_document" and r["ai_chars"] > 0)
    generated = max((r for r in train["records"] if r["split"] == "fit" and r["kind"] == "ai_paragraph"),
                    key=lambda r: fused_score(r["scores"], profile)["score"])
    cases = [("repeat-human-window", human), ("repeat-mixed-window", mixed), ("fit-generated-window", generated)]
    if profile["statistical_assets"]:
        cases = cases[:1]
    detector, checks = Detector(), []
    def blocked(*args, **kwargs):
        raise AssertionError("Nonlinear runtime verification attempted network access")
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client, \
            patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for name, record in cases:
            source = samples[record["sample_id"]]
            text = source["text"][record["char_start"]:record["char_end"]]
            input_path = target / "inputs" / f"{name}.txt"
            input_path.parent.mkdir(exist_ok=True)
            input_path.write_bytes(text.encode("utf-8"))
            response = client.post("/v1/annotate", json={"text": text, "language": "en", "policy": POLICY})
            assert response.status_code == 200, response.text
            report = response.json()
            assert len(report["segments"]) == 1 and report["segments"][0]["sha256"] == record["sha256"]
            assert report["policy"]["profile_sha256"] == profile_hash
            raw = segment_raw_scores(report["segments"][0])
            differences = {m: abs(raw[m]-record["scores"][m]) for m in profile["models"]}
            assert max(differences.values()) < 2e-5, differences
            expected = fused_score(record["scores"], profile)
            assert abs(report["segments"][0]["decision"]["ensemble_score"]-expected["score"]) < 2e-5
            cached = reapply_policy(report)
            assert cached["policy"]["id"] == POLICY
            assert cached["summary"] == report["summary"]
            assert [s["decision"] for s in cached["segments"]] == [s["decision"] for s in report["segments"]]
            save_annotation(report, text, target / name, source_path=input_path)
            save_annotation(cached, text, target / f"{name}-recombined")
            checks.append({"case": name, "sample_id": record["sample_id"], "input_sha256": record["sha256"],
                           "raw_absolute_differences": differences, "summary": report["summary"], "cache_matches_runtime": True})
        default_policy = resolve_policy(None, "en")
        health = client.get("/health").json()
        assert health["annotation_profiles"]["en"]["ready"]
        assert health["annotation_profiles"]["en"]["policy"] == default_policy
        if default_policy == POLICY:
            text = samples[human["sample_id"]]["text"][human["char_start"]:human["char_end"]]
            default = client.post("/v1/annotate", json={"text": text, "language": "en"})
            assert default.status_code == 200 and default.json()["policy"]["profile_sha256"] == profile_hash
    assert digest((ROOT / "aigc_detector/profiles" / f"{POLICY}.json").read_bytes()) == profile_hash
    detector._loaded.clear()
    gc.collect()
    result = {"status": "complete", "profile_sha256": profile_hash, "checks": checks, "network_blocked": True,
              "device": detector.device, "default_english_policy": default_policy, "profile_unchanged": True,
              "code_sha256": {f: digest((ROOT / f).read_bytes()) for f in ["aigc_detector/fusion.py", "aigc_detector/nonlinear.py",
                         "aigc_detector/annotation.py", "aigc_detector/api.py", "aigc_detector/reporting.py"]},
              "contains_source_text": False, "interpretation": "Engineering path checks, not independent effectiveness validation."}
    write_json(target / "validation.json", result)
    write_json(target / "COMPLETE.json", {"status": "complete", "profile_sha256": profile_hash})
    print(f"Completed {len(checks)} real offline API/cache checks; default={default_policy}", flush=True)


if __name__ == "__main__":
    main()
