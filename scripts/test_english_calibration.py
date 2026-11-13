"""Real-weight validation of the frozen English annotation and API contracts."""
import argparse
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from aigc_detector.api import app
from aigc_detector.engine import Detector
from aigc_detector.fusion import ENGLISH_POLICY, LEGACY_POLICY, fused_score, load_profile, reapply_policy
from aigc_detector.reporting import save_annotation
from scripts.calibrate_english import EVIDENCE
from scripts.english_academic_samples import DEFAULT_DIR, digest, read_json, verify, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    records, _, _ = verify(DEFAULT_DIR)
    samples = {r["id"]: r for r in records}
    profile, profile_hash = load_profile(ENGLISH_POLICY)
    if read_json(EVIDENCE / "FREEZE.json")["profile_sha256"] != profile_hash:
        raise ValueError("Frozen profile has changed")
    train = read_json(EVIDENCE / "train-scores.json")
    positive = next(r for r in train["records"] if r["kind"] == "ai_paragraph" and
                    fused_score(r["scores"], profile)["score"] >= profile["high_threshold"])
    human = samples["P07-human"]
    ai = samples[positive["sample_id"]]
    detector = Detector()
    def blocked(*args, **kwargs):
        raise AssertionError("Real validation attempted network access")
    checks = {}
    # Initialize Windows asyncio's local socketpair before blocking model I/O.
    with patch("aigc_detector.api.get_detector", return_value=detector), TestClient(app) as client, \
            patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        report = detector.annotate(human["text"], language="en", policy=ENGLISH_POLICY)
        assert report["policy"]["id"] == ENGLISH_POLICY
        assert report["segmentation"]["max_characters"] == profile["segment_chars"]
        assert report["summary"]["strong_signal"]["segments"] + report["summary"]["possible_signal"]["segments"] == 0
        assert "Low sensitivity" in report["interpretation"]
        save_annotation(report, human["text"], args.report_dir / "human-P07", source_path=DEFAULT_DIR / human["text_file"])
        generated = detector.annotate(ai["text"], language="en", policy=ENGLISH_POLICY)
        assert generated["summary"]["strong_signal"]["segments"] > 0
        save_annotation(generated, ai["text"], args.report_dir / "fit-positive", source_path=DEFAULT_DIR / ai["text_file"])
        checks["real_positive_example_is_fit_evidence_only"] = {"sample_id": ai["id"], "summary": generated["summary"]}
        legacy = detector.annotate(human["text"], language="en", policy=LEGACY_POLICY,
                                   segment_chars=profile["segment_chars"], min_segment_chars=profile["min_segment_chars"])
        save_annotation(legacy, human["text"], args.report_dir / "legacy-human-P07")
        cached = reapply_policy(legacy, policy=ENGLISH_POLICY)
        assert cached["summary"] == report["summary"]
        assert [s["decision"] for s in cached["segments"]] == [s["decision"] for s in report["segments"]]
        save_annotation(cached, human["text"], args.report_dir / "recombined-human-P07")
        checks["cache_recombination_matches_real_annotation"] = True
        response = client.post("/v1/annotate", json={"text": human["text"], "language": "en", "policy": ENGLISH_POLICY})
        assert response.status_code == 200
        assert response.json()["summary"] == report["summary"]
        assert response.json()["policy"]["profile_sha256"] == profile_hash
        checks["real_api_explicit_original_policy"] = True
    checks.update({"network_blocked": True, "profile_sha256": profile_hash, "device": detector.device,
                   "profile_unchanged": digest((Path(__file__).resolve().parents[1]/"aigc_detector/profiles"/f"{ENGLISH_POLICY}.json").read_bytes()) == profile_hash,
                   "status": "complete"})
    write_json(args.report_dir / "validation.json", checks)
    write_json(args.report_dir / "COMPLETE.json", {"status": "complete", "profile_sha256": profile_hash})
    print(checks, flush=True)


if __name__ == "__main__":
    main()
