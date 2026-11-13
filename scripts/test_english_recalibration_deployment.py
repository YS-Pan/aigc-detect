"""Verify the restarted localhost API and the installed CLI from another cwd."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector import __version__
from aigc_detector.fusion import ENGLISH_EXPANDED_POLICY, load_profile, segment_raw_scores
from aigc_detector.reporting import save_annotation
from scripts.calibrate_english_v2 import REPORTS, ROOT
from scripts.english_academic_samples import digest, read_json, write_json


def main():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, body=None):
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request("http://127.0.0.1:8765" + path, data=payload,
                                     headers={"Content-Type": "application/json"} if payload else {})
        with opener.open(req, timeout=600) as response:
            return json.load(response)
    health = request("/health")
    profile, profile_hash = load_profile(ENGLISH_EXPANDED_POLICY)
    assert health["version"] == __version__ == "0.7.0"
    assert health["annotation_profiles"]["en"]["ready"]
    assert health["annotation_profiles"]["en"]["profile_sha256"] == profile_hash
    assert health["annotation_profiles"]["en"]["policy"] == ENGLISH_EXPANDED_POLICY
    assert len(health["model_groups"]["en"]["models"]) == 4
    models = request("/v1/models")
    assert len(models["models"]) == 18
    source = REPORTS / "runtime/inputs/repeat-human-window.txt"
    text = source.read_text(encoding="utf-8")
    live = request("/v1/annotate", {"text": text, "language": "en"})
    save_annotation(live, text, REPORTS / "deployed-http", source_path=source)
    destination = REPORTS / "cli-crosscwd"
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8", "-m", "aigc_detector", "annotate",
               "--language", "en", "--file", str(source), "--report-dir", str(destination)]
    result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, encoding="utf-8", errors="replace")
    (REPORTS / "cli-crosscwd.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Cross-cwd CLI failed; see cli-crosscwd.log")
    cli = read_json(destination / "report.json")
    reference = read_json(REPORTS / "runtime/repeat-human-window/report.json")
    differences = {}
    for name, report in [("deployed-http", live), ("cli-crosscwd", cli)]:
        assert report["policy"]["profile_sha256"] == profile_hash
        assert report["policy"]["id"] == ENGLISH_EXPANDED_POLICY
        assert report["summary"] == reference["summary"]
        assert report["input"]["sha256"] == digest(text.encode("utf-8"))
        assert len(report["segments"]) == len(reference["segments"]) == 1
        raw, expected = segment_raw_scores(report["segments"][0]), segment_raw_scores(reference["segments"][0])
        differences[name] = {m: abs(raw[m]-expected[m]) for m in profile["models"]}
        assert max(differences[name].values()) < 2e-5
        assert abs(report["segments"][0]["decision"]["ensemble_score"] -
                   reference["segments"][0]["decision"]["ensemble_score"]) < 2e-5
    for directory in [destination, REPORTS / "deployed-http"]:
        complete = read_json(directory / "COMPLETE.json")
        assert complete["status"] == "complete"
        assert all((directory / filename).is_file() for filename in complete["files"])
        assert digest((directory / "source.txt").read_bytes()) == complete["input_sha256"]
    write_json(REPORTS / "deployment.json", {"status": "complete", "software_version": __version__,
               "profile_sha256": profile_hash, "api_process_id": health["process_id"],
               "api_url": "http://127.0.0.1:8765", "raw_detection_english_models": health["model_groups"]["en"]["models"],
               "annotation_english_models": profile["models"], "registered_classifiers": len(models["models"]),
               "cross_cwd": str(ROOT.parent), "input_sha256": digest(text.encode("utf-8")),
               "raw_absolute_differences": differences, "contains_source_text": False,
               "interpretation": "Engineering checks on a previously used input; not an independent effect test."})
    print(json.dumps({"status": "complete", "version": __version__, "profile": ENGLISH_EXPANDED_POLICY,
                      "profile_sha256": profile_hash}), flush=True)


if __name__ == "__main__":
    main()
