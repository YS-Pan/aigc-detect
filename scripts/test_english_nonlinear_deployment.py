"""Actual localhost API and cross-cwd CLI checks, preserving the v2 default."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector import __version__
from aigc_detector.fusion import ENGLISH_EXPANDED_POLICY, fused_score, load_profile, segment_raw_scores
from aigc_detector.reporting import save_annotation
from scripts.calibrate_english_nonlinear import POLICY, REPORTS, ROOT
from scripts.english_academic_samples import read_json, write_json


def main():
    destination = REPORTS / "deployment"
    destination.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, data=None):
        req = urllib.request.Request("http://127.0.0.1:8765"+path,
                    data=json.dumps(data).encode("utf-8") if data is not None else None,
                    headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=600) as response:
            return json.load(response)
    health = request("/health")
    assert health["version"] == __version__ == "0.8.0"
    assert health["annotation_profiles"]["en"]["policy"] == ENGLISH_EXPANDED_POLICY
    assert health["annotation_profiles"]["en"]["ready"]
    profile, profile_hash = load_profile(POLICY)
    source = REPORTS / "runtime/inputs/repeat-human-window.txt"
    text = source.read_bytes().decode("utf-8")
    explicit = request("/v1/annotate", {"text": text, "language": "en", "policy": POLICY})
    assert explicit["policy"]["profile_sha256"] == profile_hash
    assert explicit["software_version"] == __version__
    save_annotation(explicit, text, destination / "http-explicit", source_path=source)
    default = request("/v1/annotate", {"text": text, "language": "en"})
    assert default["policy"]["id"] == ENGLISH_EXPANDED_POLICY
    save_annotation(default, text, destination / "http-default", source_path=source)
    def run(arguments):
        process = subprocess.run([sys.executable, "-X", "utf8", "-m", "aigc_detector", *arguments],
                                 cwd=ROOT.parent, encoding="utf-8", capture_output=True, timeout=600)
        (destination / f"{arguments[0]}-stdout.json").write_text(process.stdout, encoding="utf-8")
        (destination / f"{arguments[0]}-stderr.log").write_text(process.stderr, encoding="utf-8")
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)
    run(["annotate", "--language", "en", "--policy", POLICY, "--file", str(source),
         "--report-dir", str(destination / "cli-explicit"), "--device", "cpu"])
    cli = read_json(destination / "cli-explicit/report.json")
    assert cli["policy"]["profile_sha256"] == profile_hash
    assert cli["summary"] == explicit["summary"]
    differences = {m: abs(segment_raw_scores(cli["segments"][0])[m]-segment_raw_scores(explicit["segments"][0])[m]) for m in profile["models"]}
    assert max(differences.values()) < 2e-5
    run(["recombine", "--input-dir", str(destination / "http-default"), "--policy", POLICY,
         "--report-dir", str(destination / "cli-v2-to-v3")])
    rebuilt = read_json(destination / "cli-v2-to-v3/report.json")
    assert rebuilt["policy"]["profile_sha256"] == profile_hash and rebuilt["summary"] == explicit["summary"]
    for segment in rebuilt["segments"]:
        value = fused_score(segment_raw_scores(segment), profile)
        assert value["score"] == segment["decision"]["ensemble_score"]
    write_json(REPORTS / "deployment.json", {"status": "complete", "software_version": __version__,
               "default_policy": ENGLISH_EXPANDED_POLICY, "experimental_policy": POLICY, "profile_sha256": profile_hash,
               "default_profile_sha256": health["annotation_profiles"]["en"]["profile_sha256"], "server_process_id": health["process_id"],
               "actual_http_explicit_and_default": True, "cross_cwd_cli": True, "cached_v2_to_v3_recombine": True,
               "cpu_cuda_raw_absolute_differences": differences, "contains_source_text": False})
    print("Actual HTTP, cross-cwd CPU CLI and cached v2-to-v3 recombine passed; default remains v2.", flush=True)


if __name__ == "__main__":
    main()
