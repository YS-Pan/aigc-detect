"""Render every verified input from the already computed local window scores."""
import copy
import html
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.engine import Detector
from aigc_detector.fusion import ENGLISH_EXPANDED_POLICY, fused_score, load_profile
from aigc_detector.reporting import save_annotation
from scripts.calibrate_english_v2 import EVIDENCE, REPORTS
from scripts.english_academic_samples import DEFAULT_DIR, digest, read_json, verify, write_json


def main():
    profile, profile_hash = load_profile(ENGLISH_EXPANDED_POLICY)
    freeze = read_json(EVIDENCE / "FREEZE.json")
    if freeze["profile_sha256"] != profile_hash or profile["statistical_assets"]:
        raise ValueError("This renderer requires the frozen classifier-only final profile")
    train_path = EVIDENCE / f"context{profile['segment_chars']}" / "train-scores.json"
    repeat_path = EVIDENCE / "repeat-scores.json"
    evaluation = read_json(EVIDENCE / "evaluation.json")
    if digest(train_path.read_bytes()) != freeze["train_dataset_sha256"] or \
            digest(repeat_path.read_bytes()) != evaluation["repeat_dataset_sha256"]:
        raise ValueError("Frozen numeric evidence changed")
    records = read_json(train_path)["records"] + read_json(repeat_path)["records"]
    expected = {(r["sample_id"], r["char_start"], r["char_end"]): r for r in records}
    caches, cache_hashes = {}, {}
    for mid in profile["models"]:
        path = REPORTS / f"context{profile['segment_chars']}" / f"{mid}-cache.json"
        cache = read_json(path)
        if cache["revision"] != profile["revisions"][mid]:
            raise ValueError("Cached classifier revision differs from the frozen profile")
        caches[mid] = cache["items"]
        cache_hashes[mid] = digest(path.read_bytes())

    class CachedDetector(Detector):
        def _score_texts(self, passages, mid, overlap_tokens=64, include_chunks=True, *, allow_empty=False):
            if overlap_tokens != 64 or not include_chunks:
                raise ValueError("Cache renderer must preserve original window evidence")
            values = []
            for text in passages:
                key = digest(text.encode("utf-8"))
                row = copy.deepcopy(caches[mid][key])
                if row["revision"] != profile["revisions"][mid]:
                    raise ValueError("Cached passage revision mismatch")
                values.append(row)
            return values

    detector = CachedDetector(device="cpu")
    detector.device = "cached-local-scores"
    samples, _, _ = verify(DEFAULT_DIR)
    destination = REPORTS / "all-inputs"
    destination.mkdir(exist_ok=False)
    entries = []
    for sample in samples:
        report = detector.annotate(sample["text"], language="en", policy=ENGLISH_EXPANDED_POLICY)
        assert report["policy"]["profile_sha256"] == profile_hash
        report["inference_source"] = {"kind": "reused_local_raw_window_cache", "new_model_inference": False,
                                      "cache_sha256": cache_hashes}
        for segment in report["segments"]:
            record = expected[(sample["id"], segment["char_start"], segment["char_end"])]
            assert segment["sha256"] == record["sha256"]
            for mid in profile["models"]:
                assert segment["model_scores"][mid]["ai_score"] == record["scores"][mid]
            score = fused_score(record["scores"], profile)
            assert segment["decision"]["ensemble_score"] == score["score"]
        save_annotation(report, sample["text"], destination / sample["id"],
                        source_path=DEFAULT_DIR / sample["text_file"])
        entries.append({"sample_id": sample["id"], "paper_id": sample["paper_id"], "split": sample["split"],
                        "kind": sample["kind"], "input_sha256": report["input"]["sha256"],
                        "segments": len(report["segments"]), "summary": report["summary"],
                        "html": f"{sample['id']}/report.html"})
    assert len(entries) == 115 and sum(e["segments"] for e in entries) == evaluation["full_collection"]["scored_segments"]
    split_names = {"fit": "拟合", "calibration": "门限校准", "test": "重复评估"}
    kind_names = {"human_document": "原版文档", "mixed_document": "混合文档", "ai_paragraph": "单独生成段"}
    lines = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>英文重新校准逐篇报告</title>",
             "<style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#24352f}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#1d6353}</style>",
             "<h1>英文重新校准逐篇报告</h1><p>10篇原版、30份混合文档、75处单独生成段，共678片段。复用本轮本地窗口分数；模型、参数和门限保持冻结。</p>",
             "<p>拟合与校准已参与适配；P07/P08为已知论文重复评估，没有新的独立验证。检出仍很低，未标记不能排除AIGC。</p>"]
    for kind in kind_names:
        lines.append(f"<h2>{kind_names[kind]}</h2><table><tr><th>输入</th><th>集合</th><th>片段</th><th>强档</th><th>仅疑似</th></tr>")
        for entry in entries:
            if entry["kind"] == kind:
                lines.append(f"<tr><td><a href='{html.escape(entry['html'], quote=True)}'>{html.escape(entry['sample_id'])}</a></td>"
                             f"<td>{split_names[entry['split']]}</td><td>{entry['segments']}</td>"
                             f"<td>{entry['summary']['strong_signal']['segments']}</td><td>{entry['summary']['possible_signal']['segments']}</td></tr>")
        lines.append("</table>")
    lines.append("</html>")
    (destination / "index.html").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(destination / "index.json", {"status": "complete", "profile_sha256": profile_hash, "inputs": entries,
                                           "contains_source_text": False, "new_model_inference": False})
    write_json(destination / "COMPLETE.json", {"status": "complete", "profile_sha256": profile_hash,
                    "inputs": len(entries), "segments": sum(e["segments"] for e in entries), "cache_sha256": cache_hashes})
    print(f"Complete: {len(entries)} input bundles / 678 segments; {destination / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
