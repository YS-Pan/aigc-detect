"""Render all inputs from verified local raw windows for the frozen profile."""
import copy
import gc
import html
from pathlib import Path
import shutil
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.annotation import segment_spans
from aigc_detector.engine import Detector
from aigc_detector.fusion import fused_score, load_profile, signal_revision, segment_raw_scores
from aigc_detector.reporting import save_annotation
from scripts.calibrate_english_nonlinear import EVIDENCE, REPORTS, PARENT_REPORTS, POLICY
from scripts.calibrate_english_v2 import CLASSIFIERS
from scripts.english_academic_samples import DEFAULT_DIR, digest, read_json, verify, write_json


def main():
    profile, profile_hash = load_profile(POLICY)
    freeze, evaluation = read_json(EVIDENCE / "FREEZE.json"), read_json(EVIDENCE / "evaluation.json")
    assert freeze["profile_sha256"] == profile_hash == evaluation["profile_sha256"]
    train_path, repeat_path = EVIDENCE / "train-scores.json", EVIDENCE / "repeat-scores.json"
    assert digest(train_path.read_bytes()) == freeze["train_dataset_sha256"]
    assert digest(repeat_path.read_bytes()) == evaluation["repeat_dataset_sha256"]
    records = read_json(train_path)["records"]+read_json(repeat_path)["records"]
    expected = {(r["sample_id"], r["char_start"], r["char_end"]): r for r in records}
    samples, collection, _ = verify(DEFAULT_DIR)
    assert collection["samples_sha256"] == read_json(EVIDENCE / "campaign.json")["sample_collection_sha256"]
    passages = {}
    for sample in samples:
        for start, end in segment_spans(sample["text"], profile["segment_chars"], profile["min_segment_chars"], language="en"):
            value = sample["text"][start:end]
            key = digest(value.encode("utf-8"))
            assert expected[sample["id"], start, end]["sha256"] == key
            passages.setdefault(key, value)
    location = REPORTS / f"context{profile['segment_chars']}"
    location.mkdir(parents=True, exist_ok=True)
    caches, cache_hashes, filled = {}, {}, {}
    def blocked(*args, **kwargs):
        raise AssertionError("Window evidence retrieval attempted network access")
    real = Detector()
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for mid in [m for m in profile["models"] if m in CLASSIFIERS]:
            path, original = location / f"{mid}-cache.json", PARENT_REPORTS / location.name / f"{mid}-cache.json"
            if not path.exists() and original.exists():
                shutil.copyfile(original, path)
            cache = read_json(path) if path.exists() else {"revision": profile["revisions"][mid], "items": {}}
            assert cache["revision"] == profile["revisions"][mid]
            missing = [key for key in passages if key not in cache["items"]]
            filled[mid] = len(missing)
            for offset in range(0, len(missing), 64):
                keys = missing[offset:offset+64]
                values = real._score_texts([passages[k] for k in keys], mid, 64, True, allow_empty=True)
                for key, value in zip(keys, values):
                    assert value["revision"] == profile["revisions"][mid]
                    cache["items"][key] = value
                write_json(path, cache)
                print(f"{mid}: retrieved missing windows {min(offset+64,len(missing))}/{len(missing)}", flush=True)
            real._loaded.clear()
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            caches[mid], cache_hashes[mid] = cache["items"], digest(path.read_bytes())
    stat_cache = None
    if profile["statistical_assets"]:
        path = location / "statistical-cache.json"
        stat_cache = read_json(path)
        assert stat_cache["inference"] == profile["statistical_inference"]
        for assets in profile["statistical_assets"].values():
            assert all(stat_cache["asset_revisions"][a] == revision for a, revision in assets.items())
        assert all(all(m in {r['backend_id'] for r in stat_cache['items'][key]} for m in profile["statistical_assets"]) for key in passages)
        cache_hashes["statistical"] = digest(path.read_bytes())

    class CachedDetector(Detector):
        def _score_texts(self, texts, mid, overlap_tokens=64, include_chunks=True, *, allow_empty=False):
            assert overlap_tokens == 64 and include_chunks
            return [copy.deepcopy(caches[mid][digest(t.encode("utf-8"))]) for t in texts]

    def cached_statistics(detector, texts, pinned):
        result = {m: [] for m in pinned["statistical_assets"]}
        for text in texts:
            rows = {r["backend_id"]: r for r in stat_cache["items"][digest(text.encode("utf-8"))]}
            for mid in result:
                row = copy.deepcopy(rows[mid])
                assert {a["id"]: a["revision"] for a in row["assets"]} == pinned["statistical_assets"][mid]
                result[mid].append({"raw_score": row["raw_score"], "ai_score": None, "kind": "statistical",
                                   "revision": signal_revision(pinned, mid), **{k: row[k] for k in
                                   ("token_count", "chunk_count", "chunks", "score_direction", "assets", "aggregation")}})
        return result, []

    detector = CachedDetector(device="cpu")
    detector.device = "cached-local-scores"
    destination = REPORTS / "all-inputs"
    destination.mkdir(exist_ok=False)
    entries, max_difference, category_changes = [], 0., 0
    with patch("aigc_detector.annotation.statistical_passage_scores", side_effect=cached_statistics):
        for sample in samples:
            report = detector.annotate(sample["text"], language="en", policy=POLICY)
            report["inference_source"] = {"kind": "verified_local_raw_window_cache", "cache_sha256": cache_hashes,
                                          "missing_windows_retrieved_locally": filled, "network_blocked": True}
            for segment in report["segments"]:
                record = expected[sample["id"], segment["char_start"], segment["char_end"]]
                assert segment["sha256"] == record["sha256"]
                raw = segment_raw_scores(segment)
                difference = max(abs(raw[m]-record["scores"][m]) for m in profile["models"])
                max_difference = max(difference, max_difference)
                assert difference < 2e-5
                expected_score = fused_score(record["scores"], profile)["score"]
                assert abs(expected_score-segment["decision"]["ensemble_score"]) < 2e-5
                for threshold in [profile["high_threshold"], profile["possible_threshold"]]:
                    category_changes += int((expected_score >= threshold) != (segment["decision"]["ensemble_score"] >= threshold))
            save_annotation(report, sample["text"], destination / sample["id"], source_path=DEFAULT_DIR / sample["text_file"])
            entries.append({"sample_id": sample["id"], "paper_id": sample["paper_id"], "split": sample["split"], "kind": sample["kind"],
                            "input_sha256": report["input"]["sha256"], "segments": len(report["segments"]), "summary": report["summary"],
                            "html": f"{sample['id']}/report.html"})
    assert category_changes == 0 and len(entries) == 115
    assert sum(e["segments"] for e in entries) == evaluation["full_collection"]["scored_segments"]
    split_names = {"fit": "拟合", "calibration": "门限控制", "test": "重复评估"}
    kind_names = {"human_document": "原版文档", "mixed_document": "混合文档", "ai_paragraph": "单独生成段"}
    lines = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>英文非线性融合逐篇报告</title>",
             "<style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#24352f}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#1d6353}</style>",
             f"<h1>英文非线性融合逐篇报告</h1><p>10篇原版、30份混合文档、75处单独生成段，共{sum(e['segments'] for e in entries)}片段。策略 {POLICY}，参数与门限冻结；完整原始窗口来自已核验本地缓存。</p>",
             "<p>拟合与门限控制已参与适配；P07/P08为已知论文重复评估，没有新的独立验证。未标记不能排除AIGC。</p>"]
    for kind, name in kind_names.items():
        lines.append(f"<h2>{name}</h2><table><tr><th>输入</th><th>集合</th><th>片段</th><th>强档</th><th>仅疑似</th></tr>")
        for entry in entries:
            if entry["kind"] == kind:
                lines.append(f"<tr><td><a href='{html.escape(entry['html'], quote=True)}'>{html.escape(entry['sample_id'])}</a></td>"
                             f"<td>{split_names[entry['split']]}</td><td>{entry['segments']}</td>"
                             f"<td>{entry['summary']['strong_signal']['segments']}</td><td>{entry['summary']['possible_signal']['segments']}</td></tr>")
        lines.append("</table>")
    lines.append("</html>")
    (destination / "index.html").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(destination / "index.json", {"status": "complete", "profile_sha256": profile_hash, "inputs": entries, "contains_source_text": False})
    write_json(destination / "COMPLETE.json", {"status": "complete", "profile_sha256": profile_hash, "inputs": len(entries),
               "segments": sum(e["segments"] for e in entries), "cache_sha256": cache_hashes,
               "locally_retrieved_missing_windows": filled, "maximum_raw_difference_from_numeric_evidence": max_difference,
               "threshold_decision_changes": category_changes, "network_blocked": True})
    print(f"Complete: {len(entries)} input bundles; {destination / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
