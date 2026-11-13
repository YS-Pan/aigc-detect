"""Collect attributed public examples and run the existing local detector.

Collection is deterministic and happens before scores are viewed. Detection
keeps source labels outside model inputs and blocks outbound socket connections.
"""
import argparse
import csv
import hashlib
import json
import math
import socket
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.catalog import ROOT
from network import EVENTS, get_json, open_public

HC3_REV = "09a687b8dc164b89e7df95abf15df3b216bc31c2"
BELLE_REV = "29c35e9b56c7fc91f04a613ed33896f7a19bad54"
BELLE_DOC_REV = "645084d36454cdeb4e7f1ce9b2f3426bac39f1f9"
DEFAULT_SAMPLES = ROOT / "samples/public-20260930"
LOCAL_TZ = timezone(timedelta(hours=8))
CATEGORIES = ("strong_signal", "possible_signal", "other")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(directory):
    if directory.exists():
        raise FileExistsError("Use a new sample directory; existing collections are never overwritten")
    records, evidence, snapshots = [], [], {}
    collected_at = datetime.now(LOCAL_TZ).isoformat()

    def snapshot(name, url):
        with open_public(url) as response:
            raw = response.read()
        raw.decode("utf-8-sig")
        snapshots[name] = raw
        evidence.append({"url": url, "snapshot": "sources/" + name,
                         "sha256": digest(raw), "bytes": len(raw)})
        return raw

    def add(record):
        text = record["text"]
        record.update(schema_version="public-aigc-sample/1.0", collected_at=collected_at,
                      characters=len(text), sha256=digest(text.encode("utf-8")),
                      text_file="texts/" + record["id"] + ".txt")
        records.append(record)

    hc3_meta = get_json("https://api.github.com/repos/Hello-SimpleAI/chatgpt-comparison-detection/commits/main")
    hc3_docs = "https://raw.githubusercontent.com/Hello-SimpleAI/chatgpt-comparison-detection/" + hc3_meta["sha"]
    snapshot("hc3-project.md", hc3_docs + "/README.md")
    snapshot("hc3-card.md", f"https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/raw/{HC3_REV}/README.md")
    for subset in ("open_qa", "psychology"):
        url = f"https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/resolve/{HC3_REV}/{subset}.jsonl"
        selected = 0
        prefix = hashlib.sha256()
        with open_public(url) as response:
            for line_number, raw in enumerate(response, 1):
                prefix.update(raw)
                row = json.loads(raw)
                answers = []
                for field in ("human_answers", "chatgpt_answers"):
                    choice = next(((i, t) for i, t in enumerate(row[field]) if 160 <= len(t) <= 1800), None)
                    if choice is None:
                        break
                    answers.append((field, *choice))
                if len(answers) != 2:
                    continue
                pair_id = f"hc3-{subset}-line{line_number}"
                for field, index, text in answers:
                    label = "human" if field == "human_answers" else "ai"
                    add({"id": pair_id + "-" + label, "pair_id": pair_id, "text": text,
                         "source_label": label, "kind": "official_dataset", "dataset": "Hello-SimpleAI/HC3-Chinese",
                         "subset": subset, "source_group": "hc3-" + subset, "source": url,
                         "dataset_revision": HC3_REV, "row_id": row.get("id"), "line_number": line_number,
                         "answer_field": field, "answer_index": index, "question": row["question"],
                         "source_row_sha256": digest(raw), "provenance": "publisher_labeled_answer_field",
                         "label_evidence": hc3_docs + "/README.md", "license": "CC-BY-SA-4.0",
                         "upstream_source_license": "MIT" if subset == "open_qa" else "CC0",
                         "attribution": "Biyang Guo et al., HC3 (2023); original answer authors credited by the source dataset",
                         "transformations": [], "training_overlap": {"hc3-zh": "known_entire_HC3_Chinese_training_data",
                                                                      "other_models": "not_audited"}})
                selected += 1
                if selected == 4:
                    break
        if selected != 4:
            raise RuntimeError(f"Expected four complete {subset} pairs")
        evidence.append({"url": url, "scanned_through_line": line_number,
                         "scanned_prefix_sha256": prefix.hexdigest(), "scope": "prefix_not_full_upstream_file"})

    belle_doc_base = "https://raw.githubusercontent.com/LianjiaTech/BELLE/" + BELLE_DOC_REV
    snapshot("belle-card.md", f"https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/raw/{BELLE_REV}/README.md")
    snapshot("belle-generation.md", belle_doc_base + "/data/1.5M/README.md")
    snapshot("belle-data-license.txt", belle_doc_base + "/DATA_LICENSE")
    snapshot("belle-disclaimer.txt", belle_doc_base + "/DISCLAIMER")
    url = f"https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/{BELLE_REV}/Belle_open_source_0.5M.json"
    selected = 0
    prefix = hashlib.sha256()
    with open_public(url) as response:
        for line_number, raw in enumerate(response, 1):
            prefix.update(raw)
            row = json.loads(raw)
            text = row["output"]
            chinese = sum("\u3400" <= c <= "\u9fff" for c in text)
            if not 400 <= len(text) <= 1800 or chinese / len(text) < .3 or "```" in text:
                continue
            add({"id": f"belle-line{line_number}-ai", "text": text, "source_label": "ai",
                 "kind": "official_generated_dataset", "dataset": "BelleGroup/train_0.5M_CN",
                 "source_group": "belle", "source": url, "dataset_revision": BELLE_REV,
                 "line_number": line_number, "answer_field": "output", "source_row_sha256": digest(raw),
                 "instruction": row["instruction"], "input": row["input"],
                 "provenance": "publisher_disclosed_generation_pipeline",
                 "label_evidence": belle_doc_base + "/data/1.5M/README.md",
                 "generator": "BELLE disclosed OpenAI generation pipeline; exact per-row model not published",
                 "license": "upstream_mixed_notices_see_snapshots",
                 "license_notices": {"dataset_card": "GPL-3.0", "repository_data_license": "ODC-By-1.0",
                                     "usage_restriction": "research_only_no_commercial_use"},
                 "attribution": "BELLE Group / LianjiaTech, train_0.5M_CN",
                 "transformations": [], "training_overlap": {"all_models": "not_audited"}})
            selected += 1
            if selected == 4:
                break
    if selected != 4:
        raise RuntimeError("Expected four BELLE Chinese prose outputs")
    evidence.append({"url": url, "scanned_through_line": line_number,
                     "scanned_prefix_sha256": prefix.hexdigest(), "scope": "prefix_not_full_upstream_file"})

    url = "https://www.gutenberg.org/ebooks/27166.txt.utf-8"
    with open_public(url) as response:
        raw = response.read()
    source = raw.decode("utf-8-sig")
    evidence.append({"url": url, "sha256": digest(raw), "bytes": len(raw),
                     "scope": "complete_download", "release_date": "2008-11-05", "last_update": "2021-01-04"})
    header = source[:source.index("*** START OF THE PROJECT GUTENBERG")]
    footer = source[source.index("*** END OF THE PROJECT GUTENBERG"):]
    snapshots["gutenberg-header-and-license.txt"] = (header + "\n\n" + footer).encode("utf-8")
    anchors = (("preface", "《吶喊》自序", "我在年青時候也曾經做過許多夢"),
               ("kong-yiji", "孔乙己", "魯鎮的酒店的格局"),
               ("hometown", "故鄉", "我冒了嚴寒"),
               ("village-opera", "社戲", "我在倒數上去的二十年中"))
    for slug, title, anchor in anchors:
        if source.count(anchor) != 1:
            raise RuntimeError(f"Historical excerpt anchor is not unique: {slug}")
        start = source.index(anchor)
        # End on the first sentence boundary at or after 800 original characters.
        end = source.index("。", start + 800) + 1
        text = source[start:end]
        add({"id": "luxun-" + slug + "-human", "text": text, "source_label": "human",
             "kind": "historical_authored_work", "source_group": "luxun", "source": url,
             "book": "吶喊", "title": title, "author": "魯迅 / Lu Xun (1881–1936)",
             "provenance": "named_historical_author_and_pre_generative_AI_electronic_edition",
             "label_evidence": "https://www.gutenberg.org/ebooks/27166",
             "source_document_sha256": digest(raw), "source_char_start": start, "source_char_end": end,
             "source_offset_unit": "Python Unicode code points after utf-8-sig decoding, CRLF preserved",
             "source_release_date": "2008-11-05", "source_last_update": "2021-01-04",
             "license": "public_domain_original_work; Project_Gutenberg_redistribution_notice_retained",
             "attribution": "魯迅, 吶喊; electronic edition produced by Han Huei Haung, Project Gutenberg #27166",
             "transformations": ["contiguous_excerpt_only; spelling, script, spaces and CRLF unchanged"],
             "training_overlap": {"all_models": "not_audited; familiar_classic_texts_may_occur_in_training"}})
    if len(records) != 24 or Counter(x["source_label"] for x in records) != {"ai": 12, "human": 12}:
        raise RuntimeError("Collection must contain exactly 12 human and 12 AI texts")
    directory.mkdir(parents=True)
    (directory / "texts").mkdir()
    (directory / "sources").mkdir()
    for name, raw in snapshots.items():
        (directory / "sources" / name).write_bytes(raw)
    for item in records:
        (directory / item["text_file"]).write_bytes(item["text"].encode("utf-8"))
    raw_records = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records).encode("utf-8")
    (directory / "samples.jsonl").write_bytes(raw_records)
    write_json(directory / "collection.json", {
        "schema_version": "public-aigc-collection/1.0", "collected_at": collected_at,
        "samples": 24, "source_label_counts": dict(Counter(x["source_label"] for x in records)),
        "samples_sha256": digest(raw_records), "selection_fixed_before_detection": True,
        "selection": {"hc3": "first 4 complete pairs per open_qa/psychology; first answer per field of 160–1800 chars",
                      "belle": "first 4 outputs of 400–1800 chars, >=30% CJK, no fenced code",
                      "luxun": "4 named works; exact contiguous opening excerpt, sentence end at/after 800 chars"},
        "upstream_evidence": evidence, "network_events": EVENTS,
        "interpretation": "Source labels come from provenance, never from detector scores; no independent accuracy claim"})
    print(f"Collected 24 attributed texts: {directory}", flush=True)


def verify(directory):
    collection = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    raw = (directory / "samples.jsonl").read_bytes()
    if digest(raw) != collection["samples_sha256"]:
        raise ValueError("Collection hash mismatch")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(records) != collection["samples"] or len({x["id"] for x in records}) != len(records):
        raise ValueError("Unexpected record count or duplicate IDs")
    if len({x["sha256"] for x in records}) != len(records):
        raise ValueError("Duplicate sample text")
    for item in records:
        text = item["text"]
        if not text.strip() or item["source_label"] not in ("human", "ai"):
            raise ValueError("Sample requires a nonempty text and a provenance label")
        if len(text) != item["characters"] or digest(text.encode("utf-8")) != item["sha256"]:
            raise ValueError("Sample content metadata mismatch")
        text_path = (directory / item["text_file"]).resolve()
        if not text_path.is_relative_to(directory.resolve()):
            raise ValueError("Sample text_file escapes collection directory")
        if text_path.read_bytes() != text.encode("utf-8"):
            raise ValueError("UTF-8 text snapshot differs from JSONL")
    for item in collection["upstream_evidence"]:
        if "snapshot" in item and digest((directory / item["snapshot"]).read_bytes()) != item["sha256"]:
            raise ValueError("Source evidence hash mismatch")
    if Counter(x["source_label"] for x in records) != collection["source_label_counts"]:
        raise ValueError("Source label count mismatch")
    print(f"Verified {len(records)} distinct UTF-8 texts and source evidence", flush=True)
    return records, collection


def detect(directory, report_dir):
    from aigc_detector import __version__
    from aigc_detector.engine import Detector
    from aigc_detector.fusion import REFERENCE_POLICY, load_profile
    from aigc_detector.reporting import save_annotation
    records, collection = verify(directory)
    profile, profile_hash = load_profile()
    detector = Detector()
    models = [item for item in detector.list_models() if item["id"] in profile["models"]]
    if any(not item["installed"] or item["revision"] != profile["revisions"][item["id"]] for item in models):
        raise RuntimeError("All ten installed model revisions must match the fixed profile")
    report_dir.mkdir(parents=True, exist_ok=False)
    fixed = {"created_at": datetime.now(LOCAL_TZ).isoformat(), "software_version": __version__,
             "sample_collection": str(directory.resolve()), "sample_collection_sha256": collection["samples_sha256"],
             "policy": REFERENCE_POLICY, "profile_sha256": profile_hash, "models": models,
             "model_manifest_sha256": digest((ROOT / "models/manifest.json").read_bytes()),
             "high_threshold": profile["high_threshold"], "possible_threshold": profile["possible_threshold"],
             "source_code_sha256": {name: digest((ROOT / name).read_bytes()) for name in
                                    ("aigc_detector/engine.py", "aigc_detector/annotation.py", "aigc_detector/fusion.py",
                                     "aigc_detector/modeling.py", "aigc_detector/catalog.py", "scripts/public_samples.py")},
             "segment_chars": 400, "min_segment_chars": 120, "threshold_overrides": False,
             "device": detector.device, "source_labels_used_as_model_inputs": False}
    write_json(report_dir / "run-config.json", fixed)
    rows, raw_results = [], []

    def no_network(*args, **kwargs):
        raise AssertionError("Local inference attempted a network connection")

    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "create_connection", no_network):
        for index, sample in enumerate(records, 1):
            text = sample["text"]
            raw = detector.detect(text)
            annotation = detector.annotate(text)
            if raw["input"]["sha256"] != sample["sha256"] or len(raw["results"]) != 10:
                raise RuntimeError("Raw detection input hash or model count mismatch")
            if annotation["policy"]["profile_sha256"] != profile_hash:
                raise RuntimeError("Fusion profile changed during the run")
            for item in raw["results"]:
                if item["truncated"] or not math.isfinite(item["ai_score"]) or not 0 <= item["ai_score"] <= 1:
                    raise RuntimeError("Invalid raw model result")
            if not annotation["segmentation"]["coverage_complete"] or annotation["segmentation"]["characters_covered"] != len(text):
                raise RuntimeError("Incomplete annotation coverage")
            if any(set(s["model_scores"]) != set(profile["models"]) for s in annotation["segments"]):
                raise RuntimeError("Annotation did not score every segment with ten models")
            save_annotation(annotation, text, report_dir / sample["id"], source_path=directory / sample["text_file"])
            counts = Counter(s["decision"]["category"] for s in annotation["segments"])
            category = next(c for c in CATEGORIES if counts[c])
            row = {"sample_id": sample["id"], "source_label": sample["source_label"],
                   "source_group": sample["source_group"], "characters": len(text), "sha256": sample["sha256"],
                   "segments": len(annotation["segments"]), "strong_segments": counts["strong_signal"],
                   "possible_segments": counts["possible_signal"], "other_segments": counts["other"],
                   "sample_screening_category": category,
                   "max_ensemble_score": max(s["decision"]["ensemble_score"] for s in annotation["segments"]),
                   "max_review_score": max(s["decision"]["review_score"] for s in annotation["segments"]),
                   "annotation_html": sample["id"] + "/report.html",
                   **{item["model_id"]: item["ai_score"] for item in raw["results"]}}
            rows.append(row)
            raw_results.append({"sample_id": sample["id"], "source_label": sample["source_label"], "result": raw})
            print(f"{index}/{len(records)} {sample['id']}: {dict(counts)}", flush=True)
    write_json(report_dir / "raw-detection.json", {"runs": raw_results, "status": "complete"})
    write_json(report_dir / "summary.json", {"config": fixed, "rows": rows,
               "checks": {"distinct_utf8_samples": len(records), "raw_model_sample_runs": len(records) * 10,
                          "full_annotation_coverage": True, "offline_network_blocked": True,
                          "fixed_profile_and_model_revisions": True},
               "sample_category_rule": "Any strong segment -> strong; else any possible -> possible; else other; descriptive summary only",
               "interpretation": "Scores are uncalibrated; these selected public examples are not an independent accuracy benchmark"})
    with (report_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(report_dir / "COMPLETE.json", {"status": "complete", "samples": len(records),
               "sample_collection_sha256": collection["samples_sha256"], "profile_sha256": profile_hash,
               "files": ["run-config.json", "summary.json", "raw-detection.json", "scores.csv"]})
    print(f"Complete: {report_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "verify", "detect"))
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.action == "collect":
        collect(args.sample_dir)
    elif args.action == "verify":
        verify(args.sample_dir)
    else:
        detect(args.sample_dir, args.report_dir or ROOT / "reports" / ("public-samples-" + datetime.now(LOCAL_TZ).strftime("%Y%m%d-%H%M%S")))


if __name__ == "__main__":
    main()
