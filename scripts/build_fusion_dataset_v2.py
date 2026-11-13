"""Audit local CNKI labels and freeze score-only evidence for dual-tier fitting.

No inference or network calls are made. Private reports are required to rebuild;
only numeric scores, offsets, hashes and provenance metadata enter evaluations.
The reviewed v1 alignment is retained, including its stated boundary uncertainty.
"""
import argparse
import copy
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from import_cnki import aigc_body, aigc_excerpts, aigc_summary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def han(value):
    return "".join(c for c in value if "\u3400" <= c <= "\u9fff")


def union(ranges):
    merged = []
    for start, end in sorted(ranges):
        require(isinstance(start, int) and isinstance(end, int) and 0 <= start < end,
                "Invalid half-open character interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def length(ranges):
    return sum(end - start for start, end in union(ranges))


def overlap(start, end, ranges):
    return sum(max(0, min(end, right) - max(start, left)) for left, right in union(ranges))


def rates(strong, possible, total):
    return {"total_characters": total, "strong_characters": strong,
            "possible_only_characters": possible, "any_characters": strong + possible,
            "strong_percent": 100 * strong / total,
            "possible_only_percent": 100 * possible / total,
            "any_percent": 100 * (strong + possible) / total}


def recompute_alignment(excerpt, text):
    positions = [i for i, c in enumerate(text) if "\u3400" <= c <= "\u9fff"]
    body = "".join(text[i] for i in positions)
    target = han(excerpt["text"])
    initial = difflib.SequenceMatcher(None, target, body, autojunk=False).get_matching_blocks()
    anchor = max(initial, key=lambda block: block.size)
    left = max(0, anchor.b - anchor.a - 120)
    right = min(len(body), anchor.b - anchor.a + len(target) + 120)
    blocks = [difflib.Match(b.a, b.b + left, b.size) for b in
              difflib.SequenceMatcher(None, target, body[left:right], autojunk=False).get_matching_blocks()]
    useful = [block for block in blocks if block.size >= 20]
    require(useful, "CNKI excerpt has no substantial Han anchor")
    ranges = [{"char_start": positions[b.b], "char_end": positions[b.b + b.size - 1] + 1,
               "han_characters": b.size} for b in useful]
    start = min(r["char_start"] for r in ranges)
    end = max(r["char_end"] for r in ranges)
    coverage = sum(b.size for b in useful) / len(target)
    require(coverage >= .6 and end - start <= 2 * len(excerpt["text"]), "Uncertain CNKI alignment")
    longest = max(useful, key=lambda block: block.size)
    longest_text = target[longest.a:longest.a + longest.size]
    occurrence_count = sum(body.startswith(longest_text, i) for i in range(len(body)))
    require(occurrence_count == 1, "Longest CNKI anchor is ambiguous in current prose")
    unmatched = []
    cursor = 0
    for block in useful:
        if cursor < block.a:
            unmatched.append({"han_start": cursor, "han_end": block.a,
                              "text": target[cursor:block.a],
                              "sha256": text_digest(target[cursor:block.a])})
        cursor = block.a + block.size
    if cursor < len(target):
        unmatched.append({"han_start": cursor, "han_end": len(target), "text": target[cursor:],
                          "sha256": text_digest(target[cursor:])})
    return {"char_start": start, "char_end": end, "matched_ranges": ranges,
            "matched_han_characters": sum(b.size for b in useful), "cnki_han_characters": len(target),
            "han_coverage": coverage, "longest_anchor_han_characters": longest.size,
            "longest_anchor_occurrences": occurrence_count, "unmatched_han": unmatched}


def audit_provider(private_root, report):
    stage = private_root / "cnki-stage-v2"
    manifest = read(stage / "manifest.json")
    evidence = {}
    pages = {}
    for name, fragment in (("simple", "AIGC_\u7b80\u6d01"), ("full", "AIGC_\u5168\u6587")):
        entries = [entry for entry in manifest["originals"] if fragment in entry["path"]]
        require(len(entries) == 1, "CNKI source PDF is missing or ambiguous")
        entry = entries[0]
        require(digest(stage / entry["path"]) == entry["sha256"], "CNKI PDF hash mismatch")
        page_path = stage / entry["page_records"]
        pages[name] = [json.loads(line) for line in page_path.read_text(encoding="utf-8").splitlines()]
        require(len(pages[name]) == entry["pages"], "CNKI page count mismatch")
        evidence[name] = {"pdf_sha256": entry["sha256"], "pages": entry["pages"],
                          "page_records_sha256": digest(page_path)}
    summary, sections = aigc_summary(pages["simple"])
    require(aigc_summary(pages["full"]) == (summary, sections), "CNKI report summaries differ")
    require(summary == report["summary"] and sections == report["sections"], "Frozen CNKI summary differs")
    excerpts = aigc_excerpts(pages["simple"], sections)
    source, spans = aigc_body(pages["full"])
    require(text_digest(source) == report["extracted_input"]["sha256"], "CNKI body hash mismatch")
    require(source == (stage / "aigc/source.txt").read_text(encoding="utf-8"), "CNKI body differs from saved extraction")
    marked = [span for span in spans if span["decision"]["category"] != "other" and span["text"].strip()]
    require(len(marked) == len(excerpts) == len(report["segments"]), "CNKI excerpt counts differ")
    for excerpt, span, frozen in zip(excerpts, marked, report["segments"]):
        for key in ("id", "provider_characters", "decision", "text_sha256"):
            require(excerpt[key] == frozen[key], f"CNKI excerpt {key} differs")
        require(re.sub(r"\s+", "", excerpt["text"]) == re.sub(r"\s+", "", span["text"]),
                "CNKI simple/full text differs")
        require(excerpt["decision"]["category"] == span["decision"]["category"], "CNKI label/color mismatch")
        require((frozen["char_start"], frozen["char_end"]) == (span["char_start"], span["char_end"]),
                "CNKI full-report offsets differ")
    require(sum(e["provider_characters"] for e in excerpts if e["decision"]["category"] == "strong_signal")
            == summary["marked_characters"], "CNKI strong count differs from headline")
    evidence["validation"] = {"saved_pdf_hashes_match": True, "reparsed_saved_page_records": True,
                              "simple_full_labels_and_text_agree": True, "section_and_headline_totals_agree": True,
                              "raw_pdf_text_reextracted_in_this_audit": False}
    return evidence


def build(private_root, previous_path, destination):
    previous = read(previous_path)
    alignment_path = private_root / "corpus/cnki-alignment.json"
    alignment = read(alignment_path)
    report_path = private_root / "cnki-stage-v2/aigc/report.json"
    report = read(report_path)
    thesis = (private_root / "corpus/thesis.txt").read_text(encoding="utf-8")
    require(text_digest(thesis) == alignment["input_sha256"], "Alignment source hash mismatch")
    provider_evidence = audit_provider(private_root, report)
    spans = alignment["segments"]
    require(len({s["cnki_id"] for s in spans}) == len(spans) == len(report["segments"]), "Duplicate or missing CNKI IDs")
    span_audits = []
    private_details = []
    for span, excerpt in zip(spans, report["segments"]):
        require(span["cnki_id"] == excerpt["id"] and span["category"] == excerpt["decision"]["category"],
                "CNKI alignment label mismatch")
        recomputed = recompute_alignment(excerpt, thesis)
        for key in ("char_start", "char_end", "matched_ranges", "matched_han_characters", "cnki_han_characters", "han_coverage"):
            require(span[key] == recomputed[key], f"CNKI alignment {key} changed")
        require(span["text"] == thesis[span["char_start"]:span["char_end"]], "CNKI mapped text differs")
        anchor_length = length((r["char_start"], r["char_end"]) for r in span["matched_ranges"])
        span_audits.append({"cnki_id": span["cnki_id"], "category": span["category"],
                           "provider_characters": excerpt["provider_characters"],
                           "char_start": span["char_start"], "char_end": span["char_end"],
                           "mapped_characters": span["char_end"] - span["char_start"],
                           "anchor_union_characters": anchor_length,
                           "between_anchor_characters": span["char_end"] - span["char_start"] - anchor_length,
                           **{k: recomputed[k] for k in ("matched_han_characters", "cnki_han_characters", "han_coverage",
                                                        "longest_anchor_han_characters", "longest_anchor_occurrences")},
                           "requires_extraction_scope_review": span["requires_extraction_scope_review"],
                           "unmatched_han_blocks": [{k: v for k, v in block.items() if k != "text"}
                                                     for block in recomputed["unmatched_han"]],
                           "simple_pdf_pages": sorted({x["page"] for x in excerpt["source_locations"]}),
                           "full_pdf_pages": sorted({x["page"] for x in excerpt["full_report_locations"]}),
                           "excerpt_text_sha256": excerpt["text_sha256"],
                           "mapped_text_sha256": text_digest(span["text"]), "matched_ranges": span["matched_ranges"]})
        private_details.append({"cnki_id": span["cnki_id"], "excerpt_text": excerpt["text"],
                                "mapped_text": span["text"], "unmatched_han": recomputed["unmatched_han"]})
    strong = union((s["char_start"], s["char_end"]) for s in spans if s["category"] == "strong_signal")
    possible = union((s["char_start"], s["char_end"]) for s in spans if s["category"] == "possible_signal")
    require(length(strong + possible) == length(strong) + length(possible), "CNKI categories overlap")
    require(length(strong + possible) == sum(s["char_end"] - s["char_start"] for s in spans), "CNKI intervals overlap")
    records = []
    source_audits = {}
    old_records = {(r["document"], r["id"]): r for r in previous["records"]}
    require(len(old_records) == len(previous["records"]), "Duplicate frozen score records")
    for name, source in previous["sources"].items():
        raw_path = private_root / f"raw/{name}/report.json"
        raw = read(raw_path)
        source_path = private_root / f"corpus/{name}.txt"
        mapping_path = private_root / f"corpus/{name}-map.json"
        mapping = read(mapping_path)
        text = source_path.read_text(encoding="utf-8")
        require(digest(source_path) == source["input_sha256"] == raw["input"]["sha256"], "Inference input hash mismatch")
        require(digest(raw_path) == source["raw_report_sha256"], "Raw report hash mismatch")
        require(digest(mapping_path) == source["paragraph_map_sha256"], "Paragraph map hash mismatch")
        require([m["id"] for m in raw["models"]] == previous["models"], "Model order mismatch")
        require({m["id"]: m["revision"] for m in raw["models"]} == previous["revisions"], "Model revision mismatch")
        cursor = 0
        ids = set()
        boundary_records = []
        for segment in raw["segments"]:
            start, end = segment["char_start"], segment["char_end"]
            require(start == cursor and start < end <= len(text), "Inference segments contain a gap or overlap")
            require(segment["id"] not in ids, "Duplicate inference segment ID")
            ids.add(segment["id"])
            cursor = end
            chars = end - start
            strong_chars = overlap(start, end, strong) if name == "thesis" else 0
            possible_chars = overlap(start, end, possible) if name == "thesis" else 0
            scores = [segment["model_scores"][model]["ai_score"] for model in previous["models"]]
            require(all(math.isfinite(score) and 0 <= score <= 1 for score in scores), "Invalid raw model score")
            unit = next((p["unit"] for p in mapping["paragraphs"] if p["char_start"] <= (start + end) // 2 < p["char_end"]), None)
            record = {"document": name, "unit": unit, "id": segment["id"], "chars": chars,
                      "strong": strong_chars / chars, "possible": possible_chars / chars, "raw": scores}
            require(record == old_records[(name, segment["id"])], "Audited labels or scores differ from v1")
            record.update(char_start=start, char_end=end, strong_characters=strong_chars,
                          possible_characters=possible_chars, any_characters=strong_chars + possible_chars,
                          cnki_span_ids=[s["cnki_id"] for s in spans if name == "thesis" and s["char_start"] < end and s["char_end"] > start])
            memberships = {}
            for paragraph in mapping["paragraphs"]:
                count = max(0, min(end, paragraph["char_end"]) - max(start, paragraph["char_start"]))
                if count:
                    unit_name = paragraph["unit"]
                    memberships[unit_name] = memberships.get(unit_name, 0) + count
            require(memberships and sum(memberships.values()) <= chars, "Invalid paragraph unit coverage")
            record["unit_memberships"] = [{"unit": unit_name, "characters": count}
                                           for unit_name, count in sorted(memberships.items())]
            if len(memberships) > 1:
                boundary_records.append({"id": segment["id"], "chars": chars, "legacy_midpoint_unit": unit,
                                         "unit_memberships": record["unit_memberships"],
                                         "strong_characters": strong_chars, "possible_characters": possible_chars})
            records.append(record)
        require(cursor == len(text) == raw["input"]["characters"] == source["input_characters"], "Inference coverage is incomplete")
        source_audits[name] = {"characters": len(text), "segments": len(ids), "complete_contiguous_coverage": True,
                              "unique_segment_ids": True, "raw_scores_unchanged_from_v1": True,
                              "reference_fractions_unchanged_from_v1": True,
                              "cross_unit_segments": boundary_records}
    require(len(records) == len(old_records), "Reference records were lost")
    require(sum(r["strong_characters"] for r in records) == length(strong), "Strong label total differs")
    require(sum(r["possible_characters"] for r in records) == length(possible), "Possible label total differs")
    provider_counts = {c: sum(s["provider_characters"] for s in report["segments"] if s["decision"]["category"] == c)
                       for c in ("strong_signal", "possible_signal")}
    provider = rates(provider_counts["strong_signal"], provider_counts["possible_signal"], report["summary"]["total_characters"])
    provider.update(strong_displayed_percent=report["summary"]["percent"],
                    possible_and_any_percent_are_derived=True, possible_counted_in_report_headline=False)
    same_scope = rates(length(strong), length(possible), len(thesis))
    anchor_counts = {c: sum(s["anchor_union_characters"] for s in span_audits if s["category"] == c)
                     for c in ("strong_signal", "possible_signal")}
    anchors = rates(anchor_counts["strong_signal"], anchor_counts["possible_signal"], len(thesis))
    dataset = copy.deepcopy(previous)
    dataset["schema_version"] = "aigc-reference-scores/2.0"
    dataset["privacy"] = "Raw model scores, numeric reference labels, offsets, hashes and provenance metadata only; no private prose."
    dataset["records"] = records
    reference = dataset["cnki_reference"]
    reference.update(provider_possible_characters=provider["possible_only_characters"],
                     provider_possible_percent=provider["possible_only_percent"], provider_any_characters=provider["any_characters"],
                     provider_any_percent=provider["any_percent"], same_scope_possible_characters=same_scope["possible_only_characters"],
                     same_scope_possible_percent=same_scope["possible_only_percent"], same_scope_any_characters=same_scope["any_characters"],
                     same_scope_any_percent=same_scope["any_percent"], provider=provider, same_scope=same_scope)
    dataset["label_contract"] = {"strong": "Fraction of segment characters overlapping mapped CNKI strong spans.",
        "possible": "Fraction overlapping mapped CNKI possible-only spans; disjoint from strong.",
        "review_target": "strong + possible; output possible-only must exclude output strong.",
        "unmarked": "CNKI comparator-unmarked; not verified human authorship.",
        "controls": "Both human controls are user-confirmed; identity is fitting metadata, never a runtime feature.",
        "unit_memberships": "Every paragraph unit intersecting the segment, with intersecting prose character counts. Counts exclude paragraph separators and may sum to less than chars. Use these memberships for grouped diagnostics; unit is only legacy midpoint metadata.",
        "chapter_holdout": "For each held chapter, exclude every intersecting multi-unit segment from training and testing. Test only single-unit held-chapter segments; record purged segments and excluded comparator characters.",
        "intervals": "Zero-based half-open Python Unicode offsets; union counts prevent duplicate labels.",
        "scope": "Frozen current continuous prose, excluding headings and captions; not the provider full-document denominator."}
    dataset["construction"] = {"script": "scripts/build_fusion_dataset_v2.py", "script_sha256": digest(Path(__file__)),
                               "previous_dataset_sha256": digest(previous_path), "private_inputs_required": True,
                               "labels_changed_from_v1": False, "raw_scores_changed_from_v1": False}
    dataset["sources"]["thesis"]["label_source"] = "CNKI significant and possible-only excerpts aligned to current continuous prose; unmarked is comparator-unmarked."
    audit = {"schema_version": "aigc-cnki-label-audit/1.0", "provider": provider, "same_scope": same_scope,
        "source_hashes": {"v1_dataset": digest(previous_path), "alignment": digest(alignment_path), "cnki_report": digest(report_path),
                          "builder": digest(Path(__file__)), "cnki_importer": digest(ROOT / "scripts/import_cnki.py")},
        "provider_evidence": provider_evidence, "documents": source_audits, "alignment_spans": span_audits,
        "counts": {"strong_spans": len(strong), "possible_spans": len(possible), "duplicate_span_ids": 0,
                   "overlapping_mapped_characters": 0, "unmapped_cnki_segments": 0, "records": len(records)},
        "anchor_union_sensitivity": {"same_scope": anchors,
            "extra_continuous_interval_characters": same_scope["any_characters"] - anchors["any_characters"],
            "possible_only_percent_difference": same_scope["possible_only_percent"] - anchors["possible_only_percent"],
            "label_policy": "Retain reviewed continuous v1 intervals. Anchor-only union is a sensitivity diagnostic, not a second fitted target."},
        "review": {"low_han_coverage_span_ids": [s["cnki_id"] for s in spans if s["requires_extraction_scope_review"]],
            "retained_alignment": True, "all_longest_anchors_unique": True,
            "boundary_notes": ["Spans 7, 12 and 14 contain omitted figure captions or headings; only present prose is mapped.",
                "Span 4 includes a short unmatched leading prose fragment as well as a heading and a PDF artifact; it is not reconstructed.",
                "Matched intervals end at Han anchors and may exclude adjacent punctuation; headings can align to repeated short words.",
                "The submitted PDF identity is unconfirmed, so not every difference is attributable to extraction alone."],
            "private_detail_path": "reports/fusion-20260929-v2/labels/alignment-details.json"},
        "evaluation_guidance": ["Report possible-only and any-flag rates separately, using the same denominator.",
            "Chapter holdouts must use all unit_memberships and purge intersecting cross-unit segments; midpoint-only grouping leaks boundary passages.",
            "Measure CNKI possible-span overlap under both possible-only predictions and any flagged predictions; report promotions to strong separately.",
            "Report character overlap recall, flagged-character overlap precision and excerpt hit counts, without calling these accuracy.",
            "Rate matching alone does not establish position agreement or generalization; these references participate in fitting."],
        "probability_calibrated": False, "independent_validation": False}
    write(destination / "score-dataset.json", dataset)
    audit["dataset_sha256"] = digest(destination / "score-dataset.json")
    write(destination / "label-audit.json", audit)
    write(ROOT / "reports/fusion-20260929-v2/labels/alignment-details.json", {"private": True, "segments": private_details})
    return {"dataset_sha256": audit["dataset_sha256"], "records": len(records), "provider": provider,
            "same_scope": same_scope, "anchor_union_sensitivity": audit["anchor_union_sensitivity"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, default=ROOT / "reports/fusion-20260929")
    parser.add_argument("--previous-dataset", type=Path, default=ROOT / "evaluations/thesis-cnki-20260929/score-dataset.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluations/thesis-cnki-20260929-v2")
    args = parser.parse_args()
    print(json.dumps(build(args.private_root, args.previous_dataset, args.output), ensure_ascii=False, indent=2))
