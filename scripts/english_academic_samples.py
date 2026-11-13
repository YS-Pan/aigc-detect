"""Import and verify the user-selected, local English academic calibration corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "samples/english-academic-20261003"
SOURCE_THREAD = "01a0ffb9-3c3a-74e1-98f9-6c753fa47eaf"
SEED = 20261003


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_json(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def plain_text(latex):
    """Fixed mechanical extraction, used identically for both source classes.

    Inline math and numeric citation markers are omitted. No language editing.
    Mapping remains at the block level because removal changes character offsets.
    """
    value = re.sub(r"(?<!\\)%[^\r\n]*", "", latex)
    value = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", value, flags=re.S)
    value = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", value)
    value = re.sub(r"\\(?:cite\w*|ref|eqref|label)\*?(?:\[[^]]*\])?\{[^}]*\}", "", value)
    value = re.sub(r"\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]", "", value)
    value = re.sub(r"\\([%&_#{}])", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", "", value)
    value = value.replace("\\", "").replace("{", "").replace("}", "").replace("~", " ")
    value = value.replace("``", '"').replace("''", '"').replace("---", "—").replace("--", "–")
    return re.sub(r"\s+", " ", value).strip()


def checked_text(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"Source file hash mismatch: {path}")
    return raw.decode("utf-8-sig")


def checked_span(text, span):
    value = text[span["start_char"]:span["end_char"]]
    if digest(value.encode("utf-8")) != span["sha256"]:
        raise ValueError("Source span hash mismatch")
    return value


def document_text(upstream, paper, variant=None):
    pid = paper["paper_id"]
    content = read_json(upstream / f"corpus/original/{pid}/content.json")
    source_file = variant["variant_file"] if variant else f"corpus/original/{pid}/full.tex"
    expected = variant["variant_sha256"] if variant else content["full_tex_sha256"]
    tex = checked_text(upstream / source_file, expected)
    blocks = {m.group(1): (m.group(2), m.start(2), m.end(2)) for m in re.finditer(
        r"% BEGIN BLOCK (\S+)[^\r\n]*\r?\n(.*?)\r?\n% END BLOCK \1", tex, re.S)}
    generated = {s["paragraph_id"]: s for s in variant["aigc_spans"]} if variant else {}
    parts, mapping, cursor = [], [], 0
    for b in content["blocks"]:
        if b["kind"] not in ("prose", "abstract"):
            continue
        raw, start, end = blocks[b["id"]]
        if b["id"] in generated:
            span = generated[b["id"]]["generated_span"]
            raw = checked_span(tex, span)
            start, end = span["start_char"], span["end_char"]
        text = plain_text(raw)
        if not text:
            continue
        if parts:
            cursor += 2
        mapping.append({"block_id": b["id"], "source_page": b["page"], "kind": b["kind"],
                        "complete_paragraph": b.get("complete_paragraph", False),
                        "char_start": cursor, "char_end": cursor + len(text),
                        "source_file": source_file, "source_char_start": start, "source_char_end": end,
                        "source_span_sha256": digest(raw.encode("utf-8")),
                        "source_label": "ai" if b["id"] in generated else "human"})
        parts.append(text)
        cursor += len(text)
    if set(generated) - {b["block_id"] for b in mapping}:
        raise ValueError("A generated block was excluded from extraction")
    return "\n\n".join(parts), mapping, source_file


def collect(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    source = source.resolve(strict=True)
    destination.mkdir(parents=True)
    upstream = destination / "upstream"
    # Preserve every source/experiment file, excluding reproducible build/render
    # output and temporary work. Do not touch the source workspace.
    shutil.copytree(source, upstream, ignore=shutil.ignore_patterns("temp", "build", "render", "__pycache__"))
    files = [{"path": p.relative_to(upstream).as_posix(), "bytes": p.stat().st_size,
              "sha256": digest(p.read_bytes())} for p in sorted(upstream.rglob("*")) if p.is_file()]
    write_json(destination / "sources/import-manifest.json", {
        "source_thread": SOURCE_THREAD, "source_directory": str(source), "files": files,
        "excluded_directories": ["temp", "build", "render", "__pycache__"],
        "license": "Local user-authorized research copy; no public redistribution permission inferred."})
    papers = read_json(upstream / "sources/manifest.json")["papers"]
    index = read_json(upstream / "corpus/aigc-index.json")
    for p in papers:
        checked_text(upstream / f"corpus/original/{p['paper_id']}/full.tex",
                     read_json(upstream / f"corpus/original/{p['paper_id']}/content.json")["full_tex_sha256"])
        if digest((upstream / p["source_pdf"]).read_bytes()) != p["source_sha256"]:
            raise ValueError("PDF source hash mismatch")
    # Commit the paper split before loading or observing any detector scores.
    order = random.Random(SEED).sample(sorted(p["paper_id"] for p in papers), len(papers))
    groups = {"fit": order[:6], "calibration": order[6:8], "test": order[8:]}
    split = {pid: group for group, ids in groups.items() for pid in ids}
    protocol = {"schema_version": "english-academic-protocol/1.0", "seed": SEED, "groups": groups,
                "split_unit": "entire source paper and all its derived variants/paragraphs",
                "selection_before_detector_scores": True, "models": ["hc3-en", "desklib-en", "fakespot-en", "bibbbu-xlm-roberta"],
                "segment_chars": 400, "min_segment_chars": 120,
                "fit_records": "Human full-body segments and isolated generated-paragraph segments; mixed documents are evaluation only.",
                "candidate_regularization": [0.1, 1.0, 10.0, 100.0],
                "candidate_human_weights": [1.0, 5.0, 20.0],
                "feature_family": "Regularized linear fusion of standardized logits of four raw model scores only.",
                "selection": "Paper-grouped leave-one-paper-out on fit only; maximize generated-character recall under zero human flags using training-only thresholds. Tie: more regularization, more human weight.",
                "threshold_selection": "Maximum human score from fit+calibration plus predeclared logit margins 0.5 (review) and 1.0 (strong); calibration AI scores are diagnostic only.",
                "objective": "Zero observed human flags on fit/calibration; maximize recall subject to that constraint.",
                "test_policy": "Freeze profile before reading held-out test metrics; never adjust it using test results.",
                "uncertainty": "Ten papers in one subject; zero observed flags does not prove a zero population false-positive rate."}
    write_json(destination / "protocol.json", protocol)
    samples = []

    def save(sid, text, metadata, mapping=None):
        relative = f"texts/{sid}.txt"
        target = destination / relative
        target.parent.mkdir(exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        record = {"id": sid, "text_file": relative, "sha256": digest(text.encode("utf-8")), "characters": len(text), **metadata}
        if mapping is not None:
            record["mapping_file"] = f"maps/{sid}.json"
            write_json(destination / record["mapping_file"], {"input_sha256": record["sha256"], "blocks": mapping,
                                                          "offset_convention": "Python Unicode, zero-based half-open; source mapping at block level"})
        samples.append(record)

    for p in papers:
        pid = p["paper_id"]
        text, mapping, filename = document_text(upstream, p)
        metadata = {"paper_id": pid, "split": split[pid], "source_title": p["title"], "source_year": p["year"],
                    "source_doi": p["doi"], "source_file": filename}
        save(f"{pid}-human", text, {**metadata, "kind": "human_document", "source_label": "human"}, mapping)
        for v in [v for v in index["variants"] if v["paper_id"] == pid]:
            text, mapping, filename = document_text(upstream, p, v)
            save(f"{pid}-{v['model']}-mixed", text,
                 {**metadata, "source_file": filename, "kind": "mixed_document", "source_label": "mixed", "generator": v["model"]}, mapping)
            variant = checked_text(upstream / filename, v["variant_sha256"])
            original = checked_text(upstream / v["original_file"], v["original_sha256"])
            for span in v["aigc_spans"]:
                checked_span(original, span["original_span"])
                text = plain_text(checked_span(variant, span["generated_span"]))
                save(f"{span['paragraph_id']}-{v['model']}-ai", text,
                     {**metadata, "source_file": filename, "kind": "ai_paragraph", "source_label": "ai", "generator": v["model"],
                      "paragraph_id": span["paragraph_id"], "source_span": span["generated_span"], "original_span": span["original_span"]})
    raw = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples).encode("utf-8")
    (destination / "samples.jsonl").write_bytes(raw)
    write_json(destination / "collection.json", {"status": "complete", "source_thread": SOURCE_THREAD,
                "samples_sha256": digest(raw), "protocol_sha256": digest((destination / "protocol.json").read_bytes()),
                "import_manifest_sha256": digest((destination / "sources/import-manifest.json").read_bytes()),
                "source_papers": len(papers), "human_documents": 10, "mixed_documents": 30, "ai_paragraphs": 75,
                "extraction": "prose+abstract blocks in source order; remove inline math, numeric citations and TeX commands; collapse whitespace within each block; join blocks with two LF characters",
                "privacy": "Entire batch ignored by Git. Source human labels derive from published papers; conversion errors are possible."})
    verify(destination)
    print(json.dumps({"status": "complete", "samples": len(samples), "groups": groups, "imported_files": len(files)}, ensure_ascii=False))


def verify(directory):
    collection = read_json(directory / "collection.json")
    for name, field in [("samples.jsonl", "samples_sha256"), ("protocol.json", "protocol_sha256"),
                        ("sources/import-manifest.json", "import_manifest_sha256")]:
        if digest((directory / name).read_bytes()) != collection[field]:
            raise ValueError(f"Collection hash mismatch: {name}")
    for f in read_json(directory / "sources/import-manifest.json")["files"]:
        if digest((directory / "upstream" / f["path"]).read_bytes()) != f["sha256"]:
            raise ValueError(f"Imported file changed: {f['path']}")
    records = [json.loads(line) for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    protocol = read_json(directory / "protocol.json")
    for r in records:
        raw = (directory / r["text_file"]).read_bytes()
        if digest(raw) != r["sha256"] or len(raw.decode("utf-8")) != r["characters"]:
            raise ValueError(f"Sample hash mismatch: {r['id']}")
        if r["paper_id"] not in protocol["groups"][r["split"]]:
            raise ValueError("Paper split leakage")
        r["text"] = raw.decode("utf-8")
        if "mapping_file" in r:
            mapping = read_json(directory / r["mapping_file"])
            if mapping["input_sha256"] != r["sha256"]:
                raise ValueError("Mapping hash mismatch")
            for b in mapping["blocks"]:
                source = (directory / "upstream" / b["source_file"]).read_bytes().decode("utf-8-sig")
                raw_block = source[b["source_char_start"]:b["source_char_end"]]
                if digest(raw_block.encode("utf-8")) != b["source_span_sha256"] or plain_text(raw_block) != r["text"][b["char_start"]:b["char_end"]]:
                    raise ValueError("Extracted block differs from recorded source")
        elif plain_text(checked_span((directory / "upstream" / r["source_file"]).read_bytes().decode("utf-8-sig"), r["source_span"])) != r["text"]:
            raise ValueError("Generated paragraph differs from source")
    return records, collection, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "verify"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--samples", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if args.command == "collect":
        if args.source is None:
            parser.error("collect requires --source")
        collect(args.source, args.samples)
    else:
        records, _, protocol = verify(args.samples)
        print(json.dumps({"status": "verified", "samples": len(records), "groups": protocol["groups"]}))


if __name__ == "__main__":
    main()
