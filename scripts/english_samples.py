"""Collect and verify source-labelled English examples; never infer source labels."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.catalog import ROOT
from scripts.network import EVENTS, download

HC3_REVISION = "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3"
DEFAULT_DIR = ROOT / "samples/english-public-20261002"
LOCAL_TZ = timezone(timedelta(hours=8))
BOOKS = (
    (1342, "Jane Austen", "Pride and Prejudice", 1813, "It is a truth universally acknowledged"),
    (1228, "Charles Darwin", "On the Origin of Species", 1859, "When on board H.M.S."),
    (84, "Mary Wollstonecraft Shelley", "Frankenstein", 1818, "You will rejoice to hear"),
    (205, "Henry David Thoreau", "Walden", 1854, "When I wrote the following pages"),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(directory, cached_sources=None):
    directory.mkdir(parents=True, exist_ok=False)
    sources = directory / "sources"
    sources.mkdir()
    (directory / "texts").mkdir()
    evidence, records = [], []

    def fetch(name, url):
        path = sources / name
        cached = cached_sources / name if cached_sources else None
        if cached and cached.is_file():
            # Reuse the snapshot fetched during this request, preserving its bytes.
            shutil.copyfile(cached, path)
        else:
            download(url, path)
        raw = path.read_bytes()
        evidence.append({"path": "sources/" + name, "url": url, "bytes": len(raw), "sha256": digest(raw)})
        if name.startswith("gutenberg-"):
            evidence[-1]["complete_work_archived"] = bool(re.search(
                r"\*{3}\s*END OF (?:THE )?PROJECT GUTENBERG", raw.decode("utf-8-sig"), re.I))
            evidence[-1]["snapshot_scope"] = "retrieved bytes; only the selected complete body excerpt is required"
        return raw

    def append(sample_id, text, source_label, source_group, source):
        path = "texts/" + sample_id + ".txt"
        (directory / path).write_bytes(text.encode("utf-8"))
        records.append({"id": sample_id, "text": text, "text_file": path, "language": "en",
                        "source_label": source_label, "source_group": source_group, "source": source,
                        "characters": len(text), "sha256": digest(text.encode("utf-8"))})

    for ebook_id, author, title, year, anchor in BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        raw = fetch(f"gutenberg-{ebook_id}.txt", url)
        body = raw.decode("utf-8-sig")  # Preserve original CRLF and all other characters.
        start = body.find(anchor)
        if start < 0 or body.count(anchor) != 1:
            raise ValueError(f"Book anchor is missing or ambiguous: {ebook_id}")
        end_cap = min(start + 4000, len(body))
        ends = [m.end() + start for m in re.finditer(r"[.!?][\"'’”)]*(?=\s|$)", body[start:end_cap])]
        end = ends[-1] if ends else end_cap
        if end - start < 3000:
            raise ValueError("Book excerpt is too short; source may be incomplete")
        append(f"gutenberg-{ebook_id}-human", body[start:end], "human", "gutenberg", {
            "url": f"https://www.gutenberg.org/ebooks/{ebook_id}", "download_url": url,
            "snapshot": f"sources/gutenberg-{ebook_id}.txt", "snapshot_sha256": digest(raw),
            "author": author, "title": title, "publication_year": year,
            "label_provenance": "named historical author; original publication predates language models",
            "char_start": start, "char_end": end, "offset_basis": "UTF-8-sig decoded original snapshot, including CRLF",
            "transformation": "contiguous excerpt from the specified unique body anchor; no rewriting or whitespace normalization",
            "license": "Public domain original text; retain Project Gutenberg attribution and terms in the source snapshot",
            "training_overlap": "historical literature may appear in detector/base-model training; not audited",
        })

    url = f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/{HC3_REVISION}/wiki_csai.jsonl"
    raw = fetch("hc3-wiki_csai.jsonl", url)
    fetch("hc3-data-README.md", f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/{HC3_REVISION}/README.md")
    fetch("hc3-project-README.md", "https://raw.githubusercontent.com/Hello-SimpleAI/chatgpt-comparison-detection/main/README.md")
    pairs = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        answers = {}
        for label, field in (("human", "human_answers"), ("ai", "chatgpt_answers")):
            answers[label] = next(((i, t) for i, t in enumerate(row[field]) if 300 <= len(t) <= 10000), None)
        if not all(answers.values()):
            continue
        pairs += 1
        for label, field in (("human", "human_answers"), ("ai", "chatgpt_answers")):
            index, text = answers[label]
            append(f"hc3-wiki-csai-{pairs:02}-{label}", text, label, "hc3_wiki_csai", {
                "url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3", "download_url": url,
                "revision": HC3_REVISION, "snapshot": "sources/hc3-wiki_csai.jsonl", "snapshot_sha256": digest(raw),
                "line_1based": line_number, "field": field, "answer_index": index, "question": row["question"],
                "label_provenance": "dataset publisher's human_answers / chatgpt_answers field",
                "generator": "ChatGPT (HC3 collection begun December 2022)" if label == "ai" else None,
                "license": "CC BY-SA 4.0 release; Wikipedia source CC BY-SA; Biyang Guo et al. (2023)",
                "transformation": "exact full answer string; question and metadata excluded from inference",
                "training_overlap": "released hc3-en checkpoint trained on all HC3; these are training-overlap smoke examples",
            })
        if pairs == 8:
            break
    if pairs != 8:
        raise ValueError("Could not find eight qualifying HC3 pairs")
    sample_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    (directory / "samples.jsonl").write_bytes(sample_bytes)
    collection = {"schema_version": "aigc-public-samples/1.0", "id": "english-public-20261002",
                  "created_at": datetime.now(LOCAL_TZ).isoformat(), "samples": len(records),
                  "selection_rule": "Four 3000-4000 character Gutenberg body excerpts at fixed unique anchors, then first eight wiki_csai rows with both first qualifying answers of 300-10000 characters, in file order; fixed before scoring",
                  "labels": {label: sum(r["source_label"] == label for r in records) for label in ("human", "ai")},
                  "samples_sha256": digest(sample_bytes), "upstream_evidence": evidence,
                  "network_events": EVENTS, "label_metadata_used_as_model_inputs": False,
                  "limitations": "Selected source examples, including known HC3 training overlap; not an independent accuracy benchmark. No English fusion or threshold fitting."}
    write_json(directory / "collection.json", collection)
    write_json(directory / "COMPLETE.json", {"status": "complete", "samples": len(records), "samples_sha256": digest(sample_bytes)})
    print(f"Collected {len(records)} English examples: {collection['labels']}", flush=True)


def verify(directory):
    collection = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    marker = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    raw = (directory / "samples.jsonl").read_bytes()
    if marker.get("status") != "complete" or digest(raw) != collection["samples_sha256"] or digest(raw) != marker["samples_sha256"]:
        raise ValueError("Incomplete collection or changed sample JSONL")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(records) != 20 or len({r["id"] for r in records}) != 20:
        raise ValueError("Unexpected sample count or duplicate sample IDs")
    for item in collection["upstream_evidence"]:
        content = (directory / item["path"]).read_bytes()
        if len(content) != item["bytes"] or digest(content) != item["sha256"]:
            raise ValueError("Changed upstream snapshot: " + item["path"])
    for record in records:
        text = record["text"]
        content = (directory / record["text_file"]).read_bytes()
        if content != text.encode("utf-8") or digest(content) != record["sha256"] or len(text) != record["characters"]:
            raise ValueError("Text file, JSONL and hash differ: " + record["id"])
        source = record["source"]
        snapshot = (directory / source["snapshot"]).read_bytes()
        if digest(snapshot) != source["snapshot_sha256"]:
            raise ValueError("Source snapshot hash changed")
        if record["source_group"] == "gutenberg":
            expected = snapshot.decode("utf-8-sig")[source["char_start"]:source["char_end"]]
            label = "human"
        else:
            row = json.loads(snapshot.decode("utf-8").splitlines()[source["line_1based"] - 1])
            expected = row[source["field"]][source["answer_index"]]
            label = {"human_answers": "human", "chatgpt_answers": "ai"}[source["field"]]
        if expected != text or label != record["source_label"]:
            raise ValueError("Text or label does not match publisher source")
    if {label: sum(r["source_label"] == label for r in records) for label in ("human", "ai")} != {"human": 12, "ai": 8}:
        raise ValueError("Source label count mismatch")
    print("Verified 20 UTF-8 texts, publisher fields, source positions and snapshot hashes", flush=True)
    return records, collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "verify"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--cached-sources", type=Path, help="Reuse source snapshots downloaded during the current collection")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.directory, args.cached_sources)
    verify(args.directory)


if __name__ == "__main__":
    main()
