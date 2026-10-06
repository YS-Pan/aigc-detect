"""Read the frozen three-type corpus without changing its owning workspace.

All text and character maps belong in ignored reports/. This adapter is not a
detector: known labels are projected after mechanical extraction, never supplied
to segmentation or inference. Unicode offsets are zero-based and half-open.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\project\20260911 dissertation\side quest\英文AIGC检测校准样本")
BODY_KINDS = {"abstract", "prose", "list"}
BLOCK = re.compile(r"% BEGIN BLOCK (\S+) kind=(\S+)[^\r\n]*\r?\n(.*?)\r?\n% END BLOCK \1", re.S)
TOPICS = dict(zip((f"P{i:02}" for i in range(1, 31)), ["antenna"] * 10 + [
    "microplasma", "electrochemical_storage", "acoustic_metasurface", "heat_pipe",
    "microfluidic_cooling", "fiber_current_sensor", "medical_machine_learning",
    "MEMS_anemometry", "food_chemistry", "ice_crystal_growth", "silver_electrodeposition",
    "calcium_sulfate_crystals", "additive_manufacturing", "molecular_spectroscopy",
    "fiber_grating", "dielectric_capacitors", "constrained_optimization", "autonomous_gliding",
    "transformer_circuit", "arc_nitrogen_fixation"]))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes().decode("utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def union(spans):
    out = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def overlap(spans, a, b):
    return sum(max(0, min(b, z) - max(a, x)) for x, z in union(spans))


def runs(indices):
    out = []
    for i in indices:
        if out and i == out[-1][1]:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def plain_with_offsets(latex, base=0):
    """The historical plain_text transform, with one source offset per output char.

    Replacement whitespace inherits the first replaced source offset; document
    separators have no source origin. Retained math/citation resources are removed
    identically in both labels. No language rewriting or score-based cleanup.
    """
    value, positions = latex, list(range(base, base + len(latex)))

    def sub(pattern, replacement, flags=0, group=None):
        nonlocal value, positions
        parts, mapped, end = [], [], 0
        for m in re.finditer(pattern, value, flags):
            parts.append(value[end:m.start()]); mapped.extend(positions[end:m.start()])
            new = m.group(group) if group else replacement
            parts.append(new)
            mapped.extend(positions[m.start(group):m.end(group)] if group else [positions[m.start()]] * len(new))
            end = m.end()
        parts.append(value[end:]); mapped.extend(positions[end:])
        value, positions = "".join(parts), mapped

    sub(r"(?<!\\)%[^\r\n]*", "")
    sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", re.S)
    sub(r"\\(?:begin|end)\{[^}]+\}", "")
    sub(r"\\(?:cite\w*|ref|eqref|label)\*?(?:\[[^]]*\])?\{[^}]*\}", "")
    sub(r"\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]", "")
    sub(r"\\([%&_#{}])", "", group=1)
    sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", "")
    for char in ["\\", "{", "}"]:
        sub(re.escape(char), "")
    for a, b in [("~", " "), ("``", '"'), ("''", '"'), ("---", "—"), ("--", "–")]:
        sub(re.escape(a), b)
    sub(r"\s+", " ")
    left, right = len(value) - len(value.lstrip()), len(value.rstrip())
    return value[left:right], positions[left:right]


def normalized_author(value):
    value = unicodedata.normalize("NFKD", value).lower()
    surname, given = (value.split(",", 1) + [""])[:2]
    surname = re.sub(r"[^a-z]", "", surname)
    tokens = re.findall(r"[a-z]+", given)
    full = "".join(tokens)
    if not surname or not full:
        raise ValueError("Incomplete author identity")
    return surname, full, "".join(t[0] for t in tokens), all(len(t) == 1 for t in tokens)


def related_authors(left, right):
    for a in left:
        for b in right:
            if a[0] != b[0]:
                continue
            if a[1] == b[1]:
                return True
            if a[3] or b[3]:
                short, full = (a, b) if a[3] else (b, a)
                if full[2].startswith(short[2]):
                    return True
    return False


def components(papers, edges=()):
    authors = {p["paper_id"]: {normalized_author(a) for a in p["author"].split(" and ")} for p in papers}
    groups = [{p} for p in authors]
    edges = [set(e) for e in edges]
    while True:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                linked = related_authors(set().union(*(authors[p] for p in groups[i])), set().union(*(authors[p] for p in groups[j])))
                if linked or any(e & groups[i] and e & groups[j] for e in edges):
                    groups[i] |= groups.pop(j); merged = True; break
            if merged:
                break
        if not merged:
            return sorted([sorted(g) for g in groups])


def checked_span(text, span):
    a, b = span["start_char"], span["end_char"]
    if not 0 <= a < b <= len(text) or digest(text[a:b]) != span["sha256"]:
        raise ValueError("Invalid frozen span/hash")
    for key, at in [("start_line", a), ("end_line", b - 1)]:
        if key in span and text.count("\n", 0, at) + 1 != span[key]:
            raise ValueError("Invalid frozen line number")
    return a, b


def extract_document(tex, annotation=None):
    annotation = annotation or {}
    generated = []
    for g in annotation.get("aigc_spans", []):
        a, b = checked_span(tex, g["generated_span"])
        unit = g.get("slot_id", g.get("paragraph_id", g.get("unit_id")))
        generated.append((a, b, unit))
    parts, origins, block_map, cursor = [], [], [], 0
    for m in BLOCK.finditer(tex):
        if m.group(2) not in BODY_KINDS:
            continue
        text, offsets = plain_with_offsets(m.group(3), m.start(3))
        if not text:
            continue
        if parts:
            origins.extend([None, None]); cursor += 2
        parts.append(text); origins.extend(offsets)
        block_map.append({"id": m.group(1), "kind": m.group(2), "start": cursor, "end": cursor + len(text),
                          "tex_start": m.start(3), "tex_end": m.end(3)})
        cursor += len(text)
    body = "\n\n".join(parts)
    unit_positions = defaultdict(list)
    for i, src in enumerate(origins):
        if src is not None:
            matches = [uid for a, b, uid in generated if a <= src < b]
            if len(matches) > 1:
                raise ValueError("Overlapping generated labels")
            if matches:
                unit_positions[matches[0]].append(i)
    expected = {uid for _, _, uid in generated}
    if expected != set(unit_positions):
        raise ValueError("A generated narrative unit was excluded from body extraction")
    units = [{"id": uid, "spans": runs(ids)} for uid, ids in sorted(unit_positions.items())]
    ai = union(s for u in units for s in u["spans"])
    # Every excluded generated character must be syntax/math/whitespace removed
    # by the same transform; no generated language in captions/headings is lost.
    selected = [(b["tex_start"], b["tex_end"]) for b in block_map]
    for a, b, _ in generated:
        residual = "".join(tex[i] for i in range(a, b) if not any(x <= i < z for x, z in selected))
        if plain_with_offsets(residual)[0]:
            raise ValueError("Generated prose exists outside selected body kinds")
    return {"text": body, "sha256": digest(body), "characters": len(body), "blocks": block_map,
            "origins": origins, "ai_spans": ai, "units": units}


def load_corpus(source=SOURCE):
    """Validate both schemas, full frozen documents and exact label spans."""
    source = Path(source).resolve(strict=True)
    files = {}
    def load(rel, expected=None, binary=False):
        raw = (source / rel).read_bytes()
        h = digest(raw)
        if expected and h != expected:
            raise ValueError("Upstream file hash mismatch: " + rel)
        files[rel] = {"sha256": h, "bytes": len(raw)}
        return raw if binary else raw.decode("utf-8-sig")
    papers = json.loads(load("sources/manifest.json"))["papers"]
    index = json.loads(load("corpus/aigc-index.json"))
    variants = [(v, "paragraph_compress_restore") for v in index["variants"]]
    for entry in index["additional_batches"]:
        extra = json.loads(load(entry["index"]))
        variants += [(v, v["sample_type"]) for v in extra["variants"]]
    documents, source_bodies, source_tex = [], {}, {}
    for p in papers:
        pid = p["paper_id"]
        content = json.loads(load(f"corpus/original/{pid}/content.json"))
        tex = load(f"corpus/original/{pid}/full.tex", content["full_tex_sha256"])
        load(p["source_pdf"], p["source_sha256"], binary=True)
        body = extract_document(tex)
        source_bodies[pid], source_tex[pid] = body, tex
        documents.append({**body, "id": pid + "-original", "paper_id": pid, "sample_type": "original",
                          "generator": None, "topic": TOPICS[pid], "source_file": f"corpus/original/{pid}/full.tex",
                          "section_units": [], "annotation_file": None})
    for v, kind in variants:
        pid = v["paper_id"]
        path = v.get("variant_file", v.get("variant_full_tex"))
        expected = v.get("variant_sha256", v.get("variant_full_tex_sha256"))
        tex = load(path, expected)
        annotation_file = str(Path(path).with_name("annotations.json")).replace("\\", "/")
        annotation = json.loads(load(annotation_file))
        if annotation["aigc_spans"] != v["aigc_spans"]:
            raise ValueError("Index and annotations disagree")
        for g in v["aigc_spans"]:
            for span in g.get("source_spans", [g["original_span"]] if "original_span" in g else []):
                checked_span(source_tex[pid], span)
        for s in v.get("retained_spans", []):
            a, b = checked_span(tex, s["generated_span"])
            x, z = checked_span(source_tex[pid], s["original_span"])
            if tex[a:b] != source_tex[pid][x:z]:
                raise ValueError("Retained resource changed")
        # Reverse all replacements, including multi-fragment infill removals.
        units = v.get("replacement_units") or [dict(g, source_spans=[g["original_span"]]) for g in v["aigc_spans"]]
        replacements = []
        for u in units:
            a, b = checked_span(tex, u["generated_span"])
            spans = u.get("source_spans", [u["original_span"]])
            for s in spans:
                checked_span(source_tex[pid], s)
            original = source_tex[pid][u["original_span"]["start_char"]:u["original_span"]["end_char"]]
            replacements.append((a, b, original))
        restored = tex
        for a, b, original in sorted(replacements, reverse=True):
            restored = restored[:a] + original + restored[b:]
        if restored != source_tex[pid]:
            raise ValueError("Untouched source cannot be recovered: " + path)
        body = extract_document(tex, annotation)
        section_units = [{"id": u["unit_id"], "spans": body["ai_spans"]} for u in units] if kind == "section_compress_restore" else []
        documents.append({**body, "id": f"{pid}-{kind}-{v['model']}", "paper_id": pid, "sample_type": kind,
                          "generator": v["model"], "topic": TOPICS[pid], "source_file": path,
                          "annotation_file": annotation_file, "section_units": section_units})
    audit = {"papers": len(papers), "originals": len(papers), "variants": len(variants),
             "by_type": dict(Counter(d["sample_type"] for d in documents)),
             "narrative_units_by_type": dict(Counter({k: sum(len(d["units"]) for d in documents if d["sample_type"] == k)
                                                        for k in sorted({d["sample_type"] for d in documents})})),
             "section_units": sum(len(d["section_units"]) for d in documents), "source_files": files,
             "read_only_source": str(source), "unit_note": "Section prose_slots are narrative units, not the 1112 resource-split spans or necessarily independent logical paragraphs."}
    return papers, documents, audit


def audit_duplicates(papers, documents):
    owners, labels = defaultdict(set), defaultdict(set)
    for d in documents:
        for b in d["blocks"]:
            value = d["text"][b["start"]:b["end"]]
            key = digest(re.sub(r"\s+", " ", value).strip().lower())
            if len(value) >= 80:
                owners[key].add(d["paper_id"])
                labels[key].add(overlap(d["ai_spans"], b["start"], b["end"]) > 0)
    duplicates = [{"sha256": h, "papers": sorted(ps)} for h, ps in owners.items() if len(ps) > 1]
    source_edges = []
    for field in ["doi", "source_sha256"]:
        same = defaultdict(set)
        for p in papers:
            if p.get(field):
                same[str(p[field]).lower()].add(p["paper_id"])
        source_edges += [sorted(ps) for ps in same.values() if len(ps) > 1]
    return {"cross_paper_body_duplicates": duplicates, "exact_body_label_conflicts": sorted(h for h, ls in labels.items() if len(ls) > 1),
            "same_source_edges": source_edges, "groups": components(papers, source_edges + [v["papers"] for v in duplicates]),
            "policy": "Bind duplicated sources/authors/bodies before scores. Keep label conflicts and every difficult negative; never relabel from scores."}


def previously_seen(papers, documents):
    seen = {f"P{i:02}" for i in range(1, 11)}
    old = ROOT / "samples/english-academic-20261003"
    matches = []
    if (old / "samples.jsonl").exists():
        previous = [json.loads(s) for s in (old / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        hashes = {v["sha256"] for v in previous}
        for d in documents:
            if d["sha256"] in hashes:
                seen.add(d["paper_id"]); matches.append(d["id"])
    extra = []
    for p in sorted((ROOT / "reports/english-frozen-local-20261004/new-sources").glob("*.xml")):
        x = ET.parse(p).getroot()
        authors = {normalized_author((a.findtext("name/surname") or "") + "," + (a.findtext("name/given-names") or ""))
                   for a in x.findall(".//article-meta/contrib-group/contrib") if a.find("name") is not None}
        doi = x.findtext(".//article-meta/article-id[@pub-id-type='doi']")
        related = [r["paper_id"] for r in papers if doi == r.get("doi") or related_authors(authors, {normalized_author(a) for a in r["author"].split(" and ")})]
        seen.update(related); extra.append({"id": p.stem, "sha256": digest(p.read_bytes()), "related_papers": related})
    return {"seen_papers": sorted(seen), "matching_previous_documents": matches, "previous_external_controls": extra,
            "scope": "Existing imported ten-paper collection plus four prior XML controls; no claim about base-model pretraining exposure."}


def conversion_audit(source, papers, documents):
    """Fixed first/middle/last prose blocks; diagnostics never change labels."""
    import fitz
    rows = []
    for p in papers:
        d = next(d for d in documents if d["paper_id"] == p["paper_id"] and d["sample_type"] == "original")
        blocks = [b for b in d["blocks"] if b["kind"] == "prose" and b["end"] - b["start"] >= 200]
        chosen = [blocks[i] for i in sorted({0, len(blocks) // 2, len(blocks) - 1})]
        meta = {b["id"]: b for b in read(Path(source) / f"corpus/original/{p['paper_id']}/content.json")["blocks"]}
        pdf = fitz.open(Path(source) / p["source_pdf"])
        for b in chosen:
            page = meta[b["id"]]["page"]
            human = d["text"][b["start"]:b["end"]]
            mechanical = pdf[page - 1].get_text("text", sort=False)
            token = lambda t: re.findall(r"[a-z]+", unicodedata.normalize("NFKC", t).lower())
            a, z = token(human), token(mechanical)
            # Fraction matched in order anywhere on its source page; this cannot
            # prove complete or semantically correct transcription.
            matched = sum(m.size for m in SequenceMatcher(None, a, z, autojunk=False).get_matching_blocks())
            rows.append({"paper_id": p["paper_id"], "block_id": b["id"], "source_page": page,
                         "body_sha256": digest(human), "words": len(a), "source_page_word_match_fraction": matched / max(1, len(a))})
        pdf.close()
    return {"selection": "First, middle and last prose block >=200 chars per paper, chosen before scores",
            "rows": rows, "scope": "Mechanical page-order token comparison only; neither full visual proofreading nor semantic certification. Keep conversion uncertainty and labels separate."}
