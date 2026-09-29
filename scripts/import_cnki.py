"""Import the supplied CNKI PDF bundle locally; never submit text to a service.

The parser supports the supplied Chinese personal-service report layout. It
fails on inconsistent totals and preserves every PDF page for audit. Original
PDF bytes are copied, not edited. PDF offsets refer to the extracted snapshot,
not to the provider's proprietary character counter or the submitted PDF.
"""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import fitz


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, data):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data), encoding="utf-8")


def page_records(path):
    records = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    lines.append({"text": "".join(s["text"] for s in line["spans"]),
                                  "bbox": list(line["bbox"]), "direction": list(line["dir"]),
                                  "spans": [{"text": s["text"], "bbox": list(s["bbox"]),
                                             "color_rgb": f"#{s['color']:06x}", "font_size": s["size"]}
                                            for s in line["spans"]]})
            records.append({"page": index + 1, "width": page.rect.width, "height": page.rect.height,
                            "text": page.get_text(), "lines": lines})
    return records


def clean_lines(pages):
    for page in pages:
        for line in page["lines"]:
            text = line["text"].strip()
            if abs(line["direction"][1]) > 0.1 or re.fullmatch(r"[—-]\s*\d+\s*[—-]", text):
                continue
            if text in {"https://cx.cnki.net", "知网个人AIGC检测服务", "知网个人查重服务", "官方网址 cx.cnki.net"}:
                continue
            if text:
                yield {**line, "page": page["page"]}


def clean_text(pages):
    return "\n".join(line["text"].strip() for line in clean_lines(pages))


def aigc_summary(pages):
    text = clean_text(pages)
    headline = re.search(r"AI特征值：\s*([\d.]+)%\s*AI特征字符数：\s*(\d+)\s*总字符数：\s*(\d+)", text)
    if not headline:
        raise ValueError("CNKI AIGC headline not found")
    percent, marked, total = headline.groups()
    table = text[text.index("分段检测结果"):text.index("1. 中英文摘要等")]
    sections = [{"index": int(i), "percent": float(p), "marked_characters": int(m),
                 "total_characters": int(t), "name": name.strip()}
                for i, p, m, t, name in re.findall(r"(?m)^(\d+)\n([\d.]+)%\n(\d+) / (\d+)\n([^\n]+)", table)]
    if not sections or sum(s["total_characters"] for s in sections) != int(total):
        raise ValueError("AIGC section character totals disagree")
    if sum(s["marked_characters"] for s in sections) != int(marked):
        raise ValueError("AIGC marked section totals disagree")
    return {"percent": float(percent), "marked_characters": int(marked), "total_characters": int(total)}, sections


def aigc_excerpts(pages, sections):
    lines = list(clean_lines(pages))
    text = "\n".join(line["text"].strip() for line in lines)
    offsets = []; cursor = 0
    for line in lines:
        offsets.append((cursor, cursor + len(line["text"].strip()), line))
        cursor += len(line["text"].strip()) + 1
    pattern = r"NO\.(\d+)\n(片段\d+)\n字符数：(\d+)\nAI特征：(显著|疑似)\n([\d.]+)%\n"
    matches = list(re.finditer(pattern, text))
    results = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        tail = text[match.end():end]
        stop = re.search(r"(?m)^\d+\. (?:中英文摘要等|第[^\n]+)\nAI特征值：|^说明：", tail)
        if stop:
            end = match.end() + stop.start()
        body = text[match.end():end].strip()
        prior_sections = list(re.finditer(r"(?m)^(\d+)\. (?:中英文摘要等|第[^\n]+)\nAI特征值：", text[:match.start()]))
        section = int(prior_sections[-1][1])
        locations = [{"page": line["page"], "bbox": line["bbox"], "text": line["text"]}
                     for a, b, line in offsets if a < end and b > match.end()]
        results.append({"id": f"cnki-{int(match[1]):04}", "index": int(match[1]) - 1,
                        "section_index": section, "section_name": sections[section - 1]["name"],
                        "provider_name": match[2], "provider_characters": int(match[3]),
                        "decision": {"category": "strong_signal" if match[4] == "显著" else "possible_signal",
                                     "provider_label": match[4], "counts_toward_aigc_percent": match[4] == "显著"},
                        "percent_of_section": float(match[5]), "text": body,
                        "text_sha256": sha(body.encode()), "source_locations": locations,
                        "char_start": None, "char_end": None})
    if [s["index"] for s in results] != list(range(len(results))):
        raise ValueError("Non-contiguous AIGC excerpt identifiers")
    return results


def aigc_body(pages):
    lines = list(clean_lines(pages)); body = []; active = False; section = None
    for index, line in enumerate(lines):
        text = line["text"].strip()
        next_text = lines[index+1]["text"].strip() if index+1 < len(lines) else ""
        if re.match(r"^\d+\. (?:中英文摘要等|第)", text) and next_text.startswith("AI特征值"):
            section = int(text.split(".", 1)[0]); active = False
        if text == "说明：":
            active = False
        if text == "原文内容":
            active = True; continue
        # CNKI prints the percentage/count in the page margin, in the same
        # color as the highlighted prose. It is not submitted document text.
        if re.fullmatch(r"[\d.]+%\(\d+\)", text):
            continue
        if active:
            body.append({**line, "section_index": section})
    source = ""; groups = []
    colors = {"#e60111": "strong_signal", "#d99e4c": "possible_signal"}
    for line in body:
        for span in line["spans"]:
            text = span["text"]
            if not text:
                continue
            category = colors.get(span["color_rgb"], "other")
            if not groups or groups[-1]["decision"]["category"] != category or groups[-1]["section_index"] != line["section_index"]:
                groups.append({"index": len(groups), "id": f"b{len(groups)+1:06}",
                               "char_start": len(source), "char_end": len(source),
                               "section_index": line["section_index"], "decision": {"category": category},
                               "source_locations": []})
            source += text
            groups[-1]["char_end"] = len(source)
            groups[-1]["source_locations"].append({"page": line["page"], "bbox": span["bbox"], "color_rgb": span["color_rgb"]})
        if groups:
            source += "\n"; groups[-1]["char_end"] = len(source)
    for group in groups:
        group["text"] = source[group["char_start"]:group["char_end"]]
        group["sha256"] = sha(group["text"].encode())
    if not source or {g["section_index"] for g in groups} != set(range(1, 13)):
        raise ValueError("Incomplete AIGC body section extraction")
    return source, groups


def similarity_summary(pages):
    text = clean_text(pages)
    fields = {}
    for key, label in [("total_percent", "总文字复制比"), ("excluding_self_percent", "去除本人文献复制比"),
                       ("excluding_citations_percent", "去除引用文献复制比"), ("maximum_single_source_percent", "单篇最大文字复制比")]:
        fields[key] = float(re.search(label + r"：\s*([\d.]+)%", text)[1])
    for key, label in [("duplicate_characters", "重复字符数"), ("maximum_single_source_characters", "单篇最大重复字符数"), ("total_characters", "总字符数")]:
        fields[key] = int(re.search(label + r"：\[(\d+)\]", text)[1])
    sections = [{"index": i+1, "percent": float(p), "duplicate_characters": int(c), "name": n,
                 "total_characters": int(t)} for i, (p,c,n,t) in enumerate(re.findall(
                     r"([\d.]+)%\((\d+)\)\n[\d.]+%\(\d+\)\n([^\n]+)（总(\d+)字符）", text))]
    if sum(s["total_characters"] for s in sections) != fields["total_characters"]:
        raise ValueError("Similarity section totals disagree")
    if sum(s["duplicate_characters"] for s in sections) != fields["duplicate_characters"]:
        raise ValueError("Similarity marked totals disagree")
    sources = [{"title": title.strip(), "percent": float(p), "duplicate_characters": int(n),
                "citation_raw": citation.strip(), "cited": cited == "是"}
               for title,p,n,citation,cited in re.findall(r"(?m)^\d+\n([^\n]+)\n([\d.]+)%（(\d+)）\n([^\n]+)\n是否引证：(是|否)", text)]
    return fields, sections, sources


def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "originals").mkdir(); (output / "pages").mkdir()
    records = {}; provenance = []
    for path in sorted(source.glob("*.pdf")):
        pages = page_records(path); records[path.name] = pages
        shutil.copy2(path, output / "originals" / path.name)
        write_jsonl(output / "pages" / (path.stem + ".jsonl"), pages)
        (output / "pages" / (path.stem + ".txt")).write_text("\n\n".join(f"[PDF page {p['page']}]\n{p['text']}" for p in pages), encoding="utf-8")
        provenance.append({"path": "originals/"+path.name, "original_path": str(path.resolve()), "sha256": sha(path.read_bytes()),
                           "bytes": path.stat().st_size, "pages": len(pages), "page_records": "pages/"+path.stem+".jsonl"})
    def named(fragment):
        found = [name for name in records if fragment in name]
        if len(found) != 1:
            raise ValueError(f"Expected one PDF containing {fragment}")
        return found[0], records[found[0]]
    simple_name, simple = named("AIGC_简洁")
    full_name, full = named("AIGC_全文")
    summary, sections = aigc_summary(simple)
    if aigc_summary(full) != (summary, sections):
        raise ValueError("AIGC summary and full reports disagree")
    excerpts = aigc_excerpts(simple, sections)
    if sum(s["provider_characters"] for s in excerpts if s["decision"]["category"] == "strong_signal") != summary["marked_characters"]:
        raise ValueError("AIGC excerpt totals disagree")
    body, spans = aigc_body(full)
    flagged = [s for s in spans if s["decision"]["category"] != "other" and s["text"].strip()]
    if len(flagged) != len(excerpts):
        raise ValueError(f"Full-report colored groups {len(flagged)} != summary excerpts {len(excerpts)}")
    for excerpt, span in zip(excerpts, flagged):
        a = re.sub(r"\s+", "", excerpt["text"]); b = re.sub(r"\s+", "", span["text"])
        if a != b or excerpt["decision"]["category"] != span["decision"]["category"]:
            raise ValueError(f"Full/summary excerpt mismatch: {excerpt['id']}")
        excerpt.update(char_start=span["char_start"], char_end=span["char_end"], full_report_locations=span["source_locations"])
    (output/"aigc").mkdir()
    (output/"aigc/source.txt").write_bytes(body.encode("utf-8"))
    write_jsonl(output/"aigc/segments.jsonl", excerpts)
    write_jsonl(output/"aigc/body-spans.jsonl", spans)
    meta = {"schema_version": "cnki-aigc-reference/1.0", "provider": "CNKI personal AIGC service",
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete", "summary": summary,
            "input": {"submitted_filename": "20260928.pdf", "submitted_pdf_sha256": None,
                      "provider_total_characters": summary["total_characters"]},
            "extracted_input": {"source": "source.txt", "characters": len(body), "sha256": sha(body.encode()),
                                "offset_unit": "python_unicode_code_points", "offset_convention": "zero_based_half_open"},
            "report_ids": re.findall(r"NO:([^\s]+)", clean_text(simple)+"\n"+clean_text(full)),
            "detected_at": re.search(r"检测时间：([^\n]+)", clean_text(simple))[1],
            "title": "恒定阻抗漏波天线", "sections": sections, "segments": excerpts,
            "sources": ["../originals/"+simple_name, "../originals/"+full_name],
            "policy": {"strong_signal_counted": True, "possible_signal_counted": False},
            "extraction": {"library": "PyMuPDF", "version": fitz.VersionBind, "ocr": False,
                           "full_and_simple_excerpts_match_ignoring_whitespace": True,
                           "warnings": ["Extracted Unicode offsets and counts differ from CNKI's character definition.",
                                        "Preserve PDF extraction artifacts; do not infer missing symbols or submitted-file identity."]}}
    write_json(output/"aigc/report.json", meta)
    (output/"similarity").mkdir()
    _, similarity = named("查重_简洁")
    fields, sim_sections, sources = similarity_summary(similarity)
    for name, pages in records.items():
        if name.startswith("查重_"):
            other, sections2, _ = similarity_summary(pages)
            if other != fields or sections2 != sim_sections:
                raise ValueError("Similarity reports disagree")
    write_json(output/"similarity/report.json", {"schema_version": "cnki-similarity-reference/1.0",
               "summary": fields, "sections": sim_sections, "matched_sources": sources,
               "report_ids": {name: re.search(r"报告编号:([^\s]+)", clean_text(pages))[1]
                              for name, pages in records.items() if name.startswith("查重_")},
               "detailed_evidence": ["../pages/"+Path(name).stem+".jsonl" for name in records if name.startswith("查重_")],
               "interpretation": "Similarity and AIGC are separate services. Detailed two-column evidence is retained as page lines, bounding boxes and colors."})
    write_json(output/"manifest.json", {"schema_version": "cnki-reference-bundle/1.0", "status": "complete",
               "created_at": datetime.now(timezone.utc).isoformat(), "originals": provenance,
               "aigc": "aigc/report.json", "similarity": "similarity/report.json",
               "validation": {"originals_copied_byte_for_byte": True, "aigc_section_totals_match": True,
                              "aigc_excerpt_totals_match": True, "aigc_excerpt_color_cross_check": True,
                              "similarity_section_totals_match": True},
               "checkpoint": None})
    print(json.dumps({"output": str(output), "aigc": summary, "aigc_excerpts": len(excerpts),
                      "possible_characters": sum(s["provider_characters"] for s in excerpts if s["decision"]["category"] == "possible_signal"),
                      "similarity": fields}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
