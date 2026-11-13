"""Visually checked source pairs and new-source human-only frozen controls.

New controls measure false positives only; there are no new generated positives
and no claim of independent recall validation. Does not train any neural network.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import english_frozen_local as e

EP = e.EVIDENCE / "supplement"
RP = e.REPORTS / "supplement"
REGIONS = {
    "P01-p011-b004": [[47, 553, 280, 601], [290, 457, 523, 601]],
    "P02-p010-b003": [[37, 571, 290, 703]],
    "P07-p002-b008": [[81, 630, 307, 655], [311, 310, 537, 399]],
    "P10-p003-b009": [[48, 731, 302, 751], [311, 267, 565, 376]],
}


def prose_xml(root):
    """Abstract/body prose only; figures, tables, references and math excluded."""
    excluded = {"fig", "table-wrap", "ref-list", "ack", "fn-group", "supplementary-material", "disp-formula"}
    def inline(node):
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in excluded or tag in {"inline-formula", "tex-math", "math"} or (tag == "xref" and node.get("ref-type") == "bibr"):
            return " "
        return (node.text or "") + "".join(inline(c)+(c.tail or "") for c in node)
    parts = []
    def visit(node):
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in excluded:
            return
        if tag == "p":
            value = e.clean(inline(node))
            if value:
                parts.append(value)
            return
        for child in node:
            visit(child)
    for abstract in root.findall("./front/article-meta/abstract"):
        visit(abstract)
    body = root.find("./body")
    if body is not None:
        visit(body)
    return "\n\n".join(parts), parts


def prepare():
    e.verify_freeze()
    if (EP/"declaration.json").exists():
        raise FileExistsError("Supplement already declared")
    data = e.read(e.EVIDENCE/"inputs.json")
    chosen = {b for s in data["audit_selection"] for b in s["block_ids"]}
    pairs, passages = [], {}
    def add(text):
        h = e.digest(text)
        passages[h] = text
        return h
    import fitz
    for row in e.read(e.REPORTS/"source-pairs.json"):
        if row["id"] not in chosen:
            continue
        if row["id"] in REGIONS:
            with fitz.open(e.SAMPLES/f"upstream/sources/{row['paper_id']}.pdf") as doc:
                page = doc[row["page"]-1]
                mechanical = e.clean("\n".join(page.get_text("text", clip=fitz.Rect(r)) for r in REGIONS[row["id"]]))
        else:
            mechanical = row["mechanical"]
        trimmed, similarity = e.page_excerpt(mechanical, row["converted"])
        if not trimmed:
            raise ValueError("Cannot align reviewed source "+row["id"])
        pairs.append({"id": row["id"], "paper_id": row["paper_id"], "page": row["page"],
                      "converted_hash": add(row["converted"]), "mechanical_hash": add(trimmed),
                      "word_similarity": similarity, "regions": REGIONS.get(row["id"]),
                      "note": "Reviewed against original page; residual differences include math, broken-word joining and PDF font encoding"})
    discovery = e.read(e.REPORTS/"new-source-discovery.json")
    documents, rows, sources = {}, [], []
    for i, source in enumerate(discovery["papers"], 1):
        pid = f"N{i:02}"
        xml_file = e.REPORTS/"new-sources"/(source["pmcid"]+".xml")
        root = ET.fromstring(xml_file.read_bytes())
        license_text = " ".join(" ".join(x.itertext()) for x in root.findall(".//permissions/license"))
        if "Creative Commons Attribution 4.0" not in license_text:
            raise ValueError("Unexpected license")
        text, paragraphs = prose_xml(root)
        if len(text) < 5000:
            raise ValueError("Unexpectedly short full prose extraction")
        documents[pid] = text
        sources.append({**source, "id": pid, "role": "human_calibration" if i <= 2 else "human_test",
                        "url": "https://europepmc.org/articles/"+source["pmcid"], "xml_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/"+source["pmcid"]+"/fullTextXML",
                        "license": "CC BY 4.0", "authors": [" ".join(n.itertext()) for n in root.findall(".//article-meta/contrib-group/contrib/name")],
                        "xml_sha256": e.sha(xml_file), "text_sha256": e.digest(text), "characters": len(text), "paragraphs": len(paragraphs)})
        for mode in e.MODES:
            for j, (a,b) in enumerate(e.local_spans(text, mode)):
                rows.append({"id": pid+":"+mode+f":{j:04}", "paper_id": pid, "mode": mode, "hash": add(text[a:b]), "start": a, "end": b})
    e.write(RP/"passages.json", passages)
    e.write(RP/"documents.json", documents)
    e.write(EP/"inputs.json", {"source_pairs": pairs, "new_human_windows": rows, "sources": sources, "contains_source_text": False})
    pages = sorted({(s["paper_id"], p) for s in data["audit_selection"] for p in s["pages"]})
    declaration = {"at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": e.sha(__file__),
                   "base_freeze_sha256": e.sha(e.EVIDENCE/"FREEZE.json"), "inputs_sha256": e.sha(EP/"inputs.json"),
                   "passages_sha256": e.sha(RP/"passages.json"),
                   "page_images": {f"{p}-p{n:03}.png": e.sha(e.REPORTS/"audit-pages"/f"{p}-p{n:03}.png") for p,n in pages},
                   "source_audit": "4 highest historical model windows and 6 seeded random windows; 12 original PDF pages visually inspected; incomplete bbox/reading order corrected in four reviewed blocks without changing corpus labels",
                   "new_source_policy": "N01/N02 human calibration; N03/N04 human test. Never change the base-selected candidate. Per-candidate cutoff=max(base cutoff, new calibration max+0.5). Score test only after calibration freeze.",
                   "boundary": "No new-source positives. New to this project, possible pretrained-corpus overlap unassessed. Four sources cannot validate a population zero-FPR claim.",
                   "author_audit": "Full displayed author names checked against old source metadata and each other; no shared authors found; common surnames alone do not define an identity",
                   "base_network_trained": False, "head_trained": False}
    e.write(EP/"declaration.json", declaration)
    print(json.dumps({"prepared": True, "reviewed_blocks": len(pairs), "new_documents": len(sources), "new_windows": len(rows)}), flush=True)


def verify():
    c = e.verify_freeze()
    d = e.read(EP/"declaration.json")
    for p,k in [(Path(__file__),"script_sha256"),(e.EVIDENCE/"FREEZE.json","base_freeze_sha256"),(EP/"inputs.json","inputs_sha256"),(RP/"passages.json","passages_sha256")]:
        if e.sha(p) != d[k]:
            raise ValueError("Supplement changed "+str(p))
    for src in e.read(EP/"inputs.json")["sources"]:
        if e.sha(e.REPORTS/"new-sources"/(src["pmcid"]+".xml")) != src["xml_sha256"]:
            raise ValueError("XML source changed")
    return c


def score(stage):
    c = verify()
    inputs = e.read(EP/"inputs.json")
    passages = e.read(RP/"passages.json")
    selected = {"N01","N02"} if stage == "calibration" else {"N03","N04"}
    if stage == "test":
        freeze = e.read(EP/"CALIBRATION.json")
        assert freeze["calibration_scores_sha256"] == e.sha(EP/"calibration-scores.json")
        assert freeze["declaration_sha256"] == e.sha(EP/"declaration.json")
    wanted = {r["hash"] for r in inputs["new_human_windows"] if r["paper_id"] in selected}
    if stage == "calibration":
        for p in inputs["source_pairs"]:
            wanted.update([p["converted_hash"],p["mechanical_hash"]])
    import torch
    from aigc_detector.engine import Detector
    torch.set_num_threads(1)
    detector = Detector(device="cuda",batch_size=8)
    out = {h:{} for h in sorted(wanted)}
    def blocked(*args,**kwargs): raise AssertionError("Network disabled")
    with patch.object(socket.socket,"connect",blocked),patch.object(socket,"create_connection",blocked):
        for m in e.MODELS:
            cache_path = RP/(stage+"-"+m+".json")
            cache = e.read(cache_path) if cache_path.exists() else {"declaration_sha256":e.sha(EP/"declaration.json"),"items":{}}
            assert cache["declaration_sha256"] == e.sha(EP/"declaration.json")
            _,model,_ = detector._load(m)
            model.requires_grad_(False)
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            missing = sorted(wanted-set(cache["items"]))
            for i in range(0,len(missing),24):
                batch = missing[i:i+24]
                values = detector._score_texts([passages[h] for h in batch],m,64,True)
                for h,v in zip(batch,values):
                    assert v["revision"] == c["revisions"][m]
                    cache["items"][h] = v
                e.write(cache_path,cache)
            for h in wanted: out[h][m]=cache["items"][h]["ai_score"]
            detector._loaded.clear()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(stage,m,len(wanted),flush=True)
    result = {"scores":out,"declaration_sha256":e.sha(EP/"declaration.json"),"no_training":True}
    if stage == "test": result["calibration_freeze_sha256"] = e.sha(EP/"CALIBRATION.json")
    e.write(EP/(stage+"-scores.json"),result)


def calibrate():
    verify()
    if (EP/"CALIBRATION.json").exists(): raise FileExistsError("Already calibrated")
    data,scores=e.read(EP/"inputs.json"),e.read(EP/"calibration-scores.json")["scores"]
    selection=e.read(e.EVIDENCE/"selection.json")
    thresholds={}
    for f in selection["frozen_candidates"]:
        c=f["candidate"]
        vals=[e.logit(scores[r["hash"]][c["model"]]) for r in data["new_human_windows"] if r["paper_id"] in {"N01","N02"} and r["mode"]==c["mode"]]
        thresholds[c["id"]]=max(f["threshold"],max(vals)+.5)
    e.write(EP/"CALIBRATION.json",{"at_utc":datetime.now(timezone.utc).isoformat(),"thresholds":thresholds,
                                  "calibration_scores_sha256":e.sha(EP/"calibration-scores.json"),"declaration_sha256":e.sha(EP/"declaration.json"),"selected":selection["selected"]})
    print(json.dumps({"calibrated":True,"selected_unchanged":selection["selected"]}),flush=True)


def evaluate():
    verify()
    if (EP/"evaluation.json").exists():raise FileExistsError("Preserve evaluation")
    data=e.read(EP/"inputs.json")
    test=e.read(EP/"test-scores.json")
    assert test["calibration_freeze_sha256"] == e.sha(EP/"CALIBRATION.json")
    scores={**e.read(EP/"calibration-scores.json")["scores"],**test["scores"]}
    cal=e.read(EP/"CALIBRATION.json")
    results={}
    for f in e.read(e.EVIDENCE/"selection.json")["frozen_candidates"]:
        c=f["candidate"]
        by_paper={}
        for pid in ['N01','N02','N03','N04']:
            rows=[r for r in data['new_human_windows'] if r['paper_id']==pid and r['mode']==c['mode']]
            vals=[e.logit(scores[r['hash']][c['model']]) for r in rows]
            by_paper[pid]={'windows':len(rows),'old_threshold_flags':sum(v>=f['threshold'] for v in vals),
                           'new_threshold_flags':sum(v>=cal['thresholds'][c['id']] for v in vals)}
        results[c['id']]={'by_paper':by_paper,'base_threshold':f['threshold'],'new_threshold':cal['thresholds'][c['id']]}
    paired={m:[{'id':p['id'],'converted_score':scores[p['converted_hash']][m],'mechanical_score':scores[p['mechanical_hash']][m],
                'delta_logit':e.logit(scores[p['converted_hash']][m])-e.logit(scores[p['mechanical_hash']][m])} for p in data['source_pairs']] for m in e.MODELS}
    e.write(EP/'evaluation.json',{'results':results,'selected':cal['selected'],'source_pairs':paired,
                                'calibration_freeze_sha256':e.sha(EP/'CALIBRATION.json'),'test_scores_sha256':e.sha(EP/'test-scores.json'),
                                'new_source_recall_evaluated':False,'contains_source_text':False})
    print(json.dumps({'selected':cal['selected'],'new_source_human':results[cal['selected']['id']]}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','score-calibration','calibrate','score-test','evaluate','verify'])
    a=p.parse_args().action
    if a.startswith('score-'):score(a.split('-',1)[1])
    else:globals()[a]()
