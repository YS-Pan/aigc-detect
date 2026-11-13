"""Source/context controls and local scanning using frozen classifiers only.

Private passages and full outputs stay in reports/. Evidence contains only IDs,
hashes, scores and positions. No optimizer, backward pass, or new model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import socket
import sys
import time
import unicodedata
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aigc_detector.annotation import segment_spans
from scripts.english_academic_samples import verify as verify_samples

EVIDENCE = ROOT / "evaluations/english-frozen-local-20261004"
REPORTS = ROOT / "reports/english-frozen-local-20261004"
SAMPLES = ROOT / "samples/english-academic-20261003"
PROFILE = ROOT / "aigc_detector/profiles/english-academic-20261003-v2.json"
MODELS = ["followsci-bert-en", "roberta-academic-en", "llm-detectaive-en", "cocodet-en"]
MODES = ["whole1600", "paragraph", "sentence3"]
SEED = 20261004


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def clean(text):
    """Same conservative mechanical cleanup for PDF and converted passages."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    text = text.replace("\u00ad", "").replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\[\s*\d+(?:\s*[,\-]\s*\d+)*\s*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return [m.group().lower() for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]


def page_excerpt(page_text, converted):
    """Locate first/last reliable lexical anchors; never reconstruct PDF words.

    This locates a span in the mechanical page stream using the old transcription.
    It is an alignment aid, not an independent completeness/identity verification.
    """
    tokens = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", page_text))
    a, b = words(converted), [m.group().lower() for m in tokens]
    matches = [m for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if m.size >= 3]
    if not matches:
        return "", 0.0
    start = max(0, matches[0].b - matches[0].a)
    end = min(len(b), matches[-1].b + matches[-1].size + len(a) - matches[-1].a - matches[-1].size)
    out = page_text[tokens[start].start():tokens[end-1].end()]
    return out, SequenceMatcher(None, a, words(out), autojunk=False).ratio()


def local_spans(text, mode):
    """No labels or source maps: deployment-style boundaries from input only."""
    if mode == "whole1600":
        return segment_spans(text, 1600, 400, language="en")
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S)]
    if mode == "paragraph":
        spans = []
        for start, end in paragraphs:
            spans += [(start+a, start+b) for a, b in segment_spans(text[start:end], 1600, 120, language="en")]
        return spans
    if mode != "sentence3":
        raise ValueError(mode)
    spans = []
    for start, end in paragraphs:
        # Deterministic lightweight English boundaries; abbreviations can split.
        cuts = [start] + [start+m.end() for m in re.finditer(r"[.!?][\"')\]]*\s+(?=[A-Z])", text[start:end])] + [end]
        sentences = [(a, b) for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]
        for i in range(max(1, len(sentences)-2)):
            left, right = sentences[i][0], sentences[min(i+2, len(sentences)-1)][1]
            spans += [(left+a, left+b) for a, b in segment_spans(text[left:right], 1600, 120, language="en")]
    return sorted(set(spans))


def intersects(a, b, c, d):
    return max(a, c) < min(b, d)


def make_mixture(ai, human_slot, context, fraction, position):
    """Exact paired length and AI location; context is identical in splice pair."""
    total = round(len(ai)/fraction)
    extra = total-len(ai)
    if len(human_slot) != len(ai) or len(context) < extra:
        raise ValueError("Insufficient matched human text")
    nleft = {"start": 0, "middle": extra//2, "end": extra}[position]
    before, after = context[:nleft], context[nleft:extra]
    return before+ai+after, before+human_slot+after, nleft, nleft+len(ai)


def logit(value):
    return math.log(max(float(value), 1e-30))-math.log(max(1-float(value), 1e-30))


def author_groups(papers):
    authors = {p["paper_id"]: {re.sub(r"\W", "", a.lower()) for a in p["author"].split(" and ")} for p in papers}
    groups = [{pid} for pid in authors]
    while True:
        merged = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if set().union(*(authors[p] for p in groups[i])) & set().union(*(authors[p] for p in groups[j])):
                    groups[i] |= groups.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return sorted([sorted(g) for g in groups])


def prepare():
    if (EVIDENCE/"campaign.json").exists():
        raise FileExistsError("Preserve the declaration; use a new campaign for changed rules")
    verify_samples(SAMPLES)
    samples = [json.loads(s) for s in (SAMPLES/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    papers = read(SAMPLES/"upstream/sources/manifest.json")["papers"]
    groups = read(SAMPLES/"protocol.json")["groups"]
    manifest = read(ROOT/"models/manifest.json")["models"]
    paths = [Path(__file__), ROOT/"aigc_detector/engine.py", ROOT/"aigc_detector/modeling.py",
             ROOT/"aigc_detector/annotation.py", ROOT/"aigc_detector/catalog.py", ROOT/"models/manifest.json",
             PROFILE, SAMPLES/"samples.jsonl", SAMPLES/"protocol.json", SAMPLES/"sources/import-manifest.json"]
    config = {"schema": "frozen-local-controls/1.0", "declared_at_utc": datetime.now(timezone.utc).isoformat(),
              "seed": SEED, "models": MODELS, "modes": MODES, "groups": groups,
              "author_groups": author_groups(papers), "training": "none; all base weights and classification heads frozen",
              "candidates": [{"id": mode+"::"+m, "mode": mode, "model": m} for mode in MODES for m in MODELS],
              "primary_hypothesis": "followsci-bert-en paragraph versus whole1600; sentence3 is secondary",
              "threshold": "max of all pure-human local-window logits in training/control sources plus 0.5",
              "selection": "nested leave-author-connected-paper-group-out; minimize false windows then maximize mixed block hits then fewer scans then ID",
              "margin": .5, "context_fractions": [1., .75, .5, .25], "context_positions": ["start", "middle", "end"],
              "context_policy": "intact same AI paragraph; ratio changes total length; positions within each length are controlled; exact same-length human splice and contiguous controls",
              "length_control": "additional fixed 1600 characters at AI fractions .125/.25/.5 when paragraph prefix is long enough; prefixes are explicitly not whole paragraphs",
              "inference": {"device": "cuda", "batch_size": 8, "dtype": "float32", "overlap_tokens": 64, "network_blocked": True},
              "revisions": {m: manifest[m]["revision"] for m in MODELS},
              "source_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              "expansion_gate": "no expansion or deployment unless complete selected cross-author process has zero false windows and at least 50% mixed-block recall",
              "new_independent_validation": False, "default_changes": False}
    # Save all experimental rules BEFORE extraction and any new classifier outputs.
    write(EVIDENCE/"campaign.json", config)
    passages, local, contexts, pairs, source_details, blocks = {}, [], [], [], [], []
    texts = {s["id"]: (SAMPLES/s["text_file"]).read_text(encoding="utf-8") for s in samples}
    def add(text):
        h = digest(text)
        passages[h] = text
        return h
    for s in samples:
        if s["kind"] == "ai_paragraph":
            continue
        text = texts[s["id"]]
        mapping = read(SAMPLES/s["mapping_file"])["blocks"]
        ai = [b for b in mapping if b["source_label"] == "ai"]
        for b in ai:
            blocks.append({"id": s["id"]+":"+b["block_id"], "sample_id": s["id"], "paper_id": s["paper_id"],
                           "generator": s.get("generator"), "start": b["char_start"], "end": b["char_end"]})
        for mode in MODES:
            for i, (a, b) in enumerate(local_spans(text, mode)):
                local.append({"id": s["id"]+":"+mode+f":{i:04}", "sample_id": s["id"], "paper_id": s["paper_id"],
                              "split": s["split"], "mode": mode, "start": a, "end": b, "hash": add(text[a:b]),
                              "positive": any(intersects(a, b, g["char_start"], g["char_end"]) for g in ai),
                              "kind": s["kind"]})
    import fitz
    for paper in papers:
        pid = paper["paper_id"]
        pdf_path = SAMPLES/"upstream"/paper["source_pdf"]
        if sha(pdf_path) != paper["source_sha256"]:
            raise ValueError("PDF hash changed")
        doc = fitz.open(pdf_path)
        converted = texts[pid+"-human"]
        mapping = read(SAMPLES/f"maps/{pid}-human.json")["blocks"]
        source_blocks = {b["id"]: b for b in read(SAMPLES/f"upstream/corpus/original/{pid}/content.json")["blocks"]}
        for mapped in mapping:
            b = source_blocks[mapped["block_id"]]
            old = clean(converted[mapped["char_start"]:mapped["char_end"]])
            page = doc[b["page"]-1]
            bbox = b.get("source_bbox")
            if bbox and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                rect = fitz.Rect(bbox)+(-.8, -.8, .8, .8)
                mechanical = clean(page.get_text("text", clip=rect, sort=False))
                method = "declared_source_bbox"
            else:
                mechanical, _ = page_excerpt(clean(page.get_text("text", sort=False)), old)
                method = "page_lexical_anchor_alignment"
            ratio = SequenceMatcher(None, words(old), words(mechanical), autojunk=False).ratio()
            meta = {"id": mapped["block_id"], "paper_id": pid, "page": b["page"], "method": method,
                    "converted_hash": add(old), "mechanical_hash": add(mechanical) if mechanical else None,
                    "converted_chars": len(old), "mechanical_chars": len(mechanical), "word_similarity": ratio,
                    "alignment_eligible": len(old) >= 80 and len(mechanical) >= 80 and ratio >= .6,
                    "pdf_sha256": paper["source_sha256"], "bbox": bbox,
                    "document_start": mapped["char_start"], "document_end": mapped["char_end"]}
            pairs.append(meta)
            source_details.append({**meta, "converted": old, "mechanical": mechanical})
        doc.close()
    for s in samples:
        if s["kind"] != "ai_paragraph":
            continue
        ai = clean(texts[s["id"]])
        human = clean(texts[s["paper_id"]+"-human"])
        # Text selection depends only on source and length, never detector scores.
        n = len(ai)
        if len(human) < 5*n:
            # Circular reuse is explicit for very short papers, not independent data.
            pool = (human+" ") * math.ceil((6*n)/(len(human)+1))
            recycled = True
        else:
            pool, recycled = human, False
        slot = pool[:n]
        context = " " + pool[n+1:] + " "
        def append_case(value, control, lo, hi, fraction, position, family, is_prefix):
            base = {"paragraph_id": s["id"], "paper_id": s["paper_id"], "generator": s["generator"],
                    "family": family, "fraction": fraction, "position": position, "characters": len(value),
                    "ai_start": lo, "ai_end": hi, "prefix_only": is_prefix, "human_recycled": recycled}
            contexts.append({**base, "kind": "ai_mixed" if fraction < 1 else "ai_isolated", "hash": add(value)})
            contexts.append({**base, "kind": "human_splice", "hash": add(control)})
            contexts.append({**base, "kind": "human_contiguous", "hash": add(pool[:len(value)])})
        for fraction in config["context_fractions"]:
            for position in (["start"] if fraction == 1 else config["context_positions"]):
                a, h, lo, hi = make_mixture(ai, slot, context, fraction, position)
                append_case(a, h, lo, hi, fraction, position, "intact", False)
        for fraction in [.125, .25, .5]:
            k = round(1600*fraction)
            if len(ai) < k or len(pool) < 1800:
                continue
            for position in config["context_positions"]:
                a, h, lo, hi = make_mixture(ai[:k], pool[:k], " "+pool[k+1:]+" ", fraction, position)
                append_case(a, h, lo, hi, fraction, position, "fixed1600_prefix", True)
    previous = []
    for name in ["development-data.json", "repeat-data.json"]:
        previous += read(ROOT/"evaluations/english-specialized-fusion-20261004"/name)["records"]
    human_windows = list({r["sha256"]: r for r in previous if r["kind"] == "human_document"}.values())
    selection = []
    for m in MODELS:
        row = max(human_windows, key=lambda r: r["scores"][m])
        selection.append({"reason": "highest_existing_"+m, "record": row})
    for row in random.Random(SEED).sample(human_windows, 6):
        selection.append({"reason": "seeded_random", "record": row})
    audit_selection = []
    for selected in selection:
        r = selected["record"]
        chosen = [p for p in pairs if p["paper_id"] == r["paper_id"] and intersects(p["document_start"], p["document_end"], r["char_start"], r["char_end"])]
        audit_selection.append({"reason": selected["reason"], "window_id": r["id"], "paper_id": r["paper_id"],
                                "block_ids": [p["id"] for p in chosen], "pages": sorted({p["page"] for p in chosen})})
    write(REPORTS/"passages.json", passages)
    write(REPORTS/"source-pairs.json", source_details)
    write(EVIDENCE/"inputs.json", {"local": local, "contexts": contexts, "source_pairs": pairs, "blocks": blocks,
                                   "audit_selection": audit_selection, "contains_source_text": False})
    write(EVIDENCE/"input-lock.json", {"campaign_sha256": sha(EVIDENCE/"campaign.json"), "inputs_sha256": sha(EVIDENCE/"inputs.json"),
                                   "passages_sha256": sha(REPORTS/"passages.json"), "source_pairs_sha256": sha(REPORTS/"source-pairs.json")})
    print(json.dumps({"prepared": True, "local_windows": len(local), "context_records": len(contexts),
                      "source_pairs": len(pairs), "eligible_source_pairs": sum(p["alignment_eligible"] for p in pairs),
                      "unique_passages": len(passages), "author_groups": config["author_groups"]}), flush=True)


def verify_inputs():
    c = read(EVIDENCE/"campaign.json")
    for file, expected in c["source_hashes"].items():
        if sha(ROOT/file) != expected:
            raise ValueError("Declared code/input changed: "+file)
    i = read(EVIDENCE/"input-lock.json")
    for p, k in [(EVIDENCE/"campaign.json", "campaign_sha256"), (EVIDENCE/"inputs.json", "inputs_sha256"),
                 (REPORTS/"passages.json", "passages_sha256"), (REPORTS/"source-pairs.json", "source_pairs_sha256")]:
        if sha(p) != i[k]:
            raise ValueError("Input changed: "+str(p))
    return c


def score(stage):
    c = verify_inputs()
    data, passages = read(EVIDENCE/"inputs.json"), read(REPORTS/"passages.json")
    selected = set(c["groups"]["test"] if stage == "repeat" else c["groups"]["fit"]+c["groups"]["calibration"])
    if stage == "repeat":
        verify_freeze()
    elif (EVIDENCE/"FREEZE.json").exists():
        raise FileExistsError("Development stage already frozen")
    wanted = set()
    for r in data["local"]+data["contexts"]:
        if r["paper_id"] in selected:
            wanted.add(r["hash"])
    for r in data["source_pairs"]:
        if r["paper_id"] in selected and r["alignment_eligible"]:
            wanted.update([r["converted_hash"], r["mechanical_hash"]])
    import torch
    from aigc_detector.engine import Detector
    torch.set_num_threads(1)
    detector = Detector(device="cuda", batch_size=8)
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden during inference")
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        for m in MODELS:
            target = REPORTS/stage/(m+".json")
            contract = {"campaign_sha256": sha(EVIDENCE/"campaign.json"), "inputs_sha256": sha(EVIDENCE/"inputs.json"),
                        "revision": c["revisions"][m], "stage": stage}
            cache = read(target) if target.exists() else {**contract, "items": {}}
            if any(cache.get(k) != v for k, v in contract.items()):
                raise ValueError("Cache contract mismatch")
            _, model, _ = detector._load(m)
            model.requires_grad_(False)
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            missing = sorted(wanted-set(cache["items"]))
            started = time.perf_counter()
            for offset in range(0, len(missing), 24):
                batch = missing[offset:offset+24]
                values = detector._score_texts([passages[h] for h in batch], m, 64, True)
                for h, value in zip(batch, values):
                    if value["revision"] != c["revisions"][m] or value["ai_score"] is None:
                        raise ValueError("Invalid inference")
                    cache["items"][h] = value
                cache["elapsed_seconds_this_invocation"] = time.perf_counter()-started
                write(target, cache)
                if offset % 240 == 0 or offset+24 >= len(missing):
                    print(f"{stage} {m}: {min(offset+24, len(missing))}/{len(missing)}", flush=True)
            detector._loaded.clear()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    out = {h: {} for h in sorted(wanted)}
    for m in MODELS:
        for h, v in read(REPORTS/stage/(m+".json"))["items"].items():
            if h in out:
                out[h][m] = v["ai_score"]
    payload = {"scores": out, "stage": stage, "papers": sorted(selected), "no_training": True,
               "cache_hashes": {m: sha(REPORTS/stage/(m+".json")) for m in MODELS},
               "campaign_sha256": sha(EVIDENCE/"campaign.json")}
    if stage == "repeat":
        payload["freeze_sha256"] = sha(EVIDENCE/"FREEZE.json")
    write(EVIDENCE/(stage+"-scores.json"), payload)
    print(json.dumps({"scored": stage, "unique_inputs": len(out)}), flush=True)


def threshold(rows, scores, candidate, papers, margin):
    values = [logit(scores[r["hash"]][candidate["model"]]) for r in rows if r["paper_id"] in papers and r["mode"] == candidate["mode"] and not r["positive"]]
    if not values:
        raise ValueError("Empty human calibration pool")
    return max(values)+margin


def measure(data, scores, candidate, papers, cutoff):
    rows = [r for r in data["local"] if r["paper_id"] in papers and r["mode"] == candidate["mode"]]
    positives = [r for r in rows if logit(scores[r["hash"]][candidate["model"]]) >= cutoff]
    blocks = [b for b in data["blocks"] if b["paper_id"] in papers]
    hit, covered = [], 0
    for b in blocks:
        ranges = sorted((max(b["start"], r["start"]), min(b["end"], r["end"])) for r in positives
                        if r["sample_id"] == b["sample_id"] and intersects(b["start"], b["end"], r["start"], r["end"]))
        if ranges:
            hit.append(b["id"])
        right = -1
        for a, z in ranges:
            covered += max(0, z-max(a, right))
            right = max(z, right)
    false = [r for r in positives if not r["positive"]]
    return {"human_windows": sum(not r["positive"] for r in rows), "false_windows": len(false),
            "false_unique_hashes": len({r["hash"] for r in false}), "false_documents": len({r["sample_id"] for r in false}),
            "false_papers": len({r["paper_id"] for r in false}), "mixed_blocks": len(blocks), "mixed_hits": len(hit),
            "ai_characters": sum(b["end"]-b["start"] for b in blocks), "covered_ai_characters": covered,
            "total_windows": len(rows), "hit_ids": hit, "false_ids": [r["id"] for r in false]}


def summed(rows):
    return {k: sum(r[k] for r in rows) for k in ["human_windows", "false_windows", "false_unique_hashes", "false_documents", "false_papers", "mixed_blocks", "mixed_hits", "ai_characters", "covered_ai_characters", "total_windows"]}


def objective(entry):
    m = entry["metrics"]
    return m["false_windows"], -m["mixed_hits"], m["total_windows"], entry["candidate"]["id"]


def freeze():
    if (EVIDENCE/"FREEZE.json").exists():
        raise FileExistsError("Already frozen")
    c = verify_inputs()
    data, scores = read(EVIDENCE/"inputs.json"), read(EVIDENCE/"development-scores.json")["scores"]
    fit = set(c["groups"]["fit"])
    units = [set(g)&fit for g in c["author_groups"] if set(g)&fit]
    def compare(pool):
        results = []
        papers = set().union(*pool)
        for candidate in c["candidates"]:
            folds = []
            for held in pool:
                train = papers-held
                cut = threshold(data["local"], scores, candidate, train, c["margin"])
                folds.append({"held": sorted(held), "train": sorted(train), "threshold": cut,
                              "metrics": measure(data, scores, candidate, held, cut)})
            results.append({"candidate": candidate, "metrics": summed([f["metrics"] for f in folds]), "folds": folds})
        return results
    selection = compare(units)
    chosen = min(selection, key=objective)
    nested = []
    for held in units:
        train_units = [u for u in units if u != held]
        inner = compare(train_units)
        selected = min(inner, key=objective)["candidate"]
        cut = threshold(data["local"], scores, selected, fit-held, c["margin"])
        nested.append({"held": sorted(held), "train": sorted(fit-held), "selected": selected, "threshold": cut,
                       "metrics": measure(data, scores, selected, held, cut), "inner": inner})
    controls = fit|set(c["groups"]["calibration"])
    candidates = [{"candidate": x, "threshold": threshold(data["local"], scores, x, controls, c["margin"])} for x in c["candidates"]]
    out = {"selection": selection, "selected": chosen["candidate"], "nested": nested,
           "nested_metrics": summed([f["metrics"] for f in nested]), "frozen_candidates": candidates,
           "control_papers": sorted(controls), "test_scores_used": False, "new_independent_validation": False}
    write(EVIDENCE/"selection.json", out)
    write(EVIDENCE/"FREEZE.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                  "hashes": {f: sha(EVIDENCE/f) for f in ["campaign.json", "input-lock.json", "inputs.json", "development-scores.json", "selection.json"]}})
    print(json.dumps({"selected": chosen["candidate"], "nested": out["nested_metrics"]}), flush=True)


def verify_freeze():
    c = verify_inputs()
    for f, h in read(EVIDENCE/"FREEZE.json")["hashes"].items():
        if sha(EVIDENCE/f) != h:
            raise ValueError("Frozen artifact changed: "+f)
    return c


def evaluate():
    c = verify_freeze()
    if (EVIDENCE/"evaluation.json").exists():
        raise FileExistsError("Preserve evaluation")
    data = read(EVIDENCE/"inputs.json")
    repeat = read(EVIDENCE/"repeat-scores.json")
    if repeat["freeze_sha256"] != sha(EVIDENCE/"FREEZE.json"):
        raise ValueError("Repeat not scored after freeze")
    scores = {**read(EVIDENCE/"development-scores.json")["scores"], **repeat["scores"]}
    selection = read(EVIDENCE/"selection.json")
    result = {}
    all_papers = set().union(*[set(v) for v in c["groups"].values()])
    for f in selection["frozen_candidates"]:
        candidate, cut = f["candidate"], f["threshold"]
        result[candidate["id"]] = {"threshold": cut, "all": measure(data, scores, candidate, all_papers, cut),
                                  "by_split": {k: measure(data, scores, candidate, set(ps), cut) for k, ps in c["groups"].items()},
                                  "by_paper": {p: measure(data, scores, candidate, {p}, cut) for p in sorted(all_papers)}}
    contexts = {}
    # Controlled comparisons have separate all-human, paper-held-out thresholds.
    # They describe constructed inputs, never deployed natural-document recall.
    for m in MODELS:
        grouped = defaultdict(list)
        for r in data["contexts"]:
            grouped[(r["family"], r["fraction"], r["position"], r["kind"])].append(r)
        control_thresholds = {}
        for group in c["author_groups"]:
            pool = [x for x in data["contexts"] if x["paper_id"] not in group and x["kind"].startswith("human")]
            cut = max(logit(scores[x["hash"]][m]) for x in pool)+c["margin"]
            control_thresholds.update({p: cut for p in group})
        summaries = []
        for key, rows in grouped.items():
            vals = sorted(logit(scores[r["hash"]][m]) for r in rows)
            hits = 0
            for r in rows:
                cut = control_thresholds[r["paper_id"]]
                hits += logit(scores[r["hash"]][m]) >= cut
            summaries.append({"family": key[0], "fraction": key[1], "position": key[2], "kind": key[3],
                              "count": len(rows), "median_logit": vals[len(vals)//2], "paper_held_out_flags": int(hits)})
        contexts[m] = summaries
    sources = {}
    for m in MODELS:
        eligible = [p for p in data["source_pairs"] if p["alignment_eligible"]]
        differences = [logit(scores[p["converted_hash"]][m])-logit(scores[p["mechanical_hash"]][m]) for p in eligible]
        sources[m] = {"paired_blocks": len(differences), "median_converted_minus_mechanical_logit": sorted(differences)[len(differences)//2],
                      "converted_higher": sum(v > 0 for v in differences),
                      "largest_absolute_differences": sorted([{"id": p["id"], "delta_logit": d} for p, d in zip(eligible, differences)], key=lambda x: -abs(x["delta_logit"]))[:20]}
    selected = result[selection["selected"]["id"]]
    checks = [selection["nested_metrics"], selected["by_split"]["test"]]
    passed = all(x["false_windows"] == 0 and x["mixed_hits"] >= .5*x["mixed_blocks"] for x in checks)
    out = {"results": result, "selected": selection["selected"], "nested_metrics": selection["nested_metrics"],
           "context_controls": contexts, "source_score_pairs": sources, "expansion_gate_passed": passed,
           "freeze_sha256": sha(EVIDENCE/"FREEZE.json"), "repeat_scores_sha256": sha(EVIDENCE/"repeat-scores.json"),
           "base_network_trained": False, "any_head_trained": False, "default_profile_unchanged": True,
           "new_independent_validation": False, "contains_source_text": False}
    write(EVIDENCE/"evaluation.json", out)
    print(json.dumps({"selected": out["selected"], "nested": out["nested_metrics"], "repeat": selected["by_split"]["test"], "gate": passed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "score-development", "freeze", "score-repeat", "evaluate", "verify"])
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action.startswith("score-"): score(args.action.split("-", 1)[1])
    elif args.action == "freeze": freeze()
    elif args.action == "evaluate": evaluate()
    else:
        verify_freeze()
        print(json.dumps({"freeze_verified": True}))


if __name__ == "__main__":
    main()
