"""Fetch a small, attributed sample set from the authors' original resources."""

import ast
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.catalog import ROOT
from network import get_json, open_public


def main():
    target = ROOT / "samples"
    target.mkdir(exist_ok=True)
    dataset_id = "Hello-SimpleAI/HC3-Chinese"
    meta = get_json(f"https://huggingface.co/api/datasets/{dataset_id}")
    revision = meta["sha"]
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/baike.jsonl"
    samples = []
    with open_public(url) as response:
        for line_number, raw in enumerate(response, start=1):
            row = json.loads(raw)
            human = next((text for text in row["human_answers"] if len(text) >= 100), None)
            ai = next((text for text in row["chatgpt_answers"] if len(text) >= 100), None)
            if not human or not ai:
                continue
            for field, label, text in (("human_answers", "human", human), ("chatgpt_answers", "ai", ai)):
                samples.append({"id": f"hc3-baike-line{line_number}-{label}", "text": text,
                                "source_label": label, "source": url, "dataset_revision": revision,
                                "row_id": row.get("id"), "line_number": line_number, "answer_field": field,
                                "answer_index": row[field].index(text), "question": row["question"],
                                "license": "CC-BY-SA-4.0", "kind": "official_dataset"})
            if len(samples) == 6:
                break
    if len(samples) != 6:
        raise RuntimeError("Expected three complete human/AI pairs")
    space_id = "yuchuantian/AIGC_text_detector"
    space_meta = get_json(f"https://huggingface.co/api/spaces/{space_id}")
    space_revision = space_meta["sha"]
    demo_url = f"https://huggingface.co/spaces/{space_id}/raw/{space_revision}/app.py"
    with open_public(demo_url) as response:
        source = response.read().decode("utf-8")
    example = None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Textbox":
            for keyword in node.keywords:
                if keyword.arg == "value" and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str) and value.startswith("北京大学"):
                        example = value
                        break
    if example is None:
        raise RuntimeError("Could not locate the author's Chinese demo example")
    samples.insert(0, {"id": "mpu-official-demo-zh", "text": example, "source_label": None,
                       "source": demo_url, "kind": "official_demo", "note": "Author demo has no ground-truth label."})
    long_text = "\n\n".join(sample["text"] for sample in samples[1:]) * 3
    samples.append({"id": "long-window-coverage", "text": long_text, "source_label": None,
                    "source": url, "kind": "derived_integration_fixture", "license": "CC-BY-SA-4.0",
                    "derived_from": [sample["id"] for sample in samples[1:]],
                    "note": "Concatenated and repeated source answers only to test complete long-input coverage."})
    for sample in samples:
        sample["sha256"] = hashlib.sha256(sample["text"].encode("utf-8")).hexdigest()
    (target / "smoke.jsonl").write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples), encoding="utf-8")
    (target / "demo-zh.txt").write_text(example + "\n", encoding="utf-8")
    # samples/README.md is the curated index for all collections. Regenerating
    # the smoke fixtures must not overwrite links to separately collected data.
    print(f"Saved {len(samples)} attributed samples to {target}")


if __name__ == "__main__":
    main()
