"""Source label projection and full-document evaluation, independent of GPU."""
import copy
import math

import pytest

from scripts.english_academic_samples import plain_text
from scripts import english_diverse_corpus as corpus
from scripts import english_diverse_validation as experiment


@pytest.mark.parametrize("tex", [
    r"A \textbf{frozen} paragraph with $x=1$ and \cite{smith} a 20\% result.",
    "Two---dash ``quotes'' [1, 3] ~ space. % comment\nNext line.",
    r"\begin{itemize}\item A statement.\item Second statement.\end{itemize}",
    r"\(x^2\) The \emph{conclusion} uses \ref{eq:a} evidence.",
])
def test_historical_extraction_and_character_origin(tex):
    text, offsets = corpus.plain_with_offsets(tex, 31)
    assert text == plain_text(tex)
    assert len(text) == len(offsets)
    assert offsets == sorted(offsets)
    assert all(31 <= i < 31 + len(tex) for i in offsets)
    if "20" in text:
        pos = text.index("%")
        assert tex[offsets[pos]-31] == "%"


def span(tex, part):
    a = tex.index(part)
    return {"start_char": a, "end_char": a+len(part), "sha256": corpus.digest(part),
            "start_line": tex.count("\n", 0, a)+1, "end_line": tex.count("\n", 0, a+len(part)-1)+1}


def test_section_resources_are_not_generated_and_slots_not_spans():
    tex = "% BEGIN BLOCK B1 kind=prose source_page=1\nNew words \\cite{old} more words.\n% END BLOCK B1"
    ann = {"aigc_spans": [{"slot_id": "S1", "generated_span": span(tex, "New words ")},
                          {"slot_id": "S1", "generated_span": span(tex, " more words.")}]}
    d = corpus.extract_document(tex, ann)
    assert d["text"] == "New words more words."
    assert len(d["units"]) == 1
    assert corpus.overlap(d["ai_spans"], 0, len(d["text"])) == len(d["text"])
    broken = copy.deepcopy(ann)
    broken["aigc_spans"][0]["generated_span"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="hash"):
        corpus.extract_document(tex, broken)


def test_generated_list_prose_is_included():
    tex = "% BEGIN BLOCK B1 kind=list source_page=1\n\\item Generated reasoning.\n% END BLOCK B1"
    ann = {"aigc_spans": [{"slot_id": "S1", "generated_span": span(tex, "Generated reasoning.")}]}
    d = corpus.extract_document(tex, ann)
    assert d["text"] == "Generated reasoning."
    assert len(d["units"]) == 1


def test_generated_language_in_excluded_caption_fails():
    tex = "% BEGIN BLOCK B1 kind=caption source_page=1\nGenerated caption.\n% END BLOCK B1"
    ann = {"aigc_spans": [{"slot_id": "S1", "generated_span": span(tex, "Generated caption.")}]}
    with pytest.raises(ValueError, match="excluded"):
        corpus.extract_document(tex, ann)


def test_author_aliases_do_not_merge_different_known_full_names():
    norm = corpus.normalized_author
    assert corpus.related_authors({norm("Jackson, D. R.")}, {norm("Jackson, David R.")})
    assert not corpus.related_authors({norm("Zhao, Jiajun")}, {norm("Zhao, Jianlin")})
    assert not corpus.related_authors({norm("Kim, Young Ju")}, {norm("Kim, You-Na")})
    assert corpus.related_authors({norm("Brückner, H.")}, {norm("Bruckner, H.")})


def test_group_assignment_prevents_source_or_author_leakage():
    groups = [["P01", "P02"]] + [[f"P{i:02}"] for i in range(3, 23)]
    split = experiment.split_groups(groups, {"P01"})
    assert {"P01", "P02"} <= set(split["fit"])
    assert sum(map(len, split.values())) == 22
    assert len(set(p for ids in split.values() for p in ids)) == 22
    assert split == experiment.split_groups(groups, {"P01"})


def test_mixed_window_human_characters_do_not_count_as_false_positives():
    d = {"id": "M", "paper_id": "P", "sample_type": "mixed", "ai_spans": [(2, 4), (6, 8), (9, 11)],
         "units": [{"id": "U1", "spans": [(2, 4), (6, 8)]}, {"id": "U2", "spans": [(9, 11)]}],
         "section_units": [{"id": "SEC", "spans": [(2, 4), (6, 8), (9, 11)]}]}
    original = {**d, "id": "H", "sample_type": "original", "ai_spans": [], "units": [], "section_units": []}
    windows = [{"id": str(i), "document_id": "M", "paper_id": "P", "sample_type": "mixed", "start": a, "end": b,
                "characters": b-a, "ai_characters": corpus.overlap(d["ai_spans"], a, b), "sha256": str(i)}
               for i, (a, b) in enumerate([(0, 5), (5, 9), (9, 12)])]
    windows.append({**windows[0], "document_id": "H", "sample_type": "original", "ai_characters": 0, "id": "h"})
    raw = {str(i): {"signal": p} for i, p in enumerate([.8, .2, .2])}
    cfg = {"id": "presence-linear-refit", "models": ["signal"], "center": [0], "scale": [1],
           "intercept": 0, "coefficients": [1], "cutoff": 0}
    result = experiment.metrics({"documents": [d, original], "windows": windows}, raw, cfg)
    assert (result["false_windows"], result["pure_human_windows"]) == (1, 1)
    assert (result["hit_units"], result["narrative_units"], result["hit_sections"]) == (1, 2, 1)
    assert (result["covered_generated_characters"], result["generated_characters"]) == (2, 6)
    assert result["positive_window_human_characters"] == 3
    assert result["positive_window_human_fraction"] == .6


def test_calibration_uses_only_pure_human_training_windows():
    cfg = {"id": "presence-linear-refit", "models": ["signal"], "center": [0], "scale": [1],
           "intercept": 0, "coefficients": [1]}
    rows = [{"paper_id": p, "sha256": h, "ai_characters": a} for p, h, a in
            [("F", "human", 0), ("F", "ai", 1), ("TEST", "held", 0)]]
    raw = {"human": {"signal": .5}, "ai": {"signal": .999}, "held": {"signal": .9999}}
    calibrated = experiment.calibrate({"windows": rows}, raw, cfg, {"F"})
    assert math.isclose(calibrated["cutoff"], .1)
    assert calibrated["calibration_papers"] == ["F"]


def test_acceptance_refuses_to_score_before_freeze(monkeypatch):
    monkeypatch.setattr(experiment, "verify_inputs", lambda **k: {})
    def missing_freeze():
        raise ValueError("Final freeze is missing")
    monkeypatch.setattr(experiment, "verify_freeze", missing_freeze)
    with pytest.raises(ValueError, match="freeze"):
        experiment.score("acceptance")
