import math

import pytest

from scripts import english_frozen_local as exp


def test_mixtures_preserve_whole_ai_and_match_human_controls():
    for fraction in [1., .75, .5, .25]:
        for position in ['start', 'middle', 'end']:
            ai, human, lo, hi = exp.make_mixture('GENERATED', 'humantext', ' context '*10, fraction, position)
            assert len(ai) == len(human) == round(9/fraction)
            assert ai[lo:hi] == 'GENERATED'
            assert human[lo:hi] == 'humantext'
            assert ai[:lo] == human[:lo] and ai[hi:] == human[hi:]
    with pytest.raises(ValueError):
        exp.make_mixture('generated', 'short', 'abc', .25, 'middle')


@pytest.mark.parametrize('mode', exp.MODES)
def test_local_boundaries_cover_every_nonwhitespace_without_labels(mode):
    text = 'First sentence. Second one! Third example. Fourth line.\n\n' + 'Long text. '*600 + '\n\nEnd.'
    spans = exp.local_spans(text, mode)
    assert all(0 <= a < b <= len(text) and b-a <= 1600 for a, b in spans)
    covered = {i for a, b in spans for i in range(a, b)}
    assert all(i in covered for i, c in enumerate(text) if not c.isspace())


def test_calibration_excludes_positive_and_held_out_rows():
    rows = [dict(paper_id=p, mode='paragraph', positive=y, hash=h) for p, y, h in [('A', False, 'a'), ('A', True, 'b'), ('B', False, 'c')]]
    scores = {'a': {'m': .2}, 'b': {'m': .999}, 'c': {'m': .9999}}
    candidate = {'model': 'm', 'mode': 'paragraph'}
    assert exp.threshold(rows, scores, candidate, {'A'}, .5) == exp.logit(.2)+.5


def test_any_positive_intersection_and_duplicate_scan_coverage():
    rows = [dict(id=str(i), sample_id='doc', paper_id='A', kind='mixed_document', mode='paragraph', start=a, end=b, positive=y, hash=str(i)) for i, (a,b,y) in enumerate([(0,10,False),(10,30,True),(20,40,True)])]
    data = {'local': rows, 'blocks': [dict(id='g', sample_id='doc', paper_id='A', start=20, end=25)]}
    result = exp.measure(data, {str(i): {'m': .9} for i in range(3)}, {'mode':'paragraph','model':'m'}, {'A'}, 0.)
    assert result['false_windows'] == 1
    assert result['mixed_hits'] == 1
    assert result['covered_ai_characters'] == 5
    assert not exp.intersects(0,10,10,20)


def test_author_connections_are_transitive():
    papers = [{'paper_id':'A','author':'One, A and Two, B'}, {'paper_id':'B','author':'Two, B and Three, C'}, {'paper_id':'C','author':'Three, C'}, {'paper_id':'D','author':'Four, D'}]
    assert exp.author_groups(papers) == [['A','B','C'],['D']]


def test_tiny_probabilities_are_not_flattened():
    assert exp.logit(1e-20) < exp.logit(1e-10) < exp.logit(.5)
    assert all(math.isfinite(exp.logit(x)) for x in [0,1e-30,.5,1])


def test_pdf_alignment_returns_only_source_characters():
    page = 'Heading ignored. These are the actual words of the published scientific paragraph. Footer ignored.'
    out, similarity = exp.page_excerpt(page, 'These are the actual words of the published scientific paragraph.')
    assert out in page
    assert similarity == 1


def test_new_source_xml_excludes_figure_caption_math_and_reference_list():
    from scripts.english_frozen_controls import prose_xml
    import xml.etree.ElementTree as ET
    root = ET.fromstring('<article><front><article-meta><abstract><p>Abstract text.</p></abstract></article-meta></front><body><sec><p>Human <xref ref-type="bibr">1</xref> prose <inline-formula>equation</inline-formula> ends.</p><fig><caption><p>Figure text.</p></caption></fig><table-wrap><p>Table text.</p></table-wrap><ref-list><p>Reference.</p></ref-list></sec></body></article>')
    text, paragraphs = prose_xml(root)
    assert paragraphs == ['Abstract text.', 'Human prose ends.']
    assert 'Figure' not in text and 'equation' not in text
