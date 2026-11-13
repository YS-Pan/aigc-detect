"""Correct source-audit anchor punctuation and quantify that extraction artifact.

No candidate, label, threshold or local-scanning result is changed. The first
source comparison is preserved and explicitly superseded by this narrow check.
"""
import gc
import json
from pathlib import Path
import re
import socket
import statistics
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import english_frozen_local as e
from scripts import english_frozen_controls as c


def keep_terminal_punctuation(container, excerpt):
    start=container.index(excerpt)
    end=start+len(excerpt)
    # Restore adjacent sentence punctuation only, not next paragraph/figure text.
    while end<len(container) and container[end] in '.,;:!?\"\u2019\u201d)\]':
        end+=1
    return container[start:end]


def main():
    c.verify()
    target=e.EVIDENCE/'source-boundary-check'
    private=e.REPORTS/'source-boundary-check'
    if (target/'evaluation.json').exists():
        raise FileExistsError('Preserve correction evidence')
    base=e.read(c.EP/'inputs.json')['source_pairs']
    originals={r['id']:r for r in e.read(e.REPORTS/'source-pairs.json')}
    passages,rows={},[]
    def add(text):
        h=e.digest(text)
        passages[h]=text
        return h
    import fitz
    for p in base:
        original=originals[p['id']]
        with fitz.open(e.SAMPLES/f"upstream/sources/{p['paper_id']}.pdf") as doc:
            page=doc[p['page']-1]
            regions=p['regions']
            if regions:
                container=e.clean('\n'.join(page.get_text('text',clip=fitz.Rect(r)) for r in regions))
            elif original['bbox']:
                container=e.clean(page.get_text('text',clip=fitz.Rect(original['bbox'])+(-.8,-.8,.8,.8),sort=False))
            else:
                container=e.clean(page.get_text('text',sort=False))
        excerpt,similarity=e.page_excerpt(container,original['converted'])
        mechanical=keep_terminal_punctuation(container,excerpt)
        converted=original['converted']
        no_punctuation=re.sub(r'[.,;:!?]+$','',converted)
        rows.append({'id':p['id'],'converted_hash':add(converted),'mechanical_hash':add(mechanical),
                     'no_terminal_punctuation_hash':add(no_punctuation),'word_similarity':similarity,
                     'text_identical_after_boundary_correction':converted==mechanical})
    e.write(private/'passages.json',passages)
    declaration={'code_sha256':e.sha(__file__),'parent_declaration_sha256':e.sha(c.EP/'declaration.json'),
                 'reason':'Lexical anchor matcher dropped terminal punctuation. Restore actual adjacent source punctuation; retain earlier scores as extraction-confounded diagnostics.',
                 'rows':rows,'passages_sha256':e.sha(private/'passages.json'),'selection_and_thresholds_unchanged':True}
    if (target/'declaration.json').exists():
        assert declaration==e.read(target/'declaration.json')
    else:e.write(target/'declaration.json',declaration)
    import torch
    from aigc_detector.engine import Detector
    torch.set_num_threads(1)
    detector=Detector(device='cuda',batch_size=8)
    scores={h:{} for h in passages}
    def blocked(*a,**k):raise AssertionError('Inference network blocked')
    with patch.object(socket.socket,'connect',blocked),patch.object(socket,'create_connection',blocked):
        for m in e.MODELS:
            cache_path=private/(m+'.json')
            cache=e.read(cache_path) if cache_path.exists() else {'declaration_sha256':e.sha(target/'declaration.json'),'items':{}}
            assert cache['declaration_sha256']==e.sha(target/'declaration.json')
            _,model,_=detector._load(m)
            model.requires_grad_(False)
            keys=sorted(set(passages)-set(cache['items']))
            for i in range(0,len(keys),24):
                batch=keys[i:i+24]
                values=detector._score_texts([passages[h] for h in batch],m,64,True)
                for h,v in zip(batch,values):cache['items'][h]=v
                e.write(cache_path,cache)
            for h in passages:scores[h][m]=cache['items'][h]['ai_score']
            detector._loaded.clear()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    result={}
    for m in e.MODELS:
        out=[]
        for p in rows:
            original=scores[p['converted_hash']][m]
            mechanical=scores[p['mechanical_hash']][m]
            ablated=scores[p['no_terminal_punctuation_hash']][m]
            out.append({'id':p['id'],'converted_score':original,'mechanical_score':mechanical,'no_terminal_punctuation_score':ablated,
                        'source_delta_logit':e.logit(original)-e.logit(mechanical),'punctuation_delta_logit':e.logit(original)-e.logit(ablated)})
        result[m]={'rows':out,'median_source_delta_logit':statistics.median(r['source_delta_logit'] for r in out),
                   'median_punctuation_delta_logit':statistics.median(r['punctuation_delta_logit'] for r in out),
                   'source_max_absolute_probability_difference':max(abs(r['converted_score']-r['mechanical_score']) for r in out),
                   'punctuation_max_absolute_probability_difference':max(abs(r['converted_score']-r['no_terminal_punctuation_score']) for r in out)}
    payload={'results':result,'declaration_sha256':e.sha(target/'declaration.json'),'corrected_pairs':len(rows),
             'identical_text_pairs':sum(p['text_identical_after_boundary_correction'] for p in rows),'no_training':True,'contains_source_text':False}
    e.write(target/'evaluation.json',payload)
    print(json.dumps({**{m:{k:v for k,v in value.items() if k!='rows'} for m,value in result.items()},'identical_text_pairs':payload['identical_text_pairs']}),flush=True)


if __name__=='__main__':main()
