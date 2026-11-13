"""Independent count/threshold audits and an offline experiment report."""
import argparse
from collections import defaultdict
from difflib import HtmlDiff
import gc
import hashlib
import html
import json
from pathlib import Path
import socket
import statistics
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import english_frozen_local as e
from scripts import english_frozen_controls as c


def validate_inference():
    c.verify()
    data=e.read(e.EVIDENCE/'inputs.json')
    passages=e.read(e.REPORTS/'passages.json')
    scores={**e.read(e.EVIDENCE/'development-scores.json')['scores'],**e.read(e.EVIDENCE/'repeat-scores.json')['scores']}
    chosen=[next(r['hash'] for r in data['local'] if r['split']==sp and r['positive']==label) for sp in ['fit','test'] for label in [False,True]]
    chosen += [next(r['hash'] for r in data['contexts'] if r['family']=='intact' and r['fraction']==.25 and r['position']=='middle' and r['kind']=='ai_mixed')]
    new=e.read(c.EP/'inputs.json')
    new_scores=e.read(c.EP/'test-scores.json')['scores']
    new_cut=e.read(c.EP/'CALIBRATION.json')['thresholds']['sentence3::llm-detectaive-en']
    chosen.append(next(r['hash'] for r in new['new_human_windows'] if r['paper_id']=='N04' and r['mode']=='sentence3' and e.logit(new_scores[r['hash']]['llm-detectaive-en'])>=new_cut))
    passages.update(e.read(c.RP/'passages.json'))
    scores.update(e.read(c.EP/'test-scores.json')['scores'])
    import torch
    from aigc_detector.engine import Detector
    torch.set_num_threads(1)
    detector=Detector(device='cuda',batch_size=8)
    result={}
    def blocked(*a,**k):raise AssertionError('No inference network')
    with patch.object(socket.socket,'connect',blocked),patch.object(socket,'create_connection',blocked):
        for m in e.MODELS:
            _,model,_=detector._load(m)
            model.requires_grad_(False)
            errors=[]
            for h in chosen:
                value=detector._score_texts([passages[h]],m,64,True)[0]
                error=abs(value['ai_score']-scores[h][m])
                assert error<2e-5,(m,h,error)
                assert not value['truncated']
                errors.append(error)
            result[m]={'inputs':len(chosen),'max_absolute_probability_error':max(errors)}
            detector._loaded.clear()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    e.write(e.EVIDENCE/'inference-validation.json',{'single_vs_batched':result,'network_blocked':True,'no_gradients':True})
    print(json.dumps(result),flush=True)


def independent_counts(data,scores,candidate,papers,cut):
    rows=[r for r in data['local'] if r['paper_id'] in papers and r['mode']==candidate['mode']]
    flagged=[r for r in rows if e.logit(scores[r['hash']][candidate['model']])>=cut]
    human=[r for r in rows if not r['positive']]
    false=[r for r in flagged if not r['positive']]
    blocks=[b for b in data['blocks'] if b['paper_id'] in papers]
    hits,covered=0,0
    for b in blocks:
        positions={i for r in flagged if r['sample_id']==b['sample_id'] for i in range(max(b['start'],r['start']),min(b['end'],r['end']))}
        hits+=bool(positions)
        covered+=len(positions)
    return {'human_windows':len(human),'false_windows':len(false),'mixed_blocks':len(blocks),'mixed_hits':hits,'covered_ai_characters':covered}


def audit():
    config=c.verify()
    data=e.read(e.EVIDENCE/'inputs.json')
    scores={**e.read(e.EVIDENCE/'development-scores.json')['scores'],**e.read(e.EVIDENCE/'repeat-scores.json')['scores']}
    selection=e.read(e.EVIDENCE/'selection.json')
    evaluation=e.read(e.EVIDENCE/'evaluation.json')
    all_papers=set().union(*[set(ps) for ps in config['groups'].values()])
    checked=0
    for f in selection['frozen_candidates']:
        candidate,cut=f['candidate'],f['threshold']
        maximum=max(e.logit(scores[r['hash']][candidate['model']]) for r in data['local'] if r['paper_id'] in selection['control_papers'] and r['mode']==candidate['mode'] and not r['positive'])
        assert cut==maximum+.5
        for split,papers in [('all',all_papers),*config['groups'].items()]:
            expected=evaluation['results'][candidate['id']]['all' if split=='all' else 'by_split']
            if split!='all':expected=expected[split]
            result=independent_counts(data,scores,candidate,set(papers),cut)
            assert all(expected[k]==v for k,v in result.items())
            checked+=1
    folds=0
    for outer in selection['nested']:
        held=set(outer['held'])
        assert not held&set(outer['train'])
        for candidate in outer['inner']:
            for inner in candidate['folds']:
                train=set(inner['train'])
                assert not train&set(inner['held']) and not held&(train|set(inner['held']))
                for group in config['author_groups']:
                    assert not (set(group)&train and set(group)&(held|set(inner['held'])))
                cm=candidate['candidate']
                maximum=max(e.logit(scores[r['hash']][cm['model']]) for r in data['local'] if r['paper_id'] in train and r['mode']==cm['mode'] and not r['positive'])
                assert inner['threshold']==maximum+.5
                folds+=1
    samples=[json.loads(s) for s in (e.SAMPLES/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    source={s['id']:(e.SAMPLES/s['text_file']).read_text(encoding='utf-8') for s in samples}
    passages=e.read(e.REPORTS/'passages.json')
    for r in data['local']:
        assert source[r['sample_id']][r['start']:r['end']]==passages[r['hash']]
        assert r['positive']==any(b['sample_id']==r['sample_id'] and e.intersects(r['start'],r['end'],b['start'],b['end']) for b in data['blocks'])
    pairs=defaultdict(dict)
    for r in data['contexts']:
        assert len(passages[r['hash']])==r['characters']
        key=(r['paragraph_id'],r['family'],r['fraction'],r['position'])
        pairs[key][r['kind']]=r
    for group in pairs.values():
        ai=group.get('ai_mixed',group.get('ai_isolated'))
        h=group['human_splice']
        a,b=passages[ai['hash']],passages[h['hash']]
        lo,hi=ai['ai_start'],ai['ai_end']
        assert len(a)==len(b) and a[:lo]==b[:lo] and a[hi:]==b[hi:]
        original=e.clean(source[ai['paragraph_id']])
        assert a[lo:hi]==(original[:hi-lo] if ai['prefix_only'] else original)
    # Recompute the unchanged v2 with its historical score-only evaluator.
    from scripts import english_specialized_fusion as old
    from scripts import english_hard_negative as hn
    old_data={k:[] for k in ['records','blocks']}
    for part in ['development-data.json','repeat-data.json']:
        d=e.read(old.EVIDENCE/part)
        for k in old_data:old_data[k]+=d[k]
    baseline=old.baseline()
    v2={k:hn.metrics(old_data,set(ps),baseline['model'],baseline['threshold']) for k,ps in config['groups'].items()}
    v2['all']=hn.metrics(old_data,all_papers,baseline['model'],baseline['threshold'])
    manifest=e.read(ROOT/'models/manifest.json')['models']
    verified_files=0
    for model in e.MODELS:
        for filename,expected in manifest[model]['files'].items():
            path=ROOT/'models'/model/filename
            assert path.stat().st_size==expected['bytes'],str(path)
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==expected['sha256'],str(path)
            verified_files+=1
    result={'frozen_result_groups_checked':checked,'inner_threshold_folds_checked':folds,'local_text_spans_checked':len(data['local']),
            'context_triplets_checked':len(pairs),'author_group_isolation':True,'v2_baseline':v2,
            'model_assets_verified_against_manifest':{'files':verified_files,'models':e.MODELS},'new_source_recall_evaluated':False,
            'default_profile_sha256':e.sha(e.PROFILE),'contains_source_text':False}
    e.write(e.EVIDENCE/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='v2_baseline'}),flush=True)


def report():
    audit()
    data=e.read(e.EVIDENCE/'inputs.json')
    evaluation=e.read(e.EVIDENCE/'evaluation.json')
    selection=e.read(e.EVIDENCE/'selection.json')
    new=e.read(c.EP/'evaluation.json')
    scores={**e.read(e.EVIDENCE/'development-scores.json')['scores'],**e.read(e.EVIDENCE/'repeat-scores.json')['scores']}
    selected=evaluation['selected']['id']
    per_generator={}
    hit=set(evaluation['results'][selected]['all']['hit_ids'])
    for generator in sorted({b['generator'] for b in data['blocks']}):
        blocks=[b for b in data['blocks'] if b['generator']==generator]
        per_generator[generator]={'blocks':len(blocks),'hits':sum(b['id'] in hit for b in blocks)}
    lengths={}
    for name,left,right in [('under400',0,400),('400to799',400,800),('800plus',800,100000)]:
        blocks=[b for b in data['blocks'] if left<=b['end']-b['start']<right]
        lengths[name]={'blocks':len(blocks),'hits':sum(b['id'] in hit for b in blocks)}
    contexts=[]
    isolated={r['paragraph_id']:r for r in data['contexts'] if r['family']=='intact' and r['kind']=='ai_isolated'}
    for model in e.MODELS:
        for fraction in [.75,.5,.25]:
            for position in ['start','middle','end']:
                rows=[r for r in data['contexts'] if r['family']=='intact' and r['kind']=='ai_mixed' and r['fraction']==fraction and r['position']==position]
                delta=[e.logit(scores[r['hash']][model])-e.logit(scores[isolated[r['paragraph_id']]['hash']][model]) for r in rows]
                contexts.append({'model':model,'fraction':fraction,'position':position,'count':len(delta),'score_decreased':sum(d<0 for d in delta),'median_delta_logit':statistics.median(delta)})
    corrected=e.read(e.EVIDENCE/'source-boundary-check/evaluation.json')
    source_summary={m:{'reviewed_pairs':len(v['rows']),'median_delta_logit':v['median_source_delta_logit'],
                       'median_punctuation_delta_logit':v['median_punctuation_delta_logit'],
                       'max_abs_probability_difference':v['source_max_absolute_probability_difference']} for m,v in corrected['results'].items()}
    e.write(e.EVIDENCE/'diagnostics.json',{'selected_by_generator':per_generator,'selected_by_ai_length':lengths,'context_paired_deltas':contexts,'reviewed_source_summary':source_summary,'contains_source_text':False})
    esc=html.escape
    table=[]
    for key,value in evaluation['results'].items():
        cv=next(x['metrics'] for x in selection['selection'] if x['candidate']['id']==key)
        allm,rep=value['all'],value['by_split']['test']
        table.append('<tr>'+''.join(f'<td>{esc(str(v))}</td>' for v in [key,f"{cv['false_windows']}/{cv['human_windows']}",f"{cv['mixed_hits']}/51",f"{rep['false_windows']}/{rep['human_windows']}",f"{rep['mixed_hits']}/12",f"{allm['mixed_hits']}/75"] )+'</tr>')
    audit_data=e.read(c.EP/'inputs.json')
    passages=e.read(c.RP/'passages.json')
    passages.update(e.read(e.REPORTS/'source-boundary-check/passages.json'))
    corrected_pairs={r['id']:r for r in e.read(e.EVIDENCE/'source-boundary-check/declaration.json')['rows']}
    diff=[]
    for p in audit_data['source_pairs']:
        p={**p,**corrected_pairs[p['id']]}
        diff.append(f"<details><summary>{esc(p['id'])} · 源第 {p['page']} 页 · 词序相似度 {p['word_similarity']:.3f}</summary><p><a href='../audit-pages/{p['paper_id']}-p{p['page']:03}.png'>打开原页</a></p>")
        diff.append(HtmlDiff(wrapcolumn=75).make_table([passages[p['converted_hash']]],[passages[p['mechanical_hash']]],fromdesc='现有转写的机械清洗',todesc='原PDF机械提取',context=False))
        diff.append('</details>')
    human=[]
    for pid,v in new['results'][selected]['by_paper'].items():
        src=next(s for s in audit_data['sources'] if s['id']==pid)
        human.append(f"<tr><td>{pid} ({esc(src['role'])})</td><td><a href='{src['url']}'>{esc(src['title'])}</a></td><td>{v['new_threshold_flags']}/{v['windows']}</td></tr>")
    context_table=[]
    for m,rows in evaluation['context_controls'].items():
        for fraction in [1,.75,.5,.25]:
            position='start' if fraction==1 else 'middle'
            match=[r for r in rows if r['family']=='intact' and r['fraction']==fraction and r['position']==position]
            ai=next(r for r in match if r['kind'].startswith('ai_'))
            hs=next(r for r in match if r['kind']=='human_splice')
            hc=next(r for r in match if r['kind']=='human_contiguous')
            context_table.append(f"<tr><td>{m}</td><td>{fraction:.0%}</td><td>{ai['paper_held_out_flags']}/75</td><td>{hs['paper_held_out_flags']}/75</td><td>{hc['paper_held_out_flags']}/75</td></tr>")
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>冻结模型：来源、上下文与局部扫描</title>
<style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font-family:system-ui,'Microsoft YaHei',sans-serif;color:#16232f;line-height:1.65;background:#f7fafb}}h1{{line-height:1.3}}section{{background:white;padding:24px;margin:20px 0;border:1px solid #dce5e9;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dce5e9;text-align:left;padding:8px;vertical-align:top}}.diff{{font-family:monospace}}.diff_add{{background:#d8f2dc}}.diff_sub{{background:#ffdfdf}}.diff_chg{{background:#fff0bb}}details{{overflow:auto;margin:12px 0}}a{{color:#096785}}.status{{background:#ffeecb;padding:16px;border-radius:8px}}</style>
<h1>冻结模型可行性实验</h1><p>2026-10-04 · 本地推理 · 底层权重和分类头均未训练 · 默认 v2 保留</p>
<p class="status">未达标：完整选择流程嵌套留出人工误报 3/3103、混合生成段命中 3/51；冻结后旧 P07/P08 复测 0/473、0/12。两篇新来源人工验收 1/172 误报。停止扩大本轮固定局部扫描搜索。</p>
<section><h2>范围与口径</h2><p>4 个专业模型 × 3 种扫描，共 12 个预声明方案；原十篇、三十份混合文档的 6682 个窗口；75 处生成段的 3933 条受控输入。任意 AIGC 交集即阳性；只有完全人工窗口判正是误报。所有局部扫描机会一起取人工最高分校准。P01/P02 按共同作者绑定，其余按已记录作者分组。</p><p>旧十篇均已参与多轮开发。本轮不重新宣称独立召回验证。四篇新论文只检验人工误报，未生成新的任务匹配正例；预训练材料重合情况未知。不同扫描方案的窗口数不同，不能把它们当独立论文数。</p></section>
<section><h2>局部扫描</h2><p>开发选中 {esc(selected)}。表中“选择折”参与挑选方案，不是完整嵌套成绩。所有 12 个方案在旧复测均 0/12，不能换选测试表现较好的其它方法。现有 v2 基线仍为 0/58、2/12。</p><table><thead><tr><th>固定方案</th><th>选择折人工误报</th><th>选择折命中</th><th>旧复测人工误报</th><th>旧复测命中</th><th>全资料命中</th></tr></thead><tbody>{''.join(table)}</tbody></table></section>
<section><h2>上下文对照</h2><p>下表为完整同一生成段放在中间的位置对照（100% 为单独输入）。门限由其它作者组的全部配对人工输入决定。人工拼接与 AI 拼接使用相同上下文和边界；另有自然连续人工对照。比例改变时总长度也随之改变，不能归因于比例单一因素。固定 1600 字符的前缀实验单列在数值证据中，不冒充完整段落。</p><table><tr><th>模型</th><th>生成比例</th><th>生成输入标记</th><th>人工拼接误报</th><th>连续人工误报</th></tr>{''.join(context_table)}</table><p>字符级拼接可能截断人工单词，属于构造压力测试；自然混合论文的结果见上一表，未用真值位置指导实际扫描。</p></section>
<section><h2>新的人工来源</h2><p>N01/N02 校准，锁定后只运行一次 N03/N04 验收。未按新测试结果重设门限或选择模型。来源均为 2016—2020 年公开工程论文，CC BY 4.0；取摘要与全部正文段落，排除图表说明、参考文献、公式和致谢。</p><table><tr><th>来源角色</th><th>来源</th><th>选中方案人工误报</th></tr>{''.join(human)}</table></section>
<section><h2>来源核查</h2><p>410 个原文块中 377 对满足预声明自动对齐条件；其余保留为提取失败/短片段，未删除原检测难例。4 个人工高分窗口和 6 个随机窗口涉及 29 块、12 页，已目视回查。四处机械提取补齐了跨栏内容或排除了图表干扰，原转写及人工标签保持。</p><p>检查范围内未发现足以支持重标为 AIGC 的实质语义改写证据；这不是十篇全文逐字证明。词锚定位曾遗漏句末标点，已在单独保留的 source-boundary-check 中纠正，下面差异表使用纠正后的输入。29 对中 10 对清洗文本完全相同，四模型来源 logit 差中位数均为 0。单独删除句末标点的对照显示：同一人工段的 RoBERTa Academic 分数可由 0.9977 降到 0.00130，说明边界细节足以影响分数，不能把它解释为作者或转写来源改变。残余断词、公式和标题差异也仍需区分。</p>{''.join(diff)}</section>
<section><h2>复现与边界</h2><p>脚本、模型 revision、输入/缓存/profile 哈希、冻结门限、嵌套折和逐篇计数均保存在 evaluations/english-frozen-local-20261004。正文、PDF、页面图和完整缓存仅在 Git 排除的 reports/samples 中。完整权重 SHA-256 验证覆盖四模型 29 个文件。没有修改生产配置、降低 v2 门限或训练底层网络。</p></section></html>'''
    path=e.REPORTS/'report/index.html'
    path.parent.mkdir(exist_ok=True)
    path.write_text(content,encoding='utf-8')
    print(json.dumps({'report':str(path),'selected_by_generator':per_generator,'reviewed_source_summary':source_summary}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['validate-inference','audit','report'])
    action=parser.parse_args().action
    {'validate-inference':validate_inference,'audit':audit,'report':report}[action]()
