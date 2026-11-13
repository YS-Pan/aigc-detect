"""Portable UTF-8 annotation bundles; only Python standard-library dependencies."""

import copy
import hashlib
import json
from pathlib import Path


def save_annotation(report, text, directory, *, source_path=None):
    """Save a new bundle without overwriting an existing report directory."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if report.get("status") != "complete" or report["input"]["sha256"] != digest or report["input"]["characters"] != len(text):
        raise ValueError("The complete report must match the provided source text")
    document = copy.deepcopy(report)
    cursor = 0
    for segment in document["segments"]:
        start, end = segment["char_start"], segment["char_end"]
        if start != cursor or not start < end <= len(text):
            raise ValueError("Segment spans must partition the complete source text")
        passage = text[start:end]
        if segment["sha256"] != hashlib.sha256(passage.encode("utf-8")).hexdigest():
            raise ValueError("Segment hash does not match source")
        if "text" in segment and segment["text"] != passage:
            raise ValueError("Segment text does not match source")
        segment["text"] = passage
        cursor = end
    if cursor != len(text):
        raise ValueError("The annotation does not cover the entire input")
    directory = Path(directory).resolve()
    document["source"] = {"snapshot": "source.txt", "encoding": "utf-8", "bom_in_snapshot": False,
                          **({"original_path": str(Path(source_path).resolve())} if source_path else {})}
    html = render_html(document)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "source.txt").write_bytes(text.encode("utf-8"))
    (directory / "report.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "segments.jsonl").write_text("".join(json.dumps(segment, ensure_ascii=False) + "\n" for segment in document["segments"]), encoding="utf-8")
    (directory / "report.html").write_text(html, encoding="utf-8")
    (directory / "COMPLETE.json").write_text(json.dumps({"schema_version": document["schema_version"], "status": "complete",
                    "input_sha256": digest, "files": ["source.txt", "report.json", "segments.jsonl", "report.html"]}, indent=2) + "\n", encoding="utf-8")
    return {"directory": str(directory), "json": str(directory / "report.json"),
            "jsonl": str(directory / "segments.jsonl"), "html": str(directory / "report.html"),
            "source": str(directory / "source.txt")}


def render_html(report):
    # Text is never interpolated into executable JavaScript or HTML elements.
    # Escape '<' in the JSON payload so source text cannot end the data script.
    data = json.dumps(report, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return HTML.replace("__ANNOTATION_DATA__", data)


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>AIGC 分段标注</title>
<style>
:root{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#24352f;background:#f5f6f2;line-height:1.6;font-size:15px}
*{box-sizing:border-box}body{margin:0}header{padding:25px 4vw 19px;background:#173e34;color:#fff}h1{margin:0 0 8px;font-size:27px;letter-spacing:.02em}header p{margin:4px 0;color:#d6e5dd}main{padding:20px 4vw 60px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;background:#fff;padding:14px;border:1px solid #d7dfd8;border-radius:8px;position:sticky;top:0;z-index:2}select,button{font:inherit;padding:6px 10px;background:#fff;border:1px solid #bbc9c0;border-radius:5px;cursor:pointer}button:hover{background:#eef3ed}#counts{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.count{background:#fff;border:1px solid #d7dfd8;border-left:5px solid #98a99c;border-radius:5px;padding:9px 14px}.count strong{font-size:20px;margin-right:7px}.count.strong_signal{border-left-color:#c94b47}.count.possible_signal{border-left-color:#d5a52b}.layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:22px}.document{background:#fff;padding:24px 30px;border:1px solid #d7dfd8;border-radius:8px;min-width:0}.segment{white-space:pre-wrap;overflow-wrap:anywhere;cursor:pointer;border-left:3px solid transparent;padding:9px 13px;margin:0 0 9px;border-radius:3px;scroll-margin-top:95px}.segment.strong_signal{background:#fff0ee;border-left-color:#c94b47}.segment.possible_signal{background:#fff8dd;border-left-color:#d5a52b}.segment.other{background:#f5f7f4;border-left-color:#bbc7bd}.segment.selected{outline:2px solid #2d6352}.segment.dim{opacity:.28}.segment .tag{font:12px "Segoe UI","Microsoft YaHei",sans-serif;display:block;color:#65746a;margin-bottom:6px}.body-text{line-height:1.95}aside{background:#fff;border:1px solid #d7dfd8;border-radius:8px;padding:20px;height:fit-content;position:sticky;top:85px;max-height:calc(100vh - 105px);overflow:auto}aside h2{font-size:18px;margin:0 0 12px}.detail-row{margin:8px 0;overflow-wrap:anywhere}.muted{font-size:13px;color:#64756b}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:8px 3px;border-bottom:1px solid #e8ece5}.note{background:#edf2eb;padding:12px;border-radius:5px;font-size:13px;margin-top:17px}.empty{padding:16px;color:#64756b}@media(max-width:850px){.layout{grid-template-columns:1fr}aside{position:static;max-height:none}.document{padding:15px}.controls{position:static}}
.note{overflow-wrap:anywhere}
</style></head><body>
<header><h1>AIGC 分段标注</h1><p id="metadata"></p><p id="method"></p></header>
<main><div class="controls"><label>查看 <select id="model"><option value="combined">全部模型 · 综合结果</option></select></label><label>突出 <select id="category"><option value="all">全部片段</option><option value="strong_signal">明显 AIGC 特征</option><option value="possible_signal">较可能 AIGC</option><option value="other">其他</option></select></label><button id="previous">上一个标记</button><button id="next">下一个标记</button><span id="position" class="muted"></span></div>
<div id="counts"></div><div class="layout"><article id="document" class="document"></article><aside id="detail"><h2>片段详情</h2><p class="muted">点击正文片段，查看原文位置、各模型原始判断及融合贡献。</p></aside></div>
<p class="note" id="policy"></p></main>
<script type="application/json" id="data">__ANNOTATION_DATA__</script>
<script>
'use strict';
const report=JSON.parse(document.getElementById('data').textContent), segments=report.segments;
const $=id=>document.getElementById(id), names=report.policy.labels;
const ui={model:'combined',category:'all',selected:-1};
const dual=report.policy.threshold_mode==='separate_heads';
const nonlinear=report.policy.combination.startsWith('nonlinear_');
function element(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function label(s){return ui.model==='combined'?s.decision.category:s.model_scores[ui.model].category;}
function score(s){return ui.model==='combined'?(Object.hasOwn(s.decision,'ensemble_score')?s.decision.ensemble_score:s.decision.max_ai_score):s.model_scores[ui.model].ai_score;}
function fmt(value){return value==null?'不可评分':value.toFixed(4);}
function scoreText(s){return ui.model==='combined'&&dual?`强信号分 ${fmt(s.decision.ensemble_score)} · 复核分 ${fmt(s.decision.review_score)}`:`分数 ${fmt(score(s))}`;}
for(const m of report.models){if(m.kind==='statistical')continue;const opt=element('option',m.id);opt.value=m.id;$('model').append(opt);}
document.querySelector('h1').textContent=document.title=(report.segmentation.id==='english-natural-boundaries-v1'?'英文':'中文')+' AIGC 分段标注';
$('metadata').textContent=`${report.input.characters.toLocaleString()} 字符 · ${segments.length} 个片段 · ${new Date(report.created_at).toLocaleString()} · ${report.device}`;
$('method').textContent=nonlinear?`${report.models.length} 个信号非线性融合 · 同一分数、两条门限 · 计算贡献可追溯`:dual?`${report.models.length} 个信号双头融合 · 强信号和疑似分别判断 · 模型贡献可追溯`:report.policy.reference_fitted?`${report.models.length} 个模型参考拟合融合 · 各模型原始分数和贡献可追溯`:'敏感筛查 · 任一模型高分即突出标记';
$('policy').textContent=`规则 ${report.policy.id}：强信号门限 ${report.policy.high_threshold.toFixed(6)}，复核门限 ${report.policy.possible_threshold.toFixed(6)}。${dual?'两条门限分别作用于强信号分和复核分；优先判强，其余片段再判疑似。':''}${report.policy.reference_fitted?'此配置在有限参考材料上拟合，泛化范围有限；融合分数不是 AI 写作概率。':'这些分档尚未校准，表示检测器信号。'}${report.policy.validation_warning||''}“其他”不认证人类写作。字符统计表示被标记范围，不是实际 AI 写作比例。原文 SHA-256：${report.input.sha256}`;
const nodes=segments.map(s=>{const node=element('section');node.id=s.id;node.append(element('span','','tag'),element('span',s.text??'此报告未内嵌原文','body-text'));node.addEventListener('click',()=>select(s.index));$('document').append(node);return node;});
function redraw(){const counts={};for(const k of Object.keys(names))counts[k]={n:0,c:0};segments.forEach((s,i)=>{const k=label(s);counts[k].n++;counts[k].c+=s.char_end-s.char_start;nodes[i].className=`segment ${k}${ui.selected===i?' selected':''}${ui.category!=='all'&&ui.category!==k?' dim':''}`;nodes[i].firstChild.textContent=`${s.id} · ${names[k]} · ${scoreText(s)} · 第 ${s.line_start}–${s.line_end} 行${s.decision.model_disagreement?' · 模型有分歧':''}`;});$('counts').replaceChildren(...Object.entries(counts).map(([k,v])=>{const node=element('div',undefined,`count ${k}`);node.append(element('strong',v.n.toLocaleString()),element('span',`${names[k]} · ${v.c.toLocaleString()} 字符 · ${(100*v.c/report.input.characters).toFixed(2)}% 标记范围`));return node;}));if(ui.selected>=0)details(segments[ui.selected]);}
function details(s){const target=$('detail');target.replaceChildren(element('h2',`${s.id} · ${names[label(s)]}`));target.append(element('div',`字符区间 [${s.char_start}, ${s.char_end})`,'detail-row'),element('div',`原文第 ${s.line_start}–${s.line_end} 行`,'detail-row'),element('div',`${report.policy.reference_fitted?'综合参与模型':'触发模型'}：${s.decision.triggered_models.join('、')||'未达到分档线'}`,'detail-row'));
if(dual)target.append(element('div',`强信号分 ${fmt(s.decision.ensemble_score)} / 门限 ${fmt(report.policy.high_threshold)}`,'detail-row'),element('div',`复核分 ${fmt(s.decision.review_score)} / 门限 ${fmt(report.policy.possible_threshold)}`,'detail-row'));
const table=element('table'),head=element('tr');['模型','原始分数 / 统计量','原始档位'].forEach(t=>head.append(element('th',t)));table.append(head);
for(const [name,m] of Object.entries(s.model_scores)){const row=element('tr');[name,fmt(Object.hasOwn(m,'raw_score')?m.raw_score:m.ai_score),m.kind==='statistical'?'未单独分档':names[m.category]].forEach(t=>row.append(element('td',t)));table.append(row);}target.append(table);
if(s.decision.contributions){
  const ct=element('table'),ch=element('tr');(dual?['模型','强信号','复核']:['模型','贡献']).forEach(t=>ch.append(element('th',t)));ct.append(ch);
  for(const [name,item] of Object.entries(s.decision.contributions)){const row=element('tr');row.append(element('td',name),element('td',fmt(item.logit_contribution)));if(dual)row.append(element('td',fmt(s.decision.review_contributions[name]?.logit_contribution)));ct.append(row);}
  const accounting=nonlinear?(report.policy.profile.heads.strong.kind==='tanh_mlp'?'沿标准化零点到当前分数积分梯度，分摊隐藏单元的变化；贡献加截距等于 logit。':'按坐标引起的核对数变化分摊核函数变化；贡献加截距等于 logit。'):dual?'交互项在两个模型间各分一半；贡献加上截距等于该头的 logit。':'显示各模型对综合 logit 的有符号贡献；原始分数档位不直接决定融合标签。';
  target.append(element('h2','对融合 logit 的贡献'),element('p',accounting+' 正负数值用于核对计算，不代表模型的因果重要性。','muted'),ct);
}
if(s.decision.feature_contributions?.length){
  const panel=element('details'),ft=element('table'),fh=element('tr');panel.append(element('summary',nonlinear?'展开核函数 / 隐藏单元':'展开线性、平方与交互项'));
  ['项','强信号','复核'].forEach(t=>fh.append(element('th',t)));ft.append(fh);
  const ordered=[...s.decision.feature_contributions].sort((a,b)=>Math.abs(b.logit_contributions.review)-Math.abs(a.logit_contributions.review));
  for(const item of ordered){const row=element('tr');[item.name||item.models.join(' × '),fmt(item.logit_contributions.strong),fmt(item.logit_contributions.review)].forEach(t=>row.append(element('td',t)));ft.append(row);}panel.append(ft);target.append(panel);
}
if(s.warnings.length)target.append(element('p',`提示：${s.warnings.join('、')}`,'muted'));target.append(element('p',`片段 hash：${s.sha256}`,'muted'),element('p','位置按原文 Python Unicode 字符计数，区间左闭右开。每个片段独立检测，标签的精度止于片段。','note'));$('position').textContent=`${s.index+1} / ${segments.length}`;}
function select(index,scroll=false){ui.selected=index;redraw();if(scroll)nodes[index].scrollIntoView({behavior:'smooth',block:'center'});}
function jump(direction){let hits=segments.filter(s=>ui.category==='all'?label(s)!=='other':label(s)===ui.category).map(s=>s.index);if(!hits.length){$('position').textContent='当前视图无对应标记';return;}let pos=hits.indexOf(ui.selected);pos=pos<0?(direction>0?0:hits.length-1):(pos+direction+hits.length)%hits.length;select(hits[pos],true);}
$('model').addEventListener('change',e=>{ui.model=e.target.value;redraw();});$('category').addEventListener('change',e=>{ui.category=e.target.value;redraw();});$('previous').addEventListener('click',()=>jump(-1));$('next').addEventListener('click',()=>jump(1));redraw();
</script></body></html>'''
