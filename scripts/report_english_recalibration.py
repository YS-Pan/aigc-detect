"""Write the expanded campaign's text-free summary and local report navigation."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.calibrate_english_v2 import EVIDENCE, REPORTS, ROOT, POLICY
from scripts.english_academic_samples import read_json, digest


def pct(value):
    return f"{100*value:.2f}%" if value is not None else "—"


def main():
    evaluation = read_json(EVIDENCE / "evaluation.json")
    selection = read_json(EVIDENCE / "selection.json")
    profile = read_json(ROOT / "aigc_detector/profiles" / f"{POLICY}.json")
    assert digest((ROOT / "aigc_detector/profiles" / f"{POLICY}.json").read_bytes()) == evaluation["profile_sha256"]
    chosen = selection["selected"]
    lines = ["# 英文扩展融合重新校准", "",
             "本轮完整核算10篇原版、30份含生成替换的混合文档及75处单独提取的生成段。拟合组的混合文档已进入训练，门限控制还包含校准组混合文档中确定为人工的片段；同论文的相同正文去重，论文及其全部派生版本绑定同一组。", "",
             "旧报告中的19个人工片段和12个生成段仅来自P07/P08两篇留出论文，并非全部材料。它们此前的测试结果已经看过，本轮评价属于重复评估；没有新的独立测试，也不据此宣称一般准确率或作者来源。", "",
             "| 集合 | 原版文档 | 混合文档 | 单独生成段 | 标注片段 |", "|---|---:|---:|---:|---:|"]
    names = {"fit": "拟合", "calibration": "门限校准", "test": "重复评估"}
    for split, g in evaluation["groups"].items():
        n = g["source_inputs"]
        lines.append(f"| {names[split]} | {n['human_document']} | {n['mixed_document']} | {n['ai_paragraph']} | {g['segments']} |")
    lines += ["", "含疑似档的结果如下；拟合/校准行属于参考适配证据。混合文档的覆盖按实际生成位置交集计量，人工溢出另计。", "",
              "| 集合 | 人工原版有标记 | 人工原版误标片段 | 单独提取生成段命中 | 混合正文生成块命中 | 混合生成字符覆盖 | 混合纯人工误标片段 |", "|---|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for split, g in evaluation["groups"].items():
        p, m, d = g["pure_inputs"]["any_flag"], g["mixed_documents"]["any_flag"], g["original_documents"]["any_flag"]
        lines.append(f"| {names[split]} | {d['documents_with_any_flag']}/{d['documents']} | {p['human_flagged_segments']}/{p['human_segments']} | {p['ai_paragraphs_with_any_flag']}/{p['ai_paragraphs']} | {m['generated_blocks_with_any_flag']}/{m['generated_blocks']} | {pct(m['ai_character_recall'])} | {m['pure_human_flagged_segments']}/{m['pure_human_segments']} |")
        csv_rows.append({"split": split, "original_documents": d["documents"], "original_documents_flagged": d["documents_with_any_flag"],
                         "human_segments": p["human_segments"], "human_flagged_segments": p["human_flagged_segments"],
                         "isolated_generated_paragraphs": p["ai_paragraphs"], "isolated_generated_paragraphs_hit": p["ai_paragraphs_with_any_flag"],
                         "mixed_generated_blocks": m["generated_blocks"], "mixed_generated_blocks_hit": m["generated_blocks_with_any_flag"],
                         "mixed_generated_character_recall": m["ai_character_recall"], "mixed_human_spillover_characters": m["human_characters_within_flagged_segments"],
                         "mixed_pure_human_segments": m["pure_human_segments"], "mixed_pure_human_flagged_segments": m["pure_human_flagged_segments"]})
    with (REPORTS / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    groups = list(evaluation["groups"].values())
    human_segments = sum(g["pure_inputs"]["any_flag"]["human_segments"] for g in groups)
    human_flags = sum(g["pure_inputs"]["any_flag"]["human_flagged_segments"] for g in groups)
    isolated_hits = sum(g["pure_inputs"]["any_flag"]["ai_paragraphs_with_any_flag"] for g in groups)
    mixed_hits = sum(g["mixed_documents"]["any_flag"]["generated_blocks_with_any_flag"] for g in groups)
    mixed_ai_chars = sum(g["mixed_documents"]["any_flag"]["ai_characters"] for g in groups)
    mixed_ai_flags = sum(g["mixed_documents"]["any_flag"]["ai_flagged_characters"] for g in groups)
    mixed_human_segments = sum(g["mixed_documents"]["any_flag"]["pure_human_segments"] for g in groups)
    mixed_human_flags = sum(g["mixed_documents"]["any_flag"]["pure_human_flagged_segments"] for g in groups)
    lines.append(f"| 全集合合计（混合证据性质） | 0/10 | {human_flags}/{human_segments} | {isolated_hits}/75 | {mixed_hits}/75 | {pct(mixed_ai_flags/mixed_ai_chars)} | {mixed_human_flags}/{mixed_human_segments} |")
    lines += ["", "合计包含用于拟合、门限控制及重复评估的材料，只用于核算整套材料，不能当作独立效果指标。"]
    lines += ["", f"最终profile为 `{POLICY}`，固定{profile['segment_chars']}/{profile['min_segment_chars']}字符分片。信号：" + "、".join(f"`{m}`" for m in profile["models"]) + "。",
              f"采用{chosen['family']}分数特征、L2={chosen['regularization']}、人工权重={chosen['human_weight']}、目标={chosen['target_mode']}。双头共用同一已拟合函数，强/复核门限分别取人写控制最大logit加预声明余量；门限数值只维护在profile。", "",
              "| 片段上限/最小目标 | 拟合论文留出人工误标字符 | 生成字符命中 | 证据性质 |", "|---|---:|---:|---|"]
    for length in [400, 800, 1600]:
        cs = [c for c in selection["candidates"] if c["segment_chars"] == length]
        best = min(cs, key=lambda c: (c["human_flagged_characters"], -c["ai_flagged_characters"]))
        total = sum(f["metrics"]["ai_characters"] for f in best["folds"])
        lines.append(f"| {length}/{best['min_segment_chars']} | {best['human_flagged_characters']} | {best['ai_flagged_characters']}/{total}（{pct(best['ai_flagged_characters']/total)}） | 已参与选择的逐论文留出诊断 |")
    lines += ["", "候选共1728个，比较旧四模型、新五模型、九模型、移除多语言模型、加入不同统计后端和统计后端单独融合；包含线性、平方、两模型交互、两种生成目标及人工权重/正则。先比较人工误标字符，再比较生成字符检出。同论文重复正文只在拟合去重，评价保留全部输入；没有把重复片段当作独立论文。", "",
              f"冻结profile SHA-256：`{evaluation['profile_sha256']}`。旧v1文件保留；相同两篇论文的旧v1在1600/400下为人工0/19、生成0/12，混合生成字符覆盖0。若本轮选择其它片段长度，两者不能视为完全相同分片的对照。", "",
              "零观察误标不证明总体误报率为零。语料仍限于十篇天线论文、三种相近生成任务；原版标签含PDF到LaTeX转换的不确定性。生成段单独检测与包含邻近人工正文的混合检测分开报告；标记整个混合片段时，其内人工字符只是标记溢出。未标记不能排除AIGC。", "",
              "纯分数证据入口：`evaluations/english-academic-20261003-v2/`，包含campaign、三档train-scores、candidates、selection、FREEZE、repeat-scores及evaluation；都不含正文。原文、原始窗口缓存、日志与完整报告留在Git排除的samples/reports/.runtime。", "",
              "逐篇核对：[全部115个输入的标注报告](all-inputs/index.html)，按原版、混合文档和单独生成段列出，注明拟合/校准/重复评估集合。复用已核验原始窗口缓存，报告明确记录缓存来源和hash，未再次推理或修改参数。", "",
              "验证入口：[数值原样重建](reproduction.json)、[流式统计与旧缓存一致性](streaming-real-equivalence.json)、[真实标注/API/重组](runtime/validation.json)、[最终回归日志](pytest-final.log)、[实际部署](deployment.json)、[逐组CSV](summary.csv)。各文件按实际完成状态读取，历史运行日志不能代替本轮结果。", ""]
    (REPORTS / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence = ["# 英文扩展融合纯分数证据", "", f"配置：`{POLICY}`；schema3.0；全部原版/混合/生成段按论文绑定分组。",
                "[完整本机说明](../../reports/english-recalibration-20261003-v2/README.md)和[维护验证](../../dev_doc/context/english-recalibration-validation.md)记录实际结果、工程验证与限制。原始材料和完整报告不进入Git。", "",
                "- campaign.json：打分前声明的信号、候选范围、分组、完整样本利用与已知旧测试结果。",
                "- context400/context800/context1600：fit+calibration完整输入分片的纯分数与逐论文候选比较。",
                "- selection.json：仅用fit论文留出选择的最终候选及诊断。",
                "- FREEZE.json：profile、训练数据、候选选择及campaign哈希。",
                "- repeat-scores.json：冻结后P07/P08原版、六份混合版本与十二处单独生成段的打分。",
                "- evaluation.json：三组原版误标、单独生成段命中、混合生成块/字符覆盖与人工溢出。", "",
                "- engineering-validation.json：最终114项回归、真实部署/代码hash、115个完整报告及纯数值/UTF-8/链接检查摘要，不含正文。", "",
                "这是已知参考材料上的重校准及重复评估，不是新独立验证。所有运行时特征来自固定版本模型的原始输出，论文/作者/文件身份及长度不进入融合。", "",
                "参数复现不读取正文、不加载模型：", "", "```powershell", "& '.venv/Scripts/python.exe' -X utf8 scripts/calibrate_english_v2.py reproduce", "```", ""]
    (EVIDENCE / "README.md").write_text("\n".join(evidence), encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(REPORTS / "README.md"), "summary_csv": str(REPORTS / "summary.csv")}), flush=True)


if __name__ == "__main__":
    main()
