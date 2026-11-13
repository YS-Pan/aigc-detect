"""Describe the frozen finite experiment, including unsuccessful candidates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.calibrate_english_nonlinear import EVIDENCE, REPORTS, POLICY, PROFILE, objective
from scripts.english_academic_samples import read_json, digest


def pct(value):
    return f"{100*value:.2f}%" if value is not None else "—"


def main():
    evaluation = read_json(EVIDENCE / "evaluation.json")
    selection = read_json(EVIDENCE / "selection.json")
    campaign, profile = read_json(EVIDENCE / "campaign.json"), read_json(PROFILE)
    assert digest(PROFILE.read_bytes()) == evaluation["profile_sha256"]
    chosen, config = selection["selected"], selection["selected"]["config"]
    lines = ["# 英文非线性融合实验", "",
             f"已完成{campaign['candidate_count']}个候选的拟合论文逐篇留出比较。最终冻结 `{POLICY}`：{config['family']}、{len(profile['models'])}个原始输出、{profile['segment_chars']}/{profile['min_segment_chars']}字符分片。", "",
             "全部10篇原版、30份混合文档及75处单独生成段均核算。论文及其派生版本绑定分组：6篇拟合、2篇纯人工门限控制、2篇重复评估；相同正文拟合去重，冲突标签及跨论文重复排除。论文身份、文本和长度不进入运行时融合。", "",
             "先按拟合论文留出人工误标字符最少选择，再比较混合正文中的生成字符检出，之后比较全部生成字符检出。最终参数仅拟合6篇论文，门限只读取拟合/校准组纯人工分数。P07/P08历史结果已知，本轮冻结后的复测不是新的独立验证；校准组生成分数及复测分数未用于参数/候选/门限。", "",
             "## 非线性与优化", "",
             "RBF：在线性项上加入16或48个高斯核，核中心由拟合原始分数的确定性聚类获得；宽度、L2和人工权重在预声明范围内比较。核系数由带精确Hessian和Armijo回溯的阻尼Newton法优化，核中心并未与系数联合优化。", "",
             "神经网络：4或12个tanh隐藏单元，输入权重、偏置、输出权重及截距联合优化；每个候选/论文折从两个固定种子起步，Adam 160步后用带strong-Wolfe线搜索的L-BFGS最多60步，按拟合损失选择起点。使用float64 CPU，记录损失和末次梯度；不宣称非凸问题达到全局最优。", "",
             "目标比较生成字符比例、存在生成字符；神经网络另比较带排序损失的存在目标。三个分片长度、5/9/12信号名单，共648候选、每候选6折。原始分数先按类型转换，训练加权标准化后裁到±5；人写权重和标签只用于训练。两个输出头共用同一函数，使用不同控制门限，sigmoid输出为综合筛查分。", "",
             f"最终L2={config['regularization']}、人工权重={config['human_weight']}、目标={config['target_mode']}、可学习参数{chosen['parameter_count']}。信号：" + "、".join(f"`{m}`" for m in profile["models"]) + "。", "",
             "## 逐论文候选比较", "",
             "以下每行按相同选择规则取该类/长度中的最佳候选；这些材料已经用于选择，不能称为独立准确率。", "",
             "| 分片 | 非线性 | 信号数 | 人工误标字符 | 混合生成字符命中 | 全部生成字符命中 |", "|---|---|---:|---:|---:|---:|"]
    for length, minimum in campaign["segmentations"]:
        for family in ("rbf", "tanh_mlp"):
            c = min((c for c in selection["candidates"] if c["config"]["segment_chars"] == length and c["config"]["family"] == family), key=objective)
            lines.append(f"| {length}/{minimum} | {family} | {len(c['config']['signals'])} | {c['human_flagged_characters']} | {c['mixed_ai_flagged_characters']}/{c['mixed_ai_characters']}（{pct(c['mixed_ai_flagged_characters']/c['mixed_ai_characters'])}） | {c['ai_flagged_characters']}/{c['ai_characters']} |")
    alternatives = sorted(selection["candidates"], key=lambda c: (-c["mixed_ai_flagged_characters"], c["human_flagged_characters"]))[:3]
    lines += ["", "检出最多的三个候选及其误标也保留，用于观察取舍，未按复测结果重新选择：", "",
              "| 分片 | 非线性 | 人工误标字符 | 混合生成字符命中 |", "|---|---|---:|---:|"]
    for c in alternatives:
        lines.append(f"| {c['config']['segment_chars']} | {c['config']['family']} | {c['human_flagged_characters']} | {c['mixed_ai_flagged_characters']}/{c['mixed_ai_characters']} |")
    lines += ["", "## 全部输入核算与冻结复测", "",
              "| 集合 | 原版 | 混合 | 单独生成段 | 标注片段 | 原版有标记 | 单独生成段命中 | 混合生成块命中 | 混合生成字符覆盖 | 混合纯人工误标片段 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"fit": "拟合", "calibration": "门限控制", "test": "重复评估"}
    for split, g in evaluation["groups"].items():
        n, p, m, d = g["source_inputs"], g["pure_inputs"]["any_flag"], g["mixed_documents"]["any_flag"], g["original_documents"]["any_flag"]
        lines.append(f"| {names[split]} | {n['human_document']} | {n['mixed_document']} | {n['ai_paragraph']} | {g['segments']} | {d['documents_with_any_flag']}/{d['documents']} | {p['ai_paragraphs_with_any_flag']}/{p['ai_paragraphs']} | {m['generated_blocks_with_any_flag']}/{m['generated_blocks']} | {pct(m['ai_character_recall'])} | {m['pure_human_flagged_segments']}/{m['pure_human_segments']} |")
    groups = list(evaluation["groups"].values())
    hits = sum(g["pure_inputs"]["any_flag"]["ai_paragraphs_with_any_flag"] for g in groups)
    blocks = sum(g["mixed_documents"]["any_flag"]["generated_blocks_with_any_flag"] for g in groups)
    ai = sum(g["mixed_documents"]["any_flag"]["ai_characters"] for g in groups)
    found = sum(g["mixed_documents"]["any_flag"]["ai_flagged_characters"] for g in groups)
    human_flags = sum(g["original_documents"]["any_flag"]["documents_with_any_flag"] for g in groups)
    lines += ["", f"全材料合计：原版{human_flags}/10有标记，单独生成段{hits}/75命中，混合生成块{blocks}/75命中，混合生成字符覆盖{pct(found/ai)}。合计混合拟合/控制/复测证据性质，不能当作独立效果指标。", "",
              "| P07/P08复测指标 | 上轮线性v2 | 本轮非线性v3 |", "|---|---:|---:|"]
    a, b = evaluation["groups"]["test"], evaluation["v2_baseline"]["test"]
    for label, keys in [("原版误标片段", ("pure_inputs", "human_flagged_segments")),
                        ("单独生成段命中", ("pure_inputs", "ai_paragraphs_with_any_flag")),
                        ("混合生成块命中", ("mixed_documents", "generated_blocks_with_any_flag")),
                        ("混合纯人工误标片段", ("mixed_documents", "pure_human_flagged_segments")),
                        ("标记混合片段内的人工字符", ("mixed_documents", "human_characters_within_flagged_segments"))]:
        lines.append(f"| {label} | {b[keys[0]]['any_flag'][keys[1]]} | {a[keys[0]]['any_flag'][keys[1]]} |")
    lines.append(f"| 混合生成字符覆盖 | {pct(b['mixed_documents']['any_flag']['ai_character_recall'])} | {pct(a['mixed_documents']['any_flag']['ai_character_recall'])} |")
    lines += ["", "上轮为1600/400；如本轮分片不同，片段误标数量的分母也不同，文档/真实生成位置指标用于总体比较。原版误标和混合纯人工误标与标记片段中的人工字符溢出分开核算。", "",
              "预声明的默认替换条件：拟合论文留出人工误标为零；复测原版及混合纯人工误标为零；单独生成段命中不少于v2，混合生成字符覆盖高于v2。",
              f"本轮条件{'满足' if evaluation['promotion_rule_satisfied'] else '未满足'}。默认状态见[工程交付证据](../../evaluations/english-academic-20261004-v3/engineering-validation.json)；本轮默认保留线性v2，不根据复测继续修改已冻结参数。", "",
              "十篇天线论文及相近生成任务不足以验证泛化。原版PDF到LaTeX转换也有标签不确定性。零观察误标不证明总体误报为零；未标记不能排除AIGC。", "",
              f"profile SHA-256：`{evaluation['profile_sha256']}`。旧v1/v2文件保持原hash。核/隐藏单元及每模型有符号贡献用于精确记账，不能解释为因果重要性。", "",
              "[全部115个输入标注报告](all-inputs/index.html)、[数值复现](reproduction.json)、[真实离线接口验证](runtime/validation.json)、[工程交付证据](../../evaluations/english-academic-20261004-v3/engineering-validation.json)。完整正文与缓存在Git排除的reports/samples中；公开跟踪的evaluations只有分数、标签、位置及哈希。", ""]
    (REPORTS / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (EVIDENCE / "README.md").write_text("\n".join([
        "# 英文非线性融合纯分数证据", "", f"`{POLICY}`，schema4.0；648个候选的固定范围、分组、数值和哈希均可追溯。",
        "[本机完整实验说明](../../reports/english-nonlinear-20261004-v3/README.md)与[维护验证](../../dev_doc/context/english-nonlinear-validation.md)。",
        "", "- campaign.json：复用分数前声明的候选、优化算法、分组和默认替换条件。",
        "- train-scores.json：选择分片的旧fit/calibration原始分数，字节与声明来源相同，不含正文。",
        "- selection.json：648候选、各论文折的误标/检出及优化诊断。",
        "- FREEZE.json：profile、数值、campaign、selection及拟合源码哈希。",
        "- repeat-scores.json：冻结后仅对P07/P08完整派生输入评分，记录本地缓存和网络阻断。",
        "- evaluation.json：三组完整核算、旧v2对照及预声明替换条件。",
        "- engineering-validation.json：实际运行验证和最终默认状态。", "",
        "参数与门限复现不读取正文、不加载模型、不使用复测：", "", "```powershell",
        "& '.venv/Scripts/python.exe' -X utf8 scripts/calibrate_english_nonlinear.py reproduce", "```", "",
        "P07/P08历史结果已知，复测不是新的独立验证；拟合/CV结果不能称为准确率。", ""
    ]), encoding="utf-8", newline="\n")
    print(REPORTS / "README.md", flush=True)


if __name__ == "__main__":
    main()
