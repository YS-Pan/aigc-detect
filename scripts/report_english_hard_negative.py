"""Audit frozen fits and report the bounded experiment without retuning."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import english_hard_negative as h
from aigc_detector.fusion import fused_score

NAMES = {"frozen_v2": "现有冻结 v2", "presence": "普通存在判别", "mean_human_bag_ranking": "平均人工排序",
         "hard_human_bag_ranking": "最难人工排序", "overall": "内层选择完整流程"}


def audit():
    campaign = h.verify_freeze()
    selection = h.read(h.EVIDENCE / "selection.json")
    models = h.read(h.EVIDENCE / "frozen-models.json")["models"]
    dev = h.read(h.EVIDENCE / "development-data.json")
    repeated = h.read(h.EVIDENCE / "repeat-data.json")
    data = {k: dev[k] + repeated[k] for k in ("records", "blocks")}
    all_papers = {r["paper_id"] for r in data["records"]}
    controls = [r for r in dev["records"] if not r["ai_chars"]]
    profile = h.read(h.PROFILE)
    parity, sensitivity = {}, {}
    for name, item in models.items():
        model = item["model"]
        temporary = copy.deepcopy(profile)
        temporary.update(center=model["center"], scale=model["scale"])
        temporary["heads"] = {key: {"intercept": model["intercept"], "coefficients": model["coefficients"]} for key in ("strong", "review")}
        reference = np.array([fused_score(r["scores"], temporary)["review_logit"] for r in data["records"]])
        parity[name] = float(np.max(np.abs(reference-h.predict(data["records"], model))))
        assert parity[name] < 1e-10
        control_max = float(h.predict(controls, model).max())
        global_max = float(h.predict([r for r in data["records"] if not r["ai_chars"]], model).max())
        sensitivity[name] = {
            "without_extra_margin_fit_calibration_cutoff": h.metrics(data, all_papers, model, np.nextafter(control_max, np.inf)),
            "oracle_all_known_human_max": h.metrics(data, all_papers, model, np.nextafter(global_max, np.inf)),
            "fit_papers_only": h.metrics(data, set(campaign["groups"]["fit"]), model,
                                        np.nextafter(h.cutoff([r for r in dev["records"] if r["split"] == "fit"], model, 0.), np.inf)),
        }
    leakage, hard_false = [], []
    for outer in selection["nested"]:
        held = outer["held_paper"]
        outer_train = set(campaign["groups"]["fit"]) - {held}
        for candidate in outer["inner_candidates"]:
            for inner in candidate["folds"]:
                assert held not in inner["fit_papers"] and held != inner["held_paper"]
                assert set(inner["fit_papers"]) == outer_train - {inner["held_paper"]}
                leakage.append(True)
        for name, item in outer["methods"].items():
            assert set(item["model"]["fit_papers"]) == outer_train
            training = [r for r in dev["records"] if r["paper_id"] in outer_train]
            assert h.cutoff(training, item["model"], campaign["threshold_margin"]) == item["threshold"]
            assert h.metrics(dev, {held}, item["model"], item["threshold"]) == item["metrics"]
            if name == "hard_human_bag_ranking":
                humans = [r for r in dev["records"] if r["paper_id"] == held and not r["ai_chars"]]
                for row, value in zip(humans, h.predict(humans, item["model"])):
                    if value >= item["threshold"]:
                        hard_false.append({"id": row["id"], "paper_id": held, "sha256": row["sha256"], "kind": row["kind"]})
    # Separate descriptive sensitivity calculation; no model or cutoff is saved back.
    result = {"frozen_sources_verified": True, "production_score_parity_max_errors": parity,
              "nested_inner_fold_boundaries_checked": len(leakage), "outer_models_and_cutoffs_rechecked": len(selection["nested"])*4,
              "hard_nested_false_windows": hard_false, "hard_nested_unique_false_hashes": len({r["sha256"] for r in hard_false}),
              "threshold_sensitivity_post_hoc_only": sensitivity, "retuning_performed": False,
              "contains_source_text": False}
    h.write(h.EVIDENCE / "audit.json", result)
    return result


def ratio(n, d):
    return f"{n}/{d}"


def table(rows, scope):
    lines = ["| 方法 | 完全人工窗口误报 | 原版人工窗口误报 | 混合生成段命中 | 混合生成字符覆盖 |",
             "|---|---:|---:|---:|---:|"]
    for name, m in rows.items():
        lines.append(f"| {NAMES[name]} | {ratio(m['pure_human_false_windows'],m['pure_human_windows'])} | "
                     f"{ratio(m['original_false_windows'],m['original_windows'])} | {ratio(m['mixed_block_hits'],m['mixed_blocks'])} | "
                     f"{100*m['mixed_ai_hit_chars']/m['mixed_ai_chars']:.2f}% |")
    return f"## {scope}\n\n" + "\n".join(lines) + "\n"


def report():
    result = audit()
    campaign = h.read(h.EVIDENCE / "campaign.json")
    selection = h.read(h.EVIDENCE / "selection.json")
    evaluation = h.read(h.EVIDENCE / "evaluation.json")
    models = h.read(h.EVIDENCE / "frozen-models.json")["models"]
    lines = ["# 英文缓存融合：最难人工窗口排序实验（2026-10-04）", "",
             "已完成原复盘建议的第一步。**限定九个简单线性候选后，最难人工窗口目标没有改善检出，也没有通过嵌套留出的零误报要求；保持现有 v2 默认。**本轮停止继续扩充相同分数上的候选。结论限于此次五分数、线性函数及预声明平滑目标，不能解释成所有可能的融合都已被否定。", "",
             "口径：任何包含生成字符的窗口都是正例；完全人工窗口判正才是假阳性，包括混合文档中的完全人工窗口。每个混合生成段只要与至少一个判正窗口有正长度交集即算命中。正例窗口所含人工字符不计误报。", "",
             "全部资料仍是已知的十篇论文、30份混合版本、75处单独生成段，共115输入、678窗口。来源标签保持原样；底层模型推理、权重、默认profile和接口未改动。", "",
             "## 预声明与拟合", "",
             "- 固定现有 v2 五分类器的 revision 和1600/400自然边界分片，只使用原始分数。分类器logit、中心、尺度和裁剪均逐训练折重建。",
             "- 三种目标：普通窗口存在判别、平均人工排序、最难人工排序；各配三档L2（0.01/0.1/1），共9个候选。没有新增RBF或网络搜索。",
             "- 生成段的训练bag包含与该段相交的所有混合窗口。生成段按论文等权归一化；同论文重复窗口去重但保留全部bag归属。单独生成段只用于评价。最终每种函数用6篇论文的254个唯一窗口、51个混合段bag拟合。",
             "- 普通存在判别采用按论文/类别等权的BCE。排序函数没有截距，因为公共截距在分数差和人工控制门限中抵消。两种排序都以普通存在判别解和零向量为确定性起点，按训练损失选解。",
             "- 数值优化仅在CPU使用已有PyTorch float64和L-BFGS强Wolfe线搜索，未安装依赖。192次唯一拟合全部达到预设的梯度检查标准（小于1e-5）；有限收敛检查不构成全局最优证明。", "",
             "令窗口分数为 s，温度 tau=0.1，间隔 m=0.1。最难人工分的平滑上界 H=tau*logsumexp(s_h/tau)；每个生成段相交窗口最大分的平滑下界 B=tau*logmeanexp(s_w/tau)。新目标为按论文平衡的 mean_b softplus(m+H-B)，另加L2。平均人工对照将H改为对所有人工窗口分别计算损失后按论文平均；正例bag聚合相同。平滑版本不等于精确max，但给出了对应间隔损失的保守近似。", "",
             "## 分组与防泄漏", "",
             "沿用六篇拟合、两篇纯人工校准、P07/P08复测。候选先用拟合六篇逐论文留出比较；零人工误报是可行条件，若某目标所有候选均有误报，仅保留其最少误报者作失败诊断。三种目标最终均选L2=1；普通存在判别是九候选中唯一在这轮选择折合计零误报的候选，同时0/51混合段命中。",
             "另做嵌套留出：每次外层留一篇，内层只在其余五篇中重新选择，标准化、拟合、生成段bag和门限均排除外层论文。内层选择完整流程同样会在外层遇到4个误报，不能把选择折上的零误报当稳定泛化。两篇校准只在最终人工门限中使用，其生成标签不参与训练、标准化、选择或门限。",
             "门限固定为当折训练人工最大分加预声明余量；最终用拟合及校准纯人工最大分加余量。三种目标冠军、总体选择、源码和数据哈希先写入FREEZE，之后才读取P07/P08的分数作复测。所有十篇历史上已被观察，本轮仍不是新的独立测试。", ""]
    lines.append(table(selection["nested_totals"], "嵌套逐论文留出（六篇开发论文）"))
    lines += ["最难人工排序的4次误报都来自P10：1个原版窗口及其3份混合版本中的相同人工文本，只有1个不同正文hash。这仍违反零观察误报，但不能解释为4个独立来源失败。", ""]
    lines.append(table({k:v["by_split"]["test"] for k,v in evaluation["results"].items()}, "冻结后P07/P08复测"))
    lines += ["58个完全人工窗口包含原版19个与混合版本39个。现有v2命中正例窗内的815个人工字符不属于误报。", ""]
    lines.append(table({k:v["all"] for k,v in evaluation["results"].items()}, "全部十篇材料核算（含拟合与校准）"))
    lines += ["全材料核算不是留出效果。三种新目标在单独生成段上也均为0/75；现有v2为8/75。", "",
              "## 九候选的选择折结果", "", "| 候选 | 人工误报/390 | 混合生成段命中/51 |", "|---|---:|---:|"]
    for c in selection["candidates"]:
        lines.append(f"| {c['configuration']['id']} | {c['metrics']['pure_human_false_windows']} | {c['metrics']['mixed_block_hits']} |")
    lines += ["", "## 门限余量的事后诊断", "",
              "冻结后额外核算去掉门限余量是否能恢复检出，不据此调参或部署。以下第二列只使用拟合/校准人工控制；第三列是看过全部人工标签后的排序上界，仅为描述性诊断。", "",
              "| 方法 | 去余量：混合段命中/75；人工误报/493 | 全部已见人工最大分之上：命中/75 |", "|---|---:|---:|"]
    for name, value in result["threshold_sensitivity_post_hoc_only"].items():
        a, b = value["without_extra_margin_fit_calibration_cutoff"], value["oracle_all_known_human_max"]
        lines.append(f"| {NAMES[name]} | {a['mixed_block_hits']}/75；{a['pure_human_false_windows']}/493 | {b['mixed_block_hits']}/75 |")
    lines += ["", "## 验证与复现", "",
              "新实验8项测试覆盖有限差分梯度、优化下降、留出资料不影响拟合、推理仅使用分数、重复窗口bag归属、混合窗口口径及校准正例不影响门限。最终三种冠军从冻结分数重新拟合后，预测和门限原样重现（最大误差0）。",
              f"逐项审核270个内层折的来源隔离及24个外层模型/门限；现有v2在678窗口上与生产融合函数的最大logit误差为{evaluation['baseline_production_max_abs_logit_error']:.3g}，三种新函数也通过同一生产数值求值交叉验证。",
              "运行（PowerShell，项目根目录）：", "", "```powershell",
              "& '.venv/Scripts/python.exe' -X utf8 scripts/english_hard_negative.py verify",
              "& '.venv/Scripts/python.exe' -X utf8 scripts/report_english_hard_negative.py",
              "```", "",
              "本次原始执行顺序为prepare、fit、evaluate、verify。prepare/fit/evaluate拒绝覆盖已声明、冻结或已评价结果；不能覆盖旧证据重试候选。新实验应另开目录并单独声明。",
              "证据：[预声明](../../evaluations/english-hard-negative-20261004/campaign.json)、[选择及嵌套折](../../evaluations/english-hard-negative-20261004/selection.json)、[冻结模型](../../evaluations/english-hard-negative-20261004/frozen-models.json)、[冻结证明](../../evaluations/english-hard-negative-20261004/FREEZE.json)、[逐篇结果](../../evaluations/english-hard-negative-20261004/evaluation.json)、[数值与泄漏审核](../../evaluations/english-hard-negative-20261004/audit.json)。evaluations只含分数、位置和哈希；正文、运行日志及报告在Git排除目录。", "",
              "本轮达到的是完成一次可复现且范围受限的实验；未达到零误报下混合生成段检出50%的性能目标。后续事项仍为专业人工难例与匹配生成例、底层检测器适配和新来源验收，需按后续任务单独开展。"]
    path = h.REPORTS / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "audit": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    report()
