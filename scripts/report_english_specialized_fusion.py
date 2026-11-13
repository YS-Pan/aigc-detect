"""Audit both frozen specialized-model experiments; never choose new parameters."""
from collections import Counter
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import english_specialized_fusion as sf
from scripts import english_specialized_precision as sp
from scripts import english_hard_negative as hn
from aigc_detector.fusion import fused_score


def scalar_score(scores, model):
    features = []
    for signal, center, scale in zip(model["signals"], model["center"], model["scale"]):
        p = scores[signal]
        if model.get("transform") == "native_logit":
            raw = math.log(max(p, 1e-30)) - math.log(max(1-p, 1e-30))
        elif model.get("transform") == "raw_probability":
            raw = p
        else:
            p = max(1e-5, min(1-1e-5, p))
            raw = math.log(p/(1-p))
        bound = model.get("clip_standardized", 5.)
        features.append(max(-bound, min(bound, (raw-center)/scale)))
    return model["intercept"] + sum(a*b for a, b in zip(features, model["coefficients"]))


def audit(evidence):
    with patch.object(sf, "EVIDENCE", evidence), patch.object(hn, "predict", sp.predict):
        campaign = sf.verify_freeze()
        dev, repeat = sf.read(evidence / "development-data.json"), sf.read(evidence / "repeat-data.json")
        selection, frozen, evaluation = [sf.read(evidence / name) for name in ("selection.json", "frozen-models.json", "evaluation.json")]
        assert evaluation["repeat_data_sha256"] == sf.sha(evidence / "repeat-data.json")
        data = {k: dev[k] + repeat[k] for k in ("records", "blocks")}
        for dataset in (dev, repeat):
            for path, digest in dataset["config"]["cache_hashes"].items():
                assert sf.sha(ROOT / path) == digest
        all_papers = {r["paper_id"] for r in data["records"]}
        all_models = {"frozen_v2": sf.baseline(), **frozen["models"]}
        parity, sensitivity = {}, {}
        for name, item in all_models.items():
            model, threshold = item["model"], item["threshold"]
            values = sp.predict(data["records"], model)
            independent = np.array([scalar_score(r["scores"], model) for r in data["records"]])
            parity[name] = float(np.max(np.abs(values - independent)))
            assert parity[name] < 1e-10
            assert np.array_equal(values, sp.predict([{"scores": r["scores"]} for r in data["records"]], model))
            if name != "frozen_v2":
                assert set(model["fit_papers"]) == set(campaign["groups"]["fit"])
                assert hn.cutoff(dev["records"], model, campaign["threshold_margin"]) == threshold
            expected = evaluation["results"][name]
            assert hn.metrics(data, all_papers, model, threshold) == expected["all"]
            for split, papers in campaign["groups"].items():
                assert hn.metrics(data, set(papers), model, threshold) == expected["by_split"][split]
            for paper in all_papers:
                assert hn.metrics(data, {paper}, model, threshold) == expected["by_paper"][paper]
            max_human = max(v for r, v in zip(data["records"], values) if not r["ai_chars"])
            vmap = {r["id"]: float(v) for r, v in zip(data["records"], values)}
            bag_maxima = sorted((max(vmap[i] for i in b["window_ids"]) for b in data["blocks"]), reverse=True)
            target_threshold = bag_maxima[math.ceil(.5*len(bag_maxima))-1]
            sensitivity[name] = {
                "all_known_human_max_no_margin": hn.metrics(data, all_papers, model, np.nextafter(max_human, np.inf)),
                "threshold_for_50pct_known_mixed_blocks": float(target_threshold),
                "at_50pct_known_mixed_blocks": hn.metrics(data, all_papers, model, target_threshold),
            }
        profile = sf.read(sf.PROFILE)
        production = np.array([fused_score(r["scores"], profile)["review_logit"] for r in data["records"]])
        baseline_error = float(np.max(np.abs(production - sp.predict(data["records"], sf.baseline()["model"]))))
        assert baseline_error < 1e-10
        inner_checks, false_rows = 0, []
        for outer in selection["nested"]:
            held = outer["held_paper"]
            pool = set(campaign["groups"]["fit"]) - {held}
            for candidate in outer["inner_candidates"]:
                for inner in candidate["folds"]:
                    assert held != inner["held_paper"] and set(inner["fit_papers"]) == pool - {inner["held_paper"]}
                    inner_checks += 1
            assert set(outer["model"]["fit_papers"]) == pool
            controls = [r for r in dev["records"] if r["paper_id"] in pool]
            assert hn.cutoff(controls, outer["model"], campaign["threshold_margin"]) == outer["threshold"]
            assert hn.metrics(dev, {held}, outer["model"], outer["threshold"]) == outer["metrics"]
            humans = [r for r in dev["records"] if r["paper_id"] == held and not r["ai_chars"]]
            for row, value in zip(humans, sp.predict(humans, outer["model"])):
                if value >= outer["threshold"]:
                    false_rows.append({"id": row["id"], "paper_id": held, "sha256": row["sha256"], "kind": row["kind"]})
        saturation = {}
        for sig in sf.NEW + sf.CLASS_SIGNALS:
            saturation[sig] = {}
            for kind, rows in {"pure_human": [r for r in dev["records"] if not r["ai_chars"]],
                               "mixed_positive": [r for r in dev["records"] if r["kind"] == "mixed_document" and r["ai_chars"]]}.items():
                values = np.array([r["scores"][sig] for r in rows])
                saturation[sig][kind] = {"n": len(values), "below_1e_5": int((values < 1e-5).sum()),
                    "above_1_minus_1e_5": int((values > 1-1e-5).sum()), "quantiles": np.quantile(values, [0, .5, .9, 1]).tolist()}
        chosen = frozen["models"][frozen["overall_family"]]
        values = sp.predict(data["records"], chosen["model"])
        flags = {r["id"]: bool(v >= chosen["threshold"]) for r, v in zip(data["records"], values)}
        generators = {}
        for gen in sorted({b["generator"] for b in data["blocks"]}):
            blocks = [b for b in data["blocks"] if b["generator"] == gen]
            generators[gen] = {"mixed_blocks": len(blocks), "mixed_block_hits": sum(any(flags[i] for i in b["window_ids"]) for b in blocks)}
        result = {"frozen_sources_and_cache_hashes_verified": True, "scalar_parity_max_errors": parity,
            "baseline_production_max_logit_error": baseline_error, "nested_inner_fold_boundaries_checked": inner_checks,
            "outer_models_cutoffs_metrics_recomputed": len(selection["nested"]), "all_split_paper_metrics_recomputed": True,
            "nested_false_windows": false_rows, "nested_unique_false_hashes": len({r["sha256"] for r in false_rows}),
            "nested_false_by_paper": dict(Counter(r["paper_id"] for r in false_rows)),
            "posthoc_threshold_diagnostics": sensitivity, "development_raw_score_saturation": saturation,
            "selected_by_generator_all_material": generators, "retuning_performed": False, "contains_source_text": False}
        sf.write(evidence / "audit.json", result)
        return result


def brief(metrics):
    return (f"{metrics['pure_human_false_windows']}/{metrics['pure_human_windows']}",
            f"{metrics['mixed_block_hits']}/{metrics['mixed_blocks']}",
            f"{100*metrics['mixed_ai_hit_chars']/metrics['mixed_ai_chars']:.2f}%")


def report():
    roots = [("首轮：沿用旧logit截断", sf.EVIDENCE), ("精度对照：保留微小分数／直接概率", sp.EVIDENCE)]
    unique_count = sum(sf.read(sf.EVIDENCE / f"{split}-data.json")["config"]["unique_passages"] for split in ("development", "repeat"))
    lines = ["# 四个专业英文模型的融合重调（2026-10-04）", "",
        "已完成四模型全量离线打分、42候选融合及发现数值裁剪问题后的75候选精度对照。两轮均未达到完全人工窗口零误报且混合生成段召回50%的目标；默认英文v2保持。", "",
        f"四个新模型为FollowSci、RoBERTa Academic、LLM-DetectAIve、CoCoDet。沿用固定1600/400字符分片：10篇原版、30份混合文档及75处单独生成段，共115输入、678窗口。按正文hash去重后实际为{unique_count}个输入片段，每片分别运行四个新模型；旧九分类器的分数来自hash/revision一致的缓存。运行时阻断网络，未下载权重或训练底层模型。", "",
        "## 口径与设计", "",
        "完全人工窗口判正才算误报，包含混合文档中的纯人工窗口；任意生成字符使窗口成为正例。一个生成段与任意判正窗口有正长度交集即算命中，同窗人工字符不算误报。单独生成段检出单列。", "",
        "六篇拟合、两篇人工门限校准、P07/P08重复评估。按论文绑定原版及全部派生版本；同论文重复正文去重并保留生成段bag归属。拟合仅用原版和混合窗口，孤立生成段只用于评价。每个训练折重建标准化、系数和门限；校准正例不参与选择。最终门限只看拟合/校准纯人工最大分加固定余量。", "",
        "首轮42候选包括：四新模型聚合分、七个细分类别信号、分别与v2五模型组合、与全部旧九模型组合；每组比较普通存在判别与最难人工bag排序，各三档L2。另有v2重新拟合对照及九个单信号对照。", "",
        "初轮发现旧1e-5概率裁剪把开发组所有FollowSci分数及大部分CoCoDet类别分数压成常数。保留原结果后另行声明75候选，对比保留微小分数的native logit与直接概率输入；模型名单、损失、分组和门限规则保持。该对照已知道首轮复测结果，只是后续开发诊断。", "",
        "两轮都先冻结总选择及各类冠军，再进行该轮复测核算。全部十篇长期参与开发，均不是新独立测试。没有按复测结果回改系数或门限，也未新增可调用runtime policy。", "",
        "## 核心结果", "", "| 范围／配置 | 完全人工窗口误报 | 混合生成段命中 | 生成字符覆盖 |", "|---|---:|---:|---:|"]
    evaluations = []
    for title, evidence in roots:
        audit(evidence)
        evaluation = sf.read(evidence / "evaluation.json")
        evaluations.append(evaluation)
        for scope, metrics in [("嵌套留出", evaluation["nested_totals"]),
            ("冻结复测", evaluation["results"][evaluation["selected_family"]]["by_split"]["test"])]:
            lines.append(f"| {title}；{scope} | " + " | ".join(brief(metrics)) + " |")
    v2 = evaluations[0]["results"]["frozen_v2"]
    lines.append("| 现有v2；P07/P08复测 | " + " | ".join(brief(v2["by_split"]["test"])) + " |")
    lines += ["", "58个复测人工窗口由原版19个及混合文档中的39个纯人工窗口组成；它与旧报告只列原版的19个分母不同。", ""]
    for (title, evidence), evaluation in zip(roots, evaluations):
        selection, audit_result = sf.read(evidence / "selection.json"), sf.read(evidence / "audit.json")
        lines += [f"## {title}", "", f"按开发选择折冻结的总选择为 `{selection['selected']['configuration']['id']}`。选择折参与选参，其结果不是外层留出表现。",
            f"嵌套误报共有{len(audit_result['nested_false_windows'])}个窗口、{audit_result['nested_unique_false_hashes']}个不同正文hash；逐论文为{audit_result['nested_false_by_paper']}。", "",
            "下表是冻结前选定的各类冠军；全部材料列含拟合与校准，不能当作泛化指标。", "",
            "| 冻结配置 | 选择折：误报／命中 | P07/P08：误报／命中 | 全材料：误报／命中 | 全材料单独生成段命中 |",
            "|---|---:|---:|---:|---:|"]
        for family, champion in selection["family_champions"].items():
            r = evaluation["results"][family]
            a, b, c = champion["metrics"], r["by_split"]["test"], r["all"]
            cells = [f"{m['pure_human_false_windows']}/{m['pure_human_windows']}；{m['mixed_block_hits']}/{m['mixed_blocks']}" for m in (a,b,c)]
            lines.append(f"| `{family}` | " + " | ".join(cells) + f" | {c['isolated_block_hits']}/{c['isolated_blocks']} |")
        selected = evaluation["selected_family"]
        sensitivity = audit_result["posthoc_threshold_diagnostics"][selected]
        zero, target = sensitivity["all_known_human_max_no_margin"], sensitivity["at_50pct_known_mixed_blocks"]
        lines += ["", "事后门限诊断不参与选择或部署：对本轮总选择，使用全部已知人工最大分作为零误报界限时，"
            f"混合段命中{zero['mixed_block_hits']}/75；若降门限至至少38/75命中，则有{target['pure_human_false_windows']}/493人工窗口误报。", "",
            f"证据：[预声明](../../evaluations/{evidence.name}/campaign.json)、[选择与嵌套折](../../evaluations/{evidence.name}/selection.json)、[冻结参数](../../evaluations/{evidence.name}/frozen-models.json)、[逐篇结果](../../evaluations/{evidence.name}/evaluation.json)、[数值审计](../../evaluations/{evidence.name}/audit.json)。", ""]
    lines += ["## 验证与限制", "",
        "所有冠军从冻结分数重建，预测和门限误差为0；逐项重算内外层分组、门限、全材料/分组/逐篇指标。每个冻结函数均与独立标量公式核对，旧v2同时与生产fused_score核对。原始缓存的hash与revision一致，含类别输出的求和语义已校验。工程测试与真实推理对照另见engineering-validation.json。", "",
        "结论限于本轮1600/400上下文、预声明线性存在判别/最难人工排序及两种精度表示。新模型对少量开发生成段有响应，但没有稳定提高低误报下的检出；不能据此证明所有非线性、局部表示或底层适配都无效，也不能把零观察误报解释为未来保证。", "",
        "## 复现", "", "```powershell",
        "& '.venv/Scripts/python.exe' -X utf8 scripts/english_specialized_fusion.py verify",
        "& '.venv/Scripts/python.exe' -X utf8 scripts/english_specialized_precision.py verify",
        "& '.venv/Scripts/python.exe' -X utf8 scripts/report_english_specialized_fusion.py", "```", "",
        "prepare/fit/evaluate拒绝覆盖已声明、冻结或已评估结果。原始首次顺序：prepare → score-development → fit → score-repeat → evaluate → verify；精度对照：prepare → fit → repeat → evaluate → verify。正文/缓存/完整报告仅存Git排除目录；evaluations只保留数值、偏移、hash及来源标签。"]
    path = sf.REPORTS / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "campaigns_audited": 2}), flush=True)


if __name__ == "__main__":
    report()
