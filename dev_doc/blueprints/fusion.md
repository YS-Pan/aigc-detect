# 十模型融合模块

实现：[fusion.py](../../aigc_detector/fusion.py)、[profile](../../aigc_detector/profiles/thesis-cnki-20260929-v1.json)、[拟合脚本](../../scripts/fit_fusion.py)。长文入口见 [annotation.md](annotation.md)，用户契约见[长文指南](../../user_doc/long-text-annotation.md)和[跨工作区说明](../../user_doc/dissertation-integration.md)。

## 运行时策略

默认策略为 `thesis-cnki-20260929-v1`。profile 固定十模型 roster、每个模型的 revision、标准化参数、系数、截距、400/120 字符分片和三档门限。模型集合、分片尺寸或 revision 与 profile 不一致时拒绝融合。

对每个原始分数 `p`，先裁到 profile 的概率范围，再计算 logit，减中心、除尺度；拟合尺度的下限为 0.5，标准化结果裁到 [-5, 5]。随后计算：

`ensemble_score = sigmoid(intercept + Σ(coefficient × standardized_logit))`

系数允许为正或负。输出保留原始 `ai_score`、原始档位、`ensemble_score`、`ensemble_logit`、各模型的标准化值、系数与有符号贡献。新策略下单模型原始档位仍按 0.90/0.70 记录；综合标签只由 profile 中的高分线和复核线决定。profile 门限不在其他文档中重复写数值。

融合分是当前参考材料上的筛查分，不是概率校准、作者身份判定或来源证明。报告 policy 记录 `calibrated=false`；profile 的 `fit` 记录 `probability_calibrated=false` 和 `independent_generalization_established=false`。

旧策略 `sensitive-any-model-v1` 保留任一模型达到门限即触发的敏感规则，原始分数门限默认为 0.90/0.70。它可选模型子集和自定义分片参数；这与默认融合策略的分数不能混为一谈。

## 拟合方法与约束

`fit_fusion.py` 可用 `--regularization`、`--human-control-weight` 和 `--human-limit-percent` 控制正则强度、人写控制样本权重和人工标记上限。

`scripts/fit_fusion.py` 读取模型分数及参考片段比例，不读取正文。数据须包含本人论文、wz 与 zyl 三份材料。目标值是 CNKI 显著片段在对应正文范围内的比例；用户确认的人写控制目标为零。profile 使用 L2 正则化 logit stacking。此次选择使用 0.1 的 L2 正则强度与 4 倍人工控制样本权重，并对人写控制设置严格零标记上限。人工权重只作用于拟合目标和参数估计，运行时没有作者、文本或文档身份特征。

高分线和复核线由脚本根据参考数据、CNKI 显著标记范围及人写控制约束选择，随 profile 保存。profile 和诊断记录数据 SHA-256、数据来源范围、模型 revision、超参数选择过程和限制。更新 profile 时应同时保存实际拟合证据，并说明哪些材料参与了参数与门限选择。

## CNKI 导入与范围

`scripts/import_cnki.py` 从本地 PDF 生成页面文本、版面坐标、结构化摘要与逐片段记录，不会提交文本到检测服务。当前结构化导入保留 5 份 PDF 的逐页内容，共导入 15 个 AIGC 标记片段（2 个显著、13 个疑似）；每个片段保留页码和坐标来源。

CNKI 原报告为 1.2%（1206/97762）；本人论文本地连续正文范围为 61,655 个 Unicode 字符。来源统计范围和字符定义不同，不能直接比较两个分母。标注为 2026-09-28 的提交 PDF 未提供，因此无法确认它与当前 checkpoint 或正文快照完全一致。

## 留出诊断边界

拟合样本上的分数和门限属于参考集内结果。脚本还按完整人写文档和 CNKI 阳性章节分组，把各组留出后重拟合标准化参数、系数和门限，以检查跨文档或跨章节的转移。

本轮超参数是在三份参考材料上比较后选定，所以这些留出重拟合结果仍受已看过参考材料的选择影响。它们不能称为真正独立、未接触的测试，更不能据此报告检测器准确率或“已验证泛化”。如要做独立检验，需要在参数选择前预留完全未参与拟合的文档和 CNKI 报告。

## 报告重组

长文报告 schema 为 `aigc-annotation/1.1`。报告保存 policy/profile、profile 哈希、模型 revisions、所有原始分数和窗口、融合分数及有符号贡献。

CLI `recombine` 先核对完整报告的 `COMPLETE.json`、`report.json` 与 `source.txt`，再从缓存的原始分数重新应用 policy。默认融合模式还核对 roster、profile revision 和片段尺寸。重组不加载模型、不使用 GPU，结果写到新的报告目录；已有目录不会覆盖。HTTP 没有重组端点。

[2026-10-06多来源实验](../context/english-diverse-validation-20261006.md)仅比较现有v2的人写门限校准和一次固定五分数线性存在判别重拟合（L2=10、人写权重20、1e-12 logit裁剪）；作者组内归一化和标签只用于维护期。新六来源冻结验收分别1/112、0/112，低于原样v2的5/112，未注册运行profile或改变默认。配置/门限以[evaluations](../../evaluations/english-diverse-20261006/README.md)为准，不进入实际调用参数。
