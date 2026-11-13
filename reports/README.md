# 验证产物

## 英文检测后续复盘

[2026-10-04本地核算](english-followup-review-20261004/README.md)：复现51个冻结冠军及v2，并作事后并集与单信号分布诊断。全部已知人工标签下重新划界的并集加v2为27/75混合段命中，P07/P08仍2/12；仅为已见材料诊断，不是新策略或独立验证。[重评分析](../dev_doc/context/english-detection-followup-review-20261004.md)和[纯数值证据](../evaluations/english-followup-review-20261004/README.md)保留解释与哈希。

## 专业英文四模型融合重调

[2026-10-04完整报告](english-specialized-fusion-20261004/README.md)：四新模型全量打分、42候选及75候选微小分数精度对照。两轮开发选择优胜者P07/P08均0/12混合段命中，精度轮嵌套24/390人工误报、8/51混合段命中，未达到目标，默认保持v2。[首轮数值证据](../evaluations/english-specialized-fusion-20261004/README.md)、[精度对照](../evaluations/english-specialized-precision-20261004/README.md)和[维护说明](../dev_doc/context/english-specialized-fusion-validation.md)可追溯。29项相关测试、29文件哈希及16次单条推理对照通过，旧材料不是独立测试。

## 英文最难人工窗口缓存实验

[2026-10-04报告](english-hard-negative-20261004/README.md)：执行复盘第一步，预声明9个线性候选，比较存在判别、平均人工排序和最难人工排序，完成嵌套论文留出、冻结复测和数值审核。最难人工排序嵌套4/390纯人工误报、0/51混合段命中；P07/P08复测0/58、0/12，低于v2的2/12。没有部署，默认保留v2；[纯数值证据](../evaluations/english-hard-negative-20261004/README.md)和[维护说明](../dev_doc/context/english-hard-negative-validation.md)可追溯。8项针对性测试通过，未重跑历史122项回归或底层推理。

## 补充英文模型与大模型统计

[2026-10-03本机安装入口](english-additional-20261003/README.md)：五个补充英文分类器、GPT-Neo 2.7B与Falcon 7B双模型，共新增约44GB。21资源哈希、100项回归、公开样本离线推理、显存/内存与实际API/CLI检查完成。[纯安装摘要](../evaluations/english-additional-20261003-installation/README.md)可版本管理；来源与边界见[验证说明](../dev_doc/context/english-additional-validation.md)。没有重拟合现有融合，也没有独立效果验证。

## 可选英文模型与来源样本

[2026-10-02 英文逐条分数](english-expansion-20261002/english-test/scores.csv)、[完整摘要](english-expansion-20261002/english-test/summary.json)、[中文真实回归](english-expansion-20261002/chinese-regression.json)及[实际本机部署](english-expansion-20261002/deployment-test.json)。英文20条输入包含12条来源human和8条ChatGPT，四模型全文及分段均已完成；bibbbu对全部人工材料给出≥0.90的全文分数。样本与许可见[批次说明](../samples/english-public-20261002/README.md)，检查和训练重叠限制见[验证说明](../dev_doc/context/english-expansion-validation.md)。分数与标签不能当作来源证明，这批不是独立准确率基准。

## 作者提供的本地论坛材料

[2026-09-30 图文检测汇总](kechuang-human-20260930-v2/index.html)及[详细说明](kechuang-human-20260930-v2/README.md)：14 个主题、159 条本人主帖/回复、72,525 个输入字符。来源标签按作者声明固定为 human；十模型原始检测与默认 v2 的 198 段标注已完成，0 段强信号、1 段仅疑似（357 字符）、197 段其他。按作者提供的标签，该疑似标记是一处误标观察，不能推算一般化误报率。[提取后的本地阅读入口](../samples/kechuang-human-20260930/index.html)与完整报告均由 Git 排除；[轻量验证说明](../dev_doc/context/author-forum-validation.md)记录实际检查和边界。

## 公开来源样本检测

[2026-09-30 报告入口](public-samples-20260930-24/README.md)：24 条来源明确的公开样本（人写/AIGC 各 12），完成十模型原始检测和默认 v2 分段标注。默认融合将全部片段归为其他，包括来源明确的 AIGC；单模型分数与来源标签的对照保留在报告中。样本与许可见[统一入口](../samples/README.md)，实际结果与证据边界见[公开样本验证说明](../dev_doc/context/public-samples-validation.md)。这批不是独立准确率基准。

## 当前论文参考校准

[本机报告入口](fusion-20260929/README.md)包含三篇论文的最终 JSON/JSONL/HTML、原始分数、CNKI 映射和实际验证。带正文的结果由 Git 排除；仓库内保存[纯分数证据](../evaluations/thesis-cnki-20260929/README.md)及[验证说明](../dev_doc/context/fusion-validation.md)。

## 历史工程验证

下列运行产物是对应阶段的历史证据；其中标注接口记录属于 0.2.0 与旧策略，不代表 0.3.0 默认融合 profile 的独立泛化效果。不要把本索引中的历史测试项数当作新策略验证结果。

- [smoke-test.json](smoke-test.json)：四模型、八样本的原始分数、窗口、设备及离线/一致性检查。
- [pytest.txt](pytest.txt)：输入契约和窗口边界检查输出。
- [deployment-test.json](deployment-test.json)：从调用方工作目录运行实际 HTTP 和 CLI 的结果。
- [openapi.json](openapi.json)：当次验证的历史接口快照；当前接口定义以运行中服务 `/openapi.json` 为准。
- [installation.json](installation.json)：环境、模型资源与服务交付状态快照。
- [annotation-validation.json](annotation-validation.json)：十万字符标注的耗时、完整覆盖和批处理一致性。
- [annotation-deployment.json](annotation-deployment.json)：0.2.0 标注接口与 CLI 的实际调用。
- [pytest-annotation.txt](pytest-annotation.txt)：包含三档标注和保存契约的 25 项测试。
- [十万字符示例报告](annotation-100k-20260928-223540/report.html)：公开样本派生的演示，配套 JSON、JSONL 和原文位于同目录。

这些文件记录安装验证，不是论文检测报告。来源说明见 [samples](../samples/README.md)，结论和边界见 [验证记录](../dev_doc/context/installation-validation.md)。重复执行验证脚本会覆盖对应的最新结果；需要保存历史时，先为旧结果另建带日期的副本。

长文融合当前默认 profile 为 `thesis-cnki-20260929-v2`，配置位于 [profile](../aigc_detector/profiles/thesis-cnki-20260929-v2.json)，算法与拟合限制见[融合模块说明](../dev_doc/blueprints/fusion.md)。既有报告按各自记录的 policy 理解。报告目录中的运行结果、原始正文、CNKI PDF 和其他具体报告均不入 Git；该目录只跟踪本 README。

## 英文低误报参考校准

[2026-10-03本地结果入口](english-calibration-20261003/README.md)及[真实接口报告](english-calibration-20261003-runtime-02/human-P07/report.html)：10篇来源、30份生成版本、75处生成段；冻结测试0/19人工误报，12处生成段全部漏检。[纯分数证据](../evaluations/english-academic-20261003-v1/README.md)可版本管理，样本和完整报告均Git排除。该结果没有验证提高独立检出率。

## 英文扩展重新校准

[完整本机报告](english-recalibration-20261003-v2/README.md)核算10原版/30混合/75生成段、678片段；全原版0/149误标，单独生成段合计8/75命中、混合生成块2/75命中。拟合及校准已参与适配，合计不是独立效果指标。P07/P08重复评估相同1600/400下人工0/19、生成3/12（旧v1为0/12），混合生成字符覆盖20.38%，仍有严重漏检。默认英文v2选五分类器；114项回归、实际HTTP/跨cwd CLI及数值参数重建通过。可跟踪[纯分数证据](../evaluations/english-academic-20261003-v2/README.md)，方法见[维护记录](../dev_doc/context/english-recalibration-validation.md)。

[逐篇标注索引](english-recalibration-20261003-v2/all-inputs/index.html)含全部115个输入的完整报告包，复用本轮已核验窗口分数，记录来源hash，不重复推理或重新拟合。

## 英文核融合与神经网络实验

[2026-10-04本机说明](english-nonlinear-20261004-v3/README.md)记录648个候选、Newton与Adam/L-BFGS优化、[全部115输入报告](english-nonlinear-20261004-v3/all-inputs/index.html)和122项回归。冻结v3为新五分类器/1600/400/4隐藏单元/29参数；P07/P08复测人工0/19误标、单独生成1/12命中、混合生成字符覆盖0，低于线性v2，故默认不变。全材料3/75单独生成段命中、0/75混合生成块命中，包含拟合材料，不能当作独立效果指标。[纯分数证据](../evaluations/english-academic-20261004-v3/README.md)可跟踪；[维护说明](../dev_doc/context/english-nonlinear-validation.md)记录方法、失败及限制。

[english-diverse-20261006/report/index.html](english-diverse-20261006/report/index.html)：30原版及147混合版本的完整正文阅读、精确生成标签与v2/冻结选中方案标记。结果未达目标；正文、源映射、原始缓存与标准CLI报告均仅本地保存。数值复现见[对应evaluations](../evaluations/english-diverse-20261006/README.md)。
