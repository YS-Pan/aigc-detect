# 验证产物

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

长文融合默认 profile 为 `thesis-cnki-20260929-v1`，配置位于 [profile](../aigc_detector/profiles/thesis-cnki-20260929-v1.json)，算法与拟合限制见[融合模块说明](../dev_doc/blueprints/fusion.md)。报告目录中的运行结果、原始正文、CNKI PDF 和其他具体报告均不入 Git；该目录只跟踪本 README。

[english-diverse-20261006/report/index.html](english-diverse-20261006/report/index.html)：30原版及147混合版本的完整正文阅读、精确生成标签与v2/冻结选中方案标记。结果未达目标；正文、源映射、原始缓存与标准CLI报告均仅本地保存。数值复现见[对应evaluations](../evaluations/english-diverse-20261006/README.md)。
