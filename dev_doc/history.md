# 项目阶段

## 2026-09：本地中文检测服务

建立独立虚拟环境、四个中文检测模型、统一 Python/CLI/HTTP 接口，以及参考 yadof 分层方式的使用和开发文档。实际完成状态以 [验证记录](context/installation-validation.md) 和 [变更记录](change_records/README.md) 为准。

同月升级至 0.2.0：增加用户选择的敏感筛查分段标注、可复核的 JSON/JSONL/HTML 报告，完成 [十万字符验证](context/long-text-validation.md)。

2026-09-29：在同一环境新增六个分类器，总计十模型。完成 EnsemJudge 本地分类头适配、权重及 tokenizer 追溯、离线和接口检查，详见 [扩展验证](context/model-expansion-validation.md)。

同日升级至 0.3.0：三篇论文与 CNKI 显著标记参与纯分数融合拟合；完成结构化 CNKI 包及当前论文 checkpoint、无 GPU 重合并接口和本地 Git 初始化。指定参考集达到低人工误报与略高于 CNKI 同范围的标记量，章节留出暴露明显迁移限制，详见[校准验证](context/fusion-validation.md)。
