# 调用方指南

本文主要面向按用户指令使用检测能力的 AI agent。按任务阅读，不需要先了解模型实现。

其它工作区的常规调用优先使用自包含的 [aigc-detect skill](../.agents/skills/aigc-detect/SKILL.md)，正常检测、报告保存与重组无需先阅读这里的文档。下表用于特殊情况或需要更多细节时定点查回；调用方仍负责自己的正文范围、提取和报告归位。

| 任务 | 阅读入口 |
|---|---|
| 其它工作区 agent 常规调用 | [aigc-detect skill](../.agents/skills/aigc-detect/SKILL.md) |
| 在 dissertation 工作区调用 | [dissertation-integration.md](dissertation-integration.md) |
| 长文三档标注、保存结果和高亮查看 | [long-text-annotation.md](long-text-annotation.md) |
| 启动、停止、安装、离线验证 | [install-and-run.md](install-and-run.md) |
| HTTP 请求、Python SDK、字段解释 | [api.md](api.md) |
| 显式选择英文模型、英文长文及大模型统计方法 | [english-detection.md](english-detection.md) |
| 查看公开样本及作者明确要求归档的本地人写样本、来源标签和实际检测结果 | [样本入口](../samples/README.md) |
| 修改服务或增加模型 | [开发文档入口](../dev_doc/README.md) |

检测器接收正文字符串。文件读取入口支持 UTF-8 文本、Markdown 和 LaTeX 源码，内容原样参与检测。若需要排除参考文献、代码块、公式或标记，由调用方先提取正文并保留到原文件的位置映射。

中文长文标注默认使用 `thesis-cnki-20260929-v2`：全部十个模型按 profile 版本共同计算强信号分和复核分，片段为 400/120 字符。部署包缺少 profile 或模型 revision 不符时，默认策略不能运行。强信号门限作用于 `ensemble_score`；未达到该门限时，再按独立的 `review_score` 门限决定是否列为仅疑似。两个门限分别从 profile 读取，不要求数值大小有序。旧 `thesis-cnki-20260929-v1` 和 `sensitive-any-model-v1` 可显式选用；后者默认原始门限为 0.90/0.70。长文规则、保存方式及边界见[长文指南](long-text-annotation.md)。

原始模型分数、两条融合筛查分和三档标签都是复核线索，不是“AI 文本占比”、来源证明或论文通过/不通过结论。v2 的参考数据范围、留出诊断和适用限制见[dissertation 集成说明](dissertation-integration.md)及[非线性融合验证说明](../dev_doc/context/nonlinear-fusion-validation.md)。

英文 `language=en` 的标注默认选择 `english-academic-20261003-v2`：新增五分类器全选、1600/400字符分片及固定revision。完整10原版/30混合/75生成段已核算；重复评估人工0/19误标、生成3/12命中、混合生成字符覆盖20.38%，仍有严重漏检，未标记不能排除AIGC。英文原始detect仍默认旧四模型；执行及限制见[英文指南](english-detection.md)。

0.8.0新增显式非线性实验v3：648个RBF/网络候选及Newton/Adam/L-BFGS优化完成，但冻结复测仅1/12生成段命中、混合覆盖0，故默认仍为线性v2。方法和完整材料核算见[实验验证](../dev_doc/context/english-nonlinear-validation.md)。

0.9.0已安装四个专业英文／人机协作分类器，支持按ID显式检测并保留多分类输出。用法见[英文指南](english-detection.md)；默认英文detect与v2标注名单保持。
