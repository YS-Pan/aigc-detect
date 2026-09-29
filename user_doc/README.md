# 调用方指南

本文主要面向按用户指令使用检测能力的 AI agent。按任务阅读，不需要先了解模型实现。

| 任务 | 阅读入口 |
|---|---|
| 在 dissertation 工作区调用 | [dissertation-integration.md](dissertation-integration.md) |
| 长文三档标注、保存结果和高亮查看 | [long-text-annotation.md](long-text-annotation.md) |
| 启动、停止、安装、离线验证 | [install-and-run.md](install-and-run.md) |
| HTTP 请求、Python SDK、字段解释 | [api.md](api.md) |
| 修改服务或增加模型 | [开发文档入口](../dev_doc/README.md) |

检测器接收正文字符串。文件读取入口支持 UTF-8 文本、Markdown 和 LaTeX 源码，内容原样参与检测。若需要排除参考文献、代码块、公式或标记，由调用方先提取正文并保留到原文件的位置映射。

长文标注默认使用 thesis-cnki-20260929-v1：全部十个模型按 profile 版本共同计算综合筛查分，片段为 400/120 字符。本版本提供已拟合 profile；部署包缺少该文件或模型 revision 不符时，默认策略不能运行。融合门限保存在 profile 和报告中。旧 sensitive-any-model-v1 可显式选用，默认原始门限为 0.90/0.70。长文规则、保存方式及边界见[长文指南](long-text-annotation.md)。

原始模型分数、综合筛查分和三档标签都是复核线索，不是“AI 文本占比”、来源证明或论文通过/不通过结论。跨工作区的参考数据范围和留出检验限制见[dissertation 集成说明](dissertation-integration.md)。
