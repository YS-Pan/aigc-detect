# 公开来源样本检测：2026-09-30

## 结论

已在既有 `samples/` 统一入口增加 24 条公开、带来源标签的文本，并在本机完成十模型原始全文检测和默认 v2 分段标注。**默认 `thesis-cnki-20260929-v2` 将全部 37 段归为 `other`，包括全部 12 条 AIGC 的 16 段。** 七个单模型对全部 12 条 AIGC 给出全文原始分数 ≥0.90。默认参考融合在本批问答/指令/文学示例上没有筛出来源明确的 AIGC，不能将其“其他”解释为人写。

原始分数中，12 条人写有四条被至少一个模型给出 ≥0.90；这个计数只是单模型高分与来源标签的冲突，不是给出通用误报率。没有据此修改门限、融合参数或模型。

## 采集证据

来源与抽取规则的权威入口为[样本说明](../../samples/public-20260930/README.md)，原始记录为[samples.jsonl](../../samples/public-20260930/samples.jsonl)，批次配置为[collection.json](../../samples/public-20260930/collection.json)。

- HC3-Chinese `open_qa` 与 `psychology` 各四组同题人类/ChatGPT 回答，共 16 条。标签直接来自发布者字段，使用固定数据 revision。上游许可在作者表中分别标为 MIT 与 CC0，HC3 发布层为 CC BY-SA 4.0，保留来源及署名。
- BELLE `train_0.5M_CN` 自动生成输出四条。保留作者生成说明、数据卡、DATA_LICENSE 及研究使用限制。没有逐条推断生成模型版本。
- 魯迅《吶喊》四篇开头正文的连续节选。来源电子版于 2008 年发布，作者资料和正文与站点自动摘要区分；保留字形、空格及 CRLF，记录完整下载 hash 和原字符范围。

采集选择按来源顺序、长度和指定作品固定，先于检测。24 条文本去重通过，共 9,718 个 Python Unicode 字符。JSONL SHA-256：`0b47640f578081d7c6b6f52ceca9bfc13dbc9894b12ca93982f5edd4b030d6c5`。

采集使用已有 `network.py`：api.github.com 与 Gutenberg 直连成功；raw.githubusercontent.com 与 huggingface.co 直连失败后沿用当前环境代理回退。网络事件留在 `collection.json`；未改系统代理。推理只使用 `models/` 本地权重，未将文本发送到外部推理服务。

## 固定运行配置

- 启动时间：2026-09-30 21:45:22，Asia/Shanghai。
- 软件 0.4.0，工作区 `.venv/Scripts/python.exe`，设备 CUDA。
- 默认 policy：`thesis-cnki-20260929-v2`；完整十模型与 profile 锁定 revision 一致；400/120 字符；没有覆盖任何门限。
- profile SHA-256：`4de3dd7f07e2098f4797213da33fb7f0ffdbc141a4c98fbc57748f5f4aadddf9`。门限读取 profile，实际值保存在运行配置及报告中。
- 运行配置保存模型 manifest 哈希、相关源码哈希和输入集合哈希，可区分同版本号下的工作区源码变化。
- 每条分别运行 `Detector.detect` 与 `Detector.annotate`；前者是全文 token 覆盖加权分数，后者为独立分段分数，两者不要求相等。

实际命令：

```powershell
& '.\.venv\Scripts\python.exe' 'scripts/public_samples.py' collect
& '.\.venv\Scripts\python.exe' 'scripts/public_samples.py' detect --report-dir 'reports/public-samples-20260930-24'
```

复跑必须使用新目录，不能覆盖本轮归档。

## 实际结果

| 来源标签 | 文本 | 片段 | 强信号片段 | 仅疑似片段 | 其他片段 |
|---|---:|---:|---:|---:|---:|
| AIGC | 12 | 16 | 0 | 0 | 16 |
| 人写 | 12 | 21 | 0 | 0 | 21 |

单模型原始全文分数 ≥0.90 的条数如下。0.90 是原始分数展示参考线，不是 v2 的任何融合门限；下表的计数不是准确率。

| 模型 | 人写 ≥0.90 | AIGC ≥0.90 |
|---|---:|---:|
| mpu-zh-v3 | 1/12 | 12/12 |
| mpu-zh-v3-short | 1/12 | 12/12 |
| mpu-zh-v2 | 0/12 | 12/12 |
| hc3-zh | 0/12 | 12/12 |
| ensemjudge-bert | 0/12 | 7/12 |
| ensemjudge-roberta | 0/12 | 7/12 |
| anx-bert-zh | 0/12 | 12/12 |
| paperpass-v3 | 0/12 | 1/12 |
| cmj-zh | 0/12 | 12/12 |
| bibbbu-xlm-roberta | 3/12 | 12/12 |

高原始分数与低融合分档并存，是本次实际观测；本记录不把原因未经验证地归结为具体系数或实现错误。当前 v2 的参考拟合背景见[原验证说明](nonlinear-fusion-validation.md)。

## 归档与检查

[本机报告入口](../../reports/public-samples-20260930-24/README.md)提供来源标签/检测分档对照、十模型原始分数、中位数、逐条正文和 HTML 链接。`raw-detection.json`、`scores.csv`、`summary.json`、`run-config.json` 和逐条报告保存在同一新目录。顶层及每条报告均有 `COMPLETE.json`；完整报告由 Git 排除。轻量验证数据见[verification.json](../../reports/public-samples-20260930-24/verification.json)。

实际验证覆盖：24 条 JSONL/UTF-8 快照一致且无重复；上游证据快照 hash 一致；240 个全文模型结果无截断且分数有限；37 段完整覆盖正文且每段十模型；模型 revision 与 profile 一致；报告正文、输入 hash 和完成标记相符。推理时 patch 阻止 socket connect/create_connection，24 条全部完成。脚本可编译，相关本地文档链接可解析。未修改推理 API 或融合实现，因此没有重跑与本次无关的全套模型安装测试。

## 证据边界

HC3 与 `hc3-zh` 有已知训练重叠；其余检测器对 HC3、BELLE 或经典作品的训练重叠未经审计。来源字段不是逐条法证认证；样本非随机，数量少，文体、长度和繁简存在差异。它们不能证明一般化准确率，不能代表最新生成模型或独立整篇论文效果。

本批用于固定配置下的新来源观察，没有调参，不能据此宣称独立泛化已验证，也不替代[整篇论文/CNKI 位置验证待办](../toDo/nonlinear-fusion-validation.md)。`other` 不认证人写，任一模型高分不证明 AIGC，标记字符数不是实际 AI 写作比例。
