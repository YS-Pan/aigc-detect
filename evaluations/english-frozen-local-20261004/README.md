# 冻结模型来源与局部扫描实验

2026-10-04执行；底层模型和分类头均未训练，目标未达到，默认v2保持。完整解释见[验证记录](../../dev_doc/context/english-frozen-local-validation.md)。

## 数值文件

- `campaign.json`：12候选、作者组、上下文构造、人工门限、停止规则、模型revision及代码/来源hash，先于新评分。
- `inputs.json`、`input-lock.json`：6682局部窗口、3933条上下文、410原文块的ID/hash/位置；不含正文。
- `development-scores.json`、`selection.json`、`FREEZE.json`、`repeat-scores.json`、`evaluation.json`：开发评分、完整嵌套流程、冻结后旧P07/P08复测。
- `supplement/`：29块目视来源修正、四篇新人工来源元数据与471窗口；N01/N02人工校准，锁定`CALIBRATION.json`后评分N03/N04。新来源没有生成正例，不能给独立召回结论。
- `source-boundary-check/`：来源审计词锚函数漏末尾标点后的独立修正、同文去句末标点对照。来源结论使用此版本；早期源对照含提取干扰，保留追溯。
- `audit.json`、`diagnostics.json`、`inference-validation.json`：独立计数、门限隔离、逐生成器/长度范围、上下文配对、单条/批量真实推理对照。

嵌套作者组选择3/3103人工窗口误报、3/51混合段命中；选中方案旧P07/P08为0/473、0/12，新人工验收为1/172。其它方案不可在看过测试后替换为部署方案。原十篇已参与开发。

## 复核

在工作区根目录使用本工作区Python：

```powershell
& .venv/Scripts/python.exe scripts/english_frozen_local.py verify
& .venv/Scripts/python.exe scripts/english_frozen_controls.py verify
& .venv/Scripts/python.exe scripts/report_english_frozen_local.py audit
```

`prepare → score-development → freeze → score-repeat → evaluate`为主流程；新来源补充为`prepare → score-calibration → calibrate → score-test → evaluate`。已声明/冻结/评价文件拒绝覆盖，复现新运行应使用新的实验目录并先重新声明。`report`重建离线HTML，`validate-inference`进行六输入×四模型的真实本地复跑。

论文原文、公开XML快照、模型完整输出、审计页面和正文差异报告均保存在`reports/english-frozen-local-20261004/`，不加入Git。四份公开XML的固定URL、许可、作者和SHA-256在`supplement/inputs.json`；下载沿用`scripts/network.py`直连优先，只下载公开材料，没有发送待测文本。
