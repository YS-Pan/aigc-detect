# 最难人工窗口排序的纯数值证据

这是2026-10-04在已知十篇论文缓存上完成的9候选实验，不是新的独立验证。最难人工排序在六篇嵌套留出中4/390完全人工窗口误报、0/51混合生成段命中；冻结P07/P08复测0/58误报、0/12命中，未改善现有v2的2/12，默认保留v2。

| 文件 | 职责 |
|---|---|
| [campaign.json](campaign.json) | 在拟合前声明候选、损失、优化、分组、阈值和停止条件；锁定源码、缓存、位置映射、revision及profile哈希 |
| [development-data.json](development-data.json) | 拟合/校准八篇的590窗口数值及63个混合生成段的窗口归属；校准生成标签只供评价 |
| [selection.json](selection.json) | 9候选选择折、嵌套论文留出、三类冠军、192次唯一拟合的收敛摘要 |
| [frozen-models.json](frozen-models.json) | 三类冠军的标准化、线性系数、训练元数据与人工门限；不是已部署profile |
| [FREEZE.json](FREEZE.json) | 冻结时间及预声明、模型、选择和开发数据哈希 |
| [repeat-data.json](repeat-data.json) | 冻结后读取P07/P08既有缓存的88窗口、12个混合生成段位置数值 |
| [evaluation.json](evaluation.json) | 各分组、逐论文和全材料核算、现有v2对照、目标未达成记录 |
| [audit.json](audit.json) | 生产数值交叉核对、分组边界检查及不改参数的事后门限敏感性 |
| [engineering-validation.json](engineering-validation.json) | 本轮8项测试日志、参数复现、revision/UTF-8/链接/字段审核及对应哈希 |

字段不包含正文。位置、来源标签、论文组、hash和长度仅用于拟合、控制、去重与评价；预测函数只读取五个原始分数。所有十篇历史上已被观察，旧复测也不构成新独立验证。完整解释见[维护记录](../../dev_doc/context/english-hard-negative-validation.md)和[报告](../../reports/english-hard-negative-20261004/README.md)。

在项目根目录复现：

```powershell
& '.venv/Scripts/python.exe' -X utf8 scripts/english_hard_negative.py verify
& '.venv/Scripts/python.exe' -X utf8 scripts/report_english_hard_negative.py
```

原执行顺序为prepare、fit、evaluate、verify。前三步拒绝覆盖已声明、冻结或已评价结果。进一步实验须单独声明，不能改旧参数后沿用这份FREEZE。
