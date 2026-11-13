# 专业英文融合：首轮数值证据

四新模型全量离线打分后进行42候选实验。嵌套留出14/390人工窗口误报、4/51混合段命中；开发选择优胜者冻结复测0/58误报、0/12命中。未改善v2，默认保持。全部十篇为历史开发资料，不是独立验证。

| 文件 | 内容 |
|---|---|
| [campaign.json](campaign.json) | 预声明候选、分组、损失、门限、revision及源码／来源hash |
| [development-data.json](development-data.json) | 590开发窗口、63个混合段bag及锁定原始输出 |
| [selection.json](selection.json) | 42候选选择折与完整嵌套留出 |
| [frozen-models.json](frozen-models.json) | 20类冠军参数及门限，不是运行profile |
| [FREEZE.json](FREEZE.json) | 复测前冻结的时间和hash |
| [repeat-data.json](repeat-data.json) | P07/P08的88窗口、12个混合段及新模型分数 |
| [evaluation.json](evaluation.json) | v2对照、分组／逐篇／全材料核算 |
| [audit.json](audit.json) | 数值、缓存、分组边界、误报重复hash及门限诊断 |
| [real-inference-validation.json](real-inference-validation.json) | 16次单条离线推理与批量缓存对照 |
| [engineering-validation.json](engineering-validation.json) | 两轮实际检查和证据hash汇总 |

发现旧1e-5裁剪抹平微小分数后另开[精度对照](../english-specialized-precision-20261004/README.md)，没有改本轮参数。完整解释见[验证记录](../../dev_doc/context/english-specialized-fusion-validation.md)和[报告](../../reports/english-specialized-fusion-20261004/README.md)。

```powershell
& '.venv/Scripts/python.exe' -X utf8 scripts/english_specialized_fusion.py verify
& '.venv/Scripts/python.exe' -X utf8 scripts/report_english_specialized_fusion.py
```

JSON只含数值、标识、偏移和哈希，不含论文正文。正文及完整缓存位于Git排除目录；再次实验应另建目录，不能覆盖冻结证据。
