# 专业英文融合：微小分数精度对照

首轮继承的1e-5 logit裁剪抹平了FollowSci全部和CoCoDet大部分开发分数。本轮比较保留微小数值的native logit与直接概率输入，预声明75候选；模型名单、损失、分组和人工门限规则沿用首轮。首轮复测结果已经知晓，本轮不是新独立验证。

完整选择流程嵌套留出24/390人工窗口误报、8/51混合段命中；开发选择优胜者冻结复测0/58误报、0/12命中。其它冻结冠军复测最多1/12，现有v2为2/12；默认保持，没有按复测结果重选模型或门限。

| 文件 | 内容 |
|---|---|
| [campaign.json](campaign.json) | 精度表示、75候选、来源hash和已知首轮结果声明 |
| [development-data.json](development-data.json) | 首轮相同590窗口原始数值，未重新推理 |
| [selection.json](selection.json) | 选择折、嵌套流程及优化证据 |
| [frozen-models.json](frozen-models.json) | 31类冠军的变换、中心、尺度、系数和门限 |
| [FREEZE.json](FREEZE.json) | 本轮参数冻结及hash |
| [repeat-data.json](repeat-data.json) | 冻结后引用首轮相同88窗口缓存 |
| [evaluation.json](evaluation.json) | 分组／逐篇／全材料核算 |
| [audit.json](audit.json) | 2250内层折和6外层边界、标量公式及门限诊断 |

参数未登记为运行profile。工程验证统一保存于[首轮汇总](../english-specialized-fusion-20261004/engineering-validation.json)。方法与限制见[维护记录](../../dev_doc/context/english-specialized-fusion-validation.md)及[完整报告](../../reports/english-specialized-fusion-20261004/README.md)。

```powershell
& '.venv/Scripts/python.exe' -X utf8 scripts/english_specialized_precision.py verify
& '.venv/Scripts/python.exe' -X utf8 scripts/report_english_specialized_fusion.py
```
