# 英文扩展融合纯分数证据

配置：`english-academic-20261003-v2`；schema3.0；全部原版/混合/生成段按论文绑定分组。
[完整本机说明](../../reports/english-recalibration-20261003-v2/README.md)和[维护验证](../../dev_doc/context/english-recalibration-validation.md)记录实际结果、工程验证与限制。原始材料和完整报告不进入Git。

- campaign.json：打分前声明的信号、候选范围、分组、完整样本利用与已知旧测试结果。
- context400/context800/context1600：fit+calibration完整输入分片的纯分数与逐论文候选比较。
- selection.json：仅用fit论文留出选择的最终候选及诊断。
- FREEZE.json：profile、训练数据、候选选择及campaign哈希。
- repeat-scores.json：冻结后P07/P08原版、六份混合版本与十二处单独生成段的打分。
- evaluation.json：三组原版误标、单独生成段命中、混合生成块/字符覆盖与人工溢出。

- engineering-validation.json：最终114项回归、真实部署/代码hash、115个完整报告及纯数值/UTF-8/链接检查摘要，不含正文。

这是已知参考材料上的重校准及重复评估，不是新独立验证。所有运行时特征来自固定版本模型的原始输出，论文/作者/文件身份及长度不进入融合。

参数复现不读取正文、不加载模型：

```powershell
& '.venv/Scripts/python.exe' -X utf8 scripts/calibrate_english_v2.py reproduce
```
