# 英文非线性融合纯分数证据

`english-academic-20261004-v3`，schema4.0；648个候选的固定范围、分组、数值和哈希均可追溯。
[本机完整实验说明](../../reports/english-nonlinear-20261004-v3/README.md)与[维护验证](../../dev_doc/context/english-nonlinear-validation.md)。

- campaign.json：复用分数前声明的候选、优化算法、分组和默认替换条件。
- train-scores.json：选择分片的旧fit/calibration原始分数，字节与声明来源相同，不含正文。
- selection.json：648候选、各论文折的误标/检出及优化诊断。
- FREEZE.json：profile、数值、campaign、selection及拟合源码哈希。
- repeat-scores.json：冻结后仅对P07/P08完整派生输入评分，记录本地缓存和网络阻断。
- evaluation.json：三组完整核算、旧v2对照及预声明替换条件。
- engineering-validation.json：实际运行验证和最终默认状态。

参数与门限复现不读取正文、不加载模型、不使用复测：

```powershell
& '.venv/Scripts/python.exe' -X utf8 scripts/calibrate_english_nonlinear.py reproduce
```

P07/P08历史结果已知，复测不是新的独立验证；拟合/CV结果不能称为准确率。
