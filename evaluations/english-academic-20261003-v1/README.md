# 英文纯分数校准证据

本目录不包含正文、PDF或生成响应。样本仅保存在Git排除的[samples批次](../../samples/english-academic-20261003/README.md)，详细方法及限制见[验证说明](../../dev_doc/context/english-calibration-validation.md)。冻结测试人工误报0/19，但12处生成段全部漏检；本轮不能证明提高独立检出率。

| 文件 | 用途 |
|---|---|
| campaign.json | 仅由拟合资料触发的上下文/特征搜索扩展及准则 |
| context400、context800、context1200、context1600 | 各上下文的候选profile和逐篇留出拟合诊断；400分数在根scores.json，其他长度在各子目录 |
| initial-linear-400 | 初始线性/较大余量方案，保留未检出结果 |
| train-scores.json | 最终1600/400拟合及校准分数；没有测试记录 |
| test-scores.json | 配置冻结后新算的两篇测试分数 |
| selection.json、FREEZE.json | 选择过程、参数冻结和文件哈希 |
| evaluation.json | 新旧规则在同分片上的原文、纯生成段及混合正文比较 |

模型revision与源代码哈希在数据config中；最终profile保存拟合数据、campaign哈希、标准化、系数、门限和拟合收敛证据。

复核现有结果（不重新推理）：

```powershell
& '.venv/Scripts/python.exe' -X utf8 -B scripts/english_academic_samples.py verify
& '.venv/Scripts/python.exe' -X utf8 -B scripts/calibrate_english.py evaluate `
  --test-dataset evaluations/english-academic-20261003-v1/test-scores.json `
  --evaluation reports/新的英文复核结果.json
```

重建拟合时在新目录指定`--profile`和`--diagnostics`，不要覆盖冻结profile。先按采样协议留出新的文献，再运行`score --splits fit calibration`；`test`打分需要FREEZE中的profile哈希与实际配置一致。已有数据已经使用过，重拟合不能重新称为独立测试。
