# 四个专业英文／人机协作模型安装证据

`installation.json`保存0.9.0真实工程验证的数值摘要、固定来源和文件哈希，不含待测正文。四模型的完整权重清单由[manifest](../../models/manifest.json)统一维护。

137项回归、29文件SHA-256、pip check、10官方示例×4模型离线推理、四模型CPU/CUDA与单窗公式、长文尾部覆盖、实际HTTP/跨cwd CLI、报告保存和重组已完成。

复现：

```powershell
& '.venv/Scripts/python.exe' scripts/download_models.py --specialized-english --download-workers 4
& '.venv/Scripts/python.exe' scripts/verify_models.py --models followsci-bert-en roberta-academic-en llm-detectaive-en cocodet-en
& '.venv/Scripts/python.exe' scripts/test_specialized_english.py collect --report-dir reports/specialized-new
& '.venv/Scripts/python.exe' scripts/test_specialized_english.py validate --report-dir reports/specialized-new
& '.venv/Scripts/python.exe' scripts/test_specialized_english.py deployment --report-dir reports/specialized-new
```

deployment需要已经启动当前版本的本机服务；输出目录使用新的名称，保护已完成证据。所有命令从项目根运行。完整报告位于Git排除的[报告目录](../../reports/english-specialized-20261004/README.md)。

本证据只确认安装与接口可用。官方演示样本不是目标领域独立测试，未全量重测115个学术输入、未拟合新门限或改变默认profile，不证明达到纯人工窗口零误报或混合生成段50%召回。来源与任务差异见[接入记录](../../dev_doc/context/english-specialized-validation.md)。
