# 本地中文 AIGC 检测

为其他工作区提供离线文本检测。模型、Python 虚拟环境和服务代码都位于本工作区。

## 模型

| API ID | 发布仓库 | 用途 |
|---|---|---|
| `mpu-zh-v3` | `yuchuantian/AIGC_detector_zhv3` | 中文常规文本 |
| `mpu-zh-v3-short` | `yuchuantian/AIGC_detector_zhv3short` | 中文短文本及长文本 |
| `mpu-zh-v2` | `yuchuantian/AIGC_detector_zhv2` | 官方演示保留的中文旧版 |
| `hc3-zh` | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | 早期 ChatGPT 中文检测基线 |
| `ensemjudge-bert` | `johnsonwangzs/mgtd-sys-model-weights` 的 Chinese BERT 子目录 | EnsemJudge 的独立 BERT 分类器 |
| `ensemjudge-roberta` | 同仓库的 Chinese RoBERTa 子目录 | EnsemJudge 的独立 RoBERTa-large 分类器 |
| `anx-bert-zh` | `AnxForever/chinese-ai-detector-bert` | 中文 BERT v11c |
| `paperpass-v3` | `yibo365/paperpass-v3` | 社区发布的中文学术段落分类器 |
| `cmj-zh` | `mokawa3018/cmj-chinese-aigc-text-detector` | 基于 MPU v3 继续微调 |
| `bibbbu-xlm-roberta` | `bibbbu/multilingual-ai-human-detector_xlm-roberta-base` | 包含中文的中英越多语言分类器 |

模型选择、官方依据和未纳入的候选见 [模型来源](dev_doc/context/model-sources.md)。
原始 `detect` 接口返回各模型分数，全文按 token 滑动窗口覆盖。长文 `annotate` 默认使用版本化策略 `thesis-cnki-20260929-v1`：十模型全部参与，按 400/120 字符切段，revision 必须与 profile 一致，并按 profile 中保存的门限对融合筛查分分档。融合分不是概率或来源证明。旧 `sensitive-any-model-v1` 可显式选用，默认原始门限为 0.90/0.70。

长文标注和无 GPU 重新分档命令见[用户指南](user_doc/long-text-annotation.md)。融合算法、参考集拟合和独立检验边界见[融合模块说明](dev_doc/blueprints/fusion.md)及[跨工作区指南](user_doc/dissertation-integration.md)。旧长文实测记录不能代表新 profile 的泛化准确率。

## 立即使用

在任意 PowerShell 工作目录启动服务：

```powershell
& 'D:\project\20260928 aigc detect\scripts\start-api.ps1'
```

健康检查：`http://127.0.0.1:8765/health`。交互接口文档：`http://127.0.0.1:8765/docs`。

不启动服务也可以调用命令行：

```powershell
& 'D:\project\20260928 aigc detect\.venv\Scripts\python.exe' -m aigc_detector detect `
  --file 'D:\project\20260928 aigc detect\samples\demo-zh.txt' `
  --output 'D:\project\20260928 aigc detect\reports\example.json'
```

默认运行全部 10 个模型，可用 `--models` 选择子集。首次推理会从本地磁盘加载权重。自动使用 CUDA，有需要可显式指定 CPU。

例如只运行本次新增模型：

```powershell
& '.\.venv\Scripts\python.exe' -m aigc_detector detect --file samples/demo-zh.txt `
  --models ensemjudge-bert ensemjudge-roberta anx-bert-zh paperpass-v3 cmj-zh bibbbu-xlm-roberta
```

## 文档入口

- **调用方 / dissertation agent**：[user_doc/README.md](user_doc/README.md)，随后读 [跨工作区调用](user_doc/dissertation-integration.md)。
- **长文标注与保存**：[三档标注指南](user_doc/long-text-annotation.md)，默认十模型融合分档，生成 JSON、JSONL、原文和离线 HTML。
- **维护者**：[dev_doc/README.md](dev_doc/README.md)。
- **重新安装 / 验证**：[安装与运行](user_doc/install-and-run.md)。
- **测试证据**：[验证记录](dev_doc/context/installation-validation.md) 和 [原始试跑结果](reports/smoke-test.json)。
- **十模型扩展**：[安装与验证](dev_doc/context/model-expansion-validation.md)、[测试摘要与分数表](reports/model-expansion-20260929/README.md) 和 [本轮原始试跑结果](reports/model-expansion-20260929/smoke-test.json)。
- **十万字符验证**：[验证记录与示例](dev_doc/context/long-text-validation.md)。
- **当前论文参考校准**：[结果与迁移限制](dev_doc/context/fusion-validation.md)、[可复现纯分数数据](evaluations/thesis-cnki-20260929/README.md)。

文档层级参考用户指定的 yadof 项目，保留用途分层与按需阅读；本项目以当前工作区的源码、虚拟环境和本地权重为运行单位。

Git 保留源码、文档、有来源许可的公开测试样本、纯分数校准证据和 `models/manifest.json`；论文原文、完整检测报告、CNKI 原始 PDF、模型权重及 checkpoint 不入库。当前仓库在本地 `main` 分支初始化，未配置 remote。
