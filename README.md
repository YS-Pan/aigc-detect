# 本地中英文 AIGC 检测

提供 Python、命令行和本机 HTTP 接口的离线文本筛查工具。当前源码版本为 **0.10.0**，需要 Python 3.11 或更新版本。代码与小型资源安装为 Python 包；模型另存于使用者选择的目录，安装后运行不依赖源码目录，也不依赖其他工具项目。

本仓库发布源码、测试、依赖声明和使用文档，**不包含模型权重，也未通过本次发布上传 PyPI**。模型准备阶段从上游下载；检测阶段只读取本地权重，不执行远程模型代码，不把待测正文发送给外部服务。各模型的来源、revision 和许可见[模型来源](dev_doc/context/model-sources.md)，使用前应核对所选模型的许可，尤其是 RADAR 的非商业限制。

## 能力与限制

- `detect` 返回各模型的**未校准原始分数**，通过 token 滑窗覆盖全文；默认中文十模型，显式 `--language en` 为旧英文四模型。
- `annotate` 输出三档筛查标签、原文位置、JSON、JSONL 和离线 HTML。中文默认 `thesis-cnki-20260929-v2`，固定十模型、400/120 字符分片；英文默认 `english-academic-20261003-v2`，固定五分类器、1600/400 字符分片。revision 与门限由版本化 profile 锁定。
- 分数与标签是人工复核线索，**不是 AI 文本占比、作者身份或来源证明，也不能作为论文通过或处分结论**。未标记不能排除 AIGC。
- 英文仍有严重漏检：旧论文重复评估仅命中 3/12 个生成段，混合生成字符覆盖 20.38%；2026-10-06 六个新增来源的冻结验收中，v2 人工 0/323 误标、生成段／叙述槽位仅 5/112 命中。这些局部结果不代表普遍零误报或准确率。全部 30 个来源现已见，后续复用不能再称为新盲测。详见[重校准证据](dev_doc/context/english-recalibration-validation.md)和[多来源验证](dev_doc/context/english-diverse-validation-20261006.md)。
- 中文融合也经过论文参考集拟合；参考集结果和留出重拟合诊断不构成独立泛化验证。边界见[非线性融合验证](dev_doc/context/nonlinear-fusion-validation.md)。

## 从源码安装并使用

以下为 PowerShell 示例。先克隆并安装轻量基础包；此步骤不安装 PyTorch、不下载模型：

```powershell
git clone https://github.com/YS-Pan/aigc-detect.git
cd aigc-detect
python -m pip install .
python -m aigc_detector --help
python -m aigc_detector models
```

安装推理依赖，明确选择模型目录并准备中文组，再验证资源：

```powershell
python -m pip install '.[inference]'
python -m pip check
python -m aigc_detector prepare --assets-dir 'D:/AIGC-models' --language zh
python -m aigc_detector verify --assets-dir 'D:/AIGC-models' --language zh
```

`D:/AIGC-models` 只是示例，可换成任意有足够空间的目录或已有兼容模型目录。模型体积较大，首次准备需要网络和磁盘空间；安装代码本身不下载权重。GPU 用户应先选择适合机器的 PyTorch/CUDA 版本；也可在检测时指定 `--device cpu`。资源核验不替代真实推理验证。

用 UTF-8 正文执行检测或保存完整标注报告；报告目录必须是尚不存在的新目录：

```powershell
python -m aigc_detector detect --file '正文.txt' --assets-dir 'D:/AIGC-models' --output 'result.json'
python -m aigc_detector annotate --file '正文.txt' --assets-dir 'D:/AIGC-models' --report-dir 'annotation-result'
```

英文标注先用 `prepare --annotation en` 准备固定五模型，再运行 `annotate --language en`。英文原始 `detect --language en` 的默认四模型组与标注组不同，详见[英文指南](user_doc/english-detection.md)。`--models` 可显式选择分类器；默认标注不能通过减少模型来替代。

如需 HTTP 服务，在源码目录安装 `http` extra 后启动：

```powershell
python -m pip install '.[http]'
python -m aigc_detector serve --assets-dir 'D:/AIGC-models'
```

服务默认只监听 `127.0.0.1:8766`，健康检查为 `http://127.0.0.1:8766/health`，交互接口文档为 `http://127.0.0.1:8766/docs`。完整的源码／wheel 安装、模型准备、停止与故障处理见[安装与运行](user_doc/install-and-run.md)。安装后的 CLI 可从其他目录调用。

## 模型

当前支持 22 个分类器。表中为上游仓库标识；权重不随本仓库分发。

| API ID | 上游仓库 | 用途 |
|---|---|---|
| `mpu-zh-v3` | `yuchuantian/AIGC_detector_zhv3` | 中文常规文本 |
| `mpu-zh-v3-short` | `yuchuantian/AIGC_detector_zhv3short` | 中文短文本及长文本 |
| `mpu-zh-v2` | `yuchuantian/AIGC_detector_zhv2` | 中文历史基线 |
| `hc3-zh` | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | 早期 ChatGPT 中文基线 |
| `ensemjudge-bert` | `johnsonwangzs/mgtd-sys-model-weights` 的 Chinese BERT 子目录 | EnsemJudge 独立分类器 |
| `ensemjudge-roberta` | 同仓库的 Chinese RoBERTa 子目录 | EnsemJudge 独立分类器 |
| `anx-bert-zh` | `AnxForever/chinese-ai-detector-bert` | 中文 BERT v11c |
| `paperpass-v3` | `yibo365/paperpass-v3` | 社区中文学术段落分类器 |
| `cmj-zh` | `mokawa3018/cmj-chinese-aigc-text-detector` | 基于 MPU v3 微调 |
| `bibbbu-xlm-roberta` | `bibbbu/multilingual-ai-human-detector_xlm-roberta-base` | 中文／英文组共享分类器 |
| `hc3-en` | `Hello-SimpleAI/chatgpt-detector-roberta` | 英文早期 ChatGPT 基线 |
| `desklib-en` | `desklib/ai-text-detector-v1.01` | 英文 DeBERTa 单输出分类器 |
| `fakespot-en` | `fakespot-ai/roberta-base-ai-text-detection-v1` | 英文 RoBERTa 分类器 |
| `openai-base-en` | `openai-community/roberta-base-openai-detector` | GPT-2 历史基线 |
| `openai-large-en` | `openai-community/roberta-large-openai-detector` | RoBERTa-large 历史基线 |
| `piratexx-en` | `PirateXX/AI-Content-Detector` | 英文 RoBERTa-base |
| `mage-en` | `yaful/MAGE` | Longformer，4096 token 滑窗 |
| `radar-vicuna-en` | `TrustSafeAI/RADAR-Vicuna-7B` | RoBERTa-large，限非商业用途 |
| `followsci-bert-en` | `followsci/bert-ai-text-detector` | 英文学术段落 BERT |
| `roberta-academic-en` | `andreas122001/roberta-academic-detector` | 英文科研摘要 RoBERTa |
| `llm-detectaive-en` | Space `raj-tomar001/LLM-DetectAIve` 的 deberta-base 子目录 | 人工、生成及两种机器改写四分类 |
| `cocodet-en` | `khaaaaaan/CoCoDet` | 学术评审 Human/Mix/AI 三分类 |

四个专业英文／人机协作模型按 ID 显式调用，DetectAIve 与 CoCoDet 保留全部类别分数；前者聚合包含机器润色，后者聚合 Mix+AI。接入验证不代表达到零误报或 50% 混合段召回，默认英文 v2 保持。

另支持 GPT-Neo 2.7B、Falcon 7B／7B-Instruct 的本地 Fast-DetectGPT 与 Binoculars 原始统计量，需 `statistical` extra 及对应资源。大模型的内存、设备与卸载限制见[英文指南](user_doc/english-detection.md)。非线性英文 v3 仅供显式实验，冻结复测低于线性 v2。

## 文档与验证

- **安装与首次调用**：[安装与运行](user_doc/install-and-run.md)、[用户文档入口](user_doc/README.md)。
- **Agent 跨工作区调用**：[aigc-detect skill](.agents/skills/aigc-detect/SKILL.md)，包含正文准备、CLI／HTTP、保存与结果解释。
- **接口与长文**：[API／Python SDK](user_doc/api.md)、[三档标注](user_doc/long-text-annotation.md)、[英文检测](user_doc/english-detection.md)。
- **公开样本**：[样本索引](samples/README.md)，含来源、许可与文本哈希；完整本机检测报告不随仓库发布。
- **独立包验证**：[0.10.0 变更记录](dev_doc/change_records/20261113_184500_independent-package.md)，区分已完成验证与本机联合验收余项。
- **模型与效果证据**：[模型来源](dev_doc/context/model-sources.md)、[十模型接入](dev_doc/context/model-expansion-validation.md)、[专业英文接入](dev_doc/context/english-specialized-validation.md)。
- **中文融合复现**：[当前 v2 说明](dev_doc/context/nonlinear-fusion-validation.md)、[v2 纯分数数据](evaluations/thesis-cnki-20260929-v2/README.md)、[历史 v1 数据](evaluations/thesis-cnki-20260929/README.md)。
- **维护开发**：[开发文档入口](dev_doc/README.md)、[环境记录](dev_doc/development_environment.md)。

Git 保留可公开源码、测试、文档、有来源许可的公开样本、纯分数证据和 `models/manifest.json`。私人论文、完整检测报告、CNKI PDF、模型权重、虚拟环境、checkpoint、密钥、缓存及大型运行产物保留在本地。公开源码仓库为 [YS-Pan/aigc-detect](https://github.com/YS-Pan/aigc-detect)。
