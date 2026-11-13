# 专业英文与人机协作检测模型接入

后续状态：本文所记安装完成后，四模型已在115个目标输入上完成全量评分和两轮融合比较，未达到目标，默认保持；效果证据见[融合重调记录](english-specialized-fusion-validation.md)。以下保留安装阶段的范围与事实。

核对日期：2026-10-04；软件版本：0.9.0。用户要求联网搜索并下载专业英文、人机混合文本的现成模型。本轮安装四个检查点，并接入现有 Python、CLI、HTTP 和显式旧规则标注；没有训练、拟合门限或修改任何默认 profile。

## 来源与适用范围

| 本地 ID | 官方来源与任务 | 标签和本地原始分数 | 许可说明 |
|---|---|---|---|
| `followsci-bert-en` | [FollowSci BERT](https://huggingface.co/followsci/bert-ai-text-detector)：arXiv 人工段落及模型生成的学术段落 | Human=0、AI=1；取 softmax 第1类 | 模型卡 MIT |
| `roberta-academic-en` | [NTNU RoBERTa academic](https://huggingface.co/andreas122001/roberta-academic-detector)：科研摘要与 ChatGPT 生成摘要 | human-produced=0、machine-generated=1；取第1类 | 模型卡标注 OpenRAIL，保留原卡 |
| `llm-detectaive-en` | [作者项目](https://github.com/mbzuai-nlp/LLM-DetectAIve)、[官方 Space](https://huggingface.co/spaces/raj-tomar001/LLM-DetectAIve/tree/75c5a84d0815092a455cf9b2b105c0fb4d6277d3/deberta-base)：四种写作／改写方式 | 0人工、1人工后机器润色、2机器生成、3机器生成后机器人性化；`ai_score` 为后三类之和，各类另存 | 发布 Space 未明确权重许可；不据公开可下载推断再分发授权 |
| `cocodet-en` | [CoCoDet 权重](https://huggingface.co/khaaaaaan/CoCoDet)、[作者代码](https://github.com/Y1hanChen/COCONUTS/blob/25df60884f344245ce9ddd092055a781919c850b/script/infer.py)：计算机科学会议评审的 Human/Mix/AI 内容来源 | human=0、mix=1、ai=2；`ai_score=P(mix)+P(ai)`，另存三类输出 | 模型卡 MIT；[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)模型卡 Apache-2.0 |

两类目标都有公开可下载权重，但这里的“专业英文”主要指学术段落、摘要或评审，不能外推为所有专业领域。CoCoDet 按实质内容来源分类，人工内容经机器语言润色仍可能属于 human；DetectAIve 的聚合分则把机器润色也纳入。二者都不是直接为当前天线论文的生成段插入任务训练的定位模型。

作者模型卡报告的高指标来自各自资料与口径，本项目没有复现这些指标，也没有将其作为本地准确率。FollowSci 的模型卡明确列出不专门识别部分 AI 辅助文本的限制。NTNU 系列另一个 `roberta-mixed-detector` 的 mixed 指维基和科研摘要**训练领域混合**，不能据名称解释成人机混合专用检测器。

其它检索结果：

- [CoAI academic v2](https://huggingface.co/coai/roberta-ai-detector-v2)要求登录并同意提供联系信息，本轮选择可直接下载的替代模型。
- [MixSet](https://github.com/Dongping-Chen/MixSet)提供混合情景数据、训练和评估代码；本轮没有把该数据集当作已安装的专用权重。
- [HACo-Det](https://aclanthology.org/2025.acl-long.1069/)研究词／句粒度的人机共写检测；本轮未核实到可直接安装且来源完整的作者检查点，保留为研究线索。

## 本地适配与追溯

固定 revision、每文件来源路径、SHA-256、大小和实际网络事件统一记录在 [manifest](../../models/manifest.json)。本轮为29个文件、2,098,756,851字节（约2.10 GB / 1.95 GiB），目录均在 `models/<ID>/`。Hugging Face 直连失败后按既有 `scripts/network.py` 回退当前代理，权重 CDN 仍先直连；没有修改系统代理。

下载器增加显式 `--specialized-english`，支持固定 Space 的指定子目录、非标准 `CoCoDet.bin` 原文件映射为本地 `pytorch_model.bin`，以及外部基座的 config/tokenizer。原始权重字节不转换；CoCoDet 完整检查点已包含骨干参数，只取 ModernBERT 的配置和分词器，不重复下载基座权重。保留模型卡，不下载或运行仓库 Python、优化器或训练 checkpoint。

CoCoDet 使用本地 `CoCoDetModel`：ModernBERT CLS 表示和归一化分类权重做余弦点积，乘作者推理尺度30。三个辅助头也被完整定义和严格加载；原文件由 `torch.load(weights_only=True)` 读取，`strict=True` 核对全部键。标准分类器同样拒绝加载键异常。Windows 下关闭 ModernBERT 的 reference compilation，使用已有 Torch SDPA，无需新增依赖。

真实验证发现 generic `PreTrainedTokenizerFast.prepare_for_model` 不执行 Rust 模板后处理，会漏掉 ModernBERT 的 CLS/SEP。已针对固定模板逐窗补齐，并增加真实 tokenizer 模板回归；CPU/GPU与直接作者输入路径现在一致。CoCoDet 沿用作者默认2048 token窗口，其它三个为512；重叠64、完整覆盖，文档聚合仍为token覆盖权重平均。

多分类输出增加 `class_scores`、`ai_score_classes` 和 `score_semantics`，窗口及标注片段都保留；缓存重组保留语义说明。`ai_score` 是明确类别之和，不是AI文本比例或校准的来源概率。`/v1/models` 和 CLI 清单增加 `specialized_english_models`。

## 已执行验证

- 29个新文件完整SHA-256核验、`pip check`通过，运行依赖未增加或升级。
- 137项全项目回归通过：包括多分类求和、窗口聚合、空片段、形状/非有限输出拒绝、CoCoDet余弦尺度和严格缺键拒绝、ModernBERT特殊标记以及默认名单保持。
- 10条固定官方示例（NTNU模型卡5条、DetectAIve演示4类各1条、CoCoNUTS公开首条评审）共40次真实分类推理，网络socket被阻断；不把演示例子当作独立测试。
- 四模型CPU/CUDA对照，最大分数差约 `1.16e-7`；直接tokenizer/作者分类公式与服务单窗分数差均为0。
- 四模型长文覆盖分别为1089/1090/1090/3202 token，窗口数3/3/3/2；尾token及去重权重均核验。
- 真实FastAPI检测、批量检测、标注报告保存和缓存重组通过。部署与跨工作目录CLI结果见下方轻量证据。

复现入口：[test_specialized_english.py](../../scripts/test_specialized_english.py) 的 `collect` 和 `validate`；完整输入、日志及报告在Git排除的 [本机报告目录](../../reports/english-specialized-20261004/README.md)，不含正文的证据在 [evaluations](../../evaluations/english-specialized-20261004-installation/README.md)。首次检查使用了权限异常的旧pytest目录，随后使用本轮新目录通过；最初CoCoDet特殊标记失败已修复，并以最终结果为准。

本轮是检索、下载、接入和安装验证。英文原始默认仍为旧四模型，默认标注仍为 `english-academic-20261003-v2` 五模型/1600/400。没有证明达到“纯人工窗口零误报、混合生成段召回50%”，没有重新拟合或更改原门限。后续目标语料评估与冻结验收继续记录在[英文待办](../toDo/english-calibration-validation.md)。
