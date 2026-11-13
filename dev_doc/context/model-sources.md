# 模型范围与来源

初次核对日期：2026-09-28。原有范围来自用户指定的前序 Codex 聊天 `01a0e75d-24c0-7a91-891e-ca25cac24938`。2026-09-29 用户进一步确认新增六个分类器，排除 LLM-Detector-Small-zh 和 EnsemJudge 完整大模型集成。

2026-10-02的英文四模型来源见[英文扩展记录](english-expansion-validation.md)；2026-10-03补充的五个英文分类器和GPT-Neo/Falcon统计资源见[本轮来源与验证](english-additional-validation.md)。下面的“未纳入”保留2026-09中文范围的历史语境。

## 纳入

| ID | 官方证据 | 标签依据 |
|---|---|---|
| mpu-zh-v3 | [MPU 官方项目](https://github.com/YuchuanTian/AIGC_text_detector)、[权重](https://huggingface.co/yuchuantian/AIGC_detector_zhv3) | 作者 demo：Human/AI 对应 0/1 |
| mpu-zh-v3-short | [短文本权重](https://huggingface.co/yuchuantian/AIGC_detector_zhv3short) | 同上 |
| mpu-zh-v2 | [中文 v2 权重](https://huggingface.co/yuchuantian/AIGC_detector_zhv2) | 同上 |
| hc3-zh | [官方模型卡](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta-chinese) | config：Human=0，ChatGPT=1 |

MPU 的 [官方推理代码](https://huggingface.co/spaces/yuchuantian/AIGC_text_detector/blob/191c982/app.py) 同时提供这三个中文版本。v2 作为该模型系列仍提供的中文版本一并安装；早期 beta 和所有训练种子不是本次独立扩展的范围。HC3 中文模型卡说明它在完整 HC3-Chinese 上训练，没有留出集。

模型 revisions 与每个文件 SHA-256 以 [manifest](../../models/manifest.json) 为准，避免在多份文档抄写版本造成偏差。MPU 模型库标注 Apache-2.0；HC3 模型卡未明确标注权重许可证，应保留原模型卡，若将来再分发需另行核对。HC3 样本遵循数据集卡标注的 CC-BY-SA-4.0。

## 2026-09-29 新增范围

| ID | 作者发布的权重或项目 | 标签与结构依据 |
|---|---|---|
| ensemjudge-bert | [BERT checkpoint](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/chinese-bert-sft_03-28_20-17_908) | [任务数据格式](https://github.com/NLP2CT/NLPCC-2025-Task1#data-format)：Human=0、Machine=1；BERT + 两层分类头 |
| ensemjudge-roberta | [RoBERTa checkpoint](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/chinese-roberta-sft_03-21_16-39_938) | 同上；中文 RoBERTa 的实现仍使用 BERT 结构，分类头隐藏层为 512 |
| anx-bert-zh | [AnxForever 模型卡](https://huggingface.co/AnxForever/chinese-ai-detector-bert) | config 明确 human-written=0、AI-generated=1；标准 BERT 分类器 |
| paperpass-v3 | [yibo365 模型卡](https://huggingface.co/yibo365/paperpass-v3) | 模型卡 Human=0、AI=1；标准 BERT 分类器，基座为中文 RoBERTa-large |
| cmj-zh | [CMJ 模型卡](https://huggingface.co/mokawa3018/cmj-chinese-aigc-text-detector) | config 明确 human=0、AI=1；MPU v3 继续微调 |
| bibbbu-xlm-roberta | [多语言模型卡](https://huggingface.co/bibbbu/multilingual-ai-human-detector_xlm-roberta-base) | [固定版数据代码第 59 行](https://github.com/vutuongvy101/multilingual-ai-human-text-detection/blob/1fbb02a7d77e454d4c3a21e8bcc262bd03db3190/src/multilingual_ai_detection/data/data_splitter.py#L59)：human=0、machine=1；标准 XLM-RoBERTa 分类器 |

### EnsemJudge 接入依据

[上游实现](https://github.com/johnsonwangzs/MGT-Mini/blob/132d74813136298de66c37cf67520ddb19d265fb/detector/model/neural_based/chinese_bert_detector.py) 使用 BERT pooler，经 dropout 后送入 Linear → ReLU → Dropout → Linear 分类头。普通 `AutoModelForSequenceClassification` 的单层头不匹配这些权重。本项目独立定义该层结构，复用 Transformers 的 BERT 前向计算；不导入或运行上游 Python 文件。

两个 checkpoint 仅提供 config 和 safetensors。按照[作者配置](https://github.com/johnsonwangzs/MGT-Mini/blob/132d74813136298de66c37cf67520ddb19d265fb/detector/config.py)，分别从 `hfl/chinese-bert-wwm-ext`、`hfl/chinese-roberta-wwm-ext-large` 获取 tokenizer。权重及 tokenizer 均固定 revision，原文件不改写。本项目输出两个独立分类器的分数，不代表完整 EnsemJudge 集成方案的输出或论文成绩。

### 数据覆盖与许可记录

- EnsemJudge 的比赛数据包括中文作文、新闻和学术文本，生成来源包括 GPT-4o、GLM-4-flash、Qwen-turbo。上游代码文件头限制用于比赛相关评估；检测权重仓库没有明确的独立权重许可证。本次仅记录本地安装与试跑，不据此声明可自由再分发或商业使用。
- AnxForever 模型卡标注 MIT；属于毕业设计项目，训练偏新闻、百科、技术和正式文体。主分类器与其单独发布的实验性 span 模型是不同权重，本次只安装主分类器。
- yibo365 模型卡标注 MIT；部分训练标签来自 PaperPass 分数，另含学术论文与多种生成模型文本。其账号和仓库名称不证明是 PaperPass 官方模型。
- CMJ 模型卡和 LICENSE 标注 Apache-2.0，同时提示上游及训练数据条款需一并考虑；其新增训练集较小，且与已安装 MPU 同源。本次安装 PyTorch 格式，不重复下载 ONNX 格式。
- bibbbu 模型卡标注 MIT；训练中包含中文，但 AI 样本仅来自 Qwen2.5-1.5B-Instruct，中文测试集为 90 条。checkpoint 配置写有 Transformers 5.11.0，本地已在现有 4.57.6 上完整加载并完成 GPU/CPU 试跑，见 [扩展验证](model-expansion-validation.md)。
- AnxForever 和 bibbbu 训练主要使用 256 token，配置支持 512；本地服务采用 512 token 窗口完整覆盖输入。AnxForever 的可选温度缩放与 paperpass-v3 的参考分档线均未引入，继续保留未校准分数。

此次核对的模型 API 元数据、配置、模型卡和源码依据保存在 [upstream 证据目录](../../reports/model-expansion-20260929/upstream/)。下载和试跑证据另行保存，不能把各作者自报指标合并为模型排名。

## 本次未纳入的候选

- [Desklib](https://huggingface.co/desklib/ai-text-detector-v1.01)：当前模型卡明确面向英文。
- [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)：默认评分模型 GPT-Neo 为英文模型，官方说明未给出中文检测验证。自行换中文生成模型属于新的适配工作。
- [Binoculars](https://github.com/ahans30/Binoculars)：官方说明更擅长英文；默认 Falcon 双模型未提供可据此确认的中文专用验证与门限。
- [humanize-chinese](https://github.com/swaylq/humanize-chinese)、[lynote-ai/ai-text-detector](https://github.com/lynote-ai/ai-text-detector)：规则和统计脚本，不是本次要求安装的神经检测模型。
- [AIGC-Detector-Pro](https://github.com/free-revalution/AIGC-Detector-Pro)：依赖宿主大模型分析，未提供独立离线权重。

这些排除说明只定义本轮安装范围；不表示相关方法在其他任务中无用。

2026-09-29 补充：用户明确排除 `QiYuan-tech/LLM-Detector-Small-zh`；确认本次只安装六个分类器，因此不安装 EnsemJudge 的 Qwen/GLM LoRA、混合特征模块或完整投票系统。

2026-10-04新增FollowSci学术段落BERT、NTNU学术摘要RoBERTa、LLM-DetectAIve四分类与CoCoDet评审Human/Mix/AI三分类。精确来源、许可、类别语义及未下载候选见[专业英文接入记录](english-specialized-validation.md)；当前共22分类器/3统计LM资源。新四模型显式选择，不替换默认profile。
