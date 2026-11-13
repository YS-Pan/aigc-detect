# 2026-09-30 来源明确的公开样本

本批共 **24 条独立文本，12 条人写、12 条 AIGC**。来源标签来自发布者和历史作品资料，在检测前固定。全部为中文，保留原始字形、标点、空格和换行；鲁迅作品沿用来源的繁体文本。每条单独检测，不拼接不同来源文本。

## 来源与标签依据

| 来源 | 人写 | AIGC | 标签依据与许可 |
|---|---:|---:|---|
| [HC3-Chinese open_qa](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | 4 | 4 | `human_answers` 与 `chatgpt_answers`；HC3 CC BY-SA 4.0，上游 WebTextQA/BaikeQA 为 MIT |
| [HC3-Chinese psychology](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | 4 | 4 | 同上；上游心理问答集标为 CC0 |
| [BELLE train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | 0 | 4 | 发布者披露自动生成流程；取 `output`，仅研究使用，不用于商业 |
| [魯迅《吶喊》，Project Gutenberg #27166](https://www.gutenberg.org/ebooks/27166) | 4 | 0 | 作者 Lu Xun (1881–1936)，电子版 2008-11-05 发布、2021-01-04 更新；取作品正文，不取网站自动生成摘要；来源标为美国公有领域 |

HC3 署名为 Biyang Guo et al., *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023)，并保留原答案来源。标签说明与来源许可表见[作者项目](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)。HC3 标签是数据发布者提供的来源标签，并非逐条重新联系作者认证。

BELLE 署名为 BELLE Group / LianjiaTech。作者[生成说明](https://github.com/LianjiaTech/BELLE/blob/645084d36454cdeb4e7f1ce9b2f3426bac39f1f9/data/1.5M/README.md)披露 OpenAI 生成流程：默认 `text-davinci-003`，可切换 `gpt-3.5-turbo`，但没有逐条模型版本记录，因此这里只断言自动生成，不给每条强行指定模型。其 HF 数据卡标 GPL-3.0，仓库 DATA_LICENSE 为 ODC-By，另有仅研究、禁止商业使用的声明；这些不同层次的原始声明均保存于 `sources/`，没有将全批统一重新许可。

鲁迅四条为《吶喊》自序、《孔乙己》《故鄉》《社戲》的开头连续节选，電子版製作者署名为 Han Huei Haung。原作与 Project Gutenberg 电子版本的重分发声明分别保留，见 `sources/gutenberg-header-and-license.txt`。

## 文件和复现

- [samples.jsonl](samples.jsonl)：逐条正文、来源标签、URL、revision、原行号/答案索引或全文字符范围、许可、变换说明、文本 SHA-256。
- [collection.json](collection.json)：整个 JSONL 的哈希、采集时间、先固定的选择规则、上游证据哈希和下载路线。
- `texts/`：每条正文的独立 UTF-8 文件，与 JSONL 的 `text` 字节一致。
- `sources/`：来源数据卡、作者说明、许可和 Gutenberg 电子版元信息；不保存整个上游大语料。

选择规则与具体输入在本次查看分数前固定：HC3 两个子集各取最先四组双方都有 160–1800 字符答案的问答，按答案数组取第一个满足条件者；BELLE 按文件顺序取最先四条 400–1800 字符、至少 30% 中日韩统一表意文字且无代码围栏的输出；鲁迅取四篇指定作品开头，从正文锚点开始，到至少 800 原始字符后的第一个句号结束。没有按照检测分数筛选样本，也未去除模型惯用语。

固定版本：HC3 `09a687b8dc164b89e7df95abf15df3b216bc31c2`；BELLE 数据 `29c35e9b56c7fc91f04a613ed33896f7a19bad54`，作者说明 `645084d36454cdeb4e7f1ce9b2f3426bac39f1f9`。Gutenberg 下载 URL 无不可变 revision，记录完整文件哈希及精确抽取范围；如以后内容变化，应建新批次，不能覆盖本批。

使用本工作区解释器校验与重新检测：

```powershell
& '.\.venv\Scripts\python.exe' 'scripts/public_samples.py' verify
& '.\.venv\Scripts\python.exe' 'scripts/public_samples.py' detect
```

第二条命令创建带时间的新报告目录。采集复现需另选尚不存在的目录，例如：

```powershell
& '.\.venv\Scripts\python.exe' 'scripts/public_samples.py' collect --sample-dir 'samples/public-reproduction'
& '.\.venv\Scripts\python.exe' 'scripts/public_samples.py' verify --sample-dir 'samples/public-reproduction'
```

这些命令在项目根目录执行。原始全文 `detect` 和分段 `annotate` 都运行十模型：前者提供全文 token 覆盖加权原始分数，后者使用默认 `thesis-cnki-20260929-v2`、400/120 字符、profile 固定 revision 及双头门限。结果解释见[API](../../user_doc/api.md)和[长文指南](../../user_doc/long-text-annotation.md)。

## 检测归档

已于 2026-09-30 21:45（Asia/Shanghai）启动并完成本机 CUDA 检测，软件版本 0.4.0。24 条共 9,718 个 Python Unicode 字符；原始全文检测共 240 个模型×文本结果，默认分段标注共 37 段，每段均有十模型分数。推理期间阻止 socket 网络连接，全部文本覆盖、输入哈希、模型 revision 和 profile 哈希校验通过。

| 来源标签 | 样本数 | 含强信号片段 | 仅含疑似片段 | 全部其他 |
|---|---:|---:|---:|---:|
| AIGC | 12 | 0 | 0 | 12 |
| 人写 | 12 | 0 | 0 | 12 |

样本汇总只是片段描述：出现强信号优先记强，否则出现疑似记疑似，否则为其他。默认 v2 把本批全部 37 段归为 `other`，包括 12 条来源明确的 AIGC；与此同时，原始全文检测中有七个单模型对全部 12 条 AIGC 给出 ≥0.90 的分数，也有四条人写被至少一个模型给出 ≥0.90。当前默认参考融合没有筛出本批 AIGC，单模型高分也存在误标，不能据此报告准确率或认证来源。

完整结果见[本次报告入口](../../reports/public-samples-20260930-24/README.md)、[逐条十模型分数 CSV](../../reports/public-samples-20260930-24/scores.csv)、[原始检测 JSON](../../reports/public-samples-20260930-24/raw-detection.json)与[验证说明](../../dev_doc/context/public-samples-validation.md)。每条完整报告含原文、JSON/JSONL、HTML 高亮和完成标记；完整结果不进入 Git。报告链接仅适用于此本机工作区，版本管理保留来源样本与轻量验证说明。

## 适用范围

这些是可复核的公开示例，不是随机抽样的准确率基准。`hc3-zh` 用整个 HC3-Chinese 训练，存在已知训练重叠；其余模型对各来源的训练重叠未经审计，经典名篇也可能出现在模型训练中。文学、问答和指令输出的文体及长度有差异，不能据此证明学术论文或最新模型文本的检测效果。

v2 原来在指定论文参考材料上拟合；本批使用冻结配置观察新来源，没有修改 profile、模型或门限。筛查分不是概率或来源证明，强信号/仅疑似/其他都是检测器的标签，不能代替 `source_label`。没有用本批重拟合，也没有把它记为已有“整篇独立论文/CNKI 位置验证”待办已完成。
