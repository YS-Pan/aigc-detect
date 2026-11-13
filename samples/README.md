# 检测样本

`samples/` 是本项目公开样本的统一存放位置。样本正文、元数据和来源许可保存在这里；实际运行的完整结果保存到 Git 排除的 `reports/`。私人材料默认由调用方管理；作者明确要求存入这里时，按批次单独建目录并加入 Git 排除，不能因此视为已获公开转载许可。

| 样本集 | 用途与入口 |
|---|---|
| [english-public-20261002/README.md](english-public-20261002/README.md) | 可选英文接入后的20条公开文本：12条来源人工、8条ChatGPT；含Gutenberg历史著作、HC3八组成对回答、来源哈希与四模型实际分数 |
| [public-20260930/README.md](public-20260930/README.md) | 2026-09-30 联网收集的来源明确样本，12 条人写、12 条 AIGC；含来源证据、逐条正文、复现命令和检测结果入口 |
| [kechuang-human-20260930/README.md](kechuang-human-20260930/README.md) | 作者提供的本地 MHTML：14 个主题、159 条本人帖子，按作者声明标为 human；图文与 UTF-8 正文及位置映射保存在本地，整个批次由 Git 排除 |
| [smoke.jsonl](smoke.jsonl) | 原有安装、离线运行和长窗口覆盖夹具，共 8 条；只有 6 条 HC3 回答有来源标签 |
| [demo-zh.txt](demo-zh.txt) | 官方 MPU 演示文本，无来源标签，供 CLI 示例使用 |

## 新增样本约定

- 按批次建立子目录，提供 `README.md`、`samples.jsonl`、`collection.json`、`texts/` 和 `sources/`。
- `source_label` 使用 `human` / `ai`；必须来自数据发布者的字段、公开生成过程或可核验作者资料，不能根据检测分数反推。
- 记录来源 URL、固定 revision（若可用）、原始行号/答案索引或字符范围、许可及限制、抽取规则、UTF-8 SHA-256 和采集时间。正文不做未记录的改写。
- 在查看检测结果前固定样本选择规则和配置。保留训练重叠信息；来源标签与检测器分档分别存放。
- 完整 JSON/JSONL/HTML 结果进入新的 `reports/` 子目录；目录完成后应有 `COMPLETE.json`。轻量验证说明可以进入 `dev_doc/context/`。

新增批次可复用 [public_samples.py](../scripts/public_samples.py) 的校验及检测入口；新的采集来源需另外定义并记录抽取规则。该脚本的 `collect` 专门复现本次固定来源和 24 条选择规则，不表示全网通用采集器。

## 原有 smoke 样本与历史来源说明

These samples test installation, inference, offline operation and long-text coverage. They do not establish thresholds or detection accuracy.

- `mpu-official-demo-zh`: Chinese text copied from the author's demo, no ground-truth label.
- Six HC3-Chinese answers: first three baike rows with both human and ChatGPT answers of at least 100 characters. Labels preserve the source answer field; they are not judgments made by this wrapper.
- `long-window-coverage`: six answers concatenated and repeated three times to exercise multiple windows.

HC3 source: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/resolve/09a687b8dc164b89e7df95abf15df3b216bc31c2/baike.jsonl

MPU demo source: https://huggingface.co/spaces/yuchuantian/AIGC_text_detector/raw/191c98291f204cae3e52db965ab0f944c9005067/app.py

HC3 attribution: Biyang Guo et al., How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection (2023). Dataset license: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/. The derived fixture and HC3 excerpts retain this license. The released HC3 detector was trained on the entire HC3-Chinese dataset without a held-out split. These samples therefore cannot serve as an independent accuracy benchmark. Exact revisions, answer indices and content hashes are in smoke.jsonl.

上游同时要求遵守来源数据自身限制，参见[来源许可表](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection#dataset-copyright)。本次新采集仅选表中许可已明确的 `open_qa` 和 `psychology`，没有新增 `baike` 样本。重新运行 `prepare_samples.py` 只生成 smoke 夹具，不覆盖本索引。

## 用户指定的英文学术样本

[english-academic-20261003/README.md](english-academic-20261003/README.md)保存用户指定chat的10篇原文PDF/LaTeX、30份生成版本、75处生成段及抽取正文/位置映射。整个批次Git排除；不推断公开转载许可。来源标签在检测前固定，按整篇文献划分拟合/门限校准/测试。低误报适配与未达到检出目标的结果见[验证记录](../dev_doc/context/english-calibration-validation.md)。
