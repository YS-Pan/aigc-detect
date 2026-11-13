# 英文公开样本，2026-10-02

按用户要求在代码接入后联网搜索并采集。共 20 条独立输入：12 条来源标为人工，8 条由发布者明确标为 ChatGPT。抽取规则在检测前固定，完整结果位于 [英文验证目录](../../reports/english-expansion-20261002/english-test/summary.json)，逐条全文原始分数见 [scores.csv](../../reports/english-expansion-20261002/english-test/scores.csv)。本批的原始文本、标签及来源不随分数改动。

## 来源与规则

| 来源 | 条数 | 来源标签与许可 |
|---|---:|---|
| [Jane Austen, Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) | 1 | 1813 年作者作品，human；原作公有领域，保存 Gutenberg 来源与条款 |
| [Charles Darwin, On the Origin of Species](https://www.gutenberg.org/ebooks/1228) | 1 | 1859 年作者作品，human；同上 |
| [Mary Shelley, Frankenstein](https://www.gutenberg.org/ebooks/84) | 1 | 1818 年作者作品，human；同上 |
| [Henry David Thoreau, Walden](https://www.gutenberg.org/ebooks/205) | 1 | 1854 年作者作品，human；同上 |
| [HC3 English wiki_csai](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | 16 | 八组同题 human_answers / chatgpt_answers 原始回答；发布层 CC BY-SA 4.0，Wikipedia 上游 CC BY-SA；保留 Guo et al. (2023) 署名 |

Gutenberg 从每篇正文的固定唯一锚点截取 3000–4000 字符，以上限前最后一个句末结束；原有空白、CRLF 和标记均保留。HC3 固定 revision `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`，按文件顺序选择前八组两类均有 300–10000 字符合格回答的记录，每类取第一个合格答案。完整答案原样输入，不输入问题或任何标签、作者及文件身份信息。

HC3 署名：Biyang Guo et al., *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023)。[发布者版权表](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection#dataset-copyright)明确列出 wiki_csai 的 Wikipedia / CC BY-SA 来源。本批没有使用其许可不明确的英文 finance、medicine 或 open_qa 分支。

本批保存 `samples.jsonl`、独立 `texts/`、原始 `sources/`、`collection.json` 及 `COMPLETE.json`。每条记录包含来源 URL、原快照 hash、字符范围或行号/字段/答案索引、许可、来源标签和 UTF-8 hash。`verify` 离线将逐条文本对回原快照与来源字段。Gutenberg 当前快照没有稳定 revision，以原始字节 hash 锁定。

本次取得的四个 Gutenberg TXT 响应均在作品末尾前结束，未保存完整电子书；所选完整正文节选均位于已取得范围内。原始返回字节保留用于对回位置与hash，`complete_work_archived=false`明确记录这一限制。本批是检测节选归档，不能用于复原整本作品。

## 复现

```powershell
& '.\.venv\Scripts\python.exe' scripts/english_samples.py verify
& '.\.venv\Scripts\python.exe' scripts/test_english.py --report-dir reports/新的英文测试目录
```

重新采集须用新目录，不覆盖本批：

```powershell
& '.\.venv\Scripts\python.exe' scripts/english_samples.py collect --directory samples/新的英文样本目录
```

来源网页可能更新；新批次的 hash 可能不同。`--cached-sources` 可复用本次已下载的原始来源快照，不能用修改后的文本替代来源。

## 证据边界

已发布的 hc3-en 使用全部 HC3 数据训练，没有留出，因此这里的八组 HC3 是明确的训练重叠例子。历史著作也可能出现于检测器或其基座训练中，未审计。AIGC 仅覆盖早期 ChatGPT，不代表现代生成模型、改写文本或任意英文文体。

原始分数 ≥0.90 的描述性计数：HC3 / Desklib / Fakespot / bibbbu 在人工文本中分别为 0/12、0/12、1/12、12/12，在 ChatGPT 文本中均为 8/8。0.70 门限下，人工文本高分计数分别为 0/12、1/12、2/12、12/12。英文任一模型分段规则在包含 bibbbu 时会严重误报；全文聚合分数计数和分段标注计数不是相同指标。没有据此拟合门限、融合或宣称准确率。详见[验证说明](../../dev_doc/context/english-expansion-validation.md)。
