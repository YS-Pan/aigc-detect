# 六个分类器扩展验证

日期：2026-09-29（Asia/Shanghai）。用户确认只增加六个分类器，排除 LLM-Detector-Small-zh 及 EnsemJudge 完整大模型集成。

## 安装与适配

- 新增 `ensemjudge-bert`、`ensemjudge-roberta`、`anx-bert-zh`、`paperpass-v3`、`cmj-zh`、`bibbbu-xlm-roberta`。连同原有模型共十个，均接入 Python、CLI、HTTP 检测和分段标注。
- 新增 42 个资源文件、4,966,228,202 字节（约 4.97 GB）；十模型共 67 个资源文件、6,603,702,485 字节（约 6.60 GB）。大小为十进制，不含虚拟环境和报告。
- 下载先直连，Hugging Face 直连失败后使用环境已有代理。下载时比对上游 LFS SHA-256；下载后再次完整校验本地 manifest 的所有文件，全部通过。
- EnsemJudge checkpoint 的分类头是两层 MLP。本项目在本地适配层结构，从各自 HFL 基座补齐 tokenizer，保留权重及 tokenizer 的固定 revision 和逐文件来源。
- 加载时拒绝 missing/unexpected/mismatched keys 及其他权重加载错误。专门用缺少末层 bias 的小 checkpoint 验证会抛出 `ModelUnavailable`，不返回随机初始化分数。
- 使用原有 Python 3.13.11、PyTorch 2.11.0+cu128、Transformers 4.57.6；没有增加或升级依赖。bibbbu 在现有环境成功加载及推理；`pip check` 无冲突。

## 实际验证

| 检查 | 结果 | 证据 |
|---|---|---|
| 67 个资源文件完整 SHA-256 校验 | 全部通过 | [verify-models.txt](../../reports/model-expansion-20260929/verify-models.txt) |
| 依赖一致性 | 通过 | [pip-check.txt](../../reports/model-expansion-20260929/pip-check.txt) |
| 契约测试 | 27 项通过 | [pytest.txt](../../reports/model-expansion-20260929/pytest.txt) |
| 8 样本 × 10 模型 | 80 次试跑完成，其中新增模型 48 次 | [smoke-test.json](../../reports/model-expansion-20260929/smoke-test.json) |
| 断网推理 | 阻断 socket 连接后十模型完成加载和推理 | 同上 `offline_network_blocked` |
| 单窗口模型公式 | 十模型分数差均为 0 | 同上 `author_single_window_formula` |
| EnsemJudge 分类头与 tokenizer | 分类头张量公式差为 0；fast 与作者 BertTokenizer 编码一致 | 同上相应检查 |
| CPU/CUDA | 十模型均可运行；同一短样本最大分数差约 1.37e-6 | 同上 `cpu_cuda_score_difference` |
| 长文覆盖 | 所有 token 覆盖，无静默截断；BERT 系各 9 窗，XLM-RoBERTa 7 窗 | 同上 `long-window-coverage` |
| 真实 HTTP、分段标注、显式十模型列表、批量、错误状态 | 通过 | [deployment-test.json](../../reports/model-expansion-20260929/deployment-test.json) |
| 从非项目根目录运行 CLI | EnsemJudge BERT 和 XLM-RoBERTa 的 CPU 调用通过 | 同上 |
| NLPCC 补充样本 | 第一条人工与第一条 AI 标签 × 六个新模型，共 12 次完成 | [nlpcc-smoke.json](../../reports/model-expansion-20260929/nlpcc-smoke.json) |

GPU 为 NVIDIA GeForce RTX 4060 Ti。此次 smoke 进程的 PyTorch 峰值已分配显存为 6,470.5 MiB，不含所有驱动、桌面及其他进程占用。它不能代表任意长文任务的峰值。

原本运行的四模型服务已用工作区启停脚本重启；`127.0.0.1:8765` 健康检查及实际推理均确认十模型可用。历史四模型报告保留，本次报告存放在独立目录。

## 如何解释分数

这些是运行检查，不计算准确率，也没有调整分档线。HC3 样本可能与部分模型的训练数据重叠，且只有三组人工/AI 配对，不能用于性能排名。

在这三组 HC3 样本上，EnsemJudge 两个分类器的 AI 分数均低于 0.001；paperpass-v3 对三条 AI 样本的分数也低于 0.012。XLM-RoBERTa 对其中两条人工样本给出了高于 0.999 的 AI 分数。这些实际分歧保留在原始报告中，不能把“成功返回分数”解释为“判断正确”。

为核对 EnsemJudge 的低分现象，从固定 revision 的官方 NLPCC dev.json 按文件顺序取第一条人工和第一条 AI 标签样本，选择在推理前完成。两模型对该人工样本的分数约为 0.00028、0.00012，对 AI 样本约为 0.99997；这确认模型能够对不同输入给出不同分数，但不能用这两个样本证明泛化能力。数据来源、原始文件 SHA-256、行索引、样本文本和全部分数保存在本轮报告目录。完整分数表见 [测试摘要](../../reports/model-expansion-20260929/README.md)。

默认 `models=null` 现在包含十模型。敏感筛查按任一模型触发，因此新增模型可能增加标注；复现旧结果时应显式选择原来的四个模型。此前十万字符验证只覆盖原四模型，本轮未重复十万字符测试。

来源、数据覆盖和许可证说明见 [模型来源](model-sources.md)。测试过程中还出现了既有 FastAPI/Starlette 对 httpx 的弃用提示；实际路由检查通过，本次没有为消除提示升级依赖。
