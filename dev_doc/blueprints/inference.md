# 推理模块

实现：[engine.py](../../aigc_detector/engine.py)、[catalog.py](../../aigc_detector/catalog.py)、[modeling.py](../../aigc_detector/modeling.py)。

`Detector` 持有本地模型目录、目标设备、tokenizer/model 缓存和 RLock。`auto` 选择可用 CUDA，否则 CPU。加载时验证 manifest 和文件大小，禁止远程代码，使用 PyTorch 安全的 weights-only 读取方式。

当前登记十个分类器。标准 BERT/XLM-RoBERTa 使用 `AutoModelForSequenceClassification`；`loader=bert_mlp` 的两个 EnsemJudge 权重使用本地 `BertMlpClassifier`。其 classifier 为 Linear → ReLU → Dropout → Linear，输入仍是 BERT pooler 经 dropout 的输出；隐藏尺寸和 dropout 来自原 config。加载信息中出现 missing/unexpected/mismatched keys 或 error_msgs 即拒绝推理，不使用随机初始化头。新增分类器仍沿用 float32，不应用上游可选温度校准。

正文由 fast tokenizer 一次编码，保留 offsets。窗口正文容量 = 最多 512 token 减去特殊符号数，默认重叠 64。最后一个窗口结束于完整 token 数，不多生成空尾窗。`token_start/end` 和 `char_start/end` 都为半开区间，后者可对原输入直接切片。

0.2.0 起 `_score_texts` 将多个片段的窗口组成同一批推理任务，`_probabilities` 按 batch_size 流式计算；`detect` 和分段 `annotate` 共用该实现。默认 batch_size=8。`annotate` 先按自然边界切正文，各片段独立编码，详情见 [annotation.md](annotation.md)。

每个窗口 softmax 第 1 类产生 AI 分数。设源 token i 被 c(i) 个窗口覆盖，窗口 j 的权重为其中各 token 的 1/c(i) 之和。窗口分数按此权重平均，所有窗口权重之和等于正文 token 数。此聚合是本包装器的工程选择，未经过论文级校准。

扩展模型时：登记官方来源和标签，验证是否仍为二分类及 512 长度体系；若是其他体系，应实现对应适配，不能仅改仓库名称。同步 API 的模型数量约束、测试和文档。
