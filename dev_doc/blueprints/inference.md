# 推理模块

实现：[engine.py](../../aigc_detector/engine.py)、[catalog.py](../../aigc_detector/catalog.py)、[modeling.py](../../aigc_detector/modeling.py)。

`Detector` 持有本地模型目录、目标设备、tokenizer/model 缓存和 RLock。`auto` 选择可用 CUDA，否则 CPU。加载时验证 manifest 和文件大小，禁止远程代码，使用 PyTorch 安全的 weights-only 读取方式。

当前登记22个分类器：默认中文组为原十模型，可选英文组为 HC3、Desklib、Fakespot 及共用的 bibbbu。标准 BERT/RoBERTa/XLM-RoBERTa 使用 `AutoModelForSequenceClassification`；`loader=bert_mlp` 的两个 EnsemJudge 权重使用本地 `BertMlpClassifier`。其 classifier 为 Linear → ReLU → Dropout → Linear，输入仍是 BERT pooler 经 dropout 的输出；隐藏尺寸和 dropout 来自原 config。Desklib 使用本地 `DesklibAIDetectionModel`：DeBERTa-v2 结构、attention mask 平均池化、单输出 Linear；本地适配设置 num_labels=1，不改写下载的 config。加载信息中出现 missing/unexpected/mismatched keys 或 error_msgs 即拒绝推理，不使用随机初始化头。旧OpenAI两个检查点由本地LegacyRobertaPoolerClassifier完整加载pooler，分类仍使用CLS头；2026-10-03补充的五个英文分类器显式选择，均取AI类0。所有分类器沿用float32，不应用上游可选温度校准。

正文由 fast tokenizer 一次编码，保留 offsets。窗口正文容量 = 模型窗口上限减去特殊符号数（普通512，MAGE 4096，CoCoDet 2048），默认重叠 64。最后一个窗口结束于完整 token 数，不多生成空尾窗。`token_start/end` 和 `char_start/end` 都为半开区间，后者可对原输入直接切片。

0.2.0 起 `_score_texts` 将多个片段的窗口组成同一批推理任务，`_probabilities` 按 batch_size 流式计算；`detect` 和分段 `annotate` 共用该实现。默认batch_size=8，MAGE和CoCoDet每批至多一窗；显存压力下释放最久未用的缓存，为常规前向预留2GiB、MAGE预留4GiB，仍不足时报错并提示CPU。`annotate` 先按自然边界切正文，各片段独立编码，详情见 [annotation.md](annotation.md)。

二分类窗口按catalog的ai_label_index选择softmax AI类；Desklib 对单 logit 用 sigmoid。二分类/单输出每批验证形状为[batch,2]/[batch,1]；DetectAIve/CoCoDet分别为[batch,4]/[batch,3]。设源 token i 被 c(i) 个窗口覆盖，窗口 j 的权重为其中各 token 的 1/c(i) 之和。窗口分数按此权重平均，所有窗口权重之和等于正文 token 数。此聚合是本包装器的工程选择，未经过论文级校准。语言提示仅针对实际选择的单语言模型；共享多语言模型不会单独触发中文含量提示。

扩展模型时：登记官方来源和标签，验证是否仍为二分类及 512 长度体系；若是其他体系，应实现对应适配，不能仅改仓库名称。同步 API 的模型数量约束、测试和文档。

统计后端由内置AutoModelForCausalLM加载（bfloat16、local_files_only、trust_remote_code=False、weights_only=True），顺序释放并按预算GPU/CPU卸载。detect_batch最多1024条/500,000字符，每条上下文、覆盖及归约独立，前向仍batch=1；detect委托单条batch。Neo直接消费CPU logits归约；Falcon只缓存参考logits，评分logits流式计算两个后端。临时目录必须解析在.runtime/statistical-logits下才清理，完成和异常均释放。默认512/64覆盖所有token，首token不参与下一词评分；逆重叠统计归约后再算公式。独立入口只给raw_score和方向，不给概率/门限；扩展融合可锁定这些输出，当前英文v2未选入统计信号。真实两输入/三个后端与旧缓存算法逐项一致；公式及适配来源见[补充模型记录](../context/english-additional-validation.md)，批量/重校准验证见[本轮记录](../context/english-recalibration-validation.md)。

0.9.0新增FollowSci/NTNU标准二分类器、DetectAIve四分类及本地CoCoDetModel三分类。后两者按catalog的num_labels验证形状、ai_label_indices求和，全部softmax类别在窗口和文档保存为class_scores。CoCoDet严格加载三个辅助头，CLS表示/分类权重归一化点积×30；逐窗显式添加CLS/SEP，避免generic fast tokenizer的prepare_for_model遗漏模板。关闭reference compilation，使用已有SDPA，作者默认2048token、batch至多1。见[接入记录](../context/english-specialized-validation.md)。
