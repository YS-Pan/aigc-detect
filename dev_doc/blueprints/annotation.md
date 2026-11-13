# 标注与报告模块

实现：[annotation.py](../../aigc_detector/annotation.py)、[reporting.py](../../aigc_detector/reporting.py)。用户契约见 [长文指南](../../user_doc/long-text-annotation.md)。

## 分段与推理

`segment_spans` 返回对完整原文的无重叠半开区间，优先选择句末或换行边界，限制片段大小，处理短尾片并避免拆开 CRLF。每片独立编码，由 `Detector._score_texts` 跨片批量推理。默认融合策略要求 400/120 字符分片；字符位置不经过重排或换行规范化。

language=zh使用natural-boundaries-v1，language=en使用英文句末/换行/词间空白边界english-natural-boundaries-v1。英文默认english-academic-20261003-v2，从profile读取1600/400和新五分类器名单；旧英文v1可显式复现，旧任一规则仍默认400/120。英文实际漏检和重复评估限制随policy/interpretation保存。

## 分类

annotate_text调用policy_settings/decide_fusion，要求对应profile完整名单、指定分片及固定revision。分类器原始档位按0.90/0.70展示，最终融合先判强、再判疑似。扩展schema的统计信号在分类器缓存释放后由statistical_passage_scores顺序计算，记录raw_score、方向、assets和statistical_execution；不单独分档。最终英文v2只选五分类器。报告保留全部原始输出、双头logit、模型及特征贡献，schema仍为aigc-annotation/1.2。

`sensitive-any-model-v1` 可显式选择旧 `decide` 规则：任一模型达到高分线即为 strong，否则任一达到复核线即为 possible，否则为 other。该策略默认原始门限 0.90 / 0.70，可调整；门限不按 smoke 数据标定。不可编码内容返回 null 分数并提示。

`annotate_text` 记录模型 revision、规则版本、完整覆盖、原文/片段 hash、行号、各模型窗口证据和三档汇总。新 schema 为 `aigc-annotation/1.2`。原始 detect 接口继续提供其滑动窗口分数。

## 保存

`save_annotation` 仅依赖标准库，先核验原文 hash、片段 hash 和完整无重复覆盖，再创建新的报告目录。写入 source、JSON、JSONL、HTML 后才写 COMPLETE 标记；禁止覆盖已有目录。

HTML 内嵌 JSON 中的 `<` 等字符被转义，原文通过 DOM `textContent` 展示，避免源文本成为可执行脚本。报告不依赖外部资源。字段位置含义与用户文档保持一致，不能把 UTF-16 单元或字节位置当成 Python Unicode 字符位置。

v2 HTML 在片段上显示强信号分和复核分，详情表分别显示两头贡献；展开特征项可查看线性、平方和交互贡献。每模型贡献把二元交互在两个模型间均分，只用于核对计算，不作为因果重要性解释。

HTML标题按报告语言、信号数量按实际名单显示；详情表区分原始分类分数与统计量，统计信号不进入单模型分档下拉框。页内保留实际validation_warning，不把参考拟合通称CNKI或固定十模型。

显式英文v3读取schema4.0的固定名单和非线性函数，最终为新五分类器/1600/400。HTML显示同一分数、两条门限以及核函数/隐藏单元贡献；模型贡献分摊方法按函数类别解释，不沿用多项式交互均分说法。v3重复评估1/12生成段命中、混合覆盖0且低于v2，限制随policy返回，默认仍是线性v2；见[实验验证](../context/english-nonlinear-validation.md)。

## 验证

契约测试见 `tests/test_annotation.py`，真实十万字符检查见 `scripts/test_annotation.py`，历史证据见 [长文验证](../context/long-text-validation.md)。这些旧验证只证明相应运行路径，不证明新融合 profile 的独立泛化。

0.9.0显式选择多分类模型并用旧规则时，片段model_scores保留class_scores、ai_score_classes、score_semantics，各windows保留类别输出，重组同样保留。CoCoDet聚合Mix+AI、DetectAIve聚合全部非纯人工类（含机器润色）；默认profile不变，见[英文接入记录](../context/english-specialized-validation.md)。
