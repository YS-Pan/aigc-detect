# 标注与报告模块

实现：[annotation.py](../../aigc_detector/annotation.py)、[reporting.py](../../aigc_detector/reporting.py)。用户契约见 [长文指南](../../user_doc/long-text-annotation.md)。

## 分段与推理

`segment_spans` 返回对完整原文的无重叠半开区间，优先选择句末或换行边界，限制片段大小，处理短尾片并避免拆开 CRLF。每片独立编码，由 `Detector._score_texts` 跨片批量推理。默认融合策略要求 400/120 字符分片；字符位置不经过重排或换行规范化。

## 分类

`annotate_text` 默认调用 `fusion.policy_settings` 和 `decide_fusion`，要求完整十模型 roster、profile 指定分片尺寸与每个模型固定 revision。单模型原始档位仍按 0.90 / 0.70 记录；综合三档按 profile 中的门限判定。报告保留每个原始分数、综合分、logit 和模型有符号贡献。

`sensitive-any-model-v1` 可显式选择旧 `decide` 规则：任一模型达到高分线即为 strong，否则任一达到复核线即为 possible，否则为 other。该策略默认原始门限 0.90 / 0.70，可调整；门限不按 smoke 数据标定。不可编码内容返回 null 分数并提示。

`annotate_text` 记录模型 revision、规则版本、完整覆盖、原文/片段 hash、行号、各模型窗口证据和三档汇总。新 schema 为 `aigc-annotation/1.1`。原始 detect 接口继续提供其滑动窗口分数。

## 保存

`save_annotation` 仅依赖标准库，先核验原文 hash、片段 hash 和完整无重复覆盖，再创建新的报告目录。写入 source、JSON、JSONL、HTML 后才写 COMPLETE 标记；禁止覆盖已有目录。

HTML 内嵌 JSON 中的 `<` 等字符被转义，原文通过 DOM `textContent` 展示，避免源文本成为可执行脚本。报告不依赖外部资源。字段位置含义与用户文档保持一致，不能把 UTF-16 单元或字节位置当成 Python Unicode 字符位置。

## 验证

契约测试见 `tests/test_annotation.py`，真实十万字符检查见 `scripts/test_annotation.py`，历史证据见 [长文验证](../context/long-text-validation.md)。这些旧验证只证明相应运行路径，不证明新融合 profile 的独立泛化。
