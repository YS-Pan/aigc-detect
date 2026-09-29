# 长文分段标注与结构化报告

完成日期：2026-09-28。版本：0.2.0。

## 问题和结果

原接口覆盖全文并返回窗口分数，但调用方仍需自行定位和分档。现在增加 `annotate` 流程：正文按自然边界形成完整、连续、无重叠片段，四模型跨片段批量推理，再按用户选择的“任一模型高分就突出”规则生成三档标注。

默认阈值 0.90 / 0.70 可配置，随报告保存并明确未校准。原 `detect` 分数接口继续保持其用途；未将官方试跑结果用于设定或优化门限。

## 接口与文件

- Python `Detector.annotate`、HTTP `POST /v1/annotate`、`Client.annotate`。
- CLI `annotate --file ... --report-dir ...`。
- `report.json`、`segments.jsonl`、`source.txt`、可切换模型/筛选分档的离线 `report.html`、成功完成标记 `COMPLETE.json`。
- 原文位置为 Unicode 字符半开区间，保留 CRLF、行号和 hash；报告目录必须新建以避免覆盖。
- 文档更新包括用户入口、API、跨工作区指南、架构、模块说明和术语。

## 验证

25 项契约测试、原 32 次真实模型回归、十万字符完整标注、离线检查及跨工作区 HTTP/CLI 调用均通过。实测十万字符 270 片段，四模型共 1,080 次判断，首次模型加载加推理约 9.3 秒。

完整证据和 HTML 视觉验证限制见 [长文验证记录](../context/long-text-validation.md)。
