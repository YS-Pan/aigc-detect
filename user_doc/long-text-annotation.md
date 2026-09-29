# 长文三档标注与保存

## 如何处理十万字

单次最多接受 **200,000 Python Unicode 字符**，包括标点、空格和换行。十万字正文通常可一次提交；超过限制时由调用方按章节拆分并保存章节到源文件的映射。

`annotate` 先按中文句末和换行边界切片，默认每片最多 400 字符、尽量不少于 120 字符；长句到上限时强制切开，短尾片尽量合并或重新分配。片段连续、无重叠，完整保留原文。

每片独立调用所选模型，跨片段按 8 个模型窗口一批计算。若配置的大片段超过模型 512 token 上限，内部仍用重叠窗口覆盖全文。显存主要由模型和批大小决定；耗时随片段数量增加。

片段是标注精度单位。标签不表示能精确找到一句话内的 AI 生成起止位置。[十万字符实测与限制](../dev_doc/context/long-text-validation.md)提供实际耗时和完整性证据。

## 默认策略与三档含义

默认策略为 **thesis-cnki-20260929-v1**：使用全部十个模型的原始分数，通过仅含模型分数的正则化 logit stacking 计算综合筛查分，再使用融合 profile 中的两条门限生成三档。门限由 profile 提供；这里不预先写具体值。

计算步骤是：将每个原始分数裁到 profile 规定的概率范围内，转为 logit，按 profile 的中心和尺度标准化（尺度下限 0.5，标准化值裁到 [-5, 5]），乘以含正、负值的模型系数求和并加截距，最后经 sigmoid 得到综合筛查分。该分数不是概率估计或来源证明。

默认策略要求十个模型全部参与、片段长度为 400/120 字符，并要求当前模型 revision 与 profile 逐项相同。本版本随附已拟合的 profile；如果部署包缺少该文件，或本地权重版本不符，默认策略会失败。完整拟合背景与限制见[跨工作区说明](dissertation-integration.md)及[融合模块说明](../dev_doc/blueprints/fusion.md)。

| 标签 | 默认策略的判定 |
|---|---|
| strong_signal / 明显 AIGC 特征 | 综合筛查分达到 profile 的高分线 |
| possible_signal / 较可能 AIGC | 未达到高分线，但达到 profile 的复核线 |
| other / 其他 | 未达到复核线，或片段不可完整计算 |

每份报告保留各模型原始分数、原始分档、综合分、logit、各模型的有符号贡献、profile 哈希和版本。新策略下各模型的原始档位仍按 0.90/0.70 展示，但综合标签由 profile 门限决定。

旧规则 sensitive-any-model-v1 可显式选择，兼容模型子集和自定义片段大小；默认原始门限为 0.90/0.70，任一所选模型达到门限即可触发。两套策略的分数和标签都只是复核线索。“其他”不证明人类写作，标记字符数也不是 AI 写作比例。不可编码片段会标为 no_scorable_tokens，分数为 null。参考样本拟合与独立留出检验的区别见[跨工作区说明](dissertation-integration.md)。

## 命令行：直接生成报告包

以下命令用本项目示例；正式任务替换 `--file`，并按调用方工作区规则选择输出目录：

```powershell
$reportDir = 'D:\project\20260928 aigc detect\reports\annotation-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& 'D:\project\20260928 aigc detect\.venv\Scripts\python.exe' -m aigc_detector annotate `
  --file 'D:\project\20260928 aigc detect\samples\demo-zh.txt' `
  --report-dir $reportDir
```

默认使用完整十模型 roster 和 profile 门限。新策略要求固定 400/120 切段，且所用模型 revision 与 profile 完全一致。要运行兼容旧规则或指定模型子集，显式传入 `--policy sensitive-any-model-v1`；其默认门限为 0.90/0.70。也支持 `--text`、UTF-8 `--stdin` 和 `--device cpu`。

重用已有完整报告的原始分数重新套用策略，无需再次推理：

~~~powershell
& 'D:\project\20260928 aigc detect\.venv\Scripts\python.exe' -m aigc_detector recombine `
  --input-dir 'D:\project\20260928 aigc detect\reports\previous-annotation' `
  --report-dir 'D:\project\20260928 aigc detect\reports\recombined-annotation'
~~~

重组需要完整报告包及其 `source.txt`，会核对 COMPLETE 标记、输入 hash、片段和模型版本，并写入一个新的报告目录。CLI 不覆盖已有目录；新融合策略不运行 GPU 推理。

输入支持 UTF-8/BOM 文本，保留 CRLF，移除文件开头的 UTF-8 BOM。Markdown/LaTeX 源码原样参与检测；如需只测正文，应先由调用方提取正文。

## 保存内容

必须使用新的报告目录，现有目录不会被覆盖：

```text
报告目录/
  source.txt       精确输入快照，UTF-8 无 BOM，保留原换行
  report.json      完整输入元数据、模型版本、规则、汇总及逐段记录
  segments.jsonl   每行一个片段，可流式读取
  report.html      无需联网的交互高亮报告
  COMPLETE.json    全部文件写入成功后的完成标记
```

读取已有报告包时先检查 `COMPLETE.json`，避免把写入中断的目录当成完整结果。

HTML 提供综合结果或单模型视图、三档筛选和标记跳转；点击片段显示行号、字符区间、各模型原始分数及融合贡献（若使用融合策略）。既有长文验证记录只证明其所用旧规则及模型集合的运行通路，不证明新融合策略的泛化效果；详见[验证记录](../dev_doc/context/long-text-validation.md)。

## 结构化字段

`report.json` 的全局元数据：

- `input`：全文 hash、字符数、位置单位与约定。
- `models`：实际参与的 ID、官方仓库和固定 revision。
- `policy`：策略 ID、实际门限、校准状态。新策略另记录 profile 内容及 SHA-256、完整模型 roster、版本约束、`reference_fitted=true` 和 `calibrated=false`。
- `segmentation`：分片配置、片段数、完整覆盖标记。
- `summary`：每档的片段数和标记字符数；不是 AI 写作比例。
- `segments[]`：与 JSONL 相同的逐段记录。

每片保存：

- `id/index`、`text`、`sha256`。
- `char_start/end`：相对于 `source.txt`，0 起点、左闭右开、Python Unicode 字符索引。
- `line_start/end`：1 起点行号。
- `decision`：三档标签、最高原始分数、模型分歧和可编码状态。融合策略另含 `ensemble_score`、`ensemble_logit`、各模型有符号 `contributions`、缺失模型和判定依据。
- `model_scores`：每模型独立分数、档位、token 数及内部窗口证据。窗口字符位置相对于完整原文；窗口 token 位置相对于该片段。
- `warnings`：短片、中文较少、不可编码等提示。

## HTTP / Python 保存

服务启动后，使用本项目虚拟环境运行：

```python
from pathlib import Path
from aigc_detector.client import Client
from aigc_detector.reporting import save_annotation

source = Path(r"D:\project\20260928 aigc detect\samples\demo-zh.txt")
text = source.read_bytes().decode("utf-8-sig")
report = Client(timeout=600).annotate(text)
paths = save_annotation(
    report, text,
    r"D:\project\20260928 aigc detect\reports\my-new-annotation",  # 必须尚不存在
    source_path=source,
)
print(paths)
```

跨环境也可以仅用标准库向 `POST http://127.0.0.1:8765/v1/annotate` 发送 JSON；请求与错误说明见 [API 文档](api.md)。HTTP 只返回结果，由调用方决定保存位置。`include_text=false` 可省略响应片段正文，调用 `save_annotation` 时会根据传入原文补齐并核验 hash。

这是同步接口，服务串行使用 GPU，客户端默认等待最多 600 秒。标注响应 schema 为 `aigc-annotation/1.1`；CLI、Python 和 HTTP 的默认 policy 相同。软件版本为 0.3.0。profile 门限保存在报告中，不在本指南预写具体数值。显式使用旧策略可复现 0.90/0.70 的任一模型规则。参考数据上的拟合和留出重拟合诊断不等于独立泛化验证。服务尚无后台任务队列、断点续跑或进度查询。
