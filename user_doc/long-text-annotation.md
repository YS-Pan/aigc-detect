# 长文三档标注与保存

## 如何处理十万字

单次最多接受 **200,000 Python Unicode 字符**，包括标点、空格和换行。十万字正文通常可一次提交；超过限制时由调用方按章节拆分并保存章节到源文件的映射。

默认 `language=zh` 时，`annotate` 先按中文句末和换行边界切片，默认每片最多 400 字符、尽量不少于 120 字符；长句到上限时强制切开，短尾片尽量合并或重新分配。片段连续、无重叠，完整保留原文。

每片独立调用所选模型，跨片段按 8 个模型窗口一批计算。若配置的大片段超过模型 512 token 上限，内部仍用重叠窗口覆盖全文。显存主要由模型和批大小决定；耗时随片段数量增加。

片段是标注精度单位。标签不表示能精确找到一句话内的 AI 生成起止位置。[十万字符实测与限制](../dev_doc/context/long-text-validation.md)提供实际耗时和完整性证据。

英文用 `--language en` 默认选择 `english-academic-20261003-v2`：新增五分类器全选、1600/400字符分片、固定revision。相同两篇论文的重复评估人工0/19误标、生成3/12命中、混合生成字符覆盖20.38%；不是新的独立验证，仍有严重漏检。旧英文v1可显式复现；`--policy sensitive-any-model-v1`仍默认400/120分片和原始0.90/0.70门限。中英文profile互相拒绝不同语言分片；命令、完整样本核算及缓存兼容边界见[英文指南](english-detection.md)。

## 默认策略与三档含义

默认策略为 **thesis-cnki-20260929-v2**：使用全部十个模型的原始分数，分别计算强信号分和复核分，再使用各自的 profile 门限生成三档。门限由 profile 提供；这里不预先写具体值。

两个头共用相同的分数预处理：将每个原始分数裁到 profile 规定的概率范围内，转为 logit，按 profile 中的中心和尺度标准化，再按 profile 规定的范围裁剪。强信号头沿用 v1 的线性规则；复核头通过正则化拟合，可包含线性项、单模型平方项和两个模型之间的交互项，实际项和系数保存在 profile 中。每头将系数乘以特征值后求和，加上截距，再经 sigmoid 得到各自的筛查分。两个分数刻画不同的判定目标，都不是概率估计或来源证明。

默认策略要求十个模型全部参与、片段长度为 400/120 字符，并要求当前模型 revision 与 profile 逐项相同。本版本随附已拟合的 profile；如果部署包缺少该文件，或本地权重版本不符，默认策略会失败。完整拟合背景与限制见[跨工作区说明](dissertation-integration.md)及[融合模块说明](../dev_doc/blueprints/fusion.md)。

| 标签 | 默认策略的判定 |
|---|---|
| strong_signal / 明显 AIGC 特征 | 强信号分 `ensemble_score` 达到 profile 的高分线 |
| possible_signal / 较可能 AIGC | 未达到强信号线，但独立复核分 `review_score` 达到 profile 的复核线 |
| other / 其他 | 未达到相应门限，或片段不可完整计算 |

先检查强信号头；只有它未达到高分门限时才检查复核头。因此即使两头都超过各自门限，也判为 `strong_signal`。v2 两个门限作用于不同分数，不要求 `possible_threshold < high_threshold`。覆盖门限时，两者分别需在 0–1 范围内；v1 和旧 `sensitive-any-model-v1` 仍要求单分数规则中的复核门限低于高分门限。

每份报告保留各模型原始分数、原始分档、强信号分、复核分、两头 logit 和逐模型贡献、profile 哈希和版本。v2 还提供 `feature_contributions`：每个特征列出涉及的模型、标准化特征值、两头各自的系数及 logit 贡献。两模型交互项在逐模型汇总表中平均分配给两个模型；这是用于核对计算的精确分摊，不是因果解释。模型的原始档位仍按 0.90/0.70 展示，但融合标签由两条 profile 门限决定。

旧融合规则 `thesis-cnki-20260929-v1` 可显式选用，使用单条线性融合分数和有序门限；旧规则 `sensitive-any-model-v1` 也可显式选择，兼容模型子集和自定义片段大小，默认原始门限为 0.90/0.70，任一所选模型达到门限即可触发。各策略的分数和标签都只是复核线索。“其他”不证明人类写作，标记字符数也不是 AI 写作比例。不可编码片段会标为 `no_scorable_tokens`，分数为 null。v2 拟合过程、当前诊断和证据边界见[非线性融合验证说明](../dev_doc/context/nonlinear-fusion-validation.md)及[v2 评估数据](../evaluations/thesis-cnki-20260929-v2/README.md)；v1 历史证据仍见[旧融合验证记录](../dev_doc/context/fusion-validation.md)和[v1 评估数据](../evaluations/thesis-cnki-20260929/README.md)。

## 命令行：直接生成报告包

以下命令用本项目示例；正式任务替换 `--file`，并按调用方工作区规则选择输出目录：

```powershell
$reportDir = 'D:\调用方工作区\reports\annotation-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
python -m aigc_detector annotate `
  --file 'D:\调用方工作区\正文.txt' `
  --report-dir $reportDir
```

默认使用完整十模型 roster 和 v2 profile 门限。v2 要求固定 400/120 切段，且所用模型 revision 与 profile 完全一致。要复现旧融合策略，传入 `--policy thesis-cnki-20260929-v1`；要运行兼容旧规则或指定模型子集，传入 `--policy sensitive-any-model-v1`，其默认门限为 0.90/0.70。也支持 `--text`、UTF-8 `--stdin` 和 `--device cpu`。

重用已有完整报告的原始分数重新套用策略，无需再次推理：

~~~powershell
python -m aigc_detector recombine `
  --input-dir 'D:\调用方工作区\reports\previous-annotation' `
  --report-dir 'D:\调用方工作区\reports\recombined-annotation'
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

HTML 提供双头融合结果或单模型视图、三档筛选和标记跳转；点击片段显示行号、字符区间、所有模型原始分数、强信号分和复核分及其贡献。展开特征项可核对线性、平方和交互项对两个 logit 的贡献。既有长文验证记录只证明当时所用规则及模型集合的运行通路，不证明 v2 的泛化效果；详见[长文运行验证](../dev_doc/context/long-text-validation.md)和[非线性融合验证说明](../dev_doc/context/nonlinear-fusion-validation.md)。

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

source = Path(r"D:\调用方工作区\正文.txt")
text = source.read_bytes().decode("utf-8-sig")
report = Client(timeout=600).annotate(text)
paths = save_annotation(
    report, text,
    r"D:\调用方工作区\reports\my-new-annotation",  # 必须尚不存在
    source_path=source,
)
print(paths)
```

跨环境也可以仅用标准库向 `POST http://127.0.0.1:8766/v1/annotate` 发送 JSON；请求与错误说明见 [API 文档](api.md)。HTTP 只返回结果，由调用方决定保存位置。`include_text=false` 可省略响应片段正文，调用 `save_annotation` 时会根据传入原文补齐并核验 hash。

这是同步接口，服务串行使用GPU，客户端默认等待600秒。响应schema为aigc-annotation/1.2，CLI/Python/HTTP默认policy相同，软件版本0.10.0。门限保存在profile和报告中。旧profile可显式复现，sensitive-any-model-v1保留0.90/0.70原始规则。参考拟合、重拟合留出诊断和已知论文重复评估不等于新的独立泛化验证。服务尚无后台队列、断点续跑或进度查询。

英文另有显式非线性实验v3，默认仍保留线性v2。v3同五模型/1600/400，报告展示隐藏单元贡献及解析积分梯度的模型贡献；参数优化已完成，但冻结复测检出下降。调用、全样本结果和缓存兼容见[英文指南](english-detection.md)与[实验验证](../dev_doc/context/english-nonlinear-validation.md)。

0.9.0四个专业英文／协作模型可显式按ID选择，并使用旧`sensitive-any-model-v1`标注，未进入默认v2融合。CoCoDet的Mix+AI与DetectAIve的非纯人工（含机器润色）口径不同；报告保留类别分数及解释。旧门限未对这些新模型校准，见[英文指南](english-detection.md)。
