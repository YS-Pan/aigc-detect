# API 约定

默认地址 `http://127.0.0.1:8765`，只监听本机，单 worker。

## HTTP

| 方法与路径 | 用途 |
|---|---|
| `GET /health` | 服务状态、运行设备、已安装模型 |
| `GET /v1/models` | ID、来源、revision、安装和加载状态 |
| `POST /v1/detect` | 一条文本，选定模型或全部模型 |
| `POST /v1/detect-batch` | 多条文本，共用推理选项 |
| `POST /v1/annotate` | 完整长文的三档片段标注，含原文位置和规则 |
| `GET /docs`、`GET /openapi.json` | 交互文档、接口结构 |

单条请求：

```json
{
  "text": "待检测的中文正文……",
  "models": ["mpu-zh-v3", "mpu-zh-v3-short", "mpu-zh-v2", "hc3-zh"],
  "overlap_tokens": 64,
  "include_chunks": true
}
```

`models` 省略或为 `null` 时运行全部 10 个模型；显式数组最多 10 项，空数组或未知模型会报错。上例仅选择原有四个模型。新增 ID 为 `ensemjudge-bert`、`ensemjudge-roberta`、`anx-bert-zh`、`paperpass-v3`、`cmj-zh`、`bibbbu-xlm-roberta`，完整清单由 `GET /v1/models` 提供。批量入口把 `text` 改为 `texts: ["第一段……", "第二段……"]`，响应为 `{"items": [...]}`。

限制：单条 200,000 字符；一批 1–32 条且总计不超过 500,000 字符；重叠 0–256 token。空白文本、错误类型、未知字段或不合法选项返回 HTTP 422；模型缺失返回 503。服务故障返回 500。不要把非 200 的响应当作检测结果。

## 原始检测返回字段

- `schema_version`：当前 `1.0`。
- `input.characters`、`input.sha256`：原输入长度与 UTF-8 SHA-256；默认不回显正文。
- `device`：实际使用的 `cpu` 或 `cuda`。
- `results[]`：每个模型独立结果，含官方 `repo_id` 和已安装 `revision`。
- `ai_score`：分块 softmax 的 AI 类分数，按 token 覆盖权重平均；范围 0–1，尚未校准。
- `score_min` / `score_max`：该模型各分块的最小/最大分数。
- `token_count` / `chunk_count` / `truncated`：完整输入的 token 数、窗口数、是否截断。正常完成始终 `truncated=false`。
- `chunks[]`：分块分数及 `token_start/end`、`char_start/end`。所有位置为 **0 起点、左闭右开**。字符位置按 Python Unicode 字符索引，非 UTF-8 字节或 JavaScript UTF-16 单元。
- `aggregation_weight`：每个原 token 在覆盖它的窗口之间平均分摊 1 单位权重，再对窗口求和。全部窗口权重之和等于总 token 数。
- `warnings`：短文本、中文含量低等提示。它不改变原始分数。

模型窗口含特殊符号最多 512 token，因此正文每窗通常最多 510 token。分块按各模型 tokenizer 独立计算；不同模型分数及窗口不保证可直接逐项对应。

所有模型均返回未经校准的 AI 类 softmax 分数；不应用 AnxForever 模型卡中的温度校准或 paperpass-v3 的分档线。EnsemJudge 两个 ID 各自运行一个分类器，不包含其完整集成投票方案。AnxForever 和 bibbbu 训练时主要使用 256 token，本接口按其配置支持的 512 token 覆盖输入；长文分块属于本项目的工程策略。

## Python

进程内（使用本项目虚拟环境）：

```python
from aigc_detector.engine import Detector

detector = Detector(device="auto")  # 在长驻进程中复用，避免每次重载
result = detector.detect("需要检测的中文正文", models=["mpu-zh-v3"])
```

轻量 HTTP 客户端（只依赖 Python 标准库）：

```python
from aigc_detector.client import Client

client = Client(timeout=600)
result = client.detect("需要检测的中文正文")
```

进程内 API 的非法输入抛 `ValueError`；模型缺失抛 `ModelUnavailable`。HTTP 客户端会抛出 `urllib.error.HTTPError`，调用方应保留响应错误信息。模型推理由进程内锁串行执行；HTTP 可以接收多个请求，但不应为吞吐量增加 GPU worker 副本。

## 三档标注

`POST /v1/annotate` 接受以下字段：

```json
{
  "text": "需要逐段复核的完整中文正文……",
  "models": null,
  "policy": "thesis-cnki-20260929-v1",
  "segment_chars": 400,
  "min_segment_chars": 120,
  "include_text": true
}
```

单条仍限 200,000 字符。新默认策略要求完整十模型、片段大小为 400/120 字符，且 profile 与当前模型 revision 一致。默认门限从 profile 读取；可选传入 `high_threshold` 和 `possible_threshold` 覆盖，响应会记录覆盖状态。旧策略 `sensitive-any-model-v1` 支持子集和自定义片段，默认门限为 0.90/0.70。所有门限都需满足 `0 <= possible_threshold < high_threshold <= 1`。非法策略或规则返回 422；模型缺失或版本不匹配返回 503。HTTP 只返回结果，不接受任意服务端保存路径。

响应 schema 为 `aigc-annotation/1.1`，含 `input`、`models`、`policy`、`segmentation`、`summary` 和完整 `segments[]`。`policy` 记录 profile 内容和 SHA-256；融合策略的每段 `decision` 含 `ensemble_score`、`ensemble_logit`、各模型的有符号 `contributions` 和缺失模型列表；`model_scores` 原样保留每个分类器的独立分数和窗口证据。综合筛查分不是概率或来源证明。规则、片段字段和保存说明见 [长文指南](long-text-annotation.md)。Python 为 `Detector.annotate(...)`，轻量客户端为 `Client.annotate(...)`，参数与 HTTP 对应。

原始 `detect` 的滑动窗口可能跨越多个标注片段；`annotate` 的片段独立计算。因此两个入口对同一整篇正文的结果用途不同，不应要求其全文分数和分档统计相等。

当前软件版本为 0.3.0。部署需包含已拟合的融合 profile，并安装与其锁定 revision 相同的模型权重。profile 阈值保存在 profile 和报告中；参考集拟合不等于独立泛化验证。可显式选择旧策略继续运行原有敏感筛查。

命令行另提供 `recombine`，从已有完整报告包读取 `report.json` 和 `source.txt`，根据缓存的十模型分数重新套用指定策略，不运行 GPU 推理。输入必须含有效的 `COMPLETE.json`；输出目录必须不存在。HTTP 没有对应的重组端点。
