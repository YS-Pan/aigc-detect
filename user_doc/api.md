# API 约定

默认地址 `http://127.0.0.1:8766`，只监听本机，单 worker。

## HTTP

| 方法与路径 | 用途 |
|---|---|
| `GET /health` | 服务状态、运行设备、已安装模型 |
| `GET /v1/models` | ID、来源、revision、安装和加载状态 |
| `POST /v1/detect` | 一条文本，选定模型或所选语言组 |
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

`language` 默认 `zh`，可显式传 `en`；`models` 省略或为 `null` 时运行所选语言组（中文 10 个、英文 4 个）。显式数组优先于语言组，最多22项，空数组或未知模型会报错。上例仅选择原有四个模型。新增 ID 为 `ensemjudge-bert`、`ensemjudge-roberta`、`anx-bert-zh`、`paperpass-v3`、`cmj-zh`、`bibbbu-xlm-roberta`，英文新增 ID 为 `hc3-en`、`desklib-en`、`fakespot-en`，英文组也复用 `bibbbu-xlm-roberta`。另有 `openai-base-en`、`openai-large-en`、`piratexx-en`、`mage-en`、`radar-vicuna-en` 五个显式补充英文分类器。完整清单、`model_groups`、`extra_english_models`、`specialized_english_models` 与统计后端安装状态由 `GET /v1/models` 提供。统计推理通过CLI/Python独立入口调用，不接受分类接口的 `models` 参数。批量入口把 `text` 改为 `texts: ["第一段……", "第二段……"]`，响应为 `{"items": [...]}`。

限制：单条 200,000 字符；一批 1–32 条且总计不超过 500,000 字符；重叠 0–256 token。空白文本、错误类型、未知字段或不合法选项返回 HTTP 422；模型缺失返回 503。服务故障返回 500。不要把非 200 的响应当作检测结果。

## 原始检测返回字段

- `schema_version`：当前 `1.0`。
- `input.characters`、`input.sha256`：原输入长度与 UTF-8 SHA-256；默认不回显正文。
- `device`：实际使用的 `cpu` 或 `cuda`。
- `results[]`：每个模型独立结果，含官方 `repo_id` 和已安装 `revision`。
- `ai_score`：分块softmax的AI类或声明类别之和（Desklib为单logit的sigmoid），按 token 覆盖权重平均；范围 0–1，尚未校准。
- `model_selection`：所选 `language` 与是否使用显式模型列表。
- `score_min` / `score_max`：该模型各分块的最小/最大分数。
- `token_count` / `chunk_count` / `truncated`：完整输入的 token 数、窗口数、是否截断。正常完成始终 `truncated=false`。
- `chunks[]`：分块分数及 `token_start/end`、`char_start/end`。所有位置为 **0 起点、左闭右开**。字符位置按 Python Unicode 字符索引，非 UTF-8 字节或 JavaScript UTF-16 单元。
- `aggregation_weight`：每个原 token 在覆盖它的窗口之间平均分摊 1 单位权重，再对窗口求和。全部窗口权重之和等于总 token 数。
- `warnings`：短文本、中文含量低等提示。它不改变原始分数。

除MAGE为4096、CoCoDet为2048 token外，分类模型窗口含特殊符号最多512 token，因此常规正文每窗通常最多510 token。MAGE每批至多一个窗口。分块按各模型 tokenizer 独立计算；不同模型分数及窗口不保证可直接逐项对应。

二分类器返回未经校准的 AI 类 softmax 分数，Desklib 返回单 logit 的 sigmoid；不应用 AnxForever 模型卡中的温度校准或 paperpass-v3 的分档线。EnsemJudge 两个 ID 各自运行一个分类器，不包含其完整集成投票方案。AnxForever 和 bibbbu 训练时主要使用 256 token，本接口按其配置支持的 512 token 覆盖输入；长文分块属于本项目的工程策略。

英文调用与安装见[英文指南](english-detection.md)。标注省略policy和分片参数时按显式language选默认profile：中文v2为400/120、十模型；英文 `english-academic-20261003-v2` 为1600/400、新增五分类器。P07/P08重复评估人工0/19误标、生成3/12命中、混合生成字符覆盖20.38%，仍有严重漏检且不是新的独立测试；policy.validation_warning和interpretation保留限制。旧英文v1和旧原始规则可显式使用。健康检查status按中文原始组判断；英文原始detect就绪看model_groups.en.ready，默认annotate看annotation_profiles.en.ready。后者返回实际profile/hash/名单/分片，并校验安装版本；/v1/models也返回annotation_profiles。

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
  "policy": "thesis-cnki-20260929-v2",
  "segment_chars": 400,
  "min_segment_chars": 120,
  "include_text": true
}
```

单条仍限 200,000 字符。默认策略 `thesis-cnki-20260929-v2` 要求完整十模型、片段大小为 400/120 字符，且 profile 与当前模型 revision 一致。默认门限从 profile 读取；可选传入 `high_threshold` 和 `possible_threshold` 覆盖，响应会记录覆盖状态。v2 的高分门限作用于强信号分，复核门限作用于另一条独立分数，两者分别在 0–1 范围内校验，不要求大小有序。旧融合策略 `thesis-cnki-20260929-v1` 仍使用单条融合分数，沿用有序门限规则；`sensitive-any-model-v1` 支持模型子集和自定义片段，默认原始门限为 0.90/0.70，也要求复核门限小于高分门限。非法策略或规则返回 422；模型缺失或版本不匹配返回 503。HTTP 只返回结果，不接受任意服务端保存路径。

响应 schema 为 `aigc-annotation/1.2`，含 `input`、`models`、`policy`、`segmentation`、`summary` 和完整 `segments[]`。`policy` 记录 profile 内容和 SHA-256。融合策略的每段 `decision` 保留 `ensemble_score`、`ensemble_logit` 和各模型有符号 `contributions`；v2 中这些字段表示强信号头，并增加独立的 `review_score`、`review_logit` 和 `review_contributions`。v2 的 `feature_contributions[]` 每项含 `models`、`value`、`coefficients` 和 `logit_contributions`；特征项可以是单模型线性项、平方项或两模型交互项，系数和 logit 贡献分别记录强信号头与复核头。每模型贡献表将两模型交互项的贡献平均分到两个模型，便于核对总和；它是精确的计算分摊，不表示因果影响。`model_scores` 原样保留每个分类器的独立分数和窗口证据。筛查分不是概率或来源证明。规则、片段字段和保存说明见 [长文指南](long-text-annotation.md)。Python 为 `Detector.annotate(...)`，轻量客户端为 `Client.annotate(...)`，参数与 HTTP 对应。

原始 `detect` 的滑动窗口可能跨越多个标注片段；`annotate` 的片段独立计算。因此两个入口对同一整篇正文的结果用途不同，不应要求其全文分数和分档统计相等。

当前软件版本为0.10.0。中文默认v2 profile使用schema `aigc-score-fusion/2.0`，需要固定revision的十模型；保留v1强信号线性头，复核头可含平方/交互项。英文默认v2为schema `aigc-score-fusion/3.0`，固定新增五分类器及score_transforms。profile门限保存在profile和报告中。旧thesis-v1、英文v1或原始规则可显式复现；参考拟合及留出重拟合诊断不等于独立泛化验证。

显式 `policy="english-academic-20261004-v3"` 可使用schema4.0非线性实验配置，同新五分类器/1600/400，4个tanh隐藏单元、29参数；两个头共享函数、门限不同，报告提供隐藏单元及每输入贡献。优化与648候选比较已完成，但复测1/12单独生成段命中、混合覆盖0，低于线性v2，默认不变。旧profile文件保持原hash，完整名单/revision/分片仍严格匹配。见[英文非线性验证](../dev_doc/context/english-nonlinear-validation.md)。

命令行另提供recombine，从已有完整报告包读取report.json和source.txt，根据缓存的模型原始输出重新套用兼容策略，不加载模型。输入必须含有效COMPLETE；输出目录必须不存在。HTTP没有重组端点。

英文v1为schema2.0、旧四分类器；v2为schema3.0、新五分类器。两个头共用同一已拟合函数、使用不同保守门限。扩展schema支持分类器logit及固定配置的统计raw_score变换，但最终v2未选统计后端。运行时只读profile锁定信号，文本、作者、长度、文件身份不进入融合。若统计信号被选入，model_scores保留raw_score/kind/assets/方向，ai_score为null，不按0.90/0.70单独分档；statistical_execution保存配置和加载证据。贡献新增raw_score、score_transform与standardized_value，不把统计量当概率。

policy及分片参数省略/null时读取默认profile，显式值仍校验完整名单、分片和revision。recombine省略policy时，已有英文融合缓存保留自身策略；英文旧任一规则缓存选择当前默认v2，仍须满足契约。旧四模型或400/120缓存不能套新五模型/1600/400配置，须重新推理；可显式选择旧英文v1重组兼容缓存。

0.9.0新增显式ID：`followsci-bert-en`、`roberta-academic-en`、`llm-detectaive-en`、`cocodet-en`。后两者为多分类，`ai_score`分别汇总全部非纯人工类（含机器润色）和Mix+AI内容类。`class_scores`保存按覆盖权重聚合的每类输出，`ai_score_classes`列明参与求和的类别，`score_semantics`说明含义；`chunks[].class_scores`保存各窗口类别分数。标注时字段也保留在`segments[].model_scores[ID]`，`windows[]`保留逐窗类别输出。它们不是来源概率或文本组成比例，尚未进入默认融合。来源、许可与示例见[英文指南](english-detection.md)。
