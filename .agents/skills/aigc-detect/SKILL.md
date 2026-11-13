---
name: aigc-detect
description: 在其它工作区调用本机 AIGC 检测工具，检测中英文正文、生成长文三档标注和离线报告，或从已有报告重组筛查结果。适用于本地检测与结果复核；不用于自动改写正文、判定作者身份或维护模型。
---

# 跨工作区调用本地 AIGC 检测

常规任务仅凭本 skill 和命令/API 的实际返回即可完成，不必先读检测工作区的文档系统。调用方仍按自己的工作区约定选材料、提取正文、保存报告和作最终判断。仅在下文列出的特殊情况中定点阅读检测工作区文档。

## 本机入口与边界

- 使用安装了 `local-aigc-detectors` 0.10.0 及所需 extras 的 Python，命令可从任意工作目录运行。系统有多个 Python 时选择实际安装该包的解释器。
- 默认 HTTP：`http://127.0.0.1:8766`，只监听本机。HTTP 只返回 JSON，不接受服务端输出路径。
- 权重从用户选择的模型目录加载，不启用远程模型代码，不将待测正文提交给外部服务。缺少资源时，询问用户保存根，获得明确授权后使用 `python -m aigc_detector prepare`；可用 `--assets-dir` 选择已有模型目录。
- 输入、请求、样本和结果使用 UTF-8。私人正文、完整检测报告、CNKI PDF 和 checkpoint 不加入 Git；在调用方选择的保存位置落实其忽略/备份规则。

## 选择入口

| 用户要什么 | 使用方式 |
|---|---|
| 一篇正文的可定位标记、JSON/JSONL/HTML 报告 | `annotate --file ... --report-dir ...`；一次任务优先使用此入口 |
| 连续检测多篇、复用已加载模型 | 启动一次本地 API，串行调用 `/v1/annotate` 或 `/v1/detect` |
| 每模型原始全文/窗口分数 | `detect` 或 `/v1/detect`；它不生成三档融合标注 |
| 多条独立短文原始分数 | `/v1/detect-batch`，不是长文标注批量接口 |
| 完整旧报告改用另一策略或门限 | `recombine`，只复用缓存分数，不重新推理 |

## 准备输入

输入可以是 UTF-8/BOM 的 `.txt`、Markdown 或 LaTeX 文件；文件内容原样检测，**不会自动去掉**引用、参考文献、公式、注释、代码块或 LaTeX 命令。用户要求测正文时，由调用方先提取正文并保留到原文件的字符/章节映射，使用抽取后的文件。不要因扩展名是 `.tex` 就直接把整份源码当作正文。

文件读取移除开头的 UTF-8 BOM，保留 CRLF。Python 中使用 `Path(...).read_bytes().decode("utf-8-sig")`，避免 `read_text` 将 CRLF 转为 LF 后改变 hash 和位置。

单条输入限 **200,000 Python Unicode 字符**，包含空格、标点和换行，且不能是空白。超过上限按章节/段落拆分；记录每批的顺序、输入 hash 和源位置。各批汇总需使用同一策略/模型版本，不能把标记字符占比当作 AI 来源比例。

## 一次检测并生成完整报告包

在调用方 PowerShell 运行。先将下面两处示例路径换为实际输入和按调用方约定选定的输出位置；`$reportDir` 必须尚不存在，父目录可以存在。空格路径保留引号。

```powershell
$detectorPython = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$sourcePath = 'D:\调用方工作区\本次任务\正文.txt'
$reportDir = 'D:\调用方工作区\本次任务\aigc-report-new'
& $detectorPython -X utf8 -B -m aigc_detector annotate `
  --file $sourcePath --report-dir $reportDir `
  --language zh --policy thesis-cnki-20260929-v2
if ($LASTEXITCODE -ne 0) { throw '标注未完成，读取 stderr 中的 JSON 错误。' }
```

默认中文长文策略 `thesis-cnki-20260929-v2` 必须完整使用十个中文模型、400/120 字符分片及与 profile 完全一致的模型 revision。省略 `--models`、分片和门限选项即可使用默认配置；不抄写融合门限数值，不按试跑结果临时调整门限。片段连续无重叠、覆盖全部输入；超出某模型 token 上限时内部另用窗口覆盖。模型缺失、revision 不符或 profile 缺失时保留错误，不能静默少选模型或换策略来宣称默认检测已完成。

CLI 成功输出 UTF-8 JSON；失败输出 stderr JSON，退出码为 1。`annotate` 成功摘要包含实际报告路径、策略、汇总和耗时，完整逐段结果在报告包里。

```text
报告目录/
  source.txt       精确输入，UTF-8 无 BOM，保留原换行
  report.json      完整元数据、策略/profile、汇总及逐段分数
  segments.jsonl   每行一个片段
  report.html      无需联网的高亮与筛选页面
  COMPLETE.json    全部写入成功后的完成标记
```

交付前核对退出码、上述文件和 `COMPLETE.json` 的 `status="complete"`，其 `input_sha256` 应等于 `report.json` 的 `input.sha256`；报告的字符数/hash 应对应本次实际输入。目录存在或 HTML 已写出不代表完成。报告页中的位置只对应 `source.txt`，抽取正文时依靠调用方映射回原稿。需要展示时打开生成的 `report.html`。

仅取原始分数：

```powershell
& $detectorPython -X utf8 -B -m aigc_detector detect `
  --file $sourcePath --language zh --output 'D:\调用方工作区\本次任务\raw-new.json'
```

`detect --output` 会覆盖同名文件，调用方应选新路径。`--models mpu-zh-v3 hc3-zh` 可显式选原始检测子集；不指定时运行所选语言组。不要把整篇 `detect` 的窗口分数当作 `annotate` 的片段分数：后者逐片独立计算，两种入口的分数/统计不必相等。

## 多篇任务：本地服务与报告保存

```powershell
& $detectorPython -X utf8 -B -m aigc_detector serve
```

命令在当前终端启动服务，保持该终端运行。可传 `--port` 使用另一个空闲端口，随后同步客户端地址；端口被其它程序占用时不结束未知进程。

读取GET /health确认service=local-aigc-detectors；model_groups.zh/en.ready表示原始detect语言组文件就绪，默认英文annotate看annotation_profiles.en.ready（同时核对profile revision、返回策略/hash/名单/分片）。顶层status只按中文原始组判断。健康检查不加载权重或核验全部SHA-256，不能替代实际推理成功。GET /v1/models也返回annotation_profiles及原始组。

接口同步计算，单进程串行使用 GPU。复用服务，不增加 GPU worker；长文可能超过客户端默认 600 秒，按正文长度设置更长 timeout。超时没有提供取消或进度查询机制，不能据此认为服务器已停止，也不要立即反复提交同一长文。

需要 HTTP 生成完整报告包时，用已安装检测器的 Python 执行下面的脚本，并把两个路径换为实际路径。它通过服务推理，调用本地保存函数核验原文并生成报告包，不在调用进程重载模型。

```python
from pathlib import Path
from aigc_detector.client import Client
from aigc_detector.reporting import save_annotation

source = Path(r"D:\调用方工作区\本次任务\正文.txt")
report_dir = Path(r"D:\调用方工作区\本次任务\aigc-report-new")
if report_dir.exists():
    raise FileExistsError(report_dir)
text = source.read_bytes().decode("utf-8-sig")
client = Client(base_url="http://127.0.0.1:8766", timeout=3600)
report = client.annotate(
    text, language="zh", policy="thesis-cnki-20260929-v2",
    segment_chars=400, min_segment_chars=120, include_text=True,
)
paths = save_annotation(report, text, report_dir, source_path=source)
print(paths)
```

也可在调用方自己的 Python 环境仅用标准库发 HTTP 请求，无需 PyTorch。必须绕过环境代理，使用 UTF-8 JSON，并只向上述本机地址发送正文：

```python
import json
import urllib.request

payload = {"text": text, "language": "zh",
           "policy": "thesis-cnki-20260929-v2", "include_text": True}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
request = urllib.request.Request(
    "http://127.0.0.1:8766/v1/annotate",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json; charset=utf-8"},
)
with opener.open(request, timeout=3600) as response:
    report = json.load(response)
```

这里的 `text` 按输入准备步骤读取。纯 HTTP 得到的是结果对象；若交付要求完整报告包，再使用已安装包的 `save_annotation(report, text, new_dir, source_path=...)`。只保存响应 JSON 不会自动产生 HTML、JSONL 或 COMPLETE 标记。`include_text=False` 可省略响应正文，保存函数会按传入原文补齐并核验 hash。

HTTP 原始检测用 `/v1/detect`，请求如 `{"text": text, "language": "zh", "include_chunks": true}`；批量用 `/v1/detect-batch`，将 `text` 换为 `texts: [...]`，返回 `{"items": [...]}`。批量限 1–32 条、总计 500,000 字符，每条仍限 200,000。`overlap_tokens` 只用于原始检测，范围 0–256，默认 64。不要将 `include_chunks` 传给 `annotate`，也不要将 `policy` 或 `include_text` 传给 `detect`；未知字段返回 422。

任务完成时可保留服务供后续复用。前台服务在确认无人使用后用 Ctrl+C 停止；切换 CPU 时重启并传 `--device cpu`。单次 CLI 也可直接传该参数。

## 英文与显式旧规则

工具不自动识别语言。英文原始检测用 `--language en` 或 HTTP `language="en"`；默认英文组为 `hc3-en`、`desklib-en`、`fakespot-en`、`bibbbu-xlm-roberta`。

英文标注显式指定语言后，默认采用低误报优先的 `english-academic-20261003-v2`：

```powershell
& $detectorPython -X utf8 -B -m aigc_detector annotate `
  --file $sourcePath --report-dir $reportDir `
  --language en
```

HTTP为 `{"text": text, "language": "en"}`；固定策略可传 `policy="english-academic-20261003-v2"`。省略策略/分片时锁定openai-base-en、openai-large-en、piratexx-en、mage-en、radar-vicuna-en五分类器，1600/400、固定revision及profile门限。两头共用同一函数、门限不同。全部10原版/30混合/75生成段已核算；已知两篇论文重复评估人工0/19误标、生成3/12命中、混合生成字符覆盖20.38%，仍有严重漏检且不是新的独立验证，未标记不能排除AIGC。限制随policy/interpretation返回。旧英文v1可显式复现；旧四模型或400/120缓存不能直接改成新五模型/1600/400配置，须重新推理。中英文混合文档由调用方分别选择正文和语言、保留映射。

2026-10-04增加显式非线性实验 `english-academic-20261004-v3`，同新五分类器/1600/400，4个tanh隐藏单元、29参数；普通英文任务仍默认线性v2。648个RBF/网络候选及Newton/Adam/L-BFGS优化已完成，但冻结复测仅1/12单独生成段命中、混合生成覆盖0，低于v2；不能因更复杂而把v3视为升级效果。用户明确复核实验时可传 `--policy english-academic-20261004-v3` 或同名HTTP/Python policy；相同名单/revision/分片的v2完整缓存可显式重组为v3，已有v3缓存省略policy时保留v3。完整说明见 `user_doc/english-detection.md`。

`sensitive-any-model-v1` 采用任一所选模型达到原始默认门限 0.90/0.70 即触发的规则，兼容模型子集和自定义分片；这组门限没有经过英文拟合。2026-10-02 的英文控制样本中，多语言 `bibbbu-xlm-roberta` 对全部 12 条人写全文给出 ≥0.90，因此英文四模型任一规则有明显误报限制。需要仅查英文专用模型时显式加 `--models hc3-en desklib-en fakespot-en`，仍保留 `--language en` 并说明模型集合；原始detect四模型组不随显式子集改变，默认标注由新五模型profile决定，子集结果也不意味着已校准。

中文若明确要模型子集/自定义分片，标注也须选择 `sensitive-any-model-v1`。明确复现旧单头融合时选择 `thesis-cnki-20260929-v1`；它仍要求完整中文十模型和固定版本/分片。普通任务沿用 v2。用户明确要求覆盖门限时，CLI 为 `--high-threshold` / `--possible-threshold`，HTTP 使用下划线同名字段，并保留报告中的实际值和覆盖状态：v2 两条门限分别作用于不同分数，各自须在 0–1 范围内，不要求数值有序；两个旧规则要求复核门限低于高分门限。

## 复用已有报告重组

```powershell
& $detectorPython -X utf8 -B -m aigc_detector recombine `
  --input-dir 'D:\调用方工作区\本次任务\old-complete-report' `
  --report-dir 'D:\调用方工作区\本次任务\recombined-new' `
  --policy thesis-cnki-20260929-v2
```

旧目录须有有效 `COMPLETE.json`、`report.json` 和原 `source.txt`。目标策略会校验缓存的模型集合/revision、分片、原文 hash 和完整性；不兼容时不能伪造补分或改 hash。输出必须为新目录。重组不运行模型，不适用于正文改变后的新稿，也没有 HTTP 重组端点。

## 定位与解释

- `report.json` 的 `input` 记录字符数/hash；`models` 记录实际 ID、来源和 revision；`policy` 记录规则、实际门限及融合 profile 内容/hash；`segmentation` 记录分片和完整覆盖；`summary` 记录各档片段/字符数。
- `segments[].char_start/end` 相对于 `source.txt`，**0 起点、左闭右开、Python Unicode 字符索引**，不是 UTF-8 字节或 JavaScript UTF-16 单元；`line_start/end` 是 1 起点。`text` 和片段 hash 供核对，`model_scores` 保留每模型独立原始分数/窗口。
- v2 先用 `decision.ensemble_score` 与强信号门限判 `strong_signal`；未达到时，再用独立 `review_score` 与复核门限判 `possible_signal`，其余为 `other`。两头都越线时强信号优先。不可完整计算/`no_scorable_tokens` 片段可能归 other，null 分数不能当零或人写证明。
- 原始 `ai_score`、融合两头分数和标签均为未校准筛查线索，不是概率、AI 写作比例、作者来源证明或论文通过/不通过结论；`other` 也不证明人写。短文、低中文含量、模型分歧及不可编码等 warnings 与分数一起保留。
- 融合只使用对应profile锁定的原始输出（中文十模型，英文v1四模型、v2五分类器；扩展schema可锁定统计输出）；人写控制约束只用于拟合。现有参考集拟合和留出后重拟合诊断不能描述为准确率或独立泛化已验证。CNKI 标记不是来源真值，比较时须核对双方正文范围/字符分母。

交付说明实际检测的文件/抽取范围、语言、策略和模型集合，链接完整报告及 HTML，说明完成状态及实际限制。检测请求本身不授权改论文；用户要求修订时由调用方结合正文语义人工复核标记。

## 异常与少数需要查文档的情况

| 情况 | 下一步 |
|---|---|
| CLI 失败或 HTTP 422 | 读取 `error/detail`，核对空白/长度、字段、语言、策略和分片/门限；只纠正当前请求 |
| HTTP 503、模型缺失、revision/profile 不符 | 停止宣称检测完成；核对 `/v1/models` 或 `& $detectorPython -X utf8 -B -m aigc_detector models`。不能降级冒充默认策略；安装/修复超出普通调用时明确报告 |
| HTTP 500 / 连接失败 | 保留错误；核对服务终端中的启动与错误信息。不要把错误 JSON 当结果 |
| 显存不足 | 优先单次 CLI `--device cpu`；服务设备切换按前述生命周期处理 |
| 旧报告不完整或输出目录已存在 | 保留旧目录并选择新输出；缺少 COMPLETE 的包不能作为完成结果或重组输入 |
| 缺依赖、模型或迁移机器 | 运行 `python -m aigc_detector --help` 与 `prepare --help`，按安装说明使用所选 Python 和用户确认的模型目录；不要把非交互错误当成功 |
| 新的 API 字段/响应版本、复杂保存或重组问题 | 定点读 `user_doc/api.md` 或 `user_doc/long-text-annotation.md` |
| 英文来源/特殊模型限制、需要追查英文样本证据 | 定点读 `user_doc/english-detection.md` |
| 比较 CNKI、追查融合依据/独立验证限制 | 定点读 `user_doc/dissertation-integration.md` 及其链接的相应验证证据 |
| 用户明确要求修改代码、模型、依赖或融合 profile | 转入检测工作区的 `AGENTS.md`、`dev_doc/README.md`、`dev_doc/skill/workflow.md` 与相关模块；本 skill 只覆盖调用 |

上述文档路径位于检测器发行资料中。仅阅读解决当前特殊问题所需内容；正常调用无需扫描整套文档。

显式英文旧任一规则sensitive-any-model-v1仍默认400/120及原始0.90/0.70门限；其它模型子集也须显式选择此规则。recombine省略policy时已有英文融合缓存保留其原策略，英文旧任一规则缓存选择当前默认v2；仍核对分片、完整名单及revision，不兼容不能伪造补分。

## 显式补充英文模型与大模型统计

五个补充分类器openai-base-en、openai-large-en、piratexx-en、mage-en、radar-vicuna-en已构成0.7.0默认英文v2标注融合；detect可通过models显式选这些ID，但原始detect默认仍为旧四模型。任意其它子集/组合标注使用sensitive-any-model-v1，不能把新名单交给旧四模型profile。MAGE窗口4096，其它常规分类器512；GPU缓存可能释放后重新加载。RADAR模型卡限制非商业使用。

用户明确选择大模型统计时，使用已安装检测器的 Python 运行：

```powershell
& $detectorPython -X utf8 -B -m aigc_detector statistical-detect `
  --methods fast-detect-gpt-neo fast-detect-gpt-falcon binoculars-falcon `
  --file $sourcePath --gpu-memory-gib 10 --cpu-memory-gib 20 `
  --output 'D:\调用方工作区\本次任务\statistical-new.json'
```

按实际调用方路径保存一个新输出文件；成功结果schema为aigc-statistical/1.0、status=complete，并核对input hash。三个方法共用GPT-Neo 2.7B、Falcon 7B与7B-Instruct的本地权重。7B双模型顺序加载、bfloat16、batch=1并按需GPU/CPU卸载，串行调用避免与其它GPU任务争用；默认512/64完整窗口。Fast-DetectGPT的raw_score越高越偏AI特征，Binoculars越低越偏AI特征；独立统计入口没有概率、三档判定、报告包或门限；扩展schema可锁定其原始输出，本轮候选已比较这些方法，但最终默认英文v2没有选入统计后端。不据统计量宣称已验证检出。统计入口仅CLI/Python，不能把methods发到原HTTP检测端点。models命令及/v1/models列出statistical_backends安装状态。特殊资源/来源/统计适配问题查user_doc/english-detection.md。

0.9.0已安装四个显式专业英文／协作ID：followsci-bert-en、roberta-academic-en、llm-detectaive-en、cocodet-en。原始检测可传--models加上述ID，仍用--language en；默认detect和v2标注保持。多分类结果class_scores保留全部类别，ai_score_classes/score_semantics解释聚合；DetectAIve聚合含机器润色，CoCoDet聚合Mix+AI实质内容，人工内容的语言润色可能仍属human。CoCoDet窗口2048，其它三个512。标注新名单须显式sensitive-any-model-v1，原门限未对新模型校准；安装通过不意味着目标领域零误报或50%召回。权重只在本地运行，DetectAIve发布Space未明确权重许可，不据可下载推断可再分发。具体来源与命令见`user_doc/english-detection.md`。
