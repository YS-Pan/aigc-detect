# 可选英文检测

## 专业英文与人机协作模型（0.9.0）

本轮四模型均已下载并完成离线运行验证，可通过 `--models` / HTTP `models` 显式选择：

| ID | 适用任务 | 分数含义 |
|---|---|---|
| `followsci-bert-en` | 英文学术段落二分类 | AI类分数 |
| `roberta-academic-en` | 英文科研摘要二分类 | machine-generated类分数 |
| `llm-detectaive-en` | 人工、机器生成、机器润色、机器生成后机器人性化四分类 | 除纯人工外三类之和，**包含机器润色** |
| `cocodet-en` | 学术会议评审中 Human/Mix/AI 内容三分类 | mix与ai两类之和；人工内容的语言润色可能仍属human |

```powershell
python -m aigc_detector prepare --models followsci-bert-en roberta-academic-en llm-detectaive-en cocodet-en
python -m aigc_detector verify --models followsci-bert-en roberta-academic-en llm-detectaive-en cocodet-en
python -m aigc_detector detect --language en `
  --models followsci-bert-en roberta-academic-en llm-detectaive-en cocodet-en `
  --file 'D:\调用方工作区\正文.txt' --output 'D:\调用方工作区\specialized-raw.json'
```

HTTP示例为 `{"text": text, "language": "en", "models": ["followsci-bert-en", "roberta-academic-en", "llm-detectaive-en", "cocodet-en"]}`；Python同名参数。多分类的 `results[].class_scores` 保留全部类别分数，`ai_score_classes`、`score_semantics` 解释聚合口径，每个 `chunks[]` 同时保留 `class_scores`。`ai_score` 仍是覆盖权重平均的未经校准分数，不是AI文本占比。

CoCoDet每窗最多2048 token（含CLS/SEP），其余三个512；全部输入完整覆盖。它的训练领域是计算机科学会议评审，并非任意论文正文。DetectAIve检测改写方式，也不是插入生成段的定位器。原始权重、配置、模型卡和哈希在 `models/`；DetectAIve上游未明确权重许可，不能据可下载推断允许再分发。

四个模型尚未加入已注册的运行英文profile，`detect`默认旧四模型、`annotate`默认v2五模型保持。需要用它们生成标注报告时必须显式选择 `--policy sensitive-any-model-v1`；旧0.90/0.70门限并未针对这些模型校准。来源和安装验证见[专业英文接入记录](../dev_doc/context/english-specialized-validation.md)。

2026-10-04已完成目标资料全量打分、42候选融合及75候选微小分数精度对照。两轮开发选择优胜者P07/P08混合生成段均0/12，其他冻结冠军最多1/12，现有v2为2/12；两轮嵌套留出还出现人工误报，因此默认继续保留v2。实验参数没有注册为API策略，未达到“零误报、50%混合段召回”，且这些资料不是新独立测试。方法和证据见[融合重调记录](../dev_doc/context/english-specialized-fusion-validation.md)。

## 原默认英文组与融合

默认 `language="zh"` 仍运行原有中文十模型。英文用 `language="en"` / `--language en` 显式选择；本服务不自动判断语言。`models` 非空时以显式 ID 为准，语言参数仍决定标注的分片规则。

| 英文组 ID | 权重与计算 |
|---|---|
| `hc3-en` | HC3 英文 RoBERTa；Human=0、ChatGPT=1，softmax 第 1 类 |
| `desklib-en` | Desklib v1.01 DeBERTa-v3-large；attention mask 平均池化、单输出 Linear、sigmoid |
| `fakespot-en` | Fakespot RoBERTa；Human=0、AI=1，softmax 第 1 类 |
| `bibbbu-xlm-roberta` | 复用已安装的中英越多语言分类器，softmax 第 1 类 |

新权重分别保存在 `models/hc3-en/`、`models/desklib-en/`、`models/fakespot-en/`，多语言模型共用一份权重。revision 和所有文件的 SHA-256 由 `models/manifest.json` 记录。安装与核验：

```powershell
python -m aigc_detector prepare --language en
python -m aigc_detector verify --language en
```

HTTP、进程内 Python 与轻量客户端使用同一参数：

```json
{"text": "The English text to inspect...", "language": "en", "include_chunks": true}
```

```python
from aigc_detector.engine import Detector
result = Detector().detect("The English text to inspect...", language="en")

from aigc_detector.client import Client
result = Client().detect("The English text to inspect...", language="en")
```

CLI 文件入口：

```powershell
python -m aigc_detector detect `
  --language en --file 'D:\调用方工作区\english.txt' `
  --output 'D:\调用方工作区\reports\english-example.json'
```

分数为未经校准的原始输出。默认 float32，窗口含特殊符号最多 512 token，重叠 64，按 token 覆盖权重聚合并覆盖全部输入。Desklib 上游示例使用更长的截断参数；本项目沿用本地 config 的 512 上限与完整覆盖策略。Fakespot 上游建议的清洗函数没有应用，正文原样检测以保留位置；这里的结果不等于其带清洗的完整 Apollo 系统。各种模型的分数不能直接理解为作者来源概率。

英文长文标注默认采用低误报优先的 `english-academic-20261003-v2`。显式选择语言即可：

```powershell
python -m aigc_detector annotate `
  --language en `
  --file 'D:\调用方工作区\english.txt' `
  --report-dir 'D:\调用方工作区\reports\english-annotation-example'
```

HTTP 为 `{"text": text, "language": "en"}`；Python 为 `Detector().annotate(text, language="en")`。需要固定策略时可显式传 `policy="english-academic-20261003-v2"`。省略策略及分片参数时读取 profile：`openai-base-en`、`openai-large-en`、`piratexx-en`、`mage-en`、`radar-vicuna-en` 五分类器全选，1600/400 字符分片、固定 revision、已拟合线性函数与两条不同门限。`detect` 默认英文组仍为旧四模型；使用本轮融合分档应调用 `annotate`。

全部材料为10篇原版、30份混合文档和75处提取的生成段；按论文绑定其全部派生版本，分为6篇拟合、2篇门限校准、2篇重复评估。旧报告19个人工片段/12个生成段只来自最后两篇论文，不能当作总样本数。旧v1仅用原版和单独生成段拟合，混合文档只评估；本轮将拟合组18份混合文档加入训练，并以校准组混合文档中的纯人工片段控制门限，同论文相同正文去重。最终配置完整评价678片段；三组的原版/混合/生成段分别为6/18/51、2/6/12、2/6/12。九分类器与三个统计输出、三档分片及1728个候选均已比较。

旧v1在P07/P08为人工0/19误标、生成0/12命中、混合生成字符覆盖0。冻结后的新v2在相同论文和1600/400分片上为 **人工0/19误标、生成3/12命中（其中强档1处）**；六份混合文档命中2/12处生成块，生成字符覆盖20.38%，纯人工0/39片段误标，但标记片段同时包含815个人工字符。整套10篇原版均未标记；单独生成段合计8/75命中，混合生成块合计2/75命中。合计包含拟合材料，不能当作独立效果指标。此前P07/P08结果已经被查看，本轮是重复评估，检出仍很低，没有新的独立泛化验证。“其他”不能排除AIGC；限制随 `policy.validation_warning` 和 `interpretation` 保存。详见[重新校准证据](../dev_doc/context/english-recalibration-validation.md)、[完整本机报告](../reports/english-recalibration-20261003-v2/README.md)和[本地样本](../samples/english-academic-20261003/README.md)。

英文分片优先使用英文句末、换行和词间空白，保持连续完整覆盖；报告标为 `english-natural-boundaries-v1`。句号规则属于工程近似。中英文profile互相拒绝不同语言分片。旧英文v1保留，显式 `--policy english-academic-20261003-v1` 可复现；已经使用英文融合的缓存在省略recombine策略时保留该缓存的原策略。旧任一规则缓存省略策略时选择当前英文默认v2。重组仍要求目标profile的完整模型名单、revision及分片：旧四模型缓存不能改成新五模型融合，旧400/120缓存也不能改成1600/400，须重新推理。相同五模型/1600/400/revision的旧规则缓存可显式重组到v2。

2026-10-04已完成更复杂的非线性实验：RBF核融合用Newton法优化系数，小型tanh网络用双种子Adam和L-BFGS联合优化权重、偏置和截距；三档分片、5/9/12信号及648个候选只在拟合论文内比较。冻结的 `english-academic-20261004-v3` 为同一新五分类器、1600/400、4隐藏单元/29参数，纯数值参数和门限可精确复现。但P07/P08重复评估仅1/12生成段命中、混合生成覆盖0，低于线性v2，故 **默认保留v2**。全10原版/30混合/75生成段再次核算，v3单独生成段3/75、混合生成块0/75命中；这包含拟合材料，不是独立效果指标。

需要复核实验时，CLI显式 `--policy english-academic-20261004-v3`，HTTP/Python为 `policy="english-academic-20261004-v3"`。同五模型/1600/400/revision的v2完整报告可显式recombine到v3，无需重新推理；省略recombine策略时已有v3报告保留v3。报告保留固定profile、核/隐藏单元及模型计算贡献和检出不足警告。详见[非线性实验验证](../dev_doc/context/english-nonlinear-validation.md)、[完整本机说明](../reports/english-nonlinear-20261004-v3/README.md)和[全部115个输入报告](../reports/english-nonlinear-20261004-v3/all-inputs/index.html)。

复现旧任一模型规则时显式传 `--policy sensitive-any-model-v1`。该规则仍默认400/120分片与0.90/0.70原始门限，兼容模型子集和自定义分片，没有经过英文融合拟合。新增英文专用模型子集仍需显式选择该规则。

`GET /v1/models` 返回22个登记分类器、原始检测 `model_groups`、`extra_english_models`、`specialized_english_models`、统计后端安装状态和 `annotation_profiles`。`GET /health` 的status按中文原始组判断；英文原始检测就绪看 `model_groups.en.ready`，英文默认标注就绪看 `annotation_profiles.en.ready`，后者还核对profile revision并返回策略、hash、名单和分片。更新代码后重启空闲的常驻服务，不中断其它调用方的任务。

来源明确的英文样本与复现入口见[样本批次](../samples/english-public-20261002/README.md)，实际验证与限制见[英文扩展记录](../dev_doc/context/english-expansion-validation.md)。英文 HC3 训练包含全部 HC3，样本结果不能当作独立准确率。

2026-10-02 这批 12 条来源标为人工的文本中，bibbbu 全部给出 ≥0.90 的全文 AI 分数；它在这批英文控制材料上没有区分能力。英文四模型任一规则因此会产生严重误报。需要仅查看新增英文专用模型时，可显式传 `models=["hc3-en", "desklib-en", "fakespot-en"]`，并保留 `language="en"`；这仍不能替代独立英文验证或校准。

## 2026-10-03 补充模型

默认英文原始detect四模型及旧v1 profile保留。以下五分类器可显式输出原始分数，且已共同构成默认英文v2标注融合：

| ID | 结构 / 窗口 | AI 类与来源限制 |
|---|---|---|
| `openai-base-en` | RoBERTa-base / 512 | 第0类；GPT-2历史基线，MIT |
| `openai-large-en` | RoBERTa-large / 512 | 第0类；GPT-2历史基线，MIT |
| `piratexx-en` | RoBERTa-base / 512 | 第0类；训练披露较少，Artistic-2.0 |
| `mage-en` | Longformer-base / 4096 | 第0类；多领域训练，Apache-2.0；未应用作者文本清洗 |
| `radar-vicuna-en` | RoBERTa-large / 512 | 第0类；对抗训练，模型卡限制非商业使用；名称中的7B指训练文本生成器 |

旧 OpenAI 权重的 pooler 使用本地结构完整接入，分类仍取 CLS 头；缺失或不匹配权重继续报错。分类器使用 float32，MAGE 每批至多一个窗口。GPU 缓存在显存不足时释放最久未用的模型，为加载和推理留出空间；再次选中时从本地重新加载。

```powershell
python -m aigc_detector prepare --annotation en
python -m aigc_detector detect --language en `
  --models openai-base-en openai-large-en piratexx-en mage-en radar-vicuna-en `
  --file 'D:\调用方工作区\english.txt' `
  --output 'D:\调用方工作区\reports\extra-english-example.json'
```

Python / HTTP 的models使用相同IDs。这五模型的完整三档融合用默认英文v2；任意其它子集/组合的原始分数标注需显式选择 `sensitive-any-model-v1`。新模型不能交给旧四模型profile。各分类器仍可显式组合输出原始分数。

## 大语言模型统计后端

三个后端共用三份本地权重：

| 后端 ID | 权重 | 原始统计量方向 |
|---|---|---|
| `fast-detect-gpt-neo` | GPT-Neo 2.7B，同时作参考与评分模型 | 越高越偏 AI 特征 |
| `fast-detect-gpt-falcon` | Falcon 7B 参考模型 + 7B-Instruct 评分模型 | 越高越偏 AI 特征 |
| `binoculars-falcon` | 同一 Falcon 双模型 | 越低越偏 AI 特征 |

```powershell
python -m pip install 'local-aigc-detectors[statistical]'
python -m aigc_detector prepare `
  --backends fast-detect-gpt-neo binoculars-falcon --download-workers 8
python -m aigc_detector verify --backends fast-detect-gpt-neo binoculars-falcon
python -m aigc_detector statistical-detect `
  --methods fast-detect-gpt-neo fast-detect-gpt-falcon binoculars-falcon `
  --file 'D:\调用方工作区\english.txt' `
  --gpu-memory-gib 10 --cpu-memory-gib 20 `
  --output 'D:\调用方工作区\reports\statistical-example.json'
```

安装统计 extra 后无需重复装依赖。两个 Falcon 后端一起调用时共用一轮参考/评分前向计算。此入口目前提供 CLI 与 `StatisticalDetector` Python 类，没有统计推理 HTTP 端点。

本机单卡16GB显存不能同时容纳两份7B原始权重，采用bfloat16、模型前向batch=1、顺序加载和Accelerate卸载。Neo直接流式归约；Falcon只临时保存参考logits，评分logits流式计算统计量。缓存进入工具的可写运行目录，请求结束或异常时清理。`StatisticalDetector.detect_batch`一次最多1024条且总计500,000字符，每条独立上下文；两个Falcon后端共用前向。大后端串行运行，CPU可显式选择但7B耗时增加。本轮两个真实输入的三种输出与旧缓存算法逐项一致。

默认512 token窗口、重叠64，完整覆盖；首token没有前文，不参与下一词统计。长文按token逆重叠权重合并统计项再计算公式，这是本项目的适配。模型前向bfloat16，统计softmax/log-softmax以float32计算，归约float64。独立统计入口返回raw_score、方向、窗口、revision和设备映射，没有概率或单独三档门限。扩展融合schema可以锁定统计输出及其配置，本轮已实际纳入候选比较，但最终默认profile没有选入统计后端；不沿用作者在其它配置上的门限。

安装更多或更大的模型不等于解决英文鉴别。安装证据见[补充模型记录](../dev_doc/context/english-additional-validation.md)，实际融合比较、完整样本利用和仍然严重的漏检见[重新校准证据](../dev_doc/context/english-recalibration-validation.md)。

2026-10-04又完成了[不训练底层网络的来源、上下文与局部扫描实验](../dev_doc/context/english-frozen-local-validation.md)。四个专业模型的自然段／三句组等12方案均未改善P07/P08混合段检出；开发选中方案在两篇新人工来源上还出现1/172窗口误报，默认继续保留v2。来源回查发现提取边界、断词和末尾标点也能显著改变分数；不要据高分重标人工来源或通过修改正文迎合检测器。新实验只在维护脚本运行，不能将其名称传给标注API。

2026-10-06对新增20来源及两种额外Astra生成方式完成[完整177输入验证](../dev_doc/context/english-diverse-validation-20261006.md)。预留六来源验收中，现有v2纯人工窗口误报0/323，但只命中5/112个生成段/整节叙述槽位（真实段落2/64，槽位3/48），生成字符覆盖3.11%；整节整体另列1/5。两条固定改进更差，默认与调用参数保持；未标记仍不能排除AIGC，零观察误报不能证明总体零误报。全部逐主题/生成器/方式结果和[本地完整报告](../reports/english-diverse-20261006/report/index.html)已保存。
