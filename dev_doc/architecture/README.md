# 系统架构

## 数据流

```text
准备期：用户选择资源目录 → 官方模型仓库 → direct-first 下载器 → <模型目录>/<id> + manifest
运行期：调用方正文 → HTTP / CLI / Python → Detector → 本地 tokenizer + 模型
                                             ↓
                             完整 token 窗口 → 独立模型分数 + 原文位置
标注期：完整正文 → 按语言/profile自然边界分片 → profile锁定的原始输出（中文十模型/英文五分类器） → 固定版本融合 → 三档筛查标签
```

## 组件职责

- `.agents/skills/aigc-detect/`：其它工作区 agent 的自包含调用契约和 Codex 界面元数据；dissertation 通过 Junction 共用真实目录。常规调用不要求读检测文档系统，特殊问题才查回用户指南；不改变推理实现或调用方材料职责。
- `catalog.py`：22分类器、原始detect中文十模型/英文四模型、五个补充英文分类器和三份causal-LM资源/统计后端；标注名单由profile锁定，英文默认v2选择补充五分类器，既有profile保留。
- `engine.py`：参数检查、本地加载、模型缓存、滑动窗口、推理、聚合。
- `modeling.py`：EnsemJudge两层头、Desklib平均池化头和旧OpenAI RoBERTa pooler的本地适配；`assets.py`校验完整权重索引。`statistical.py`顺序加载本地GPT-Neo/Falcon，多个独立输入共用加载、流式归约，保留原始统计量。
- `annotation.py`：自然边界分片、策略调用、位置与模型分歧记录。
- `fusion.py`：版本化纯分数融合；v2 使用强信号和复核两个头，复核头可含平方及两模型交互项。校验 profile、完整模型集合、片段配置及 revision，重新计算报告标签并记录有符号贡献。v1 线性融合和旧任一模型规则保留。
- `nonlinear.py`：schema4.0英文实验profile的固定RBF/tanh网络标准库求值及精确贡献记账；参数在维护期优化，运行时不训练。实验v3复测低于线性v2，英文默认保留v2。
- `reporting.py`：核验原文对应关系，写入 JSON、JSONL、源文本和离线 HTML 报告。
- `api.py`：请求验证、错误状态、健康与模型清单、OpenAPI。
- `cli.py`：文件/标准输入和 JSON 输出；固定本机地址启动 HTTP。
- `client.py`：标准库 HTTP 客户端，绕过环境代理访问本地服务。
- 安装包的准备和核验命令提供模型下载与哈希检查；`scripts/` 保留样本获取、融合 profile 拟合、CNKI PDF 导入、真实权重测试及源码工作区的可选服务脚本。

## 不变量

1. 推理不依赖网络，不执行模型仓库代码；模型加载明确使用 `local_files_only=True` 和 `weights_only=True`。
2. 中文十模型及 HC3/Fakespot 英文分类头的第 1 类为 AI；Desklib的单logit经sigmoid；五个补充英文分类器的AI类均为0。来源包含作者源码、模型配置、模型卡和任务数据说明，统一登记在 catalog。EnsemJudge 从 checkpoint 子目录读取权重，tokenizer 从固定 HFL 基座读取；两者各自可追溯。
3. 全部可编码 token 被覆盖。长文没有前 512 token 的静默截断。
4. 单模型聚合显式扣除重叠重复权重。长文默认用 thesis-cnki-20260929-v2：全十模型、profile 固定的 revision 与 400/120 分片。强信号分与复核分分别使用各自门限，先判强，其余片段再判疑似。保留全部原始分数、双头贡献及交互项证据；融合分数不是概率或来源证明。
   旧 sensitive-any-model-v1 可显式使用，任一所选模型达到默认原始门限 0.90/0.70 即触发。
5. 模型按需缓存，串行锁限制GPU请求；分类器默认batch_size=8、float32，MAGE与CoCoDet最多一窗。统计后端bfloat16、前向batch=1、GPU/CPU卸载和顺序加载；detect_batch最多1024输入/500,000字符，各输入上下文独立。统计原始输出不单独分档，扩展融合须锁定资产及窗口/算术配置。
6. 服务仅绑定 127.0.0.1，正文不进入默认日志。模型、输入 hash 和 revision 随结果提供。
7. 加载时核对完整权重键集合；缺失、不匹配或多余的权重均拒绝提供分数，防止自定义分类头被普通分类头替换后静默随机初始化。
8. 融合门限和系数从版本化 profile 读取，不按 smoke 样本设置。运行时只使用对应profile的原始模型分数；正文、作者和文档身份不能成为特征。显著及疑似的参考覆盖、完整文档或章节留出重拟合诊断分别报告；参与调参的材料不得称为独立测试或用于宣称准确率。
9. 语言通过language=zh/en显式选择。原始detect默认中文十模型/英文四模型，标注由profile选完整名单。英文默认english-academic-20261003-v2（1600/400、新五分类器），重复评估3/12生成段命中、混合生成字符覆盖20.38%，仍有严重漏检，未标记不能排除AIGC。旧英文v1保留；已有英文融合缓存重组默认保留原策略。健康状态按中文原始组判断；annotation_profiles单独返回默认融合的就绪与revision检查。

## 保存边界

`samples/` 统一保存有来源许可的公开检测样本；安装 smoke 夹具保留原路径，额外来源按批次归档。每批来源标签、抽取规则、许可、revision/位置及 UTF-8 内容哈希独立于检测结果；完整结果进入新的 `reports/` 子目录。样本及复现入口见[公开样本索引](../../samples/README.md)和[资源模块](../blueprints/assets-and-validation.md)。

作者明确指定的个人论坛材料可作为本地例外按批次放入 `samples/`，但须先排除该目录的 Git 跟踪。`kechuang_samples.py` 在推理前按作者头部提取已保存的正文，保留逐帖位置映射、原档哈希和独立的作者声明标签；图文阅读页与检测纯文本分开，引用及网页控件不进入模型输入。没有改变 Detector 或融合的运行边界。

源码、文档、有来源许可的公开测试样本、`models/manifest.json` 和不含论文正文的轻量证据可版本管理。论文输入文本、完整检测报告、CNKI 原始 PDF、模型权重、checkpoint、`.venv`、`.runtime`、`logs` 和缓存不进入 Git；`reports/README.md` 是报告目录中唯一应跟踪的文件。下载清单记录来源事实；健康检查只做文件存在/大小检查，完整性通过 verify_models 的 SHA-256 检查确认。

dissertation 保有正文提取、章节筛选、结果归档和科学判断的责任。本项目不自动读取或更改它的论文文件。

英文学术样本保存在Git排除的samples/english-academic-20261003。完整10原版/30混合/75生成段按论文绑定分组；本轮拟合纳入混合上下文，同论文重复正文去重；原始输出、位置数值、哈希及冻结证据可保存到evaluations，不含正文。三档长度/九分类器/三统计输出的候选比较及已知论文重复评估见[重新校准验证](../context/english-recalibration-validation.md)，历史v1见[旧记录](../context/english-calibration-validation.md)。

0.9.0新增四个专业英文／协作模型。DetectAIve与CoCoDet保留多分类输出，ai_score按catalog声明的类别索引求和。CoCoDet由本地ModernBERT+余弦头适配，完整严格加载；原有默认名单/profile保持。见[本轮记录](../context/english-specialized-validation.md)。

四模型后续[融合重调与精度对照](../context/english-specialized-fusion-validation.md)已完成，全部115输入／678窗口均核算。实验参数只属于维护期脚本，未登记为运行profile；两轮嵌套留出及复测未达目标，生产数据流和默认v2保持。

随后[冻结局部扫描实验](../context/english-frozen-local-validation.md)增加维护期来源／上下文配对与自然段、三句组前向评分。12个候选全部只读冻结模型输出，扫描边界不接收标签，所有纯人工扫描机会共同校准；没有训练底层或分类头，也没有接入运行profile。作者组嵌套、旧复测及两篇新来源人工验收未达目标，生产默认和数据流保持。

[多来源英文验证](../context/english-diverse-validation-20261006.md)新增维护期跨schema语料适配、逐字符来源/标签投影、来源作者分组和完整正文冻结评价。运行时仍为锁定五个原始分数；实验只重拟合上层线性映射或校准人工门限，没有训练底层/分类头、增加生产profile或改变默认接口。文本、源映射与完整缓存只放入Git忽略的reports。
