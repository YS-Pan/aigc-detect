# 资源和验证

2026-10-04的[冻结模型可行性实验](../context/english-frozen-local-validation.md)由[english_frozen_local.py](../../scripts/english_frozen_local.py)、[english_frozen_controls.py](../../scripts/english_frozen_controls.py)、[english_source_boundary_check.py](../../scripts/english_source_boundary_check.py)及[审计报告脚本](../../scripts/report_english_frozen_local.py)实现。涵盖410原文块来源对照、3933条上下文输入、6682局部窗口和四篇公开XML人工来源；模型与分类头均不训练。12方案嵌套和冻结复测未改善，新来源人工验收1/172误报，未注册运行profile。25项相关测试、独立门限/计数审计、24次真实单条复跑和四模型29文件哈希验证通过；初版来源词锚漏末尾标点已在独立证据中修正，不能把提取干扰解释为语义转写。

2026-10-04的[最难人工窗口缓存实验](../context/english-hard-negative-validation.md)仅使用既有分数与来源位置；[english_hard_negative.py](../../scripts/english_hard_negative.py)锁定源码/缓存/映射hash，prepare、fit、evaluate依次声明、拟合冻结、读取旧复测，verify原样重建参数与门限。[报告审核脚本](../../scripts/report_english_hard_negative.py)核对生产求值、嵌套分组及门限；[8项测试](../../tests/test_english_hard_negative.py)覆盖梯度、隔离、bag去重及混合正例口径。仅数值进入evaluations；正文、日志和报告留在Git排除目录。未新增推理、权重或依赖，实验未改善检出。

模型下载与核验入口为安装包的 `prepare`、`verify` 命令。样本入口：[prepare_samples.py](../../scripts/prepare_samples.py)（安装夹具）、[public_samples.py](../../scripts/public_samples.py)（公开样本采集、校验和本地检测）。

旧 `scripts.network` 模块名直接别名到 `aigc_detector.network`，下载函数与网络事件、直连失败状态共用同一实现，保留旧开发脚本导入及状态替换的兼容。依赖范围以 `pyproject.toml` 为准；`requirements-lock.txt` 保存指定机器的运行／测试快照，构建依赖单独在隔离环境解析。

下载器对每个重定向目标分别先直连，元数据源回退代理不会迫使可直连的权重CDN走代理。主机直连失败后在进程内记住失败，使用环境代理回退；不更改全局设置。模型 API 提供 commit revision 与 LFS SHA-256，下载实际权重及 tokenizer 文件，验证后原子替换 `.partial`。既有 manifest 固定后续下载 revision。仅保存允许的模型资源文件，不拉取或执行仓库Python。支持safetensors/bin分片与索引，检查weight_map完整性；显式`--download-workers 1..8`使大权重按64MiB字节范围并发续传，核对每段Content-Range、完整大小和官方SHA-256后再原子替换。

`prepare --models <ID...>` 可选择子集；省略时默认中文十模型，`--language en` 选择英文四模型，`--annotation en` 选择默认英文标注的五模型。`verify` 支持相同选择及 `--all`。tokenizer 白名单包含英文 BPE 的 vocab.json/merges.txt 及 Desklib 的 spm.model，不执行上游 Python。catalog 的 `subfolder` 限定检测 checkpoint；`tokenizer_repo_id/tokenizer_revision` 指定独立 tokenizer 来源。EnsemJudge 只取两个分类器的子目录，其 HFL tokenizer 另行下载，不下载基座预训练权重或其他大模型。每个新增资源记录原始 repo/revision/path、本地 SHA-256 和大小，manifest 在一个模型完整下载后原子替换，保留此前网络事件。

样本脚本从官方 MPU demo 解析示例常量，不执行 demo；从 HC3 固定 revision 获取 baike 前三组满足长度条件的成对回答，保留行号、答案字段和索引。另构造长文夹具验证窗口覆盖。样本来源和许可见 [samples/README.md](../../samples/README.md)。

`samples/` 是统一公开样本入口，原有 smoke 文件保留，新采集按批次建子目录。`public_samples.py collect` 只复现 `public-20260930` 的固定来源和 24 条规则，复用 `network.py` 直连优先、失败后环境代理回退；不执行来源代码。采集先于检测，保存上游快照、revision/位置、许可和输入 hash；`verify` 核对 JSONL、逐条 UTF-8 文本和证据哈希。`detect` 复用本地 Detector，为每条分别计算全文十模型原始分数和默认 v2 的 400/120 分段结果，核对 profile/模型版本、全文覆盖，并在推理时阻止 socket 连接。标签元数据不进入模型输入；结果只写入新的报告目录，完成标记最后写入。两种检测入口的分数不可视为相同。`prepare_samples.py` 不再覆盖统一样本索引。

作者提供的科创存档入口为 [kechuang_samples.py](../../scripts/kechuang_samples.py)：只使用标准库解析本地 MHTML 的已呈现内容，按主帖/回复作者头部精确选择正文，排除引用和媒体控件；按 HTML 声明严格解码、恢复表情与单份 MathJax LaTeX，保存 UTF-8 逐帖/主题正文、位置映射、可用图片、图文阅读页与原档哈希。`collect` 拒绝覆盖目录，`verify` 核对选择后的文本、主题拼接、位置和资源哈希。复用 `public_samples.py detect` 的通用校验及本地十模型入口；默认 v2 和 profile 不作修改。作者标签与网页历史提示分别保存，样本批次按明确请求放入 `samples/kechuang-human-20260930/` 并由 Git 排除。实际结果见[作者论坛材料验证](../context/author-forum-validation.md)。这不是通用网页抓取或 OCR 工具，不补齐未保存分页。

英文样本入口为 [english_samples.py](../../scripts/english_samples.py)：按固定锚点提取四个 Gutenberg 历史作者作品节选，按文件顺序选择 HC3 wiki_csai 前八组合格的人类/ChatGPT 原始回答。保存原始快照、作者年代、字段、revision、位置、许可和 UTF-8 哈希，离线 verify 逐条对回来源。只使用 wiki_csai 的明确 CC BY-SA 来源，避免许可不明确的英文分支。真实英文验证 [test_english.py](../../scripts/test_english.py) 对20条文本分别计算四模型全文分数及显式旧规则分段报告，同时核对离线网络阻断、作者公式、CPU/CUDA、长窗和 API；不拟合英文门限或融合。

验证分层：

1. verify_models：全部文件与已记录 SHA-256 比较。
2. pytest：窗口边界和去重权重、不合法请求、模型缺失契约。
3. smoke_test：真实权重 × 官方样本、单窗口公式、EnsemJudge 分类头张量公式与 fast/slow tokenizer 对照、CPU/CUDA、长文、离线禁止 socket 连接、FastAPI 检测和标注路由。支持 `--output` 保存独立报告。
4. 实际本机 HTTP 与跨 cwd CLI：验证调用方部署方式。

依赖精确版本记录在 requirements-lock，原始真实模型结果保存在 reports。公开说明不把 smoke 分数当作阈值或准确率。

英文学术校准样本入口为[english_academic_samples.py](../../scripts/english_academic_samples.py)：复制指定本地语料包（排除build/render/temp），哈希核验、固定整篇分组、对两类正文执行相同TeX提取并保存块级位置映射，整个批次Git排除。[calibrate_english.py](../../scripts/calibrate_english.py)负责本地离线纯分数打分、拟合论文留出选择、profile冻结和测试；test推理必须核对FREEZE，默认候选打分只包含fit/calibration。真实报告/API/重组验证由[test_english_calibration.py](../../scripts/test_english_calibration.py)执行。结果及检出不足见[英文校准验证](../context/english-calibration-validation.md)。

0.6.0新增五个英文分类器与三个统计LM资源；download_models支持--extra-english/--backends，verify_models --all核验全部21份资源。大模型可选依赖为Accelerate 1.15.0、psutil 7.2.2（当前lock已包含）。真实验证由scripts/test_additional_english.py提供classifiers/statistical两个部分，使用既有公开样本，屏蔽推理网络连接；完整结果进入独立reports目录，不能当作独立准确率或据此拟合门限。

0.7.0的[calibrate_english_v2.py](../../scripts/calibrate_english_v2.py)完整利用拟合组混合上下文，比较三档分片/九分类器/三统计输出，冻结后重新评估两篇已知论文，核算全部10原版/30混合/75生成段。数值证据保存在evaluations/english-academic-20261003-v2，不含正文；raw缓存、日志和完整报告留在被忽略的reports中。reproduce从数值原样重建参数/门限。test_english_recalibration.py验证真实默认/显式API、完整报告和缓存；test_english_recalibration_deployment.py验证常驻HTTP与跨cwd CLI。tests/test_english_expanded.py和test_statistical_batch.py覆盖原始统计方向、上下文隔离、资源复用、无泄漏选择/门限、空间交集和冻结保护。114项回归通过；实际结果与仍然严重的漏检见[重新校准验证](../context/english-recalibration-validation.md)。

0.8.0的[calibrate_english_nonlinear.py](../../scripts/calibrate_english_nonlinear.py)复用哈希锁定的旧三档数值，进行648候选/6折非线性选择，固定源码及profile后才复测；reproduce原样重建。build_english_nonlinear_reports.py核对原始窗口与冻结数值并生成115包；test_english_nonlinear.py与test_english_nonlinear_deployment.py验证真实离线API/常驻HTTP、跨cwd CLI和缓存重组。tests/test_nonlinear_fusion.py核对独立公式、抵消时贡献记账、优化下降及确定性、冻结边界和显式接口。122项回归通过，未新增权重或依赖；复测下降故保留线性默认，见[实验验证](../context/english-nonlinear-validation.md)。

build_english_recalibration_reports.py从原始窗口缓存重建全部115个输入的完整报告，验证每片hash/原始分数/融合结果对应冻结数值；明确标注缓存来源与未再次模型推理。完整报告及逐篇HTML索引在Git排除的本轮reports/all-inputs中，不复制到evaluations。

0.9.0的download_models --specialized-english安装四个专业英文／协作检查点，新增29文件约2.10GB；当前verify_models --all覆盖22分类器/3统计LM，共25资源。支持固定Space子目录、显式权重文件名及外部基座配置；CoCoDet原字节只重命名不转换，ModernBERT仅取config/tokenizer。test_specialized_english.py提供collect、validate、deployment，见[来源和验证](../context/english-specialized-validation.md)与[轻量证据](../../evaluations/english-specialized-20261004-installation/README.md)。下文21资源属于0.6.0历史阶段。

[english_diverse_corpus.py](../../scripts/english_diverse_corpus.py)只读接入30原版/147混合的schema1与schema2，校验文件/跨度/逆向恢复，统一抽取abstract/prose/list并保留字符来源映射；整节247个实际生成叙述槽位与1112个资源拆分跨度分别管理。[english_diverse_validation.py](../../scripts/english_diverse_validation.py)声明来源作者角色、离线评分177完整输入、两候选冻结和六来源验收；[audit_english_diverse.py](../../scripts/audit_english_diverse.py)独立公式/逐字符核算、数值复现和跨cwd完整正文CLI验证。173项测试通过；[完整结果与限制](../context/english-diverse-validation-20261006.md)说明检出失败、保持v2及语料已见边界。
