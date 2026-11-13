# 项目阶段

## 2026-09：本地中文检测服务

建立独立虚拟环境、四个中文检测模型、统一 Python/CLI/HTTP 接口，以及参考 yadof 分层方式的使用和开发文档。实际完成状态以 [验证记录](context/installation-validation.md) 和 [变更记录](change_records/README.md) 为准。

同月升级至 0.2.0：增加用户选择的敏感筛查分段标注、可复核的 JSON/JSONL/HTML 报告，完成 [十万字符验证](context/long-text-validation.md)。

2026-09-29：在同一环境新增六个分类器，总计十模型。完成 EnsemJudge 本地分类头适配、权重及 tokenizer 追溯、离线和接口检查，详见 [扩展验证](context/model-expansion-validation.md)。

同日升级至 0.3.0：三篇论文与 CNKI 显著标记参与纯分数融合拟合；完成结构化 CNKI 包及当前论文 checkpoint、无 GPU 重合并接口和本地 Git 初始化。指定参考集达到低人工误报与略高于 CNKI 同范围的标记量，章节留出暴露明显迁移限制，详见[校准验证](context/fusion-validation.md)。

同日升级至 0.4.0：新增学习 CNKI 显著与疑似并集的非线性复核头，保留强档行为，默认切换到 v2。加入 135 候选搜索、扰动与消融诊断，修复跨章片段进入留出训练的漏洞。参考论文仅疑似由 0.43% 增至 6.75%，实际疑似位置覆盖为 58.52%；泛化限制仍明显，详见[双头融合验证](context/nonlinear-fusion-validation.md)。

## 2026-10：可选英文模型

0.5.0新增HC3英文、Desklib和Fakespot，并复用bibbbu为英文组；默认中文十模型及两个thesis profile保持原配置。20条来源明确英文材料完成四模型全文与分段试跑，记录bibbbu在人工材料上的严重误报。实现、工程验证及证据限制见[英文扩展记录](context/english-expansion-validation.md)和[完成变更](change_records/20261002_181638_english-models.md)。

2026-10-03：用户指定10篇英文学术来源、30份生成版本、75处生成段复制到本地samples；英文标注默认加入1600/400四模型低误报参考配置。冻结测试0/19人工误报、0/12生成检出，明确没有达到兼顾检出的目标；见[英文校准验证](context/english-calibration-validation.md)。

2026-10-03：0.6.0补充五个显式英文分类器及GPT-Neo/Falcon大模型统计方法，总新增约44GB；21资源校验和本机工程检查完成。默认组和原融合保持，新增结果未进行独立效果验证；见[补充模型验证](context/english-additional-validation.md)。

同日0.7.0完整核算10原版/30混合/75生成段，混合上下文纳入重校准，比较三档/九分类器/三统计输出及1728候选；英文标注默认选择新五分类器融合。重复评估人工0/19、生成3/12、混合生成字符覆盖20.38%，仍有严重漏检且不是新独立验证。114项回归、真实本地API/跨cwd CLI、参数重建和全部115输入报告完成；见[重校准记录](context/english-recalibration-validation.md)。

2026-10-04：0.8.0新增RBF核及tanh网络的英文显式实验融合；648个候选、Newton/双种子Adam/L-BFGS优化、冻结后复测、115输入报告及122项回归完成。最终v3复测1/12单独生成段命中、混合生成覆盖0，低于线性v2，故保留v2默认；见[非线性实验记录](context/english-nonlinear-validation.md)。

2026-10-04：0.9.0下载并接入FollowSci、NTNU academic、LLM-DetectAIve和CoCoDet四个专业英文／协作检测器，登记22分类器。完成多分类语义保留、ModernBERT本地适配、离线与部署验证，未改默认融合或拟合新门限。见[本轮记录](context/english-specialized-validation.md)。

同日四新模型完成115输入全量打分及42候选融合；发现旧裁剪抹平微小分数后另做75候选精度对照。两轮完整嵌套流程未过零误报条件，开发选择优胜者复测均0/12混合段命中，默认保持v2。29项相关测试、16次单条推理对照和冻结参数重建完成；见[融合重调记录](context/english-specialized-fusion-validation.md)。
