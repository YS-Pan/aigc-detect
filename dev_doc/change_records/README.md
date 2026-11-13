# 已完成变更

- [20261113_193920_github-publication.md](20261113_193920_github-publication.md)：GitHub 发布集合与历史隐私审计、独立安装说明、当前依赖快照、旧网络入口兼容及 203 项回归。

- [20261113_184500_independent-package.md](20261113_184500_independent-package.md)：0.10.0 独立安装、按需资源准备及分类器真实模型对照；统计和联合部署余项在记录中标明。

- [20261006_141604_english-low-recall-review.md](20261006_141604_english-low-recall-review.md)：扩充三类英文语料后的原因复盘、原始分数与高生成比例诊断、真实段落50%召回误报代价及无本地训练候选；CPU数值独立复现，默认保持。

- [20261004_165800_english-frozen-local.md](20261004_165800_english-frozen-local.md)：禁止底层训练约束下完成来源/上下文配对、12局部扫描方案和四篇新人工来源；嵌套与新人工验收均未达目标，保留v2，25项测试通过。

- [20261004_161058_english-followup-review.md](20261004_161058_english-followup-review.md)：后续三轮工作重评、51个冻结方案互补性及单信号诊断，更新来源／上下文对照和底层适配优先级；运行配置保持。

- [20261004_155248_english-specialized-fusion.md](20261004_155248_english-specialized-fusion.md)：四新模型全量打分、42候选及75候选精度对照，嵌套与复测未达目标，保留v2；29项测试与真实离线对照完成。

- [20261004_151222_english-specialized-models.md](20261004_151222_english-specialized-models.md)：四个专业英文／协作模型下载及多分类接入，ModernBERT特殊标记适配，137项回归与真实离线/部署验证；默认融合保持。

实质改动记录放在本目录，小修放在 [minor/](minor/README.md)。记录完成事实和验证，不把计划当结果。

- [20261004_141456_english-hard-negative.md](20261004_141456_english-hard-negative.md)：完成复盘第一步的9候选缓存实验、嵌套论文留出和冻结复测；最难人工排序仍漏检并出现留出误报，保留v2，停止本轮扩大搜索。

- [20261004_124640_english-review.md](20261004_124640_english-review.md)：近期英文检测复盘、纯人工窗口零误报/混合生成段50%口径、固定评分门限取舍与后续难例/领域适配建议；只做缓存诊断，默认不变。

- [20261004_094234_english-nonlinear.md](20261004_094234_english-nonlinear.md)：648个RBF/网络候选、Newton/Adam/L-BFGS优化、冻结v3复测下降与保留线性默认，115报告、122项回归及0.8.0本机部署完成。

- [20261003_221843_english-recalibration.md](20261003_221843_english-recalibration.md)：完整10原版/30混合/75生成段重校准、12信号/三档比较、默认五分类器融合、115报告与114项回归；重复评估改善但仍有严重漏检。

- [20261003_193713_english-additional-models.md](20261003_193713_english-additional-models.md)：五个补充英文分类器、GPT-Neo/Falcon统计资源、直连优先下载、21资源校验与本机CPU/GPU/API/CLI验证；不改原融合。

- [20261003_164122_english-calibration.md](20261003_164122_english-calibration.md)：复制10篇英文学术来源、30份生成版本与75处生成段；四模型低误报参考配置、冻结测试、92项契约与真实调用验证，并明确独立生成检出仍为0。

- [20261002_195813_cross-workspace-skill.md](20261002_195813_cross-workspace-skill.md)：自包含 aigc-detect 调用 skill、dissertation 目录链接、两工作区入口与维护同步及契约验证。

- [20261002_181638_english-models.md](20261002_181638_english-models.md)：可选英文四模型、Desklib本地适配、20条来源样本、默认中文回归与英文误报观察。

- [20260930_224335_author-forum-samples.md](20260930_224335_author-forum-samples.md)：作者论坛 MHTML 的本人正文提取、独立本地样本目录与十模型默认 v2 检测。

- [20260930_215144_public-samples.md](20260930_215144_public-samples.md)：公开样本统一入口、24 条来源采集、十模型原始检测与默认 v2 漏标观察。

- [20260929_111401_nonlinear-fusion.md](20260929_111401_nonlinear-fusion.md)：双头非线性融合、疑似标签学习、参数搜索与敏感性诊断、跨章留出修复。

- [20260929_103441_reference-fusion.md](20260929_103441_reference-fusion.md)：三篇论文参考融合、CNKI 结构化与 checkpoint 归档、离线重合并和 Git 初始化。

- [20260929_091459_six-classifiers.md](20260929_091459_six-classifiers.md)：新增六个分类器、EnsemJudge 本地适配、十模型离线与接口验证。

- [20260928_221930_initial-deployment.md](20260928_221930_initial-deployment.md)：四个中文检测模型、统一接口、真实试跑与文档系统。
- [20260928_223900_long-text-annotation.md](20260928_223900_long-text-annotation.md)：敏感筛查三档标注、JSON/JSONL/HTML 报告和十万字符验证。

- [20261006_132010_english-diverse.md](20261006_132010_english-diverse.md)：30来源三类生成语料、177完整输入、六来源冻结验收及两条固定纯分数改进；失败保留v2，173项测试和真实跨cwd CLI通过。
