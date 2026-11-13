# 中文与英文纯分数融合模块

2026-10-04新增[专业英文融合重调](../context/english-specialized-fusion-validation.md)：四新模型聚合及类别分数全量打分，42候选后针对1e-5裁剪抹平微小分数另做75候选精度对照。由`english_specialized_fusion.py`和`english_specialized_precision.py`维护期脚本完成分组、冻结、复测和重建；精度表示存入实验参数。两轮均未通过目标，未注册新运行profile，生产融合及默认v2保持。

2026-10-04的维护期缓存实验新增[english_hard_negative.py](../../scripts/english_hard_negative.py)：固定五分数/1600分片/线性函数，对比存在判别、平均人工bag排序和最难人工bag排序共9候选。位置标签只定义拟合bag与评价；训练、标准化和门限逐论文重建，嵌套留出后冻结三类冠军再读取旧复测缓存。没有新增运行policy；最难人工排序复测0/12混合段命中，默认仍为v2。方法、失败与停止条件见[验证记录](../context/english-hard-negative-validation.md)。

实现：[fusion.py](../../aigc_detector/fusion.py)、[nonlinear.py](../../aigc_detector/nonlinear.py)、[v2 profile](../../aigc_detector/profiles/thesis-cnki-20260929-v2.json)、[拟合脚本](../../scripts/fit_fusion_v2.py)。用户契约见[长文指南](../../user_doc/long-text-annotation.md)，实验见[v2 验证](../context/nonlinear-fusion-validation.md)。

## 版本与输入约束

默认策略为 `thesis-cnki-20260929-v2`，profile schema 为 `aigc-score-fusion/2.0`。固定完整十模型、每个模型的 revision、400/120 字符分片、标准化、特征项、两头系数及门限。不满足名单、revision 或分片配置时拒绝融合。

名单以 `CHINESE_MODELS` 的原十模型为准，不随可选英文 catalog 项扩展。英文现场标注拒绝 thesis profile；缓存重组还校验 `segmentation.id=natural-boundaries-v1`，即使英文报告显式使用中文十模型也不能套用中文参考融合。

`thesis-cnki-20260929-v1` 保留原单分数线性融合，原[拟合脚本](../../scripts/fit_fusion.py)仍生成 v1 ID。`sensitive-any-model-v1` 保留任一模型规则，支持模型子集和自定义分片，原始默认门限为 0.90/0.70。

## 双头非线性计算

十个原始分数分别裁到 profile 范围，转为 logit、减中心、除尺度，再裁到 `±clip_standardized`，得到标准化值 `z_i`。特征 `feature_terms` 使用零起点模型索引：

- `[i]`：线性项 `z_i`。
- `[i,i]`：平方项 `z_i²`。
- `[i,j]`：交互项 `z_i × z_j`，其中 `i < j`。

完整二次展开有 10 个线性项、10 个平方项、45 个交互项。每个头独立计算：

`head_logit = intercept + Σ coefficient[k] × feature[k]`

`head_score = sigmoid(head_logit)`

强头沿用 v1 拟合方法，非线性项系数为零，保留既有强信号行为。复核头学习 CNKI 显著与疑似的并集覆盖比例。所有运行时特征均由十个原始模型分数派生，正文、作者、文档身份和位置不进入特征。

1. 强信号分达到 `high_threshold`：强档。
2. 其余片段的复核分达到 `possible_threshold`：仅疑似。
3. 其余：其他。缺任何模型分数时不计算双头，记录缺失原因。

v2 两条门限作用于不同分数，不要求 `possible_threshold < high_threshold`；v1 和旧规则仍要求有序门限。双头不是互斥类别概率，也没有概率校准。

## 报告与计算证据

报告 schema 为 `aigc-annotation/1.2`。`ensemble_score`、`ensemble_logit`、`contributions` 保留字段名，在 v2 表示强头。新增 `review_score`、`review_logit`、`review_contributions` 和 `feature_contributions`，后者逐项保存模型名称、特征值、两头系数和贡献。

每模型贡献包括线性项、平方项及二元交互的一半，贡献之和加截距等于该头 logit。这用于核对计算，不能解释成因果重要性。HTML 显示双头分数、双列贡献和可展开的全部特征项。原始分数和窗口证据不变，单模型原始档位仍按 0.90/0.70 展示。

policy 保留完整 profile、SHA-256、门限所属分数、是否覆盖门限，以及 `calibrated=false`。完整模型 revision 随报告保存。

## 拟合与调试

[数据构建脚本](../../scripts/build_fusion_dataset_v2.py)核验原始报告、正文 hash、CNKI 对齐和章节交集，只将数值、偏移和哈希写入[纯分数证据](../../evaluations/thesis-cnki-20260929-v2/README.md)。600 段原始分数沿用上一轮；旧数据已有疑似标签，但旧 fitter 只学习显著标签。

强头目标是显著覆盖比例，复核头目标是显著与疑似并集比例。用户确认的人写控制目标为零，额外权重只用于拟合。候选比较线性、平方、完整二次展开及 L2 正则和控制权重，使用 NumPy 阻尼 Newton 法和回溯线搜索，无新运行时依赖。

门限由参考数据选择：强档延续既有目标，疑似按“仅疑似”字符比例单独选择，并满足人写控制标记上限。无法满足的候选明确记录不可行。运行时不按文档配额截取片段，也不根据文件身份改门限。

评估同时记录比例、CNKI 实际位置交集、候选优化收敛、门限间隔、分数扰动和整组留出重拟合。参数选择与稳定性筛选都用过参考材料，完整选择过程随证据保存；比例相近本身不证明位置一致或准确率。

## 标签与评估范围

同一 61,655 字符正文中，CNKI 显著 1,171 字符，仅疑似 3,893 字符。原报告分母为 97,762：显著 1,206，13 处疑似合计 4,214；疑似比例由片段字符数推导，原摘要只计显著，不能混用两种口径。

15 处对齐保留页码、哈希和唯一最长汉字锚。第 7、12、14 处含被抽取排除的题注或标题；第 4 处含未匹配短正文，没有猜测补齐。连续疑似区间与严格锚并集相差 13 字符。原送检 PDF 与当前 checkpoint 是否相同尚无法确认。

文档和章节留出时重新拟合标准化、系数与门限。章节采用全部 `unit_memberships` 检查，含留出章节的跨界片段从训练和测试同时排除，并记录数量，避免用中点归章产生泄漏。超参数选择已见过三份材料，所以这些仍是迁移诊断，不能称为独立测试或泛化准确率。具体失败与限制见[v2 验证](../context/nonlinear-fusion-validation.md)。

## 重组与保存

CLI `recombine` 核对旧完整包的 COMPLETE 标记、源文本、输入 hash、片段、模型 roster 和 revision，以缓存原始分数重算选定 policy，不加载模型。结果写入新目录，不覆盖旧包；HTTP 没有重组端点。含正文的完整报告只保存在被 Git 忽略的 `reports/`。

## 英文低误报profile

历史 `english-academic-20261003-v1`沿用schema2.0，两头共用同一线性函数、门限不同，严格锁定旧四模型及1600/400分片；可显式复现。任意不匹配的语言/名单/分片/revision拒绝融合。

[calibrate_english.py](../../scripts/calibrate_english.py)仅以纯分数做论文组内候选搜索，人工权重和零标记约束只用于拟合；测试在FREEZE之后重新推理，参数不变。源码保留线性/平方/交互候选能力，本轮最终选择线性。独立测试人工0/19误报，生成0/12检出；运行policy及interpretation明确记录检出不足。完整证据与复现见[英文验证](../context/english-calibration-validation.md)和[纯分数目录](../../evaluations/english-academic-20261003-v1/README.md)。

## 英文扩展v2

显式language=en默认采用 [english-academic-20261003-v2](../../aigc_detector/profiles/english-academic-20261003-v2.json)，schema3.0。锁定名单可为登记英文分类器及统计后端的唯一非空子集；最终选择新五分类器、1600/400分片及线性函数，双头共用函数、门限不同。名单不会随catalog变化。分类原始分数转logit，Fast-DetectGPT为identity，Binoculars为negative_identity；标准化、裁剪和特征项由profile保存。统计后端必须锁定asset revision及512/64、bfloat16和归约配置；其raw_score不按0.90/0.70单独分档，也不伪装成ai_score概率。

[calibrate_english_v2.py](../../scripts/calibrate_english_v2.py)先声明campaign，再对拟合/校准的全部原版、混合文档、生成段打分。同论文正文hash去重、冲突标签和跨论文重复排除；拟合按论文及人工/含生成桶归一化，目标比较生成字符比例或存在生成字符。三档长度、八种信号组合、三特征族、四L2、三人工权重、两目标共1728候选，仅用拟合论文逐篇留出选择；最终门限只取拟合/校准纯人工最大logit加固定余量。完整声明信号和候选缺失时禁止最终冻结。校准生成分数和test分数不参与参数/候选/门限。

FREEZE锁定profile、训练数值、campaign和selection哈希；随后才运行test-only重复打分，禁止重新选择或覆盖训练分数。全部10原版/30混合/75生成段及678片段分别报告。P07/P08原版0/19误标、单独生成3/12命中、混合生成字符覆盖20.38%；是已经知道旧测试结果后的重复评估，仍有严重漏检。重组省略策略时已有英文融合缓存保留原策略，旧原始规则缓存选择当前默认v2；仍核对精确名单/分片/revision。参数复现只需冻结数值，不读取正文或加载模型。实际工程证据与限制见[重新校准验证](../context/english-recalibration-validation.md)。

## 英文显式非线性v3

`english-academic-20261004-v3`采用schema4.0，保留schema3.0的分数变换、名单、revision、分片和统计合同；heads改为固定的RBF或tanh_mlp参数，两个头函数须完全相同，仍分别使用profile门限。运行只用标准库；模型输入贡献和核/隐藏单元贡献加截距均精确等于logit。网络采用解析直线路径积分梯度，RBF采用坐标核对数变化分摊；是记账而非因果解释。

[calibrate_english_nonlinear.py](../../scripts/calibrate_english_nonlinear.py)锁定三档旧fit/calibration分数来源，比较648个候选、6论文折；[nonlinear_english_fit.py](../../scripts/nonlinear_english_fit.py)提供固定核中心下的Newton法和双种子Adam/L-BFGS网络参数联合优化。只在拟合组选择，拟合/校准纯人工控制门限，FREEZE之后复测，不按复测调参。最终选择新五分类器、1600/400、4隐藏单元、29参数。P07/P08仅1/12单独生成段命中、混合覆盖0，低于线性v2，故默认不变；v3需显式调用。完整全样本核算、收敛及失败候选见[实验验证](../context/english-nonlinear-validation.md)。

0.9.0专业英文四模型尚未进入任何拟合profile。它们只可显式原始检测或使用旧任一规则，多分类的全部原始类别与聚合语义在缓存重组时保留；既有profile文件和门限未改变，见[接入记录](../context/english-specialized-validation.md)。

[2026-10-06多来源实验](../context/english-diverse-validation-20261006.md)仅比较现有v2的人写门限校准和一次固定五分数线性存在判别重拟合（L2=10、人写权重20、1e-12 logit裁剪）；作者组内归一化和标签只用于维护期。新六来源冻结验收分别1/112、0/112，低于原样v2的5/112，未注册运行profile或改变默认。配置/门限以[evaluations](../../evaluations/english-diverse-20261006/README.md)为准，不进入实际调用参数。
