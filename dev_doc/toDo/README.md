# 待办

当前安装、接口和文档交付任务的进展与验证见 [验证记录](../context/installation-validation.md)。

## 待用户触发

当前没有待用户启动的检测任务。[dissertation 参考校准](dissertation-evaluation.md)已完成；后续更换论文、提取方式或模型版本时按维护触发条件重新验证。

[双头融合后续独立验证](nonlinear-fusion-validation.md)：在获得新材料时先保留整份独立文档，固定配置后再检测；本轮参考适配已经完成，不把已有参考数据继续称为独立测试。

## 自动维护触发

见 [auto/README.md](auto/README.md)。这些是维护时的检查条件，不是后台定时任务。

[英文低误报与检出后续验证](english-calibration-validation.md)：扩展重校准和648候选非线性优化均已完成，完整使用10原版/30混合/75生成段。非线性v3复测下降，默认保留v2的3/12单独生成段及20.38%混合生成覆盖，仍有严重漏检。取得新资料或另行要求训练时保留新的独立文献，不用旧重复评估继续调参后宣称独立。

2026-10-04[复盘与缓存诊断](../context/english-detection-review-20261004.md)明确用户以混合正文生成段召回50%、完全人工窗口零误报为目标，允许含AIGC的正例窗口包含人工内容。随后授权的[人工高分端缓存实验](../context/english-hard-negative-validation.md)已完成：9候选及嵌套留出未改善，最难人工排序P07/P08混合0/12，默认保持v2的2/12；按停止条件结束本轮搜索。尚未执行的专业匹配难例、底层适配及新来源验收见上述待办。

2026-10-04[专业英文四模型接入](../context/english-specialized-validation.md)和[目标论文融合重调](../context/english-specialized-fusion-validation.md)均已完成；42候选及75候选精度对照没有通过目标，默认保持v2。剩余匹配资料、底层适配及新资料冻结验收见[英文待办](english-calibration-validation.md)。

[后续重评](../context/english-detection-followup-review-20261004.md)已完成51个冻结函数的互补性核算。事后重新划界并集加v2仍仅27/75；后续优先来源与上下文对照、少量底层适配及新来源验收，未启动训练。

随后授权的[冻结模型来源／上下文／局部扫描阶段](../context/english-frozen-local-validation.md)已完成，用户明确禁止训练底层网络。12方案嵌套及旧复测未达标，两篇新人工验收1/172误报，未扩大或部署；这批对照不再待执行。仅保留新的任务匹配混合资料、完整来源提取维护和冻结后新正负例验收，详见[更新后的剩余事项](english-calibration-validation.md#冻结模型可行性阶段已完成)。

2026-10-06[30来源/147版本完整验证](../context/english-diverse-validation-20261006.md)已完成新混合正例接入、来源作者隔离及六来源冻结验收；两条非神经改进均未达标，默认v2保持。30篇资料均已见，接入与本次验收不再待执行；剩余目标及禁止底层训练边界见[更新后的英文待办](english-calibration-validation.md#2026-10-06新语料接入及正负例验收已完成)。
