# 开发文档入口

本项目沿用参考 yadof 的 `user_doc` / `dev_doc` 分层，以及 architecture、blueprints、context、toDo、change_records 的职责划分。文档只描述此项目实际实现与证据。

## 按任务阅读

1. 先读 [维护流程](skill/workflow.md) 和 [用户入口](../user_doc/README.md)。
2. 修改系统行为时读 [架构](architecture/README.md)；遇到分数、门限、分块术语时读 [术语](terminology.md)。
3. 查看 [模块索引](blueprints/README.md)，只读本次涉及的模块说明。
4. 查看 [待办索引](toDo/README.md)，确认是否与当前任务相关。
5. `context/` 按文件名及链接进入；无需每次通读历史实验。
6. `change_records/` 是已完成的历史，`obsolete/` 是失效材料；只在追查原因时阅读。

## 导航

| 文档 | 职责 |
|---|---|
| [architecture/README.md](architecture/README.md) | 当前组件、数据流、边界及不变量 |
| [blueprints/README.md](blueprints/README.md) | 实现模块与扩展位置 |
| [development_environment.md](development_environment.md) | 已验证机器与依赖复现 |
| [terminology.md](terminology.md) | 分数、token、窗口、样本等含义 |
| [context/model-sources.md](context/model-sources.md) | 模型选择与官方来源 |
| [context/english-specialized-validation.md](context/english-specialized-validation.md) | 四个专业英文／协作模型，来源许可、多分类语义、ModernBERT适配和0.9.0验证 |
| [context/english-specialized-fusion-validation.md](context/english-specialized-fusion-validation.md) | 四新模型全量打分、42候选与75候选精度对照；嵌套误报及复测检出不足，默认保持 |
| [context/installation-validation.md](context/installation-validation.md) | 安装和试跑证据 |
| [context/model-expansion-validation.md](context/model-expansion-validation.md) | 六个分类器扩展、十模型离线及接口验证 |
| [context/english-calibration-validation.md](context/english-calibration-validation.md) | 英文样本复制、低误报参考适配、冻结测试与检出不足限制 |
| [context/english-recalibration-validation.md](context/english-recalibration-validation.md) | 完整10原版/30混合/75生成段重校准、12信号/三档比较、五分类器默认及重复评估限制 |
| [context/english-nonlinear-validation.md](context/english-nonlinear-validation.md) | 648个RBF/网络候选、Newton/Adam/L-BFGS优化、冻结v3复测下降与保留线性默认 |
| [context/english-detection-review-20261004.md](context/english-detection-review-20261004.md) | 英文开发复盘、混合段50%及纯人工窗口零误报口径、固定门限取舍、困难负例与领域适配建议 |
| [context/english-detection-followup-review-20261004.md](context/english-detection-followup-review-20261004.md) | 后续三轮重评、51个冻结方案互补性诊断、来源转换与局部表示／底层适配优先级 |
| [context/english-frozen-local-validation.md](context/english-frozen-local-validation.md) | 禁止底层训练约束下的来源／上下文对照、12局部方案和四篇新人工来源；未达标、保留v2 |
| [context/english-hard-negative-validation.md](context/english-hard-negative-validation.md) | 最难人工窗口排序的9候选缓存实验、嵌套论文留出与冻结复测失败、保留v2 |
| [context/english-expansion-validation.md](context/english-expansion-validation.md) | 可选英文四模型、Desklib本地适配、20条来源样本与误报观察 |
| [context/english-additional-validation.md](context/english-additional-validation.md) | 补充五个英文分类器与本地大模型统计方法的来源、安装、资源和验证限制 |
| [context/long-text-validation.md](context/long-text-validation.md) | 十万字符标注、结构化报告和三档规则验证 |
| [context/fusion-validation.md](context/fusion-validation.md) | 三篇论文参考校准、CNKI 同范围比较、迁移限制和实际验证 |
| [context/nonlinear-fusion-validation.md](context/nonlinear-fusion-validation.md) | v2 双头非线性融合、疑似位置覆盖、稳定性取舍和无泄漏留出诊断 |
| [context/public-samples-validation.md](context/public-samples-validation.md) | 24 条公开来源样本、十模型原始分数与默认 v2 漏标观察、归档和证据边界 |
| [context/author-forum-validation.md](context/author-forum-validation.md) | 本地 MHTML 的作者正文提取、14 主题/159 帖、十模型检测与一处默认 v2 疑似误标观察 |
| [toDo/README.md](toDo/README.md) | 待用户触发及自动触发的未来事项 |
| [change_records/README.md](change_records/README.md) | 已完成变更索引 |
| [history.md](history.md) | 主要阶段摘要 |
| [skill/README.md](skill/README.md) | 文档维护约定 |

当前工作区不依赖其他项目的 Python 环境；不把 yadof 的框架、集群或 wheel 接收流程复制到本项目。

[2026-10-06多来源英文验证](context/english-diverse-validation-20261006.md)：30来源、三种生成方式、177完整输入的先分组冻结验收；两条非神经改进未达目标，保留v2。

[扩充样本后的低检出复盘](context/english-low-recall-review-20261006.md)：三类生成协议、工程完整性及原始分数复核；高生成比例仍漏检，固定v2达到50%真实段落召回的误报代价，以及无需本地训练的新信号候选。所有新增门限诊断都是已见材料上的事后分析。
