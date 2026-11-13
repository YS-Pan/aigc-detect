# 工作区约定

## 入口

- 调用已有检测能力：先读 [user_doc/README.md](user_doc/README.md)，按任务进入接口或跨工作区指南。
- 修改代码、模型、依赖或文档结构：先读 [dev_doc/README.md](dev_doc/README.md) 和 [维护流程](dev_doc/skill/workflow.md)，随后按导航读取相关模块。
- 所有 Markdown、样本、请求和 JSON 结果使用 UTF-8。

## 操作

- 使用已安装本包的所选 Python；开发时显式记录解释器和依赖，普通运行无需源码目录或专用虚拟环境。
- 模型下载先直连；确实失败后可使用当前配置的代理。沿用 `scripts/network.py`，不改系统代理。
- 推理只加载用户选择的模型目录中的本地权重，不启用远程模型代码，不把待测文本提交给外部服务。
- 模型版本和文件哈希由模型目录的 `manifest.json` 记录。更新权重同时更新来源说明和验证证据。
- 原始检测输出是未校准分数。中文长文默认使用 `thesis-cnki-20260929-v2`：十模型全选、400/120 字符分片、model revision 与 profile 一致；两条融合门限分别作用于强信号分和独立复核分。英文显式 `language=en` 默认标注使用 `english-academic-20261003-v2`：新增五分类器全选、1600/400 分片、固定 revision；两头共用同一分数函数而门限不同。完整10篇原版、30份混合文档及75处生成段已核算；P07/P08重复评估人工0/19误标、生成3/12命中、混合生成字符覆盖20.38%，仍有严重漏检，不能据“其他”排除AIGC。重复评估不是新的独立测试。英文原始detect默认组仍为旧四模型。旧英文v1、`thesis-cnki-20260929-v1` 与 `sensitive-any-model-v1` 可显式选择；后者原始默认门限为0.90/0.70。标注见[长文指南](user_doc/long-text-annotation.md)及[英文指南](user_doc/english-detection.md)。
- 融合只以对应profile锁定信号的原始输出为运行时特征（中文十模型，英文v1四模型、v2五分类器；扩展schema支持锁定的统计输出）；不能使用文本、作者、长度或文件身份特征。人写控制材料的加权和零标记约束只用于拟合。综合筛查分不是概率或来源证明，参考集拟合或留出重拟合诊断不得表述为准确率或独立泛化已验证。显式英文实验 `english-academic-20261004-v3` 为schema4.0、同新五分类器/1600/400、4隐藏单元；648个RBF/网络候选及优化已完成，但复测1/12生成命中、混合覆盖0，低于v2，默认保留线性v2。
- 更新融合 profile 时，保持 profile 中模型 revision、profile 哈希、拟合参数和门限可追溯；同步更新拟合证据。不得把拟合材料上的结果描述为独立测试。
- 外部工作区的文本选择、预处理、检测报告归位由调用方负责。本工作区负责通用推理与接口。

## Git 与隐私

- 本地开发分支为 `main`；发布目标 `origin` 为 `https://github.com/YS-Pan/aigc-detect.git`。2026-11-13 用户已授权本次整理提交、配置该 remote 并推送；后续发布按当次授权范围执行，不强推或覆盖远端已有内容。
- 保留 `models/manifest.json` 和有来源许可的公开测试样本；排除虚拟环境、权重、原始私人材料、完整报告、CNKI PDF、checkpoint 和缓存。用户及 peer 的论文正文或完整检测报告不得加入 Git。

## 收尾

同步受影响的用户指南、架构、模块说明；记录实际验证及限制。未来事项写入 `dev_doc/toDo/`，已完成变更写入 `dev_doc/change_records/`。文档维护规则见 `dev_doc/skill/`。

0.9.0显式学术英文／协作模型：followsci-bert-en、roberta-academic-en、llm-detectaive-en、cocodet-en。后两者保留class_scores，分别汇总非纯人工（含机器润色）和Mix+AI内容类；原英文detect及v2标注默认不变。来源、许可、ModernBERT特殊标记和验证见[接入记录](dev_doc/context/english-specialized-validation.md)。安装验证不等于达到零误报／50%混合段召回。

四新模型已完成115输入／678窗口全量核算、42候选及75候选精度对照；两轮开发选择优胜者复测混合段均0/12，精度对照嵌套24/390人工误报、8/51命中，默认仍为v2。实验参数未注册为运行profile；后续避免1e-5裁剪抹平微小类别信号。证据及已见材料限制见[融合重调](dev_doc/context/english-specialized-fusion-validation.md)。

2026-10-04按用户“不训练底层神经网络”的要求完成[冻结局部实验](dev_doc/context/english-frozen-local-validation.md)：来源／上下文对照、12扫描方案；作者组嵌套3/3103人工误报、3/51混合段命中，旧复测0/473、0/12，两篇新人工验收1/172误报。没有训练模型或分类头，未注册profile，默认v2保持。P01/P02须因共同作者绑定；初版来源词锚漏末尾标点已在source-boundary-check中修正。新来源仅负例，不构成独立召回验证。

2026-10-06[30来源三类英文冻结验收](dev_doc/context/english-diverse-validation-20261006.md)已完成：30原版/147混合/177完整输入；新六来源v2人工0/323、生成段/叙述槽位5/112。两条固定纯分数改进为1/112与0/112，没有可靠改善，默认v2保持，173项测试通过。底层/分类头均未训练；全部30来源现已见，不得重新当作新盲测。
