# 工作区约定

## 入口

- 调用已有检测能力：先读 [user_doc/README.md](user_doc/README.md)，按任务进入接口或跨工作区指南。
- 修改代码、模型、依赖或文档结构：先读 [dev_doc/README.md](dev_doc/README.md) 和 [维护流程](dev_doc/skill/workflow.md)，随后按导航读取相关模块。
- 所有 Markdown、样本、请求和 JSON 结果使用 UTF-8。

## 操作

- 使用本工作区 `.venv/Scripts/python.exe`，避免混入其他工作区的环境。
- 模型下载先直连；确实失败后可使用当前配置的代理。沿用 `scripts/network.py`，不改系统代理。
- 推理只加载 `models/` 的本地权重，不启用远程模型代码，不把待测文本提交给外部服务。
- 模型版本和文件哈希由 `models/manifest.json` 记录。更新权重同时更新来源说明和验证证据。
- 原始检测输出是未校准分数。长文默认使用 `thesis-cnki-20260929-v1`：十模型全选、400/120 字符分片、模型 revision 与 profile 完全一致；融合门限读取 profile，不得预写或按 smoke 样本调整。旧 `sensitive-any-model-v1` 可显式选择，原始默认门限为 0.90/0.70。标注入口见 [长文指南](user_doc/long-text-annotation.md)。
- 融合只以十个模型的原始分数为运行时特征；不能使用文本、作者或文件身份特征。人写控制材料的加权和零标记约束只用于拟合。综合筛查分不是概率或来源证明，参考集拟合或留出重拟合诊断不得表述为准确率或独立泛化已验证。
- 更新融合 profile 时，保持 profile 中模型 revision、profile 哈希、拟合参数和门限可追溯；同步更新拟合证据。不得把拟合材料上的结果描述为独立测试。
- 外部工作区的文本选择、预处理、检测报告归位由调用方负责。本工作区负责通用推理与接口。

## Git 与隐私

- Git 当前在本地 `main` 分支初始化，未配置 remote。没有明确授权时不添加 remote、不推送。
- 保留 `models/manifest.json` 和有来源许可的公开测试样本；排除虚拟环境、权重、原始私人材料、完整报告、CNKI PDF、checkpoint 和缓存。用户及 peer 的论文正文或完整检测报告不得加入 Git。

## 收尾

同步受影响的用户指南、架构、模块说明；记录实际验证及限制。未来事项写入 `dev_doc/toDo/`，已完成变更写入 `dev_doc/change_records/`。文档维护规则见 `dev_doc/skill/`。
