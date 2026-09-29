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
| [context/installation-validation.md](context/installation-validation.md) | 安装和试跑证据 |
| [context/model-expansion-validation.md](context/model-expansion-validation.md) | 六个分类器扩展、十模型离线及接口验证 |
| [context/long-text-validation.md](context/long-text-validation.md) | 十万字符标注、结构化报告和三档规则验证 |
| [context/fusion-validation.md](context/fusion-validation.md) | 三篇论文参考校准、CNKI 同范围比较、迁移限制和实际验证 |
| [toDo/README.md](toDo/README.md) | 待用户触发及自动触发的未来事项 |
| [change_records/README.md](change_records/README.md) | 已完成变更索引 |
| [history.md](history.md) | 主要阶段摘要 |
| [skill/README.md](skill/README.md) | 文档维护约定 |

当前工作区不依赖其他项目的 Python 环境；不把 yadof 的框架、集群或 wheel 接收流程复制到本项目。
