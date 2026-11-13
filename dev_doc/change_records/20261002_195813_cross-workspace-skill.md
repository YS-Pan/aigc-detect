# 其它工作区 agent 调用 skill 与 dissertation 链接

完成时间：2026-10-02 19:58:13（Asia/Shanghai）。用户要求新增一个使其它工作区 agent 大部分情况下无需读检测工作区文档就能正确调用工具的 skill，并在 `D:\project\20260911 dissertation` 添加链接。

## 交付

- 在本工作区 `.agents/skills/aigc-detect/` 维护自包含 `SKILL.md` 与 `agents/openai.yaml`。覆盖 UTF-8/BOM/CRLF、正文抽取及位置映射、跨 cwd 的本工作区解释器、CLI/HTTP 选择、服务复用、完整报告包保存和完成核验、中文 v2、英文及模型子集旧规则、无推理重组、分数解释与异常处理。
- 常规任务仅凭 skill 与实际命令/API 返回完成；安装、运行异常、新字段、CNKI/融合证据追溯和明确的实现维护才定点进入相应文档。不复制推理代码，不增加运行依赖、权重、门限或服务端点。
- dissertation 的 `.agents/skills/aigc-detect` 为指向本工作区真实 skill 目录的 Windows Junction；核对类型、目标、内容 SHA-256 和 `samefile`。源内容维护一份，后续更新通过链接可见。
- 同步检测工作区 README、用户入口、dissertation 集成说明、架构、接口蓝图及文档维护契约。接口、语言组、默认策略、报告保存或启停变化时须同步 skill。
- 同步 dissertation 的 `devDoc/README.md` 调用入口和 `devDoc/workspace.md` 的 skill 表/恢复边界；`tools/doc-discovery/coverage.json` 只精确登记新外部 skill 链接的排除，避免论文摘要发现跨入检测文档系统。其余六项内部 skill 发现保持原规则。
- dissertation 两项 devDoc 文档已本地提交为 `b0926ff`。Junction、覆盖配置和本任务 temp 记录位于该仓库外，随工作区保存；链接不能代替外部真实 skill、环境和权重。检测仓库已有未提交变化保留，本任务不将它们顺带提交，不配置 remote 或推送。

## 实际验证与限制

- skill-creator `quick_validate.py` 在源目录与 dissertation 链接目录均通过；界面元数据由其生成器生成，short_description 长度符合契约。默认自动选择策略未关闭。
- 5 段 PowerShell 示例语法解析、2 段 Python 示例 AST 解析通过。核对实际 HTTP 请求 schema 的默认 v2 与 400/120、英文显式请求、原始/批量字段，以及 `save_annotation` 签名；7 个特殊情况文档路径存在。
- 从 dissertation cwd 使用检测工作区 `.venv/Scripts/python.exe` 成功运行 CLI 帮助及 `models`，确认跨目录 editable 导入、中文十模型与英文四模型清单；另核对 annotate/recombine 帮助选项。
- dissertation 默认文档检查 `errors=[]`、`scan_complete=true`、330 项默认摘要/585 项有效摘要；仅此新外部 Junction 被精确排除。当前任务完整读取并确认基线 330 项，变化后复用 329 项、补读 workspace 摘要与改变的全局入口，最终 `input_read_complete=true`，无待读项。
- 两工作区相关 Git diff 空白检查通过，dissertation devDoc 提交后工作区干净。未读写私人检测正文，未重跑模型推理，未作独立 agent 前向试用。以上验证覆盖 skill 格式、命令/请求契约及链接，不是检测效果、准确率或独立泛化验证。

普通沙箱命令本轮因 setup refresh 初始化错误不能启动；随后通过受审批的命令完成已授权读取、外部链接/入口更新和验证，没有修改系统代理或全局 Git safe.directory。
