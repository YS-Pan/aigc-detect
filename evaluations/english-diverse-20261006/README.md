# 英文多来源完整正文冻结验收

2026-10-06执行，30来源、147生成版本、177完整输入；三类生成方式。新验收6来源预先绑定全部派生版本。两条预声明改进未达目标，默认保持英文v2。解释及分层结果见[验证记录](../../dev_doc/context/english-diverse-validation-20261006.md)。

## 文件

- `campaign.json`、`inputs.json`、`input-lock.json`：先于评分的来源/作者角色、完整窗口、精确生成单位/跨度、原始语料和代码hash；无正文。
- `source-and-assets-audit.json`、`conversion-audit.json`：387源文件、35资产文件、40旧完整输入等价、90个人工块机械PDF词序对照及其限制。
- `development-scores.json`、`baseline-diagnosis.json`、`selection.json`：拟合/开发/人工校准原始分数、两条候选及15个作者组留出。人工校准的生成分数未进入选择。
- `frozen-configurations.json`、`FREEZE.json`：两条固定纯分数参数、人工界限、选中ID和完整流程锁。实验配置未注册运行profile。
- `acceptance-scores.json`、`evaluation.json`：FREEZE之后六篇新来源的完整原版与混合版本；逐类型/生成器/主题/论文指标。
- `independent-audit.json`、`compatibility.json`：独立公式/逐字符计数、纯数值参数复现、作者组隔离、六来源bootstrap及完整正文跨cwd CLI/缓存重组。

原样v2新验收0/323人工误报、5/112生成单位命中；其中真实段落2/64、整节叙述槽位3/48，整节整体另列1/5。选中门限校准1/112，固定线性重拟合0/112。三者均未达50%；零观察误报不等于总体零误报。旧十篇与本轮拟合/开发计数不是独立测试。

## 复核已有运行

工作区根目录使用本项目解释器；这些命令不重新训练底层或向外提交正文：

```powershell
& .venv/Scripts/python.exe -X utf8 -B scripts/english_diverse_validation.py verify
& .venv/Scripts/python.exe -X utf8 -B scripts/audit_english_diverse.py final
```

`report`可由冻结数值及本地正文重建离线页面。首次运行顺序为`declare → score-development → diagnose → freeze → score-acceptance → evaluate → report`。已存在的声明和FREEZE拒绝覆盖；FREEZE后禁止重新评分开发或重新选择。需要新实验须创建新的命名目录/声明，并使用新未见验收资料；不能把这30篇重新称为盲测。

完整正文、源映射、原始模型窗口结果、源清单中的具体论文内容与实际标准报告保存在[本地reports入口](../../reports/english-diverse-20261006/report/index.html)，不在本数值目录保存。现有工作树历史未提交代码保留；本轮提交为新增实验/证据及导航增量，复现锁定的是当前文件hash，不能把旧Git初始提交当作完整0.9.0运行环境。
