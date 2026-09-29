# 本项目参考校准数据

这里保存 600 个片段的十模型原始分数、参考标签比例、字符数和来源哈希，不含论文正文。完整文本、映射和 HTML 报告保存在本机 `reports/fusion-20260929/`，不纳入 Git。

## 文件

- `score-dataset.json`：163 个本人论文片段、229 个 wz 片段、208 个 zyl 片段，模型列顺序与 revision 固定。
- `evaluation.json`：最终参考集结果、逐片段融合分数及留出重拟合诊断。
- `control-weight-experiment.json`：比较 L2 强度 0.1/1、人工样本权重 1/2/4/8 的结果。
- `validation.json`：本次实际接口、测试及报告检查摘要。

本人论文的目标值为每片段中 CNKI **显著**标记所占比例。CNKI 疑似片段另行记录，不计入原报告 1.2% 的目标。未显著标记的正文表示比较报告未标记；用户明确确认的两篇人工论文目标为零。人工样本的字符权重乘 4，默认门限约束两篇人工参考论文均零标记。

## 复现配置

在工作区根目录，用本工作区 Python 执行：

```powershell
& '.\.venv\Scripts\python.exe' scripts/fit_fusion.py `
  --dataset evaluations/thesis-cnki-20260929/score-dataset.json `
  --profile reports/fusion-reproduction/profile.json `
  --diagnostics reports/fusion-reproduction/evaluation.json `
  --regularization 0.1 --human-control-weight 4 --human-limit-percent 0
```

拟合只需要已保存的数值，不加载模型。配置和数据 JSON 使用 UTF-8、LF 换行，保证 Git 保存后 SHA-256 不因换行转换改变。运行时融合只依赖每片段的十个分数；来源身份和哈希用于追溯。

## 结果含义

三篇都参与了参数、超参数和门限选择，结果属于参考集拟合。章节留出诊断暴露了明显的迁移限制，不能从这三篇推断对新论文的检测准确率。详细统计、提取范围、知网片段覆盖和实际测试见[验证记录](../../dev_doc/context/fusion-validation.md)。
