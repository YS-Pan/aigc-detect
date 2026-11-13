# dissertation agent 调用入口

检测器以 Python 包安装，并从用户选定的模型目录读取权重。
目标使用方：`D:\project\20260911 dissertation`。

常规调用使用 `$aigc-detect`：该 skill 的真实指令维护在 [SKILL.md](../.agents/skills/aigc-detect/SKILL.md)。只读该 skill 即可完成常规检测、报告包保存和重组，不需预先读本页或检测文档系统；本页保留集成背景、CNKI 比较和拟合证据边界，特殊任务按需读取。恢复时须同时恢复 skill、Python 包与权重。

## 推荐调用方式

1. 按 dissertation 自己的工作流选定需要复核的生成结果，提取正文。
2. 启动一次本地 API；后续多个文件复用服务中的已加载模型。
3. 需要定位三档片段时调用 `/v1/annotate`；仅需原始窗口分数时调用 `/v1/detect`。保留所选模型结果、规则和输入 hash。长文默认策略 `thesis-cnki-20260929-v2` 使用完整十模型和 400/120 字符分片；需要复现原有四模型时显式选择 MPU 三版本与 `hc3-zh`，并在 `annotate` 中同时指定 `policy="sensitive-any-model-v1"`。也可显式指定旧融合策略 `thesis-cnki-20260929-v1` 复现单头规则。
4. 将报告写到 dissertation 工作区规则允许的位置；用字符位置定位需人工复核的正文。标注可保存成 JSON、JSONL 和 HTML，见 [长文指南](long-text-annotation.md)。

```powershell
python -m aigc_detector serve
```

调用方可在自己的 Python 环境直接使用标准库，不需安装 PyTorch：

```python
import json
import urllib.request
from pathlib import Path

text = Path(r"D:\project\20260911 dissertation\由调用方选择的正文.txt").read_text(encoding="utf-8-sig")
payload = json.dumps({"text": text, "include_chunks": True}, ensure_ascii=False).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:8766/v1/detect", data=payload,
    headers={"Content-Type": "application/json; charset=utf-8"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=600) as response:
    result = json.load(response)
# 按 dissertation 的文件归位规则保存 result。
```

上面的正文路径是占位说明，调用方必须选择实际文件。超过接口长度限制时，按章节/段落分批提交，并保留段落到源文件的映射。

## 无服务命令行方式

```powershell
python -m aigc_detector detect `
  --file 'D:\调用方工作区\正文.txt' `
  --models mpu-zh-v3 mpu-zh-v3-short mpu-zh-v2 hc3-zh
```

命令输出纯 JSON 到 stdout，错误到 stderr，失败退出码为 1。真实任务用选定正文替换示例文件；`--output` 可指定结果路径。源码文件会原样输入模型，服务不会自动去除 Markdown/LaTeX 标记。

## 参考数据与 v2 融合目标

默认策略 `thesis-cnki-20260929-v2` 使用十个模型的原始分数作为运行时特征，要求与 profile 锁定的模型 revision 和 400/120 字符分片完全一致。v2 保留 v1 的强信号线性规则，并加入独立的复核头拟合 CNKI 的强标记与疑似标记；profile 可列出标准化 logit 的线性项、单模型平方项和模型对交互项。人写控制材料的加权和零标记约束只用于拟合。融合不读取正文、作者或文件身份作为特征。两个筛查头各有自己的门限，保存在 profile 中；强信号优先判定，不能按两个门限的数值大小推断类别。拟合方法、所选参数及当前诊断结果见[非线性融合验证说明](../dev_doc/context/nonlinear-fusion-validation.md)和[v2 可复现评估数据](../evaluations/thesis-cnki-20260929-v2/README.md)。

CNKI 原报告的字符统计与本地提取正文范围不同；两者不能直接作为同一字符分母比较。CNKI 标签是外部系统的比较标记，不是作者身份真值。v1 原始比较记录和可复现纯分数数据继续保留在[旧融合验证记录](../dev_doc/context/fusion-validation.md)和[v1 评估数据](../evaluations/thesis-cnki-20260929/README.md)。

当前 checkpoint 与标注为 2026-09-28 的提交 PDF 是否完全相同无法确认，因为该提交 PDF 未提供，导入记录没有提交文件的 SHA-256。不要把当前 checkpoint 的结果说成对那份已提交 PDF 的复现。

## 拟合与独立检验

v2 profile 从十个模型的原始分数拟合，不使用正文文本、文件名或作者名作为特征。拟合所用的论文片段及 CNKI 标记也参与了参数或门限选择，因此这些材料上的结果属于参考集内拟合，不能当作独立测试或泛化准确率。CNKI 报告中的标记比例只描述其自身的文本范围和标记统计，不能作为本检测器的准确率。

独立留出检验要求训练、超参数选择和门限选择都没有见过留出的整份文档或章节。按句子随机拆分同一篇论文仍可能共享写作者和上下文，不能替代文档级独立留出。整篇或整章留出后重拟合的结果属于诊断；如果超参数或特征项已经根据同一组参考材料选择，这仍不等于真正未接触的独立测试。因此当前不能据此宣称准确率或独立泛化已验证。具体分组和诊断证据见[非线性融合验证说明](../dev_doc/context/nonlinear-fusion-validation.md)。

## 解释和职责

- `ai_score`、`ensemble_score` 和 `review_score` 都是筛查分，不是“AI 占比”、概率或来源证明；每份报告保留原始模型分数和两头贡献供复核。
- 长文默认用 v2 十模型融合 profile，固定 400/120 切段并校验模型 revision。`ensemble_score` 是强信号分，`review_score` 是独立复核分；强信号优先。profile 门限保存在生成的 profile 文件中，不应提前手填或根据 smoke 样本调整。
- 旧融合 profile `thesis-cnki-20260929-v1` 可显式选用以复现单头线性规则；`sensitive-any-model-v1` 可显式选用以复现任一模型达到 0.90/0.70 门限即标记的敏感筛查规则。
- 本服务负责离线推理、全文覆盖、来源版本和结构化结果；调用方负责正文抽取、章节选择、报告归档和最终判断。
