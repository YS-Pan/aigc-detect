# dissertation agent 调用入口

检测服务工作区：`D:\project\20260928 aigc detect`。
目标使用方：`D:\project\20260911 dissertation`。

## 推荐调用方式

1. 按 dissertation 自己的工作流选定需要复核的生成结果，提取正文。
2. 启动一次本地 API；后续多个文件复用服务中的已加载模型。
3. 需要定位三档片段时调用 `/v1/annotate`；仅需原始窗口分数时调用 `/v1/detect`。保留所选模型结果、规则和输入 hash。当前默认十模型；需要复现原有四模型时显式选择 MPU 三版本与 `hc3-zh`，并在 `annotate` 中同时指定 `policy="sensitive-any-model-v1"`。
4. 将报告写到 dissertation 工作区规则允许的位置；用字符位置定位需人工复核的正文。标注可保存成 JSON、JSONL 和 HTML，见 [长文指南](long-text-annotation.md)。

```powershell
& 'D:\project\20260928 aigc detect\scripts\start-api.ps1'
```

调用方可在自己的 Python 环境直接使用标准库，不需安装 PyTorch：

```python
import json
import urllib.request
from pathlib import Path

text = Path(r"D:\project\20260911 dissertation\由调用方选择的正文.txt").read_text(encoding="utf-8-sig")
payload = json.dumps({"text": text, "include_chunks": True}, ensure_ascii=False).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:8765/v1/detect", data=payload,
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
& 'D:\project\20260928 aigc detect\.venv\Scripts\python.exe' -m aigc_detector detect `
  --file 'D:\project\20260928 aigc detect\samples\demo-zh.txt' `
  --models mpu-zh-v3 mpu-zh-v3-short mpu-zh-v2 hc3-zh
```

命令输出纯 JSON 到 stdout，错误到 stderr，失败退出码为 1。真实任务用选定正文替换示例文件；`--output` 可指定结果路径。源码文件会原样输入模型，服务不会自动去除 Markdown/LaTeX 标记。

## 本轮参考数据与拟合目标

新策略 `thesis-cnki-20260929-v1` 已使用指定的本人论文正文、CNKI 标记范围及 wz、zyl 两篇人写论文控制材料完成拟合。拟合采用仅分数的 L2 正则化 logit stacking；比较正则强度与控制材料权重后，选用的正则强度为 0.1、人工控制样本权重为 4 倍。profile 对人写控制设严格零标记约束，并让本人论文的筛查范围靠近 CNKI 标记。人工控制权重只改变训练样本权重；运行时融合仍只读取十个模型分数，不读取文本、作者或文档身份。最终融合门限保存在 profile 中，不在文档里重复抄写数值。

导入的 CNKI 报告汇总为 1.2%（1206/97762）；本地提取的连续正文范围为 61,655 个 Unicode 字符。两者统计范围和字符定义不同，不能直接把两个分母当作同一测量值。结构化导入保留了 5 份 PDF 的逐页文本与版面位置，共提取 15 个标记片段（2 个显著、13 个疑似）；片段可追溯到页码和坐标。CNKI 标签是外部系统的比较标签，不是作者身份真值。

当前 checkpoint 与标注为 2026-09-28 的提交 PDF 是否完全相同无法确认，因为该提交 PDF 未提供，导入记录没有提交文件的 SHA-256。不要把当前 checkpoint 的结果说成对那份已提交 PDF 的复现。

## 拟合与独立检验

profile 从十个模型的原始分数拟合，不使用正文文本、文件名或作者名作为特征。拟合所用的论文片段及 CNKI 标记也参与了参数或门限选择，因此这些材料上的拟合指标属于参考集内结果，不能当作独立测试或泛化准确率。CNKI 的 1.2% 只描述其自身标记统计，不能作为本检测器的准确率。

独立留出检验要求训练、超参数选择和门限选择都没有见过留出的整份文档或章节。按句子随机拆分同一篇论文仍可能共享写作者和上下文，不能替代文档级独立留出。脚本会按人写文档和 CNKI 阳性章节分组，排除留出组后重拟合标准化参数、系数与门限。不过超参数是在这三份参考材料上比较和选择的，所以这些结果只能称为留出重拟合诊断，不能称为真正独立或未接触测试。本轮没有独立泛化已验证或准确率结论。

## 解释和职责

- `ai_score` 和 `ensemble_score` 都是筛查分，不是“AI 占比”、概率或来源证明；每份报告保留原始模型分数和融合贡献供复核。
- 长文默认用十模型融合 profile，需固定 400/120 切段并校验模型 revision。profile 的门限保存在生成的 profile 文件中，不应提前手填或根据 smoke 样本调整。
- 旧规则 `sensitive-any-model-v1` 可显式选用，适合复现任一模型达到 0.90/0.70 门限即标记的敏感筛查规则。
- 本服务负责离线推理、全文覆盖、来源版本和结构化结果；调用方负责正文抽取、章节选择、报告归档和最终判断。
