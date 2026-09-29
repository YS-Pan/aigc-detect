# 系统架构

## 数据流

```text
安装期：官方模型仓库 → direct-first 下载器 → models/<id> + manifest
运行期：调用方正文 → HTTP / CLI / Python → Detector → 本地 tokenizer + 模型
                                             ↓
                             完整 token 窗口 → 独立模型分数 + 原文位置
标注期：完整正文 → 400/120 自然边界片段 → 十模型原始分数 → 固定版本融合 profile → 三档筛查标签
```

## 组件职责

- `catalog.py`：中文模型 ID、官方仓库、标签含义。
- `engine.py`：参数检查、本地加载、模型缓存、滑动窗口、推理、聚合。
- `modeling.py`：EnsemJudge 两层分类头的本地适配，沿用 Transformers 的 BERT 前向计算。
- `annotation.py`：自然边界分片、策略调用、位置与模型分歧记录。
- `fusion.py`：旧任一模型敏感规则和版本化 score-only logit stacking；校验 profile、完整模型集合、片段配置及 revision，重新计算报告标签并记录有符号贡献。
- `reporting.py`：核验原文对应关系，写入 JSON、JSONL、源文本和离线 HTML 报告。
- `api.py`：请求验证、错误状态、健康与模型清单、OpenAPI。
- `cli.py`：文件/标准输入和 JSON 输出；固定本机地址启动 HTTP。
- `client.py`：标准库 HTTP 客户端，绕过环境代理访问本地服务。
- `scripts/`：下载、样本获取、哈希验证、融合 profile 拟合、CNKI PDF 结构化导入、真实权重测试和服务生命周期。

## 不变量

1. 推理不依赖网络，不执行模型仓库代码；模型加载明确使用 `local_files_only=True` 和 `weights_only=True`。
2. 十个分类头的第 1 类为 AI。来源包含作者源码、模型配置、模型卡和任务数据说明，统一登记在 catalog。EnsemJudge 从 checkpoint 子目录读取权重，tokenizer 从固定 HFL 基座读取；两者各自可追溯。
3. 全部可编码 token 被覆盖。长文没有前 512 token 的静默截断。
4. 单模型聚合显式扣除重叠重复权重。长文默认用 thesis-cnki-20260929-v1：全十模型、profile 固定的 revision 与 400/120 分片，通过正则化 logit stacking 生成筛查分。保留全部原始分数及各模型有符号贡献；融合分数不是概率或来源证明。
   旧 sensitive-any-model-v1 可显式使用，任一所选模型达到默认原始门限 0.90/0.70 即触发。
5. 模型在一个进程中按需缓存；串行锁限制同时使用 GPU 的推理请求。默认 batch_size=8、float32、eval/inference_mode。
6. 服务仅绑定 127.0.0.1，正文不进入默认日志。模型、输入 hash 和 revision 随结果提供。
7. 加载时核对完整权重键集合；缺失、不匹配或多余的权重均拒绝提供分数，防止自定义分类头被普通分类头替换后静默随机初始化。
8. 融合门限和系数从版本化 profile 读取，不按 smoke 样本设置。参考集拟合指标与独立留出检验分开报告；参考结果不得表述为准确率或已验证的泛化效果。

## 保存边界

源码、文档、有来源许可的公开测试样本、`models/manifest.json` 和不含论文正文的轻量证据可版本管理。论文输入文本、完整检测报告、CNKI 原始 PDF、模型权重、checkpoint、`.venv`、`.runtime`、`logs` 和缓存不进入 Git；`reports/README.md` 是报告目录中唯一应跟踪的文件。下载清单记录来源事实；健康检查只做文件存在/大小检查，完整性通过 verify_models 的 SHA-256 检查确认。

dissertation 保有正文提取、章节筛选、结果归档和科学判断的责任。本项目不自动读取或更改它的论文文件。
