# GitHub 源码发布核对

本次按用户授权整理原工作区中的 0.10.0 源码、测试、依赖、公开样本、纯分数证据及使用文档，发布目标为 [YS-Pan/aigc-detect](https://github.com/YS-Pan/aigc-detect)。源码和 Codex project 保持原位；未将迁移临时候选混入，未改系统安装或下载模型，不包含 PyPI 上传。

## 发布集合与历史

- 远端起点为空；本地原有 `8b0fe44`、`0136784` 两次提交保留。`origin` 配置为指定 HTTPS 地址，正常推送 `main` 并建立 `origin/main` 上游跟踪，未强推。
- 历史 110 个 blob 合计约 4.63 MB，最大约 2.74 MB；未发现论文正文、完整报告、PDF、权重、checkpoint、缓存或虚拟环境。当前发布集合再次检查了路径、尺寸、常见凭据模式及 evaluations JSON 的正文类字段，未发现应排除内容。
- 模型 manifest 仅保存来源、revision、尺寸及哈希。公开样本保留来源、署名及许可；私人论文、作者提供的论坛材料和完整报告由忽略规则排除。补充忽略 `build/`、`dist/` 和 wheel。
- 对带字节哈希的样本、纯分数证据、模型 manifest 和包内 profile／资源索引禁用 Git 换行转换，防止新克隆破坏来源或冻结证据哈希；保留上游样本的原有空白。
- README 改为可独立执行的 GitHub 源码安装、extras、资源准备及检测步骤，并明确未校准分数、严重英文漏检和已见材料的证据限制。完整本机报告不随仓库发布。
- 依赖入口改为复用 `pyproject.toml` 的 extras；运行／测试快照更新为当前 Windows Python 3.13.17 环境，移除无关 HTCondor。运行快照中的 setuptools 70.2.0 与隔离构建要求 `>=78` 分开管理。
- 修复旧 `scripts.network` 包装层：旧模块名直接别名到包内实现，保留下载函数和可变状态的兼容，不复制网络实现。

## 本次实际验证

解释器为 `C:/Program Files/Python313/python.exe`（3.13.17），Torch 2.10.0+cu128、Transformers 5.22.1、FastAPI 0.162.0、Uvicorn 0.54.0、Accelerate 1.15.0、psutil 7.2.2。

- `python -m pytest -q --basetemp tmp/release-final-pytest-20261113 -p no:cacheprovider --tb=short`：203 passed，7 个 Torch 弃用警告；宿主 pytest-asyncio 另提示未来默认值。旧 `.pytest-tmp` 有历史 ACL 问题，因此使用新的已忽略目录。
- `python -m pip check`：无依赖冲突；锁文件版本与当前安装逐项核对。`pip install --dry-run '.[http,statistical]'` 解析成功，未更改正式安装。
- `python -m pip wheel --no-deps --wheel-dir tmp/release-dist .`：隔离构建成功。轻量 wheel 约 123 KB；在新建的临时 venv 中通过普通离线 pip 安装及 `pip check`，未安装 Torch 或 Transformers。
- 在源码目录外用该轻量安装检查 5 个帮助／列表命令、25 份资源索引、5 个 profile、随包 skill 与用户文档。非交互缺资源时返回 `PreparationRequired`，未创建模型或配置目录。
- 发布文档 47 个相对链接均指向公开集合中的文件；包内 JSON 可按 UTF-8 解析；`git diff --check` 通过。
- 最终暂存集合为 373 文件、约 53.9 MB；203 份样本／证据／资源文件与工作区原始字节一致。只从暂存区导出的新源码目录中，24 条公开样本与 20 条英文样本的来源、位置及哈希核验通过；在该目录使用 `python -m pytest -q --basetemp .pytest-tmp -p no:cacheprovider --tb=short` 再次得到 203 passed、7 个 Torch 弃用警告，确认无需被排除的本机材料。

本次未重复真实模型推理。既有 0.10.0 的分类器和统计后端对照见 [独立包记录](20261113_184500_independent-package.md)，其范围与未完本机联合验收须按原记录理解。203 项工程测试和安装检查不证明检测准确率、零误报或独立泛化能力。

## 实际发布结果

2026-11-13 19:46（Asia/Shanghai）核对：源码提交 [`a0bc65df9cc346fdeedda34282f4e93d0b1c54ef`](https://github.com/YS-Pan/aigc-detect/commit/a0bc65df9cc346fdeedda34282f4e93d0b1c54ef) 已推送；`git ls-remote origin refs/heads/main` 与本地 HEAD 一致。GitHub 接口已从 `main` 读回新版 README，仓库为 public、默认分支 `main`。本记录的发布结果补记以随后文档提交随同保存；最终分支头可由仓库提交历史核对。
