# 运行环境

工作区：`D:\project\20260928 aigc detect`。解释器：`.venv/Scripts/python.exe`，基于系统 Python 3.13.11。

已识别硬件：NVIDIA GeForce RTX 4060 Ti，16 GB 显存；NVIDIA 驱动 591.74。选择 CUDA 12.8 的 PyTorch wheel，Transformers 固定 4.57.6 以适配这些 BERT 分类器。

本次已验证 PyTorch 2.11.0+cu128、Transformers 4.57.6、FastAPI 0.141.1、Uvicorn 0.54.0。全部模型可在 GPU 和 CPU 加载运行，具体检查见 [验证记录](context/installation-validation.md)。

2026-09-29 在同一环境扩展到十个分类器，没有新增或升级依赖；全部十模型的 GPU/CPU 加载及推理检查见 [扩展验证](context/model-expansion-validation.md)。

随后导入 CNKI 报告时，在本工作区环境增加可选的 PyMuPDF 1.26.7，固定版本见 `requirements-evaluation.txt`。0.3.0 的分数融合使用 Python 标准库，拟合脚本使用环境已有的 NumPy；未更新模型权重。

运行依赖的精确版本见根 `requirements-lock.txt`；真实推理使用的 Python、PyTorch、Transformers 和 GPU 信息由 [smoke-test.json](../reports/smoke-test.json) 保存。包的范围声明在 `pyproject.toml`，当前部署使用工作区 editable 安装，使其他 cwd 也能调用。

安装与复现命令以 [安装指南](../user_doc/install-and-run.md) 为准。模型总磁盘大小和完整性以 `models/manifest.json` 为准。
