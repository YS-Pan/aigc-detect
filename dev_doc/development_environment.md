# 运行环境

## 当前环境（2026-11-13）

开发源码保持在 `D:\project\20260928 aigc detect`。当前运行路线为系统解释器 `C:/Program Files/Python313/python.exe`，Python 3.13.17；安装包版本为 0.10.0，使用 PyTorch 2.10.0+cu128 和 Transformers 5.22.1。普通运行使用所选解释器中的已安装包，不要求工作区 `.venv`、源码目录或其他工具项目。

代码轻量安装、可选 extras、资源准备、分类器和统计后端的新旧对照证据及剩余本机联合验收边界见 [0.10.0 独立包记录](change_records/20261113_184500_independent-package.md)。该记录中的候选测试与本机迁移证据不等于本次 GitHub 发布重新运行了全部真实模型。

依赖声明以 `pyproject.toml` 为准，精确依赖快照见根 `requirements-lock.txt`；实际解释器与安装状态应由同一 Python 查询。用户选择的模型目录独立于源码，模型 revision、大小和哈希由其 `manifest.json` 记录。安装及构建命令见 [安装指南](../user_doc/install-and-run.md)。

## 历史环境与验证（0.1.0—0.9.0）

下列记录描述 2026 年 9—10 月各次交付时的环境与结果，不代表当前安装状态。旧解释器为 `.venv/Scripts/python.exe`，基于当时系统 Python 3.13.11。

当时识别的硬件为 NVIDIA GeForce RTX 4060 Ti，16 GB 显存；NVIDIA 驱动 591.74。旧路线选择 CUDA 12.8 的 PyTorch wheel，Transformers 固定 4.57.6 以适配 BERT 分类器。

初始交付验证 PyTorch 2.11.0+cu128、Transformers 4.57.6、FastAPI 0.141.1、Uvicorn 0.54.0；当时模型的 GPU 和 CPU 加载结果见 [验证记录](context/installation-validation.md)。

2026-09-29 在同一环境扩展到十个分类器，没有新增或升级依赖；全部十模型的 GPU/CPU 加载及推理检查见 [扩展验证](context/model-expansion-validation.md)。

随后导入 CNKI 报告时，在本工作区环境增加可选的 PyMuPDF 1.26.7，固定版本见 `requirements-evaluation.txt`。0.3.0 的分数融合使用 Python 标准库，拟合脚本使用环境已有的 NumPy；未更新模型权重。

0.4.0 的双头二次融合继续仅用标准库运行，参数拟合使用已有 NumPy 的阻尼 Newton 法；没有引入 SciPy 或新增模型依赖。本地 editable 包和管理的 API 已更新到 0.4.0，验证见[双头融合记录](context/nonlinear-fusion-validation.md)。

2026-10-02 更新本地 editable 包及受管理API为0.5.0，登记13个分类器，默认仍为中文十模型，英文组四模型。没有增加或升级运行依赖，Desklib使用已有快速分词器；十三模型哈希、pip check、中英文CPU/CUDA及实际本机部署检查通过，见[英文扩展验证](context/english-expansion-validation.md)。

2026-10-03更新editable包和本机API为0.6.0；新增五个英文分类器及三份因果LM资源，登记18分类器/三个统计后端。只增加官方Accelerate 1.15.0、psutil 7.2.2，未升级已有Torch/Transformers。16GB显存采用顺序加载和CPU内存卸载，全部统计后端的GPU及Neo纯CPU检查通过；Falcon纯CPU模式没有实测。实际峰值、完整文件验证和限制见[补充模型验证](context/english-additional-validation.md)。

随后0.7.0完成英文扩展重校准，离线更新editable包、pip check通过并重启已确认空闲的受管理API。18分类器/三统计后端及权重、依赖版本保持；三档上下文全12信号实际打分，最终标注默认新五分类器。114项回归、CUDA真实默认/显式API、完整报告、缓存重组、跨cwd CLI及实际本机HTTP通过；两个真实输入的统计流式输出与旧缓存逐项一致。效果及仍然严重的漏检见[本轮验证](context/english-recalibration-validation.md)。

2026-10-04更新editable包和受管理API至0.8.0，pip check通过，未新增或升级依赖。RBF及tanh网络运行仅用标准库，维护拟合用已有NumPy/Torch float64 CPU、Newton/Adam/L-BFGS；122项回归、真实CUDA离线API与CPU/常驻HTTP/跨cwd CLI和缓存重组通过。非线性复测下降，英文默认保留线性v2；见[实验验证](context/english-nonlinear-validation.md)。

历史真实推理所用 Python、PyTorch、Transformers 和 GPU 信息保存在本机 `reports/smoke-test.json`，完整报告不随 Git 发布。上述阶段使用工作区 editable 安装，使其他 cwd 也能调用。

历史模型总磁盘大小和完整性快照见 `models/manifest.json`；当前实际资源以所选模型目录的清单为准。

2026-10-04更新本地editable包和空闲受管理API至0.9.0。新增四份学术／协作检测权重约2.10GB，没有增加或升级运行依赖。CoCoDet使用现有ModernBERT/SDPA，本地关闭reference compilation；四模型CPU/CUDA、离线40次官方示例推理、长窗、API/跨cwd CLI和137项回归通过。证据及限制见[专业英文接入记录](context/english-specialized-validation.md)。
