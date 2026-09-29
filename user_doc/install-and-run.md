# 安装、运行与验证

## 当前本地布局

```text
<工作区>/
  .venv/                 Python 虚拟环境
  aigc_detector/         Python、HTTP、CLI 实现
  models/manifest.json   模型版本、文件大小及 SHA-256
  models/<model-id>/     实际模型权重与 tokenizer
  samples/              有来源记录的试跑样本
  reports/              验证报告
  scripts/              下载、校验、试跑、服务启停
```

## 重新创建环境

在工作区根目录执行。系统需有 Python 3.13；CUDA wheel 自带运行库，本机 NVIDIA 驱动需兼容。

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip install --no-deps --no-build-isolation -e .
& '.\.venv\Scripts\python.exe' scripts/download_models.py
& '.\.venv\Scripts\python.exe' scripts/prepare_samples.py
```

`requirements-lock.txt` 保存本次实际依赖版本，复现时可在创建环境后 `pip install -r requirements-lock.txt`，然后安装本项目。

模型下载先尝试直连。某主机在本次下载进程中直连失败后，后续请求回退到环境代理；不修改系统设置。`models/manifest.json` 保存实际网络路线。可续传 `.partial`；下载后校验官方 LFS SHA-256，并记录所有文件的本地 SHA-256。已有 manifest 时使用其中的固定 revision。

下载器支持 `--models ensemjudge-bert ensemjudge-roberta anx-bert-zh paperpass-v3 cmj-zh bibbbu-xlm-roberta`，只处理所列模型。EnsemJudge 从固定子目录读取检测权重，并从各自 HFL 基座获取 tokenizer；manifest 分别记录权重与 tokenizer 的来源和版本。不会下载该仓库中的 7B/9B 权重。

全部权重下载完后，推理设置离线模式、禁用遥测、`local_files_only=True`、`trust_remote_code=False`。标准分类器使用 Transformers；EnsemJudge 的两层分类头由本项目 `modeling.py` 适配。加载时拒绝缺失、不匹配或多余的权重键；旧格式权重通过 PyTorch `weights_only=True` 加载。

## 服务生命周期

```powershell
& '.\scripts\start-api.ps1'                    # 后台，无可见窗口
& '.\scripts\start-api.ps1' -Device cpu        # 停止旧实例后切换设备
& '.\scripts\start-api.ps1' -Foreground        # 当前终端，可 Ctrl+C
& '.\scripts\stop-api.ps1'                     # 停止本脚本记录的实例
```

默认端口 8765，可用 `-Port` 更改。启动脚本复用已在该端口运行的本服务；若已有实例，新的设备参数不会改变它。仅监听 `127.0.0.1`。运行记录在 `.runtime/`，服务日志在 `logs/`；日志默认不记录正文或每次访问。

启动未设为开机自启。机器重启或服务退出后，再调用启动脚本。没有完整模型时健康检查显示 `models_missing`；完整文件存在显示 `ok`，首次推理才加载模型，因此健康检查不代替真实推理验证。

## 检查

```powershell
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' scripts/verify_models.py
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' scripts/smoke_test.py
```

最后一个命令使用真实权重，对 8 条样本运行全部模型，并检查单窗口公式、EnsemJudge 分类头与 tokenizer 一致性、CPU/GPU 一致性（可用时）、长文覆盖、HTTP 检测及标注路由与离线行为。结果默认写入 `reports/smoke-test.json`；可用 `--output reports/自选目录/smoke-test.json` 保留历史报告。它不生成准确率或门限。

模型清单扩展后，应停止并重新启动已有服务，让该进程读取新的代码和清单。

pytest 的临时目录固定在工作区 `.pytest-tmp`，避免 Windows 系统临时目录的其他会话权限影响测试。

长文标注验证入口是 `scripts/test_annotation.py`：构造有来源的 100,000 字符测试夹具，检查全文覆盖、批处理一致性和离线运行，保存新的带时间戳报告目录。当前脚本使用默认融合配置；夹具只用于工程检查，不参与拟合。当前使用说明见 [长文标注](long-text-annotation.md)，0.2.0 的历史实测数据见 [长文验证记录](../dev_doc/context/long-text-validation.md)。

CNKI PDF 导入是可选维护功能，另外安装 `requirements-evaluation.txt`。普通检测服务不需要 PyMuPDF。纯分数拟合入口见 [融合模块](../dev_doc/blueprints/fusion.md)。

## 常见问题

- **导入失败**：确认使用本项目 `.venv`，并完成本项目的 editable 安装。
- **模型缺失/损坏**：先运行 `verify_models.py`；重新执行下载脚本会检查已有文件和哈希。
- **显存不足**：用 `-Device cpu` 启动，或在 Python 中指定 `Detector(device="cpu")`；避免多个 GPU 服务副本。
- **端口占用**：启动脚本不会结束未知进程；选择其他端口并同步调用方地址。
- **网络下载失败**：直连失败后使用环境已有代理；相关实现只影响下载，不影响离线推理和本地客户端。
