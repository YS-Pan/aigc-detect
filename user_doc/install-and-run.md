# 安装、运行与验证

## 安装

需要 Python 3.11 或更新版本。当前 GitHub 交付为源码；本次发布不包含 PyPI 上传。以下为 PowerShell 示例，使用所选 Python 安装轻量基础包，只包含代码和小型资源，不安装 PyTorch 或下载模型：

```powershell
git clone https://github.com/YS-Pan/aigc-detect.git
cd aigc-detect
python -m pip install .
python -m aigc_detector --help
python -m aigc_detector models
```

需要检测时，在源码目录安装 `inference` extra；HTTP 服务选 `http`，统计方法选 `statistical`。后两项包含推理依赖：

```powershell
python -m pip install '.[inference]'
# 按需要选择：python -m pip install '.[http]' 或 '.[statistical]'
python -m pip check
```

基础安装与功能安装均不下载模型。先按下文显式准备所需资源，然后执行检测；源码安装完成后可以从任意目录运行包。

### 构建与安装本地 wheel

需要分发安装文件时，在源码目录构建 wheel。构建不下载模型，产物位于 `dist/`：

```powershell
python -m pip wheel --no-deps --wheel-dir dist .
```

在目标机器用所选 Python 安装生成的 wheel，例如：

```powershell
python -m pip install '<wheel 文件路径>'
python -m aigc_detector --help
python -m aigc_detector models
```

本地 wheel 也可直接选择 extras，以下路径替换为实际的 `.whl` 文件路径：

```powershell
python -m pip install '<wheel 文件路径>[inference]'
python -m pip check
```

只选择需要的 extras。GPU 用户应按机器 CUDA 和 PyTorch 的安装说明选择兼容 wheel；已有满足版本要求的包可复用。运行不要求保留源码目录或固定的 Python 虚拟环境。

如需让 agent 使用随安装包提供的 `aigc-detect` skill，先用同一个 Python 查看其位置：

```powershell
python -c "from importlib.metadata import distribution; d=distribution('local-aigc-detectors'); print(next(d.locate_file(f).resolve() for f in d.files if f.as_posix().endswith('skills/aigc-detect/SKILL.md')))"
```

读取显示的 `SKILL.md`；需要自动发现时，将其所在的整个 `aigc-detect` 目录复制到所用 agent 的 skill 发现目录，并按该 agent 的方式启用。安装包不会自动修改 agent 设置。

## 准备模型

交互式首次使用需要模型的功能时，工具询问大文件保存根；输入目录后再下载所需文件。显式选择目录，例如本机的 `D:/AgentTools`，则模型目录为 `D:/AgentTools/models/aigc-detect`。用户选择的目录会保存，取消则不下载。

也可显式准备或复用已有模型目录。`--assets-dir` 是**模型目录本身**，其中包含 `manifest.json` 和各模型子目录：

```powershell
python -m aigc_detector prepare --assets-dir '<模型目录>' --language zh
python -m aigc_detector prepare --assets-dir '<模型目录>' --annotation en
python -m aigc_detector verify --assets-dir '<模型目录>' --annotation en
```

`--language zh|en` 对应原始 detect 的语言组，`--annotation zh|en` 对应相应默认标注策略；两者的英文模型组不同。指定 `--models <ID...>` 或 `--backends <方法...>` 只准备所需模型。可先用 `prepare --help` 查看选项。已有完整且符合 manifest 的资源会复用；推理只使用本地权重，不执行远程模型代码。

库、HTTP 和非交互命令遇到缺失资源时返回准备提示，不读取输入或隐式下载。先交互选择路径，或由用户明确给出路径和下载授权后执行 `prepare`。模型版本、文件大小及哈希保存在资源目录的 manifest；下载失败保留原有可用资源和错误信息。

## 调用

```powershell
python -m aigc_detector detect --file '<输入文本>' --output '<新结果.json>'
python -m aigc_detector annotate --file '<输入文本>' --report-dir '<新报告目录>'
python -m aigc_detector serve
```

服务只监听 `127.0.0.1`，默认端口为 8766；可用 `serve --port` 改变，并同步客户端地址。前台服务可用 Ctrl+C 退出。健康检查 `http://127.0.0.1:8766/health` 只检查资源状态，不加载权重或替代真实推理。服务串行使用 GPU，调用方可设置超时；客户端超时不等于服务器推理已停止。

从源码运行的 Windows 用户也可用 `scripts/start-api.ps1` 和 `scripts/stop-api.ps1` 管理后台实例；启动时 `-Python` 可指定已安装包的解释器。日志及受身份校验保护的进程记录位于 `%LOCALAPPDATA%/local-aigc-detectors/`。停止前确认没有其他调用方正在使用服务。

缺失模型或 revision 与 profile 不符时，保留错误并准备正确资源，不能通过减少模型冒充默认标注。英文默认标注固定五模型和 1600/400 分片，中文默认标注固定十模型和 400/120 分片。具体范围和筛查结果限制见[长文指南](long-text-annotation.md)与[英文指南](english-detection.md)。

## 检查与故障

- `verify` 检查所选资源；实际功能还需要对选用模型执行一次离线推理。健康检查不证明推理成功。
- 导入失败时确认运行的 Python 与安装包、extras 所用 Python 相同。显存不足可选择 `--device cpu`。
- 端口占用时选择其他端口，不结束未知进程。模型下载先直连，确实失败后可使用现有代理，不修改系统代理。
- 统计后端的 Neo/Falcon 资源、设备限制和输出语义见[英文指南](english-detection.md)。
