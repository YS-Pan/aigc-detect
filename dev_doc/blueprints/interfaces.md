# 接口模块

实现：[api.py](../../aigc_detector/api.py)、[cli.py](../../aigc_detector/cli.py)、[client.py](../../aigc_detector/client.py)。字段的权威说明在 [用户 API 文档](../../user_doc/api.md)。

FastAPI 同步端点在线程池运行，单例 Detector 用 RLock 串行推理。请求层禁止额外字段，限制字符数和 batch。内部非法参数转换为 422，缺失模型转换为 503；未知运行故障保留 500，不伪造分数。

检测、批量和标注请求的模型列表长度上限随 `len(CATALOG)` 计算，当前为 10；省略列表则运行全部模型。更新 catalog 后须重启常驻服务。

`annotate` 和 `POST /v1/annotate` 默认使用 `thesis-cnki-20260929-v1`。该策略需要完整十模型、400/120 字符分片，以及与 profile 完全相同的模型 revision；profile 文件位于 `aigc_detector/profiles/`。profile 门限不硬编码在接口文档里。子集运行需显式选择 `sensitive-any-model-v1`，其原始分数门限默认为 0.90/0.70。

CLI 从 UTF-8 文件、文本参数或 stdin 获取正文，成功输出 JSON，失败输出 stderr JSON 并返回退出码 1。HTTP serve 固定本机地址与单 worker。`client.py` 只使用标准库并关闭代理继承，因此调用方无需模型依赖。CLI 的 `recombine` 可从带有 `COMPLETE.json` 的完整旧报告中读取原始模型分数和 `source.txt`，重新应用融合或旧规则；它校验输入 hash、模型 roster、revision 和分片参数，不加载模型、不运行 GPU 推理，并要求新报告目录不存在。重组只通过 CLI 暴露，HTTP 没有该端点。当前软件版本 0.3.0，标注报告 schema 为 `aigc-annotation/1.1`。

PowerShell start-api 支持后台/前台模式，分别记录 venv 启动器和服务 worker 的 PID、启动时间、可执行文件路径；后台窗口隐藏，日志进入 logs。stop-api 逐个匹配这些身份后，只停止两个已验证进程，避免误停其他 Python 作业。worker PID 来自健康检查。
