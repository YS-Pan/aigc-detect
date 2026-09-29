# 资源和验证

下载入口：[download_models.py](../../scripts/download_models.py)、[network.py](../../scripts/network.py)。样本入口：[prepare_samples.py](../../scripts/prepare_samples.py)。

下载器先直连。主机直连失败后在进程内记住失败，使用环境代理回退；不更改全局设置。模型 API 提供 commit revision 与 LFS SHA-256，下载实际权重及 tokenizer 文件，验证后原子替换 `.partial`。既有 manifest 固定后续下载 revision。仅保存允许的资源文件，不拉取或执行第三方 Python 源码。

`download_models.py --models <ID...>` 可选择子集。catalog 的 `subfolder` 限定检测 checkpoint；`tokenizer_repo_id/tokenizer_revision` 指定独立 tokenizer 来源。EnsemJudge 只取两个分类器的子目录，其 HFL tokenizer 另行下载，不下载基座预训练权重或其他大模型。每个新增资源记录原始 repo/revision/path、本地 SHA-256 和大小，manifest 在一个模型完整下载后原子替换，保留此前网络事件。

样本脚本从官方 MPU demo 解析示例常量，不执行 demo；从 HC3 固定 revision 获取 baike 前三组满足长度条件的成对回答，保留行号、答案字段和索引。另构造长文夹具验证窗口覆盖。样本来源和许可见 [samples/README.md](../../samples/README.md)。

验证分层：

1. verify_models：全部文件与已记录 SHA-256 比较。
2. pytest：窗口边界和去重权重、不合法请求、模型缺失契约。
3. smoke_test：真实权重 × 官方样本、单窗口公式、EnsemJudge 分类头张量公式与 fast/slow tokenizer 对照、CPU/CUDA、长文、离线禁止 socket 连接、FastAPI 检测和标注路由。支持 `--output` 保存独立报告。
4. 实际本机 HTTP 与跨 cwd CLI：验证调用方部署方式。

依赖精确版本记录在 requirements-lock，原始真实模型结果保存在 reports。公开说明不把 smoke 分数当作阈值或准确率。
