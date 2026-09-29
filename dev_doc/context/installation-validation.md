# 安装与验证记录

完成日期：2026-09-28（Asia/Shanghai）。

本文保留初次四模型部署的历史证据。2026-09-29 新增六个分类器及当前十模型状态见 [扩展验证](model-expansion-validation.md)。

## 安装结果

- 四个中文权重均已安装：MPU v3、MPU v3 Short、MPU v2、HC3 Chinese。
- 官方 LFS SHA-256 验证通过；再次独立核验全部 25 个模型资源文件通过。模型资源约 1.64 GB（十进制）。
- Python 3.13.11；PyTorch 2.11.0+cu128；Transformers 4.57.6。`pip check` 无依赖冲突。
- 实测 CUDA 可用，设备为 RTX 4060 Ti 16GB；`.venv` 文件约 4.42 GiB。
- Hugging Face 直连超时后，使用环境代理完成下载；未改系统代理。下载请求路线记录在模型 manifest。

## 验证结果

| 验证 | 结果 | 证据 |
|---|---|---|
| 8 条样本 × 4 模型 | 32 次均运行成功 | [smoke-test.json](../../reports/smoke-test.json) |
| 作者单窗口公式对照 | 四模型分数差为 0 | 同上 `author_single_window_formula` |
| CPU / CUDA 对照 | 最大绝对差约 9.32×10⁻⁹ | 同上 `cpu_cuda_score_difference` |
| 长文全文覆盖 | 每模型 3,999 token、9 个窗口，未截断 | 同上 `long-window-coverage` |
| 禁止网络的推理 | 阻断 socket 连接后，四模型仍完成加载和推理 | 同上 `offline_network_blocked` |
| 输入与接口契约 | 7 项通过 | [pytest.txt](../../reports/pytest.txt) |
| 真实 HTTP / 批量 / 模型筛选 / 错误 422 | 通过 | [deployment-test.json](../../reports/deployment-test.json) |
| 从 dissertation cwd 使用 CLI | 通过；输入仍为本项目官方样本 | 同上 |
| 后台启动、身份核对停止、重启 | 通过；重启后再次验证 HTTP 与 CLI | [installation.json](../../reports/installation.json) |

GPU 试跑过程中，PyTorch 记录的峰值已分配显存约 1,704.8 MiB；它不包含所有驱动/桌面占用，也不代表任意请求的最大显存需求。

测试工具输出一个 Starlette 关于 httpx 未来弃用的提示；当前接口测试已通过。首次 pytest 的系统临时目录权限问题通过工作区内专用 `.pytest-tmp` 解决。没有待修复的试跑失败。

## 样本及结论边界

[samples/smoke.jsonl](../../samples/smoke.jsonl) 包含作者中文演示文本、HC3 baike 三组人类/ChatGPT 回答、一个由上述回答构造的长文覆盖夹具；每条都有来源信息。[样本说明](../../samples/README.md)记录许可证和转换。

测试只确认流程可运行，没有生成准确率或标定门限。模型在部分样本上的分数存在明显差异，原始分数完整保存，不据此选择或淘汰模型。HC3 样本与 HC3 模型训练来源重叠，不能当独立评测集。

## 交付状态

API 已通过启停和重启检查，本次交付时运行于 `http://127.0.0.1:8765`，设备 CUDA。运行 PID 等易变信息以 `.runtime/api-8765.json` 和 `/health` 为准。未设置开机自启。

未读取 dissertation 的正式正文用于检测，未修改该工作区。下一任务入口是 [跨工作区调用指南](../../user_doc/dissertation-integration.md)。
