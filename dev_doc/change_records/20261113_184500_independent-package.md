# 0.10.0 独立包与资源准备

代码、小型 profile、资源索引、离线核验与 skill 随 Python wheel 安装。基础安装不拉取推理库或模型；所需推理、HTTP 和统计依赖由 extras 选择。资源目录由使用者指定，交互式准备先确认，非交互调用缺资源时返回准备提示。模型文件及清单和可写运行状态不依赖源码目录。HTTP 默认 127.0.0.1:8766；旧开发脚本委托包内准备和核验入口。

保留中文和英文原有模型 revision、窗口覆盖、融合 profile 与分数含义。Transformers 5 的 fast tokenizer 使用适配后的单序列特殊 token，统计推理使用可写 scratch 并移除 5.x 忽略的加载参数。模型推理仍只访问本地权重。

## 已验证

- 完整候选回归 183 项通过；后续对资源准备、错误分支及旧入口的针对性回归 28 项通过。22 个原模型 tokenizer 的中英文固定输入在 4.57.6 与 5.22.1 下 token、offset 和窗口一致。
- 旧 0.9.0 和候选 0.10.0 用相同公开中英文样本，分别在 Condor 14363.0、14365.0 运行中文十模型、英文五模型默认标注、两组默认原始检测和 Desklib/CoCoDet。两作业均退出 0、已确认进程回收；输入、token、窗口、profile、分数及标签在预设浮点绝对/相对 1e-4 范围内一致。两种软件的 Torch/Transformers 版本不同，不能据此外推所有输入或识别准确率。
- 旧版 Neo 在 Condor 14367.0 完成并确认回收，4 GiB GPU / 10 GiB CPU 模型映射预算下 GPU/CPU 混合运行；进程树 RSS 采样峰值 13,150,007,296 字节，Torch GPU allocated 峰值 4,248,695,808 字节，临时 logits 已清理。模型映射预算不是整个进程内存或显存的硬限；本机调度须按实测申请足够作业 RAM 与 GPU 配额。
- 候选 Neo 初试发现无 home 执行账户下路径提前求值，以及 Transformers 5 将 GPT-Neo 旧注意力缓冲报告为多余键；修复为显式状态目录惰性解析，并仅对已核实的旧非持久缓冲做精确兼容，保留其余权重不匹配错误。14378.0 重跑退出 0 并确认回收；新旧 token、窗口、revision 与原始统计值一致，候选 Torch GPU allocated 峰值 4,091,341,824 字节，scratch 已清理。忽略的旧缓冲键记录在结果 `loads` 中。
- 旧版 Falcon 在 Condor 14369.0 一次运行 Fast-DetectGPT 和 Binoculars，两个 7B 模型顺序使用 GPU/CPU 映射，作业退出 0 且已确认进程回收、临时 logits 已清理；进程树 RSS 采样峰值 15,944,609,792 字节。本次为作业申请 22 GiB RAM 和 7164 MiB GPU 配额，模型映射预算仍为 4/10 GiB；这是当前样本与机器的测量，不是硬隔离或最大负载证明。
- 候选 Falcon 14374.0 两方法同样完成并确认回收；500 字符公开样本的 token、窗口、两个模型 revision 和两项原始统计值与旧版一致。候选单进程 RSS 采样峰值 12,975,738,880 字节，临时 logits 已清理。
- 候选源码从无关工作目录用 CPU 运行中文短样本（14379.0，2 CPU、4 GiB RAM、无 GPU 申请），输入、模型 revision、79 token、窗口及分数与旧版 GPU 基线一致；作业退出 0 且已确认回收。
- 在隔离 Python 中完成轻量 wheel 安装及包数据检查；系统 Python 的用户安装从源码外目录可导入并找到安装后的 skill。旧目录下的真实权重仍只读，未因本次验收下载新模型。

## 验收余项

正式机器级安装后的 CPU 短例、HTTP 与 embedding 并存以及消费者切换仍由本机联合验收记录跟进。原始作业日志、对照 JSON 和临时运行证据位于迁移任务的 `tmp/20261113-tool-migration-impl/`；清理前应选留所需证据。

本机 Neo 的调度示例为 `aigc-stat-old-fast-detect-gpt-neo-job.json`：`gpu_memory_mib=7164`、`memory_mib=16384`、`cpus=8`，检测参数 `--gpu-memory-gib 4 --cpu-memory-gib 10`。这些是该次机器及样本的设置，不是所有机器或更大模型的通用上限；作业配额与应用映射预算分别核算。
