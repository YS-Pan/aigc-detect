# 英文模型扩展与来源样本验证

日期：2026-10-02。用户授权执行聊天“调研英文 AIGC 检测模型”中的扩展方案，并要求改代码后联网查找明确的英文人工与 AIGC 内容进行测试。本次新增 HC3 英文、Desklib v1.01、Fakespot，并复用 bibbbu 多语言模型；不纳入本轮讨论中的大语言模型统计后端、受限许可研究候选或 GPT-2 历史检测器。

## 本地实现与来源

软件版本 0.5.0。Python、CLI、检测/批量/标注 HTTP 接口新增 `language=zh/en`；省略仍是原中文十模型，英文组为四模型，显式 IDs 优先。health 保持按默认中文组判断，新增各组 ready。原中文两个 profile 的文件、参数、门限、十模型 revision 和中文分片均保留；profile 校验和拟合脚本按中文 roster 校验，不随 13 项 catalog 扩大。英文现场及缓存重组均拒绝中文参考融合。

| ID | 官方来源与固定 revision | 标签 / 加载 | 许可事实 |
|---|---|---|---|
| hc3-en | [Hello-SimpleAI](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta/tree/d2b342c61775d5dd0221808a79983ed3b86ffd86)；d2b342c61775d5dd0221808a79983ed3b86ffd86 | Human=0、ChatGPT=1；标准 RoBERTa | 模型卡未明确独立权重许可证，本次只本地安装试跑 |
| desklib-en | [Desklib](https://huggingface.co/desklib/ai-text-detector-v1.01/tree/5fdea974cd4287c61674951ec78803aa274e2fb7)；5fdea974cd4287c61674951ec78803aa274e2fb7 | 本地 DeBERTa、mask 平均池化、Linear(hidden,1)、sigmoid | 模型卡 MIT |
| fakespot-en | [Fakespot](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/tree/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db)；f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db | Human=0、AI=1；标准 RoBERTa | 模型卡 Apache-2.0 |
| bibbbu-xlm-roberta | [既有来源](model-sources.md)；b4aa2e06d8428b4d27f2ef625267044d74b2eeb8 | 共用既有目录与分类器 | 既有模型卡 MIT |

新模型分别保存到 `models/hc3-en/`、`models/desklib-en/`、`models/fakespot-en/`。下载沿用 network.py，Hugging Face 直连失败后使用当前环境代理，没有更改系统代理。官方元数据、config、模型卡与源说明保存到 `reports/english-expansion-20261002/upstream/`；每个实际资源的 repo/revision/path/大小/SHA-256 已入 manifest。没有下载或执行上游 Python 推理代码。

全部推理为 float32、本地权重，禁用远程代码。Desklib 本地适配根据上游单行分类头将 num_labels 设置为1，不改写其下载 config；tokenizer.json 由 DebertaV2TokenizerFast 直接读取，未增加 SentencePiece 或其他运行依赖。窗口上限沿用 config 的512，完整滑窗而非只截断首窗。Fakespot 未引入上游建议的文本清洗，以保留源位置；它的结果不等于带清洗的完整 Apollo 系统。

## 来源样本

[english-public-20261002](../../samples/english-public-20261002/README.md) 共20条：四篇 Gutenberg 早期作者作品节选及 HC3 wiki_csai 前八组成对回答。标签计数为 human=12、ai=8。Gutenberg 原作由 Austen、Darwin、Shelley、Thoreau 创作，发表年代分别为1813、1859、1818、1854。HC3 AI 字段明确为 ChatGPT 回答，人写标签来自发布者字段；问题、作者及任何标签信息不作为模型输入。

采集使用明确 CC BY-SA 的 wiki_csai 分支。HC3 数据 revision 固定为 `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`，Gutenberg 锁定实际原始快照 hash。每条原文、位置、字段、索引、许可、来源和 hash 完整保存，verify 离线对回原始来源。选择规则在评分前固定，没有按检测结果筛选样本。样本 JSONL SHA-256 为 `8409e25e3157abc2866400cba4de2c891fa0c40dabe62672d9275052a53cbd18`。

四个 Gutenberg TXT 响应在书尾前结束；保存的是本次取得的原始字节快照，不能视为完整电子书。四个所选完整节选均在保留范围内，源位置和文本hash核验通过。collection 的 `complete_work_archived=false`明确保存这个采集限制，没有补写或推断缺失的后半作品。

联网还核对了 RAID 发布者说明及数据集元数据，其小批量 filter 接口先返回 HTTP 500，简化查询重试后超时；没有把未取得的材料记成测试样本。这批 AIGC 只覆盖早期 ChatGPT。

## 实际检查

在本工作区 .venv 与 RTX4060Ti 16GB 上完成：

- `verify_models.py --all`：十三模型全部登记文件大小与 SHA-256 一致；`pip check` 无依赖冲突。本地 editable 包已安装为0.5.0。
- 最终 pytest：86项契约/回归通过，含原中文双头融合、英文模型组、可选模型缺失不影响中文健康、英文分片、单输出头、离线 checkpoint roundtrip 及英文缓存不能套中文 profile。
- 既有中文 smoke：8条输入 × 原十模型，共80次全文模型试跑；单窗口、EnsemJudge张量公式与分词器、CPU/CUDA、长窗、网络阻断及HTTP检测/标注均通过。默认中文结果仍使用原模型集。
- 英文：20条 × 四模型，共80次全文模型试跑，并对每条保存400/120独立分片标注。所有 token 窗口及字符片段完整覆盖，源 hash 一致。评分时屏蔽 socket.connect/create_connection，无外部推理连接。
- 英文四模型作者公式比较：最大绝对分数差为4.77e-7；Desklib独立计算mask平均池化、checkpoint Linear 与sigmoid，差为1.19e-7。CPU/CUDA 最大差为5.07e-7。
- 另用历史正文重复三次验证四模型长窗口，并验证英文检测、批量、标注HTTP路由。英文默认中文profile请求返回422，不加载模型。
- 实际本机0.5.0服务在 `127.0.0.1:8765`启动并验证：中英文HTTP检测、批量、标注，模型缓存及错误契约，另一个cwd中的中英文CLI全部通过。原受管理进程已经未运行，本次启动新服务；详见[deployment-test.json](../../reports/english-expansion-20261002/deployment-test.json)。

第一次 pytest 因既有 .pytest-tmp/.pytest_cache 属于不同账号而失败于临时目录建立；改用新的 `.runtime/pytest-english-20261002-*` 并禁用共享cache完成验证，未更改旧目录权限。依赖发出的 FastAPI TestClient 和 torch.jit.script 弃用提示没有改变本次检查结果。

完整英文报告：[summary.json](../../reports/english-expansion-20261002/english-test/summary.json)、[scores.csv](../../reports/english-expansion-20261002/english-test/scores.csv)、[原始输出](../../reports/english-expansion-20261002/english-test/raw-detection.json)；每条子目录含原文、JSON、JSONL、HTML、COMPLETE。中文回归：[chinese-regression.json](../../reports/english-expansion-20261002/chinese-regression.json)。完整报告和权重由Git排除，本说明与许可明确的公开样本可跟踪。

## 描述性结果与限制

下表为全文原始分数，门限0.90仅用于复现既有原始高分线的计数，没有用这些样本拟合：

| 模型 | human 分数范围 | human ≥0.90 | ai 分数范围 | ai ≥0.90 |
|---|---:|---:|---:|---:|
| HC3 English | 0.00028–0.32007 | 0/12 | 0.99816–0.99924 | 8/8 |
| Desklib | 0.00332–0.85529 | 0/12 | 0.99566–0.99997 | 8/8 |
| Fakespot | 0.00064–0.94231 | 1/12 | 0.99907–0.99992 | 8/8 |
| bibbbu | 0.99797–0.99975 | 12/12 | 0.99962–0.99974 | 8/8 |

在0.70线下，人工高分计数分别为0、1、2、12。bibbbu 在这批英文人写控制材料上全部高分，没有有效区分，加入任一模型分段规则会产生严重误报。可显式只选择三个新增英文专用模型，保留四模型原始结果供审阅；这项观察不能推导出其他模型的一般准确率。

英文分段规则的标签是片段分数的任一触发，全文分数是滑窗覆盖权重均值，两者不能等同。这里没有建立英文融合、概率校准或验证过的英文门限。HC3 checkpoint明确使用全部HC3训练，因此同题回答属于训练重叠 smoke；Gutenberg历史文本也未审计基座/检测训练重叠。精选20条材料不能作为独立测试、模型排名或现代生成模型泛化准确率。
