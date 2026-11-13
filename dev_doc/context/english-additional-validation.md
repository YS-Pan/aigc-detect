# 补充英文模型安装与本机验证

核对日期：2026-10-03。用户要求参考最近两天本项目的讨论，增加英文检测模型，包括本机资源能够运行的较大模型，并优先直连下载。本记录只说明安装、接口和资源检查；没有重新拟合英文融合或进行独立效果评估。

## 讨论与范围依据

已读取本项目的四个相关 chat：`调研英文 AIGC 检测模型`（`01a0fbde-0615-7020-9a5b-17a7dfb7dcd2`）、`添加英文模型并测试`（`01a0fbf5-75e4-72f3-87c4-a19480f954a1`）、`添加跨工作区调用 Skill`（`01a0fc65-af12-7122-bfd4-fae89f86db38`）、`校准英文 AIGC 检测`（`01a100b5-8dd8-72f0-8a6c-b0c35cdde99a`）。当前四模型英文 profile 在冻结测试中的12处生成段全部漏检，增加模型不能被提前描述为解决了该问题。

| 新资源 ID | 官方来源 | 本地接入 |
|---|---|---|
| `openai-base-en` | [OpenAI RoBERTa-base](https://huggingface.co/openai-community/roberta-base-openai-detector) | 125,237,762参数；GPT-2历史基线；AI为第0类；MIT |
| `openai-large-en` | [OpenAI RoBERTa-large](https://huggingface.co/openai-community/roberta-large-openai-detector) | 356,411,394参数；AI为第0类；MIT |
| `piratexx-en` | [PirateXX](https://huggingface.co/PirateXX/AI-Content-Detector) | 124,647,170参数；AI为第0类；Artistic-2.0 |
| `mage-en` | [MAGE](https://huggingface.co/yaful/MAGE)、[作者项目](https://github.com/yafuly/MAGE) | 148,660,994参数；Longformer，4096 token；AI为第0类；Apache-2.0 |
| `radar-vicuna-en` | [RADAR](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B) | 355,361,794参数；RoBERTa-large；AI为第0类；非商业使用限制；7B指训练文本生成器 |
| `gpt-neo-2.7b` | [GPT-Neo 2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B) | Fast-DetectGPT的同模型参考/评分资源；MIT |
| `falcon-7b` | [Falcon 7B](https://huggingface.co/tiiuae/falcon-7b) | Fast-DetectGPT / Binoculars参考资源；Apache-2.0 |
| `falcon-7b-instruct` | [Falcon 7B Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | 两种统计方法的评分资源；Apache-2.0 |

模型 revision、文件大小、SHA-256及标签依据的固定来源以[manifest](../../models/manifest.json)和[catalog](../../aigc_detector/catalog.py)为权威位置。本地模型卡保留，推理不启用远程代码；上游推理源码只作为接入依据，未执行。

Fast-DetectGPT公式对应作者项目commit `971b05202bac2bb504d60c0ac0812fea7a8f7c82`；Binoculars公式对应commit `c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8`。本项目独立实现公式，采用完整覆盖的窗口聚合、bfloat16模型前向与float32分块统计，归约使用float64；这些工程适配需另行验证，不搬用作者门限。两种统计量的方向不同，输出没有概率、三档判定或融合。

原[Meta Llama3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)及[Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)仓库均要求人工账号授权，本机未认证读取config返回401。因此未安装这两个受限权重；没有读取账号令牌或代为接受协议。

## 下载与环境

硬件为RTX 4060 Ti 16GB，Windows可见物理内存约47.2GiB；工作盘下载前可用约1441GiB。本工作区Python 3.13.11、PyTorch 2.11.0+cu128及Transformers 4.57.6保持原版本。只增加官方wheel `accelerate==1.15.0` 和 `psutil==7.2.2`，通过直连PyPI下载并核验其SHA-256，本地安装后 `pip check`通过；editable包更新为0.6.0。

沿用 `scripts/network.py` 的直连优先策略，分别处理每个重定向主机。`huggingface.co`直连实际失败后使用当前配置代理获取元数据/部分小文件，权重CDN `us.aws.cdn.hf.co`可直接连接。大文件采用最多8个64MiB分段请求、断点续传、精确Content-Range检查和最终完整SHA-256校验；没有改变系统代理。

## 已完成的工程检查

- 五个新增分类器在既有20条公开样本上完成100次GPU推理，另逐模型检查CPU/CUDA差异及softmax第0类公式；无网络连接尝试。最大CPU/CUDA分数差为约1.2e-7，公式差为0。
- MAGE完成超过4096 token的多窗口覆盖检查；实际HTTP应用测试覆盖新增ID、原始检测及显式旧敏感标注规则。
- 全部18个分类器在单卡串行运行，显存压力下释放最久未使用模型，CUDA峰值分配12,147,783,168字节（约11.3GiB），没有OOM。
- GPT-Neo在1218 token公开人工片段上完成离线GPU统计检测；参数2,651,307,520，CUDA峰值分配5,817,880,576字节（约5.4GiB）。其原始分数不能用于推断来源。
- Falcon 7B下载后完成512 token离线前向检查，参数6,921,720,704，CUDA峰值约10.0GiB，进程内存观测峰值约5.5GiB；默认10GiB GPU预算使部分层使用CPU，无缺失/不匹配权重。checkpoint中的embedding与输出head完整逐元素相等，支持原生模型共享权重的加载行为。
- 从`D:\project`调用真实CLI，五个新增分类器及GPT-Neo均通过，1218 token的统计结果与先前试跑相同。边下载边调用的当次分类器/Neo耗时约233/182秒，包含冷加载和磁盘争用，不能当作纯前向速度基准。
- 最终代码回归100项全部通过，`pip check`通过。首次重复检查因旧pytest临时目录ACL拒绝访问出现19项fixture错误；改用本轮工作区内的新临时目录后通过，没有删除旧目录或修改其权限。

最终安装与部署检查全部完成：新增8份资源合计44,002,660,555字节（约44.0GB）；全部21份资源再次离线SHA-256核验通过，依赖检查通过。三个统计后端分别在完整公开人工节选和生成回答上运行；每个输入只加载三次LM，两个Falcon方法共用计算，全部窗口完整覆盖且未应用门限。GPT-Neo的CPU检查实际设备映射为纯CPU；CPU与GPU模式均让Accelerate按预算分配，拒绝磁盘卸载，禁止临时state-dict卸载。

完整统计试跑中，GPT-Neo的CUDA峰值约5.4GiB；两个Falcon各约6.9GiB、进程内存观测峰值约15.0GiB。实际映射取决于当次空闲显存和CPU预算，和前述初次512 token试跑不同；记录的是分配峰值与采样到的进程RSS，不能据此给所有输入作硬上限承诺。CPU Neo观测峰值约6.5GiB。加载、模型切换及当次磁盘争用包含在耗时内，本轮不提供吞吐基准。

真实本机HTTP与项目外CLI验证通过，五个新分类器的全文分数完全一致；默认英文标注仍使用原四模型profile并保留漏检警示。API为0.6.0、18个分类器及三个统计后端均就绪；测试后重新启动本轮创建的实例，清除测试模型缓存，保持按需加载。接口仅监听127.0.0.1:8765。

详细本机证据位于[运行目录](../../reports/english-additional-20261003/)：`smoke/classifiers.json`、`smoke/mage-long-coverage.json`、`all-classifier-memory.json`、`neo-first-gpu.json`、`falcon-first-gpu.json`、三个统计方法的逐输入JSON、`smoke/neo-cpu.json`、`verify-all.log`、`pytest-memory-budget-final.log`及`deployment.json`。完整结果与上游快照均由Git排除；可版本管理的[轻量摘要](../../evaluations/english-additional-20261003-installation/README.md)只保存安装和资源证据，不含正文。

## 效果边界

公开样本只用于安装试跑，包含HC3训练来源，未审计其与其他模型训练数据的重叠，没有拟合门限，也不是独立准确率基准。更大的分类器仍可在人工材料上给出高分或漏掉生成材料；模型数量、参数规模、训练领域、生成模型覆盖分别影响结果，不能仅据安装或试跑证明英文鉴别可靠。

原13个分类器的manifest条目与三个融合profile逐项/逐文件核对保持不变，默认中文十模型和英文四模型不变。新增五个分类器通过显式ID调用，因果语言模型只进入独立统计后端；旧英文profile的全部漏检警示保留。后续效果验证见[待办](../toDo/english-calibration-validation.md)，本次没有执行训练、替换旧权重或发送待测正文到外部服务。
