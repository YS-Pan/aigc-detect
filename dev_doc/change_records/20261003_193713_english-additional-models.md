# 补充英文模型与本地大模型统计

完成日期：2026-10-03。根据用户要求读取最近两天本项目的相关chat，新增五个英文分类器及三份较大因果语言模型权重，总新增约44GB。登记18个分类器与三个统计后端，包和本机API更新为0.6.0。

OpenAI base/large、PirateXX、MAGE和RADAR通过显式ID使用；MAGE支持4096 token及batch=1，旧OpenAI pooler由本地结构完整加载。新增GPT-Neo 2.7B、Falcon 7B/7B-Instruct，用于Fast-DetectGPT与Binoculars原始统计；顺序加载、bfloat16、GPU/CPU预算分配、临时CPU logits缓存和完整窗口覆盖，不搬用上游门限，不启用远程模型代码。两个Falcon方法同次请求共用前向。

下载继续直连优先，按重定向主机分别尝试；元数据主站失败后回退当前代理，主要权重CDN直连。分片索引、64MiB范围续传和最多8并发经过实际下载及SHA验证；不改系统代理。仅增加Accelerate 1.15.0与psutil 7.2.2，官方wheel直连并核验后本地安装。原Llama3 8B仓库要求账号授权、未认证返回401，本轮未安装。

实际完成21份资源哈希复查、100项代码回归、五分类器100次公开样本GPU调用及CPU/CUDA/作者类公式检查、MAGE长窗、18分类器缓存释放、三个统计后端两种公开输入及CPU Neo、真实HTTP与项目外CLI。CPU预算模式使用自动设备映射；原13模型条目和三个profile完全保持，中文十模型、英文四模型默认列表不变。API测试后清理本轮测试缓存并恢复按需加载。

同步用户指南、架构、模块、术语和跨工作区共享skill，检查链接与UTF-8；完整结果继续Git排除。实现、来源、资源峰值、限制和纯安装摘要见[验证说明](../context/english-additional-validation.md)及[轻量证据](../../evaluations/english-additional-20261003-installation/README.md)。公开样本有训练重叠风险，本轮没有独立效果评估、门限拟合或新融合，不宣称解决了英文AIGC鉴别。
