# 接口模块

实现：[api.py](../../aigc_detector/api.py)、[cli.py](../../aigc_detector/cli.py)、[client.py](../../aigc_detector/client.py)。字段的权威说明在 [用户 API 文档](../../user_doc/api.md)。

其它工作区的常规 agent 调用契约在 [aigc-detect skill](../../.agents/skills/aigc-detect/SKILL.md)，随发行包提供。接口、语言组、默认策略、保存或启停行为变化时同步 skill 内的命令和约束，确保常规任务仍可只读 skill 完成；少数安装、异常和证据追溯才导航回用户指南。

FastAPI 同步端点在线程池运行，单例 Detector 用 RLock 串行推理。请求层禁止额外字段，限制字符数和 batch。内部非法参数转换为 422，缺失模型转换为 503；未知运行故障保留 500，不伪造分数。

模型列表上限随len(CATALOG)计算，当前22。原始detect/batch省略名单时按language选择中文十模型或英文四模型；annotate省略名单时使用profile锁定名单，英文默认新五分类器。CLI使用--language，客户端转发同名选项。health的status按中文原始组判断，model_groups返回原始组；annotation_profiles按语言返回默认策略、hash、名单、分片及安装revision就绪。更新代码后重启空闲常驻服务。

annotate及/v1/annotate中文默认thesis-cnki-20260929-v2，完整十模型/400/120；英文默认english-academic-20261003-v2，完整新五分类器/1600/400。分片和门限为null/省略时读profile，仍校验固定revision和语言边界。英文两头共用同一函数、门限不同，报告明确3/12生成段命中和20.38%混合生成字符覆盖来自已知论文重复评估，仍有严重漏检。旧英文v1可显式复现。已有英文融合缓存重组省略policy时保留原策略，旧原始规则缓存选当前默认；名单/分片/revision不匹配须重新推理。任意子集标注仍需显式旧任一规则，默认原始门限0.90/0.70。

CLI从UTF-8文件/参数/stdin读取正文，成功输出JSON，失败stderr JSON及退出码1。HTTP固定本机地址/单worker，默认端口8766；标准库client关闭代理继承。recombine读取有效COMPLETE、report.json/source.txt并校验hash、名单、revision和分片，不加载模型，目标目录必须不存在。HTTP没有重组端点。软件版本0.10.0，标注schema仍为aigc-annotation/1.2。

英文实验非线性v3通过显式policy=english-academic-20261004-v3调用，完整五分类器/1600/400与schema4.0；默认保留线性v2。已有v3缓存省略重组策略时保留v3；相同名单/revision/分片的v2缓存可显式改用v3，原始分数不变。复测下降与优化范围见[实验验证](../context/english-nonlinear-validation.md)。

PowerShell start-api 支持后台/前台模式，记录所选 Python 和服务 worker 的 PID、启动时间、可执行文件路径；后台窗口隐藏，日志与状态进入用户目录。stop-api 逐个匹配身份后停止已验证进程，避免误停其他 Python 作业。worker PID 来自健康检查。普通安装可直接通过 CLI 前台启动，无需这些源码脚本。

statistical-detect CLI使用独立methods参数；两个Falcon方法共用前向，512/64及内存预算随结果记录。Python为StatisticalDetector，可批量复用LM加载，输入上下文独立；返回原始统计量，不给单独门限。HTTP没有独立统计推理端点。schema3.0标注支持固定配置的统计信号，但本轮默认v2最终只选五分类器；这些分类器也可显式通过原detect接口调用。

0.9.0新增specialized_english_models清单及多分类结果class_scores、ai_score_classes、score_semantics。detect/annotate请求字段不变，四模型显式ID可选，默认profile不变。类别分数在窗口、标注片段和重组结果保留；聚合语义进入返回说明。见[接入记录](../context/english-specialized-validation.md)。
