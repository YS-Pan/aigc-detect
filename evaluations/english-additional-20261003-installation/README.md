# 英文补充模型安装证据

2026-10-03，0.6.0。[summary.json](summary.json)保存新增八份资源的repo/revision、大小、文件数、实现哈希、CPU/CUDA检查、内存观测、网络路由和原配置不变证据，不保存正文。

五个补充分类器为OpenAI RoBERTa base/large、PirateXX、MAGE、RADAR；另有GPT-Neo 2.7B和Falcon 7B/7B-Instruct，供三个统计后端使用。全部21份资源SHA-256复查、100项回归、100次公开样本分类器推理、三个统计后端在两种公开输入上的离线调用、CPU Neo及实际HTTP/跨目录CLI检查通过。

这份摘要是安装与接口证据，不是独立准确率、误报率或校准证据。既有公开样本可能与训练集重叠；没有重拟合融合，原13个模型条目与三个profile不变，默认英文仍为四模型。三种统计后端只输出原始统计量，不沿用作者门限。

来源、许可和实际限制见[验证说明](../../dev_doc/context/english-additional-validation.md)，调用见[英文指南](../../user_doc/english-detection.md)。完整本机运行产物和上游快照在Git排除的[reports目录](../../reports/english-additional-20261003/)。
