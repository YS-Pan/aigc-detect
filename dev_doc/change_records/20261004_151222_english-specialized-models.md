# 专业英文与人机协作模型接入（0.9.0）

已联网核对并下载FollowSci学术段落BERT、NTNU学术摘要RoBERTa、LLM-DetectAIve四分类和CoCoDet三分类，共29文件、约2.10GB。新增固定Space子目录和外部基座配置下载支持，保留revision、文件哈希、来源和许可说明。推理全部本地，不执行上游仓库代码。

完成ModernBERT余弦头、全部辅助头严格加载、CLS/SEP分窗适配；多分类的类别分数和聚合语义通过Python/CLI/HTTP/标注及缓存重组保留。登记22分类器，四个新模型显式选择；英文默认detect及v2融合保持，没有重新训练或拟合门限。

实际验证：137项回归、29文件SHA-256、pip check、10官方示例共40次离线模型推理、四模型CPU/CUDA及公式、长文完整覆盖、真实HTTP/跨cwd CLI、标注保存/重组通过。本地editable包和已确认空闲的受管理API更新为0.9.0。依赖未增加或升级。

初次真实验证定位并修复generic tokenizer遗漏ModernBERT特殊符号的问题；最终单窗分数与直接作者公式相符。旧pytest临时目录权限问题通过新测试目录解决。

本次为安装验证，不是目标论文准确率测试；尚未证明零人工窗口误报与50%混合段召回。说明与证据见[来源和验证](../context/english-specialized-validation.md)、[轻量证据](../../evaluations/english-specialized-20261004-installation/README.md)、[调用指南](../../user_doc/english-detection.md)。后续效果评估仍见[英文待办](../toDo/english-calibration-validation.md)。
