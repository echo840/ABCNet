# 预训练模型迁移指南

由于在新版本中我们对模型的结构进行了大量的重构和修复，MMOCR 1.x 并不能直接读入旧版的预训练权重。我们在网站上同步更新了所有模型的预训练权重和log，供有需要的用户使用。

此外，我们正在进行针对文本检测任务的权重迁移工具的开发，并计划于近期版本内发布。由于文本识别和关键信息提取模型改动过大，且迁移是有损的，我们暂时不计划作相应支持。如果您有具体的需求，欢迎通过 [Issue](https://github.com/open-mmlab/mmocr/issues) 向我们提问。
