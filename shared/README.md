# 共用工具

`download_model.py` 提供可复用的模型/数据文件下载、断点处理与校验功能。
同构和异构工程共用此入口，不复制下载器，也不把权重或数据集放入 Git。

从仓库根目录执行 `python shared/download_model.py --help` 查看参数。
现有模型目录不因源码重新组织而搬迁或重新下载。
