# 共用模型下载与分发

`transfer_image.py --config <配置>` 直接将已缓存 Docker 镜像通过固定主机密钥的
SSH 流传到对端 docker load，不落地镜像 tar；核对 image ID 后清理本次临时
授权和密钥。它只确认传输与镜像身份，不能替代容器运行、依赖和模型验收。
同厂商 TP/PP 的准备入口复用此工具，详见
`homogeneous/experiments/pp_baseline/README.md`。

`download_model.py` 复用节点已有 Python/curl，从 ModelScope 的不可变 revision 读取文件清单与 SHA256。可用 `--mirror hf-mirror` 优先下载国内镜像的 safetensors，失败时使用官方 ModelScope；两种来源均必须通过官方清单的大小与哈希检查。元数据始终从固定版本获取。支持已有 Python 3.7 环境、断点续传及多文件并发，不安装新环境。

完成权重结构验收后写 `download_manifest.json`、`SHA256SUMS`、`DOWNLOAD_COMPLETE`。下载日志和报告放在 `artifacts/downloads/`，模型留在用户模型目录。
默认验收原始 BF16；下载标准量化权重时显式指定 `--quantization awq` 或 `--quantization gptq`，检查 W4/group128 配置、打包 dtype、投影数量与索引完整性，不把 INT4 模型当成浮点模型验收。下载校验不等于后端推理验收。

`distribute_model.py --config <配置>` 等待下载完成标志，再用 SCP 分发。配置包含 source_directory、task_directory、report_directory、download_report、task_name、source_ip、workers 和 peers；每个 peer 为 `[别名, 内网IP, SSH端口, 已信任的HostKeyAlias]`。使用 `--check` 只验证节点认证。运行配置和已信任 known_hosts 放在忽略的 artifacts 中，不提交认证信息。

目标目录为 `~/models/<模型名>`；磁盘不足时可预先建立用户专属数据盘目录及模型目录软链接。已有完成模型仅允许校验复用；未完成目录必须带相同任务的 `TRANSFER_IN_PROGRESS` 标志才可继续。逐文件复用已校验内容，SCP 后再次验证 SHA256，所有文件通过后才复制完成标志。单节点失败写入报告，不阻塞其他节点。

`transfer_auth.py` 复用原分发任务的限源、restrict 临时公钥认证方式。私钥只在下载源节点生成，不传出节点。任务结束只移除本任务授权及临时私钥，保留模型、配置和必要日志，不修改其他容器或服务。

Qwen3-8B/32B 当前分发使用各 3 个并发节点；源下载各 8 个并发文件。下载、分发后台执行，不依赖本地窗口持续连接。源节点 `artifacts/distribution/<模型名>/status.json` 记录完成或失败状态。此轮启动、SSH 认证及 64 MiB SCP 速度探测已验证；全量传输结果待后台完成。
