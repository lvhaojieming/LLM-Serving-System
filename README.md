# LLM-Serving-System

工程按同构部署与异构算力适配分开组织，共用模型下载工具和一份 Git 历史。

| 目录 | 职责 | 状态 |
|---|---|---|
| [`homogeneous/`](homogeneous/README.md) | 已完成的昇腾同构集群、实例扩缩容、Gateway、官方 vLLM Router、AWQ/GPTQ 专家池及同厂商 TP/PP 实验 | 保留现有实现与配置 |
| [`heterogeneous_compute/`](heterogeneous_compute/README.md) | 不同厂商算力适配、跨厂商 TP2/PP2 执行、通信探针、各厂商基线配置及性能采集 | 昇腾 + 摩尔 BF16 功能验收通过；性能优化待完成 |
| [`shared/`](shared/README.md) | 可复用的模型/数据下载工具 | 不含模型权重 |

同构与异构按**是否跨硬件厂商**区分。同厂商跨节点 TP/PP 仍属于同构；
不同厂商各跑一个独立模型实例的对照属于多厂商基线，不代表跨厂商 PP。
当前真正跨厂商 PP 为昇腾前 20 层 + 摩尔后 20 层，两端各自 TP=2。

## 同构工程

```bash
cd homogeneous
python scripts/control.py
python -m pytest tests experiments/bf16_distributed/test_distributed.py
```

同构包的安装入口为 `homogeneous/pyproject.toml`；需要安装时从仓库根目录执行
`python -m pip install -e ./homogeneous`。原 `heteroserve` 包名和管理器参数不变。
配置草稿与本地产物分别位于 `homogeneous/state/`、`homogeneous/artifacts/`。
已有宿主机的固定容器、Kubernetes 工作负载、持久化路径和暂停状态不因目录整理改变。

## 异构工程

```bash
python heterogeneous_compute/manage.py status --ssh-config <cluster_ssh_config>
python heterogeneous_compute/manage.py start --ssh-config <cluster_ssh_config>
python heterogeneous_compute/validate.py --ssh-config <cluster_ssh_config>
python -m unittest discover -s heterogeneous_compute/tests -v
```

配置入口是 `heterogeneous_compute/configs/ascend_musa.json`。
当前执行器复用 vLLM 模型组件，采用 CPU 中转、参考注意力和串行 FIFO；
尚非原生 vLLM 引擎跨厂商组网，不具备连续批处理、流水线重叠或正式速度验收。
支持范围、镜像、依赖、启动/排空命令和复现步骤见对应目录 README。

Git 只保留代码、配置、文档和必要结果摘要。模型、数据集、原始请求日志、
checkpoint、运行状态及凭据均不提交；有效原始结果继续在本地 `artifacts/` 中保存。
