# 异构原生 AWQ 接入

`serving.py` 复用三个厂商的独立原生 AWQ 实例、现有 Gateway 和官方
`vllm-router==0.1.15`。配置集中在 `configs/serving.json`，不创建 K8s 集群或容器。
跨厂商 PP 执行器仍由原 `manage.py` 管理，本入口不将 AWQ 请求拆成跨厂商 PP。

| 入口模型名 | 服务 | 分配 |
| --- | --- | --- |
| `awq` | 官方 Router 按 power_of_two 分配 | 三个 AWQ 实例 |
| `ascend-awq` | `.24:18424` | 昇腾芯片 0/1，TP2，项目 INT4 适配器 |
| `musa-awq` | `.97:18497` | 摩尔设备 6/7，复用已有 TP2 实例 |
| `iluvatar-awq` | `.60:18460` | 天数设备 2/3，复用已有 TP2 实例 |
| `embedding` | `.24:18427` | 昇腾芯片 2，Qwen3-Embedding-0.6B BF16 |

统一地址 `http://192.168.162.60:18600`，生成用 `/v1/chat/completions`，
embedding 用 `/v1/embeddings`。昇腾容器映射八张物理卡对应的十六个芯片设备，
上表是进程分配，不代表 TP2 用满八张卡。保持 `resource_limits_enabled=false`。

AWQ 权重来源与算子差异见 `configs/native_awq.json`。摩尔和天数现有实例
显式使用 eager；昇腾新启动默认厂商图模式，并开启 cudagraph metrics。
实际图捕获／回放必须依据运行证据，不能仅凭省略 eager 参数判断。
本轮不启用原 AWQ/GPTQ 学习路由头，其标签不代表三家硬件。`auto` 不开放。
Embedding 单独提供向量接口，下载模型不等于恢复旧训练头。

## 下载与复现

官方小模型是 [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。
`.24` 的现有容器内执行以下命令，复用现有可写 `/quantization` 挂载：

```bash
python3 /workspace/shared/download_model.py \
 --model Qwen/Qwen3-Embedding-0.6B \
 --revision dd3e37eb506b3addbd930037796e25d349ae9010 \
 --destination /quantization/models/Qwen3-Embedding-0.6B \
 --report /workspace/artifacts/model-download/embedding/download.json --workers 4
```

逐文件校验大小、SHA-256 和 safetensors 结构。单文件模型与多分片索引均支持。
下载完成标志、源清单和依赖版本保留于模型目录和报告目录。
官方 Router 隔离依赖复用 `/workspace/artifacts/runtime/vllm-router`；wheel 版本、
来源与 SHA-256 见 `artifacts/dependencies/router-manifest.json`。
启动保留厂商原有 `PYTHONPATH` 和共享依赖，不重新建立环境。

工作站在仓库根目录执行：

```powershell
python heterogeneous_compute/serving.py sync --ssh-config "$env:USERPROFILE/.ssh/config"
python heterogeneous_compute/serving.py start --ssh-config "$env:USERPROFILE/.ssh/config"
python heterogeneous_compute/serving.py status --ssh-config "$env:USERPROFILE/.ssh/config"
python heterogeneous_compute/serving.py verify --ssh-config "$env:USERPROFILE/.ssh/config"
```

`sync` 仅同步本入口需要的源码、配置和测试。启动不等待长时间加载，
完整覆盖就绪后再验收。控制器在 `.60` 容器内执行健康检查和请求。
`--only router gateway` 可只操作接入层。停止本入口管理的进程：

```powershell
python heterogeneous_compute/serving.py stop --ssh-config "$env:USERPROFILE/.ssh/config"
```

默认按 Gateway、Router、embedding、昇腾 AWQ 的顺序停止，Gateway 先本地排空。
Gateway/Router/embedding 校验 boot ID、PID 启动时间与进程组，昇腾 AWQ
复用量化部署器的模型／命令校验。不恢复旧 PP 实验。
摩尔和天数 AWQ 为外部复用实例，本入口不负责重启或停止它们。

`/ready` 的 `expert_coverage_complete=true` 才表示全部配置入口就绪；
HTTP 200 也可能只表示部分服务可用。验收要求三厂商 `/workers` 健康、
各入口普通与完整 SSE 请求正确，并通过官方 Router 的 worker 计数验证小批
并发请求确实分配到三家。Embedding 检查批量数量、1024 维、有限数值、
重复输入一致性及不同输入区分。

日志、PID、状态和验收报告位于 `artifacts/serving/`，不提交原始运行产物。
功能验收不代表完成吞吐上限、持续稳定性、同条件性能比较或 K8s 生命周期验收。
