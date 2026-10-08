# LLM Serving System

日常统一管理入口：`python3 scripts/control.py`。启动直接显示实例工作台，输入 `awq-01` 或 `gptq-01` 编辑；`+` 新增向导，`c` 保存，`u` 发布，`jobs` 查看任务。草稿自动保留并恢复，运行/保存/待修改值分开显示；节点、流量和高级操作继续复用现有六组功能。
参数清单、修改、部署和验收用法见 [总控操作说明](docs/control.md)。

面向昇腾的 Kubernetes 推理工程。实例生命周期由 Kubernetes 管理，负载均衡与动态发现复用官方 vLLM Router。
当前 AWQ 池有四个模型副本，分别运行在 `.209`、`.210`、`.211`、`.216`；GPTQ 池有一个模型副本，运行在 `.216` 的另一张 NPU。
五个模型现分别为 `awq-01`～`awq-04`、`gptq-01`，共用部署模板，拥有独立 Deployment 和实例参数。
AWQ/GPTQ 的单节点 TP2、PP2 及 TP2/PP2 组合已实测普通/SSE 推理；当前生产配置保持 TP1/PP1 基线，跨节点并行后置。
每个专家池使用一个官方二级 Router 维护本池入口计数；同池跨 Router 副本协调后置。
一级概率 Router 已从 `MLsys_inference` 迁入 `heteroserve.routing`，Gateway 通过池 Service 接入二级路由。
入口配置为 `deploy/gateway.json`，AWQ 与 GPTQ 两个池均已启用；禁用专家时明确返回 503，不自动改选。
应用只提供真实模型准入、健康检查、排空和发现一致性接口，不重新实现推理引擎或负载均衡算法。

实验采用独立的跨物理主机 K3s 容器集群，管理凭据由实验控制平面生成。
原共享集群的 kubelet、凭据和暂停服务保持原状。实验基础设施长期复用，参数通过配置调整。

| 入口 | 用途 |
|---|---|
| `scripts/control.py` | 配置草稿、集中保存、CLI、回退及稳定实例操作 |
| `scripts/control_menu.py` | 六组中文操作界面和明确的专家/实例上下文 |
| `scripts/manage_instances.py` | 独立工作负载、单实例发布、资源检查及实际卡号查询 |
| `scripts/manage_nodes.py` | 复用现有管理器的节点容量检查、排空、保留容器退出与重新接入 |
| `docs/control.md` | 总控参数、节点操作、保护条件及失败恢复说明 |
| `deploy/lab/cluster.json` | 五机实验集群（四个推理节点及 `.217` 控制节点）的固定容器 ID、镜像、端口及网段 |
| `scripts/manage_lab.py` | 预检、创建或复用实验节点、查询状态与跨机验收 |
| `deploy/lab/npu.json` | 固定实验节点的昇腾镜像与验证资源预算 |
| `scripts/manage_npu.py` | 驱动占用预检、NPU 分配与计算验收、受限清理 |
| `deploy/k8s/` | 通用项目权限、网络与暂停状态的 NPU 验证模板 |
| `scripts/check_kubernetes.py` | Kustomize 渲染、固定 OpenAPI schema 与部署约束检查 |
| `docs/deployment.md` | 部署边界、权限、版本和验收步骤 |
| `deploy/pools/ascend-{awq,gptq}.json` | 各专家的模型版本、副本、节点及独立推理参数 |
| `deploy/pools/defaults.json` | 两个专家池共用的部署默认值及 Router 依赖 |
| `deploy/k8s/pools/` | 可渲染的专家与 vLLM Router 部署入口 |
| `scripts/manage_pool.py` | 渲染、发布、复用 Router 依赖及真实模型验收 |
| `scripts/verify_lifecycle.py` | 扩缩容、发现、恢复、排空和维护验收 |
| `deploy/monitoring/` | 一套固定镜像的 Prometheus/Grafana/kube-state-metrics |

在获准使用节点且具备宿主机容器权限的管理机中运行：

```bash
python3 scripts/manage_lab.py plan
python3 scripts/manage_lab.py preflight
python3 scripts/manage_lab.py up
python3 scripts/manage_lab.py status
python3 scripts/manage_lab.py verify
```

`verify` 只在实验集群的验证命名空间创建 CPU HTTP 探针，验证跨机 Pod IP、Service/DNS 和 Deployment 重建。
它不启动已有推理服务，也不证明 NPU 分配或真实模型推理已经通过。

五节点网络及四个推理节点的 NPU 分配和计算已验证。真实 AWQ 专家的普通/SSE 推理、
双副本发现、扩缩容、Pod 删除恢复、API 读取失败时的缓存过期和在途 SSE 排空均已实际验收。
详细范围、故障恢复时间和仍未验证的项目见 `docs/deployment.md`；不能把实验通过当作所有目标集群的生产认证。

当前实验资源策略为 `resource_limits_enabled=false`：外层容器的 CPU、内存及进程上限和项目 Pod 的
CPU/内存 limits 已取消，项目验证配额已删除；NPU 按卡分配保留。原地操作入口为 `scripts/manage_lab.py unlimit`。

本地验证复用已有 Python 环境，开发依赖见 `requirements-dev.txt`。

```bash
python -m pytest -q tests
```

不得将 CPU 验收、静态 YAML 校验或普通 Node Ready 当作昇腾生产部署验收。

专家池操作在已有部署目录执行，不需要重建镜像或模型容器：

```bash
python3 scripts/manage_pool.py prepare-router
python3 scripts/manage_pool.py apply
python3 scripts/manage_pool.py router
python3 scripts/manage_pool.py verify
python3 scripts/verify_lifecycle.py routing
```

`worker.py` 在实例中管理已存在的 vLLM 后端，启动后必须真实生成验收答案才能 Ready。
`router.py` 使用官方 vLLM Router 的数据面和 worker 管理 API；补充 EndpointSlice/Pod UID 核对、
有限发现缓存和请求准入。所有服务使用内部 Service，不默认公开推理端口。
