# 跨机器 Kubernetes 部署与验收

当前工程已接入三台物理服务器的独立 K3s 实验节点，使用真实昇腾模型进行准入和生命周期验收。
Kubernetes 管理节点、设备资源及 Deployment；官方 vLLM Router 负责负载均衡和原生 Pod 发现。
应用接口补充真实模型健康、在途排空、Pod UID 核对及有限发现缓存。

## 实验与现有集群的职责

现有共享集群使用其原有 kubelet 和管理凭据。实验控制平面和 worker 在独立 Docker 容器中运行，
使用独立数据卷、API 端口、容器网络和 Pod/Service 网段。所有 kubectl 操作显式指向实验控制平面中的 kubeconfig。

这条路线需要获准使用宿主机，并有创建 privileged 实验节点容器的权限。
它不需要原共享集群的应用部署权限，也不会赋予原共享集群的权限。
privileged 容器不能作为对宿主机管理员的安全隔离边界；设备归属必须单独确认。

用户已指定 `.209` 至 `.217`，共九台候选节点。当前 `.217` 为控制节点，`.209`、`.210` 为推理节点。
三只节点容器长期复用并锁定 ID；外层 CPU、内存、swap、进程上限和项目 CPU/内存 limits 已按用户要求取消。
原 4 CPU、8 GiB 只保留为创建预算记录。应用保留合理的调度 requests 和一实例一张 NPU 的 requests/limits。
模型缓存临时卷有 20 GiB 上限；监控数据保留 7 天、TSDB 数据规模上限 5 GB。

## 版本与已知边界

实验版本固定为 `v1.34.12+k3s1`，ARM64 镜像固定平台 manifest digest 和 image config digest。
这些值由 K3s 发布页、Docker Hub 平台记录以及 `.217` 的真实镜像检查交叉确定。
镜像配置 ID 允许复用经校验的 Docker archive，避免在每台节点重复等待外网下载。

这是兼容性实验基线，不是长期生产版本选择。Kubernetes 1.34 上游支持在 2026-10-27 结束。
目前实验节点实际内置 containerd 为 `2.2.7-k3s1`；不能用宿主机的 `1.6.16` 或报告中的候选 `2.1.4` 代替它。
已实测设备插件、非特权模型 Pod、真实推理及故障恢复；这不等于厂商已认证整个嵌套版本矩阵。
正式生产集群仍须确认发行版支持周期、驱动/固件/CANN/runtime 对应矩阵及目标集群验收。

目前九台候选节点中，`.212` 使用 cgroup v2，其余使用 cgroup v1。
Usernetes rootless 路线尚不满足大多数节点的前置条件，因此当前评估 K3s 的 rootful 容器路线。
K3s 官方提供 Docker 中运行 server/agent 的方式；这不等于官方认证了本项目的跨机网络和嵌套 Ascend 组合。

## 可重复的验收顺序

1. 预检真实路由、Docker 网段、端口、可用内存及已有资源归属。遇到冲突停止，不覆盖宿主机配置。
2. 创建或复用两个实验节点，确认它们分别关联 `.217` 和 `.209` 的真实地址，全部 Ready。
3. API server dry-run 后部署两个受限 CPU 探针，分别固定到两个实验 worker 环境。
4. 双向访问对端 Pod IP 和 Service DNS，校验响应中的 Pod UID、节点名和实际 Pod IP。
5. 删除一个验证 Pod，确认 Deployment 创建新 UID，并重新通过 Ready 检查。
6. 确认 NPU 归属和可用卡，选择经过验证的设备插件/runtime组合，验证设备发现、分配、隔离和小矩阵计算。
7. 部署一个真实模型，验证身份、非流式与完整流式输出、取消、排空和重建后的重新验收。
8. 扩大节点和副本数量，执行资源预算、故障恢复、滚动更新及目标集群验收。

第 2～5 步通过后，只能声明跨物理主机的 Kubernetes 基础控制与网络链路通过。
第 6～7 步通过前，不声明具备已验证的 NPU 模型控制能力。

`verify` 的完整机器可读结果写入 `artifacts/kubernetes/lab-verification.json`，运行产物不提交到仓库。
必要的验收摘要可以写入正式文档。首次镜像传输失败或 API 不可用时，保留具体诊断，不伪造通过状态。

## 操作约束

已有模型容器、权重、checkpoint 与暂停进程保持原状。管理脚本只能创建或复用带有本项目归属标签的实验节点，
遇到同名外部资源或配置不匹配时拒绝执行。`up` 不会自动重启已停止的实验节点。
不得挂载原 `/etc/kubernetes`、原 `/var/lib/kubelet` 或 Docker socket 给实验节点。
加入令牌通过标准输入传递，私有状态目录权限为 0700，令牌文件为 0600，日志和 Git 中均不保存令牌。

NPU 资源不能由“节点可登录”推定为空闲，也不能把 Docker 中其他任务占用的卡视为 Kubernetes 未分配的空闲卡。
裸机上电、固件和驱动维护属于宿主机运维范围；Kubernetes 的实例管理权限不自动包含这些能力。

## 已取得的结果 2026-10-07

- `.217` 与 `.209` 的独立实验节点均已 Ready，实验控制平面内的 CoreDNS 正常。
- 两个 CPU HTTP 探针已部署并就绪。初始跨节点 TCP 请求超时，而服务器地址和对端 Pod 的 ICMP 测试成功。
  关闭实验节点 `flannel.1` 的 `tx-checksum-ip-generic` 后，双向 Pod IP、双向 Service/DNS 请求及 Pod 删除重建均通过。
  该对照结果支持嵌套 VXLAN 的校验和卸载兼容问题；没有改动宿主机物理网卡。
  修复由仅限实验节点网络命名空间的 DaemonSet 维护，禁用 API token，仅申请 NET_ADMIN，不挂载宿主机文件。
  两个维护 Pod 均为 Running/Ready；在 `.209` 实验接口重新打开该卸载后，7 秒复查已由维护 Pod 自动关闭，
  随后再次通过全部四项跨机请求与 Pod 删除重建验收。
- 实验容器中调整反向路径过滤没有解决超时，已撤销该实验设置；宿主机配置未改动。
- `.209` 复用 `moqe-loss-expert`，在进程环境中仅选择物理 0 号 NPU，执行确定性小矩阵测试。
  PyTorch 与 torch-npu 均为 2.5.1，设备识别为 Ascend910B4，进程可见设备数为 1。
  精确矩阵结果和 CPU 参考结果均通过，最大绝对误差为 0；退出后 npu-smi 未发现该测试遗留进程。
  这是已有容器的硬件计算验收，不是 Kubernetes NPU 分配、容器设备隔离或模型推理验收。
- `.217` 存在训练账号运行的 `reserve_npus.py`，涉及全部八张卡；其归属和预留没有修改。
- 原共享集群和原来暂停的模型服务保持原状。

### NPU Pod 设备分配与计算验收：已通过

复用 `.209` 的现有设备插件镜像和模型镜像，导入实验节点的 containerd，未重建外层 Docker 容器。
设备插件在实验集群识别八张健康卡，节点公布 `huawei.com/Ascend910` capacity/allocatable 均为 8。
两只非 privileged 验证 Pod 各请求一张卡，取得不同物理设备；其余七张卡的设备打开检查均被拒绝。
初次两只 Pod 的 torch-npu 设备初始化失败，`drvGetDevNum` 返回 87。

检查 `.209` 宿主机内核日志发现 `uda_occupy_dev_by_ns: Conflict open udevid`，时间与验证进程吻合。
宿主机头文件确认 87 为 `DRV_ERROR_RESOURCE_OCCUPIED`。`/proc/uda/namespace_node` 的独占映射
`ns_id=0` 将全部八张卡关联到 `root_tgid=2679173`，对应旧容器 `vllm-ascend`（ID 前缀 `474713d061e6`）。
该容器仅有四个 Bash 进程，仍保留驱动独占设备映射；没有模型计算进程不等于设备已释放。
取消进程可见卡过滤、以及临时让验证 Pod 共用实验节点 PID 命名空间，都没有解决失败。
共用 PID 命名空间的对照不是正式部署配置，对照结束后删除了这两只失败验证 Pod。
用户确认该旧容器归属并授权停止后，停止了它，保留原容器、文件和 ID；独占映射随之消失。
普通 Pod 的错误 87 随之消失，证明旧容器设备占用是本次失败原因。
没有安装新的昇腾 runtime、重装驱动、重置 NPU 或改动原共享集群。

后续冷编译暴露出两个独立的配置问题：CANN 尝试访问 `/root/.tvm_test_data` 被拒绝，
以及 3 GiB 的验证 Pod 内存限额触发 OOMKilled。修复为每只 Pod 独立的 `HOME=/tmp` 和缓存目录，
设置 `TE_PARALLEL_COMPILER=1`、`OMP_NUM_THREADS=2`，验证 Pod 内存限额为 6 GiB。
为保持外层实验节点的 8 GiB 配置不变，两只 Pod 同时持有设备分配，依次执行计算。
不据此声明同时冷编译、并行模型吞吐或完整的多租户安全隔离已经验收。

2026-10-07 16:06（北京时间）的验收结果：两只普通 Pod 分别获分配物理 5 号和 2 号卡，
torch/torch-npu 均为 2.5.1，Ascend910B4，进程可见设备数均为 1，其他七张设备打开均被拒绝。
精确矩阵测试及三次 CPU 参考比较均通过，最大绝对误差为 0。
Pod 不使用 privileged、hostPID 或 hostNetwork，删除测试 Pod 后释放分配。
机器可读结果保留在 `artifacts/kubernetes/npu-isolation.json`，不提交运行产物。
上述是先期设备验收。随后真实模型 Pod 已部署并通过普通及完整 SSE 推理，三只外层容器均未重建。

### 复现 NPU 验证

当前资源策略已按用户要求设为 `resource_limits_enabled=false`：两只固定 Docker 容器原地取消
CPU、内存、swap 总量及进程数上限，项目 Pod 模板取消 CPU/内存 limits 和 emptyDir 大小限制，
删除项目验证命名空间的 `validation-budget` 配额。调度 requests、NPU 按卡分配与设备隔离保留。
K3s 系统组件的默认资源设置、节点 Pod 容量及物理硬件容量不属于本次项目配额修改。
`cluster.json` 中的 4 核、8 GiB 是原创建预算记录；当前策略为 false，不会应用这些上限。
Docker 20.10 的零值更新会保留旧配置；原地更新使用 `CPUQuota=-1` 和 cgroup v1 内核无上限内存值。
两台节点实际 cgroup 已核实：`cpu.cfs_quota_us=-1`，内存及 memory+swap 均为
`9223372036854771712`，`pids.max=max`；这是内核的无上限表示值。
Docker 的旧 `NanoCpus` 字段可能仍记录创建时的值，判断当前 CPU 上限应检查实际 cgroup 及 `CpuQuota`。
前述 6 GiB、依次冷编译是首次成功验收的条件；目前验证计算顺序仍为依次执行，但不再强制该内存上限。
原地取消 Docker 上限的可复现入口为 `python3 scripts/manage_lab.py unlimit`，随后执行 `verify` 和
`python3 scripts/manage_npu.py plugin` 同步项目模板。不会重建外层容器或启动原暂停的模型服务。

在已有正式部署目录 `/root/zhangjinhao/LLM-Serving-System` 操作，复用设备插件、模型镜像和驱动：

```bash
python3 scripts/manage_npu.py preflight
python3 scripts/manage_npu.py status
python3 scripts/manage_npu.py isolation
python3 scripts/manage_npu.py isolation-results
python3 scripts/manage_npu.py cleanup
python3 scripts/manage_npu.py preflight
```

参数保存在 `deploy/lab/npu.json`。`isolation-results` 会先确认两只 Pod 就绪并持有分配，
再依次启动计算，校验返回结果的 Pod UID；拒绝将 privileged 或共用 PID 命名空间的对照 Pod 作为通过结果。
`preflight` 检查驱动独占映射，发现占用或映射格式异常则失败，不自动停止任何容器或重置设备。
复现前确认卡的归属及其他宿主机任务；测试结束执行 `cleanup`，只删除归属标签匹配的两只验证 Pod。

校验和设置属于本次嵌套实验环境的配置，不应无条件复制到其他生产集群。
DaemonSet 使用实验节点已有的固定 K3s 镜像和 ethtool，节点重启后重新执行接口检查。

可重复的已有容器硬件检查入口是 `scripts/check_npu.py`。
使用该入口前应确认设备归属和占用，限制当前进程的可见设备，并复用匹配的 CANN / torch-npu 环境。
不要根据微型矩阵迭代时间推断模型吞吐或延迟。

## 当前专家与路由部署

`deploy/pools/ascend-awq.json` 定义当前专家池，`deploy/lab/npu.json` 固定已复用的运行镜像及 vLLM 参数。
模型是 Qwen3-14B-AWQ 经既有工具转换的 `moqe_ascend_int4` 布局，转换清单 SHA-256 为
`70a289a9eb8ed35d8a62aa466738c2cd5906df68df844c7731c98840230654f9`；两台目标节点的六个权重分片均逐一校验。
模型和适配器在目标节点预先准备，再以只读挂载提供给 Pod。Kubernetes 不负责自动复制权重。

当前两个模型副本和一个二级路由副本使用内部 Service。每个模型 Pod 申请 `huawei.com/Ascend910: 1`，
CPU/内存使用调度 requests（4 CPU、16 GiB），不恢复已经取消的 CPU/内存 limits。
路由使用官方 ARM64 `vllm-router==0.1.15`，wheel 及两个缺失依赖的版本和 SHA-256 均固定在池配置。
依赖通过校验后发布到现有节点卷，只补缺失项，不修改原模型环境，不构建新镜像。

`worker.py` 启动已存在的 vLLM 后端，并验证模型身份和实际生成结果后才 Ready。
请求容量上限为每实例 8 个在途请求；vLLM 引擎同时执行序列数为 2，其余允许请求在引擎内等待。
排空立即拒绝新请求；在途 SSE 继续处理，240 秒预算到期时明确取消，容器退出总预算为 300 秒。
首次启动预算为 1200 秒。健康检查失败使实例先退出就绪，再由 liveness 触发恢复；正常排队不直接触发重启。

`router.py` 不实现负载均衡算法。官方 Router 负责请求转发、Power-of-two 选择、原生 Pod 发现及指标。
补充接口对照 Pod UID 与 EndpointSlice 就绪/退出状态，发现读取失败后保留最多 15 秒缓存，
到期拒绝新分流；既有请求继续执行。它也通过官方 `/workers` 管理 API 清除已失效注册地址。
节点失联实验实际发现原生注册地址仍可能残留，因此保留这项一致性补充，不将它伪装成原生内置保证。

驱动及本地模型需要精确的只读 hostPath，项目 Namespace 对 Pod Security admission 采用必要例外。
模型和 Router 容器均非 privileged，不使用 hostPID/hostNetwork，禁止权限提升并移除额外 capabilities。
Router 的 ServiceAccount 仅有项目 Namespace 内服务发现读取权限，无 Secret 读取或工作负载写权限。

当前 `router_replicas: 1`，同一进程统一维护池入口在途计数，模型端仍逐实例执行 8 请求上限。
这不提供 Router 入口高可用；Router 故障或更新期间入口会中断。重启后的入口计数从零开始，
模型端仍保护已有请求的实例容量；跨副本协调、持久化计数和 token 工作量统计留到后续实现。
下表节点失联时仍有 Router 接流量的记录来自此前双 Router 验收，不能作为当前单 Router 的高可用保证。
生产集群如采用经过认证的设备 runtime、CSI/PVC，可替换本地存储入口并收紧 Namespace 策略。

## 实测验收记录

以下时间是本次实验观测值，不是通用恢复时间或性能承诺。机器可读记录保存在忽略 Git 的 `artifacts/kubernetes/lifecycle/`。

| 场景 | 已取得的结果 |
|---|---|
| 三节点网络 | 六个方向的 Pod IP、六个方向的 Service/DNS 和 Pod 重建通过 |
| 单卡真实推理 | 普通生成和完整 SSE 均返回预期答案，包含 `[DONE]`，记录 Pod UID/节点/物理卡号 |
| 双副本发现 | `.209`、`.210` 专家均被发现并实际接到请求 |
| 手动扩缩容 | 2→3→2；新增实例自动发现，Router Pod UID 不变 |
| 发现读取失败 | 临时拒绝 API list 权限，缓存到期后新请求返回 503，恢复权限后不重启 Router 即恢复 |
| 删除 Pod | 新 UID 替代旧 UID，约 101 秒恢复就绪与发现 |
| SSE 排空 | 输出期间删除模型 Pod，44 秒的 SSE 完整结束；约 93 秒恢复目标副本 |
| 引擎失去响应 | SIGSTOP 仅作用于自有模型进程组；约 43 秒退出 Ready、225 秒容器重启、322 秒恢复 |
| 自愿维护 | cordon/drain `.209`，两个专家在 `.210` 恢复，保留一份路由容量；约 98 秒完成容量迁移，随后 uncordon 并恢复分布 |
| 节点容器失联 | 约 56 秒检测 NotReady，约 450 秒在另一节点补足专家副本；最低保留 1 个专家和 1 个 Router，真实推理仍可完成 |
| 安全移除与重加入 | `.210` 排空后移出集群，保留原 Docker ID 重加入；Node UID 改变，设备重新识别，约 311 秒恢复完整基线 |
| 缩容与更新排空 | 分别在 SSE 输出期间执行缩容、滚动更新，约 43 秒的输出均完整收到 `[DONE]`；更新约 196 秒恢复目标容量 |
| 更新失败与回滚 | 刻意使用未缓存的固定摘要，检测 `ErrImageNeverPull`；回滚至原固定镜像，约 99 秒恢复 |

未做物理服务器断电、NPU 硬件损伤注入，也未宣称能恢复已被截断的生成请求。
节点容器停止测试与真实物理机故障必须分别记录；不能用前者替代后者的生产认证。

## 常用操作与回滚

```bash
python3 scripts/manage_pool.py render
python3 scripts/manage_pool.py prepare-router
python3 scripts/manage_pool.py apply
python3 scripts/manage_pool.py router
python3 scripts/manage_pool.py verify
python3 scripts/verify_lifecycle.py routing
python3 scripts/verify_lifecycle.py scale
python3 scripts/verify_lifecycle.py delete
python3 scripts/verify_lifecycle.py drain
python3 scripts/verify_lifecycle.py cache
python3 scripts/verify_lifecycle.py shortage
python3 scripts/verify_lifecycle.py scale_drain
python3 scripts/verify_lifecycle.py update_drain
python3 scripts/verify_lifecycle.py rollback
```

故障和维护命令只针对配置中固定的自有实验节点，结束后恢复原 Docker ID 和部署目标。
运行 `engine`、`maintenance`、`node_failure`、`remove_rejoin` 前确认没有本项目之外的流量依赖这些实例。
更新使用 `maxSurge:0/maxUnavailable:1`，无需备用卡；单副本更新会中断，双副本更新保留部分容量。
PDB `minAvailable:1` 约束自愿驱逐，不替代更新策略或保证节点故障下的可用性。
回滚应在实验控制平面中 `kubectl rollout undo deployment/<pool> -n heteroserve`，
同时恢复对应 Git 版本的运行 ConfigMap 和池配置，再做真实模型与路由验收；不能只回滚副本数量。

## 监控

由于没有已有监控，本次只创建一套固定 ARM64 镜像的 Prometheus、Grafana 和 kube-state-metrics。
`deploy/monitoring/stack.yaml` 可复现配置；镜像平台 digest 固定在 `images.json`。
Prometheus 保留 7 天、TSDB 数据规模上限 5 GB；数据使用独立 Retain 本地 PV，清理实验不删除有效监控数据。
Grafana 使用生成的私有 Secret，管理员密码只保存在 `.217` 的
`/root/zhangjinhao/LLM-Serving-System/state/monitoring/grafana-admin-password`（0600），不进入 Git、命令参数或日志。
一次校验命令的错误输出曾包含初始凭据，已通过标准输入重置密码、更新 Secret 并验证新凭据。

已确认全部推理抓取目标正常，能够查询 3 个 Ready 节点、16 个 NPU 资源、Pod 重启、专家就绪及原生 vLLM TTFT 指标。
Grafana 已自动加载 `heteroserve` 面板，共 8 个图表。服务仅使用内部 Service；按需使用受控端口转发访问。

## 参考资料

- [K3s Docker server/agent](https://docs.k3s.io/advanced#running-k3s-in-docker)
- [K3s 跨节点网络选项](https://docs.k3s.io/networking/basic-network-options)
- [K3s 固定版本发布](https://github.com/k3s-io/k3s/releases/tag/v1.34.12%2Bk3s1)
- [Kubernetes 发布与支持周期](https://kubernetes.io/releases/)
- [Usernetes 前置要求](https://github.com/rootless-containers/usernetes#requirements)
- [昇腾官方设备占用排查](https://www.hiascend.com/document/caselibrary/detail/topic_0000002238354198)
- [CANN 算子编译并发配置](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/envvar/envref_07_0020.html)
