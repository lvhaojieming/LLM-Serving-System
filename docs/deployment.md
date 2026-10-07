# 跨机器 Kubernetes 部署与验收

目标是验证 Kubernetes 真正创建、放置、重建和管理跨机器工作负载，再接入昇腾 NPU 和真实模型。
当前实验集群只管理明确标记的实验资源；应用层重构等待部署阶段验收。

## 实验与现有集群的职责

现有共享集群使用其原有 kubelet 和管理凭据。实验控制平面和 worker 在独立 Docker 容器中运行，
使用独立数据卷、API 端口、容器网络和 Pod/Service 网段。所有 kubectl 操作显式指向实验控制平面中的 kubeconfig。

这条路线需要获准使用宿主机，并有创建 privileged 实验节点容器的权限。
它不需要原共享集群的应用部署权限，也不会赋予原共享集群的权限。
privileged 容器不能作为对宿主机管理员的安全隔离边界；设备归属必须单独确认。

用户已指定 `.209` 至 `.217`，共九台候选节点。当前只启用 `.217` server 和 `.209` agent。
每个实验节点容器限制 4 CPU、8 GiB 内存和 16384 PID。嵌套节点上报的 Node capacity 可能与外层容器配额不同，
因此验证工作负载还有独立 ResourceQuota，不能直接用 Node capacity 宣称可用实验资源。

## 版本与已知边界

实验版本固定为 `v1.34.12+k3s1`，ARM64 镜像固定平台 manifest digest 和 image config digest。
这些值由 K3s 发布页、Docker Hub 平台记录以及 `.217` 的真实镜像检查交叉确定。
镜像配置 ID 允许复用经校验的 Docker archive，避免在每台节点重复等待外网下载。

这是兼容性实验基线，不是长期生产版本选择。Kubernetes 1.34 上游支持在 2026-10-27 结束。
目前实验节点实际内置 containerd 为 `2.2.7-k3s1`；不能用宿主机的 `1.6.16` 或报告中的候选 `2.1.4` 代替它。
昇腾 runtime/Device Plugin 与这个组合的兼容性尚未验收。NPU 阶段需要重新锁定并验证完整版本矩阵。

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
两台外层实验容器 ID 保持不变；真实模型 Pod 尚未启动，后续仍须进行真实推理及目标集群验收。

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

## 参考

- [K3s Docker server/agent](https://docs.k3s.io/advanced#running-k3s-in-docker)
- [K3s 跨节点网络选项](https://docs.k3s.io/networking/basic-network-options)
- [K3s 固定版本发布](https://github.com/k3s-io/k3s/releases/tag/v1.34.12%2Bk3s1)
- [Kubernetes 发布与支持周期](https://kubernetes.io/releases/)
- [Usernetes 前置要求](https://github.com/rootless-containers/usernetes#requirements)
- [昇腾官方设备占用排查](https://www.hiascend.com/document/caselibrary/detail/topic_0000002238354198)
- [CANN 算子编译并发配置](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/envvar/envref_07_0020.html)
