# 同厂商跨节点 TP2 / PP2 基线

每组两台同厂商、同型号节点，各用两个逻辑加速设备，整模型共四个 rank。
使用厂商原生 `vllm serve --tensor-parallel-size 2 --pipeline-parallel-size 2`，
通过配置选择原生 Ray 或多节点 multiprocessing，不使用跨厂商参考执行器。

| 厂商 | 节点 | 镜像软件 |
|---|---|---|
| 昇腾 910C | .23 / .24 | vLLM-Ascend 0.11.0 A3 |
| 摩尔 S5000 | .97 / .96 | MUSA 4.3.5 镜像，vLLM 0.9.3.dev |
| 海光 BW1100 | .252 / .251 | 两端已有的厂商 vLLM 0.25.1 镜像 |
| 天数 BI-V150 | .58 / .60 | CoreX 4.5.0，vLLM/插件 0.23.0 |

同厂商两端必须使用同一个镜像 ID。驱动随宿主机变化，逐节点记录版本和
必要库校验值；不是所有厂商使用同一个二进制环境。MLU590 缺少兼容的
用户 vLLM 环境；MetaX 只有一个节点；BI-V200 不与 BI-V150 混作同型号基线。

## 复用与配置

复用 .97 的 `heteroserve-musa` 和 .58 的 `heteroserve-iluvatar`。
原服务及跨厂商执行器继续使用原卡；本实验配置选择其他空闲卡。
昇腾旧容器只挂载 0/1，不能在其中访问空闲的 2/3，因此新增长期复用的
`heteroserve-ascend-pp`；第二节点原先没有本项目推理容器，也新增长期容器。
海光复用两端已有的 0.25.1 镜像并使用独立长期容器；0.18.1 的 Docker 29
导出只得到元数据，未通过新容器运行验收，不能把 image inspect 成功当成环境可用。
不重建已有容器，不恢复 CPU/内存/PID 限制，不操作旧 Kubernetes 实验或暂停服务。

昇腾仅挂载主机选中的两张逻辑卡；CANN 在该挂载子集中从 0 编号，故容器内
`ASCEND_RT_VISIBLE_DEVICES=0,1`，实际主机卡号仍保存为配置中的 `[2,3]`。
Ray UNIX socket 临时根必须足够短，使用 `/tmp/moqe-pp/<vendor>/s<stage>`。
不启动可选 Ray Client server，不新增依赖。

.96 未安装 MThreads 容器运行时，镜像中的 libmusa 为占位文件。配置指定
复用宿主机 `libmusa.so.4.3.8`，复制到现有实验容器并校正链接，记录其 SHA256。
.97 原容器使用已有 mthreads runtime；不更改宿主机驱动。

.58 的同一 NIC 主地址为 192.168.0.12，实验地址为 192.168.162.58。
原生 Gloo 会使用主地址，.60 原路由经网关无法连接。配置在 .60 添加唯一
`192.168.0.12/32 via 192.168.162.58`，直接到该主地址所属的同一个对端。
不改默认路由、不替换已有路由、不改节点 IP。`network.py` 校验路由和对端身份，
路由保留作为此服务的通信依赖。此路径已通过原生四 rank 推理验收。

## 启动与测速

从仓库根目录，使用已有 Python 和 SSH 配置：

```powershell
python homogeneous/experiments/pp_baseline/manage.py prepare --ssh-config '<cluster_ssh_config>'
python homogeneous/experiments/pp_baseline/manage.py start --ssh-config '<cluster_ssh_config>'
python homogeneous/experiments/pp_baseline/manage.py benchmark --ssh-config '<cluster_ssh_config>'
python homogeneous/experiments/pp_baseline/manage.py status --ssh-config '<cluster_ssh_config>'
```

`--vendor ascend|musa|hygon|iluvatar` 只操作选中组。prepare 使用
`shared/transfer_image.py` 内网流式 docker save / SSH docker load，保留已信任
主机密钥校验，不落地镜像 tar，传完移除任务密钥与对端临时授权。
start 拒绝重复启动和覆盖已保留运行状态；失败时先确认进程归属，保留必要
诊断，再清理本次失败 rank 和状态，不能使用 ray stop/killall 清理共享服务。

### 在原容器内增加跨节点实例

每个 TP2/PP2 实例在两台机器上各使用两个芯片。每厂商四个实例共使用
两节点各八个芯片，不能把它写成 TP4/PP2，也不能将芯片数当作板卡数。
BI-V150 每板两个设备，每设备 32 GiB；每节点八个芯片对应四张板卡。

配置的 `vendors.<vendor>.instances` 声明额外实例，原 `nodes`、`ray_port`、
`api_port` 仍代表保留的基础实例。例如基础实例使用 `[4,5] / [0,1]` 时：

```json
"instances": {
  "pp-02": {"devices": [[0,1], [2,3]], "ray_port": 33000, "api_port": 18398},
  "pp-03": {"devices": [[2,3], [4,5]], "ray_port": 33200, "api_port": 18399},
  "pp-04": {"devices": [[6,7], [6,7]], "ray_port": 33400, "api_port": 18400}
}
```

配置校验拒绝同一节点芯片重叠、API/Ray 端口范围重叠和非法实例名。
部署前还应检查真实设备占用；配置声明不是资源预留，不能停止其他任务。
模型、精度、厂商环境及两端容器沿用同一配置。每实例独立 Ray 集群、
API 端口和 `/tmp/moqe-pp/<vendor>/<instance>/`，复用同一套官方 vLLM PP。
四个现有容器使用 `ipc=host`，共享内存以容器内 `df /dev/shm` 的实际挂载
为准；不能把 Docker inspect 中未生效的默认 `ShmSize` 当作容量。
节点内部使用共享内存，跨节点 PP 仍通过网络通信。
摩尔同时显式设置 `--disable-custom-all-reduce`、`MCCL_P2P_DISABLE=1` 和
`MCCL_SHM_DISABLE=0`，复用已验证的标准 MCCL 同机 SHM 通道配置。
`ipc=host` 只能证明共享内存挂载可用，不能单独证明 TP collective 使用 SHM；
验收还需检查工作进程环境、引擎的 `disable_custom_all_reduce=True`，以及
MCCL INFO 日志中的 SHM 通道。跨节点 PP 的 Socket 通信继续保留。
原 AWQ/TP2 的提速数据不能直接当作 FP16/TP2/PP2 的性能结果。

```powershell
python homogeneous/experiments/pp_baseline/manage.py start --config artifacts/baselines/same-vendor-pp/deployment-config.json --ssh-config '<cluster_ssh_config>' --vendor musa --instance pp-02
python homogeneous/experiments/pp_baseline/manage.py status --config artifacts/baselines/same-vendor-pp/deployment-config.json --ssh-config '<cluster_ssh_config>' --vendor musa --instance pp-02
```

`--instance` 只选择该实例，不启动同池其他实例；省略时继续使用原基础实例。
额外实例可用 `env` 覆盖该实例的通信端口环境变量，共用环境的其余值保持不变。
昇腾 `nodes[].container_devices` 声明容器挂载的完整主机芯片列表，`devices`
选择该实例使用的两个主机芯片。启动前核对实际 Docker 挂载，按照挂载子集
计算 `ASCEND_RT_VISIBLE_DEVICES`：只挂载 `[2,3]` 时选择 `[2,3]` 对应 `0,1`；
挂载全部 `0..15` 时同样选择对应 `2,3`。不能在多实例共享全卡容器中固定为 `0,1`。
`nodes[].env` 保存各节点独有的环境变量，优先于厂商共用环境。例如 A3 的
`HCCL_LOGIC_SUPERPOD_ID` 需要按实际超节点划分配置，不能让设备 SDID 相同的
两台机器误入同一逻辑超节点。`ray_non_carry_over_env_vars` 使用官方 vLLM
`ray_non_carry_over_env_vars.json` 保留这类节点局部配置，合并而不覆盖已有排除项。
实例快照、日志、验收结果和可选性能结果分别保存在
`artifacts/baselines/same-vendor-pp/<vendor>/<instance>/`，基础实例的历史路径不变。
启动时上传已解析的独立配置，避免并行实例覆盖共用 `config.json`。
真实回答在判定成功前保存，失败也保留具体输出。性能记录从配置读取
FP16/BF16；本入口沿用原 eager 基线，不能标作图模式。

只部署模型时不要执行 `benchmark`。回归检查可用已有 Python 标准库运行：
`python homogeneous/tests/test_pp_baseline.py -v`。

2026-10-28 扩容验收：Qwen3-14B、FP16、TP2/PP2、eager，每厂商四个实例。
基础实例保留运行，新增实例复用相同四个容器与镜像；CPU/内存/PID 上限未恢复。

| 厂商/实例 | 第一节点芯片 | 第二节点芯片 | API 端口 | Ray 基础端口 |
|---|---|---|---|---|
| 摩尔 base | .97: 4,5 | .96: 0,1 | 18397 | 32200 |
| 摩尔 pp-02 | .97: 0,1 | .96: 2,3 | 18398 | 33000 |
| 摩尔 pp-03 | .97: 2,3 | .96: 4,5 | 18399 | 33200 |
| 摩尔 pp-04 | .97: 6,7 | .96: 6,7 | 18400 | 33400 |
| 天数 base | .58: 6,7 | .60: 0,1 | 18358 | 32600 |
| 天数 pp-02 | .58: 8,9 | .60: 2,3 | 18359 | 33600 |
| 天数 pp-03 | .58: 10,11 | .60: 4,5 | 18360 | 33800 |
| 天数 pp-04 | .58: 12,13 | .60: 6,7 | 18361 | 34000 |

13:01 北京时间验收：八个 API 均健康且实际贪心算术生成正确，队列归零，
SMI 确认每节点八个目标芯片有模型显存占用。只完成部署和功能验收，
没有启动吞吐/稳定性压测，不能将此验收当作容量结论。
本机保留 `artifacts/baselines/same-vendor-pp/deployment-config.json` 和
`replica-acceptance.json`，包含实际启动参数、版本、回答与设备占用证据。

### 昇腾主机 TCP/IP 路径

2026-10-28 昇腾两节点对齐到现有 AWQ 环境镜像
`heteroserve/ascend-migration:heteroserve-ascend-w4`（vLLM 0.26.0，
vLLM-Ascend `155f68974c802633c19650ece430d41ee6802fa6`），模型仍使用原始
Qwen3-14B FP16，不使用 AWQ 权重。`.24` 复用原容器，`.23` 原 AWQ 环境为
0.11 且两端版本不兼容，因此同步已有镜像并创建一个长期复用的同环境容器；
原容器和旧 PP 容器均保留，旧模型服务保持暂停。

用户确认仅连接主机网络。该部署通过官方 Ray Compiled Graph 的主机通道
传递 PP 中间张量，同机 TP 继续使用 HCCL。运行配置：

```json
"VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "0",
"VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": "shm",
"VLLM_USE_RAY_WRAPPED_PP_COMM": "0",
"VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM": "0"
```

这里 `shm` 是 Ray 的节点内共享内存及跨节点 gRPC 通道，不是跨主机共享同一块
`/dev/shm`。这是官方执行器配置，不修改 vLLM 的模型切分或张量传输实现。
当前环境的 Ray V2 默认路径调用 worker 内部的 PP HCCL send/recv；
选择 Compiled Graph 后，由 Ray 的 `execute_model_ray` 直接传递
`IntermediateTensors`，避免要求跨节点的 NPU 数据面网络。
保留 `--no-async-scheduling` 和 eager 模式作为该配置的明确执行条件。

诊断中，按节点设置不同的 `HCCL_LOGIC_SUPERPOD_ID` 消除了 EI0014 SDID 重复，
直接 HCCL PP 的下一条错误为两端设备 IP 均为 `0.0.0.0`（EI0015）。对目标
2 号芯片的只读查询确认 RoCE Link DOWN、IPv4/IPv6 均未配置。未修改硬件
SDID、网卡 IP、驱动或交换机。以上说明该次配置和现场条件，不表示 HCCL
本身不支持跨节点。官方诊断参考：
[superDeviceId 重复](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclug_000046.html)。

旧成功实验是 0.11/BF16/Compiled Graph `auto`，不能把它与 0.26/FP16/Ray V2
默认路径视为相同配置；历史结果和必要诊断分别保留。

当前昇腾分配如下；`.23` 的 4–9 号芯片属于旧实验已停用的三组设备，继续避开。
本次 4/5、6/7 的尝试同样遇到 TSD/AICPU 初始化失败，保留诊断后改用此前
已验收设备，没有重启整机或重置驱动。

| 实例 | .23 芯片 | .24 芯片 | API 端口 | Ray 基础端口 |
|---|---|---|---|---|
| base | 2,3 | 2,3 | 18323 | 32000 |
| pp-02 | 0,1 | 0,1 | 18324 | 34200 |
| pp-03 | 10,11 | 4,5 | 18325 | 34400 |
| pp-04 | 12,13 | 6,7 | 18326 | 34600 |

两台各使用八个芯片，整组为四个独立 TP2/PP2 实例。
部署验收与诊断保留在本机 `artifacts/baselines/same-vendor-pp/` 下的
`ascend-awq-acceptance.json`、`ascend-hccl-diagnosis.json` 和
`ascend-history-comparison.json`，有效旧结果保留于原目录。

海光 Ray 的原生 `pp_broadcast` 初始化仍停滞，配置已切换 `executor=mp`：
两端分别传 `--nnodes 2 --node-rank 0/1 --master-addr <head> --master-port <port>`，
第二节点使用 `--headless`。镜像已实际解析接受这些官方参数，但本轮服务初始化
超时，客户端记录为 failed，未产生有效速度数据。其他三组保留已验证的 Ray 路径。

原生引擎须确认两台 Ray 节点各提供两个设备，并通过真实贪心算术 decode。
benchmark 复用 .58 现有 vLLM 0.23.0 官方客户端和既有 ShareGPT/tokenizer：
BF16、Eager、关闭前缀缓存，200 请求/档、3 预热、并发 1/4/8，固定输出 64
token、ignore EOS、temperature=0、seed=20261010。
数据集 SHA256 与原单节点/跨厂商测量固定一致。

客户端保存官方逐请求数据、TTFT/TPOT/ITL/E2EL 的均值与 P50/P95/P99，
按整四卡实例报告吞吐。完成后检查成功请求数、输出预算、跨档输入 token
长度序列和吞吐重算，自动生成 summary.json；拒绝覆盖结果。

服务器日志、阶段状态、部署身份和配置位于各节点
`artifacts/baselines/same-vendor-pp/<vendor>/`；性能原始数据全部位于 .58 同目录
的 `performance/`。这些运行产物不提交仓库。确认正常开始和进度推进后停止
轮询，由后台任务继续；用户询问时再读取并整理正式结果。

## 比较边界与当前验证

旧单节点基线是 TP2/PP1；本组是跨节点 TP2/PP2。海光原基线使用 0.18.1，
本组使用两端已有且可启动的 0.25.1，应明确软件版本也发生变化。
跨厂商旧组使用串行参考执行器，新组使用自定义连续批处理执行器；本组使用
原生 vLLM。两者作为系统部署对照，
不能把差值全部归因于跨厂商通信。

2026-10-14：昇腾、天数、摩尔原生四 rank 均通过模型验收，官方 benchmark
三档各 200 请求，三组均 600/600 成功。摩尔于 16:14（北京时间）完成；已核验
输出预算、各档输入长度及吞吐重算，保留 [摩尔结果](results/musa.json)。
摩尔并发 1/4/8 的输出吞吐分别为 4.76/95.31/169.65 token/s；并发 1 明显偏慢，
原因尚未定位，不能据此直接作硬件性能排名。

海光 Ray 四 rank 已分层加载，但原生 PP broadcast 初始化停滞，Socket 配置
仍未解决；后续原生多节点 mp 同样未通过服务初始化，测速客户端最终就绪等待
超时。海光本轮为失败状态，暂无有效速度数据，不会自动重启。
