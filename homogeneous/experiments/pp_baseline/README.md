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

海光 Ray 的原生 `pp_broadcast` 初始化仍停滞，配置已切换 `executor=mp`：
两端分别传 `--nnodes 2 --node-rank 0/1 --master-addr <head> --master-port <port>`，
第二节点使用 `--headless`。镜像已实际解析接受这些官方参数，正在做模型验收；
截至本次记录尚无该路径的速度结论。其他三组保留已验证的 Ray 路径。

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
跨厂商组使用串行参考执行器，本组使用原生 vLLM；两者作为系统部署对照，
不能把差值全部归因于跨厂商通信。

2026-10-14：昇腾和天数原生四 rank 已通过模型验收。天数官方 benchmark 三档
600/600 成功；昇腾并发 1 的 200/200 成功，其余档继续后台运行。
摩尔已解决缺失驱动库和 MCCL 的默认跨节点传输错误，Socket 配置通过真实
推理验收，关闭诊断 INFO 日志后正式 benchmark 已开始。海光 Ray 四 rank 已
分层加载，但原生 PP broadcast 初始化停滞，Socket 配置仍未解决；正在验证
原生多节点 mp。海光尚未通过完整模型验收，不能报成已成功测试。
