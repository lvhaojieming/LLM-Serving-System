# 摩尔 S5000 / 天数 BI-V150 的 FlagCX 推理实验

正式配置：`heterogeneous_compute/configs/musa_iluvatar_flagcx.json`。
当前六节点部署复用 `.96` 的 `heteroserve-musa-pp` 和 `.60` 的
`heteroserve-iluvatar-pp`，基础实例各用设备0/1；`pp-02`、`pp-03`、`pp-04`
通过同一配置提供另外三个实例，每台共用8个芯片。不新建容器、不添加
CPU/内存/PID 上限。此前 `.97[4,5]` / `.60[0,1]` 实验的原配置和结果
保存在旧结果目录中；当前运行输出放在 `replicated/pp-01` 等子目录。

Qwen3-14B BF16 使用机内 TP2、跨厂商 PP2。摩尔执行 embedding 和前 20 层，
天数执行后 20 层、norm 和 LM head。当前配置复用官方本地 TP、模型算子和
厂商 Flash Attention，以及现有项目跨厂商执行器；
这不是厂商原生 vLLM Engine 的跨厂商运行，也不是 P/D 分离。
现有严格权重加载检查验证各阶段没有缺失参数。

跨阶段 BF16 hidden states 和 residual 的设备地址直接交给 `flagcxSend/Recv`。
TCP 控制连接只传 JSON 元数据、调度信息和反馈 token。
FlagCX 本次选择 Socket 网络后端，其内部可以进行主机中转；不声称 RDMA 或
设备间零拷贝。机内 TP 继续使用原生 MCCL / CoreX NCCL。

## 依赖与构建

两端原 PyTorch ABI 不同，保持已有 Torch、vLLM、驱动及工具链。
`bridge.cc` 仅封装 FlagCX 公共 C 接口，不依赖 PyTorch C++ ABI。

固定 FlagCX commit `4dd8862e09be73c8e54f49b763186e80c365f195`，
JSON 子模块 commit `a0e9fb1e638cfbb5b8b556b7c51eaa81977bad48`。
库和源码保留在两端 `/workspace/artifacts/dependencies/flagcx/<commit>/`，
包含 `source.json`、打包源码、构建日志和 `FlagCX/build/lib/`。
源码压缩包及含 JSON 头文件的构建压缩包 SHA256 已记录在正式配置中。
不复制库到系统目录，不覆盖厂商通信库。

重建时在现有容器内使用该源码和正式构建入口：

```bash
bash /workspace/heterogeneous_compute/platforms/flagcx/build.sh \
  /workspace/artifacts/dependencies/flagcx/4dd8862e09be73c8e54f49b763186e80c365f195/FlagCX \
  musa   # 天数节点使用 iluvatar
```

已观察到默认 Socket 配置在大于一个网络 chunk 的消息上发生长度错误。
当前配置将 `FLAGCX_NET_BUFFER_SIZE` 和 `FLAGCX_NET_CHUNK_SIZE` 均设为 4 MiB，
使同一连接同时只保留一个网络槽；Socket 使用一线程、一个数据 socket。
此设置已通过两条 PP lane 的变长、重复、双向 BF16 精确值检查，最大张量
1024×5120。它是本实验的实际通信配置，不能忽略该限制解释吞吐。

图模式只覆盖 MLP 和本地 TP，不包含 attention、KV 和 PP 通信。FlagCX 配置
必须在建立 PP 通道前预热1–4 token的全部图形状。运行期间首次捕获需要设备
同步，与正在等待对端的 FlagCX 设备流重叠时曾发生停滞。启动日志中的
`graphs_prewarmed_before_pp` 记录各 rank 的预热计数；运行验收还需确认实际
请求的图回放计数继续增长。预热成功本身不代表并发或长输出验收通过。

摩尔端还曾在异步发送线程读取 `Event.elapsed_time` 时出现
`MUSA error: device not ready`，导致通道关闭。设备张量路径现在先显式同步
两个生产者事件，再读取计时和交给 FlagCX，避免只依赖跨流 wait 的完成状态。
传输异常会同时写入 rank 日志与时间线，不以零吞吐的失败请求作为测速结果。

## 运行与验证

在本地项目根目录执行，SSH 配置使用原有主机别名：

```powershell
python heterogeneous_compute/manage.py probe --ssh-config '<ssh_config>' --config heterogeneous_compute/configs/musa_iluvatar_flagcx.json
python heterogeneous_compute/manage.py start --ssh-config '<ssh_config>' --config heterogeneous_compute/configs/musa_iluvatar_flagcx.json
python heterogeneous_compute/manage.py status --ssh-config '<ssh_config>' --config heterogeneous_compute/configs/musa_iluvatar_flagcx.json
python heterogeneous_compute/manage.py benchmark --ssh-config '<ssh_config>' --config heterogeneous_compute/configs/musa_iluvatar_flagcx.json
```

`probe` 每条 lane 用 1/128/512/1024 token 四种尺寸、三轮变更内容，检查精确往返。
`start` 要求四份成功报告与当前配置哈希匹配。它只选择该配置的进程，不覆盖原
`ascend_musa.json`，不操作其他服务。重复启动和重复压测会被拒绝。

当前 API 为 `http://192.168.162.96:18297/v1`，其余实例端口18298–18300。
结果、各 rank 日志、传输时间线和通信校验位于两端
`/workspace/artifacts/baselines/musa-iluvatar-flagcx/replicated/`。

测速客户端在天数 `.60` 的现有容器运行，先做三条顺序问答和三条并发问答的
准确答案检查；失败则不进入测速。之后调用现有 vLLM 0.23 官方 `bench serve`：
synthetic random 数据，输入约 256 token、固定输出 64 token、seed 20261010，
并发 1/4/8，每档 200 请求、3 次预热、greedy、ignore EOS。
这组输入不是旧 ShareGPT 基线，不能直接据此计算与旧结果的加速比。

`performance/status.json` 表示实际进度；完整逐请求结果和命令保存在
`performance/random-c*.json`、`random-c*-command.json`，各档自动验证请求数、
输出长度、相同输入长度序列和吞吐重算，通过后生成 `performance/summary.json`。
短问答通过仅代表有限功能验收，不代表批大小数值不变性或完整模型质量验收。

正常退出使用同一命令的 `stop` 动作，先排空当前请求；初始化或通信故障无法
排空时，`abort` 仅终止当前配置的精确 Python 进程并保留容器。
`stop-probe` 只终止该配置的通信探针。
