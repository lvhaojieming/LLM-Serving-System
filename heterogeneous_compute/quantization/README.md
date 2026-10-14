# Qwen3-14B W4A16 权重生成

2026-10-14 当前选择：AWQ 自量化权重已通过导出校验，保留用于复现；GPTQModel NPU 自量化在第一层 down_proj 的 AICPU QR 上失败，按用户要求停止重跑。未采用的自定义 Hessian 实现、msModelSlim 安装临时文件和失败 GPTQ offload 缓存已清理，保留必要失败日志。用户已选择直接下载社区 `JunHowie/Qwen3-14B-GPTQ-Int4`，不将其标为 Qwen 官方发布。

社区模型的 ModelScope revision 固定为 `63ca45ac5d22ed49897a9e55112a1c32360869fc`，配置为标准 GPTQ、bits=4、group_size=128、desc_act=false、checkpoint_format=gptq。ModelScope 下载及 SCP 回 `.24` 已完成并通过哈希校验；两节点目录均为 `~/models/Qwen3-14B-GPTQ-Int4`。复用 `shared/download_model.py --quantization gptq --mirror modelscope` 和通用分发器。

复用原始 BF16 模型与固定 ShareGPT 校准集，使用同一入口通过 `--algorithm awq|gptq` 切换算法。复用同构项目的 SHA256 与 tokenizer 校验函数，不复制量化后端。

校准集 256 条、最大长度 512，seed 20261014；与现有官方 benchmark 的 200 条输入无重合。`prepare_calibration.py` 在现有 vLLM benchmark 客户端环境中运行，复用已下载的数据集，不重新下载完整语料。

当前权重生成环境为 `.24` 已缓存的官方 `quay.io/ascend/vllm-ascend:v0.26.0rc2-a3` 镜像，长期复用任务容器 `heteroserve-quantization`。其 Torch/NPU/Transformers 版本与配置中一致；安装 GPTQModel 时约束这些版本，不修改其他推理容器。依赖安装日志和完整环境快照保存在 `/quantization/`。新容器用于空闲单卡量化，避免占用正在压测的推理卡。

标准输出：AWQ GEMM（非对称），GPTQ legacy（对称），bits=4、group_size=128、desc_act=false。原 BF16 权重加载为 FP16 进行校准，LM head 不量化。已有昇腾适配器支持这两种标准格式的整数布局解析，但新环境整模型推理验收仍需独立进行。

量化设备不决定部署设备。上述标准 checkpoint 与昇腾专用 `moqe_ascend_int4` 转换权重分开管理：标准权重用于后续后端验收，专用转换或加载时重排复用已有昇腾适配代码。其他厂商仍需逐一验证格式、零点约定和实际 W4A16 算子支持，不能仅凭能读取模型文件就认定兼容或加速。

后续目标为同一套量化结果的多厂商部署：昇腾、摩尔、天数、海光、寒武纪。每个后端分别记录镜像与库版本、AWQ/GPTQ 加载结果、实际执行算子和精度；格式转换必须验证整数与尺度保持一致，不能通过重新校准掩盖不同权重。某后端不支持对应算法时明确标记，不自动回退成 FP16 并报告为 W4A16。验收后使用同一官方 vLLM benchmark 输入与参数测量延迟和吞吐。

```bash
python3 /workspace/heterogeneous_compute/quantization/run.py \
  --config /workspace/heterogeneous_compute/quantization/config.json --algorithm awq
python3 /workspace/heterogeneous_compute/quantization/run.py \
  --config /workspace/heterogeneous_compute/quantization/config.json --algorithm gptq
```

模型位于宿主机 `/home/zhangjinhao/quantization/models/`；日志位于 `logs/awq.log`、`logs/gptq.log`，退出码在同名 `.exit` 文件。容器中统一挂载为 `/quantization/`。原始权重挂载只读。

Transformers 5 导出时可能重新序列化 tokenizer。发布先确认模型词表、merges、特殊 token ID 和所有校准输入的 token ID 一致，再恢复原始 tokenizer 文件。已保存但未发布的模型可用同一命令加 `--publish-only` 完成验证与发布，不重复量化。补充发布日志与退出码为 `logs/<algorithm>.publish.log/.exit`；最终完成标志以正式模型目录中的 `QUANTIZATION_COMPLETE` 为准，不能仅看旧进程的退出码。

输出先写 `.incomplete`，检查量化配置、280 个 INT32 packed Linear、浮点 LM head 和 tokenizer 一致性后发布正式目录，保存逐文件哈希与依赖版本。`QUANTIZATION_COMPLETE` 只表示权重生成及结构验证完成，不表示推理速度、精度或后端加载已经验收。禁止覆盖已有模型与未完成输出。

量化权重、校准文本、缓存和原始日志不提交 Git。推理复用现有昇腾 INT4 后端；当前不重建旧部署，不恢复已暂停服务。

## 当前 W4A16 部署与官方测速

复用 `.24` 的 `heteroserve-quantization`、原始和量化权重，容器 ID 保持不变。
推理采用 vLLM 0.26.0+empty、Torch 2.10.0+cpu、torch-npu 2.10.0.post4、
Transformers 5.14.1、镜像内 CANN 9.1.0；不新增环境或安装推理依赖。

`shared/ascend_int4/` 将既有 `ascend_int4_test` 的整数解码、重排和 NPU 矩阵
乘法纳入正式项目，扩展当前 vLLM 的自定义量化注册与参数加载接口。
这是自定义 Dense AWQ/GPTQ 兼容适配，不是厂商官方 Dense AWQ/GPTQ 支持。
不修改厂商 site-packages，只有显式启用的服务及其子进程安装注册。
原始权重和 tokenizer 不变，不重新校准、不重新量化、不保留全模型 FP16 展开。
用户已有旧原型作为来源保留，没有删除其工作。

当前仅验证 **TP1/PP1、FP16 激活、group128、AWQ GEMM/GPTQ legacy**；
GPTQ act-order、非连续 g_idx、其他 group size、TP2 等均拒绝或未验证。
LM head 和 embedding 保持浮点，280 个标准量化 Linear 在 vLLM 中融合成
160 个实际执行 Linear。它们的 INT32 参数以每权重 4 bit 存储，调用
`torch_npu.npu_weight_quant_batchmatmul`，不会静默回退为浮点权重 matmul。

两种算法的实际 q_proj/down_proj 在 token 数 1/32 时与 CPU FP32 反量化数学
对照，相对 L2 误差均低于 0.0006。AWQ/GPTQ 整模型及 FP16 对照均实际输出
5 / 巴黎 / 3；每个 W4 模型的 160 个融合层均记录实际 INT4 算子调用。
这些是有限功能验收，不代表 PPL、下游任务质量或完整的量化误差评估。

统一入口与参数为 `deploy.py`、`deployment.json`。顺序执行 AWQ → GPTQ →
FP16 对照，固定复用一张空闲逻辑 NPU 0、同一环境、TP1/PP1、Eager、
关闭 prefix cache、context 2048、max_num_seqs 8、gpu_memory_utilization 0.60。
与过去 TP2/BF16 基线分开；这次 FP16 是同资源、同激活精度的浮点控制组。
AWQ 是本项目自量化，GPTQ 是所选社区权重，不能将二者质量差异全归因于算法。

从客户端仓库根目录运行（Windows 可用隐藏后台进程启动）：

```powershell
python heterogeneous_compute/quantization/deploy.py --action run --ssh-config '<cluster_ssh_config>'
```

控制器必须保持运行，但不持续轮询 benchmark 进度：它阻塞等待官方 benchmark
进程结束，再只排空自己创建的模型进程并切换下一配置，保留容器和模型。
失败会记录并停止，不修改共享服务、驱动或暂停的旧服务。
单独部署、验证、停止可在现有容器内用同一入口的
`--action start|probe|stop --algorithm awq|gptq|fp16` 完成。
保留镜像原有 CANN/PYTHONPATH；入口自行添加项目路径，不用
`docker exec -e PYTHONPATH=/workspace` 覆盖 CANN 的 acl 绑定路径。

测速复用 `.58` 的 vLLM 0.23.0 官方 `vllm bench serve` 与既有 ShareGPT：
每模型并发 1/4/8、各 200 请求、3 预热、64 output tokens、ignore EOS、
temperature 0、seed 20261010。三模型共 1800 正式请求。
校验固定数据集 SHA256、成功数、输出长度、各档输入长度和吞吐重算；
输出 TTFT/TPOT/ITL/E2EL 与 P50/P95/P99。

控制器状态与本地原始摘要在 `artifacts/quantization/inference/`；服务端验收与
日志在 `.24` 同目录；官方原始测速结果在 `.58` 同目录的各算法
`performance/`。完成后只提交必要结果摘要，保留权重、配置和复现信息。
