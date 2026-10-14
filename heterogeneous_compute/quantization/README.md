# Qwen3-14B W4A16 权重生成

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
