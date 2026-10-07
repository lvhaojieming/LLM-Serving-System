# LLM Serving System

面向昇腾的 LLM 推理系统。当前先完成 Kubernetes 部署与控制能力验收，验收后再重构应用层。

实验采用独立的跨物理主机 K3s 容器集群，管理凭据由实验控制平面生成。
原共享集群的 kubelet、凭据和暂停服务保持原状。实验基础设施长期复用，参数通过配置调整。

| 入口 | 用途 |
|---|---|
| `deploy/lab/cluster.json` | 两机实验集群的固定镜像、端口、网段及资源上限 |
| `scripts/manage_lab.py` | 预检、创建或复用实验节点、查询状态与跨机验收 |
| `deploy/lab/npu.json` | 固定实验节点的昇腾镜像与验证资源预算 |
| `scripts/manage_npu.py` | 驱动占用预检、NPU 分配与计算验收、受限清理 |
| `deploy/k8s/` | 通用项目权限、网络与暂停状态的 NPU 验证模板 |
| `scripts/check_kubernetes.py` | Kustomize 渲染、固定 OpenAPI schema 与部署约束检查 |
| `docs/deployment.md` | 部署边界、权限、版本和验收步骤 |

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

当前两机网络及 `.209` 的两只普通 Pod NPU 分配与矩阵计算已通过，复现步骤和适用边界见 `docs/deployment.md`。
真实模型 Pod 尚未验收。

当前实验资源策略为 `resource_limits_enabled=false`：外层容器的 CPU、内存及进程上限和项目 Pod 的
CPU/内存 limits 已取消，项目验证配额已删除；NPU 按卡分配保留。原地操作入口为 `scripts/manage_lab.py unlimit`。

本地验证复用已有 Python 环境，开发依赖见 `requirements-dev.txt`。

```bash
python -m pytest -q tests
```

不得将 CPU 验收、静态 YAML 校验或普通 Node Ready 当作昇腾生产部署验收。
