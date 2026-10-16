# cambricon-mlu-runtime v1 —— 寒武纪 MLU590 基座镜像（登记式基线）

## 用途与现状

- **登记对象**：`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`
  （`@sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2`）。
- **现行发布物**：寒武纪 MLU590 线的统一基座，设备上下文与训练路径同底座。锁定声明见
  [`../../../stack.lock.cambricon.v1.yaml`](../../../stack.lock.cambricon.v1.yaml)。
- **repro_status**：🟢 FlagOS 官方镜像（公开 Harbor，匿名可按 digest 拉取）。
- **登记式基线（ADR-0007）三条差异**：① 镜像整体为上游官方产物，无本组配方层 ⇒ `image.baseline.yaml`
  全部条目 `from: base_image`，`recipe: []`；② 镜像内无 `.git` ⇒ 无自证 commit 锚，版本主锚为
  Harbor digest，辅锚为 pip 版本号与镜像 Label 的上游配方 commit；③ 本组 cambricon 后端不固化进镜像，
  经 `WORKSPACE_ROOT` 挂载使用。
- **运行环境**：[`demo/`](demo/)（`docker-compose.yml` + `env.baseline.yaml`）。
- **已知限制**：镜像自身无缺陷类限制；使用面约束 5 条（档位由宿主驱动决定 / 宿主无 neuware /
  triton 导入顺序与 `PYTHONPATH` / 本镜像不含 vLLM / 硬件能力 `(5, 0)` 不支持 FP8 与 MX）见
  [`lock.yaml`](lock.yaml) 的 `notes` 与 `gaps`。
- **分发方式**：公开 Harbor registry，按 digest 拉取，无需凭据。

## 复现（demo 起容器 + 容器内环境核对）

### 前置条件

| 项 | 要求 |
|---|---|
| 镜像 | 本机已有 `@sha256:e55b420e…`（无则按下方「获取」拉取） |
| 运行环境 | 宿主驱动 6.2.x 线（实测 v6.2.29）；宿主无需安装 neuware |
| 卡 | `cnmon` 挑 0% 利用率 / 0 MiB 显存的空闲卡；无带卡容器独占，但勿干扰他人容器 |
| 模型 | 本 demo 只做环境核对，不需要模型 |

### 步骤（仓根执行）

```bash
# 0) 挑空闲卡
cnmon | head -40

# 1) 起容器（strict 会核对 image id 与基线声明一致）
dev/lib/up.sh dev/images/cambricon-mlu-runtime/v1/demo --strict

# 2) 容器内环境核对（一律用 bash -lc：/opt/triton 只在经 shell 启动时进 PYTHONPATH）
docker exec -it flagrt-mlu-v1-demo-mlu2 bash -lc \
  'python3 -c "import torch, torch_mlu; print(\"device_count=\", torch.mlu.device_count()); print(torch.mlu.get_device_name(0))"'
#    预期：device_count= 8 / MLU590-M9

docker exec -it flagrt-mlu-v1-demo-mlu2 bash -lc \
  'MLU_VISIBLE_DEVICES=3 python3 -c "import torch, torch_mlu; print(torch.mlu.device_count())"'
#    预期：1（选卡变量按授予顺序密集重编号）

docker exec -it flagrt-mlu-v1-demo-mlu2 bash -lc \
  'MLU_VISIBLE_DEVICES=4,5 python3 -c "
import torch, torch_mlu
for d in range(torch.mlu.device_count()):
    a = torch.randn(1024, 1024, device=f\"mlu:{d}\"); b = torch.randn(1024, 1024, device=f\"mlu:{d}\")
    print(d, bool(torch.isfinite(a @ b).all()))
"'
#    预期：两卡均 True

# 3) 用完收尾（勿动他人容器）
WORKSPACE_ROOT=$PWD docker compose \
  -f dev/images/cambricon-mlu-runtime/v1/demo/docker-compose.yml down
```

## 获取

```bash
docker pull harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3@sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2
docker image inspect harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3@sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2 \
  --format '{{.Id}}'
#   预期 image id：sha256:e887e0ca4333a27185817fb8fef6ce95d5d9220c5a9725641187d602cabeea96
```

<!-- changelog:
2026-10-15 —— 首次登记（登记式基线，ADR-0007 同款流程）：FlagOS 官方寒武纪运行时基座镜像
  flagos-runtime-cambricon-neuware4.4.3:2.2.0，Harbor digest 锚；新增 image.baseline.yaml
  （manual-registration）与 demo/（compose + env.baseline.yaml，baseline.py export 生成）。
  镜像真身在 tza-0a06-ai02-em9 按 digest 拉取落地，image id 实测，带卡验收通过
  （device_count=8 / 单卡重编号 / 双卡 matmul finite）。寒武纪后端接入证据由 device-context
  方向自行维护与合入。
-->
