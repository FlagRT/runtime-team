# ascend-infer-vllm v1 —— 华为昇腾官方推理镜像（登记式基线）

## 用途与现状

- 本目录登记**华为昇腾官方**推理镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`
  （image id `sha256:2e56022ae5b3…`，registry digest `sha256:5cf8a2b6…`，rc 标签防 GC 主锚），
  **现行推理腿发布物（2026-10-14 发版）**，见 [`dev/stack.lock.910c.v4.yaml`](../../../stack.lock.910c.v4.yaml) 的 `lock.infer`。
  非本组构建、非 FlagTree 血统：版本纪元对齐 ascend3.5（CANN 9.0.0 / py3.11 / torch 2.10.0 / vLLM 0.20.2），
  整体为被动继承的官方产物 —— 上游配方在 `vllm-project/vllm-ascend` 仓 `docker/`。
- **登记式基线（ADR-0007）**：本目录 `image.baseline.yaml` 为手工登记，字段口径同 ENV-SPEC §3.1；
  与自建基线的差异 —— ① 全部条目 `from: base_image`（无本组配方层）；② commit 取自镜像内 `.git`
  （自证锚，2026-10-14 探针实测），check 走 INFO 分支由 image id + digest 兜底；③ 服务启动不在此目录，
  统一经 [`serve_standard.sh`](../../device-context/prototype/scripts/serve_standard.sh)（组内启动标准，唯一入口）。
- 现行运行环境 [`env/v1/`](env/v1/docker-compose.yml)（docker-compose.yml + env.baseline.yaml，ADR-0008）：
  只管容器生命周期，`up.sh --strict` 照常核对 image id；含宿主 /usr/local/Ascend/driver 挂载与单 die 映射。
  [`demo/`](demo/docker-compose.yml) 为冻结的第 0 版验收记录（其 compose 缺驱动挂载，勿再作起点）；
  起服务在容器内跑 serve_standard.sh。
- `repro_status`: 🟢 digest 锁定、公开 registry 可拉（2026-09-09 验证）；镜像不可重建是**设计边界**（官方产物）。
- 已知限制：**无**。不携带训练腿限制 A-2~A-4 任何一条。
- 分发：公开 registry 按 digest 拉取 `docker pull quay.io/ascend/vllm-ascend@sha256:5cf8a2b6…`；
  离线归档见 [`ARCHIVE.md`](ARCHIVE.md)。

## 复现（env/v1 起容器 + serve_standard 起服务）

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（image id `sha256:2e56022a…`） | 公开 registry 按 digest 拉取；image id 由 `up.sh --strict` 按镜像基线核对 |
| 运行环境 | 本目录 `env/v1/`（现行；`demo/` 为冻结验收记录） | 随仓；`INFER_DIE` 选定单 die、宿主驱动挂载为内置要求 |
| 卡 | 1+ × 910C | 启动前 `docker ps` 确认无其他带卡容器（昇腾独占）；选卡 `DEV` 传给 serve_standard.sh |
| 模型 | serve_standard.sh 需模型路径（HF 缓存须给到 snapshots/<hash>；实体目录直接给目录） | `FLAGRT_MODELS_DIR` 指向实体目录 |

### 步骤（仓根执行）

```bash
docker ps                          # 无其他带卡容器
INFER_DIE=<die> FLAGRT_MODELS_DIR=<模型目录> dev/lib/up.sh \
  dev/images/ascend-infer-vllm/v1/env/v1 --strict
docker exec -it flagrt-infer-v1-env-v1-910c bash    # 容器内：
DC_BACKEND=ascend MODEL=<模型路径> DEV=0 STOP_AFTER=1 \
  bash /workspace/dev/device-context/prototype/scripts/serve_standard.sh
# verdict：SERVE_STANDARD_PASS (ready=1 smoke=1)
INFER_DIE=<die> docker compose -f dev/images/ascend-infer-vllm/v1/env/v1/docker-compose.yml down
```
