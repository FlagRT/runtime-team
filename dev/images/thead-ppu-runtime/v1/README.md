# thead-ppu-runtime v1 —— FlagTree 官方 PPU 基座镜像（登记式基线）

## 用途与现状

- 本目录登记 **BAAI·FlagTree 官方** PPU 基座镜像
  `flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base`
  （image id `sha256:f5366712d6a0…`，Harbor digest `sha256:50a906e6ed9d…`，版本主锚），
  **平头哥 PPU（真武 ZW810E）现行基座**，见 [`dev/stack.lock.thead-ppu.v1.yaml`](../../stack.lock.thead-ppu.v1.yaml)。
  非本组构建：整体为被动继承的 FlagTree 官方产物（ppu3.6 线：PPU SDK v2.1.0 / py3.12 /
  torch 2.10.0 / vLLM 0.19.0+cu130），上游配方在 BAAI·FlagTree 官方 PPU 镜像线。
- **登记式基线（ADR-0007，同 ascend-infer-vllm/v1 先例）**：本目录 `image.baseline.yaml`
  为手工登记，字段口径同 ENV-SPEC §3.1；与自建基线的差异 —— ① 全部条目 `from: base_image`
  （无本组配方层）；② 镜像内 `/opt/vllm` 无 `.git`（2026-10-15 探针实测）⇒ 无自证 commit 锚，
  版本主锚 = Harbor digest，辅锚 = pip 版本号；③ 本组 PPU 后端（runtime ppu 0.1.0）不在镜像内，
  在仓内 `dev/device-context/prototype/runtime/`，经 WORKSPACE_ROOT 挂载使用、不固化进镜像。
- demo 运行环境 [`demo/`](demo/docker-compose.yml)（docker-compose.yml + env.baseline.yaml）：
  只管容器生命周期，`up.sh --strict` 照常核对 image id；两条 PPU 特有启动坑（hostname 解析 /
  NCCL 网卡名）已内置规避，纪律说明见 demo 文件头注释。
- `repro_status`: 🟢 digest 锁定、公开 Harbor registry 可拉（2026-10-15 验证）；镜像不可重建是
  **设计边界**（官方产物）。
- 已知限制：**无**（本镜像自身）。PPU 侧已知平台缺陷（退出期弱引用悬垂，根因在平台侧、已由
  本层规避）见 device-context 方向 `docs/PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md`（kistich 分支）。
- 分发：公开 Harbor registry 按 digest 拉取
  `docker pull harbor.baai.ac.cn/flagtree/flagtree-ppu-py312-…@sha256:50a906e6ed9d…`。

## 复现（demo 起容器 + 容器内环境核对）

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | `flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base`（image id `sha256:f53667…`） | 公开 Harbor registry 按 digest 拉取；image id 由 `up.sh --strict` 按镜像基线核对 |
| 运行环境 | 本目录 `demo/` | 随仓 |
| 卡 | 1+ × PPU-ZW810E（PTG_0：16 卡） | `ppu-smi` 挑空闲卡（0% util / 2MiB 显存）；无带卡容器独占（同 P800） |
| 模型 | 容器内核对不需要；服务化等 serve_standard.sh 增补 ppu 分支 | — |

### 步骤（仓根执行，机器 = PTG_0）

```bash
ppu-smi | tail -24                        # 挑空闲卡
dev/lib/up.sh dev/images/thead-ppu-runtime/v1/demo --strict
docker exec -it flagrt-ppu-v1-demo-ptg0 bash
# 容器内环境核对（本 demo 的验收动作）：
python3 -c "import torch; print(torch.cuda.device_count())"              # 预期 16（映射全卡）
CUDA_VISIBLE_DEVICES=2 python3 -c "import torch; print(torch.cuda.device_count())"   # 预期 1
# 用完收尾：
WORKSPACE_ROOT=$PWD docker compose -f dev/images/thead-ppu-runtime/v1/demo/docker-compose.yml down
```

> 服务化口径：组内唯一服务入口 serve_standard.sh 当前无 ppu 分支（该方向的 PPU 交接在
> kistich/device-context 分支待合入）。该分支合入后，服务启动统一走 serve_standard.sh；
> 本目录不另设第二服务入口。

## 获取

```bash
docker pull harbor.baai.ac.cn/flagtree/flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04@sha256:50a906e6ed9dd4cb8f00be446b731f2e3738d4c6ec4296d46380e71fb5f5bd53
docker tag <拉取后 image id> flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base
```

PTG_0 本机已有该镜像真身（5 个 FlagTree 工作负载容器共用，勿动）。

<!--
changelog:
2026-10-15 —— 首次登记（登记式基线，ADR-0007 同款流程）：BAAI·FlagTree 官方 PPU 基座镜像，
digest 锁定；新增 image.baseline.yaml（manual-registration）+ demo/ 运行环境 +
lock.yaml。PPU 后端接入证据（conformance 13/13 等）在 kistich/device-context 分支，
由该方向自行合入，本目录只登记镜像基座。
-->
