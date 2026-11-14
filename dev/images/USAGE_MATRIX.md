# 基座使用矩阵 · 现行环境 × 代码仓（USAGE_MATRIX）

> **一页看全**：现行生效的每个基础环境（芯片 × 腿）用什么镜像、什么运行环境目录、各层组件与代码仓的来源和版本锚。
> **数据源**：四份现行锁定声明（`dev/stack.lock.910c.v4.yaml` / `stack.lock.kunlunxin.v1.yaml` / `stack.lock.cambricon.v1.yaml` / `stack.lock.thead-ppu.v1.yaml`）＋ 各 `v<N>/image.baseline.yaml` 与 `env/v1/env.baseline.yaml`。本表只做汇总呈现，**版本事实以各 baseline / lock 为准**。
> **更新纪律**：任何镜像入锁 / 换锁 / 运行环境换版时，随 `image_list.md` 一并刷新本表（现状-only；过程与理由进 ADR，不写这里）。
> 生成：2026-11-14（基线取自 dev-1.0）。

**形态图例（单元格前缀）**：**基座**=厂商/官方基座自带（被动继承，不可重建）｜**配方**=本组配方带入（重建可变）｜**覆盖**=env/vN 运行环境挂载生效（优先于镜像固化）｜**上游**=上游发布镜像自带（版本锚=digest）｜**—**=不在此环境。

| 层 · 行项 | ① 910C·训练 | ② 910C·推理 | ③ 昆仑芯·训练 | ④ 昆仑芯·推理 | ⑤ 寒武纪·训练 | ⑥ 寒武纪·推理 | ⑦ 平头哥·训推同底 |
|---|---|---|---|---|---|---|---|
| **锚点 · 现行镜像** | `ascend-train-comm` **v3**（[目录](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/ascend-train-comm/v3)）· id `e13a15d0…`（27 机 `7028028…`） | `ascend-infer-vllm` **v1**（[目录](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/ascend-infer-vllm/v1)）· `vllm-ascend:v0.20.2rc1-a3` · digest `5cf8a2b6…` | `kunlun-operator-runtime` **v2**（[目录](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/kunlun-operator-runtime/v2)）· id `cac5ddae5f62…` | `kunlun-infer-vllm` **v1**（[目录](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/kunlun-infer-vllm/v1)）· `kunlunxin001…:tele4` · digest `e180a0dd…` | `cambricon-mlu-runtime` **v1**（[目录](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/cambricon-mlu-runtime/v1)）· `flagos-runtime-cambricon-neuware4.4.3:2.2.0` · digest `e55b420e…` | `cambricon-infer-vllm` **v1**（[目录](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/cambricon-infer-vllm/v1)）· `flagos-app/vllm0.20.2-…:2.2.0-0.2.2` · digest `76a673a8…` | `thead-ppu-runtime` **v2**（[目录](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/thead-ppu-runtime/v2)）· id `1ba7ead1…`（基座 digest `50a906e6…`） |
| **锚点 · 现行环境** | [env/v1](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/ascend-train-comm/v3/env/v1)（FlagCX 覆盖；`demo/`=冻结第 0 版） | [env/v1](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/ascend-infer-vllm/v1/env/v1) | [env/v1](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/kunlun-operator-runtime/v2/env/v1)（GEMM compat，opt-in） | —（登记式，无 env 目录；验证宿主=在跑容器） | [demo](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/cambricon-mlu-runtime/v1/demo)（容器 `flagrt-mlu-v1-demo-mlu2`） | [demo](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/cambricon-infer-vllm/v1/demo) | [env/v1](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/images/thead-ppu-runtime/v2/env/v1) |
| **芯片基座 · 宿主驱动/固件**（宿主机侧，不入镜像；固件随驱动栈、基线未单独登记） | 昇腾驱动 25.5.0（910c-11 实测 2026-10-20） | 同左 | 宿主侧（旧机实测 5.0.21.47；新机未回填） | 同左＋tag 明示栈代际 `v5.0.21.43` | 6.2.x 线定档位；两机实测分叉（Mlu-1=6.5.52 / Mlu-2=6.2.29，2026-11-05） | 同左 | 宿主侧（版本未回填） |
| **芯片基座 · 厂商计算栈 SDK**（镜像内 L1） | 基座 **CANN 9.0.0** | CANN 9.0.0（官方栈同纪元） | 基座 **XTDK 3.6.0.1 ＋ XRE 5.13.0.0 ＋ XCCL 20260209_9**（自证 `_versions.txt`） | 上游 xVLLM 发布栈（xmlir 1.0.0.1） | 基座 **NEUWARE 4.4.3**（cntoolkit 4.4.3 / cnnl 2.1.829 / mluops 1.8.1 / cncl 1.29.4） | NEUWARE 4.4.3（flagos-app 线） | 基座 **PPU SDK v2.1.0**（CUDA wrapper cu130） |
| **芯片基座 · 设备插件**（Route A，torch 命名空间） | 基座 torch_npu **2.10.0**（PrivateUse1=`npu`） | 基座 torch_npu **2.10.0**（wheel） | 基座 torch_xmlir（xmlir **1.0.0.1**；设备 API=**torch.cuda 符号重写**） | 上游 xmlir **1.0.0.1**（wheel） | 基座 torch_mlu **1.29.2** ＋ torch_mlu_ops 1.8.0（device_type=`mlu`） | 基座 torch_mlu **1.29.2** | 无专用插件——**torch.cuda 命名空间**（PPU SDK 提供） |
| **芯片基座 · 集合通信库** | HCCL（CANN）＋ FlagCX（见活跃库层） | 原生 hccl | XCCL ＋ FlagCX v0.10.0（固化） | 上游 flagcx **0.13.0**（editable） | CNCL 1.29.4（MLU_LINK 片间；RDMA 未启用 = C-4） | CNCL（neuware 栈） | accl-p（libpccl）＋ NCCL（网卡覆盖纪律） |
| **基础栈 · Python** | 基座 3.11.15 | 官方栈 3.11 | 基座 3.10.18（conda env） | tag `cp310` | 官方栈 3.10 | 官方栈 3.10（同基座族 flagos-base:2.2.0） | 基座 3.12.3 |
| **基础栈 · PyTorch** | 基座 2.10.0+cpu | 官方栈 2.10.0 | 基座 2.9.0+cu129 | 上游 2.9.0+cu129 | 基座 2.7.1+cpu | 上游 2.7.1+cpu | 基座 2.10.0 |
| **基础栈 · transformers** | 未单独 pin（基线无此条目） | 基座 **5.5.3** | 未单独登记 | 上游 4.57.6 | 未单独登记 | 上游 **5.17.0** | 基座 4.57.0 |
| **基础栈 · triton（编译层）** | 配方 3.5.1（FlagTree `15ec1a6c` 线构建） | 基座 3.5.0 ＋ triton_ascend 3.2.1（华为 wheel） | 由 flagtree wheel 提供（0.6.1+xpu3.6 线） | 上游 3.5.0 通用版（treenone，非 FlagTree 构建物） | 厂商 triton-mlu 3.2.0+mlu1.7.2（/opt/triton） | 上游 triton-mlu 3.2.0+mlu1.7.2（wheel） | 派生层提供 3.6.0（flagtree wheel） |
| **活跃库 · vllm（上游）** | 基座 `bc150f50`（0.20.2，editable） | 基座 `bc150f50`（0.20.2+empty） | 基座 0.13.0（无 commit） | 上游 0.20.2+empty（wheel；上游 Label `09935a6b`） | 无 | 上游 0.20.2+flagos（wheel，无 commit） | 基座 0.19.0+cu130（wheel；/opt/vllm 无 .git，无 commit 锚） |
| **活跃库 · vllm-ascend（华为插件）** | — | 基座 `367b8e62`（0.20.2rc1，镜像内 git） | — | — | — | — | — |
| **活跃库 · Torch-FL** | 配方固化 `162582d` | — | — | — | — | — | — |
| **活跃库 · FlagCX** | 固化 `4e0e0cb` → **覆盖 `62fe615`（生效；修 A-2 原生 DDP）** | —（走 hccl） | 固化 `3b43bdb2`（v0.10.0，基座自带） | 上游 0.13.0 | — | — | — |
| **活跃库 · FlagGems** | 固化 `f7ae8e6b` | — | 固化 `73c5aff1` → **覆盖 `4a00b845`（`xliu969/kunlun-gemm-compat`，opt-in；修 K-1 编译）** | 上游 5.0.0 | 基座 5.4.0rc2.post3（验证=当前不可替换） | 基座 5.4.0rc2.post3 | — |
| **活跃库 · vllm-plugin-FL** | 固化 `8b059122`（release/0.2＋补丁；**910C 线挂起不跟进**，默认路径不 import） | **硬纪律 `unset VLLM_PLUGINS` 显式禁用** | 固化 `f9d95f43`（v0.1.0，基座自带；纪律 `VLLM_FL_PREFER=vendor`，K-2） | 上游 0.2.0+`g38e7dbc`（加载报错不阻断） | — | 上游 **0.2.2（寒武纪线必需，勿 unset）** | —（serve ppu 分支不需要） |
| **活跃库 · FlagTree** | 配方固化 `15ec1a6c`（triton_v3.5.x） | —（无 FlagTree 构建物） | 索引 wheel `0.6.1+xpu3.6`（sha256 `31863cfe…`，`--no-deps`） | —（treenone） | —（无寒武纪线） | — | 派生 wheel `0.7.0+ppu.gite223f42c`（commit `e223f42c`，sha256 `b57daff7…`；同时提供 triton 3.6.0） |
| **活跃库 · 自建仓**（非固化，验证时经 WORKSPACE_ROOT / pip 挂载） | device-context runtime（`use("ascend")`）；FlagPerf 独立使用 | 同左（serve 统一入口） | 同左（`kunlun`）；FlagMemory connector 见④ | **FlagMemory xpu connector 宿主栈**（挂载） | device-context `cambricon` 挂载 | 同左 | device-context `ppu` 挂载；FlagMemory ppu connector |

## 备注

- **壁仞 BR_0 未入锁定声明**（接入中，镜像 `birensupa-smartinfer-vllm:26.07.09` 走厂商官方线定档；入锁待总组收拢）。
- **四个方向交付仓（FlagMemory / flagos-device-context / framework-adapter / FlagScheduler）不在任何基座镜像内**——经挂载或 pip 在验证时使用，不属于基座固化资产；`FlagPerf` 是唯一不出现在任何基座里的本组仓（benchmark 工具，独立使用）。
- **"固化 → 覆盖"仅两处**（①FlagCX `4e0e0cb`→`62fe615`、③FlagGems `73c5aff1`→`4a00b845`）——即镜像＋环境双版本方案（ADR-0008）的现行全部落点；其余环境生效版本=镜像字节。
- **无 commit 锚的格子**均为上游官方产物（镜像内无 .git，版本锚=Harbor digest）或索引/离线 wheel（锚=版本号＋sha256）——ENV-SPEC 登记式基线既定口径，不是缺口。
- 昆仑芯两腿并存为**有意分栈**（runtime 腿 vllm 0.13.0 缺 simple_cpu_offload_connector，KV connector 场景用 infer 腿；ADR-0026）；寒武纪/910C 同口径两腿分栈。
