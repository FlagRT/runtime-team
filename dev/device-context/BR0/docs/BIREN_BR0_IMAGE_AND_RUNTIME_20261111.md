# 壁仞 BR_0 · 镜像选型与设备上下文接入报告（2026-11-11）

> 机器：`wz-server008`（`Biren-1`，8 × Biren166M）｜ 接入实例目录：`BR0/`（手册铁律：新建实例、只迁规范与方法）
> 时间：2026-11-11 10:23–11:45 CST ｜ 执行：hliu553
> 结论：**镜像已定档、后端已落地、离线自检 83/0/1、真机 smoke 46/0**。可用卡 1/3/4/5/6/7（卡 0/2 故障见另报告）。

---

## 1 镜像选型（结论 + 依据）

**定档**：`birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx`（机内已有，17.5 GB）

### 1.1 为什么不是 FlagOS 官方线

按镜像选型纪律「唯一 source = `flagos-ai/build-infra::configs.yaml`」，实际克隆核对：

```
git clone --depth 1 git@github.com:flagos-ai/build-infra.git
grep -n -i 'biren' configs.yaml   → 0 命中
```

⇒ **FlagOS 官方镜像体系里没有壁仞线**（既无 `flagos-runtime-*biren*`，也无 FlagTree 条目）。
因此本实例与寒武纪同属「厂商官方镜像线」：**版本锚 = 镜像 tag + 机内 digest，不按 FlagOS 线号理解**。
（对照：910C/P800/PPU 走 FlagOS 官方线；MLU590 走 `flagos-runtime-cambricon-*`；壁仞无该线。）

### 1.2 候选与实测对比（同一台机，`docker run --rm` 不挂卡探查）

| 候选镜像 | 大小 | torch / 厂商插件 | `import torch_br` | 判定 |
|---|---|---|---|---|
| **birensupa-smartinfer-vllm:26.07.09-…-br1xx** | 17.5 G | 2.8.0+cpu / `torch-br 1.8.7.20800+br1xx` | ✅（source 厂商环境脚本后） | **选定** |
| birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx | 10 G | 2.8.0+cpu / `torch-br 1.11.0.20800+br1xx` | ❌ 缺 `libbesu.so.1` | 备用（更新 torch-br 但库面不全） |
| biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107 | 25.6 G | 2.8.0+cpu / `torch-br 1.8.7` + `biren-ml-py 1.12.0` | ❌ `/usr/lib/libbesu.so.1*` 为 **0 字节占位** | 用于**算子/内核开发**，非运行时腿 |

**决定性依据不是"看版本号"，而是"机内已有可用的现成组合"**：机内在跑的 xbench 容器（`xbench-native-dev-biren-20261111-101500`）用的就是 `smartinfer-vllm:26.07.09`，在该容器内 `import torch_br` → `2.8.0+cpu / 1.8.7.20800+br1xx` 一次成功。**同一个镜像、同样的入口**，是可复现的落地路径。

> ⚠️ 探查中踩到的假象（都记入 §4 坑清单）：裸镜像里 `libsupti.so.1` / `libbesu.so.1` 找不到或为 0 字节，**不是镜像坏**——壁仞 runtime 在容器启动时把宿主侧库注入 `birensupa/sdk/<ver>/…`，并由 `brsw_set_env.sh` 设 `LD_LIBRARY_PATH`；`docker exec` 不继承入口进程环境，所以必须显式 source。

---

## 2 容器与运行参数（定案）

```bash
docker run -d --name dc-biren-hliu553 \
  --network host --shm-size 32g \
  -v $HOME/hliu553_biren:/work \
  -v /share/hf_cache:/share/hf_cache:ro \
  -e BIREN_VISIBLE_DEVICES=5,7 \
  --restart unless-stopped \
  birensupa-smartinfer-vllm:26.07.09-…-br1xx /bin/bash -c 'sleep infinity'
```

三条硬纪律（实测得出）：

| # | 纪律 | 依据 |
|---|---|---|
| 1 | **不要传 `--device`** | 宿主 `daemon.json` 是 `default-runtime: biren`，壁仞 container-toolkit **自动注入**卡节点与库；手工 `--device /dev/biren/card_N` 会与之冲突（首版容器因此 `no device`，ErrorCode 100） |
| 2 | 选卡用 **`BIREN_VISIBLE_DEVICES=<物理卡号>`** | 与 `CUDA_VISIBLE_DEVICES` 同义、**物理卡号**；容器内 0 基重编号（本实例 5,7 → `supa:0`/`supa:1`） |
| 3 | 命令前必须 source 厂商环境 | `. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set`；否则 `import torch_br` 报 `libsupti.so.1: cannot open shared object file` |

---

## 3 接入判据（逐级实测）

### 3.1 L2 最小判据（真机，卡 5/7）

```
torch: 2.8.0+cpu | torch_br: 1.8.7.20800+br1xx
supa device_count: 2
matmul64 sum=-1009.2964 nan=0 inf=0
mm PU1: ['PrivateUse1: registered at /home/jenkins/agent/workspace/BR_PACKAGE/buildSystem/...']
supa has: device_name=True mem_get_info=True props=True current_device=True
```

⇒ **路线 A（厂商官方 torch 插件）确认**：设备 API 走 `torch.supa`（`torch.cuda.device_count()==0`、`torch.xpu` 亦为 0）。

### 3.2 后端落地：`runtime/backends/biren/`

- `name = "biren"` · `device_type = "supa"`；注册进 `registry._KNOWN_BACKENDS`（第五个实例）
- **能力声明（逐项实测，未测不声明）**：`device` / `memory` / `memory_alloc_stat` / `stream` / `event` /
  `multidevice` / `recovery_probe` / `device_state` / `device_state_control`
- **明确不声明**：`stream_priority`（能设≠有效果，未做回读判据）· `bounded_sync`/`sync_timeout`（原生无 timeout，
  仅轮询近似）· `recovery_real` / `graph_capture` / `memory_alloc` / `record_stream` / `context_*` / `error_map`（未探测）
- 事件适配：**E3 型缺口已修**——未 record 的 `Event.query()` 原生返回 True，本层用 `_recorded` 标志修正

### 3.3 离线契约自检

```
python3 scripts/backend_offline_check.py --backend biren
→ 83 通过 / 0 失败 / 1 跳过（跳过=stub 能力边界，非失败）
```

> 顺带修了工具侧一个**真缺口**：`fresh_import` 的厂商模块清理名单与 `VENDOR_ROOTS` 阻断名单都硬编码，
> 少了 `torch_br` ⇒ 「缺厂商扩展时必须报错」这条判据会因上一轮残留而**静默通过**。
> 已补两处（`torch_br`），并把 `_stub_biren` 加进 `_STUBS`（此前工具只认四家）。

### 3.4 真机 smoke（后端无关通用自检）

```
runtime/smoke_runtime.py --backend biren
→ 46 通过 / 0 失败
关键项：device_count n=2 · memory_stats {total 65024, used 0, free 65024, allocated 0}
       Stream/Event 可用 · Event.record + wait_host 有界返回 · probe_device(0) True
       recover_device 返回统一契约 dict · translate_error 回填 backend=biren
       未声明 error_map ⇒ graded_by=message_hint、mapped=False、error_code=None（诚实性判据全过）
```

---

## 4 坑清单（本实例新增，按手册 §9 归档）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | `import torch_br` → `libsupti.so.1: cannot open shared object file` | `docker exec` 不继承镜像入口设置的环境（`brsw_set_env.sh`） | 命令前 source 该脚本（**不是**镜像坏） |
| 2 | 容器内 `no device`（ErrorCode 100） | 手工 `--device` 与 `default-runtime: biren` 的自动注入冲突 | 去掉 `--device`，用 `BIREN_VISIBLE_DEVICES` |
| 3 | 裸镜像里 `libbesu.so.1.12.0` 为 **0 字节** | 库由 runtime 在容器启动时注入，裸镜像只有占位 | 不要以裸镜像 `ls -l` 大小判断完整性 |
| 4 | `Event()` 未 record 时 `query()` 返回 True | 厂商插件事件语义缺省 | 适配层 `_recorded` 修正（同 PPU/MLU590） |
| 5 | `torch.supa.synchronize()` / `Stream.synchronize()` 无 timeout | 厂商 API 形态 | 轮询 + deadline，超时**上报**（不保证中断） |
| 6 | 离线自检「缺厂商扩展应报错」静默通过 | 工具的 `VENDOR_ROOTS` 与 `fresh_import` 清理名单硬编码、缺 `torch_br` | 两处已补；新增厂商记得**两处都注册** |

---

## 5 证据与脚本

| 文件 | 内容 |
|---|---|
| `BR0/probes/biren_resource_20261111_out/` | 首轮资源普查（宿主/卡/驱动/SDK/镜像候选），5 份脚本 |
| `BR0/probes/biren_card02_20261111_out/` | 卡 0/2 故障专项（239 行内核时间线） |
| `BR0/probes/biren_image_20261111_out/` | 镜像内容探查脚本（本轮 13 次迭代，含失败尝试） |
| `BR0/docs/BIREN_BR0_RESOURCE_SURVEY_20261111.md` | 资源普查报告 |
| `BR0/docs/BIREN_CARD02_FAULT_REPORT_20261111.md` | 卡 0/2 故障报告（供上游） |
| `runtime/backends/biren/` | 后端实现（`backend.py` 含能力声明依据与 known_issues 九字段） |

## 6 下一步（未做，按手册顺序）

> **2026-11-11 14:56 更新**：第 1、2 项已完成 —— conformance **13/13 + 6/6**、多流 16 项已逐项对照
> （探针 `STREAM_SEMANTICS_PASS 8/8`；`stream_priority` **不声明**，依据是「构造收参但回读恒 0」）。
> 详见 `BR0/docs/BIREN_BR0_CONFORMANCE_AND_STREAM_20261111.md`，证据 `BR0/probes/biren_conf_stream_20261111_out/`。

1. ~~**conformance 13 + 6**（真机）~~ —— ✅ **13/13 + 6/6，`rc=0`**（2026-11-11）；
2. ~~多流 16 项基线~~ —— ✅ 已逐项对照：12 项实测通过 / 1 项实测不具备有效优先级 / 1 项不适用 / 2 项本轮未做；
3. 两条腿：训练腿（需确认集合通信后端名——**必须探测，不得兜底**）、推理腿（镜像自带 vllm 0.21.0）——**用户裁定本轮不做**；
4. 服务化：`serve_standard.sh` 需补 `biren)` 分支（新芯片必经项）；
5. 职责审计 78 项。
