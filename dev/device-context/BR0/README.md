# 壁仞 BR_0 · 第五实例看板

> 分支：`kistich/device-context` ｜ 更新：**2026-11-11** ｜ 负责人：Kistich（hliu553）
> 🆕 **2026-11-11 集合通信后端名探测（训练腿前置，16:35–16:51）**：后端名实测为 **`sccl`**（`torch_br/__init__.py:179` `register_backend("sccl", …, devices=["supa","cpu"])`），**SCCL 版本 `10.8.0`**；2-rank supa 双卡 `all_reduce` **值精确匹配**（`BIREN_DIST_ALLREDUCE_PASS`，rc=0）；错名 `bccl` → `AssertionError: Unknown backend type` ⇒ **无静默回落**。⚠️ 两条边界：`all_reduce` **拒绝 CPU 张量**（`Tensors must be SUPA and dense`）；`BIREN_VISIBLE_DEVICES` **不许进程内改**（改则设备上下文静默挂住）。报告 `docs/BIREN_BR0_DIST_BACKEND_20261111.md`，证据 `probes/biren_dist_20261111_out/`。
> 🆕 **2026-11-11 判据推进**：真机 **conformance 13/13 + 推理 6/6**（各 `rc=0`）、多流探针 **`STREAM_SEMANTICS_PASS 8/8`**；
> 多流 **16 项基线逐项对照** = **12 项实测通过 / 1 项实测「不具备有效优先级」/ 1 项不适用 / 2 项本轮未做**（训推两腿按裁定不做）。
> 关键实测：`Stream(priority=N)` 对 N∈{0,±1,−2,−5,7} 构造全成功但 `stream.priority` **回读恒 0**、`Stream.priority_range()`
> 抛 `RuntimeError: greatest_priority <= -1 INTERNAL ASSERT FAILED` ⇒ `stream_priority` / `_control` / `_readback` **三键全不声明**。
> 报告 `docs/BIREN_BR0_CONFORMANCE_AND_STREAM_20261111.md`，原始证据 `probes/biren_conf_stream_20261111_out/`。
> **2026-11-11 接入起步**：SSH/sudo/docker 打通 → 资源普查 → 镜像定档 → `biren` 后端落地 → **离线契约自检 83/0/1** → **真机 smoke 46/0**。
> 报告 `docs/BIREN_BR0_IMAGE_AND_RUNTIME_20261111.md`、`docs/BIREN_BR0_RESOURCE_SURVEY_20261111.md`；卡故障专项 `docs/BIREN_CARD02_FAULT_REPORT_20261111.md`。
> **本目录 = 第五实例（壁仞 Biren166M，代号 BR_0）的芯片专属资产**；
> 芯片无关的规范与原型在 `../prototype/`，前四实例在 `../910C/`、`../P800/`、`../MLU590/`、`../PPU/`。
> 接入方法见 **《新芯片接入手册》** `../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`。

---

## 0. 状态

> 🟡 **当前状态（2026-11-11）**：**接入进行中，设备上下文与多流两道判据已过；训练腿/推理腿/服务化/职责审计尚未做。**

| # | 判据 | 结果 | 证据 |
|---|---|---|---|
| 1 | 离线契约自检（无设备） | **83 通过 / 0 失败 / 1 跳过**（跳过＝stub 能力边界） | `../prototype/scripts/backend_offline_check.py --backend biren` |
| 2 | 真机 smoke | **46 / 0**（`device_count=2`、`memory_stats 65024/0/65024`、Stream/Event/`record+wait_host`、`probe_device`、`recover_device` 契约 dict、`translate_error` 诚实性） | `docs/BIREN_BR0_IMAGE_AND_RUNTIME_20261111.md` §3.4 |
| 3 | conformance 设备上下文与多流 13 例 | **13/13** `CONFORMANCE_PASS`，`rc=0` | `probes/biren_conf_stream_20261111_out/biren_conf_20261111_144929/{conf13.log,conf13.json}` |
| 4 | conformance 推理 6 例 | **6/6** `CONFORMANCE_PASS`，`rc=0` | 同上 `{conf6.log,conf6.json}` |
| 5 | 多流 16 项基线 | **逐项对照**（12 实测通过 / 1 实测不具备有效优先级 / 1 不适用 / 2 未做）；探针 **`STREAM_SEMANTICS_PASS 8/8`** | `docs/BIREN_BR0_CONFORMANCE_AND_STREAM_20261111.md` §4；`…/biren_stream2_20261111_145225/` |
| 5b | **集合通信后端名探测**（训练腿前置） | ✅ **`sccl`**（SCCL `10.8.0`）；2-rank `all_reduce` PASS；错名报错无回落 | `docs/BIREN_BR0_DIST_BACKEND_20261111.md`；`probes/biren_dist_20261111_out/` |
| 6 | 训练腿（两 rank / 50 步） | ⛔ **本轮不做**（用户裁定） | — |
| 7 | 推理腿（前向 / 服务化） | ⛔ **本轮不做**（用户裁定） | — |
| 8 | 错误闭环（四类注入） | ⛔ 未做 | — |
| 9 | 职责响应审计 78 项 | ⛔ 未做 | — |

⭐ **多流 16 项实际读数**（与 `STREAM_SEMANTICS_PASS 8/8` 不是同一套计数，别混说）：

| 子项 | 读数 |
|---|---|
| S1 流内顺序 | `5.000000`（期望 5.0） |
| S2 无隐式同步 | A=`7.0000` / B=`11.0000` |
| S3 跨流可见性 | `z 全 3 = True` |
| S4 `wait_stream` 传递 | `b 全 10 = True` |
| S5 多流并发 | hi=`5.0000` / lo=`8.0000`；⚠️ **重叠率实测 ±2% 内（无可测收益）**，见 §2 第 5 条 |
| S8 默认流 vs 命名流 + `record_stream` | `2.0000` / `4.0000`；跨流复用后 `10.0000` |
| S9 错误隔离（API 级） | 流 A 注入 `RuntimeError`（shape 不匹配）后流 B 仍 `6.0000`；**设备级为语义推断，未真实触发** |
| S10 生命周期与配额 | **1000 次**创建销毁（`1.01 s`）后新流仍可用 `9.0000` |
| S11 跨流内存 | `6.0000` |
| S12 流优先级 | **不具备有效优先级**（回读恒 0、range 抛内部断言）⇒ 三键不声明 |
| S13 多设备流绑定 | `[2.0, 3.0]`（双卡各自独立流） |
| S14 同步超时 | `wait_host(200ms)` 返回 `False`、耗时 200 ms（不永久阻塞）；原生 `synchronize()` **无 timeout 形参** ⇒ 只做「超时上报」语义 |
| S15 跨进程 IPC | ⚪ 标注不适用（与 PPU 同口径） |
| S16 流数量配额 | 与 S10 同证据（1000 次无泄漏） |
| S6 集合通信与流绑定 | ⛔ 未做（属训练腿；**集合通信后端名尚未探测**） |
| S7 图捕获流语义 | ⛔ 未做（`graph_capture` 未声明；`torch.supa` 有 `SUPAGraph` 类但**类存在 ≠ 语义已验证**） |

---

## 1. 环境速查（先看这里）

| 项 | 值 |
|---|---|
| 机器 | `wz-server008`｜宿主 `Ubuntu 22.04.5 LTS`｜192 线程 / 1.5 TiB 内存｜**有公网**（GitHub 直连 200） |
| SSH | `ssh Biren-1` → `hliu553@123.159.196.74:38022`（公钥免密已配；密码在 `~/.ssh/creds_biren.txt`，**不进任何 git 仓**） |
| sudo | 口令**同登录密码**；`sudo -S` 已验证（`(ALL:ALL) ALL`） |
| docker | 组权限已自解决（`sudo usermod -aG docker hliu553`，2026-11-11）；⚠️ **组身份在登录时注入**，旧会话/ControlMaster 复用连接不生效，重开即好 |
| 加速卡 | `Biren166M`，单卡 **65024 MiB（64 GB）**；原 8 张，**2026-11-11 起 6 张**（见 §4） |
| 驱动 / 软件栈 | 内核模块 `biren 1.12.0`｜**BR-SMI 1.12.0**｜**SUPA 1.12**；SDK 在 `/usr/local/birensupa/` |
| 存储 | `/` 余 425G（家目录可写）｜`/share` 3.5T **96% 满且只读**｜`/data` 91% **只读**（有 `hf_cache`）⇒ **数据落 `~/`** |
| 容器 | `dc-biren-hliu553`（`--restart unless-stopped`），工作目录挂 `~/hliu553_biren → /work` |
| 取卡 | `-e BIREN_VISIBLE_DEVICES=5,7`（按**设备节点序号**，不是 `brsmi` 显示索引 —— 见 §2 第 4 条） |

**一条命令跑完已过的判据**（宿主侧执行，脚本在 `probes/biren_conf_stream_20261111_out/scripts/`）：

```bash
bash biren_conf_run.sh      # conformance 13 + 6
bash biren_stream_rerun.sh  # 多流 8 子项 + i6 连续三次
bash biren_facts_run.sh     # S12 流优先级三问 + i6 随工作量缩放
```

---

## 2. 关键认知（实测得出，接入前必读）

1. **FlagOS 官方没有壁仞镜像线** —— `flagos-ai/build-infra::configs.yaml` grep `biren` **0 命中**（与寒武纪同属
   「走厂商官方镜像」）。故镜像事实只能从**镜像 tag + ID** 来，不能按 `configs.yaml` 推断。
   定档镜像 `birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx`
   （镜像 ID `sha256:081634a3…`，**本地无 digest**）。决定性依据：机内在跑的 xbench 容器**同镜像同入口可用**（可复现路径，不是版本推断）。
2. **设备 API 是 `torch.supa`**（pip 包 `torch-br`，`PrivateUse1`）——`torch.cuda.device_count()` 与 `torch.xpu` 均为 0。
   后端据此 `name="biren"` / `device_type="supa"`。
3. **容器里任何 python 调用前必须 source 厂商环境脚本**：
   `. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set`。
   漏掉 ⇒ `import torch_br` 报 `ImportError: libsupti.so.1: cannot open shared object file`。
   原因是镜像把库路径放在**入口脚本**里，而 `docker exec` 起的是新进程、不继承入口进程环境（**不是镜像坏**）。
4. **两条取卡铁律**（2026-11-11 现场踩到）：
   ① `BIREN_VISIBLE_DEVICES` 选的是**设备节点序号** `/dev/biren/card_N`，**不是 `brsmi` 显示索引** ——
      卡掉线后 `brsmi` 会重排索引而节点名不变（现 `brsmi 3 = 85:00.0 = card_5`），照 `brsmi` 索引挑会挑到别人的卡；
   ② `brsmi` 读数正常 **≠** 卡可用（新故障卡复位后仍 `N/A` + 128 MiB 残留）⇒ **取卡前跑最小算子探针**。
5. **吞吐/重叠类读数必须先预热**：某形状**首次在命名流上执行**要付一次性编译代价（实测循环首轮 `512²@512²` **2153.67 ms**，
   第 2–4 轮 `0.29–0.50 ms`）；i6 那条 `-21150%` / `-75237%` 的负重叠率就是这么来的。预热并放大工作量后重叠率收敛 **±2% 内**
   （n/batches = 512/4、1024/8、2048/8、2048/16、4096/8 两轮独立测量）⇒ **本栈这套 3 流 + 4 事件流水线没有可测重叠收益**，
   属性能域，**不作验收判据、也不据此归因厂商**。单步原语本身不慢：流上下文切换 `0.058 ms`、H2D `non_blocking` `0.098 ms`、matmul512 `0.214 ms`。
6. **不要传 `--device`**：宿主 `/etc/docker/daemon.json` 是 `default-runtime: biren`，toolkit 会自动注入卡与库；
   手工 `--device` 反而干扰注入（首版容器因此报 `ErrorCode 100, no device`）。
7. **裸镜像里的 `.so` 是 0 字节占位**，首次容器启动时才注入真实库 ⇒ **别用 `ls -l` 判断镜像完整性**。
8. **`Event()` 未 `record()` 时原生 `query()` 返回 True（误报已完成）** —— 与 PPU / MLU590 同族；本层 `BirenEventAdapter`
   用 `_recorded` 标志修正为 `False`（`known_issues` 条目 `BIREN-EVENT-UNRECORDED-QUERY-TRUE`）。

9. **集合通信后端名是 `sccl`（不是 nccl/hccl/bccl）**，由 `import torch_br` 时注册（`torch_br/__init__.py:179`，`devices=["supa","cpu"]`，注册条件是 `torch_br._C._c10d_supa_init` 可用）。`dist.is_backend_available()` 对 16 个常见厂商名（nccl/hccl/cncl/bccl/bcl/supa/biren/rccl/accl/pccl/xccl…）**全部 False**，猜名猜不到 ⇒ **必须以源码注册处为准**。实用面 **supa-only**（CPU 张量 `all_reduce` 被拒）；训练腿用 `DC_DIST_BT=sccl`。
10. **`BIREN_VISIBLE_DEVICES` 是容器级变量，进程内改会挂住**（不是无效）——实测 `mp.spawn` worker 里改它导致整段判据静默超时（连「错名必须报错」都不报错）；选卡交给 `-e`，进程内只用 `torch_br.supa.set_device(rank)`。

---

## 3. 环境版本组合

| 层 | 值 | 取法 |
|---|---|---|
| 宿主 OS / 内核 | `Ubuntu 22.04.5 LTS` / `5.15.0-119-generic` | `hostnamectl` / `uname -r` |
| 驱动 | `biren 1.12.0`（SUPA 1.12） | `modinfo biren` / `brsmi` 首行 |
| 镜像 | `birensupa-smartinfer-vllm:26.07.09-…-br1xx`，ID `sha256:081634a3…` | `docker inspect` |
| 容器内 torch | `2.8.0+cpu` | 探针 `[env]` 行 |
| 厂商插件 | `torch_br 1.8.7.20800+br1xx` | 同上 |
| 设备命名空间 | `torch.supa`，`device_count=2` | 同上 |
| vLLM（镜像自带） | `0.21.0` | `import vllm` |
| 统一运行时 | `runtime:biren=0.1.0`（`__version__`） | `conf13.json` 的 `env` |

> ⚠️ 版本事实的**唯一来源**是本表左列的命令读数；镜像 tag 里的 `26.07.09` / `pt2.8.0` / `br1xx` 是**厂商命名**，
> 不要拿它反推 torch 或 SDK 版本（本轮实测容器内 `torch 2.8.0+cpu`，SDK 目录为 `/usr/local/birensupa/sdk/1.10.0.0-24145.lkg`）。

---

## 4. 阻塞与需要协调的事项

| # | 事项 | 现状（2026-11-11） | 归属 |
|---|---|---|---|
| 1 | **卡池故障扩散** | 原 8 张 → **6 张**：`03:00.0`、`63:00.0` 当日彻底 `device remove`；`05:00.0`（13:34）、`83:00.0`（14:22）**新发同型故障**（`smmu context fault(Read) fsr=0x402` → `PQM hang recovery fail` → `found spc hang, mask 0xffffffff` → `hot reset`），复位后仍 `N/A` + 128 MiB 残留 | 提供方/硬件（工单对象是 **Bus-Id `03/63/05/83` 四张**，**不是**交接清单写的 7 号卡） |
| 2 | 我方在用两张卡 | `85:00.0`(`card_5`)、`a5:00.0`(`card_7`) **日志干净**（最后事件仅我方重建容器的 `SIGTERM`） | 本方向 |
| 3 | ~~集合通信后端名~~ | ✅ **已探测（11-11）：`sccl` / SCCL 10.8.0**，2-rank `all_reduce` PASS，错名报错无回落 ⇒ 训练腿可直接用 | 已完成 |
| 4 | `serve_standard.sh` 缺 `biren)` 分支 | 未补（新芯片必经项） | 本方向 |
| 5 | `/share` 96% 满、`/data` 只读 | 镜像与模型优先复用机内已有，新增数据落 `~`（余 425G） | 环境约束 |

---

## 5. 目录内容

| 路径 | 内容 |
|---|---|
| `docs/BIREN_BR0_CONFORMANCE_AND_STREAM_20261111.md` | **conformance 13+6 与多流 16 项逐项对照**（含 S12 优先级三问、i6 负重叠率定性、探针文案错归属修正、当日卡况） |
| `docs/BIREN_BR0_IMAGE_AND_RUNTIME_20261111.md` | 镜像选型三条依据、容器参数定案、三级判据、六条坑清单 |
| `docs/BIREN_BR0_RESOURCE_SURVEY_20261111.md` | 资源普查（宿主/卡/驱动/SDK/存储/镜像候选）+ **§2.1 当日卡况更新** |
| `docs/BIREN_CARD02_FAULT_REPORT_20261111.md` | 卡故障专项（时间线 + 内核原文，可直接作上游报告附件） |
| `probes/biren_conf_stream_20261111_out/` | conformance 与多流证据（`MANIFEST.md` 含 md5 与期望读数）+ 7 份复跑脚本 |
| `probes/biren_image_20261111_out/` | 镜像内容探查脚本（含失败尝试，留作反证） |
| `probes/biren_resource_20261111_out/` | 首轮资源普查脚本 |
| `probes/biren_card02_20261111_out/` | 卡故障内核时间线日志（239 行）与复核脚本 |
| 后端实现 | `../prototype/runtime/backends/biren/backend.py`（能力声明依据 + `known_issues` 九字段） |

---

## 6. 已完成 / 后续

**已完成**：环境打通（SSH/sudo/docker）→ 资源普查 → 镜像定档 → 后端落地（13 抽象 + `build()` + `supports()` 如实声明 9 键）
→ 离线契约自检 83/0/1 → 真机 smoke 46/0 → conformance 13/13 + 6/6 → 多流 16 项逐项对照 → **集合通信后端名探测（`sccl`，2-rank all_reduce PASS）**。

**顺手修的工具缺口**：`prototype/scripts/backend_offline_check.py` 的 `VENDOR_ROOTS` 阻断名单与 `fresh_import` 清理名单
**都硬编码且少了 `torch_br`** ⇒ 「缺厂商扩展必须报错」这条判据会因上一轮残留而**静默通过**；两处已补，并新增 `_stub_biren`（工具此前只认四家）。

**后续（按手册顺序）**：① ~~集合通信后端名探测~~ ✅（`sccl`）→ 训练腿（两 rank / 50 步）→ ② 推理腿前向与服务化（镜像自带 vllm 0.21.0）
→ ③ `serve_standard.sh` 补 `biren)` 分支 → ④ 错误闭环四类注入 → ⑤ 职责响应审计 78 项 → ⑥ 接入完成前按手册 §7 + manifest §1 做**同口径核验**。
