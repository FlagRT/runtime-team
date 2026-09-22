# 寒武纪 MLU590 接入方案（第三个芯片实例）

> 日期：2026-09-22 ｜ 负责人：Kistich（hliu553）｜ 方向：设备抽象与执行上下文（device-context）
> 性质：**接入工作方案 + 真机执行手册**（本实例专属，不迁移）
> 依据：《新芯片接入手册》`../../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`（8 步流程）
> 　　　《运行时层接口约定》`../../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` §2（Backend 插件接入规范）
> 定位：**按接入规范新建实例** —— 迁移的是**规范与方法**；910C / P800 的**实现与结论不迁移**。

---

## 0. 当前进度（结论先行）

| 手册步骤 | 内容 | 状态 |
|---|---|---|
| 第 0 步 | 环境普查（7 项前置风险） | ✅ **09-22 完成** —— `CAMBRICON_MLU_ENV_REPORT_20260922.md` |
| — | 镜像渠道调研与**定档** | ✅ **09-22 完成** —— `CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` |
| 第 1 步 | 厂商 PyTorch 栈判别（A/B/C/D 四路径） | ⏳ **待容器**（预期路径 **C：PrivateUse1**，见 §2） |
| 第 2 步 | 镜像就绪 5 条判据（含依赖链完整自检） | ⏳ 待容器 |
| 第 3 步 | **实现 backend（13 抽象）** | ✅ **09-22 完成（代码层）** —— `prototype/runtime/backends/cambricon/` |
| 第 3.5 步 | **离线契约自检**（无设备） | ✅ **35 通过 / 0 失败**（新增工具，见 §4.2） |
| 第 4 步 | conformance 13 例 + 推理 6 例 | ⏳ 待容器 |
| 第 5 步 | 两条腿（2 卡训练 / 单卡推理 + 服务化） | ⏳ 待容器 |
| 第 6 步 | 错误闭环（四类注入） | ⏳ 待容器 |
| 第 7 步 | 多流 16 项基线逐项比对 | ⏳ 待容器 |
| 第 8 步 | 证据归档 + 验收清单 13 项 | ⏳ |

**⇒ 当前唯一硬阻塞：`docker` 组权限**（+ `/srv/hliu553`）。
镜像侧、代码侧均**已不阻塞**（镜像定档且实测可匿名拉取；后端已落地并通过离线自检）。

**⚠️ 一句话边界**：上表的"完成"指**代码层完成**。
本实例**尚未在任何寒武纪设备上跑过一次** ——
`torch.mlu` 的真实 API 形态、conformance 是否通过、两条腿能否跑通，
**全部必须到容器内实测才能下结论**。

---

## 1. 环境与阻塞（第 0 步结论摘要）

| 项 | 实测值 |
|---|---|
| 机器 | `Mlu-1` = 10.1.1.21（`tza-0a06-ai01-em9`）、`Mlu-2` = 10.1.1.22（`tza-0a06-ai02-em9`），SSH 免密可用 |
| 加速卡 | 各 **8 × MLU590-M9**，单卡 **96 GB**，健康 `Good` |
| 宿主驱动 / 固件 | **v6.2.29** / v1.5.0；`cnmon` = CNMON v6.2.29（宿主工具） |
| MLU 软件栈 | 宿主**无 `/usr/local/neuware`** ⇒ 软件栈必须走**容器镜像** |
| 数据盘 | `/srv` **11 T**（09-22 实测：`3.3 T 已用 / 6.7 T 可用`，33%）；docker 数据目录本就是 `→ /srv/var/lib/docker` 的符号链接 |
| 共享用户 | 同机他人：`gpfs, liangfan1, daizijian, huangxiang, qiyiyan, leihuhu` ⇒ **共享机用卡纪律照 P800 办** |
| **验收模型** | ✅ **共享 HF 缓存里已有**，无需下载：<br>`/srv/data/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`<br>（`drwxr-xr-x root root` 全局可读，**只读复用，不得写入**；与另两实例**同一模型** ⇒ 结果可比） |

**硬阻塞（需 root / 管理员，两台各一次）**：

```bash
sudo mkdir -p /srv/hliu553 && sudo chown -R hliu553:hliu553 /srv/hliu553 && sudo chmod 750 /srv/hliu553
sudo usermod -aG docker hliu553        # 执行后需重新登录 SSH 生效
```

**镜像定档**（已定，无需再申请）：

```text
harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0
digest sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2   (2.5 GiB)
栈：py3.10 / torch 2.7.1+cpu / torch-mlu 1.29.2+torch2.7.1 / torch-mlu-ops 1.8.0 / triton 3.2.0+mlu1.7.2
官方标注宿主驱动前置 6.2.15 —— 我们实测 v6.2.29，**同 6.2.x 线**
```

> 定档理由与另一档（`neuware4.7.2`，需宿主驱动 **6.5.48**）的处置见
> `CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.1 定档决策 / §0.2 驱动升级上报预案。

---

## 2. 厂商栈判别（第 1 步，已给出**预期**与判别命令）

**预期路径：C —— 厂商私有命名空间（PrivateUse1）**
`torch_mlu` 把 MLU 注册为 PyTorch 的 PrivateUse1，设备串前缀 **`mlu`**：

```python
import torch, torch_mlu
torch.mlu.device_count()      # → 8
torch.device("mlu:0")         # → 有效设备串
```

故本后端 `name = "cambricon"`（厂商标识）、`device_type = "mlu"`（设备串前缀）——**两者不同名是刻意的**（后端名按厂商、设备串按命名空间），与前两家同一约定。

⚠️ **PrivateUse1 是进程级单例** ⇒ 本后端**不得与 `torch_npu` / `torch_fl` 同进程混用**。

**进容器第一件事就是跑这几条判别命令**（手册 §2 原文，逐条记录输出）：

```bash
python3 -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
python3 -c "import torch.npu;  print('npu:',  torch.npu.device_count())"   # 预期 ImportError（寒武纪无此栈）
python3 -c "import torch.mlu;  print('mlu:',  torch.mlu.device_count())"   # ← 预期 8
python3 -c "import torch.xpu;  print('xpu:',  torch.xpu.device_count())"   # 预期失败
python3 -c "from vllm.platforms import current_platform; print(current_platform)"
```

**最后一条决定推理腿形态**：打印具体平台 → 可直接起服务；打印 `UnspecifiedPlatform`
→ 认不出设备，必须另有厂商移植版 vLLM 或平台插件（寒武纪已知有厂商移植版 `Cambricon/vllm-mlu`，
但**镜像内是否已装、形态是哪一种，未验证**）。

---

## 3. 已完成的接入动作（第 3 步，代码层）

### 3.1 新增后端：`prototype/runtime/backends/cambricon/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 导出 `CambriconBackend` / `CambriconEventAdapter` / `build` |
| `backend.py` | **13 个抽象方法** + `build()` 工厂 + `supports()` 如实声明 + `known_issues()` + `info()` + `device_state()`；内含 `CambriconEventAdapter`（事件语义适配） |

**五域覆盖**：设备（count / set_device / memory_stats / probe）/ 内存 / 流-事件（create / current / context / 有界同步 / 有界主机等待）/ 错误翻译 / 状态恢复。

### 3.2 能力声明（**如实声明，未验证的一律不声明**）

| 能力 | 声明 | 原因 |
|---|---|---|
| `device` `memory` `stream` `event` `multidevice` | ✅ | 实现齐备；8 卡为实测环境事实 |
| `bounded_sync` | ✅ | 主机侧 `wait_host` 真有界（轮询实现）；流同步为**超时上报**语义（同前两家口径，已如实标注） |
| `recovery_probe` | ✅ | 探针级探活 |
| `device_state` | ✅ | 复用芯片无关的四态机（进程内状态机，不依赖厂商原语） |
| `error_map` | ❌ | **厂商错误码是否透出到 Python 层未实测** ⇒ 无凭据不声明；分级只走 `message_hint` / `default`，且**若底层骨架意外给出 `code_map` 一律如实降级标注**，不冒充码表命中 |
| `recovery_real` | ❌ | **是否有设备级重置/重建原语未实测** ⇒ 不写一条猜的重建序列（写了会制造"看起来支持"的假象） |
| `graph_capture` | ❌ | 图捕获入口是否存在未验证 |
| `stream_priority` | ❌ | 该 API 形态未验证；且同类 API 在昆仑芯上会触发 PyTorch 自身 `INTERNAL ASSERT`，故 `stream_priority_range()` **主动返回 None、不乐观透传** |

### 3.3 框架侧改动（都是"登记"，未改任何接口签名）

| 文件 | 改动 |
|---|---|
| `runtime/backends/registry.py` | `_KNOWN_BACKENDS` 加入 `"cambricon"`（自动发现的登记点） |
| `runtime/smoke_runtime.py` | 第 [6] 节真实后端自检的挑选顺序加入 `cambricon`（依赖缺失会如实 SKIP 后继续试下一家） |
| `runtime/__init__.py` | docstring 补上后端名示例 |
| `runtime/proto/proto_train_leg.py` | **寒武纪不给 `DC_DIST_BT` 就报错退出**（见下） |
| `scripts/serve_standard.sh` | 新增 `cambricon` 分支 + `card_snapshot` 支持 `cnmon` / `torch.mlu` 查卡降级 |
| `docs/SERVICE_STARTUP_STANDARD_20260920.md` | 登记 `cambricon` 分支（标注"尚未真机验证"） |

**为什么训练腿要"不给就报错"**：集合通信后端名**同一套代码跨芯片就不同**
（910C = `flagos`、P800 = `flagcx`，手册 §9 坑 3 已实证），寒武纪**不可类推**。
而 `proto_train_leg.py` 的兜底是 `gloo` —— 那会**静默退化为纯 CPU 集合通信**：
训练脚本照样跑完、loss 照样下降，但**设备侧通信根本没被验证**。
这类"看起来通过"的结果比失败更糟，故显式拦住并给出探测方法。

### 3.4 后端内的 `known_issues()`（已实测的环境约束，非厂商缺陷）

| id | 级别 | 内容 |
|---|---|---|
| `MLU-DRIVER-TIER-CONSTRAINT` | info | 宿主驱动 v6.2.29（6.2.x 线）⇒ 只能用 `neuware4.4.3` 档；`neuware4.7.2` 要求 6.5.48 |
| `MLU-HOST-NO-NEUWARE` | info | 宿主无 `/usr/local/neuware` ⇒ 一切验证必须在带卡容器内 |

> **本实例当前没有任何厂商缺陷结论**（尚未进容器）。容器内实测后按手册 §9 模板补条目。
> 清单为空也要如实为空 —— 不凑数。

---

## 4. 本地已验证（**不是真机结论**）

### 4.1 静态检查与回归

| 项 | 结果 |
|---|---|
| 语法编译（7 个改动/新增 py 文件） | ✅ 全部通过 |
| `bash -n scripts/serve_standard.sh` | ✅ 通过 |
| 本机 `smoke_runtime.py`（无 torch 环境） | ✅ **28 通过 / 0 失败**；4 个真实后端如实 SKIP ⇒ **新增后端未造成回归** |
| `registry.discover()` 是否被新后端打断 | ✅ 不打断：cambricon 模块可导入（依赖是**延迟导入**），缺 `torch_mlu` 时只在调用期报错并被 discover 容错跳过 |

### 4.2 ⭐ 新增：**离线契约自检**（`prototype/scripts/backend_offline_check.py`）

**动机**：手册第 4 步（写 backend）与第 5 步（跑 conformance）之间有一段空档 ——
代码写完但机器没到位。这段时间最容易犯的是**实现层面的错**（抽象方法没实现全、
有界同步其实没上界、事件语义没修、错误翻译冒充码表、能力声明与实现不一致），
**这些不需要真实芯片就能查出来**。

```bash
python3 prototype/scripts/backend_offline_check.py --backend cambricon
# → 离线自检结果: 35 通过 / 0 失败
```

用 stub 把 `torch.mlu` 命名空间"空跑"一遍，8 组 35 条判据，逐条对应《接口约定》
或 conformance 用例的同口径判据（代码内注明 F1 / E2-v2 / E3 / R1-R5）：

| 组 | 覆盖 | 关键判据 |
|---|---|---|
| ① 发现/实例化 | ABC 强制 13 抽象方法齐全 | 实例化即失败于缺方法 |
| ② 设备域 | count / set_device / memory_stats / probe | 结构 `{total_mb,used_mb,free_mb}`、**用完还原当前设备** |
| ③ 流-事件 + 有界同步 | 真超时 / 已完成不误判 | 未完成 + `timeout_ms=0` → 必抛 `TimeoutError`；已完成 → 正常返回；实测 **200 ms 按时返回** |
| ④ 事件语义 | E3 / E2-v2 | **stub 刻意做成"未 record 也返回 True"的坏实现**，验证适配层确实修正为 `False` |
| ⑤ 错误翻译 | F1 三投影 + 诚实性 | 形状错 → `L2_PARAM`；未声明 `error_map` ⇒ `mapped` 必须为 `False` |
| ⑥ 恢复/设备状态 | R1-R5 | `recover_device` 返回 dict；`real` 如实说明不支持 |
| ⑦ 能力自洽 | `info().capabilities` ↔ `supports()` | 无不一致项 |
| ⑧ 缺厂商扩展 | **不得静默降级** | 抛 `RuntimeError` 且文案给出下一步动作 |

> ⚠️ **结论边界（勿外推）**：本自查**只证明实现逻辑与契约形态**，
> **不能替代 conformance**，也**不能证明后端在真机上能用**。脚本末行会重复这句。
> 该工具已回写进《新芯片接入手册》**§4.4**（新芯片接入的可复用资产）。

---

## 5. 真机执行手册（拿到 `docker` 权限后**按序执行**）

> 纪律：**每一步的原始输出都要落盘归档**（`probes/`，`.log` 已加 `!*.log` 例外）。
> **共享机用卡**：先 `cnmon` 看占用，挑**空闲**卡并在记录里写明用了哪张。

### A1 拉镜像 + 起容器

```bash
IMG=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0
docker pull "$IMG"
docker image inspect "$IMG" --format '{{index .RepoDigests 0}}'   # 核对 digest 是否 = e55b420e…

CT=dc-mlu590-hliu553
docker run -dit --name "$CT" \
  --net=host --shm-size=64g \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/cambricon_dev0 --device=/dev/cambricon_dev1 \
  --device=/dev/cambricon_dev2 --device=/dev/cambricon_dev3 \
  --device=/dev/cambricon_dev4 --device=/dev/cambricon_dev5 \
  --device=/dev/cambricon_dev6 --device=/dev/cambricon_dev7 \
  --device=/dev/cambricon_ctl \
  -v /usr/bin/cnmon:/usr/bin/cnmon:ro \
  -v /srv/data/hf_cache:/hf_cache:ro \
  -v /srv/hliu553:/work -w /work \
  -v /etc/localtime:/etc/localtime:ro \
  "$IMG" bash
docker exec -it "$CT" bash
```

**参数依据（不是编的）**：设备节点来自官方 `base/cambricon-neuware4.4.3.md` 给的
`--device /dev/cambricon_dev0 --device /dev/cambricon_ctl`；
`-v /usr/bin/cnmon` 与 `--cap-add=SYS_PTRACE --shm-size` 来自 FlagGems 官方周测配置
`FlagGems/.github/configs/weekly/MLU590-M9DE.yml`（该配置跑的正是 **MLU590-M9** 这台卡型）。
⚠️ 寒武纪**无容器 toolkit**（不像昇腾 `Ascend-docker-runtime`）⇒ 直接给设备节点，不用 `--runtime`。

### A2 环境普查（第 0 步复跑，作正式证据）

```bash
OUT=/work/preflight bash /work/prototype/scripts/preflight_env.sh
```

### A3 厂商栈判别（§2 的五条命令，逐条留输出）

### A4 镜像就绪 5 条判据（含**依赖链完整**自检）

```bash
python3 -c "import torch, torch_mlu; print('devs', torch.mlu.device_count())"      # 判据 1+2：预期 8
python3 -c "import triton; print('triton', triton.__version__)"                    # 判据 5（对照 P800 的坑）
python3 -c "from vllm.platforms import current_platform; print(current_platform)"  # 判据 4：推理腿形态
python3 -c "import torch.distributed as d; print('dist ok')"                       # 判据 3 的前置
cnmon | head -30                                                                   # 看卡与占用
```

### A5 放入原型 + 运行时代码层验证

```bash
# 方式一（有出网）：git clone 分支
cd /work && git clone -b kistich/device-context git@github.com:FlagRT/runtime-team.git rt
# 方式二（无出网）：从本机拷
#   scp -r dev/device-context/prototype Mlu-1:/srv/hliu553/

cd /work/rt/dev/device-context/prototype
python3 scripts/backend_offline_check.py --backend cambricon     # 期望 35/0（与本地一致）
python3 runtime/smoke_runtime.py --backend cambricon             # 接入自检
python3 runtime/conformance/runner.py --backend cambricon                       # 13 例
python3 runtime/conformance/runner.py --backend cambricon --cases infer_cases    # 推理 6 例
```

**这一步是接入完成的判定线**：13/13 + 6/6 全绿，或未支持项有**如实 stub-skip 说明**。

### A5b ⚠️ 先探测集合通信后端名（**训练腿的前置**，不可跳）

```bash
python3 - <<'PY'
import torch, torch.distributed as dist
print("mlu:", torch.mlu.device_count())
for bt in ("cncl", "cpu:gloo,mlu:cncl", "flagcx", "cpu:gloo,mlu:flagcx"):
    try:
        import os; os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29511")
        dist.init_process_group(bt, rank=0, world_size=1, timeout=__import__("datetime").timedelta(seconds=30))
        print("OK  ", bt); dist.destroy_process_group()
    except Exception as e:
        print("FAIL", bt, type(e).__name__, str(e)[:80])
PY
```

把可用者写进 `DC_DIST_BT`（形如 `cpu:gloo,mlu:<backend>`）后再跑训练腿；
**探测结果回填本文档与 `runtime/backends/cambricon/backend.py` 顶部「未实测清单」第 10 条**。

### A6 多流 16 项基线

```bash
DC_BACKEND=cambricon python3 prototype/probes/probe_stream_semantics_full.py   # 期望 STREAM_SEMANTICS_PASS 8/8
```

### A7 训练腿（2 卡，50 步；门禁档）

```bash
DC_BACKEND=cambricon MLU_VISIBLE_DEVICES=0,1 \
DC_DIST_BT=<A5b 探测结果> \
DC_ROOT=/work/rt/dev/device-context \
DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
DC_OUT_DIR=/work/scratch MAX_STEPS=50 BATCH=4 SEQ=128 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py
```

判据：两 rank 均 `TRAIN_LEG_PASS 6/6`，loss 正常下降。
参考锚点：910C `15.4497 → 11.15` / 2117 tok/s；P800 `15.4488 → 11.1481` / 3482 tok/s。
**50 步不是随便定的**（挂死类缺陷往往在第 n 次通信才出现）。

### A8 推理腿（前向 + 服务化）

```bash
# 前向
DC_BACKEND=cambricon MLU_VISIBLE_DEVICES=0 \
DC_MODEL=<同上 snapshot 路径> python3 runtime/proto/proto_infer_leg.py
# 服务化（统一入口，不要自建脚本）
DC_BACKEND=cambricon DEV=0 STOP_AFTER=1 \
MODEL=<同上 snapshot 路径> bash prototype/scripts/serve_standard.sh
```

判据：维度 **1024**、语义区分度、吞吐与 p50/p90、真实异常注入 → `L2_PARAM/raise` 且业务继续；
服务化另加 **超长输入 → HTTP 400 → L2_PARAM/raise** 与**同卡共存**。
参考锚点：910C 区分度 0.4123 / 108 句·s⁻¹；P800 0.4102 / 30.70 句·s⁻¹。
⚠️ 停机必须连 `EngineCore` 子进程一起 `kill -9`（P800 实测残留占卡 73850 MiB）。

### A9 错误闭环（四类注入）

```bash
DC_BACKEND=cambricon python3 runtime/proto/proto_error_recovery_loop.py
```

判据：四类注入（参数/资源/执行/致命）分级与处置正确、业务继续。
⚠️ 调用纪律：传**完整原始异常/服务错误消息，不得截断**（截断会把参数错退化成无意义重放）。

### A10 归档 + 回填

- 原始日志/JSON → `MLU590/probes/`（`.log` 已有 `!*.log` 例外；**写"已入库"前必须 `git ls-files` 实测确认**）
- 回填：本文档 §0 进度表、§6 验收清单、`backend.py` 的「未实测清单」与 `_capabilities`
- 更新 `MLU590/README.md`、主看板、`STATUS.md`

---

## 6. 验收清单（手册 §8 的 13 项，逐项标当前状态）

| # | 项 | 判据 | 当前状态 |
|---|---|---|---|
| 1 | 环境打通 | 连得上 / 有权限 / 有可写目录 / 会挑空闲卡 | 🟡 SSH ✅、资源 ✅、**`docker` 组 ⛔** |
| 2 | 厂商栈判别 | 明确落 A/B/C/D 哪条路径并记录证据 | 🟡 预期 **C（PrivateUse1 / `mlu`）**，⏳ 待实测确认 |
| 3 | 镜像就绪 | 5 条判据全过（含依赖链完整自检） | 🟡 镜像**已定档 + 实测可匿名拉取**；5 条判据 ⏳ 待容器 |
| 4 | backend 落地 | 13 抽象 + `build()` + `supports()` 如实声明 | ✅ **代码层完成**（+ 离线自检 35/0） |
| 5 | conformance 13 例 | 全绿或如实 stub-skip | ⏳ |
| 6 | conformance 推理 6 例 | 全绿 | ⏳ |
| 7 | smoke 自检 | 全通过 / 0 失败 | ⏳（本机 28/0 是无 torch 环境的回归，不算） |
| 8 | 训练腿 | 两 rank `TRAIN_LEG_PASS 6/6` | ⏳（前置：A5b 集合通信后端名） |
| 9 | 推理腿（前向） | 维度 / 范数 / 区分度 / 时延 / 异常分级 | ⏳ |
| 10 | 推理腿（服务化） | 含超长输入防御与同卡共存 | ⏳ |
| 11 | 错误闭环 | 四类注入分级处置正确、业务继续 | ⏳ |
| 12 | 已知问题如实声明 | `known_issues()` 结构化 + 开跑前告警 | ✅ 已实现（当前仅环境约束 2 条，**无厂商缺陷结论**） |
| 13 | 证据归档 | 原始日志/JSON 入版本库 + `git ls-files` 实测 | 🟡 第 0 步证据已入库；本轮 ⏳ |

---

## 7. 风险与应对

| # | 风险 | 应对 |
|---|---|---|
| 1 | **`torch.mlu` API 形态与预期不符**（`mem_get_info` 缺失/参数不同、`Stream.synchronize` 不收 timeout） | 后端已按**多条兜底链**写（显存三条取值路径、`synchronize` 两种调用形态），并把每条标为「未实测」；实测后回填 |
| 2 | **推理腿形态未知**（厂商移植版 vs 社区版 + 插件） | `current_platform` 一跑即知；`serve_standard.sh` 的 cambricon 分支**刻意不预设任何厂商专用环境变量**（前两家变量互不通用，手册 §9 坑 5） |
| 3 | **集合通信后端名不同** | 已用"不给就报错"拦住静默退化；A5b 专门探测 |
| 4 | **共享机误用他人占用的卡** | P800 曾因此撤销一个"缺陷"结论 ⇒ 用卡前 `cnmon` 挑空闲卡、记录用卡、异常先换卡复测 |
| 5 | **老档位（4.4.3）相关问题** | 不得凭版本号升级宿主驱动；走 `CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.2 的四条门槛 + 上报模板 |
| 6 | 停机残留占卡 | `serve_standard.sh` 已含 `pkill -9` EngineCore 清理 |
| 7 | 结论被外推 | 所有产出标注取得时的**档位/条件**（同 P800 的「KL3 未设置条件下取得」纪律） |

---

## 8. 不做什么（职责边界）

| 不做 | 归属 |
|---|---|
| 生产级性能调优 | 精度 / 调优方向 |
| 算子实现、通信库优化、显存池调优、调度策略 | 各自方向 |
| 擅自改上游口径 / 动公共资产 | 由总组裁定 |
| 规模化验收（Megatron-LM-FL / vllm-plugin-FL 适配） | 框架适配方向 |

**问题路由**：接入过程暴露的问题按"设备因素归设备"分流 —— 属五域内的先修；
算子/通信/显存/调度/性能一律对外提交。
