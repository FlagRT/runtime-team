# 新芯片接入手册（New Chip Onboarding Manual）

> 版本：v1.0 ｜ 日期：2026-09-20 ｜ 维护：device-context 子方向（Kistich）
> **读者**：接入第 3 家及以后芯片的工程师；以及需要判断"某芯片接进来要多久"的评审方。
> **适用前提**：统一运行时原型 `dev/device-context/prototype/` 已就位（自包含，可整体拷出）。
> **本手册的目标**：把一次新芯片接入，从"摸索"变成"照单执行"。
> **写法约定**：每条要求与坑都给出**实测依据**（现象 / 数字 / 报错原文），可独立阅读，不写成"请看某文件某章"。
> 凡本方向未取得证据的，明确标注「未验证」，不臆造。

---

## 0. 一句话主张与实测锚点

**新增一家芯片 = 实现一个 backend（13 个抽象方法）+ 跑通 conformance（13 例 + 6 例）。**

已有两家实例的实际成本：

| 实例 | 芯片 | 结果 | 接入耗时 |
|---|---|---|---|
| 第 1 家 | 华为昇腾 910C | conformance 13/13 + 6/6；两条腿全闭环 | 基线实例（自建原型期） |
| 第 2 家 | 昆仑芯 P800 | conformance 13/13 + 6/6；smoke 42/0；训练腿 6/6；推理腿前向 13/13、服务化 10/10；错误闭环两设置各 5/0/0 | **单日完成**（月度目标：单个 ≤5 人天） |

⇒ 第 2 家的"单日完成"是**接入动作本身**（写 backend + 跑 conformance），
不含**环境打通**（机器/驱动/镜像/网络）与**厂商缺陷排障**。后两项才是真正的工期风险，见 §1 与 §7。

---

## 1. 第 0 步：把环境风险前置（比写代码更耗时）

**P800 的实测教训：接入动作只花一天，环境打通卡了好几天。** 因此接入前先把下面 7 项列表问清楚：

| # | 要确认的事 | P800 的实际情况（作为对照） |
|---|---|---|
| 1 | **连得上吗** | SSH 走**非标准端口 26008**；`~/.ssh/config` 里 `Host` 段缺 `Port` 行时，表现为 `Connection refused`，**极易误判为网络/VPN 问题** |
| 2 | **有权限吗** | 需加入 `docker` 组（否则每次 sudo）；需自有可写数据目录（`/data2/hliu553`） |
| 3 | **镜像怎么来** | 官方手册要求 `docker pull`（59.9 GB）+ `docker load`（32 GB 产出包）；本机镜像库**已有**该镜像 ⇒ 两步全跳过 |
| 4 | **谁是共享用户** | 该机为共享机（25 人在线 / 22 个容器）；**用卡前必须挑空闲卡**，见 §7 坑 2 |
| 5 | **磁盘够吗** | 先 `findmnt -T /var/lib/docker` 看 docker 数据目录的真实挂载点（P800 是 bind mount 到 `/data1`，5.8 T） |
| 6 | **网络通不通** | 镜像/whl 源是否可达：P800 上 `resource.flagos.net` 可达（HTTP 200），`flagos-pypi-hosted` 源可装 3.3 GB 的 `flagtree` wheel |
| 7 | **拓扑清楚吗** | P800：XPU0-3 属 NUMA0、XPU4-7 属 NUMA1；组内走 XL 私有链路、跨组走 SYS；**2 卡实验优先取同组相邻卡**（如 6,7） |

**用卡纪律（共享机）**：`xpu-smi` 挑**连续且空闲**的卡并记录用卡。
P800 实测曾因误用被他人占用的卡（观察到他人卡 1 占用 166→502 MiB / 100%），
把现象误判为"通信库适配缺陷"——换成空闲的卡 6/7 后三类通信全通过，该结论已撤销。

**一键执行上面 7 项**（拿到机器就先跑这一条，产出报告可直接作为环境报告证据）：

```bash
bash prototype/scripts/preflight_env.sh              # 自动探测；输出到 /tmp/dc_preflight/
OUT=/srv/<user>/preflight bash prototype/scripts/preflight_env.sh   # 指定输出目录
```

脚本只读、不装任何东西；缺项**如实打印「未取得 / 不可用」，不猜测、不补零**。
其中两处是踩过的坑：第 4 项必须看 **docker 数据目录的真实挂载点**（`findmnt -T $(docker info --format '{{.DockerRootDir}}')`，
P800 曾因 `du -x` 跨文件系统即停而误判根分区容量）；第 2 项的芯片信息在**容器内通常拿不到厂商工具**
（`npu-smi` / `xpu-smi` / `cnmon` 都是宿主工具）⇒ 脚本自动降级为 torch 侧查询。

---

## 2. 第 1 步：判别厂商 PyTorch 栈（决定走哪条路）

这是**接入手册的第一项**，因为它决定后面所有动作的形态。

**判别命令（拿到机器就跑）**：

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch.npu; print('npu:', torch.npu.device_count())"          # 昇腾系
python -c "import torch.mlu; print('mlu:', torch.mlu.device_count())"          # 寒武纪系
python -c "import torch.xpu; print('xpu:', torch.xpu.device_count())"          # 通用 XPU 命名空间
python -c "from vllm.platforms import current_platform; print(current_platform)"  # 推理腿服务化可行性
```

**四条路径与判据**：

| 路径 | 特征 | 处理方式 | 已知实例 |
|---|---|---|---|
| **A. 厂商官方 PyTorch 扩展**（首选） | 能 `import torch_<vendor>` 且 `device_count() > 0` | 直接按其命名空间实现 backend | 昇腾 `torch_npu`（`npu`） |
| **B. 复用 `torch.cuda` 命名空间** | `torch.cuda.device_count() > 0`，但设备实际是国产芯片 | 按 `device_type="cuda"` 实现；**必须靠 `vendor` 字段区分厂商**（见 §5 修订建议 1） | 昆仑芯 P800（XPytorch + `torch_xray` 符号重写，编译标志 `USE_XPU=OFF`） |
| **C. 私有命名空间（PrivateUse1）** | `import torch_xxx` 后设备串是自有前缀 | 可用；但**同一进程只能激活一家**（PrivateUse1 是单例），接口约定须写明"进程级切换" | `torch_npu`→`npu`、`torch_fl`→`flagos`、`torch_mlu`→`mlu` |
| **D. 无 PyTorch 集成，只有底层 C API** | 上述全部失败 | 需自研薄桥接层，或确认 `torch_fl` 是否已为该厂商编译 backend —— **成本最高，须先评估再动手** | 未遇到 |

**⚠️ 关于 `torch_fl` 的定位（常见误解）**：`torch_fl` **不是**"没有标准命名空间时的兜底"。
它是**编译期绑定单一加速器**的（编译时传 `ACCELERATOR=<vendor>`，换厂商要重编译），
平台矩阵只覆盖特定几家（`cuda / metax / ascend / ppu / dcu / gcu / musa / bpu / tsingmicro`，
**没有 cambricon、没有 biren**）。它的定位是"设备接入层的**参照实现**"与全组预研 B 线，
主线是 **Route A：各芯片厂商官方插件**。
对昇腾实测：必须 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 且先 `import torch_fl` 再 `import torch`，
**禁止与 `torch_npu` 共存**（镜像自带校验脚本直接报错）。

**推理腿的服务化判据（决定要不要额外补环境）**：

```bash
python -c "from vllm.platforms import current_platform; print(current_platform)"
```

- 打印出**具体平台**（如 `<vllm_fl.platform.PlatformFL>`）→ 可以起服务
- 打印出 **`UnspecifiedPlatform`** → 社区 vLLM 认不出设备，**必须另有平台插件或厂商移植版 vLLM**，否则推理腿的服务化形态无法验证

P800 正是第二种：容器里是**社区 upstream vLLM 0.13.0**（`vllm/platforms/` 只有 cpu/cuda/rocm/tpu/xpu，
**没有 kunlun**），靠 `vllm-plugin-FL` 提供 platform 才可用。
而 910C 相反：用的是华为官方移植版 `vllm-ascend`（自带 ascend 平台），
同一份 `vllm-plugin-FL` 在昇腾上**没有后端、必须禁用**（设 `VLLM_PLUGINS=fl` 后
`current_platform.device_type` 变空，抛 `RuntimeError: Device string must not be empty`）。

⇒ **同一插件跨芯片可用性可以完全相反**，这是选型时必须逐个确认、不能类推的一点。

---

## 3. 第 2 步：镜像就绪判据

镜像不需要"算子多、模型多"，标准是**能被本原型接管**。五条判据：

| # | 判据 | 命令 | 不满足的后果 |
|---|---|---|---|
| 1 | 设备可见 | `torch.cuda.device_count()` / `torch.npu.device_count()` > 0 | smoke 直接跑不了 |
| 2 | 厂商 torch 栈可导入 | 见 §2 判别命令 | 没有接入点 |
| 3 | 训练腿有可用集合通信 | 探测 `nccl / xccl / kccl / flagcx` 哪条可用 | 训练腿无法验证 |
| 4 | 推理腿有 vLLM 平台路径 | 见 §2 `current_platform` | 服务化形态无法验证 |
| 5 | **依赖链完整可导入** | 见下 | 表现为"设备问题"，实为包问题 |

**第 5 条是本手册最容易踩、也最难自查的一条**。P800 上它有两层，且**都表现为同一句报错**
`RuntimeError: Failed to infer device type`：

**第一层：镜像里没有 `triton`。** 官方 `-base`（及 `-base-ssh` 变体）开箱不含 `triton`——
site-packages 里既无目录、pip 也无记录。完整断链（需 `VLLM_LOGGING_LEVEL=DEBUG` 才看得到）：

```
vllm_fl/__init__.py:6             → from vllm_fl.utils import get_op_config
vllm_fl/utils.py:8                → import flag_gems
flag_gems/testing/__init__.py:3   → from flag_gems import runtime
flag_gems/runtime/configloader.py:4 → import triton
→ ModuleNotFoundError: No module named 'triton'
```

补齐（官方手册 1.2 节原文；实测源可达、wheel 3.3 GB、约 2 分 24 秒）：

```bash
python3 -m pip uninstall -y triton          # 反复执行至彻底卸载
python3.10 -m pip install flagtree===0.7.0rc3+xpu3.6 \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
```

**第二层：`flag_gems` 的子模块不完整。** 即使 `triton` 就位，`vllm_fl` 仍依赖
`flag_gems.runtime.backend.device.DeviceDetector`，而 site-packages 里那份 `flag_gems`
（`pip show` 看得见包）该子模块**导不进来** ⇒ 必须：

```bash
export PYTHONPATH=/env/FlagGems/src
```

**判据化的自检命令**（接入时先跑，别等起服务）：

```bash
python -c "import triton; print('triton', triton.__version__)"
PYTHONPATH=/env/FlagGems/src python -c "import flag_gems, vllm_fl; print('FL stack OK')"
```

> 镜像选型与入档/入锁流程见另一份《镜像选择与确定指南》；P800 的镜像选型与官方 `-base`
> 获取补齐三步见两实例配置手册。本手册只强调：**"官方镜像开箱即用"这句话要亲自验证**。

---

## 4. 第 3 步：实现 backend（13 个抽象方法）

**新建 `runtime/backends/<vendor>/`，不要复制任何既有芯片的实现。**

### 4.1 必须实现的 13 个抽象方法（`backends/base.py`）

| 域 | # | 方法 | 签名摘要 |
|---|---|---|---|
| **设备** | 1 | `device_count` | `() -> int` |
| | 2 | `set_device` | `(ordinal) -> None` |
| | 3 | `memory_stats` | `(ordinal) -> dict` |
| | 4 | `probe_device` | `(ordinal) -> bool` |
| **流 / 事件** | 5 | `create_stream` | `()` |
| | 6 | `create_event` | `()` |
| | 7 | `current_stream` | `()` |
| | 8 | `stream_context` | `(native_stream)` |
| | 9 | `synchronize` | `(ordinal, timeout_ms=None) -> None` |
| | 10 | `synchronize_stream` | `(native_stream, timeout_ms) -> None`（**有界**） |
| | 11 | `wait_event_host` | `(native_event, timeout_ms) -> bool`（**有界**） |
| **错误** | 12 | `translate_error` | `(exc, location="") -> FlagosError` |
| **恢复** | 13 | `recover_device` | `(ordinal, mode="probe", reason="") -> dict` |

**四个可选/带默认实现的方法**（按需覆盖）：

| 方法 | 用途 |
|---|---|
| `supports(capability) -> bool` | 能力如实声明；conformance 据此生成 stub-skip 报告 |
| `known_issues() -> list` | 结构化已知缺陷（见 §7 坑 5），让下游"读到后端即知坑" |
| `stream_priority_range()` | 流优先级范围；不支持则返回 `None` |
| `info() -> dict` | 汇总后端信息（含 `known_issues`） |

**两个必须声明的类属性**：`name`（后端名）、`device_type`（设备串前缀）。

> ⚠️ 当前 `device_type` 同时承担了"设备串前缀"与"厂商标识"两个语义，
> 而 P800 用的取值是 `"cuda"`（复用 cuda 命名空间）——**将来接 NVIDIA 也是 `cuda`，会撞名**。
> 建议同时声明 `vendor` 字段，见 §5 修订建议 1。

### 4.2 三条实现纪律

1. **如实声明，不伪造能力**：不支持就用 `supports()` 声明为 False，conformance 会生成 stub-skip 报告。
   P800 实测：厂商错误码不透出到 Python 层 ⇒ `error_map` 能力**如实跳过** `vendor_code_map` 用例，
   **不伪造、不补零**——这是接入规范的要求。
2. **有界同步必须真的是有界的**：`synchronize_stream` / `wait_event_host` 的 `timeout_ms` 要真正生效。
   910C 实测：未 record 的 event `wait_host(200ms)` 返回 `False`、耗时 201 ms（不永久阻塞）。
3. **错误翻译返回统一类型**：必须是 `api/errors.py` 的 `FlagosError` 实例，
   且 `category` 必须是 api 层的 `ErrorCategory` 枚举（跨模块枚举混用会踩 §5 修订建议 4 的坑）。

### 4.3 登记与自动发现

提供 `build()` 工厂函数并在 `registry` 登记名字；`registry.discover()` 扫描已安装插件。
注册表已预留常见后端名，新增一家通常**不需要改框架代码**。

### 4.4 无设备时的离线契约自检（2026-09-22 新增，来自第 3 家接入实践）

**动机（实测痛点）**：第 4 步（写 backend）与第 5 步（跑 conformance）之间有一段空档 ——
代码写完但机器/容器还没到位（权限未开、镜像未取到）。这段时间最容易犯的是**实现层面的错**：
抽象方法没实现全、有界同步其实没上界、事件语义没修（未 record 的 `query()` 误报"已完成"）、
错误翻译冒充码表命中、能力声明与实现不一致。**这些问题不需要真实芯片就能查出来。**

```bash
python3 prototype/scripts/backend_offline_check.py --backend <vendor>
```

用 stub 厂商命名空间把后端"空跑"一遍，覆盖 8 组检查：
① 发现/实例化（**ABC 会在实例化时强制 13 个抽象方法齐全**）；② 设备域 4 项；
③ 流/事件域 + **有界同步是否真有界**；④ 事件语义 E3/E2-v2；⑤ 错误翻译 F1 与诚实性
（**含"他厂码串入"负向测试**，见下）；⑥ 恢复与设备状态；⑦ 能力声明自洽 + `known_issues` 结构；
⑧ **缺厂商扩展时不得静默降级**。

**判据来源**：逐条对应《接口约定》或 conformance 用例的同口径判据（代码内注明 F1 / E2-v2 / E3 / R1-R5）。

#### 4.4.1 四家已内置 stub（2026-09-22 起）

stub 已通用化为 `_vendor_stub(ns_name, vendor_modules, mem_mode)`，**新增一家 = 在 `_STUBS` 里加一行**：

| 后端 | 设备命名空间 | 需一并对造的顶层模块 | 当前结果（本机、无设备） |
|---|---|---|---|
| `cambricon` | `torch.mlu` | `torch_mlu` | **38 通过 / 0 失败 / 0 跳过** |
| `kunlun` | `torch.cuda`（XPytorch） | — | 38 / 0 / **1 跳过** |
| `flagos` | `torch.flagos` | `torch_fl` | 31 / 0 / **2 跳过** |
| `ascend` | `torch.npu` | `torch_npu` | 28 / 0 / **2 跳过** |

> ⚠️ **沉痛教训（第 3 家接入的收口产出）**：本工具**原先只为 `cambricon` 一家内置 stub**，
> 于是 `--backend kunlun\|ascend\|flagos` **直接被拒** ⇒ 那三家**从未被自检过**，
> 结果在 910C 第一实例上长期存在两处缺陷（声明了 `device_state` 却无实现；
> `info()["supports"]` 手写了第二份键名清单、与 `_capabilities` 对不上）
> 直到把工具扩到四家才暴露。**"只有一家能跑的自检"= 其余各家的盲区。**
> 明细见 `BACKEND_SYMMETRY_AUDIT_20260922.md`。

#### 4.4.2 显式 SKIP 机制（stub 是语义空间的，别让它越界）

stub 无法模拟真实算子与厂商运行时（例如昇腾的有界同步走 `acl` 原语、探活走真实设备计算）。
凡 stub 覆盖不到的判据，一律**显式 SKIP 并写明原因**，汇总行区分
`X 通过 / Y 失败 / Z 跳过（stub 能力边界，非失败）` ——
**既不误报 FAIL，也不把"跳过"混进"通过"**。

#### 4.4.3 两条新增判据（都可防住"声明与实现不符"）

| 判据 | 防的是 |
|---|---|
| `device_state 可调用`（缺方法判 FAIL，**不让整轮崩掉**） | 可选能力"声明了却不实现"（ABC 拦不住；原先该调用会让整轮自检崩掉、连汇总都打不出） |
| `info()['supports'] 键集合 == 能力全集` | `info()` 手写第二份键名清单导致**键名漂移**（smoke 既有的"取值一致"判据覆盖不到：两边都取 `False`） |

#### 4.4.4 证据卫生（同样适用于所有验证脚本）

- **不得硬编码单一厂商的错误码或厂商专有文案**：需要错误码时从**实际异常**中提取，拿不到就如实写"无厂商码"。
  （曾把昇腾码 `507046` 写进错误闭环文案，在无厂商码的后端上直接产出**假证据**。）
- **注入用例必须自描述且期望参与判定**：把"这条注入期望什么"写进结果记录
  （如 `expectation` / `expect_matched`），不符即判该条失败 —— 不能只看"业务还在跑"就放过。

> ⚠️ **结论边界（必须写清，勿外推）**：本自查**只证明实现逻辑与契约形态**，
> **不能替代 conformance**，也**不能证明后端在真机上能用**。
> 厂商 API 真实形态（某函数是否存在/是否收参数）、厂商错误码是否透出、
> 集合通信后端名、以及 conformance 是否通过 —— **一律必须真机实测**。
> 脚本末行会重复这句边界，避免被脱离语境引用。

---

## 5. 第 4 步：跑通 conformance（13 例 + 6 例）

```bash
python3 runtime/conformance/runner.py --backend <vendor>                    # 设备上下文与多流 13 例
python3 runtime/conformance/runner.py --backend <vendor> --cases infer_cases # 推理 6 例
python3 runtime/smoke_runtime.py --backend <vendor>                          # 接入自检
```

**通过判据**：13/13 与 6/6 全绿，或未支持项有**如实 stub-skip 说明**。
13 例覆盖：事件 record/wait（含超时边界、未 record 查询）、错误翻译、恢复评估、
流内顺序、跨流显式依赖、结果可见性、显式传输、页锁定异步拷贝、在途保护、拓扑路径。

**这一步是接入完成的判定线**：两家实例分别是 13+6 与 13+6，且**逐用例状态完全一致**。

---

## 6. 第 5 步：两条腿验证（设备层的能力实证）

**刻意用最小形态**（`transformers` + `AdamW` + `torch.distributed`；推理用原生 vLLM），
目的是把设备因素单独暴露出来——框架越厚，设备问题越容易被框架行为掩盖。
反例很实在：P800 的 KL3 挂死只有在纯 torch 形态下才定位到函数级。

### 6.1 训练腿（2 卡，50 步 / batch 4 / seq 128）

```bash
CUDA_VISIBLE_DEVICES=6,7 DC_BACKEND=<vendor> DC_ROOT=<prototype> \
DC_MODEL=<...>/snapshots/<hash> DC_OUT_DIR=<out> MAX_STEPS=50 BATCH=4 SEQ=128 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py
```

**判据**：两 rank 均 `TRAIN_LEG_PASS 6/6`（设备绑定 / 模型加载 / 反向 / 集合通信同步 /
三类通信对照 / 无死锁），且 loss 正常下降。
参考锚点：910C `15.4497 → 11.15`、2117 tok/s；P800 `15.4488 → 11.1481`、3482 tok/s。

**50 步不是随便定的**：挂死类缺陷往往在第 n 次通信才出现。
P800 实测挂死点游走在第 3–4、41–50、101–120 次通信之间——50 步是能稳定覆盖该区间的档位。

### 6.2 推理腿（单卡前向 + vLLM 服务化）

**前向**判据：向量维度、首条范数、语义区分度、吞吐与 p50/p90 时延、真实异常注入的分级与处置。
参考锚点：910C 维度 1024 / 区分度 0.638；P800 维度 **1024** / 区分度 **0.6392**（两实例 `detail` 逐字相同）。

**服务化**判据（`--runner pooling --convert embed`，与两实例同口径）：
服务就绪、维度/范数、语义区分度、吞吐与 p50、**超长输入 → HTTP 400 → 统一分级 L2_PARAM/raise 且业务继续**、
**服务与设备上下文同卡共存**（同卡跨流计算正确）。
参考锚点：910C 区分度 0.4123 / 108 句/s；P800 0.4102 / 30.70 句/s。

> ⚠️ 该版本 vLLM **没有 `--task` 参数**，embedding 服务必须用 `--runner pooling --convert embed`
> （按 `--task embed` 起会报 `vllm: error: unrecognized arguments: --task embed`）。
> ⚠️ `vllm serve` 被杀主进程后 **EngineCore 子进程会残留并持续占卡**（P800 实测卡 6 仍被占 73850 MiB / 96 GiB），
> 下次启动报 `Free memory ... less than desired GPU memory utilization` ⇒ 停机逻辑必须连子进程一起 `kill -9`。

### 6.3 多流 Stream 基线（16 项）

多流是本方向的另一半职责。接入新后端时，应按接口约定给出的**多流验收基线 16 项**逐项比对
（P800 目前只验了"跨流 Event 依赖"等少数项，16 项逐项比对列入 10 月计划）。

---

## 7. 第 6 步：错误闭环（设备侧的最后一块）

```bash
python3 runtime/proto/proto_error_recovery_loop.py --backend <vendor>
```

**判据**：四类注入（参数类 / 资源类 / 执行类 / 致命类）各自得到正确的分级与处置，
且**业务能继续**。参考锚点：910C 推理腿 5 闭环 / 0 失败；P800 两设置各 5 闭环 / 0 失败。

> **调用纪律（实测结论）**：调 `translate_error` 时必须传**完整的原始异常或服务错误消息，不得截断**。
> 实测：vLLM 超长输入返回的完整错误体 → 正确分级 L2_PARAM/raise；
> 只传 "HTTP 400" 或被截断 → 退化为 L3_EXECUTION/replay（对参数错误做无意义重放）。

---

## 8. 验收清单（接入完成的判定）

复制这张表逐项打勾，全绿即接入完成。

| # | 项 | 判据 | 910C | P800 |
|---|---|---|---|---|
| 1 | 环境打通 | 连得上 / 有权限 / 有可写目录 / 会挑空闲卡 | ✅ | ✅ |
| 2 | 厂商栈判别 | 明确落在 A/B/C/D 哪条路径，并记录证据 | ✅ A（`npu`） | ✅ B（`cuda`） |
| 3 | 镜像就绪 | 5 条判据全过（含**依赖链完整**自检） | ✅ | ✅ |
| 4 | backend 落地 | 13 抽象 + `build()` + `supports()` 如实声明 | ✅ | ✅ |
| 5 | conformance 13 例 | 全绿或如实 stub-skip | ✅ 13/13 | ✅ 13/13 |
| 6 | conformance 推理 6 例 | 全绿 | ✅ 6/6 | ✅ 6/6 |
| 7 | smoke 自检 | 全通过 / 0 失败 | ✅ 37/37 | ✅ 42/0 |
| 8 | 训练腿 | 两 rank `TRAIN_LEG_PASS 6/6` | ✅ | ✅ |
| 9 | 推理腿（前向） | 维度 / 范数 / 区分度 / 时延 / 异常分级 | ✅ 10/10 | ✅ 13/13 |
| 10 | 推理腿（服务化） | 含超长输入防御与同卡共存 | ✅ 10/10 | ✅ 10/10 |
| 11 | 错误闭环 | 四类注入分级处置正确、业务继续 | ✅ 5 闭环 | ✅ 两设置各 5 闭环 |
| 12 | 已知问题如实声明 | `known_issues()` 结构化 + 前后端告警 | ✅ | ✅ |
| 13 | 证据归档 | 原始日志/JSON 入版本库（**注意 `*.log` 例外规则**），不是"声称已归档" | ✅ | ✅ |

**第 13 条的教训**：仓库根 `.gitignore` 有 `*.log` 通用规则，P800 曾因此导致 8 份证据日志
**历次提交都写"已归档"实则从未入库**。修法是在 `probes/.gitignore` 写 `!*.log`；
**写"已入库"前必须用 `git ls-files <path>` 实测确认**，不能凭 `git add` 没报错就认定。

---

## 9. 跨芯片坑清单（接入时按此逐条排查）

| # | 坑 | 现象 / 触发条件 | 应对 |
|---|---|---|---|
| 1 | **依赖链不完整伪装成设备问题** | `Failed to infer device type`（看着像设备问题） | 先跑 §3 的自检命令；缺 `triton` 就补 `flagtree`，缺子模块就设 `PYTHONPATH` |
| 2 | **共享机上误用被占用的卡** | 设备侧报错，看起来像通信库缺陷 | 用卡前 `xpu-smi` 挑空闲卡；出现异常先换卡复测再下结论（P800 曾因此撤销一个"缺陷"结论） |
| 3 | **同一 FlagCX 在不同芯片注册的后端名不同** | 910C 是 `flagos`；P800 是 `flagcx` 且需显式 `import flagcx` | 不要照搬；探测可用后端再写进配置 |
| 4 | **`timeout` 杀不掉挂死进程** | 挂死点持 GIL 自旋，SIGTERM 被推迟（实测存活 73 分钟） | 必须 `kill -9` 按 PID 强杀，再 `xpu-smi` 复查卡释放 |
| 5 | **同一插件跨芯片可用性相反** | `vllm-plugin-FL` 在昆仑芯**必需**、在昇腾**必须禁用** | 逐个实测，不类推 |
| 6 | **HF 缓存路径层级** | 传缓存根目录报 `Unrecognized model ... Should have a model_type key` | 必须给到 `snapshots/<hash>`；注意有的脚本自带 `resolve_model()`、有的没有 |
| 7 | **镜像大小有两套口径** | `docker images`（磁盘占用，含共享层）vs `image inspect Size`（镜像层之和），差近 3 倍 | 引用时必须注明口径，否则被误读成版本差异 |
| 8 | **多环境对照不能只看总数** | 只看"N/N 通过"会漏掉语义差异 | 按**逐用例状态**与**逐条 `detail` 字符串**比对；共享机上的**单次吞吐不可直接对比**，须交替复测 |
| 9 | **vLLM 停机残留占卡** | EngineCore 子进程残留（实测 73850 MiB / 96 GiB） | 停机连子进程一起 `kill -9` |

### 已知厂商缺陷的结构化声明（模板）

接入过程中遇到厂商缺陷时，按这个结构写进 `known_issues()`，让下游"读到后端即知坑"：

```python
{
  "id": "KL3_EVENT_SPIN",
  "summary": "KL3 事件同步在集合通信场景下概率性永久自旋",
  "trigger": "环境变量 <X>=1 且存在设备侧集合通信（两要素，缺一不挂）",
  "repro_rate": "18 次运行 16 次（≈89%）；挂死步数游走第 3–120 次通信",
  "irrelevant": "数据量 / 张量形状 / reduce op / 用卡对 / 同步间隔 均无关（逐一单变量排除）",
  "layer": "厂商运行时/驱动层（算子层与编译层已硬证据排除）",
  "evidence": "3 处自旋帧均在厂商 libxpucuda.so，偏移 +0x94080",
  "workaround": "不设置该环境变量；我方验证路径源码中 flag_gems 0 处引用 ⇒ 不依赖它",
  "workaround_cost": "该变量是 FlagGems 对应后端的官方推荐变量 ⇒ 关闭是否损失设备异常上报须上游确认；本方向不擅自改锁定口径",
  "report_to": "芯片厂商（主）+ 相关中间件（抄送）",
  "affects": "多卡集合通信场景；单进程设备上下文路径不受影响（错误闭环两设置逐字节一致即证据）"
}
```

**厂商问题上报模板（可直接套用）**：

```
1. 现象：<一句话>；复现率 <n/m>
2. 最小复现：<脚本 + 命令 + 约 <n> 秒出结果>
3. 判别条件：<需要哪些条件，哪些因素已排除>
4. 函数级定位：<栈帧 + 库名 + 偏移>
5. 影响面：<哪些场景受影响，哪些不受影响>
6. 已做的排除：<算子层 / 编译层 / 镜像变体 / 容器参数>
7. 我方规避方式与代价：<...>
```

> 第 6 项很关键：P800 的 KL3 缺陷最终在**两个不同镜像上一致重现**
> （现用变体 16/18、官方 `-base` 3/3），挂死现场特征一致（进程 `Rsl` 自旋、`utime` 累积至约 9700、
> 卡 100% 利用率而显存仅 366 MiB）。**这条排除了"是不是你们镜像的问题"**，是上报时最有力的一句。

---

## 10. 接入之后：本方向交付什么、不交付什么

| 做 | 不做 |
|---|---|
| 原型接入（backend + conformance + 两条腿 + 错误闭环） | 生产级性能调优（归精度/调优方向） |
| 暴露问题 → **五域内的先简单修复** | 算子实现、通信库优化、显存池调优、调度策略 |
| 已知问题如实声明（`known_issues`）+ 厂商上报 | 擅自改锁定镜像口径或公共资产 |
| 把原型 release 给运行时层其他子方向做验证与迭代 | 规模化验收（Megatron-LM-FL / vllm-plugin-FL 归框架适配方向） |

**问题路由**：接入过程中暴露的问题按"设备因素归设备"分流——
属本层五域的先修；算子/通信/显存/调度/性能一律对外提交。
