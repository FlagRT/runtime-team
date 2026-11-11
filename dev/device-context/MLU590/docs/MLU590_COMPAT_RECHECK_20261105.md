# 寒武纪 MLU590 · 兼容性对照表逐格复核（2026-11-05 真机）

> 机器：`Mlu-1` = `tza-0a06-ai01-em9`（10.1.1.21，验收机）· `Mlu-2` = `tza-0a06-ai02-em9`（10.1.1.22）
> 时间：**2026-11-05 15:55:14 – 16:07:07 CST** ｜ 负责人：Kistich（hliu553）
> 容器：`dc-mlu590-hliu553`（本方向自有；锁定镜像 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`）
> 证据目录：`MLU590/probes/mlu_compat_20261105_out/`（原始 `.out` + 6 份可复跑脚本 + 探针 `.py`）
> 判据口径：`docs/CHIP_COMPAT_MATRIX_20261104.md` 的三级采集（L0 宿主 / L1 容器 / L2 行为）与剩余待核项 P1。

---

## 0 结论先行

| 格 | v0.10 判定 | **v0.11 判定** | 本轮依据（自包含） |
|---|---|---|---|
| 操作系统 | 🟡 部分验证（容器 OS「由包名推断」） | **✅ 已验证** | 宿主 `grep PRETTY_NAME /etc/os-release` → `Ubuntu 20.04.6 LTS`、`uname -r` → `5.15.0-139-generic`；容器 `/etc/os-release` → **`Ubuntu 22.04.5 LTS`**（原推断值被实测坐实） |
| 驱动 | ✅（`v6.2.29`） | **✅（内容更新，判定不变）** | `Mlu-1` 运行态 **`6.5.52`**（cnmon / `/sys/module/cambricon_drv/version` / `modinfo` 三处一致）；`Mlu-2` 仍 **`6.2.29`**；两台 `dpkg` 元数据都写 `6.2.29-7.ubuntu20.04`。固件均 `v1.5.0` |
| 开发包（SDK） | ✅ | ✅ 不变 | 容器 `cat /usr/local/neuware/version.txt` → `Neuware Version 4.4.3`；`TOOLKIT=cntoolkit_4.4.3-1.ubuntu22.04_amd64.deb`；镜像 `docker inspect` ID `sha256:e887e0ca4333a27185817fb8fef6ce95d5d9220c5a9725641187d602cabeea96` 与 `dev/images/cambricon-mlu-runtime/v1/image.baseline.yaml` 的 `image.id` 逐字一致 |
| 框架（PyTorch） | ✅ | ✅ 不变 | 容器 `torch 2.7.1+cpu` · `torch_mlu 1.29.2+torch2.7.1` · `torch_mlu_ops 1.8.0+torch2.7.1` · py `3.10.20`，与基线逐字一致 |
| 算子库（实际使用 · 厂商） | ✅ | ✅ **证据升级** | 注册层实测：`aten::mm`/`add.Tensor`/`mul.Tensor`/`silu`/`relu`/`_softmax`/`native_layer_norm`/`triu` 的 `PrivateUse1` 全部 `registered at /torch_mlu/torch_mlu/csrc/aten/generated/RegisterMLU.cpp:43487 [kernel]`；厂商路径进程内 `'flag_gems' in sys.modules = False` |
| 算子库（替代 · FlagGems 系） | ⚪ 待核验（唯一 ⚪ 格） | **🟡 部分验证（结论：当前不可替换）** | 本轮做了结论性验证：`enable()` 成功且 **8/9 算子的 dispatch 被接管**，但 **4/9 算子执行失败**、**真实前向失败** ⇒ 不是「没测」，而是测出「不可用」，详见 §4 |

**计数变化**（24 格不变）：✅ 17 → **18**；🟡 6 → **6（不变）**；⚪ 1 → **0**；⛔ 0。⭐ **两处变化互相抵消**（操作系统 🟡→✅ 减 1、替代库 ⚪→🟡 加 1）⇒ 合计仍 24 格；首轮误写成 🟡7，已由主表逐格计数复核更正。

---

## 1 采集纪律与连通性（实测，不是推断）

| 项 | 读数 |
|---|---|
| 连通性 | 15:55:14 `ssh Mlu-1` / `ssh Mlu-2` 均成功；`nc -z` 两台 `:22` 全 OPEN |
| 执行方式 | 全部走「本地写脚本 → `ssh <host> 'bash -s' < 脚本`」，脚本首行打印 `SELF_CHECK`，本地 `md5 -q` 留证 ⇒ 避免多层 ssh 嵌套引号被吞、避免跑到旧版 |
| 带卡名额 | 采集前 `cnmon`：**8 卡全 `0 MiB / 98304 MiB`、`No running processes found`**；只取卡 2（`MLU_VISIBLE_DEVICES=2`）；采集后仍为 8 卡全空闲 |
| 容器 | 只启动本方向自有容器 `dc-mlu590-hliu553`（原 `Exited (137) 2 days ago`）；`docker inspect` 显示 `NetworkMode=host`、8 个 `/dev/cambricon_dev*` + `ctl/gdr/ipcm*`、`/srv/hliu553→/work`、`/srv/data/hf_cache→/hf_cache:ro`。**未动任何他人容器** |
| 环境误判（记录备查） | `docker run --rm` 首次报 `failed to create endpoint ... adding interface veth... to bridge docker0 failed: Device does not exist`（宿主 docker0 网桥缺失）⇒ L1 改 `--network none` 后正常。**这是宿主网络配置，不是设备故障、也不是算子库问题** |

**L1 的两个入口条件**（都不算算子库缺陷，先排除再判据）：

```
docker run --rm --network none ... /flagos/bin/python3 l1_probe.py
  → import triton / import flag_gems 均失败：ModuleNotFoundError: No module named 'triton'
    （原因：直调 python 未走 login shell，/opt/triton 不在 PYTHONPATH；
     换 bash -lc 后 triton 3.2.0 @ /opt/triton/triton/__init__.py 正常）
docker run --rm --network none ...（不挂设备）
  → import flag_gems：RuntimeError: No MLU devices are available
    （FlagGems 在 import 期就要求设备可用 ⇒ 这格必须带卡才能判）
```

补齐 `PYTHONPATH=/opt/triton` 并带卡后，两个入口条件都消除，才进入 §4 的判据。

---

## 2 六格逐格证据（命令原文 → 读数）

| 层 | 命令（宿主 / 容器视角分开） | 本轮读数 | 与既有基线 |
|---|---|---|---|
| OS · 宿主 | `grep PRETTY_NAME /etc/os-release` · `uname -r` | `Ubuntu 20.04.6 LTS` / `5.15.0-139-generic`（Mlu-1、Mlu-2 同） | 与 09-22 归档一致 |
| OS · 容器 | 容器内 `grep -E '^(NAME|VERSION|PRETTY_NAME)=' /etc/os-release` | `Ubuntu 22.04.5 LTS (Jammy Jellyfish)` / `VERSION_ID=22.04` | **由「包名推断」升级为实测** |
| 驱动 · 运行态 | `cnmon` 首段 · `cat /sys/module/cambricon_drv/version` · `/sbin/modinfo cambricon_drv \| grep ^version` | Mlu-1：`CNMON v6.5.52 / Driver v6.5.52`、`6.5.52`、`version: 6.5.52`；固件 `v1.5.0`；Mlu-2：`CNMON v6.2.29 / Driver v6.2.29` | 见 §3（版本已分叉） |
| 驱动 · 包元数据 | `dpkg -l \| grep cambricon` · `zgrep cambricon /var/log/dpkg.log*` | 两台均 `cambricon-mlu-driver-dkms 6.2.29-7.ubuntu20.04`；日志最后一次驱动安装是 **2026-09-09 11:52** 的 `6.2.29-7` | 与运行态不同源，见 §3 |
| 开发包 | 容器 `cat /usr/local/neuware/version.txt` · `env \| grep -E 'TOOLKIT\|PKGS\|NEUWARE'` | `Neuware Version 4.4.3`；`NEUWARE_HOME=/usr/local/neuware`；`TOOLKIT=cntoolkit_4.4.3-1.ubuntu22.04_amd64.deb`；`PKGS=cnnl_2.1.829-… / cnnlextra_2.2.7-1 / mluops_1.8.1-1 / cncl_1.29.4-1 / cnclep_1.1.1-1`（均 `.ubuntu22.04`） | 与 `image.baseline.yaml` L72–L82 逐字一致 |
| 框架 | 容器 `pip list` 过滤 · `python -c 'import torch, torch_mlu'` | `torch 2.7.1+cpu` · `torch_mlu 1.29.2+torch2.7.1` · `torch_mlu_ops 1.8.0+torch2.7.1` · `triton 3.2.0+mlu1.7.2` · `flag_gems 5.4.0rc2.post3` · py `3.10.20` | 与 `image.baseline.yaml` L25–L58 逐字一致 |
| 算子库 · 实际使用 | `torch._C._dispatch_dump_table(op)` 取 `PrivateUse1` 行 | 全部指向 `RegisterMLU.cpp:43487`（厂商） | 由「9 段 PASS 间接推断」升级为**注册层直接证据** |
| L2 行为 | 带卡 `torch.mlu.device_count()` + 9 算子 + 真实前向 | `device_count=1`；9 个算子厂商路径全 OK、无 NaN/Inf；Qwen3-Embedding-0.6B 前向 `last_hidden_state[0,0]` head `1.726331, -21.718626, 0.018317, -5.072930, 3.850179, 4.286869, -2.279402, 18.445930`、sum `-50.397224` | 证明**现行驱动 6.5.52 + 锁定镜像 4.4.3 档**可协同工作 |

---

## 3 驱动分叉专项（本轮最容易被误读的一处）

| 读数来源 | `Mlu-1`（验收机） | `Mlu-2` | 是否可比 |
|---|---|---|---|
| `cnmon` 首段（运行态） | `CNMON v6.5.52 / Driver v6.5.52` | `CNMON v6.2.29 / Driver v6.2.29` | ❌ **不同机器**，是分叉不是冲突 |
| `/sys/module/cambricon_drv/version` | `6.5.52` | `6.2.29`（09-22 归档同值） | 同上 |
| `modinfo cambricon_drv` | `version: 6.5.52`，文件 `/lib/modules/5.15.0-139-generic/updates/dkms/cambricon-drv.ko` | — | 与 cnmon 同源 |
| `dpkg -l` 包元数据 | `6.2.29-7.ubuntu20.04` | `6.2.29-7.ubuntu20.04` | ⚠️ **包管理器元数据 ≠ 运行态内核模块**，两个口径互不否定 |
| 文件时间 | `/usr/bin/cnmon` `2026-10-28 11:18:54`；`/sys/module/cambricon_drv/version` `2026-10-28 17:09:29` | `/usr/bin/cnmon` `6月25 10:01` | 指向 Mlu-1 在 **10-28** 单独换过驱动；`dpkg.log` 无 10-28 安装记录 |

**可比性 / 旧值处理（必须一起读）**：

1. `6.2.29` 在 09-22 环境报告、10-08 复跑、10-14 preflight 里都是**当时的实测值**，且 `Mlu-2` 现在仍是这个值 ⇒ **历史归档不改写成 6.5.52**，只在「现行验收机口径」登记新值。
2. 归档里「宿主驱动 `6.2.29`（6.2.x 线）⇒ 只能选 `neuware4.4.3` 档（`4.7.2` 档官方标注要求 `6.5.48`）」这条选档依据，**对 Mlu-1 的条件已变**（现行 `6.5.52` 已越过 `6.5.48` 门槛）。但本方向**没有在 4.7.2 档做过任何验证**，本轮也不做档位升级判定 ⇒ 只登记「选档依据已过期，需总组 / 镜像方向重新评估」，**锁定镜像与验收基线一律不变**。
3. `11-03` 的 9 段 `overall: PASS` 归档中**未记录当次 `cnmon` 读数**（本轮已扫 `MLU590/probes/*20261103*` 无命中）⇒ 按「能用事实就不要反推」，**不写「11-03 验收在 6.5.52 下取得」这个推断**。能写的是：本轮 11-05 在 `6.5.52` 下实测设备可见、9 算子与真实前向全通过（§2）。
4. 本轮**未重跑 11-03 的 9 段判据**——这是范围裁定，不等于「已证明无回归」。

---

## 4 FlagGems 那一格：结论性验证（⚪ → 🟡）

### 4.1 三层证据

| 层 | 判据 | 实测 |
|---|---|---|
| ① 我方路径是否启用 | `grep -rInE "flag_gems\|use_gems\|enable()\|USE_FLAGGEMS\|VLLM_FL_PREFER" MLU590/probes prototype/scripts` → **0 命中**；vendor 进程打印 `'flag_gems' in sys.modules` | `False`。训练腿 / 推理腿 / 多流 / conformance / 服务化消费方**全部走厂商原生算子路径** |
| ② 能设 + 有效果（不是「接受了参数」就算） | 带卡进程 `import flag_gems 5.4.0rc2.post3` → `flag_gems.enable()` → 逐算子 `torch._C._dispatch_dump_table` 前后对比 | `ENABLE_OK`；**8/9** 个算子的 `PrivateUse1` 从 `RegisterMLU.cpp:43487` 变为 `/dev/null:63`（FlagGems 动态注册）；唯一未变的是 `aten::matmul`（不在 FlagGems 覆盖内，仍厂商） |
| ③ 能跑通吗 | 两个**独立进程**、同 seed、同输入、同 `MLU_VISIBLE_DEVICES=2`、同 `PYTHONPATH=/opt/triton`，唯一变量＝是否 `enable()`；9 算子逐个 try/except + 真实前向 | 见 §4.2 / §4.3 |

### 4.2 逐算子对照（9 项，固定种子，fp32，64³ GEMM 同款 K-5 判据维度）

| 算子 | 厂商路径 | FlagGems 启用后 | 数值最大差（前 8 元素） | 判定 |
|---|---|---|---|---|
| `gemm64`（64×64 @ 64×64） | OK | OK | `0.000000` | ⚠️ **归属待证**，见下方判据强度说明 |
| `matmul`（32×64 @ 64×16） | OK | OK | `0.000000` | dispatch **未变** ⇒ 仍是厂商路径 |
| `add` | OK | **FAIL** | — | `KeyError: 'Keyword argument task_type was specified but unrecognised'` |
| `mul` | OK | **FAIL** | — | 同上 `task_type` |
| `silu` | OK | **FAIL** | — | 同上 `task_type` |
| `relu` | OK | OK | `0.000000` | dispatch 已变且跑通 |
| `softmax` | OK | OK | `0.000000`（sum `32.0` vs `32.000004`，差 `4e-6`） | dispatch 已变且跑通 |
| `layernorm` | OK | **FAIL** | — | `KeyError: 'Keyword argument BLOCK_ROW_SIZE was specified but unrecognised'` |
| `triu` | OK | OK | `0.000000` | dispatch 已变且跑通 |

**判据强度（不能过度解读的地方）**：

- 数值比对只取输出**前 8 个元素 + 全张量 sum + NaN/Inf 计数**，不是全张量逐元素比对 ⇒ 覆盖面如实写清，不宣称「逐元素一致」。
- `gemm64` 虽然 dispatch 已变、数值与厂商逐字一致且未触发 triton 报错，但**没有单独证明它确实执行了 FlagGems 的 GEMM 实现**（可能内部回落厂商）。按「能用事实就不要反推」标为**实现归属待证**，**不**把它当作「FlagGems GEMM 可替换」的证据；需算子方向确认。
- 4 个失败项的完整栈：`triton.testing.do_bench` → `triton/runtime/autotuner.py:kernel_call` → `triton/runtime/jit.py:run` → `KeyError`。`layernorm` 换成 `BLOCK_ROW_SIZE`。⇒ 这是 **FlagGems 传给 triton 的自动调优参数与容器内寒武纪 `triton 3.2.0+mlu1.7.2`（`/opt/triton`）接口不匹配**的实测现象与栈位置；按「外部依赖结论三层」纪律，**只登记现象，不归因到厂商/网络层，也不升格为厂商缺陷**，修复属算子 / 编译方向。

### 4.3 真实前向（同一模型、同一 snapshot）

| 路径 | 结果 |
|---|---|
| 厂商原生 | `AutoModel.from_pretrained('/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/')` → `.to('mlu:0')` → 前向 **OK**，head/sum 见 §2 |
| `flag_gems.enable()` 后 | **FAIL**：`KeyError: 'Keyword argument task_type was specified but unrecognised'`（同一 triton autotuner 栈） |

### 4.4 判定

**🟡 部分验证（结论：当前不可替换）**。理由：⚪ 的含义是「本组未做结论性验证」，本轮已做——能加载、**能设且有效果**（8/9 dispatch 被接管），但 **4/9 算子与整段真实前向失败** ⇒ 结论是「当前不可替换」，不是「未知」。覆盖面＝9 个常用算子 + 1 个真实前向；**未做全量算子库普查与性能对照**（属算子域，本方向不单独下结论）。历史 `lock.yaml` 里「组外实测口径：FlagGems _cambricon 后端显著偏慢、宜作 fallback」仍是**组外口径**，本轮没有性能判据，不并入本结论。

---

## 5 待核项收窄与边界

| 项 | 状态 |
|---|---|
| 对照表 §6.2 的 MLU590 P1 两项（容器 OS、FlagGems 结论性验证） | ✅ **本轮结案**（2026-11-05，证据见本报告与 `mlu_compat_20261105_out/`） |
| FlagGems ↔ triton-mlu autotuner 接口不匹配（`task_type` / `BLOCK_ROW_SIZE`） | 待算子 / 编译方向处理；本方向**声明面不变**（vendor 原生），不影响验收 |
| Mlu-1 驱动 `6.5.52` 下的镜像选档依据 | 待总组 / 镜像方向重评；本方向不换档、不改锁定镜像 |
| 11-03 归档缺当次 `cnmon` 读数 | **显式标注「未记录」**，本轮不反推、不补写 |
| 全量算子库普查 / 性能对照 / `gemm64` 实现归属 | 未验证（属算子域，本轮不做） |
| 本轮未重跑 11-03 的 9 段判据 | 范围裁定，**不等于已证明无回归** |
| 资产影响 | 未改任何默认配置、未改共享模型资产、未动他人容器；本轮**启动**了我方容器 `dc-mlu590-hliu553`，采集结束时为 `Up`（8 卡仍全空闲） |

---

## 6 第三方自查：可复跑脚本与期望读数

执行方式（在能连到 `Mlu-1` 的机器上）：

```
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step0.sh   # 宿主盘点 / cnmon / docker ps / 磁盘
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step1.sh   # 驱动溯源 + L1 容器版本（--network none，不挂设备）
ssh Mlu-2 'bash -s' < scripts/mlu_compat_step1.sh   # 对照机驱动读数
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step2.sh   # 注册层两路径 + 容器配置
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step3.sh   # L2 两进程注册层 + 数值基线
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step4.sh   # 9 算子独立判据 + 真实前向
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step5.sh   # 9 算子 dispatch 归属对照
```

| 脚本 | md5 | 期望读数（关键行） |
|---|---|---|
| `mlu_compat_step0.sh` | `7aabab71d8c636dfe5d91b9374acbc41` | `CNMON v6.5.52 Driver v6.5.52`（Mlu-1）；8 卡 `0 MiB` |
| `mlu_compat_step1.sh` | `bfdb5a702b890855ba4d81a71835ece6` | `Ubuntu 22.04.5 LTS`；`torch 2.7.1+cpu`；`Neuware Version 4.4.3`；镜像 ID `sha256:e887e0ca4333…` |
| `mlu_compat_step2.sh` | `8d165db027fd1f556137786eb40c57b6` | vendor 路径 `RegisterMLU.cpp:43487` + `flag_gems_in_sys_modules: False`；不挂卡时 `RuntimeError: No MLU devices are available` |
| `mlu_compat_step3.sh` | `55f4c6fb21b16ae70f297685991d57e0` | `ENABLE_OK`；`REG_CHANGED aten::mm True` / `aten::matmul False` |
| `mlu_compat_step4.sh` | `d5a16cfedf50af4477b98c485010f12c` | `GEMS_FAIL` 4 项；`FORWARD vendor ok=true` / `FORWARD gems ok=false` |
| `mlu_compat_step5.sh` | `15b1b6615b2fa0328dca316b1931df60` | 9 行 `DISPATCH_ROW`，8 行 `changed=True`、1 行 `changed=False` |

6 份脚本均通过 `bash -n`。原始输出按时间戳分文件、只追加不覆盖：
`l2_*_160008.out`、`op_*_160334.out`、`fwd_*_160334.out`、`dispatch_map_160707.out` 为**结论所依据**的那轮；
`*_155928.out` 与 `*_160233.out` 是两次失败尝试（缺 `PYTHONPATH` / `sys.argv` 参数传递），**保留作为反证**，不得删。

---

## 7 本轮对对照表的具体改动（v0.10 → v0.11）

1. 主表 MLU590「操作系统」格：容器 OS 由「由包名推断」改为实测 `Ubuntu 22.04.5 LTS`，🟡 → ✅。
2. 主表 MLU590「驱动」格：改为 `Mlu-1 6.5.52`（运行态，10-28 起）/ `Mlu-2 6.2.29` / `dpkg 6.2.29-7` 三值并列，固件 `v1.5.0`；✅ 不变但加形态说明。
3. 主表 MLU590「算子库（替代 · FlagGems 系）」格：⚪ → 🟡，写入「能接管但当前不可替换」的实测结论与失败根因栈。
4. §2.1 计数：✅ 18 / 🟡 6 / ⚪ 0 / ⛔ 0（🟡 两处变化抵消），并说明 ⚪ 归零不是「测出可用」而是「测出不可用」。
5. §3.3 由「本轮网络不可达，沿用历史记录」改写为本轮实测证据表 + 驱动分叉说明。
6. §6.2 的 MLU590 P1 两项标结案；新增三项「不属本方向」的待办（见 §5）。
7. §6.4 追加 11-05 MLU590 采集记录；§8 记 v0.11 修订；§5.3 登记 FlagGems ↔ triton-mlu autotuner 接口不匹配。
