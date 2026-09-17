# 昆仑芯 P800 设备上下文 · 基线实测报告（2026-09-14）

> 目的：容器已起，按**我方五域**逐项取基线，识别缺失项并按适配方案 §1.3 判定归属。
> 复现脚本与原始结果：[`dev/device-context/P800/probes/`](../probes/)
> 容器：`hliu553-device-context-p800`（镜像 `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`）

---

## 0. 结论速览

| 域 | 状态 | 说明 |
|---|---|---|
| 设备抽象 | ✅ **可用** | 8 卡可见、显存查询准确、set/current device 正常 |
| 多流 Stream / Event | ✅ **基本可用** | 创建 / 上下文 / 跨流 Event 依赖 / record_stream **全部正确**；**流优先级子项 ❌ 上游缺陷** |
| 错误码翻译 | ⚠️ **可行但有硬约束** | 异常类型与消息模板正确，但**厂商错误码在 Python 层拿不到**（见 §3.2） |
| 状态恢复 | ⚠️ **未取到证据** | 本次仅确认 `empty_cache` 存在；设备级重置原语**仍待测** |
| 插件机制 | ✅ **已验证** | 镜像已含 `flagtree 0.6.1+xpu3.6` 与 `flag_gems`，算子级实测通过 |

**净结论**：**五域中 2 域完全可用、2 域可用但受上游约束、1 域待测**；
本次识别出 **1 个上游缺陷 + 1 条上游约束 + 1 条设计依据**（§3）。

---

## 1. 调用契约（必须先固化，否则后续全部失败）

| 项 | 取值 |
|---|---|
| 环境激活 | `source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda` |
| **⚠️ 坑：默认 `python3`** | `/root/miniconda/bin/python3` = **Python 3.13.11（conda base）** —— **不是**目标环境。目标环境为 **Python 3.10.18** |
| **必需环境变量** | `export XPU_EVENT_KL3_ENABLE=1`（官方手册明确要求） |
| **设备 API** | **走 `torch.cuda`，不是 `torch.xpu`** |
| 关键包版本 | `torch 2.9.0+cu129`、`flagtree 0.6.1+xpu3.6`、`flag_gems 5.3.4.post1.dev12`、`triton 3.6.0`、`transformers 4.57.1`、`flagcx 0.10.0`（editable `/env/FlagCX/plugin/torch`） |
| **torch 编译标志（理论级证据）** | `USE_CUDA=ON, USE_CUDNN=ON, USE_NCCL=1, USE_XCCL=OFF, `**`USE_XPU=OFF`**`（`torch.__config__.show()` 原文） |

### 1.1 「XPU 走 cuda 命名空间」的三条独立证据

1. `torch.xpu.is_available()` → **False**，且报 `AssertionError: Torch not compiled with XPU enabled`；
2. `torch.cuda.is_available()` → **True**，`device_count = 8`，`torch.cuda.get_device_name(0) = "GPU"`；
3. **官方 FlagTree xpu3.6 单测的 `conftest.py` 里 `--device` 默认值就是 `'cuda'`**：
   ```python
   parser.addoption("--device", action="store", default='cuda')
   ```
   （来源：`FlagTree/third_party/xpu/python/test/unit/conftest.py`）

4. **编译标志** `USE_XPU=OFF`（`torch.__config__.show()`）—— 这是**理论级证据**：
   该 torch 根本没有编译 XPU 支持，`torch.xpu` 只是 Python 侧属性存在、功能为空。

> ⚠️ **一处既有判断偏差**：xliu969 的 P800 实测文档记有「torch.xpu 亦存在 → 双通道」，
> 但该结论仅来自 `hasattr(torch,'xpu')`（其探针未测 `is_available()`）。
> 实测 + 编译标志均表明 **只有 CUDA 单通道**，建议对齐时提示更正。

进程启动时会打印：

```
XCCL /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir/xccl//so/libbkcl.so loaded
SYMBOL_REWRITE torch success
```

> 机制解释：该栈用 **XPytorch（`/env/xpytorch-cp310-torch290-ubuntu2004-x64.run`）+ `torch_xray` 符号重写**，
> 把 CUDA 命名空间下的调用重定向到 P800；因此 `torch.xpu` 无实现、`torch.cuda` 才是入口。
> **对我方接入的直接含义**：`kunlun` 后端应以 `torch.cuda` API 实现设备/流/事件映射，
> 不能照搬 910C（`torch_npu`）或 FlagOS（`torch_fl`）的命名空间假设。

---

## 2. 五域基线实测结果

### 域 1 · 设备抽象

| 项 | 结果 |
|---|---|
| `device_count` | **8** |
| `get_device_name(0)` | `"GPU"`（**注意：不是 `P800`**，若按型号字符串分支需额外处理） |
| `current_device` | 0 |
| `mem_get_info(0)` | free **98272 MiB** / total **98304 MiB**（= 96 GiB） |
| `memory_allocated(0)` | 0 |
| `mem_get_info` 原始字节 | `(103045660672, 103079215104)` 一致 |
| **设备可见性变量** | **`CUDA_VISIBLE_DEVICES`** —— 实测 `=2` → `device_count()=1`；`=2,5` → `2`（910C 的 `ASCEND_RT_VISIBLE_DEVICES` 对应物就是它，**不是** XPU 侧变量） |

### 域 2 · 多流 Stream / Event

| 子项 | 结果 | 判定 |
|---|---|---|
| `create_stream` | `torch.cuda.Stream()` → `Stream` | ✅ |
| `current_stream` | `Stream` | ✅ |
| `stream_context`（`with torch.cuda.stream(s)`） | 2048² 加法 = 8388608.0（正确） | ✅ |
| **跨流 Event 依赖**（`s2.wait_event(e)`） | 期望 45056.0，实得 **45056.0** | ✅ **语义正确** |
| `record_stream` | OK | ✅ |
| Event 计时（`enable_timing=True`） | 4096² matmul = **15.859 ms** | ✅ |
| **`Stream.priority_range()`** | ❌ `RuntimeError: greatest_priority <= -1 INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDAStream.h":188` | ❌ **上游缺陷**（§3.1） |

### 域 4 · 错误（异常类型与消息）

| 用例 | 异常类型 | 消息要点 |
|---|---|---|
| 显存不足 | `torch.OutOfMemoryError` | `Tried to allocate 14901.16 GiB. GPU 0 has a total capacity of 96.00 GiB of which 95.97 GiB is free` |
| 设备序号越界 | `torch.AcceleratorError` | `CUDA error: invalid device ordinal` |
| 流优先级 | `RuntimeError` | `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188` |

### 域 5 · 状态恢复与分布式

| 项 | 结果 |
|---|---|
| `torch.distributed.is_available()` | True |
| **可用后端列表** | `flagcx, gloo, kccl, mpi, nccl, ucc, undefined, xccl` |
| 备注 | **`flagcx` 与 `xccl` / `kccl` 均已注册**；`libbkcl.so` 随进程加载；字面 `bkcl` 未出现在 backend_list |
| `torch.cuda.empty_cache` | 存在且可调用 |
| 设备级重置原语 | **未取到证据，仍待测** |

---

## 3. 缺失项清单（含归属判定）

### 3.1 【上游缺陷】Stream 优先级范围非法导致 PyTorch INTERNAL ASSERT

| 项 | 内容 |
|---|---|
| **现象** | 调用 `torch.cuda.Stream.priority_range()` 直接抛 `RuntimeError`，且是 PyTorch C++ 层 `INTERNAL ASSERT FAILED` |
| **最小复现** | `python3 -c "import torch; print(torch.cuda.Stream.priority_range())"`（需先激活环境 + `XPU_EVENT_KL3_ENABLE=1`） |
| **错误原文** | `greatest_priority <= -1 INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDAStream.h":188, please report a bug to PyTorch. Unexpected CUDA stream priority range` |
| **根因判断** | PyTorch 期望「最大优先级 ≤ -1」，而该栈后端上报了非法区间 → 属**后端上报值不合规**，非我方实现问题 |
| **稳定性** | **单进程隔离复现，rc=1，稳定** |
| **影响** | 我方 `RuntimeBackend.stream_priority_range()`（可选能力）在昆仑芯上**不可用** |
| **归属** | **上游（XPytorch / 昆仑芯 torch 后端的 stream priority 实现）** → 对外提交 |
| **我方动作** | ① `kunlun` 后端 `supports()` **如实声明不支持流优先级**，conformance 该项**如实跳过**；② 封装层**禁止透传**该调用，避免触发 C++ 断言；③ 出对外提交单（现象 + 最小复现 + 原文 + 定位证据） |

### 3.2 【上游约束】厂商错误码在 Python 层不可得

| 项 | 内容 |
|---|---|
| **观察** | 三个用例的 Python 异常消息中**均未出现昆仑芯错误码**；最多只有 `CUDA error: invalid device ordinal` 这类转述 |
| **唯一一次见到号码** | 在第一轮**同进程混合**用例的进程退出钩子里：<br>`terminate called after throwing an instance of 'pybind11::value_error'`<br>`what(): [RUNTIME ERROR]: error code= 101, invalid device ordinal;`<br>—— 来自 `torch_xmlir._prepare_to_exit`，**非稳定可得** |
| **影响** | 910C 上我们靠**错误码映射表**（108 条）做 L1–L4 分级；昆仑芯侧**没有码可映射** |
| **归属** | **上游（错误上报层未把厂商码透出到 Python 异常）** → 对外提交诉求：请求在异常中携带厂商错误码 |
| **我方动作（不阻塞）** | 昆仑芯错误映射表**先以「异常类型 + 消息模板」为键**建立，暂不依赖数字码；<br>`translate_error` 如实标注 `is_grade_confident=False` 的场景 |

### 3.3 【设计依据】同进程内一处错误会污染后续调用

| 项 | 内容 |
|---|---|
| **观察** | 同一进程内先 `set_device(99)` 触发越界错误后，**紧接着的 OOM 用例报的是同一个 `invalid device ordinal`**，而非 OOM |
| **隔离后** | 每个用例独立进程运行 → OOM 正常报 `torch.OutOfMemoryError`，越界正常报 `AcceleratorError` |
| **⚠️ 更正记录** | 我最初把该现象读作「**设备级**错误粘滞」——**不准确**。隔离实验证明它**不跨进程**，只在**同进程内**发生 |
| **归属** | 待进一步定位（可能属运行时 context 失效语义，也可能与本栈映射层实现有关）→ **先记为观察项，不下结论** |
| **我方动作** | 作为「**错误隔离分层**」设计依据：错误发生后同进程的后续调用结果**不可信**，<br>恢复动作应优先考虑**重建进程/上下文**而非原地继续 |

---

## 4. 与 910C 的对照（跨芯片原语差异）

| 维度 | 910C（昇腾） | P800（昆仑芯） | 对我方后端设计的影响 |
|---|---|---|---|
| 设备命名空间 | `torch_npu` / FlagOS `torch_fl` | **`torch.cuda`**（XPytorch + 符号重写） | 不能假定命名空间，须由后端自行绑定 |
| 设备可见性 | `ASCEND_RT_VISIBLE_DEVICES` | 待确认（`XPU_EVENT_KL3_ENABLE` 为**事件开关**，非可见性） | 待补测 |
| 流优先级 | FlagOS 后端无 | **有接口但上游缺陷** | 两个平台该项都属「如实声明不支持」 |
| 跨流 Event 依赖 | 支持 | **支持（实测通过）** | 可对齐 |
| 有界同步 | pyACL 有 / FlagOS 无 | **未取到证据** | 待测 |
| 集合通信后端 | HCCL | **`xccl` / `kccl` / `flagcx` 均注册，`libbkcl.so`** | 分布式验证可复用 |
| 错误码 | 108 条映射表 | **Python 层不可得**（§3.2） | 映射表策略需按平台分支 |
| 带卡容器并发上限 | 3（超限 `acl.init()`=500000） | **未测** | 待测 |

---

## 5. 结论与下一步

**已证实**：容器可用、8 卡可见、FlagGems 算子级正确（`add` max diff = 0.0）、多流语义与跨流依赖正确。

**下一步（按最小变更、单变量原则）**

| # | 动作 | 目的 |
|---|---|---|
| 1 | 补测**设备级重置原语**（是否有 `reset_device` / context 重建） | 填 `recover_device` 的 real 模式证据 |
| 2 | 补测**有界同步**（`Stream.synchronize()` 是否支持超时） | 对齐 910C 的三态语义 |
| 3 | ~~补测**设备可见性环境变量**~~ | ✅ **已完成**：即 `CUDA_VISIBLE_DEVICES`（见 §2 域1） |
| 4 | 补测**带卡容器并发上限** | 若存在，提总组入约束清单 |
| 5 | 建立**昆仑芯错误映射表 v0**（以异常类型 + 消息模板为键） | 填 `translate_error` |
| 6 | 出 **2 张对外提交单**（§3.1 流优先级缺陷、§3.2 错误码不透出） | 非我方项，按渠道提交 |
| 7 | 实现 `kunlun` backend 并跑 conformance 13 例 | 接入完备度度量 |
