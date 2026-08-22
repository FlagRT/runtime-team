# 推理插件接入（阶段 4）执行记录

> ⚠️ **已归档（2026-08-22）**：本文件为设备层路线变更（B→A，见 [../FlagOS设备层路线变更指南.md](../FlagOS设备层路线变更指南.md)）前的 **B 线/旧案产物**，不进入 A 线交付路径。
> 仍有效结论：昇腾 910c B 线推理链路闭环方法论（单卡→TP 验证流程）。
> 需 A 线重验：A 线昇腾链路走 torch_npu 官方配置，链路验证需重跑。

> 日期：2026-08-15（建档）｜ 执行人：xliu969 ｜ 状态：🟢 单卡 + 多卡 TP 闭环全部完成（TP=2 数值正确 2026-08-16；TP=4 见多卡 TP 验证文档）
> 前置：阶段 3 全部完成（16 卡集合通信 + torch_fl DDP 集成）｜ 调研：《阶段4启动评估-vllm-plugin-FL推理插件-20260815.md》
> 📌 本文为阶段 4 **唯一主文档**（后续会话续写）；交接快照见《推理插件接入-阶段4进度交接.md》；多卡 TP 验证见《推理插件接入-阶段4多卡TP验证-执行记录.md》

## 1. 结论摘要

**单卡最小闭环验证已完成大半，三个关键事实颠覆了交接时的预期：**

1. **vllm 0.20.2 在 CPU-only torch 容器可装可 import**（aarch64 wheel 齐全）。pip 元数据要求 `torch==2.11.0`，但实测降到 `torch==2.10.0+cpu` 后 import 正常——**torch 版本约束可绕过**（vllm 本体代码与 2.10 兼容）。
2. **torch_fl 自带 torch_npu 兼容 shim**：`import torch_fl` 后自动挂载 `torch.npu` 属性（11 个接口）并注册 `sys.modules["torch_npu"]` 别名（torch_fl/__init__.py:710-727）——插件里 25 处 torch_npu/torch.npu 引用**大部分在 import 层已被 shim 覆盖**。shim 缺 3 样：`empty_cache`、`NPUGraph`、`_inductor`；且 `is_available()` 恒 False（设计如此，防 transformers 误判）。
3. **真正的硬挂点 = triton 版本冲突**：vllm 0.20.2 依赖树带官方 triton 3.5/3.6，与 torch 2.10 内置 `_TritonLibrary`（torch/__init__.py:2805 注册 triton namespace）**二进制双注册冲突**（TORCH_LIBRARY RuntimeError）；aarch64 无 triton 3.4 wheel 可退；且官方 triton 无 ascend 后端（FlagGems ascend 算子需要 triton-ascend/FlagTree 系 3.2.x）。

**另一个关键情报**：flagos-ai 官方 ascend CI（.github/configs/ascend.yml → `flagscale/vllm-plugin-fl:ascend-vllm0.20.2-a3-ci`，Dockerfile 见 docker/ascend/Dockerfile）**基于 vllm-ascend 官方镜像（torch_npu 生态）构建，不含 torch_fl**——即路线 A（torch_fl 替换 torch_npu）是 flagos-ai 未做过的新工作，改造量高于交接时的 3~5 人日预估。改造分两块：环境编排（triton/torch 版本决策）+ 插件代码 25 处引用。

**挂点 C 定论（triton）**：官方 triton 3.5/3.6 在无 NVIDIA GPU 机器上**根本不可行**——triton driver 层要求恰好 1 个 active driver（triton/runtime/driver.py `_create_driver`），而 nvidia driver 的 `is_active()` 在无 GPU 时恒 False（driver.py:739-742）。libcuda stub + TRITON_LIBCUDA_PATH 只能绕过编译层的 assert，绕不过 driver 层。**FlagGems ascend 算子必须配昇腾系 triton**（FlagTree 3.2.x fork / triton-ascend，driver 基于 CANN）。

**路线 A 的 torch 版本约束放宽**：torch_fl 的 .so 跨 torch 小版本 ABI 兼容（2.10 编译的 _C.so 在 torch 2.11.0 下 import 正常）——之前以为 torch_fl 严格限 torch 2.10，实测放宽。且 torch 2.10/2.11 的 `_TritonLibrary`（torch/__init__.py:2805/2824）可 patch 为惰性 DEF_FRAGMENT 消除与 triton 3.5+ 的 namespace 冲突（venv 内环境调整，已验证）。

## 1b. 【重大进展 2026-08-15 下午】triton 决策已解 + 全链路 import 打通 🎉

**昇腾 triton 复用方案（零编译）**：vllm-ascend 官方镜像（本机已有）内含 **triton_ascend 3.2.1**（昇腾定制 triton，cp311 wheel，含 `triton/backends/ascend/` 完整后端 + bishengir 编译器工具）。docker create（不启动、不占 DrvMng 额度）+ docker cp 取出，拷入 py311 venv 即可用——**免 FlagTree 源码编译**（省 1~2 小时）。

**关键：python 版本须 3.11**（triton_ascend 是 cp311 wheel；vllm 0.20.2 的 cp38-abi3 wheel 跨版本可用；torch 2.10.0+cpu 有 cp311 wheel）。py311 来源：vllm-ascend 镜像的 /usr/local/python3.11.15（docker cp 拷入容器，同 Ubuntu 22.04 glibc 兼容）。

**py311 环境组装（venv311，全部成功）**：
1. torch 2.10.0+cpu（pytorch 官方源）✅
2. vllm 0.20.2（--no-deps，避开 torch==2.11.0 pin）+ 依赖循环补装 ✅
3. triton_ascend 3.2.1（镜像拷贝，triton.__version__=3.2.0，backends=['ascend'] 唯一）✅
4. flag_gems（源码 editable）+ vllm-plugin-FL（源码 editable）✅
5. **torch_fl 0.1.0（py311 源码编译，ACCELERATOR=ascend 走 CMake ascend 分支免 nvcc）** ✅
6. flagcx 0.13.0（py311 源码编译）✅
7. .pth 自动引导（flagos_boot.py：import torch_fl + shim 补丁 + factory 兜底）

**测试结果（venv311，插件零改动）**：
- `import vllm_fl` ✅ / `register()` → PlatformFL ✅ / device_type=npu / dist_backend=flagcx（FLAGCX_PATH 环境变量触发）✅
- **推理链路实测**：EngineCore 启动 ✅ → model 加载 ✅ → 权重加载（gate_up fused 挂点，见下）→ 推理执行曾到 Qwen3 attention（03:08 前）

**已修复的运行时挂点（torch_fl 源码 + venv 引导，插件零改动）**：
1. torch.device("npu"/"cann") 失败 → torch_fl `_remap` 加 npu/cann 别名（可 PR）
2. worker.py:396 `import torch_npu._inductor` → shim 注册 `_inductor`（指向 torch_fl.compile.inductor_backend）+ `empty_cache`/`NPUGraph`/`mem_get_info`/`max_memory_allocated` 补齐
3. triton_ascend PCH 需 `torch_npu.__file__`/`torch_npu._C`（raw stream）/`torch_npu.version` → shim 补齐（stream 走默认流 handle 0）
4. triton_ascend launcher 需 at_npu（taskqueue）→ `TRITON_ENABLE_TASKQUEUE=false`
5. torch_fl `set_device` 收 torch.device 报错 → index 转换
6. torch.device 局部类不可 pickle（spawn）→ 动态 __module__/__qualname__ 可导入
7. **torch function mode 是 thread-local**（vllm execute_model 在后台线程）→ torch_fl patch Thread.start 推 mode + 继承 ACL 设备；另加 factory wrap 双保险（boot 层 torch.empty 等 device 参数 remap）
8. **allocator 子线程 aclrtGetDevice 失败**（device=-1）→ ascend_memory.h get_device_index fallback 0（可 PR，需重编译 torch_fl）
9. **torch set_data 不允许 cpu↔privateuseone 跨设备**（vllm device_loading_context 触发）→ patch vllm utils.py（venv 内）用 param 替换语义
10. **flagos(PrivateUse1) 不参与 torch 默认 device/`with torch.device` 机制**（vllm model 实例化在 CPU）→ patch vllm base_loader（venv 内）model 实例化后显式 `.to(target_device)`

**当前挂点（下一轮接续）**：~~`split_with_sizes: backend not registered`~~ —— **已解决**（见下）。🎉🎉🎉 **单卡最小闭环跑通（2026-08-15 05:18）**：

```
Prompt: 'Hello, my name is', Generated: " Xiaoyu, and I'm a"
[probe] REASONING OK
exit=0（完整退出，收尾清理也通过）
```

**Qwen3-4B 在 NPU 单卡通过 vllm-plugin-FL + torch_fl 完成真实推理生成**——阶段 4 第 1 步（单卡最小闭环验证）完成。**全部适配在环境层**（torch_fl 源码增强 + flagos_boot + venv 内 vllm patch），**插件代码零改动**。挂点链完整清单与解法见 §2。

**小坑（已解决）**：
- torch_fl 编译：默认 ACCELERATOR=cuda 要 nvcc——必须 `ACCELERATOR=ascend`（CMakeLists.txt:37）
- .pth 多行只执行 import 行——引导逻辑须放独立模块（flagos_boot.py）
- vllm 缺依赖（cloudpickle/cbor2 等）——自动补装循环
- torch 2.10 的 mode pop API 是 `_pop_torch_function_stack`（非 _pop_on...）

**改造量评估（更新版）**：
- 环境层（新发现，**前置决策点**）：triton 必须换昇腾系（FlagTree 3.2.x 源码编译 或 triton-ascend）——官方 triton 出局；torch 版本可 2.10 或 2.11（torch_fl ABI 兼容）；vllm 0.20.2 与 torch 2.10/2.11 都兼容
- 代码层：platform/worker 层 3 类小挂点（is_available 语义、_inductor、empty_cache/NPUGraph）→ 可改 shim 或改插件；dispatch 算子层（npu_grouped_matmul 等 torch_npu 专用算子）→ FlagGems/FlagTree 回退（P0 算子清单未到，覆盖度待定）
- 通信层：flagcx 已有（阶段 3 资产），dist_backend_dict npu→hccl 待改

## 2. 执行环境（最终版）

- 容器 flagos-fl-dev-910c（host 网络 · CANN 9.0.0 · 无 torch_npu，与阶段 3 相同）
- **实验 venv：`/root/vllm-venv311`（Python 3.11.15，拷自 vllm-ascend 镜像 /usr/local/python3.11.15，与 tf-venv-integration 完全隔离）**
- venv311 内：torch 2.10.0+cpu + vllm 0.20.2 + **昇腾 triton（triton_ascend 3.2.1，拷自 vllm-ascend 镜像，零编译）** + flag_gems（源码 editable）+ vllm-plugin-FL（源码 editable）+ **torch_fl/flagcx（py311 重新编译：`ACCELERATOR=ascend`）**
- site-packages 放 `flagos_torchfl.pth`（import flagos_boot）自动加载 shim + 环境适配
- **模型 Qwen3-4B 已下载**（ModelScope，14 文件，/workspace/models/Qwen3-4B）

### 挂点清单（完整，全部已解——推理阶段挂点用 💡 标记）

| # | 挂点 | 解法 |
|---|------|------|
| 1 | vllm 依赖 torch==2.11.0 vs torch_fl 要 2.10 | pip 元数据硬 pin 可绕过（--force-reinstall 降 2.10.0+cpu）|
| 2 | 官方 triton 3.5/3.6 driver 层无 GPU 不可用 | 复用 vllm-ascend 镜像 triton_ascend 3.2.1（cp311，零编译）|
| 3 | torch_fl 编译要 nvcc | `ACCELERATOR=ascend`（CMakeLists.txt:37 ascend 分支）|
| 4 | torch.npu/torch_npu 接口缺口 | torch_fl shim 补齐：empty_cache、_inductor、NPUGraph、mem_get_info（ctypes 调 aclrtGetMemInfo）、torch_npu._C（raw stream）、version、stream |
| 5 | torch.device("npu"/"cann") 不被认识 | torch_fl _remap 加 npu/cann → flagos 别名（可 PR）|
| 6 | set_device 收 torch.device（C++ 要 int）| 修 flagos.set_device / get_device_properties（index 转换）|
| 7 | torch.device 局部类不可 pickle（spawn）| 动态改 __module__/__qualname__ 为 flagos_boot.device |
| 8 | TorchFunctionMode 栈 thread-local | torch_fl Thread.start patch（子线程 push mode）|
| 9 | 子线程 ACL 上下文不可用（allocator -1）| torch_fl C++ get_device_index fallback 0（重编译）|
| 10 | set_data 不允许 cpu↔privateuseone | patch vllm utils.py（param 替换语义）|
| 11 | flagos 不参与默认 device/with 机制 | patch vllm base_loader（model.to + ModelWeightParameter 子类恢复：__dict__/weight_loader/MRO 方法/property）|
| 12 | torch.empty(device="npu") 特殊线程失效 | flagos_boot factory wrap（不依赖 mode）|
| 13 | 💡 split_with_sizes 在 flagos 无 kernel | flagos_boot Tensor.split fallback → flag_gems split_with_sizes_copy |
| 14 | 💡 detach_ 无注册（torch.tensor C++ 内部直调）| torch.library.impl("aten::detach_", "PrivateUse1", no-op) |
| 15 | 💡 torch.npu.stream 缺失 + _DefaultStreamHandle 属性 | shim 补 stream；_DefaultStreamHandle 补 device/stream_id；stream() 对 default handle no-op |
| 16 | 💡 torch.accelerator.synchronize 无支持（收尾清理）| flagos_boot patch（current_accelerator().type=="flagos" → torch.npu.synchronize）|
| 17 | flag_gems.device 名不一致（lift_fresh）| flagos_boot 只改 flag_gems.device 模块属性（不动 DeviceDetector）|
| 18 | flagcx 初始化 device="cann" | torch_fl _remap 加 cann 别名 |
| 19 | fused gate_up/QKV 加载形状 assert | base_loader 子类恢复（#11）|

### 插件层（vllm-plugin-FL，待改造，本次未动）
- worker.py:396 `import torch_npu._inductor`（shim 已覆盖，插件仍硬编码）
- worker.py:87 empty_cache / platform.py:132-136 内存统计（shim 已覆盖）
- ascend impl 需 npu_grouped_matmul 等 torch_npu 专用算子（MoE 模型触发；Qwen3-4B dense 未触发）

## 3. 操作链路与复现步骤

```bash
# 1. 建实验 venv + 装 vllm（--no-deps 后手动降 torch）
docker exec flagos-fl-dev-910c bash -lc 'python3 -m venv /root/vllm-venv'
docker exec flagos-fl-dev-910c bash -lc '/root/vllm-venv/bin/pip install vllm==0.20.2'   # 会装 torch 2.11.0
docker exec flagos-fl-dev-910c bash -lc '/root/vllm-venv/bin/pip install "torch==2.10.0+cpu" "torchaudio==2.10.0+cpu" "torchvision==0.25.0+cpu" --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps'

# 2. 装 flag_gems + vllm-plugin-FL（源码，不编译 C++）
docker exec flagos-fl-dev-910c bash -lc '/root/vllm-venv/bin/pip install --no-build-isolation -e /workspace/FlagGems'
docker exec flagos-fl-dev-910c bash -lc 'cd /workspace/vllm-plugin-FL && /root/vllm-venv/bin/pip install --no-build-isolation --no-deps -e /workspace/vllm-plugin-FL'

# 3. 拷 torch_fl + flagcx（复用阶段 3 编译产物，免编译）
SP=/root/tf-venv-integration/lib/python3.12/site-packages; VP=/root/vllm-venv/lib/python3.12/site-packages
cp -r $SP/torch_fl $SP/torch_fl-0.1.0.dist-info $SP/flagcx $SP/flagcx-0.13.0.dist-info $VP/

# 4. 自动加载 torch_fl shim
echo "import torch_fl" > /root/vllm-venv/lib/python3.12/site-packages/flagos_torchfl.pth

# 5. 探测脚本（挂点链）
docker exec flagos-fl-dev-910c bash -lc 'timeout 180 /root/vllm-venv/bin/python /workspace/scripts/vllm_fl_probe.py 1'
```

## 4. 测试结果明细（挂点链，自底向上实测）

| # | 环节 | 结果 | 说明 |
|---|---|---|---|
| 1 | vllm 0.20.2 + torch 2.10.0+cpu import | ✅ | 无 CUDA/CANN 环境 import 正常；torch 2.11 约束只是 pip 元数据 |
| 2 | torch_fl shim（.pth 自动加载） | ✅ | torch.npu 11 接口 + torch_npu 模块别名就位；`device_count()=16` |
| 3 | flag_gems + vllm_fl import | ✅ 已解 | 无 torch_fl 时 `torch.npu` AttributeError（FlagGems gen_torch_device_object，backend/__init__.py:346）；有 shim 后消除 |
| 4 | triton 3.6.0 nvidia 后端初始化 | ✅ 已解 | 无 GPU 环境 libcuda.so.1 assert（triton/backends/nvidia/driver.py:40）；TRITON_LIBCUDA_PATH + stub .so 可绕过编译，但运行时 dlopen 仍要 LD_LIBRARY_PATH |
| 5 | triton 3.5/3.6 vs torch 2.10 | ✅ 已解 | TORCH_LIBRARY triton namespace 双注册冲突（torch 2.10 内置 _TritonLibrary，torch/__init__.py:2805）；patch 为惰性 DEF_FRAGMENT 后可消除（已验证） |
| 6 | triton driver 层（无 GPU） | ✅ 已解（昇腾 triton 方案） | `_create_driver` 要求恰好 1 个 active driver；nvidia `is_active()` 无 GPU 恒 False（driver.py:739）→ 官方 triton 无解，必须昇腾系 triton |
| 7 | torch_fl shim 接口覆盖 | ⚠️ 部分 | 缺 `empty_cache`(worker.py:87 会用)、`NPUGraph`(compilation/graph.py:47)、`_inductor`(worker.py:396)；`is_available()` 恒 False → ascend.py:43 判定后端不可用 |
| 8 | torch_fl .so 跨版本 ABI | ✅ 发现 | 2.10 编译的 _C.so 在 torch 2.11.0 下 import 正常（torch 版本约束放宽） |
| 9 | torch_npu 专用算子 | ✅ 未触发（dense 模型） | fused_moe.py:11/mm_encoder_attention.py:21 模块级 import 可被 shim 满足，但 npu_grouped_matmul 等算子调用 shim 没有（Qwen3-4B 非 MoE，可能不触发） |

## 5. 避坑清单

- **vllm 0.20.2 的 torch==2.11.0 是 pip 元数据硬 pin，但运行时与 2.10 兼容**——别被 pip 解析劝退，--force-reinstall 降级即可
- **triton 3.5+/3.6 与 torch 2.10 双注册冲突**：torch 2.10/2.11 内置 `_TritonLibrary`（torch/__init__.py:2805/2824）占用 triton namespace；venv 内 patch 为惰性 DEF_FRAGMENT 可消除（改 torch 源文件，非插件代码）
- **官方 triton 在无 NVIDIA GPU 机器 driver 层不可用**（is_active 恒 False，driver 需恰好 1 个 active）——stub libcuda 只能绕编译 assert，最终必须换昇腾系 triton（FlagTree/triton-ascend）
- torch_fl shim 的 `is_available()`=False 是**故意的**（torch_fl/__init__.py:676 注释：防 transformers/accelerate 误判真实 NPU）——vllm_fl 若依赖 is_available 判定会踩坑，改造时需改用 device_count()>0 或 vendor 检测
- torch_fl/flagcx 的 .so 缺失依赖（libtorch_cpu.so 等）由进程内 import torch 提供，跨 venv 拷贝 .so 可行（本次已验证）；且 .so 跨 torch 小版本 ABI 兼容（2.10 → 2.11 验证通过）
- **LD_LIBRARY_PATH 必须追加而非覆盖**：容器 /etc/profile.d 注入的 CANN 路径（libmsprofiler.so 等）不能丢，否则 torch_fl backend 自动加载失败
- torch 2.11 的 `_import_device_backends()` 会**自动加载 torch.backends entry points**（torch_fl 注册了）——无 CANN LD_LIBRARY_PATH 时会挂；可用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 禁用（torch_fl 显式 import 即可）
- .pth 文件自动加载：site-packages 下任意 .pth 内的 import 语句在 site 初始化时执行（先于业务代码）；若 import 抛异常会被静默吞掉（难排查）
- 容器内 git 操作需 safe.directory（FlagGems/vllm-plugin-FL 已配）

## 6. 失败经验（专章）

1. **官方 triton 路线在 torch_fl 环境走不通**（本轮最大发现，挂点 C 定论）：两层障碍——① torch 2.10/2.11 内置 _TritonLibrary 与 triton 3.5+ C++ TORCH_LIBRARY 双注册冲突（可 patch 消除）；② triton driver 层要求恰好 1 个 active driver，无 GPU 时 nvidia driver is_active() 恒 False（**不可绕**）→ FlagGems ascend 必须配昇腾系 triton（FlagTree 3.2.x / triton-ascend）。逐层绕 libcuda（stub 库）是死路
2. flagos-ai 官方 ascend CI 用 torch_npu（vllm-ascend 镜像），**无 torch_fl 参照**——路线 A 无现成模板；且其 CI 中 flag_gems 的 triton 方案需确认（vllm-ascend 镜像自带官方 triton，flag_gems ascend 算子如何编译存疑——可能其 CI 不触发 flag_gems triton 编译路径）
3. LD_LIBRARY_PATH 覆盖 CANN 路径 → torch_fl backend 加载失败（libmsprofiler.so），症状是 .pth 静默失效（torch_fl 未加载、torch.npu 不存在）——排查半天才定位
4. torch 2.11+cu130 的 aarch64 wheel 带 CUDA 组件，加载 stub libcuda 会 import 崩溃——stub 方案只适用于纯 CPU torch

## 7. 遗留风险与待办

1. ~~**triton 决策（路线 A 第一前置）**~~ **已解决（2026-08-15）**：复用 vllm-ascend 官方镜像内 triton_ascend 3.2.1（cp311，docker cp 零编译拷入 venv311），免 FlagTree 源码编译
2. **路线 B 容器窗口**（用户未响应默认跳过）：起 vllm-ascend 官方镜像容器（本机已有 18GB）需先停一个旧容器（DrvMng 上限≈3）——**未做容器实验**，参照行为改为静态情报（镜像 ENV：CANN 9.0.0 + python 3.11.15 + SOC 910_9391；官方 Dockerfile 构成）。**注：自有闭环（单卡+TP=2/4 数值正确）已跑通后，路线 B 参照价值大减，建议降级不做（可消解容器决策）**
3. ~~**模型权重**~~ **已完成**：Qwen3-4B 已下载（ModelScope，/workspace/models，宿主盘持久）
4. 挂点清单已比交接文档更新（25 处引用 vs 原 6 条），详见会话记录
5. **P0 算子清单**（2026-08-16 到期，待确认是否已交付）作为 dispatch 层覆盖度依据——MoE 模型（如 Qwen3-30B-A3B）触发 npu_grouped_matmul 等 torch_npu 专用算子，是插件层剩余改造量的关键未知点
6. **补丁 PR 上游**（待用户确认 git 身份/账号）：vllm-plugin-FL 2 补丁 + torch_fl 4 补丁，提交说明见《阶段4挂点补丁-PR提交说明-20260816.md》

## 8. 产物清单

- 实验 venv：容器内 /root/vllm-venv（torch 2.10.0+cpu + vllm 0.20.2 + flag_gems + vllm-plugin-FL + torch_fl + flagcx；含 .pth 自动加载 torch_fl、torch `_TritonLibrary` patch）
- 模型权重：/workspace/models/Qwen3-4B（ModelScope，14/14 文件 7.6G，2026-08-15 下载完成，宿主盘持久）
- 探测脚本：宿主 /home/xliu969/workspace/scripts/vllm_fl_probe.py（分步探测 import 链）、scripts/probe_torchfl_npu.py（torch_fl shim 接口画像）、scripts/probe_vllm_report.py（vllm 依赖报告解析）
- 探针报告：容器内 /tmp/vllm_report.json（vllm 0.20.2 依赖解析，含 aarch64 wheel 平台确认）
- 本轮全部为容器内改动 + 宿主脚本，**宿主配置/驱动/阶段 3 资产零改动**

---

## 附录 A 下一步（接续顺序）

1. **用户决策**：路线 B 起容器前停哪个旧容器（x-benchmark 同事容器 / bridge-old 已退出）
2. 路线 B 对照：vllm-ascend 镜像起容器跑 Qwen3-4B 单卡（torch_npu 生态参照行为）
3. 模型下载：ModelScope Qwen/Qwen3-4B → /workspace/models
4. triton 路线决策（与 flagos-ai 沟通或本地验证 FlagTree fork）
5. 路线 A 改造（platform → worker → 通信 → dispatch，每层验证）
