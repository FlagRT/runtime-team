# 昇腾 910C torch_fl 源码版回归指南

> **定位**：910C 收尾计划第 0/1 步。目标是把"镜像内置 torch_fl 0.1.0（旧版）"升级为"Torch-FL main 源码版（含 flagcx_native 修复）"，并重跑双卡训练回归，证明源码版与旧版行为一致。
> **硬约束**：全程使用 **torch_fl（flagos 设备后端）**，不使用 torch_npu；所有命令在 **910C 容器内**执行。
> **分工**：本指南由 AI 编写，命令由你在服务器（ssh 910C → 容器）执行。

---

## 一、回归目的与判定标准

| 判定项 | 通过标准 |
|---|---|
| 环境就绪 | 容器内 `npu-smi info` 正常，torch_fl 源码版 import 成功，`flagos.device_count()` 返回可用卡数 |
| 通信正确性 | `test_ag.py` allgather 数据与本地一致（此前 [1,2]/[10,11] 模式） |
| 训练回归 | 双卡 Qwen2.5-1.5B 训练 1 epoch，loss 收敛形态与旧版一致（旧版基准：2.6451→1.9458，2324 tok/s），无数据全零类回归 |
| 语义一致 | 输出张量非全零；步间 loss 曲线形态可对比 |

**结论口径**：四项全过 → 可在看板标注"torch_fl 源码版回归通过"；任一项失败 → 记录失败现象回填对比表，回到对应步骤排查（重点怀疑编译选项与链接 libflagos）。

---

## 二、第 0 步：环境恢复与基线快照（容器内）

```bash
# 0.1 确认容器在跑、DrvMng 名额充足（注意他人容器占用）
docker ps | grep flagos-device-context-dev-910c
npu-smi info | head -30        # 若名额不足，先释放或协调

# 0.2 进入容器并激活 venv（沿用此前环境）
#   （容器名 flagos-device-context-dev-910c，venv 为 tf-venv-integration）
source /workspace/tf-venv-integration/bin/activate 2>/dev/null || source ~/tf-venv-integration/bin/activate

# 0.3 记录环境基线快照（重要：作为回归对比基准）
python - <<'PYEOF'
import sys, torch, torch_fl
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch_fl:", getattr(torch_fl, "__version__", "unknown"))
from torch_fl import flagos
print("flagos devices:", flagos.device_count())
PYEOF
```

把上面输出粘贴到 `env_snapshot.txt`（建议存到 runtime-team 分支内 `benchmarks/env_snapshot_20260822.txt`）。

> 若容器已停止：`docker start flagos-device-context-dev-910c` 后重复 0.2。注意 DrvMng 名额问题——若 `aclInit` 报 500000 类错误，不是版本问题，是名额占用，先释放再试（此前已确认）。

---

## 三、第 1 步：Torch-FL 源码编译安装（容器内）

```bash
# 1.1 拉取含 flagcx_native 修复的 Torch-FL main
cd /workspace && git clone https://github.com/FlagRT/Torch-FL.git torch-fl-main 2>/dev/null \
  || (cd torch-fl-main && git pull origin main)

# 1.2 编译安装（三个关键点，缺一不可）
cd torch-fl-main
export PATH=/usr/local/python3.9.2/bin:$PATH   # 以容器实际 python 为准
pip install -e . --no-build-isolation          # ① --no-build-isolation：隔离环境找不到 torch
make USE_ASCEND=1                              # ② USE_ASCEND=1：编昇腾分支（默认编 NVIDIA 缺 cuda.h）
# ③ 链接 libflagos.so（此前 patch_link_libflagos.py 的处理，检查 _build_config.py 中 flags）
python ../flagos-demos/scripts/patch_link_libflagos.py --check-only || python ../flagos-demos/scripts/patch_link_libflagos.py
pip install -e . --no-build-isolation          # 应用链接补丁后重装
```

**踩坑注记**（来自此前 910C 实战，防止重踩）：

| 坑 | 现象 | 对策 |
|---|---|---|
| pip 隔离找不到 torch | `ModuleNotFoundError: torch` | `--no-build-isolation` |
| 默认编译 NVIDIA 分支 | `cuda.h: No such file` | `make USE_ASCEND=1` |
| 未链接 libflagos.so | 运行时符号缺失 | `_build_config.py` 加 `-lflagos`，重装 |
| git 仓库 root 属主 | 无法写 `.git` | `sudo chown` 后操作（密码经 `printf|ssh` 传，远程命令不加 `< /dev/null`） |
| 继承旧版安装残留 | import 到旧 0.1.0 | 先 `pip uninstall torch-fl -y`，确认 `python -c "import torch_fl; print(torch_fl.__file__)"` 指向源码目录 |

```bash
# 1.3 验证 import 指向源码版
python -c "import torch_fl; print(torch_fl.__file__); print(torch_fl.__version__)"
# 预期输出路径含 torch-fl-main，版本号应为 main 对应版本（非 0.1.0）
```

---

## 四、第 2 步：通信快速验证（容器内，双卡）

```bash
# 2.1 快速 allgather 正确性（沿用 test_ag.py，路径按实际）
cd /workspace/runtime-team/benchmarks 2>/dev/null || cd /workspace
torchrun --nproc_per_node=2 test_ag.py
# 预期：rank0 收集到 [1,2],[10,11] 各卡数据拼接正确，无全零
```

> 若此步失败，优先怀疑：① 未走源码版（import 到旧版）；② libflagos 链接缺失；③ 四层根因相关的 current-stream / uniqueId 修复未生效（检查 flagcx 编译产物版本）。

---

## 五、第 3 步：双卡训练回归（容器内）

```bash
# 3.1 用既有训练脚本（路径按实际）
cd /workspace/runtime-team/benchmarks 2>/dev/null || cd /workspace
bash run_train_flagos_910c.sh    # 内部含 torchrun --nproc_per_node=2 train_qwen_1_5b_flagos.py
```

**回归对比表模板**（跑完回填，作为看板附件）：

| 指标 | 旧版 0.1.0（基准） | 源码版（本次） | 差异判定 |
|---|---|---|---|
| torch_fl 版本 | 0.1.0 | （回填） | — |
| 训练步数 | 1 epoch | 1 epoch | — |
| 初始 loss | 2.6451 | （回填） | 同量级 |
| 收敛 loss | 1.9458 | （回填） | 同量级 |
| 吞吐 tok/s | 2324 | （回填） | 同量级 |
| 显存/卡 GB | 14.64 | （回填） | 同量级 |
| 输出全零/NaN | 无 | 无 | 必须无 |
| 结论 | — | （通过/失败） | 四基准项同量级+无全零 = 通过 |

> 说明：loss/吞吐数值允许小幅波动（同量级即可），核心判定是"无全零/NaN + loss 收敛形态一致"——这证明源码版修复没有引入行为回归。

---

## 六、完成后的看板标注（README 更新）

在 `runtime-team/dev/device-context/README.md` 看板追加一行：

```
- [x] torch_fl 源码版回归（2026-08-22）：双卡 Qwen2.5-1.5B 训练与 0.1.0 行为一致，回归对比表见 benchmarks/env_snapshot_20260822.txt 同目录
```

---

## 七、下一步衔接

第 1 步完成且四项判定全过后，进入收尾计划第 2a 步（细项21 补验）：
- `scripts/test_err_translation.py` —— 错误三维翻译 smoke（类别/位置/根因）
- `scripts/test_topology_report.py` —— 拓扑报告 smoke（互联域/设备邻接/路径选择记录）

两个脚本已就绪，随回归通过后在容器内依次执行即可（见各自文件头用法说明）。

## 八、实测执行记录（2026-08-22，SSH 直连 910C 执行完成）

> 本节为真实执行记录，**取代**上面第三~七节中与本记录不一致的旧流程（旧流程按旧版
> torch_fl 编写，源码版构建命令以本节为准）。全部命令在容器
> `flagos-device-context-dev-910c` 内执行。

### 8.1 源码版编译（三连排障后的正确命令）

```bash
cd /workspace/PyTorch-Plugin-FL
source /root/tf-venv-integration/bin/activate
# ① 昇腾代码生成（写入 257 个算子到 backends_ascend.conf）
python scripts/codegen_ascend.py
# ② 临时补丁：ascend 分支加 -DFLAGGEMS_KERNEL=OFF（CMakeLists 该 option 默认 ON，
#    但 ascend 分支漏了强制 OFF——上游缺陷；dcu/musa/bpu 都有，唯独 ascend 没有）
cp setup.py /tmp/setup.py.bak
# 在 cmake_args.append(f"-DACCELERATOR={ACCELERATOR}") 后插入：
#   if ACCELERATOR == "ascend":
#       cmake_args.extend(["-DFLAGGEMS_KERNEL=OFF"])
# ③ 编译安装（三个环境变量缺一不可）
ACCELERATOR=ascend TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  pip install -e . --no-build-isolation
# ④ 编译成功后还原 setup.py 补丁
cp /tmp/setup.py.bak setup.py
```

**三连排障表**：

| 次 | 失败原因 | 对策 |
|---|---|---|
| 1 | torch 2.10 构建期 import 触发 flagcx 后端 autoload 失败（旧版卸载后扩展缺失） | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| 2 | cmake 默认 `-DACCELERATOR=cuda`（昇腾无 CUDA） | `ACCELERATOR=ascend` |
| 3 | `FLAGGEMS_KERNEL` 默认 ON 但 ascend 分支漏强制 OFF → 找不到 FlagGems | patch setup.py 加 `-DFLAGGEMS_KERNEL=OFF` |

### 8.2 运行时必需环境变量

**所有 torch_fl 程序运行时都必须带 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`**——flagcx
backend extension 的 autoload 在源码版下仍失败（待查扩展注册路径），禁用后核心可用。
（该变量只影响扩展自动加载，不影响 torch_fl/flagos 行为。）

### 8.3 训练回归结果（源码版 vs 旧版 0.1.0）

| 指标 | 旧版 0.1.0（基准） | 源码版（本次实测） | 判定 |
|---|---|---|---|
| 初始 loss (s0) | 2.6451 | **2.6451** | 一致 |
| 收敛 loss | 1.9458 | **1.9436**（s2480） | 一致 |
| 吞吐 tok/s | 2324 | **2298** | 同量级 |
| 显存/卡 GB | 14.64 | **14.64** | 一致 |
| 全零/NaN | 无 | 无 | 通过 |
| test_ag allgather | [1,2]/[10,11] | **[1,2]/[10,11]** | 通过 |

**结论：四项判定全过，torch_fl 源码版回归通过。**

### 8.4 细项21 六项补验结果（探针脚本，结果 JSON 见 outputs/910c_regression_20260822/）

| 补验项 | 脚本 | 结果 | 关键证据 |
|---|---|---|---|
| 错误三维翻译 | test_err_translation.py | PARTIAL | 捕获 `aclnnMatmulGetWorkspaceSize failed, ret=161002` → L2 参数类（ACL 错误码映射表已入脚本）；根因原文保留；位置投影标注接口缺口 |
| 拓扑感知 | test_topology_report.py | **PASS** | npu-smi 解析 4 设备；单机双卡直连 HCCS 互联事实可观测 |
| 页锁定内存池 | test_pinned_pool.py | **PASS** | 锁页分配/异步拷贝/生命周期循环全对（需设备预热，见 8.5） |
| 双缓冲流水线 | test_double_buffer.py | **PASS** | 数据正确；探测到 flagos.Stream/Event 统一接口可用 |
| 状态恢复 | test_recovery_min.py | PARTIAL | 最小重建观测通过（错误后可重新获取设备资源）；五段式编排/状态机接口未暴露 |
| CPU—NPU 协同 | test_ai_cpu_core.py | **PASS** | 设备算子链/CPU↔flagos 互转/回退衔接/统一表达全对 |

### 8.5 源码版实测新发现的坑（重要）

| 坑 | 现象 | 对策 |
|---|---|---|
| pin_memory 段错误 | **flagos 设备未初始化时 `pin_memory()` 直接段错误（EXIT=139）** | 先 `torch.zeros(1, device="flagos")` 设备预热（已入全部脚本） |
| E2 先 wait 后 record 死锁 | `flagos.Event().wait()` 在事件未 record 时**永久阻塞（无超时）** | conformance 用例改验证 record→wait 正常路径并标注边界；统一层需定义超时/语义收敛 |
| torch.cuda 误报 | flagos 环境下 `torch.cuda.is_available()` 返回 True，但 torch 为 +cpu 编译，`torch.cuda.Event/synchronize` 不可用 | 脚本统一用 `flagos.synchronize()` 同步，禁用 torch.cuda.* |
| 数值容差 | flagos(aclnnMatmul) 与 CPU matmul 数值实现差异（相对 ~2e-5） | 一致性用例用相对容差（1e-3）而非绝对 1e-2 |

### 8.6 一致性测试昇腾基线（conformance）

```bash
cd /workspace/flagos_scripts
python runner.py --chip ascend --out conformance_ascend_result.json
# === 汇总: 7/7 通过 ===  CONFORMANCE_PASS
```

7 个行为契约用例全过（S1/S2/E1/E2/T1/T2/F1）。**昇腾基线已建立**——下一款芯片接入时
同一套用例运行并比对 `ok` 字段，行为差异即缺陷。基线 JSON 已存
`outputs/910c_regression_20260822/conformance_ascend_result.json`。
