# 重建 ascend-train-comm v2

在 `ascend-operator-runtime:v2` 之上叠加 FlagCX，全新血统首次构建。

## 方式

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`。父层
`flagrt/ascend-operator-runtime:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-arm64`
需先建（见 `../../ascend-operator-runtime/v2/REBUILD.md`）。需要构建期网络
（`docker build --network=host`）——FlagCX 的两个 git submodule
（`third-party/json`、`third-party/googletest`）构建期直接 clone，均为公开仓库。

## 构建上下文需备齐

| 项 | 来源 |
|---|---|
| `Dockerfile` | = 本目录 `Dockerfile.repro`（build.sh 会拷） |
| `src/FlagCX/` | `git archive` 自 `FlagCX @ 4e0e0cbcbf721169ca82348080f8353aebfe2c31`（FlagRT 组织仓公开主干 tip） |
| `assets/verify_flagcx_runtime.py` / `verify_flagcx_p2p.py` | 本目录 `assets/` |

## 相比 v1 的改进

见 `lock.yaml` 的 `improvement_over_v1` 字段：FlagCX 层改用公开 commit
`4e0e0cbcbf721169ca82348080f8353aebfe2c31`（`FlagRT/FlagCX` 组织仓公开主干，
同步自 `flagos-ai/FlagCX` 公开上游），`USE_ASCEND=1 pip install .
--no-build-isolation` 直接从干净 checkout 端到端构建。相比 v1 依赖的 owner
私有 commit `55eb2ffff698…` + 2 个私有 patch（`lock.yaml:gaps`，需回收已装包
vendor），v2 **没有 gaps 段**——这是真正可从零复现的训练腿通信层。

## 验证（2026-09-12 实测，真机 2 卡）

| 判据 | 结果 |
|---|---|
| `docker build --network=host` | 通过，端到端从源码编译（无需 vendor 任何回收件） |
| FlagCX `.so` 存在 + 链接检查（`verify_flagcx_runtime.py --static`） | 通过（`libascendcl.so` / `libhccl.so` 齐全，未链 `torch_npu`） |
| `verify_flagcx_runtime.py --device`（真机 2 卡，torchrun） | 通过：两个 rank 均 `device_all_reduce_ok: true`，`torch_npu_coexistence_check: passed` |
| `dev/communication/probes/communication_correctness.py`（真机 2 卡，torchrun，复用现成脚本，未重复造轮子） | **40/40 通过**（all_reduce / all_gather / p2p / async_all_reduce，fp32 + bf16） |

命令（在带 2 张卡的容器内，容器名 `flagos-cand-train-910c-v2`，验证完已
`docker stop/rm`）：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/usr/local/python3.11.15/bin/torchrun --nproc_per_node=2 --master_port=<port> \
  /opt/flagrt/verify_flagcx_runtime.py --device
/usr/local/python3.11.15/bin/torchrun --nproc_per_node=2 --master_port=<port> \
  dev/communication/probes/communication_correctness.py --iterations 20 --out-dir /tmp/comm-results
```

## 已知问题：进程退出期 SIGABRT（已隔离，未修复）

成功完成集合通信、结果数值正确之后，进程退出（无论显式
`dist.destroy_process_group()` 还是让 Python 自然退出触发 atexit）会以
`free(): invalid pointer` 报错并 SIGABRT。**已用最小复现隔离，与 torch_npu
共存无关**——完全不 import torch_npu 的
`dev/communication/probes/communication_correctness.py` 在写出 PASS 结果文件
之后于进程退出时复现同一崩溃；剥离到只剩 `import torch_fl` + `dist.init_
process_group("flagos")` + 一次 `all_reduce` + 退出的最小脚本同样复现。推测是
FlagCX 与 Torch-FL 各自的退出期清理（atexit/析构函数）顺序冲突导致的
double-free 类问题。

**处理方式**：不在本次重建范围内根因修复（需要 FlagCX/Torch-FL 侧协作定位）。
`assets/verify_flagcx_runtime.py` 已调整为在 `destroy_process_group` 之前打印
JSON 结果，`dev/communication/probes/communication_correctness.py` 本身也是先
写结果文件再退出——两者都不会让这个已知问题掩盖真实的通信正确性结果。**批处理式
短生命周期任务不能依赖进程退出码判断成功与否，需检查落盘/打印的结果本身。**
详见 `lock.yaml:known_issues`，跟进见 `dev/images/TODO.md`。

## 另一个真机实测发现：triton "2 active drivers" 冲突（已在 v2 basefix，非本层）

与本层无直接关系，但在同一次真机验证中发现并修复，记在
`../../ascend-operator-runtime/v2/assets/patch_triton_ascend_flagtree.py` 里
（第 4 条патч）：Torch-FL 的生态兼容 shim 把 `torch.cuda.is_available()` 打成
`lambda: flagos.device_count() > 0`，导致 FlagTree triton 的顶层驱动自动发现
误判"CUDA 也是 active"，与真正的 NPUDriver 冲突报错。已在 operator-runtime/v2
层修好（详见该层 REBUILD.md）。
