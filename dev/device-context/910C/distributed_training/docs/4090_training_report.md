# 4090-1 两卡 1.5B 训练报告（FlagCX NVIDIA 适配）

> 完成日期：2026-08-25 ｜ 环境：4090-1（8×RTX 4090 24GB，本报告用 GPU 1/3 两卡）
> 软件：torch 2.11.0+cu130 ｜ FlagCX kistich/ascend-dev1.0 @ 4fbd9e9（nvidia adaptor）
> 模型：Qwen2.5-1.5B（bf16）｜ 数据：wikitext-2-raw-v1 ｜ 与 910C 同款训练脚本（CUDA 版）

## 一、结果总览（与 910C / B 线三平台对照）

| 平台 | 通信后端 | 训练步数 | 最终 loss | 吞吐 (tok/s) | checkpoint |
| --- | --- | --- | --- | --- | --- |
| 910C | torch_npu + 官方 HCCL | 2481 全程 | 1.9472 | 5428 | ✓ |
| 910C | torch_npu + FlagCX(hccl) | 2481 全程 | 1.9501 | 4157 | ✓ |
| **4090-1** | **CUDA + FlagCX(nvidia)** | **2481 全程** | **1.9432** | **1538** | **✓** |
| 4090-1 | CUDA + NCCL（torch_fl 历史） | 2481 全程 | ~1.94 | — | ✓ |

- **loss 收敛一致**（1.943-1.950），FlagCX 在 NVIDIA 平台的 DDP 训练完整闭环
- 4090 吞吐低于 910C：消费级卡 + **无 NVLink**（梯度 allreduce 走 PCIe P2P），
  且 bf16 在 4090 上无 tensor core 加速（消费级限制），属预期性能

## 二、部署要点（4090-1 host，无 sudo）

| 项 | 说明 |
| --- | --- |
| Python | `~/dvfs/.venv_4090`（torch 2.11.0+cu130；base conda 无 torch） |
| 编译工具 | gcc 9.4 / make / cmake；`/usr/local/cuda` → cuda-13.0（与 torch cu130 匹配） |
| nlohmann json | 单头文件下载到 `~/include/nlohmann/json.hpp`，`make JSON_INCLUDE_DIR=~/include` |
| NCCL 头/库 | pip 包 `nvidia/nccl`（`~/.venv_4090/.../site-packages/nvidia/nccl/`）→ `CCL_INCLUDE`/`CCL_LIB`/`CCL_LINK=-l:libnccl.so.2` |
| C++ 标准 | **`HOST_COMPILER="g++ -std=c++17"`**（CUDA 13 的 cccl 要求 C++17） |
| 插件安装 | pip editable wheel 报 "Can't mix absolute and relative paths" → 改用 `setup.py build_ext --inplace` + `.pth` 注册 |
| GPU 选择 | 8 卡共用机器，用 `CUDA_VISIBLE_DEVICES=1,3` 挑空闲卡 |

## 三、踩坑与修复（FlagCX nvidia 适配）

1. **NCCL IB 探测段错误**：comm init 时 `ncclIbMatchVfPath(path2=null)` → SIGSEGV。
   修复：`NCCL_IB_DISABLE=1`（4090 单机训练不需要 IB，走 NVLink/P2P/TCP）。
   已并入脚本环境变量。
2. **collective stream 语义**：flagcxBackend 的 nvidia 分支 `getStreamByIndex(0)`
   用自建缓存流 → 与 torch 当前流无 happens-before → 双卡 allgather 第二次调用
   rank1 数据全 0（910C 昇腾分支同根因的 nvidia 变体）。
   修复：`getStreamByIndex(0)` 返回 `at::cuda::getCurrentCUDAStream(deviceId_).stream()`
   （FlagCX commit 4fbd9e9）——collective 与 tensor 计算同流，天然有序。

## 四、验证脚本（flagos-demos/scripts/）

- `test_ag_cuda.py`：双卡 allgather（预期 out=[1,2] / out2=[10,11] 全对）
- `train_qwen_1_5b_cuda.py`：两卡 DDP 训练（backend=flagcx，内置 NCCL_IB_DISABLE=1；
  MODEL_PATH 支持本地路径或 HF 模型名）
- `patch_nvidia_current_stream.py`：nvidia 当前流修复的可复现 patch

## 五、运行方式（4090-1）

```bash
export LD_LIBRARY_PATH=$HOME/FlagCX/build/lib:$LD_LIBRARY_PATH
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,3          # 挑空闲卡
~/dvfs/.venv_4090/bin/torchrun --nproc_per_node=2 --master_port=29523 \
  train_qwen_1_5b_cuda.py
```
