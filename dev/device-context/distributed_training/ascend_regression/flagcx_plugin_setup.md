# FlagCX torch 插件安装与验证（新容器/新环境速查）

> **适用**：昇腾 910C（A 线 torch_npu 环境）新容器/新 venv；FlagCX 已编译（`/workspace/FlagCX/build/lib/libflagcx.so` 存在）
> **日期**：2026-08-26（实测沉淀，flagos-hliu553-dev-910c 验证通过）
> **一句话**：**不用 pip install，用 .pth 注册 + LD_LIBRARY_PATH 指向 libflagcx.so**

---

## 一、前置检查（3 项，缺一不可）

```bash
# ① venv 与 torch_npu
source /root/tf-venv-integration/bin/activate   # 以实际 venv 为准
python -c "import torch, torch_npu; print(torch.npu.device_count())"   # 期望 >=1（DrvMng 名额正常）

# ② FlagCX 已编译（libflagcx.so）
ls -la /workspace/FlagCX/build/lib/libflagcx.so

# ③ 插件 Python 扩展已编译（py3.12）
ls /workspace/FlagCX/plugin/torch/flagcx/_C.cpython-312-*.so
```

若 ③ 缺失才需要编译插件（一般仓库里已有 py3.12 产物，**无需重建**）。

## 二、安装（核心 2 步）

```bash
# 1) .pth 注册插件包路径（替代 pip install -e，pip 该路径会 wheel 构建失败）
echo /workspace/FlagCX/plugin/torch > /root/tf-venv-integration/lib/python3.12/site-packages/flagcx.pth

# 2) 运行时 LD_LIBRARY_PATH（libflagcx.so 所在目录，**每次运行前必须 export**）
export LD_LIBRARY_PATH=/workspace/FlagCX/build/lib:$LD_LIBRARY_PATH
```

## 三、验证（2 条命令）

```bash
# ① import + backend 注册
cd /tmp && python -c "
import flagcx
import torch.distributed as dist
print('flagcx import ok:', flagcx.__file__)
print('backend registered:', hasattr(dist.Backend, 'FLAGCX') or 'flagcx' in (dist.Backend.backend_list if hasattr(dist.Backend, 'backend_list') else []))
"
# 期望：flagcx import ok + backend registered True

# ② 双卡 allgather 数据验证（A 线 torch_npu）
cd /workspace/dev/device-context/benchmarks
torchrun --nproc_per_node=2 --master_port=29511 test_ag_npu.py
# 期望：[agtest] rank0/1: out=[1, 2]  out2=[10, 11]
```

## 四、常见坑

| 坑 | 现象 | 对策 |
|---|---|---|
| `pip install -e .` 失败 | `Failed to build installable wheels`（pyproject wheel 构建问题） | **用 .pth 注册，不用 pip**（_C.so 已编译在仓库） |
| 忘记 LD_LIBRARY_PATH | import 后段错误 / `libflagcx.so: cannot open shared object` | 运行前 `export LD_LIBRARY_PATH=/workspace/FlagCX/build/lib:$LD_LIBRARY_PATH` |
| venv 路径不一致 | .pth 写错 site-packages | `python -c "import site; print(site.getsitepackages())"` 确认 |
| DrvMng 名额耗尽 | `torch.npu.device_count()=0` / npu-smi `-8020 device is used` | 停多余容器后**重启目标容器**（DrvMng 客户端重新注册） |
| torch_npu 版本不匹配 | import torch_npu 报错 | 与 torch 同版本安装（torch_npu==2.10.0 ↔ torch 2.10.0） |

## 五、一键脚本

```bash
# 简化安装（自动检测 venv 并写 .pth，输出验证命令）
bash flagos-demos/scripts/setup_flagcx_plugin.sh [/path/to/venv]
```

见同目录 `setup_flagcx_plugin.sh` 源码。
