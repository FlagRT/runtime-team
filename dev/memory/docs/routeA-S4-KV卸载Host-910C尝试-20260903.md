# S4 KV 卸载到 Host —— 910C 昇腾尝试记录

> 日期:2026-09-03 ｜ 执行人:xliu969(agent 代跑) ｜ 机器:16× Ascend910C 全卡空闲
> 探针:`dev/memory/probes/routeA_s4_kv_host_offload_910c.py`
> 环境:910C 当前可用推理容器(vllm 0.20.2 + triton_ascend 3.2.2 + vllm-plugin-FL)
> 上游对照:P800 已跑通(vllm 0.13,《[vllm-0.13-allocator与offload调研-20260822](vllm-0.13-allocator与offload调研-20260822.md)》§4 + `routeA_s4_kv_*`)
>
> **关键:下述两处阻塞都在 vLLM 层(平台门 + CUDA 扩展),与设备栈无关——昇腾换任何设备层组合都会撞同样的门。**

## 结论速览

🔴 **910C 昇腾栈上,vLLM 0.20.2 官方 KV-cache→Host 卸载(native `OffloadingConnector`)当前不可用**,硬阻塞在平台门与 CUDA 扩展缺失两处,均非参数可绕。与 P800 的差异是**平台能力**(`is_cuda_alike`),不是配置问题。

| 阶段 | P800(vllm 0.13) | 910C(vllm 0.20.2) |
|---|---|---|
| `KVTransferConfig` 构造 | OK(`num_cpu_blocks`) | OK,但 **extra_config 键改为 `cpu_bytes_to_use`(字节)**;`num_cpu_blocks` 报 `cpu_bytes_to_use must be specified` |
| `CPUOffloadingSpec` 初始化 | OK | OK(内部按 `cpu_bytes_to_use // kv_bytes_per_offloaded_block` 自算块数) |
| `get_handlers()` 平台门 | 通过(`is_cuda_alike()==True`) | 🔴 **`Exception: CPU Offloading is currently only supported on CUDA-alike GPUs`**(`v1/kv_offload/cpu/spec.py:84`);`PlatformFL` device_name=npu → `is_cuda_alike()==False` |
| 传输算子 | `ops.swap_blocks` 可用 | 🔴 **`vllm._C` 加载失败**(`ImportError: libcudart.so.13`)→ `torch.ops._C_cache_ops.swap_blocks_batch` 不存在(`AttributeError`),`cpu_gpu.py:315` 必失败 |

## 复现命令(容器内)

```bash
docker exec -it flagos-fl-dev-910c bash
cd /workspace
# 正确的 0.20.2 配置(cpu_bytes_to_use),仍会撞平台门:
env ASCEND_RT_VISIBLE_DEVICES=0 VLLM_PLUGINS=fl VLLM_FL_USE_FLAGGEMS_ATTN=1 DO_NOT_TRACK=1 \
  S4_CONNECTOR=OffloadingConnector S4_CPU_BYTES_GB=4 S4_MAX_TOKENS=4 S4_XFER=0 \
  /root/vllm-venv312/bin/python -u dev/memory/probes/routeA_s4_kv_host_offload_910c.py
```

## 逐阶段证据

### 1. API 变更:`num_cpu_blocks` → `cpu_bytes_to_use`(0.13 → 0.20.2)

`vllm/v1/kv_offload/cpu/spec.py:19-24`(0.20.2):

```python
cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
if not cpu_bytes_to_use:
    raise Exception("cpu_bytes_to_use must be specified in kv_connector_extra_config")
```

块数不再由调用方给,改为 `int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block`(spec.py:30-40)。
→ P800 探针里绕 `num_cpu_blocks=0` 接线缺陷的写法在 0.20.2 上作废,直接传 `cpu_bytes_to_use`。

### 2. 平台硬门:`is_cuda_alike()`

`vllm/v1/kv_offload/cpu/spec.py:83-87`:

```python
def get_handlers(self, kv_caches):
    if not self._handlers:
        if not current_platform.is_cuda_alike():
            raise Exception("CPU Offloading is currently only supported on CUDA-alike GPUs")
```

实测 `PlatformFL`:`is_cuda_alike()=False` / `is_cuda()=False` / `is_out_of_tree()=True` / `device_name=npu` / `dispatch_key=PrivateUse1`。
P800 的 `PlatformFL`(xpytorch,USE_CUDA=ON)`is_cuda_alike()=True`,所以同一条 native 路径在昆仑芯放行、在昇腾拦下。

`SimpleCPUOffloadConnector`(注册表另一条 CPU 卸载路径)更不可行:`v1/simple_kv_offload/cuda_mem_ops.py`
直接调 CUDA driver API(`cudaHostRegister` / `cuMemcpyBatchAsync`)。

### 3. 即使绕过平台门:传输算子不存在

`探针 S4_FORCE_CUDA_ALIKE=1` 在父进程 patch `is_cuda_alike→True` 不生效(EngineCore 是独立子进程,重新 import)。
但根因已明确:`vllm._C` 是 CUDA 构建,本镜像无 `libcudart.so.13` →
`Failed to import from vllm._C: ImportError('libcudart.so.13: ...')`;
`torch.ops._C_cache_ops.swap_blocks_batch` 直接 `AttributeError`。
而 `CpuGpuOffloadingHandlers.transfer_async`(`cpu_gpu.py:315`)正是调 `ops.swap_blocks_batch` + `torch.cuda.Stream/Event`。

## 落地建议(下一步方向,未做)

官方 native 路径要在昇腾栈可用,需要其一:

1. **vllm-plugin-FL 侧补 CPU-offload 适配**:提供昇腾版 handlers —— 用 `aclrtMemcpyAsync`(H2D/D2H)+ ACL stream/event 复刻 `SingleDirectionOffloadingHandler` 语义,并让 `CPUOffloadingSpec` 的平台门对 `PlatformFL` 放行(patch 或上游加 `is_kv_offload_supported()` 钩子)。工作量对标 device-context 组 FlagCX 去 torch_npu 适配。
2. **换非算子级方案**:`LMCacheConnectorV1`(若 LMCache 有昇腾后端)或自研 Host 溢出层(挂 evict_blocks + aclnn D2H,0.13 调研一度认为官方路径可省)—— 即「KV 溢出无内置」的结论在昇腾仍成立。
3. **确认目标 vLLM 版本**:若昇腾最终锁 0.13(与 P800 对齐),P800 已验证的 native 路径能否直接复用需实测——0.13 的 `CPUOffloadingSpec` 同样有 `is_cuda_alike` 门 + `swap_blocks` CUDA 算子依赖,昇腾(非 cuda-alike、`vllm._C` 缺 libcudart)未必放行。故版本对齐**大概率不能**解开本阻塞,倾向维持 0.20.2。

## 产物

- 探针 `dev/memory/probes/routeA_s4_kv_host_offload_910c.py`(P800 版的昇腾移植;含 `S4_CONNECTOR` / `S4_CPU_BYTES_GB` / `S4_FORCE_CUDA_ALIKE` 开关)
- 容器内日志:`/root/s4_910c_run{1,2,3}.log`
- NPU 全程无残留(procs clean,卡回落基线 ~3GB)
