# 昇腾 910C venv 补丁台账(2026-09-02 全卡复现会话)

> ## 🧊 已冻结路线工装
> 本台账与 `*.diff` / `*.patch` 服务于 **torch_fl + vllm-plugin-FL 设备栈**在 910C 的复现,该路线已冻结(见 [../README.md](../README.md))。仅留档,**勿在其上继续开发**。
>
> 用途(历史):记录 venv312 相对基础镜像/上游源码的增量——触发挂点、根因、改动文件、diff 归档路径、镜像固化方式。

## 0. 背景与教训(为什么有这份台账)

2026-09-02 卡空闲后做 memory 子方向全卡复现,执行 Qwen3-4B 推理闭环
(上次被同事 sglang 占卡阻塞的项)。**暴露问题:旧机(venv311)的 venv 内补丁
从未入库**,新机 venv312 组装时全部丢失,导致 8+ 个挂点需重新逐个排查/修复。
本台账 + `patches/` diff 归档即为此类补丁的**唯一权威载体**。

## 1. 补丁清单(按 boot 加载顺序/依赖序)

### P1. flagos_boot.py —— venv 引导模块(单文件承载大部分 boot 层补丁)
- 位置:site-packages/flagos_boot.py(由 flagos_torchfl.pth 单行 `import flagos_boot` 加载)
- 仓内副本(权威):`dev/memory/probes/flagos_boot.py`(已同步,改后须 docker cp)
- 职责:见文件头 docstring;2026-09-02 新增 5 项(带注释锚点):
  1. `torch_npu.__spec__` 修正 + `__path__`(triton_ascend `_get_package_dir("torch_npu")`
     找不到 include 目录 → npu_utils JIT 编译失败;旧 shim 只给 origin 无 submodule_search_locations)
  2. `torch_npu._C` / `torch_npu.version` 注册进 sys.modules + `_npu_getCurrentRawStream[NoWait]`
     (vllm_fl/EngineCore 子进程 `import torch_npu._C` 报 "is not a package")
  3. `torch.npu.NPUGraph`、`torch.npu.reset_peak_memory_stats`(vllm_fl platform/graph import 期直读)
  4. `flag_gems.device` 对齐 'npu'→'flagos'(lift_fresh 后端名判定;旧机挂点 #17,重建版曾漏)
  5. **factory wrap(§1b)**:torch.empty/zeros/... 显式 remap device='npu'→flagos,
     不依赖 thread-local mode(triton benchmark 线程无 mode 栈)
  6. **compat op 宽松注册(§5c)**:`aten::_has_compatible_shallow_copy_type` 对
     PrivateUse1/CPU 返回 True —— EngineCore(spawn)进程内 torch_fl CatchAll 宽松版
     被 composite 默认实现盖回,`p.data=p.data.to("flagos")`(vllm device_loading_context)
     抛 "incompatible tensor type"(旧机挂点 #9 根因,旧机是 venv patch vllm utils.py)
  7. `get_device_properties` int 兼容 wrapper(torch.cuda.get_device_capability 传
     torch.device 对象 → flagos 收 int 崩溃;注意 torch_fl/__init__.py:919 已把
     torch.cuda.get_device_properties 绑到原函数,boot 须重新绑定)

### P2. triton_ascend 3.2.2 源码补丁(npu_utils.cpp)
- 位置:site-packages/triton/backends/ascend/npu_utils.cpp
- 改动:USE_TORCH_NPU 段去掉 `torch_npu/csrc/...` include 与 at_npu 符号依赖;
  `triton_allocate_sync_block_lock` 用 `at::empty(device=PrivateUse1)` 替代
  `at_npu::native::allocate_workspace`;`triton_async_launch` 同步直调替代 OpCommand
- 根因:venv312 只有 stub torch_npu(无真库符号),编译期 OK 但 import 期
  RTLD_NOW 解析 at_npu 符号失败
- 修复后须 `rm -rf ~/.triton/cache`(旧 so 缓存 key 失效)
- diff 归档:`patches/triton_ascend_npu_utils.cpp.diff`(拷自容器 /tmp/patch_export 前身,
  以本文撰写时容器内文件为准重新 diff)

### P3. vllm 0.20.2 venv patch(base_loader.py)
- 位置:site-packages/vllm/model_executor/model_loader/base_loader.py
- 改动:initialize_model 后显式 `model.to(target_device)`(target != cpu 时)
- 根因:torch 2.10 的 `with torch.device("flagos")` 默认设备上下文对 PrivateUse1
  **不生效**,模型参数建在 CPU(vllm CUDA 依赖该上下文建参);旧机挂点 #11 同款
- diff 归档:`patches/vllm_base_loader_model_to.diff`

### P4. triton_ascend 3.2.2 npu_utils.cpp workspace 悬空指针(本次新发现)
- 位置:同 P2 文件,`triton_allocate_workspace_legacy` 函数
- 改动:static `unordered_map<uint64_t, at::Tensor>` 按 size 保活 + mutex;不再返回
  临时 tensor 的 storage 指针
- 根因:P2 改造时 workspace 用 `at::empty(...).storage().data()` —— 临时 tensor 表达式
  结束即析构、内存归还缓存池 → **悬空指针**。需要 workspace 的 kernel(unified_attention,
  workspace_size=31812)异步执行期间写悬空内存 → AICore 100% 死循环;workspace_size=0
  的 kernel(rms_norm 等)不受影响。真 torch_npu 版调 at_npu 持久 workspace 池,无此问题
- 症状:单卡推理跑到 attention 后卡死(实测 19 分钟无输出,AICore 100%)
- 修复后须 `rm -rf ~/.triton/cache`(so 缓存 key 含源码 hash)
- diff 归档:`patches/P4-triton-npu_utils-workspace-dangling.patch`(相对 nightly 原版,
  含 P2 全部改动 + P4 增量;P2 单独 diff 见其文件头注释)

## 2. 镜像固化指示(把补丁打进城,而非容器内找补)

目标形态:以 `harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev`
为基础镜像,Dockerfile 内完成 venv 组装时**按本台账依次 apply**,产出新 tag 供
`docker run` 直接使用。容器只做运行,不做组装。

```dockerfile
# 示意(待落成 dev/memory/镜像/ 下完整 Dockerfile,勿直接照抄)
FROM harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev

# 1) venv 组装(见复现验证记录 §4b 顺序):torch 2.10.0+cpu → triton_ascend 3.2.2
#    (docker cp 自 quay.io/ascend/vllm-ascend:nightly-main-a3,cp312)→ torch_fl 编译
#    (ACCELERATOR=ascend FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=0)→ vllm 0.20.2 --no-deps
#    → flag_gems/vllm-plugin-FL editable → flagcx(FLAGCX_ADAPTOR=ascend, fix 分支)

# 2) P1 flagos_boot:COPY dev/memory/probes/flagos_boot.py $SP/flagos_boot.py
#    + 确保 flagos_torchfl.pth 存在(单行:import flagos_boot)

# 3) P2 triton npu_utils.cpp:COPY patches/triton_ascend_npu_utils.cpp.diff /
#    RUN cd $SP/triton/backends/ascend && patch -p0 < /triton_ascend_npu_utils.cpp.diff
#    RUN rm -rf /root/.triton/cache

# 4) P3 vllm base_loader:COPY patches/vllm_base_loader_model_to.diff /
#    RUN cd $SP/vllm/model_executor/model_loader && patch -p0 < /vllm_base_loader_model_to.diff
```

固化命令(宿主,一次性收敛容器内 diff):
```bash
# P2
docker exec flagos-fl-dev-910c bash -c \
  'diff -u <(docker run --rm quay.io/ascend/vllm-ascend:nightly-main-a3 cat \
    /usr/local/python3.12.13/lib/python3.12/site-packages/triton/backends/ascend/npu_utils.cpp) \
    /root/vllm-venv312/lib/python3.12/site-packages/triton/backends/ascend/npu_utils.cpp'
#  → 落 patches/triton_ascend_npu_utils.cpp.diff
# P3
docker exec flagos-fl-dev-910c bash -c \
  'diff -u /tmp/base_loader.py.bak \
    /root/vllm-venv312/lib/python3.12/site-packages/vllm/model_executor/model_loader/base_loader.py'
#  → 落 patches/vllm_base_loader_model_to.diff(容器内 .bak 为原始备份)
```

## 3. 遗留(未固化/待决策)

- torch_fl 源码补丁 `ascend_memory.h` get_device_index fallback 在
  xliu969/wip-ascend-mem 分支(ebc8762)留档,合入 dev-1.0 需人工 merge(两分支该文件有差异)
- flagcx fix 分支 local-ascend-fix(153cdfd)未 push;上镜像需推 fork 分支供拉取
- vllm-plugin-FL 的 venv patch(若有)与上游 dev-1.0 的关系未核 —— P3 若上游已修可省
- 完整 Dockerfile 与 tag 命名未落盘(本文 §2 为指示,待实现)
