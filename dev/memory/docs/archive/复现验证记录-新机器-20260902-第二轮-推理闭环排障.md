# memory 子方向全卡验证执行记录 —— 新机器 2026-09-02(二轮)

> 日期:2026-09-02 ｜ 执行人:xliu969(agent 代跑) ｜ 机器:16× Ascend910C 全卡空闲窗口
> 前置:一轮已复现分配器画像(5/5)、flagcx 双卡、链路 import(见《复现验证记录-新机器-20260902.md》)
> 本轮目标:全卡验证 + 打通单卡推理闭环(Qwen3-4B,唯一未复现项)+ **补丁体系化留档**

## 0. 结论速览

| 项 | 状态 |
|---|---|
| 分配器画像(1 卡,62.4GiB 满血档) | ✅ 5/5(一轮已绿,本轮回看) |
| flagcx 全卡 allreduce(16 进程×16 卡) | ✅ 16/16,校验和 136 正确 |
| vllm 链路 import / PlatformFL / KV cache 分配 | ✅ |
| 权重加载(7.6GB,1.13s) | ✅(修完挂点链后) |
| **单卡推理闭环(Qwen3-4B 生成)** | ✅ **链路打通(exit=0,确定性输出)**——但输出为乱码文本且极慢(4 tok/69s),数值正确性未达(见 §4) |
| TP=2/4 数值复现 | ⛔ 依赖输出数值正确,未达(下轮) |
| 补丁台账 + 镜像固化指示 | ✅ `dev/memory/docs/archive/patches/README.md`(P1-P4) |

## 1. 本轮挂点链(新机器 venv312 缺失,逐一修复)

旧机 venv311 的修复大多以 **venv 内手工 patch** 存在、未入库 → 新机 venv312 组装时未重打,
本轮逐个重建并**全部落盘归档**(补丁体系化的直接由头)。

1. **torch_npu.__spec__ 无 submodule_search_locations** → triton `_get_package_dir("torch_npu")`
   找不到 include → npu_utils JIT 编译失败。boot 重建 ModuleSpec + 指向真实 stub 包。
2. **stub include 不完整**(缺 third_party/acl/inc/graph/operator.h)→ 从本地
   `quay.io/ascend/vllm-ascend:nightly-main-a3` docker create + docker cp 完整 include 覆盖。
3. **npu_utils.so RTLD_NOW 加载失败**:undefined `at_npu::native::allocate_workspace`(3.2.2
   新增 USE_TORCH_NPU 段引用真 torch_npu 库,venv312 只有 stub)。改写 npu_utils.cpp 去 at_npu 依赖。
4. **torch.npu.NPUGraph 缺失**(vllm_fl graph.py:47)→ boot 双挂 torch.npu + torch_npu。
5. **flag_gems lift_fresh "requires a npu tensor"**(flag_gems.device='npu' vs tensor 'flagos')
   → boot 覆盖 flag_gems.device(旧机挂点 #17)。
6. **EngineCore `import torch_npu._C` / `.version` 失败**(boot 只挂属性未注册 sys.modules)
   → boot 注册子模块 + `_npu_getCurrentRawStream[NoWait]`。
7. **torch.npu.reset_peak_memory_stats 缺失**(PlatformFL.get_current_memory_usage)→ boot 镜像。
8. **factory wrap(旧机挂点 #12)**:triton benchmark 线程无 mode 栈,`torch.empty(device='npu')`
   不 remap → boot 加不依赖 mode 的显式工厂包装。
9. **set_data 跨设备被拒(旧机挂点 #9)**:`p.data = p.data.to(flagos)` 报 "incompatible tensor
   type"。单测通过但 EngineCore(spawn)必现 —— torch_fl 的 CatchAll 宽松注册在子进程被
   composite 默认盖回。boot §5c 显式以 PrivateUse1/CPU key 注册宽松版。
10. **vllm base_loader 缺 model.to(旧机挂点 #11)**:torch 2.10 `with torch.device("flagos")`
    上下文对 PrivateUse1 不生效(实测 ctx 内 empty 仍落 CPU)→ 参数全在 CPU。venv patch
    base_loader.py 在 initialize_model 后显式 `.to(load_device)`(已归档 P3)。
11. **get_device_properties 收 torch.device 崩溃**:torch.cuda.get_device_capability 传 device
    对象 → flagos 只收 int。boot wrapper + 重新绑定 torch.cuda.get_device_properties
    (torch_fl/__init__.py:919 绑的是原函数,boot 须覆盖)。
12. **triton workspace 悬空指针(本次新发现,归档 P4)**:`triton_allocate_workspace_legacy`
    用 `at::empty(...).storage().data()` 返回**临时 tensor 指针**(表达式结束即析构、内存回池)。
    需要 workspace 的 kernel(flash-attention workspace_size=31812)异步执行期间写悬空内存 →
    AICore 100% 死循环;workspace_size=0 的 kernel(rms_norm 等)不受影响。改 static per-size
    缓存保活。

## 2. 复现(已绿项,全部重跑确认)

### flagcx 全卡 allreduce —— 16 进程 × 16 卡
- 命令范式:`ASCEND_RT_VISIBLE_DEVICES=<n> FLAGCX_ADAPTOR=ascend` 16 进程,MASTER_PORT=29600
- 结果:16/16 OK,数值 136(=16×17/2)全 rank 正确(比一轮双卡更进一步,补全新卡档位)

### 分配器画像
- device 0,probe_allocator_profile.py:5/5 全绿,HBM 空闲 62.4GiB 满血档,与 08-17 语义一致

## 3. 推理闭环排障进度

修完 §1 挂点 1-12 后,单卡(ASCEND_RT_VISIBLE_DEVICES=0)可跑到:
- engine 初始化 ✓ 权重加载 1.13s ✓ KV cache 分配(174,624 tokens)✓
- prefill 首个算子链:rms_norm ✓ rotary_embedding ✓ 之后卡住

### 卡点定位过程(已排除项)
1. 卡 `unified_attention`(vllm 默认 TRITON_ATTN)launch,AICore 100% → 怀疑 workspace 悬空
   → 修 P4 后仍卡(该 kernel 用 SIMT 模板编译,mix_mode=mix)。
2. 切 `VLLM_FL_USE_FLAGGEMS_ATTN=1`(flag_gems AttentionFLBackend,不依赖真 torch_npu)
   → 卡点移到 flag_gems **linear**(qkv 投影的第一个 matmul)。
3. **triton launch 链路本身健康**:最小 add kernel 0.2ms/次 ✓;最小 matmul(dot)kernel 能执行。
4. **flag_gems linear 单算子**:M=1/4096×12288 首次 13s(编译),同 shape 二次 ~200ms;
   但 **torch 原生 matmul 同 shape 10ms**。→ flag_gems triton AIC 路径慢且(疑似)数值偏大
   (bf16 大 K 场景),torch_fl/aclnn 原生路径健康。
5. 结论方向:该场景(vllm Qwen3-4B + flagos)应让 matmul/linear 走 **torch 原生(aclnn)**,
   flag_gems 覆盖 AIC 密集算子在本环境是负优化/异常;dispatch policy/实现优先级待调整。
   (未完成,留给下轮)

## 4. 未决问题(下轮入口)

### 4a. 闭环已通但输出乱码 + 极慢(69s/4 tok)——数值正确性未达

- 触发方式:`VLLM_FL_USE_FLAGGEMS_ATTN=1`(flag_gems AttentionFLBackend)
  单卡跑 qwen3_mini_probe:exit=0,两次独立运行输出逐字一致
  (`'case家乡opia(function'`),确定非随机 → 系统性数值错
- **已排除**:triton launch 链路(add kernel 0.2ms 正常);torch 原生 matmul
  (bf16 K=4096 err=0.4997, rel 0.16%,健康);flag_gems linear 单算子可执行
  (err=0.8 略大但同量级,非乱码主因)
- **劫持链定位(重要)**:vllm 内算子并非全走 dispatch——flag_gems 的
  `apply_gems_patches_to_vllm`(FlagGems/src/flag_gems/patches/patch_vllm_all.py:643)
  **全局 patch vllm 类方法**:RMSNorm.forward_cuda / RotaryEmbedding.forward_cuda /
  PagedAttention.write_to_paged_cache / SiluAndMul.forward_cuda /
  FlashAttentionImpl.forward 等 → custom_gems_* 版;linear/matmul 则经
  `_VLLM_OPS_IMPLS` + dispatch_key 注入(py-spy 栈证实 linear 落在
  flag_gems/ops/linear.py)。**当前整条算子链(含 rotary=位置编码、attention、
  KV cache reshape)都在 flag_gems triton kernel 上执行** → 慢(~20×)与乱码同源嫌疑。
  该 patch 由谁触发、可否按平台跳过,待核(vllm_fl import 链或 flag_gems 自动)。
- **修复方向**:a) 逐算子数值对照(用 torch 原生实现逐一替换 flag_gems kernel 二分定位);
  b) 检查 apply_gems_patches_to_vllm 在 flagos 平台是否应跳过(rotary/attention/
  paged_cache 是乱码高嫌疑);c) 与旧机 venv311 对照——旧机 TP 验证输出正常,
  需核对旧机 vllm-plugin-FL/flag_gems 版本是否无此 patch 链或 dispatch 行为不同;
  d) 查 flag_gems 编译/执行是否走了 SIMT fallback 等降级路径(精度受损)。
- **参考**:VLLM_FL_PREFER=reference 不可用(plugin 代码 NameError:
  reshape_and_cache_flash 未定义,reference.py 侧 bug)

### 4b. flag_gems 慢 ~20× 的 AIC kernel 是否禁用(走 aclnn)

- flag_gems linear 大 shape:首次 13s(编译),热 ~200ms;torch 原生 10ms
- 若 4a 定位到 flag_gems AIC 路径不可救,考虑 dispatch policy per-op 禁
  flag_gems 的 linear/matmul(该类算子 flagos 上 aclnn 原生更快更准)

## 5. 补丁体系化(用户要求交付,已完成)

见 `dev/memory/docs/archive/patches/README.md`:
- P1 flagos_boot.py(仓内权威副本 = probes/flagos_boot.py,本次 7+ 处新增均有注释锚点)
- P2 triton npu_utils.cpp(USE_TORCH_NPU 段去 at_npu 依赖)
- P3 vllm base_loader.py(model.to target)
- P4 triton npu_utils.cpp workspace 悬空(static per-size 保活)
- 每项含:位置 / 改动 / 根因 / diff 归档路径 / 镜像固化 Dockerfile 指示
- **遗留**:Dockerfile 本体未落盘(README §2 为指示);P2/P4 是同一文件的两处改动,
  归档为相对 nightly 原版的单文件 diff(覆盖两者)

## 6. 环境快照

- 容器 flagos-fl-dev-910c;venv /root/vllm-venv312(py3.12.13)
- vllm 0.20.2;triton_ascend 3.2.2(USE_TORCH_NPU 段为 3.2.2 新增);torch_fl(ebc8762 编译);
  vllm-plugin-FL dev-1.0;模型 /workspace/models/Qwen3-4B
- 新探针:probes/qwen3_offline_tp.py(4 prompts,TP 参数化)、qwen3_mini_probe.py(单 prompt 诊断)、
  op_smoke.py / triton_smoke.py / triton_mm_smoke.py / matmul_compare.py / linear_shape_probe.py /
  linear_twice.py(算子级隔离用)
