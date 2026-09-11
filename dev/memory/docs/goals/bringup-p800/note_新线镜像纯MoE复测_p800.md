# 新线镜像纯 MoE 复测 — 2026-08-22

> 执行：xliu969（Hermes 协助）｜ 设备：P800（8× OAM，驱动 5.0.21.47）｜ 卡号：XPU 2（其余 1-7 空闲，仅 XPU0 被 zhenghaojia 容器占用）
> 容器：flagos-newline-moe（新线官方镜像，一次性，任务后 stop 保留）
> 镜像：harbor.baai.ac.cn/flagrelease-public/kunlunxin001-gems5.0.0-treenone-triton3.0.0-cx0.13.0-plugin0.2.0-vllm0.20.2-cp310-pt29-x64-v5.0.21.43:tele4
> 模型：/models/Qwen3-30B-A3B（= 宿主 /data2/xliu969/code/runtime-team/models/Qwen3-30B-A3B，61.1GB，Qwen3MoeForCausalLM 纯 MoE）
> 前置参考：[纯MoE-昆仑芯-20260822.md](archive/纯MoE-昆仑芯-20260822.md)（旧线 4 参/5 参 TypeError 失败）、[新线镜像-MoE复测-静态预检-20260822.md](archive/新线镜像-MoE复测-静态预检-20260822.md)（gems 5.x 全系未补 renormalize）
> 本文档为昆仑芯 MoE 复测系列**结论主文档**，整合自《官方镜像复测-MoE》《纯MoE-昆仑芯》《新线镜像-MoE复测-静态预检》（均已归档 docs/archive/）
> 探针：dev/memory/probes/routeA_s3_offline.py（docker cp 至容器 /tmp/，3 prompts / max_tokens=64 / temp=0）

---

## D. 一句话判定（先给结论）

> **昆仑芯纯 MoE 在新线镜像上「解锁到可生成」：eager 模式完整跑通（EXIT=0，3.2 tok/s，生成 192 tokens），
> 旧线的 `flag_gems.topk_softmax` 4参/5参 TypeError 已被插件 0.2.0 的 dispatch 多后端降级机制绕开。
> 但：① 默认模式（graph capture）在昆仑芯上不可实际使用——FULL_AND_PIECEWISE 不被 KunlunxinAttentionBackend 支持，
> 自动降级 PIECEWISE 后 51 个 capture size 每个 ~42s（全程约 35 分钟），900s 超时未完成；② eager 生成文本质量差
> （首 token 正确，如 "Paris"/"John"，随后退化为重复/乱码），expert GEMM 来自厂商 xtorch_ops 内核，正确性未过关；
> ③ 使用旧线 env 口径 `VLLM_FL_PREFER=flagos|vendor` 会触发**新的崩溃点**：flag_gems 5.0.0 wheel 注册的
> `_moe_C::moe_align_block_size` 为 6 参 schema（旧 vllm 签名），vllm 0.20.2 以 7 参调用（新增 `expert_map`）→ RuntimeError。
> 即：**从「必崩」推进到「能跑、能生成、但质量与默认模式不可用」，MoE 内核本身（fused_experts）首次在昆仑芯触达并执行。**

---

## A. 栈版本基线（容器内实测，非镜像标签推断）

| 组件 | 版本 | 安装方式 | 备注 |
|---|---|---|---|
| flag-gems | **5.0.0** | wheel（site-packages/flag_gems/） | kunlunxin 后端 `topk_softmax` 仍 **4 参**（runtime/backend/_kunlunxin/fused/topk_softmax.py:56），与静态预检一致 |
| vllm-plugin-fl | **0.2.0+g38e7dbc** | editable（.pth → /workspace/vllm-plugin-FL/） | dispatch 架构（flaggems/vendor/reference 三后端） |
| vllm | **0.20.2** | site-packages | V1 引擎，modular MoE（MoEPrepareAndFinalize*），`vllm._C` 未编译（ModuleNotFoundError，插件 `_patch_custom_ops` 兜底注册 schema） |
| triton | 3.0.0+f69a5406 | site-packages | |
| flagcx | 0.13.0 | site-packages | |
| torch | 2.9.0+cu129 | site-packages | 含 torch_xmlir（symbrewrite） |
| xtorch-ops | 0.1.2935+50a5d6a4 | site-packages | **MoE expert GEMM 厂商内核来源** |
| torch-plugin / xmlir | 0.1.0 / 1.0.0.1 | site-packages | |

插件加载机制（vllm 0.20.2）：**仍是 `VLLM_PLUGINS` 环境变量**（无 `--plugin` CLI 参数）。
入口点注册于 pyproject.toml：`vllm.platform_plugins: fl = "vllm_fl:register"` + `vllm.general_plugins: fl = "vllm_fl:register_model"`，
经 vllm/plugins/__init__.py 的 `load_plugins_by_group` 按 `VLLM_PLUGINS` 白名单加载（default group `vllm.general_plugins`）。
`register()` 做 vendor 初始化 + custom-op schema 兜底 + 平台 patch；`register_model()` 里 `register_router()`
将 vllm 原生 FusedTopKRouter/GroupedTopKRouter/FusedTopKBiasRouter 的 `_compute_routing` 猴子补丁为 FL 版本
（vllm_fl/ops/fused_moe/router.py:257-261），使 topk 路由走 CachedOp dispatch。

---

## B. 插件 0.2.0 自适应/降级机制（实测修正静态预检的理解）

### B.1 静态预检预期 vs 实测

- **静态预检预期**：插件 0.2.0 靠 `topk_softmax_flaggems` 的 try/except（5 参→4 参+手动 renorm）自适应救场。
- **实测修正**：`kunlunxin.yaml` 的 `flagos_blacklist` **显式包含 `topk_softmax`**
  （vllm_fl/dispatch/config/kunlunxin.yaml:85），flaggems 实现被策略层禁用、从未执行。
  **真正救场的是 dispatch manager 的多后端降级**（kunlunxin.yaml `strict: false`），运行日志实证：
  ```
  Op 'topk_softmax' using 'vendor.kunlunxin' (kind=vendor, vendor=kunlunxin)
  Op 'topk_softmax' switched from 'vendor.kunlunxin' to 'reference.torch' (kind=reference, vendor=None)
  ```
  vendor.kunlunxin 的 topk_softmax 实现（4 参 flag_gems 调用）失败 → 自动切 reference.torch 纯 torch 实现 → 通过。
  默认模式与 eager 两次运行日志完全一致。

### B.2 插件 0.2.0 代码层自适应（存在但本次未触达）

flaggems 后端 impl 中确有 try/except 自适应（vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py:56-71）：
```python
def topk_softmax_flaggems(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False):
    from flag_gems import topk_softmax
    try:
        topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize)  # 5 参
    except:
        topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)                # 4 参
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)                     # 手动归一化
    return topk_weights, topk_indices
```
**代码质量备注（可作 issue 附注）**：`topk_softmax_flaggems` 在同一文件 **L56 与 L75 重复定义两遍**（逐字节相同，
Python 以后定义者生效，无功能影响，属重复代码/合并残留）。

### B.3 dispatch 架构变化（0.1.0 → 0.2.0）

- 0.1.0：`_fl_ops.py` 无条件 5 参调 flag_gems.topk_softmax（monkey-patch vllm FusedMoE.select_experts）→ 4 参后端必崩。
- 0.2.0：vllm_fl/dispatch/ 三后端（flaggems / vendor / reference）+ CachedOp 缓存 + 策略文件
  （kunlunxin.yaml：per-op 后端顺序 + flagos_blacklist + strict:false 自动降级）+ `replace_router_with_fl` 路由层 patch。
  topk_softmax 的 4/5 参签名差异被「vendor 失败 → reference 兜底」吸收，**不再 TypeError**。

---

## C. 运行结果

### C.1 预检口径 `VLLM_FL_PREFER=flagos|vendor`（旧线 env 口径，暴露新崩溃点）

配置：`CUDA_VISIBLE_DEVICES=2 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=flagos|vendor
USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 DO_NOT_TRACK=1`，默认模式（enforce_eager=False）

**❌ EXIT=1，但失败点已前进：topk_softmax 通过，崩在 fused_experts 前置的 moe_align_block_size**

```
RuntimeError: _moe_C::moe_align_block_size() expected at most 6 argument(s) but received 7 argument(s).
Declaration: _moe_C::moe_align_block_size(Tensor topk_ids, int num_experts, int block_size,
Tensor(a!) sorted_token_ids, Tensor(b!) experts_ids, Tensor(c!) num_tokens_post_pad) -> ()
```

崩溃链路：qwen3_moe forward → self.experts（FusedMoE）→ moe_runner._forward_impl →
unquantized_fused_moe_method → modular_kernel → vllm fused_moe.py:2062 `_prepare_expert_assignment` →
vllm moe_align_block_size.py:90 `ops.moe_align_block_size`（**7 参**，含 `expert_map`）→
vllm/_custom_ops.py:2353 → `torch.ops._moe_C.moe_align_block_size`（**6 参 schema**）→ RuntimeError

**根因（新失败点，flag_gems wheel 与 vllm 0.20.2 的 API 漂移）**：
- flag_gems 5.0.0 `patches/patch_util.py:57-60` 注册 `_moe_C::moe_align_block_size` 为 **6 参** schema
  （vllm ≤0.1x 签名，无 `expert_map`；vllm._C 未编译时由该兜底定义生效）；
- vllm 0.20.2 `_custom_ops.py:2353-2360` 以 **7 个位置参数**调用（新增 `expert_map: Tensor | None`）；
- 触发条件：`VLLM_FL_PREFER=flagos|vendor` 时 `use_flaggems()`（vllm_fl/utils.py:93-95，要求 PREFER 恰为
  "flagos" 或空）返回 False → `select_unquantized_moe_backend_oot`（fused_moe_utils.py:219-222）不选
  `TritonExpertsFL` → fused_moe 落回 **vllm 原生路径** → 撞 7参/6参签名不匹配。
- 结论：**该崩溃是「旧线 env 口径 × 新线组件」的兼容性问题；新线正确口径（见 C.2）下 fused_moe 走插件路径，
  完全绕开此调用**。但 flag_gems 5.0.0 的 6 参 schema 与 vllm 0.20.2 7 参调用本身就是待修漂移
  （任何落回原生 fused_moe 的配置都会撞上），值得作为 issue 提交厂商。

### C.2 新线口径 `VLLM_FL_PREFER=flagos` — 默认模式（enforce_eager=False）

配置：`... VLLM_FL_PREFER=flagos ...`，其余同 C.1。

**⏱ EXIT=124（900s timeout），但引擎初始化全链路已通过，卡死在 CUDA graph capture**

| 阶段 | 耗时 | 结果 |
|---|---|---|
| 权重加载（16 shards） | 13.39s | ✅ |
| torch.compile + profile_run 内存摸底（**旧线崩溃点，本次通过**） | 214.04s（含首次编译） | ✅ MoE 前向实际执行通过（topk_softmax 降级 reference.torch） |
| KV cache 分配 | — | ✅ 227,472 tokens / 5.55x 并发 |
| CUDA graph capture | 未完成 | ⚠️ 见下 |

graph capture 关键日志：
```
WARNING CUDAGraphMode.FULL_AND_PIECEWISE is not supported with KunlunxinAttentionBackend backend
        (support: AttentionCGSupport.NEVER); setting cudagraph_mode=PIECEWISE
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 2/51 [01:24<34:21, 42.08s/it]
```
即：昆仑芯 + 插件 attention 后端不支持整图捕获 → 自动降级 PIECEWISE → 51 个 capture size × ~42s ≈ **35 分钟**，
900s 预算内只完成 2/51。**默认模式（graph）在昆仑芯当前不可实际使用**（时间成本维度），与旧线「graph capture 引擎
初始化即崩」性质不同——本次是 capture 能跑但极慢。

### C.3 新线口径 `VLLM_FL_PREFER=flagos` — eager 模式（S3_ENFORCE_EAGER=1）

**✅ EXIT=0 —— 昆仑芯纯 MoE 首次完整跑通并生成**

| 指标 | 数值 |
|---|---|
| 加载耗时（含模型初始化，跳过 compile/graph） | 163.5s |
| 加载后显存 | used 95.34GB / total 103.08GB（free 7.74GB） |
| 生成 | 3 prompts × 64 tokens = 192 tokens，60.65s，**3.2 tok/s**（3 请求共享批处理，单卡 eager） |

生成文本（temperature=0.0）：
```
Prompt: 'Hello, my name is'
Generated: ' John, my, I, I, I, I, I, I, I, I, I, I, III, I, I... III, I, III, I, III, III, III, I, III[\nI'

Prompt: 'The capital of France is'
Generated: ' Paris, and theII, theII[\nThe The capital, and, and, andII[\n, andII[\nThe, andII...'

Prompt: '量子计算的基本原理是什么？请简要说明。'
Generated: ' ...\n\n### 什么是。 什么是？ ？ ？ ？@ ？@ ？@ ？@ 20. 1. 20 20 ？@ 20 ？ ？ ？ ？ 2'
```
**质量判定**：首 token 正确（"John"、"Paris"），随后迅速退化为 token 重复/乱码 —— 链路与内核**可执行**，
但**生成正确性未过关**（与混合架构模型 eager 乱码问题同类别：能跑、输出不对）。未在本次范围深挖（需逐层
定位 attention/KV/softmax/expert GEMM 数值，建议后续用 reference 全后端对照或单层数值比对）。

dispatch 归属（eager 日志实证）：
- attention_backend：vendor.kunlunxin（KunlunxinAttentionBackend，CUSTOM attention 注册）
- rms_norm / rotary_embedding：default.flagos（flag_gems）
- **topk_softmax**：vendor.kunlunxin →（失败）→ **reference.torch**（纯 torch 兜底，4/5 参之争终结于此）
- **fused_experts（expert GEMM）**：插件 OOT 后端 TritonExpertsFL（`select_unquantized_moe_backend_oot`
  在 out-of-tree + use_flaggems 时选定），其 `fused_experts_impl` 被 vendor patch 替换为
  **kunlunxin `_klx_fused_experts`（xtorch_ops 厂商内核）**：
  gen_block_statistic → moe_pre_sorted → moe_fc(w1) → silu → moe_fc(w2) → post（权重求和）
  （vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py:160-169 + impl/fused_moe/fused_moe.py:26-）
- moe_align_block_size：插件路径 `_prepare_expert_assignment`（fused_moe_utils.py:255-295）走
  CachedOp/naive 分支，**未触达** vllm 原生 7 参调用

---

## E. 结论与剩余风险

### E.1 结论（对照旧线）

| 维度 | 旧线（4.2.1rc0 + 0.1.0 + vllm 0.13） | 新线（5.0.0 + 0.2.0 + vllm 0.20.2） |
|---|---|---|
| topk_softmax | ❌ TypeError（5参调4参），profile_run 即崩 | ✅ dispatch 降级 reference.torch，默认+eager 均通过 |
| fused_experts / expert GEMM | 未触达（topk 先行崩溃） | ✅ **首次触达并执行**（xtorch_ops 厂商内核） |
| 默认模式 | 崩于初始化 | ⚠️ 初始化通过，graph capture PIECEWISE ~35min 不可实际使用 |
| eager 生成 | 无 | ✅ 跑通，3.2 tok/s，**但输出退化（首 token 正确后重复乱码）** |
| 一句话 | 必崩 | **链路解锁（可生成），正确性与默认模式可用性未过关** |

### E.2 剩余风险（按严重度）

1. **生成质量**：eager 输出首 token 正确后退化重复 —— **已于 2026-09-01 定位根因**：
   厂商插件 `patch_decode_attention`（decode_paged_attention→prefill_attention prefix_cache，
   无条件应用）为退化源，与 expert GEMM 无关（纯 torch 参考 A/B 同样乱码，dense 模型同退化）；
   禁用补丁后生成正常、解码提速近 2x。详见 [新线栈decode生成退化-根因定位-20260901.md](新线栈decode生成退化-根因定位-20260901.md)。
2. **默认模式不可用**：KunlunxinAttentionBackend 声明 AttentionCGSupport.NEVER → FULL_AND_PIECEWISE 降级
   PIECEWISE，51 sizes × ~42s ≈ 35min capture；生产默认配置（enforce_eager=False）不可行，需
   限制 cudagraph capture size 或等待厂商图捕获优化。
3. **`_moe_C::moe_align_block_size` 6参 vs 7参漂移**（flag_gems 5.0.0 wheel vs vllm 0.20.2）：
   本测试用 `VLLM_FL_PREFER=flagos` 绕开（fused_moe 走插件路径），但任何落回 vllm 原生 fused_moe 的配置
   必崩 —— issue 素材：patch_util.py L57-60 注册 6 参 schema，建议补 `Tensor? expert_map=None` 参数。
4. **topk_softmax 依赖 reference.torch 兜底**：vendor 实现失败路径被策略吸收（strict:false），
   正确性 OK（纯 torch）但性能非最优；flag_gems kunlunxin 后端补 `renormalize` 后可将 topk_softmax
   移出 flagos_blacklist 走 flaggems 内核（静态预检 §6.3 的 issue 建议仍成立）。
5. **vllm._C 未编译**：靠插件 `register_op_schemas` 兜底 schema + flag_gems patch_util 注册 impl，
   稳定性依赖组件间签名对齐（本次已暴露 6/7 参一类漂移风险）。
6. 性能未测：3.2 tok/s 为 eager 单卡 3 请求共享批处理，无 graph、无 TP、无性能调优意图。

---

## F. 原始证据位置与复现命令

### 日志（容器 /logs/ = 宿主 dev/memory/benchmarks/out/）
- `newline_pure_moe_default.log` —— C.1 旧口径默认模式（moe_align_block_size 6/7 参崩溃，完整 traceback）
- `newline_pure_moe_default_prefer_flagos.log` —— C.2 新口径默认模式（init 全通过，graph capture 超时 EXIT=124）
- `newline_pure_moe_eager.log` —— C.3 新口径 eager（EXIT=0，生成文本 + dispatch 日志）

### 复现命令
```bash
# 容器（一次性命名 flagos-newline-moe，注意镜像 Entrypoint=/bin/bash，须用 --entrypoint 覆盖或直接传命令）
docker run -d --name flagos-newline-moe --network host --ipc host --shm-size 512g \
  --device /dev/xpu0:/dev/xpu0:rwm ... --device /dev/xpu7:/dev/xpu7:rwm --device /dev/xpuctrl:/dev/xpuctrl:rwm \
  -v /data2/xliu969/code/runtime-team/models:/models:ro \
  -v /data2/xliu969/code/runtime-team/dev/memory/benchmarks/out:/logs \
  -e TZ=Asia/Shanghai --entrypoint /bin/bash \
  harbor.baai.ac.cn/flagrelease-public/kunlunxin001-gems5.0.0-treenone-triton3.0.0-cx0.13.0-plugin0.2.0-vllm0.20.2-cp310-pt29-x64-v5.0.21.43:tele4 \
  -c 'sleep infinity'
# 探针
docker cp dev/memory/probes/routeA_s3_offline.py flagos-newline-moe:/tmp/
# eager（新线正确口径）
docker exec flagos-newline-moe bash -lc 'source /root/miniconda/bin/activate python310_torch29_cuda
export CUDA_VISIBLE_DEVICES=2 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=flagos \
  USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 DO_NOT_TRACK=1 S3_ENFORCE_EAGER=1
S3_MODEL=/models/Qwen3-30B-A3B python -u /tmp/routeA_s3_offline.py'
# 默认模式：去掉 S3_ENFORCE_EAGER=1（预期卡 graph capture ~35min）；旧口径对照：VLLM_FL_PREFER=flagos|vendor（预期 moe_align_block_size 6/7 参崩溃）
```

### 环境变量结论
- vllm 0.20.2 插件加载：**`VLLM_PLUGINS=fl` 环境变量**（无 --plugin 参数），入口 `vllm_fl:register` / `vllm_fl:register_model`
- **env 口径变更**：插件 0.2.0 要求 `VLLM_FL_PREFER=flagos`（或留空），**旧线 `flagos|vendor` 会令 use_flaggems()=False，
  使 fused_moe 落回 vllm 原生路径并触发 moe_align_block_size 6/7 参崩溃**——复测必须用新口径。

## 清理状态
- 容器 flagos-newline-moe 已 stop（未删除，保留供复查）；日志已落盘 /logs（宿主 dev/memory/benchmarks/out/）。
- 未修改任何仓库文件（仅新增本文档）；全程无 git 操作。
