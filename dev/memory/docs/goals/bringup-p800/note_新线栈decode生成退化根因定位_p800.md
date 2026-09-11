# 新线栈 decode 生成退化 — 根因定位（2026-09-01）

> 执行：xliu969（Hermes 协助）｜ 设备：P800（XPU 2）｜ 容器：flagos-newline-moe（新线官方镜像，一次性）
> 镜像：kunlunxin001-gems5.0.0-treenone-triton3.0.0-cx0.13.0-plugin0.2.0-vllm0.20.2-…:tele4
> 上游：[新线镜像纯MoE复测-20260822](新线镜像纯MoE复测-20260822.md) E.2 剩余风险 #1（eager 生成质量退化）
> 本文档为 **issue #5（昆仑芯问题反馈清单）的根因定位结论主文档**

---

## 一句话结论

> **新线栈（vllm 0.20.2 + plugin 0.2.0）eager 生成的"首 token 正确、随后退化重复/乱码"，
> 根因是厂商插件 `patch_decode_attention`（patch.py:348-416）——它把 decode 阶段的
> `xtorch_ops.decode_paged_attention` 无条件替换为 `prefill_attention(is_prefix_cache=True)`。
> 该补丁本是为 Qwen3.6-27B layer 43+ 的 decode NaN 所写，但在 P800 新线栈上 decode 输出错误
> （疑似 prefix-cache 语义/自注意力处理问题），**与 MoE 无关、与 expert GEMM 无关**——
> dense 模型（Qwen3-4B）同样退化。禁用该补丁（恢复原生 decode_paged_attention）后
> dense 与纯 MoE 模型生成均恢复正常，且解码吞吐提升近一倍（dense 11.0 → 20.5 tok/s）。**

---

## A. 排查链（逐层隔离，全部实测）

| 步骤 | 实验 | 结果 | 结论 |
|---|---|---|---|
| 1 | 厂商 fused_experts（xtorch_ops）+ 厂商 decode 补丁 | ❌ 乱码（复现 08-22 报告） | 基线确认 |
| 2 | **纯 torch 参考 fused_experts**（EngineCore 内注入，[REF-MOE] 标记确认生效）+ 厂商 decode 补丁 | ❌ 同样乱码 | **expert GEMM 无罪**（参考实现数学等价仍退化） |
| 3 | dense 模型 Qwen3-4B（无 MoE）+ 厂商 decode 补丁 | ❌ 同样乱码（Dr,,,,/？函数？循环） | **与 MoE 无关**，decode 阶段通用问题 |
| 4 | dense Qwen3-4B + **禁用厂商 decode 补丁**（原生 decode_paged_attention） | ✅ 正常（"Paris. The capital of Germany is Berlin…"，20.5 tok/s） | **补丁 = 退化源** |
| 5 | 纯 MoE Qwen3-30B-A3B + 禁用厂商 decode 补丁 | ✅ 正常（"John. I am a student…"/量子计算正确阐述，13.6 tok/s） | **issue #5 根因闭环** |

关键对照（同模型、同 prompt、同 greedy 采样，仅切换 fused_experts / decode attention）：
- vendor fused_experts + 补丁：`'John, I, I, I, I, I, I, I, I, III…'`（13.1 tok/s）
- torch 参考 fused_experts + 补丁：`'John, and I am, I, I, I, I, I, I…'`（13.1 tok/s）
- vendor fused_experts + 无补丁：`'John. I am a student. I am 13 years old. I am from China…'`（13.6 tok/s）

> 注：步骤 2 与 1 的乱码"细节不同、模式相同"——fused_experts 实现差异会传导到输出，
> 但退化模式由上游（decode attention）决定，两种实现都救不回来。

## B. 根因细节：patch_decode_attention（厂商 NaN 工作补丁）

- 位置：`/workspace/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py:348-416`
- 动机（代码注释）：`xtorch_ops.decode_paged_attention produces NaN on certain layers during
  decode (observed on layer 43+ of Qwen3.6-27B). Using prefill_attention with is_prefix_cache=True
  provides correct results.`
- 应用方式：`apply_kunlunxin_patches()`（patch.py:46）**无条件调用**，对全部模型生效；
  替换对象是 `KunlunxinPagedAttention.forward_decode`（静态方法，vllm_fl/…/attention.py:625 原实现）
- 替换后的 decode 实现：`prefill_attention(is_causal=True, is_prefix_cache=True, …)`，
  以 `kv_prefix_start_loc = cumsum(seq_lens)` 描述每个 decode 请求的已缓存 KV 长度，
  以 `block_table` 传 paged 块表，`alpha=scale`
- **疑似错误点（待厂商确认）**：prefix-cache 语义下 query 的 KV 长度/自注意力位置处理——
  若 kv 长度把当前 decode token 自身计入（seq_lens 含新 token 的 KV），query 会自注意力，
  每步 decode 输出被自身 KV 污染，形成"首 token 对、后续逐步退化"的典型模式；
  同时该路径比原生 decode_paged_attention 慢近一倍（dense 11.0 vs 20.5 tok/s）
- **修复建议（给厂商）**：① 恢复原生 `decode_paged_attention` 作为 P800 默认 decode 路径
  （本机实测 Qwen3-4B/Qwen3-30B-A3B 均无 NaN，原生路径正常）；② 若 Qwen3.6-27B 等模型确实
  需要该补丁，应**按模型/按条件启用**（如检测到 NaN 层才降级），而非无条件全局替换；
  ③ 补丁内 prefix-cache 语义（kv_prefix_start_loc / 自注意力）需复核修正

## C. 附带发现（实验过程中的坑，已固化为探针注释）

1. **EngineCore 是独立子进程**：主进程 monkeypatch 不生效（vllm 0.20.2 V1 引擎）。
   A/B 注入必须改插件源码（容器内编辑，改前备份 .orig_bak）——见
   [probes/routeA_s3_moe_ab.py](../probes/routeA_s3_moe_ab.py) 与 [probes/ref_moe_impl.py](../probes/ref_moe_impl.py)
2. **flag_gems bmm autotuner 在 EngineCore 上下文内 ZeroDivisionError**（do_bench estimate_ms=0，
   CUDA event 计时异常，与 README「event-timing 回退」已知问题同族）；einsum 会被降级到该路径
3. **flag_gems nonzero triton 内核 error 719 launch failure**（EngineCore 上下文内）；
   torch.nonzero/掩码索引在引擎内不可靠 → 参考实现改用 **CPU 侧 numpy 排序 + GPU 侧
   2D mm（稠密推理同路径）/ gather / index_add_**，全部为引擎内已验证算子
4. **vllm 0.20.2 无内置纯 torch MoE 参考**（UnquantizedMoeBackend.TORCH 已移除），
   A/B 参考实现需自研（本子方向已固化在 probes/ref_moe_impl.py）
5. **权重布局**：Qwen3-30B-A3B w1=[E, 2F=1536, H=2048]（中间维 2F），w2=[E, H, F]——
   与 vLLM 常见 einsum 写法（末维 2F）相反，参考实现首版因此形状报错

## D. 复现与验证命令

```bash
# 容器内禁用补丁（一次性容器 flagos-newline-moe；改前自动备份 .nanpatch_bak，恢复 = cp 回去）
P=/workspace/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py
cp $P $P.nanpatch_bak
# 把 patch.py:46 的 patch_decode_attention() 注释掉（sed 或手改）
# 验证: EngineCore 日志不再出现 "Patched KunlunxinPagedAttention.forward_decode"

# dense 对照（Qwen3-4B）
docker cp dev/memory/probes/routeA_s3_offline.py flagos-newline-moe:/tmp/
docker exec flagos-newline-moe bash -lc 'source /root/miniconda/bin/activate python310_torch29_cuda
export CUDA_VISIBLE_DEVICES=2 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=flagos USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 DO_NOT_TRACK=1
S3_MODEL=/models/Qwen3-4B S3_ENFORCE_EAGER=1 python -u /tmp/routeA_s3_offline.py'

# 纯 MoE 对照（Qwen3-30B-A3B，issue #5 目标模型）
S3_MODEL=/models/Qwen3-30B-A3B S3_ENFORCE_EAGER=1 python -u /tmp/routeA_s3_offline.py
```

## E. 日志与产物

- `benchmarks/out/moe_ab_vendor.log` —— 步骤 1（vendor fused_experts + 补丁，乱码基线）
- `benchmarks/out/moe_ab_reference6.log` —— 步骤 2（参考 fused_experts + 补丁，[REF-MOE] 标记）
- `benchmarks/out/dense_qwen3_4b_newline.log` —— 步骤 3（dense + 补丁，乱码）
- `benchmarks/out/dense_qwen3_4b_nopatch.log` —— 步骤 4（dense 无补丁，正常 ✅）
- `benchmarks/out/moe_nopatch.log` —— 步骤 5（MoE 无补丁，正常 ✅）
- 参考实现与 A/B 探针：`probes/ref_moe_impl.py`、`probes/routeA_s3_moe_ab.py`

## 影响与后续

- issue #5 状态：**根因已定位**（厂商插件 patch_decode_attention），等待厂商修复或确认疑似点；
  子方向侧已有可用规避（禁用补丁），但注意**这是容器内一次性修改**，正式修复须走厂商 issue
- 08-22 报告中 E.2 剩余风险 #1（生成质量）关闭；#2（graph capture 35min）仍开放；
  #3/#4（moe_align_block_size 6/7 参漂移）不受影响
