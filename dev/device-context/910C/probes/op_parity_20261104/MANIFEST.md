# 910C 算子库逐算子对照证据（2026-11-04）

> 目的：把「算子库只有端到端间接证据」补成逐算子实测。正文见 `docs/CHIP_COMPAT_MATRIX_20261104.md` **§3.1d**。
> 环境：`dc-probe-910c-1104`（镜像 `flagrt/ascend-operator-runtime-comm:2.0.0-…-cann9.0-…-torch2.10-…`）｜`ASCEND_RT_VISIBLE_DEVICES=2`｜`torch 2.10.0+cpu` / `torch_npu 2.10.0` / `flag_gems 0.0.0+f7ae8e6b934a` / triton `3.5.1`（经 `flagtree 0.6.0+ascend.git15ec1a6c`）

## 一、逐算子对照（`op_parity_ascend.py`）

做法：vendor 原生路径跑真实前向（Qwen3-Embedding-0.6B，3 条文本）→ `TorchDispatchMode` 记录**实际执行的每个 aten 算子**及输入/输出 →
`flag_gems.enable()` 后**用同一输入重放** → 与 vendor 输出比对（`rel` + `allclose(1e-3)`）。
纪律：in-place 算子跳过；每算子限 3 个签名；记录预算 600 MB。

| 项 | 读数 |
|---|---|
| 唯一 aten 算子 | **27** |
| 重放条数 | **304** |
| 结论 | **PASS 184 / FAIL 0 / SKIP 120** |
| 覆盖且数值一致 | **18 个算子**（`slice` `neg` `view` `t` `mul` `unsqueeze` `transpose` `rsqrt` `mean` `matmul` `lift_fresh` `expand` `cat` `all` `add` `pow` `sin` `embedding`） |

## 二、SKIP 的三类（混写会误判）

| 类别 | 条数 | 定性 |
|---|---|---|
| `silu`(22) + `lift_fresh`(56) 重放报错 | 78 | **探针自身姿势问题**：`op_min.py` 证明两者在 FlagGems 下**正常** ⇒ 不计缺陷 |
| `triu`(21) `CompilationError` | 21 | **确证缺陷**：`op_min.py` 可独立复现（vendor 正常） |
| `npu.npu_fusion_attention_v3`(20) | 20 | flag_gems 未覆盖（厂商融合算子）⇒ vendor 回落，属预期 |

## 三、端到端：启用 FlagGems 后前向直接失败（`gems_forward.py`）

```
[gems-forward] FAILED  CompilationError: at 44:4:
   ModuleNotFoundError("No module named 'shmem'")
```
- 归因：`shmem` 在 `flag_gems/**.py` 与 `triton/language/**` **均搜不到** ⇒ 由 **flagtree/triton-ascend 编译期**引入，属**后端栈依赖缺口**。
- 镜像内无该包、无 wheel；本机**无外网**（`pypi` `http=000`）。

## 文件

| 路径 | 说明 |
|---|---|
| `op_parity_ascend.py` | 探针本体（记录 + 重放 + 比对；可复用到其它家） |
| `op_min.py` | **最小独立复验**：区分「重放失败」与「kernel 失败」 |
| `gems_forward.py` | 启用 FlagGems 后跑真实前向 |
| `out/console.log` | 探针控制台（含 `[env]` / 计数 / `[summary]`） |
| `out/op_parity.tsv` | **逐算子逐条**结果（`op / covered / rel / verdict / sig / note`） |
| `out/summary.json` | 机读摘要（含 `record_skipped_by_reason` 与 top ops） |
| `out/op_min.log` | 最小复验输出（`gems.silu`/`lift_fresh` 正常、`gems.triu` 失败） |
| `out/gems_forward.log` | 启用后前向失败的完整栈 |
