# 推理执行记录：Qwen3-4B TP 数值等价性验证（O1-O3 + TP=4）

- **日期**：2026-09-01
- **分支**：kistich/device-context（A 线，torch_npu + vLLM + vllm-ascend）
- **目标**：对齐同事 `xliu969/common-compose-align` 历史开发记录（模型/镜像/实验顺序），
  以 Qwen3-4B 为基准模型复现"TP=1/2 逐字一致"结论，并对照 B 线两条缺陷
- **状态**：✅ 全部通过

---

## 1. 结论先行

| 验证项 | 结果 | 对照对象 |
|---|---|---|
| O1 模型对齐 | ✅ Qwen3-4B 下载并校验（7.6G，dense） | 同事昇腾线基准模型 |
| O2 单卡冒烟 | ✅ `DENSE_INFER_PASS`，98.2 tok/s（greedy） | 同事 B 线加载 24.6s / P800 96.5 tok/s |
| O3 TP=2 数值等价 | ✅ `TP_COMPARE_PASS` 4/4 逐字一致 | 同事 B 线"TP=1/2 逐字一致" |
| TP=4 数值等价（附加） | ✅ `TP_COMPARE_PASS` 4/4 逐字一致 | 同事 B 线 TP=4 链路 |
| 缺陷① bool×int→embedding 查全 row1 | ✅ A 线不存在 | 同事 B 线 TP=2 专属缺陷 |
| 缺陷② flagcx 异步无同步→NaN | ✅ A 线不存在（无 NaN） | 同事 B 线缺陷，P2 已证，此处复核 |

**核心结论**：同构 910C A 线推理栈（torch_npu + vLLM 0.20.2 + vllm-ascend）在
TP=1/2/4 下数值完全等价（greedy 解码逐字一致），B 线两条设备上下文缺陷在 A 线均不存在。

---

## 2. 环境与对齐点

| 维度 | 同事历史（昇腾 B 线/A 线） | 本次（A 线） | 判定 |
|---|---|---|---|
| 模型 | Qwen3-4B（ModelScope，7.6G，`/workspace/models/Qwen3-4B`） | Qwen3-4B（hf-mirror，7.6G，`/mnt/raid/hliu553/models/Qwen3-4B`） | ✅ 同模型 |
| 推理栈 | A 线 = torch_npu + vllm 0.20.2 | torch 2.10.0+cpu + torch_npu 2.10.0 + vllm 0.20.2 + vllm-ascend（镜像 `vllm-ascend:v0.20.2rc1-a3`） | ✅ 方向一致 |
| 实验顺序 | 单卡 dense 闭环 → TP=2 数值对比 → TP=4 链路 | O2 单卡 → O3 TP=2 → TP=4 | ✅ 一致 |
| 关键环境变量 | — | **不设 `VLLM_PLUGINS=fl`**（见 §4 坑①） | 重要 |

模型 config 校验：`model_type=qwen3`、36 层、hidden 2560、vocab 151936、bf16（dense 4B，非 MoE）。

---

## 3. 执行结果（Qwen3-4B，seed=42，greedy 解码，max_tokens=64）

| TP | 加载(s) | 推理(s) | 吞吐(tok/s) | 判定 | 与 TP=1 逐字对比 |
|---|---|---|---|---|---|
| 1 | 14.8 | 2.6 | **98.2** | `DENSE_INFER_PASS` | 基准 |
| 2 | 21.9 | 3.5 | 72.3 | `DENSE_INFER_PASS` | **4/4 一致**（text+token_ids） |
| 4 | 22.4 | 6.1 | 42.1 | `DENSE_INFER_PASS` | **4/4 一致**（text+token_ids） |

- NaN 复核：TP=1/2/4 均无 NaN（缺陷②对照通过）
- 输出语义正常、无乱码（缺陷①对照通过）
- 性能观察：小模型（4B）+ 小 batch（4×64 tokens）下 TP 越大吞吐越低（通信开销 > 计算收益），符合预期；
  TP 验证关注数值等价性，不关注吞吐放大
- 对照：同事 B 线 Qwen3-4B 加载 31.89GiB/24.6s、KV 占 76%（我们加载 15-22s，同量级）

结果文件（`benchmarks/inference/results_qwen3_tp_compare/`）：
- `dense_qwen3_tp1_greedy.json` / `dense_qwen3_tp2_greedy.json` / `dense_qwen3_tp4_greedy.json`（greedy，正式结论）
- `dense_qwen3_tp1.json` / `dense_qwen3_tp2.json`（随机采样对照，FAIL 属预期，见 §4 坑②）
- 日志：`log_qwen3_tp_greedy_20260901.log` / `log_qwen3_tp4_greedy_20260901.log`

---

## 4. 两个关键坑（务必记录，后续复用）

### 坑① A 线禁止设置 `VLLM_PLUGINS=fl`

- **现象**：设置 `VLLM_PLUGINS=fl` 后 vLLM 报
  `RuntimeError: Device string must not be empty`（`current_platform.device_type` 为空）
- **根因**：vllm-plugin-FL 是 FlagOS 原生栈（torch_fl）插件，在 A 线（torch_npu + vllm-ascend）
  会干扰 ascend platform 的选择；P0 跑通时该变量实际为"未设置"（见 `dense_infer_result.json` env）
- **规则**：A 线容器内运行推理一律**不设** `VLLM_PLUGINS=fl`（保留 `DO_NOT_TRACK=1`）

### 坑② TP 逐字对比必须以 greedy 解码为对照

- **现象**：默认 `SamplingParams`（temperature=1.0 随机采样）下 TP=1 vs TP=2 输出 0/4 一致
  （前 7-14 token 一致后发散，语义均正常）
- **根因**：TP=2 下 logits 经 HCCL all_reduce，浮点累加顺序与 TP=1 不同 → 微小浮点差被随机采样放大
- **修复**：`SamplingParams(temperature=0.0)`（greedy，argmax 对微扰鲁棒）→ 4/4 一致
- **规则**：任何 TP 数值等价性对比，必须使用 greedy 解码 + 固定 seed（同事 B 线"逐字一致"同理）

---

## 5. 复现方法

```bash
# 容器内（flagos-infer-910c），脚本位于 raid
docker exec -it flagos-infer-910c bash

# TP=2 对比（O2+O3 一键：TP=1 → TP=2 → 逐字对比）
bash /mnt/raid/hliu553/run_qwen3_tp_compare.sh 2 greedy

# TP=4 对比
bash /mnt/raid/hliu553/run_qwen3_tp_compare.sh 4 greedy
```

脚本改动（`benchmarks/inference/`）：
- `dense_infer_qwen_1_5b_npu.py`：新增 `--tp` / `--seed` / `--greedy`；result 新增 `outputs_full`（完整 text + token_ids）
- `compare_tp_outputs.py`（新）：TP=1 vs TP=N 逐字对比，text + token_ids 双信号，NaN 复核，`TP_COMPARE_PASS/FAIL`
- `/mnt/raid/hliu553/run_qwen3_tp_compare.sh`（新）：一键串 O2→O3→对比，支持 `[N] [greedy]`

---

## 6. 待办

- [ ] git 提交脚本与文档改动（PR 目标 dev-1.0，与 FlagCX 缺陷修复合并呈现）
- [ ] 若需复现同事 B 线缺陷①/②的完整对照，可同步归档 B 线记录
