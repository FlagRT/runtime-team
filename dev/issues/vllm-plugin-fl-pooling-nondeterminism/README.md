# 昆仑芯 pooling 输出数值不确定（根因未知）

## 问题与现状

- **一句话**：在已修复 [`vllm-plugin-fl-pooling-dispatch`](../vllm-plugin-fl-pooling-dispatch/README.md)
  （embed() 不再崩溃）的现行昆仑芯推理环境下，**同一句话重复调用 `llm.embed()` 两次，余弦相似度
  不是预期的 1.0**，偏差幅度远超 fp16 舍入噪声应有的量级。
- **归属方向**：未定。根因未知，不下结论归属——可能是厂商底层算子/attention 后端的非确定性，
  也可能是 `vllm-plugin-FL` 适配层的另一个缺口，没有深入排查。
- **状态（2026-11-22）**：**仍命中，根因未定位，无修复、无验收判据**。本 case 包目前只是
  现象记录，不是已有判据等待复验的那种 case。
- **不要和什么混为一谈**：这跟另一份不同报告
  （`inbox/kunlun芯显存问题排查.md`，袁先乙提交，厂商 `vllm-kunlun 0.11.0` 的两个缺陷之一
  ——prefix-caching 命中路径位置/偏移错乱）**不是同一机制**：那份报告是完全不同的软件栈
  （厂商自己的 `vllm-kunlun` 插件，不是我方现行用的 `vllm-plugin-FL`），而且本 case 已经
  显式排除了 prefix-caching 相关性（见下方实测），那份报告的缺陷至今未在现行栈上验证过。
  两者分开跟踪，不互相印证、不互相替代判读。

### 已排除的假说

- **跟 prefix-caching 机制无关**：显式 `enable_prefix_caching=False` 重测，cos 反而略低
  （0.9235 vs 默认开启时 0.9361~0.9416），不是"关掉缓存就恢复 1.0"，说明这个偏差不是由
  prefix-caching 命中引起的。

### 版本集合（分版表态）

| 坐标 | 状态 | 依据 |
|---|---|---|
| `kunlun-infer-vllm/v1`（锚点 digest `sha256:e180a0dd4267ebdfbce4ea5434710870a6ea7e96118457538cc02442907d54b1`，已打上 pooling-dispatch 补丁的环境） | **仍命中，根因未知** | 2026-11-22 P800-C 容器 `flagrt-kunlun-infer-v1-env-v1-p800a` 三组实测，见下方「复现」与 [`assets/evidence/pooling-cos-20261122.txt`](assets/evidence/pooling-cos-20261122.txt) |

## 复现

### 前置条件

与 [`vllm-plugin-fl-pooling-dispatch`](../vllm-plugin-fl-pooling-dispatch/README.md) 完全相同
（同一镜像、同一容器、同一模型），**额外前提**：容器内已经打上该 case 的一行补丁（否则
`llm.embed()` 会先崩在 dispatch 问题上，走不到这里）。

### 单入口命令（现象记录，不是判定脚本）

```bash
# 容器内，已 export 三个网络变量（同上一个 case）
python3 dev/issues/vllm-plugin-fl-pooling-nondeterminism/assets/record_pooling_cos.py \
  --model /home/models/Qwen3-Embedding-0.6B/

# 对照组：显式关闭 prefix caching
python3 dev/issues/vllm-plugin-fl-pooling-nondeterminism/assets/record_pooling_cos.py \
  --model /home/models/Qwen3-Embedding-0.6B/ --disable-prefix-caching
```

### 预期结果（现象，不是判定）

脚本只打印 `COS = <值>`，退出码 0 表示收数完成、2 表示执行出错，**不代表任何修复验收结论**。
2026-11-22 P800-C 三组读数：

| 场景 | cos | max_abs_diff |
|---|---|---|
| 同批次内两条相同文本 | 0.9361 | — |
| 两次完全独立调用（默认 `enable_prefix_caching=True`） | 0.9416 | 0.0509 |
| 两次独立调用，显式 `enable_prefix_caching=False` | 0.9235 | 0.0556 |

三组同量级（都在 0.92~0.94），预期应接近 1.0（确定性场景）。证据截选：
[`assets/evidence/pooling-cos-20261122.txt`](assets/evidence/pooling-cos-20261122.txt)。

## 修复验收

**根因未定位，暂无验收判据，待排查。** 不编造判据——这个 case 包目前的作用是把"现象 +
已排除的假说"留档，供下次有人接手排查或厂商确认时直接复现，不是一个可以判 PASS/FAIL 的
验收脚本。

| 脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据 |
|---|---|---|---|---|
| [`assets/record_pooling_cos.py`](assets/record_pooling_cos.py) | 收数 | 见「单入口命令」 | 不做判定；产物供排查引用 | 2026-11-22 P800-C：三组读数见上表，证据 [`assets/evidence/pooling-cos-20261122.txt`](assets/evidence/pooling-cos-20261122.txt) |

### 后续排查方向（未执行，供参考）

- 换一个 prompt/种子多测几次，确认偏差是否稳定复现同一数值（当前只测了 2~3 次，样本量不足）。
- 对比 CPU/其他非昆仑芯 backend 跑同一模型同一输入的 cos，排除是模型本身（如 pooling 层
  dropout 之类）的问题。
- 检查 kunlun_ops attention 后端是否有已知的非确定性归约顺序（类似 B15 昆仑芯 GEMM
  数值问题的性质，但未核实是否同一根因）。

<!-- changelog:
2026-11-22 —— 首次登记：[`vllm-plugin-fl-pooling-dispatch`](../vllm-plugin-fl-pooling-dispatch/README.md)
  修复验证过程中的附带发现。已排除与 prefix-caching 的相关性。根因未定位，暂不立项排查，
  留档供后续接手。
-->
