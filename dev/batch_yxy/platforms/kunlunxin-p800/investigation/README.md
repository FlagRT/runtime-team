# 昆仑芯 P800 数据（归档）

> 相关文档：缺陷排查全记录 [INVESTIGATION_20261117.md](INVESTIGATION_20261117.md)；
> score()/classify() 入口测试计划与现状 [SCORE_TEST_PLAN_STATUS_20261118.md](SCORE_TEST_PLAN_STATUS_20261118.md)。

| 项 | 值 |
| --- | --- |
| 平台 | 昆仑芯 P800 OAM ×8（本机 `xpu-smi` 驱动 5.0.21.47），测试使用单卡 `xpu2`（容器内相对索引 0） |
| 镜像 | `chenyunlong/vllm-kunlun-runtime:0.11.0-xpytorch20260226`（本机 docker 本地镜像） |
| 运行时 | vLLM 0.11.0 + vllm-kunlun 0.11.0 + kunlun_ops 0.1.58，torch 2.5.1+cu118，Python 3.10 |
| 模型 | Qwen3-Embedding-0.6B（本机 `/data1/flagperf-kzhang519/models/` ModelScope 副本，sha256 校验一致，只读；st 配置剔除的链接目录规避 vllm 补丁版本地路径误判 bug） |
| 关键开关 | `enforce_eager=True`；基准默认**未开** prefix caching；单卡；`XPU_VISIBLE_DEVICES=0` |
| 引擎配方 | `VLLM_PLUGINS=kunlun VLLM_ENABLE_V1_MULTIPROCESSING=0`（进程内引擎，规避 EngineCore 子进程平台绑定时序缺陷，见 §4） |
| 分支 | `main` + 本次 P800 适配（KUNLUN 画像登记、`vllm_compat.llm_model_config` 版本兼容、P800 验收脚本） |

## 1. 调度映射正确性（32 请求链路，`examples/sched_engine_p800_vllm.py`）

三到达顺序（inorder / reverse / interleaved）全部通过：**32/32 恰好一次回填**、
向量维度 1024、封口原因符合预期（batch_limit×3 + wait_limit×3）、断言零失败。
证据：`data/flagscheduler-32req-summary.json`。

数值参考：legacy vs scheduled 逐元素 0/32、cos mean 0.05 —— 受 §3 平台缺陷影响，
按 910C 口径仅作参考，不作为映射结论依据。

## 2. 组批矩阵（n=192 mixed，`benchmarks/bench_matrix_compare.py`）

口径与 910C 相同：A = 一次直连；B = 档位封口不分组；C = 档位 + 长度分桶；
每配置预热 1 轮、3 轮取中位数。`max_total_tokens=32768`。

| 配置 | req/s | e2e p50 | p95 | p99 | 批次 | bs p50 | padding | vs A max_abs / cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | **87.6** | 2192.2 | 2192.2 | 2192.2 | 1 次调用 | n/a | n/a | 基线 |
| B32 | 82.1（-6%） | 1632.4 | 2338.4 | 2338.4 | 6 | 32 | 0.562 | 0.64 / -0.066 |
| B64 | 83.0（-5%） | 1399.4 | 2214.6 | 2313.9 | 6 | 34 | 0.479 | 0.82 / -0.052 |
| B128 | 76.0（-13%） | 1562.4 | 2421.1 | 2526.6 | 6 | 34 | 0.479 | 0.82 / -0.052 |
| C32 | 80.7（-8%） | **381.0** | 2380.2 | 2380.2 | 12 | 17 | 0.115 | 0.70 / -0.066 |
| C64 | 81.2（-7%） | 383.8 | 2363.6 | 2363.6 | 10 | 17 | 0.085 | 0.70 / -0.066 |
| C128 | 75.1（-14%） | 488.3 | 2555.0 | 2555.0 | 9 | 17 | 0.093 | 0.70 / -0.066 |

- A 自身跨轮完全一致（max_abs=0.0，matched=192）。
- **吞吐**：外层组批不提升（-5%~-14%），方向与 910C 一致、幅度更小；B64/B128 被 32768 token 预算钳成与 B32 相同的 6 批（bs≈34），档位未真正拉开（与 910C 同现象）。
- **延迟**：C 分桶 p50 从 2192ms 降至 381ms（**-83%**，三平台中降幅最大）；padding 从 0.48–0.56 降至 0.085–0.115。
- **数值一致性**：所有 B/C vs A 的 cos 为负 —— 平台长短混批缺陷所致（§3），**不能**像 910C/NVIDIA 那样解读为"批组成漂移"。

n64 快检（`data/bench-vllm-mixed-n64.json`）与矩阵结论一致。

## 3. 平台缺陷：pooling 输出错误（2026-11-17 根因定位，两个独立缺陷）

证据：`probes/results/xpu_pooling_defects_20261117.json`（含全部判别实验数据）。

**先说映射**：4×4 交叉余弦矩阵与顺序交换实验证明**结果-请求对应始终正确**
（损坏跟随文本、不跟随槽位，无置换特征）——调度映射验收不受任何影响。

**缺陷一：prefix caching 命中路径**。`enable_prefix_caching=True`（vllm 0.11
默认开）时，KV 命中的长文本（多块）pooling 输出错误。最小复现：同一 300tok
文本连发两次，cos=0.546。CPU transformers 真值判定：冷缓存（真实 prefill）
cos=1.0 正确，命中路径 cos=0.546 错误。**规避：`enable_prefix_caching=False`。**

**缺陷二：跨 prefill 调度步**。单次 embed 调用总 token 超过
`max_num_batched_tokens`（16384）时，prefill 被切成多步，**第二个及以后
步的全部序列损坏**（铁证：16×2048 两步调用，步1 全对、步2 8/8 全坏）。
交叉比对证明损坏向量与批内一切真实嵌入近正交（cos 0.01~0.12，无任何
≈1.0 匹配）——**排除"读到其他序列 KV"的串扰假设**，指向"块表指向无效
内存、从垃圾数据计算"。触发受会话块分配布局影响（同形状调用在不同会话
历史下可全部正确）。与批内条数无关（单步内 64 条全对）；步数越多坏越多。
**规避：单次调用总 token ≤16384——FlagScheduler 外层组批的
`max_total_tokens` 预算控制天然满足。**

**安全包络（实测）**：`enable_prefix_caching=False` + 单次调用总 token ≤16384
时，64 条任意混长批逐位正确（cos=1.0）。

**影响面升级（2026-11-17 生成路径验证）**：两缺陷均非 pooling 专属——
同一权重改 `task=generate` 实测：① 缓存命中使同一长提示的贪心解码从第 1 个
token 分叉；② 多步批量调用 16/16 输出与单独调用完全分叉（0~1/32 重合，
排除 fp16 噪声；解码阶段跨步 KV 读取使步 1 序列也全坏）。**即 vllm serve +
默认 prefix caching 或任何多步连续调度下的生成输出同样不可信**，严重程度
高于此前仅按 embedding 评估的结论。

**对既往数据的解释**：n64/n192 基准的 B/C vs A 数值 FAIL 全部由缺陷二造成
（各路径单次调用均超 16384，各自确定性损坏→跨路径 cos 为负）；A 跨轮逐位
一致是"同构调用→同样步划分→同样错误"的表现；11-10 诊断中"短文本损坏"实为
参照调用命中缓存（缺陷一）导致的参照自身错误。性能数字（吞吐/延迟）的测量
路径与输出对错无关，仍然有效；数值一致性列在缺陷修复前不作数。
两缺陷待向昆仑芯独立上报，最小复现各 3 行。

## 4. 环境适配记录（供后来者复现）

1. **EngineCore 子进程起不来**：`KunlunPlatform` 未实现 `set_device`，且 spawn
   子进程内插件激活时序缺陷使 `current_platform` 落到基类 → `NotImplementedError`。
   规避：`VLLM_ENABLE_V1_MULTIPROCESSING=0`（进程内引擎）。
2. **本地模型路径误判**：vllm 0.11.0 `get_hf_file_to_dict` 对本地路径调
   `hf_hub_download` 且异常捕获列表缺 `ValueError`；模型目录带 sentence-transformers
   根配置时触发。规避：剔除 st 配置的符号链接目录（权重本体 sha256 不变）。
3. **选卡**：`XPU_VISIBLE_DEVICES`（昆仑芯官方栈），torch 设备经 `torch.cuda`
   重定向可见。
4. **仓库侧兼容修复**：`vllm_compat.llm_model_config()` 兼容 0.11 的
   `llm.llm_engine.model_config` 与 0.20+ 的 `llm.model_config`；单测 251 项全绿。
