# 910C 生成式显存画像报告（OneRec-8B）

> 状态：🟢 **冒烟 + 加载基线 + decode 增长 + 归还闭环 + 输入/并发对照 + gmu 对照已实测**（2026-09-23，npu1-27 davinci1/Phy-1）｜ 负责人：chenyuanxin
> 对应：战略文档 §3 验收标准第 4 条、§5 memory 第 1 条（生成式补测）；对照基线：[profile_显存画像_910c.md](profile_显存画像_910c.md)（embedding，2026-09-10）
> 模型：`OpenOneRec/OneRec-8B`（revision `29f95b3d…40ff`，Qwen3ForCausalLM，bf16）；model_list 此前无生成记录，本报告补上首次生成验证。
> 服务入口：`serve_standard.sh`（kistich/device-context，DC_BACKEND=ascend，TP=1，MAX_MODEL_LEN=4096，EAGER=1）；910C 标准环境不启用 FL（口径见 STATUS 路线节）。
> 探针：`probes/910c/gen-load_910c.py`（精确长度请求驱动 + 并发/取消 + 1s /metrics KV 采样）、`gen-matrix_910c.sh`（编排）、`gen-parse_910c.py`（结算行解析），复用 `hbm-sampler_910c.py`（宿主机）。
> 数据来源：`benchmarks/out/gen910c/`（37 个文件：HBM CSV ×4、KV CSV ×14、load JSON ×14、vllm.log ×2 等）。

---

## 1. 环境（标准环境，与 embedding 画像同栈）

| 项 | 值 | 来源 |
|---|---|---|
| 镜像 / 容器 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` / `mem-profile-910c`（仅挂 davinci1） | 实测 |
| 设备 | davinci1（宿主 npu-smi NPU 0 / Chip 1），容器内 `ASCEND_RT_VISIBLE_DEVICES=0`；idle HBM 2892 MiB | 实测 |
| 模型权重 | `/data_lib/models/OneRec-8B`（共享盘，root:root 0755，自下载后 sudo 移入；model_list 待回写此路径） | 实测 |
| 关键运行参数 | gmu=0.9（F 组 0.4）、max_model_len=4096、enforce_eager=1、TP=1 | serve_standard.sh |
| HBM 采样 | 宿主机 hbm-sampler，~2s 分辨率（容器内无 npu-smi） | 实测 |
| KV 指标 | `/metrics` 轮询 `vllm:kv_cache_usage_perc`（1s），与 HBM 双口径同步采集 | 实测 |

## 2. 方法（新增点，相对 embedding 画像）

- **双指标同步**：设备 HBM（npu-smi，池外真值）+ 引擎 KV 使用率（/metrics，池内动态）。embedding 时期只有前者，无法观测池内占用。
- **精确长度驱动**：gen-load 按 tokenizer 构造指定 token 数的 prompt，响应 usage 实测为准（构造值与实测差 +4，special tokens）。
- **受控变量**：输出长度（C）、输入长度（B）、并发（D）、取消（E）、gmu（F），每组独立起服务。

## 3. 冒烟（首次生成验证）

`SERVE_STANDARD_PASS (ready=1 smoke=1)`：服务就绪 40s，`/v1/completions` 生成 8 tokens。

## 4. A 组：加载基线（gmu=0.9）

| 项 | OneRec-8B | 对照：Qwen3-4B | 对照：Embedding-0.6B |
|---|---|---|---|
| 权重 | **15.66 GiB**（bf16 8B） | 7.52 GiB | 1.13 GiB |
| KV 池预分配 | **39.23 GiB / 285,568 tokens** | 47.42 GiB / 345,216 tok | 53.79 GiB / 503,552 tok |
| 最大并发（4096 tok/req） | **69.72×** | 84.28× | 61.47× |
| peak activation（vLLM 结算，合成估计） | 0.24 GiB | 0.18 GiB | 0.22 GiB |
| chip HBM 峰值 | **59,416 MiB（90.7%）** | 59,511 MiB | ~59.7 GiB |

账本：HBM 峰值 ≈ idle 2.82 + 权重 15.66 + KV 池 39.23 + 激活/杂项 ~0.3 ≈ 58.0 GiB，与采样峰值吻合。**池大小 = gmu 预算 − 权重 − 激活**，故权重越大池越小（并发换权重）。

## 5. C 组：decode KV 线性增长（单请求，prompt=128）

| 输出 tokens | 实测（prompt/completion） | KV 使用率峰值 | 用时 s | 结束后 KV |
|---|---|---|---|---|
| 64 | 132/64 | 0.090% | 2.5 | 0（归还） |
| 256 | 132/256 | 0.179% | 10.1 | 0 |
| 1024 | 132/1024 | 0.404% | 39.5 | 0 |

KV 使用率与输出 token 数严格成比例；单 token KV 成本 ≈ **0.14 MiB/token**（39.23 GiB / 285,568 tok = 0.145，两口径互证）。

## 6. E 组：完成/取消归还闭环

- 完成 ×3 轮（prompt 128 + 256 tok）：3/3，KV 峰值 0.134%，每轮归 0，无累积。
- 中途取消（2 并发，6s 掐断，max_tokens 3000）：断连后 KV 回落 0，后续请求正常——**取消请求的 KV 块被引擎回收，容量无流失**。
- ⚠️ 验证边界：本取消为客户端断连触发的引擎侧回收，未验证调度器抢占式回收（需容量打满场景，暂缓）。

## 7. 双口径核心证据（HBM vs KV）

CE 全程宿主采样（209 行）：加载稳态后 192 行 HBM 59,294–59,436 MiB，**波动仅 142 MiB**；同期 KV 使用率 0→0.4%→0 往复多次。

→ 生成式 KV 块是**真实需求且动态涨落**（区别于 embedding 的纯余量），但 vLLM 预分配将其封闭在池内，设备 HBM 恒平。**判泄漏不能看 npu-smi 单曲线，必须看 kv_cache_usage_perc + 空闲块。**

## 8. B/D 组：输入长度与并发

| 组 | 变量 | KV 峰值 | 结果 |
|---|---|---|---|
| B | 输入 1028/2052/3588（+64 出） | 0.404% / 0.762% / 1.300% | 均归还；3588 组与 0.14 MiB/tok 吻合 |
| D | 并发 4/16/64（516+256） | 0.538% / 2.332% / 5.919% | 全部完成，c64 未触发排队（离 69.72× 尚远） |

## 9. F 组：gmu 0.9 vs 0.4

| 配置 | KV 池 | KV tokens | 最大并发 | HBM 稳态 | c8 负载 |
|---|---|---|---|---|---|
| gmu0.9 | 39.23 GiB | 285,568 | 69.72× | 90.7% | — |
| gmu0.4 | **8.59 GiB** | 62,464 | **15.25×** | **28,098 MiB（42.9%）** | 8/8 完成，KV 峰值 4.1%，归还正常 |

**生成式与 embedding 的关键差别**：KV 池对生成式是真实需求，压 gmu 直接换走并发容量（69.7×→15.3×），不是 embedding 那种"纯浪费的余量"。同一旋钮，两种语义。

## 10. 关键结论（供总组 / device-context / 调度）

- **生成式-on-vLLM 账本**：HBM ≈ idle 2.82 + 权重 + gmu × 可见 HBM + 激活 ~0.3（GiB）；与 embedding 同构，**唯一大杠杆仍是 gmu**；激活在两种负载下都是 0.2–0.6 GiB 零头。
- **双指标口径**（本报告核心方法论产出）：设备 HBM 回答"池多大"，/metrics 回答"池用多少"；预分配使前者恒平，判泄漏/判容量必须看后者。
- **单 token KV 成本 0.14 MiB/token**（OneRec-8B，TP=1）→ 最大并发 ≈ KV 池 tokens ÷ 单请求上下文；`max_model_len` 与 gmu 对并发是等比例的两个旋钮。
- 归还闭环健康（完成/取消均回收、无累积），标准栈上 Memory 侧暂无需要适配的缺口。

## 11. 配置建议（观测范围内）

| 场景 | 建议 | 依据 |
|---|---|---|
| 独占芯生产实例 | gmu=0.9 | c64 无压力；容量 69.7 路 × 4096 tok |
| 共享芯 / 本方向实验 | gmu=0.4 | HBM 42.9%，c≤15；轻负载 8/512×c8 实测够用 |
| 生成较短的业务 | 先砍 `MAX_MODEL_LEN` 再动 gmu | max_model_len 4096→1024 并发 ×4，不占额外显存；需业务确认上下文分布 |
| 长期 | 上线后按真实流量 `/metrics` 回调 gmu | 压测是 worst-case；真实 KV 峰值大概率远低于 10% |

## 12. 局限与待办

- 采样 2s 分辨率，短 prefill 峰值可能被低估；vLLM 结算 peak_activation 为合成估计非实测。
- 未测：调度抢占、prefix caching、graph 模式（EAGER=0）、alloc-conf 对生成式的作用（embedding 报告 §5.2 的悬置推测）、FL 插件同用例对照。
- 探针脚本未提交 git；模型已落 `/data_lib/models/OneRec-8B`，model_list.md 路径待回写；容器复用时需加 `-v /data_lib:/data_lib`。
