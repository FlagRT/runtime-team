# P800 规范验证跑 —— 跑前声明（PROFILE-PROTOCOL v0.2 首次全程执行）

> 日期：2026-10-25。本文件在开机**前**写定（原则 4），跑完后只能补"实测结果"栏，
> 不得改判据。目的：①按 v0.2 产出 P800 完整基线+对照数据；②检验规范本身可执行性。

## 环境锚点

- 栈：vLLM 0.20.2+empty + vllm-plugin-fl 0.2.0+g38e7dbc（kunlunxin vendor）
- 机/卡：P800 节点（VM-0-2-ubuntu）xpu5，容器 mem-profile-p800；起跑前真值
  95.97/96 GiB free（2026-10-25 实测），独占口径
- 模型：/work/models/OneRec-8B（Qwen3ForCausalLM，15.63 GiB checkpoint）
- 对照项：kv-transfer-config 有无（XpuSimpleCPUOffloadConnector，offload.xpu_offload，
  Host 池 16 GiB）；其余配置两引擎逐字一致

## §3 服务配置固定表（两引擎共用）

TP=1｜单 API server｜max-model-len 4096｜enforce-eager 开｜gmu 0.8 +
kv-cache-memory-bytes=64GiB（池锁定）｜max_num_seqs/chunked prefill/prefix caching
=引擎默认（记录生效值）｜temperature=0

## 腿次序与有效判据（开机前声明）

| 腿 | 定档 | 支持的结论 | 有效判据（不满足=无效，不入正文） |
|---|---|---|---|
| P 池画像 | 零生成请求，就绪后 +2/+5/+10min 采样 | 池物理常数（后续腿的分母） | 池 tok 必须来自引擎日志行原文 |
| S1 prefill | 256/1024/3072 tok ×3 轮，max_tokens=1 | 纯 prefill 时延曲线 | 输出必须=1 tok；prompt 实测 tok 偏离目标 >10% 无效 |
| S2 decode | 输入 256 tok，输出 512 tok，4 轮（首轮 sanity） | 单流 decode 稳态速度 | completion_tokens≥128；报中位数+最差 |
| B 档位 | c=28/57/102/125（池 25/50/90/110%），prompt≈3968 tok，输出 128，每档 2 轮 | 容量曲线+饱和分界线 | 每档须报池利用率峰值；<30% 档只标"未饱和参考"；"饱和"结论须见 capacity waiting>0 或驱逐>0 |
| A/B 对照 | OFF 引擎全腿 → ON 引擎 S2+B 全腿 | 卸载开关对照 | cost 档（≤50%）预期零差；expand 档（110%）关腿必须见 capacity waiting>0，开腿必须见 DMA store/load 计数 |

## 正确性验收（ON 引擎必过）

- 同 prompt 同 temperature=0：S2/B 各档 OFF 与 ON 的输出文本比对（含召回路径）；
- DMA：marker 文件 store/load 分开计数 + host RSS 起止 + 完成归还。

## 预注册的预期（防事后挑数）

- S2：OFF 单流 ~13.8 tok/s 量级（10-23 口径）；ON 预期同量级（无池压无 DMA）。
- B 档 OFF：c28/c57 未饱和；c102 逼近；c125 出现 capacity waiting。
- B 档 ON：c125 不排队（Host 兜底），DMA 出现。
- 批量 decode 病态预期复现（聚合吞吐随并发不升反降）——独立记录，报 vendor 线。

## 实测结果

→ 已完成，见同目录 `RESULT_规范验证跑_P800_20261025.md`。
→ 预注册 5 条：4 中 1 修正（ON c125 仍排队，收益为完工 -17% 而非消除排队）。
→ 规范抓出 2 个数据 bug、2 个规范漏洞、1 次执行违规；无效数据 4 批未入正文。
→ 卡已还。
