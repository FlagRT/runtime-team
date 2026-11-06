# PPU 收尾证据 raw 素材（2026-11-06）

环境锚：FlagMemory xliu969/tier-impl@536bf3b；跑卡镜像 flagrelease_ppu_vllm024_gems_540:1012（id sha256:6cd05dcd…08527）；主机回归镜像 flagtree-ppu-3.6-dev:20261031；PPU-SMI 1.17 / Driver 1.5.1-1d747a；卡 PPU0（跑前 ppu-smi 快照证空闲）。

| 文件 | 内容 | 采集命令 |
|---|---|---|
| ppu-smi-before.txt | 跑前全机占卡快照（PPU0 空闲依据） | ppu-smi |
| test_ppu_tier_manager_realcard.log | manager 端到端真卡用例（单文件 -v） | 容器内 pytest -v |
| ppu-realcard-full.log | 真卡全套（-m realcard，15 passed） | 容器内 pytest -m realcard |
| host-regression.log | 主机全套（10 失败均为环境性：ascend_profile 在 flagtree 正规环境复跑 25/25 过，遮蔽坑同首轮 PPU 验证记载） | pytest |

泄漏核查（IP/主机名与家目录/账号/口令密钥四类）：零命中，无需脱敏。

## 2026-11-06 追加：connector 级真模型 A/B 对照（收尾最后一项）

环境锚：同上（镜像/仓/模型同，本轮用卡 PPU1，跑前空闲见 ab-ppu-smi-before.txt；容器 fm-ab-a / fm-ab-b 用毕即删）。结论摘要见 ab-summary.txt。

| 文件 | 内容 | 采集命令 |
|---|---|---|
| ab-serve-a.log / ab-serve-a.stdout.log | A 轮 serve 全日志（无 connector 基线） | 容器内重定向 /evid |
| ab-serve-b.log / ab-serve-b.stdout.log | B 轮 serve 全日志（含 [PpuOffload-DMA/SCHED] 标记：store 123 批 398 块、load 4 批 8 块） | 容器内重定向 /evid |
| ab_client.py | A/B 客户端（20 条固定中文短 prompt，greedy，并发 4，可选 --recall） | python3 |
| ab-client-a.json | A 轮客户端输出（吞吐 27.19 tok/s，p50 4.804s） | ab_client.py |
| ab-client-b.json | B 轮客户端输出（吞吐 17.28 tok/s，p50 4.180s；含 recall 首放+重放） | ab_client.py --recall |
| ab_recall_probe.py / ab-recall-probe.out | 召回触发剧本第一版（filler 不足未覆写，重放走 GPU 命中，hit_len=0） | 容器内 python3 |
| ab-recall-load2.out | 召回触发剧本第二版（8 并发长 filler 覆写 → load DMA 4 批 + hit_len 最高 80，实锤） | 容器内 python3 heredoc |
| ab-ppu-smi-before.txt / ab-ppu-smi-after-a.txt / ab-ppu-smi-after-b.txt | 三时点占卡快照 | ppu-smi |
| ab-summary.txt | 指标对照 + 标记计数 + 一致性结论 | 人工汇总 |

泄漏核查：本轮文件为日志/JSON/脚本，含主机内网地址 127.0.0.1 与仓路径，无账号口令/IP 敏感新增。
