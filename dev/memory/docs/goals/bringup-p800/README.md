# 🟢 bringup-p800 —— 昆仑芯 P800 bring-up（进行中）

> 目标：在 FlagOS 官方栈（厂商 torch + FlagGems + FlagCX + vllm-plugin-FL + vLLM）上把昆仑芯 P800 跑通、画像、补齐显存/KV 卸载能力——开放式目标，无锁定终点。
> 权威方案：[../../common/design_显存与缓存管理权威方案.md](../../common/design_显存与缓存管理权威方案.md)（本目标是该方案在 P800 芯片上的具体执行）。

| 文件 | 内容 |
|---|---|
| `profile_P800可用性实测_p800.md` | P800 端到端可用性实测（2026-08-21）：官方栈跑通 Qwen3-4B（96.5 tok/s），分层验证 + 阻塞项清单 |
| `profile_P800显存画像_p800.md` | P800 显存画像 V1-A 专项（2026-08-22）：eager 模式 7 阶段全绿，KV 556,352 tokens/76.40GiB |
| `issue_昆仑芯问题反馈清单_p800.md` | 拟提交智源/FlagOS 的问题反馈清单：causal_conv1d / topk_softmax / 文档滞后等 |
| `note_P800适配执行记录_p800.md` | P800 环境适配执行记录（2026-08-20）：驱动/容器/xpytorch/flag_gems vendor 识别 |
| `note_新线镜像纯MoE复测_p800.md` | 昆仑芯 MoE 复测系列结论主文档（2026-08-22）：绕开 topk_softmax 阻塞、expert GEMM 首次触达 |
| `note_新线栈decode生成退化根因定位_p800.md` | decode 生成退化根因定位（2026-09-01）：厂商 patch_decode_attention 补丁为退化源 |
| `survey_vllm0.13-allocator与offload调研_p800.md` | vllm 0.13.0 allocator/offload 机制调研（2026-08-22）：对照 0.20.2，KV CPU 卸载原生可用 |
| `history/` | 已被上述文档整合/取代的过程性记录，仅留历史对照，见 [history/README.md](history/README.md) |
