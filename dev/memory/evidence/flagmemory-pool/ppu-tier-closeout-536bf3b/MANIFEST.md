# PPU 收尾证据 raw 素材（2026-11-06）

环境锚：FlagMemory xliu969/tier-impl@536bf3b；跑卡镜像 flagrelease_ppu_vllm024_gems_540:1012（id sha256:6cd05dcd…08527）；主机回归镜像 flagtree-ppu-3.6-dev:20261031；PPU-SMI 1.17 / Driver 1.5.1-1d747a；卡 PPU0（跑前 ppu-smi 快照证空闲）。

| 文件 | 内容 | 采集命令 |
|---|---|---|
| ppu-smi-before.txt | 跑前全机占卡快照（PPU0 空闲依据） | ppu-smi |
| test_ppu_tier_manager_realcard.log | manager 端到端真卡用例（单文件 -v） | 容器内 pytest -v |
| ppu-realcard-full.log | 真卡全套（-m realcard，15 passed） | 容器内 pytest -m realcard |
| host-regression.log | 主机全套（10 失败均为环境性：ascend_profile 在 flagtree 正规环境复跑 25/25 过，遮蔽坑同首轮 PPU 验证记载） | pytest |

泄漏核查（IP/主机名与家目录/账号/口令密钥四类）：零命中，无需脱敏。
