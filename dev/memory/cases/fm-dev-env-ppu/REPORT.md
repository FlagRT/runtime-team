# 验收样例 · fm-dev-env-ppu —— 自建派生环境 + FM 开发线真卡验证

> 规范：[`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md) §0（四件套）。

## 用途与判据

- **验收对象**：FM dev-1.0 线在 PPU 真卡上的验证（载体 = fix-ppu-copyin-test：换行换入
  用例数据行修正，已合入 FM dev-1.0@3b7cdb1）。
- **运行环境**：**自建派生**——对齐母本与差异清单见 [`env/README.md`](env/README.md)
  （母本 = FM 在 PPU 的现行定档环境；唯一差异 = FM 挂载锚切 FM dev-1.0 线头）。
- **单入口**：`PPU_CARD=<n> bash dev/memory/cases/fm-dev-env-ppu/run.sh`
  （选卡纪律：ppu-smi 挑零负载卡；机器 = PTG_0，VPN 门控）。

## 判据表

| 判定项 | 判据 | 结果 |
|---|---|---|
| C1 环境核对 | `up.sh --strict` 全绿（镜像 id + **FM dev-1.0 头 commit** 守门——派生环境的门） | **通过**（镜像 6cd05dcdcf30…／FM commit=3b7cdb1 核对 PASS） |
| C2 锚核对 | 容器内 FM 检出 = 3b7cdb1（git rev-parse 完整 SHA 逐字相等） | **通过** |
| C3a 修复用例 | `test_ppu_backend_realcard.py` 2/2（修复前 1 failed） | **通过**（2 passed，11.6s） |
| C3b 不回归 | `test_ppu_tier_manager_realcard.py` 3/3 | **通过**（3 passed，11.6s） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **通过**（summary 4/4） |

## 通过态

首次真机执行：2026-11-19，机器 PTG_0，卡 14（执行前后 0% / 26MiB 基线、回落核验干净）。
派生环境守门验证成立：同一镜像下，母本环境锁 v0.3.0 发版 commit（58e60a9）、本环境锁
FM dev-1.0 头（3b7cdb1）——`up.sh --strict` 对两者的拒绝/放行各得其所。修复用例翻绿
（1 failed → 2/2）与 manager 不回归（3/3）在本环境一次跑全。

## 修复背景

2026-11-19 样例 fm-wheel-install-ppu 开发中暴露「H2D 数值不落」；五层探针（E1 主线程
单拷贝 → E2 后台线程 → E3 裸 copy_+stream → E4 同步 → E5 完整 DmaCopyBackend 双流）
排除产品代码，E6 双写法验证坐实：契约 = 源行按条目记录定位，用例却把数据写进了目标行
——产品行为正确、用例数据行写错。修复 = 数据行 `cpu[3]`→`cpu[2]` + 断言语义注明
（FM 08af9f3，PTG_0 真机 1 failed → 2 passed，已合入 FM dev-1.0@3b7cdb1）。

## 跟线维护

FM dev-1.0 推进后：宿主检出 ff → 重导本目录 env.baseline（`baseline.py export`）→
更新 run.sh 的 `FM_COMMIT_EXPECT` → 本表锚更新。

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)
- 完整日志：执行机 `/tmp/fm-dev-env-ppu-<时间戳>/`（不入仓）

<!-- changelog
2026-11-19 首建（FM 锚=开发分支 08af9f3）
2026-11-19 首次真机执行全绿（PTG_0 卡14，summary 4/4）
2026-11-19 修复合入 FM dev-1.0@3b7cdb1 后重锚至线头；按审核意见去除教学性说明
-->
