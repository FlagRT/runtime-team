# 验收样例（S5 教学示范第三件）· fm-dev-env-ppu —— 自建派生环境 + FM 开发线真卡验证

> **定位**：三部曲之三——①纯引用（fm-smoke-ascend）②引用+临时加装（fm-wheel-install-ppu）
> ③**自建派生环境**（本件）。规范权威 = [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md) §0。

## 用途与判据

- **验收对象**：FM 分支 `xliu969/fix-ppu-copyin-test`（换行换入用例数据行修正）在 PPU
  真卡上的验证——**顺便承载的真实特性开发**，不是虚拟 case。
- **运行环境**：**自建派生**（对齐声明与差异清单见 [`env/README.md`](env/README.md)）——
  母本 = FM 在 PPU 的现行定档环境（tiering-async-ppu/env），唯一差异 = FM 挂载锚切本分支。
- **单入口**：`PPU_CARD=<n> bash dev/memory/cases/fm-dev-env-ppu/run.sh`
  （选卡纪律：ppu-smi 挑零负载卡；机器 = PTG_0，VPN 门控）。

## 判据表

| 判定项 | 判据 | 结果 |
|---|---|---|
| C1 环境核对 | `up.sh --strict` 全绿（镜像 id + **FM 开发分支 commit** 守门——派生环境的门） | **通过**（镜像 6cd05dcdcf30…／FM commit=08af9f318dc6 核对 PASS） |
| C2 锚核对 | 容器内 FM 检出 = 分支头（git rev-parse 实证，完整 SHA 逐字相等） | **通过** |
| C3a 修复用例 | `test_ppu_backend_realcard.py` 2/2（修复前 1 failed——数据行修正的翻绿验证） | **通过**（2 passed，11.6s） |
| C3b 不回归 | `test_ppu_tier_manager_realcard.py` 3/3 | **通过**（3 passed，11.6s） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **通过**（summary 4/4） |

## 通过态

**首次真机执行：2026-11-19，机器 PTG_0，卡 14（执行前后 0% / 26MiB 基线、回落核验干净）。**
派生环境守门验证成立：同一镜像下，母本环境锁 v0.3.0 发版 commit、本环境锁开发分支头
08af9f3——`up.sh --strict` 对两者的拒绝/放行各得其所。修复用例翻绿（1 failed → 2/2）与
manager 不回归（3/3）在本环境一次跑全。开发线推进后按 env/README 重导 baseline 跟线。

## 背景故事（为什么有这个修复——教学：一次完整的缺陷定位）

2026-11-19 样例②开发中暴露「H2D 数值不落」；五层探针（E1 主线程单拷贝 → E2 后台线程 →
E3 裸 copy_ + stream → E4 同步 copy_ → E5 完整 DmaCopyBackend 双流）逐层排除产品代码，
E6 双写法验证坐实：**契约=源行按条目记录定位，用例却把数据写进了目标行**——产品行为正确、
`test_ppu_backend_realcard.py` 用例数据行写错。修复 = 数据行 `cpu[3]`→`cpu[2]` +
断言语义注明（FM 分支 08af9f3，PTG_0 真机 1 failed → 2 passed）。样例②的
`scripts/smoke_e2e.py` 同源错误一并修正于该分支同批。

## 教学对照（做首件样例时看这里）

| 教学点 | 本样例的落点 |
|---|---|
| 自建派生的第一句话 | `env/README.md` 派生声明（对齐哪个母本、差异清单） |
| 自建三步 | 拷贝改差异 → 同镜像拷登记桩 → 宿主检出停目标 commit 后 `baseline.py export` |
| 开发线怎么跟 | ff 检出 → 重导 baseline → REPORT 注明当前锚（strict 门随线头走） |
| 就绪等待的正确姿势 | import 成功为准（不依赖宿主残留的 build.done——样例②踩坑③） |
| 判定不接 tee | run.sh 注明（样例②踩坑②） |

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)
- 完整日志：执行机 `/tmp/fm-dev-env-ppu-<时间戳>/`（不入仓）

---

<!-- changelog
2026-11-19 首建（S5 教学示范第三件：自建派生环境）
2026-11-19 首次真机执行全绿（PTG_0 卡14，summary 4/4；派生守门+修复翻绿+不回归三项实证）
-->
