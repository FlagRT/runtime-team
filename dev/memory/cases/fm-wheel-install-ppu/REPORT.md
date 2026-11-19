# 验收样例 · fm-wheel-install-ppu —— 继承环境 + pip 安装交付组件（FM wheel）

> 规范：[`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md) §0（四件套）。

## 用途与判据

- **验收对象**：FlagMemory v0.3.0 发布 wheel（GitHub Release 附件）在 PPU 现行环境内的
  安装面与功能面（分层管理组件的消费形态）。
- **运行环境**：直接引用 FM 在 PPU 的现行定档环境
  [`dev/memory/cases/tiering-async-ppu/env/`](../../tiering-async-ppu/env/)
  （flagrelease vllm0.24 定档镜像；不用 thead-ppu env/v1 的原因：该镜像 vllm 0.19，
  FM connector 需 ≥0.24，见 FM RUNBOOK 边界登记——引用环境须核对组件版本依赖）。
  引用与加装方式见 [`env/README.md`](env/README.md)。
- **单入口**：`PPU_CARD=<n> WHEEL=<wheel路径> bash dev/memory/cases/fm-wheel-install-ppu/run.sh`
  （选卡纪律：ppu-smi 挑零负载卡；机器 = PTG_0，VPN 门控）。

## 判据表

| 判定项 | 判据 | 结果 |
|---|---|---|
| C1 环境核对 | wheel sha256 自证 + `up.sh --strict` 全绿（镜像 id + FM 挂载 commit） | **通过**（镜像 6cd05dcdcf30…／FlagMemory commit 58e60a9 核对 PASS） |
| C2a 继承面 | 装 wheel 前 `__file__` = /workspace/FlagMemory（环境的源树 pip-editable 形态） | **通过** |
| C2b 安装面 | 装 wheel 后 `flagmemory.__version__`==0.3.0 且 `__file__` 在 site-packages（wheel 生效，未被源码挂载遮蔽） | **通过** |
| C3 功能面 | 分层管理真卡三用例 3/3（wheel 形态执行） | **通过**（3 passed，11.7s） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **通过**（summary 3/3） |

## 通过态

首次真机执行：2026-11-19，机器 PTG_0，卡 14（执行前后 0% / 26MiB 基线、回落核验干净）。
双形态对照成立：装前源树（/workspace/FlagMemory）→ pip 装 v0.3.0 wheel → 装后
site-packages；三用例在两种形态各 3/3。容器销毁即恢复源树形态（幂等重跑验证）。

## 过程发现（如实记录）

开发中 `scripts/smoke_e2e.py`（自写最小搬运闭环参考脚本）的 H2D 数值检查在源树与
wheel 两形态同构失败（6/7）；FM 原测试同用例同环境复现 1 failed。当日定位定论（五层
探针 E1–E6）：产品代码正确，是用例把数据写进了目标行而非条目记录源行——FM 分支
fix-ppu-copyin-test 修正（1 failed → 2 passed，已合入 FM dev-1.0@3b7cdb1），本样例
smoke_e2e.py 同源错误同批修正。定位过程与翻绿验证见样例 fm-dev-env-ppu。

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)
- 完整日志：执行机 `/tmp/fm-wheel-install-ppu-<时间戳>/`（不入仓）

<!-- changelog
2026-11-18 首建
2026-11-19 首次真机执行全绿（PTG_0 卡14，双形态对照）
2026-11-19 smoke 同源错误修正（FM 08af9f3 批）；按审核意见去除教学性说明
-->
