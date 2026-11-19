# 验收样例（S5 教学示范第二件）· fm-wheel-install-ppu —— 继承环境 + pip 安装交付组件（FM wheel）

> **定位**：与 fm-smoke-ascend（纯引用式）配对——本件演示**引用现行环境之上，自己额外加装
> 交付组件**：环境不挂 FlagMemory（thead env/v1 的 `mounted_repos` 为空），以发布 wheel
> pip 安装进容器再验收。消费方视角：手里只有 wheel，没有 FM 源码仓。
> 规范权威 = [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md) §0。

## 用途与判据

- **验收对象**：FlagMemory v0.3.0 发布 wheel（GitHub Release 附件）在 PPU 现行环境内的
  **安装面**与**功能面**（分层搬运异步闭环——FM 分层管理组件的消费形态）。
- **运行环境**：直接引用现行环境目录
  [`dev/images/thead-ppu-runtime/v2/env/v1`](../../../images/thead-ppu-runtime/v2/env/v1/)
  （引用与加装方式的教学注解见 [`env/README.md`](env/README.md)），`up.sh --strict` 机器核对。
- **单入口**：`PPU_CARD=<n> WHEEL=<wheel路径> bash dev/memory/cases/fm-wheel-install-ppu/run.sh`
  （选卡纪律：ppu-smi 挑零负载卡；CUDA_VISIBLE_DEVICES 授予；机器 = PTG_0，VPN 门控）。

## 判据表

| 判定项 | 判据 | 结果 |
|---|---|---|
| C1 环境核对 | wheel sha256 自证 + `up.sh --strict` 全绿 | 待执行 |
| C2 安装面 | 容器内 `flagmemory.__version__`==0.3.0 且 `__file__` 在 site-packages（wheel 形态生效、未被源码挂载遮蔽） | 待执行 |
| C3 功能面 | 异步搬运最小闭环 8 检查全过（懒加载真 DmaCopyBackend / D2H·H2D 事件确认+数值逐字一致 / 锁页探测真锁页） | 待执行 |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | 待执行 |

## 通过态

（首次真机执行后填写；PTG_0 为 VPN 门控机，窗口未开时本表如实保持「待执行」）

## 教学对照（做首件样例时看这里）

| 教学点 | 本样例的落点 |
|---|---|
| 继承环境之上加装的三种方式（层包/挂载/运行时手动） | `env/README.md` 对照表 |
| 「安装的是 wheel 那份」怎么自证 | C2 的 site-packages 路径检查（PYTHONPATH 遮蔽坑） |
| 消费方没有源码仓时判据怎么写 | `scripts/smoke_e2e.py`——自写最小功能脚本，不依赖 FM tests/ |
| 交付组件的版本责任 | run.sh 头部下载 URL + sha256 锚，执行前自证 |
| 升级新版本的全部动作 | `env/README.md` 末节（换 WHEEL+sha256+期望版本号三处） |

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)
- 完整日志：执行机 `/tmp/fm-wheel-install-ppu-<时间戳>/`（不入仓）

---

<!-- changelog
2026-11-19 首建（S5 教学示范第二件：继承环境+pip 安装 FM wheel，判据表待真机执行回填）
-->
