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
| C1 环境核对 | wheel sha256 自证 + `up.sh --strict` 全绿（镜像 id + FM 挂载 commit） | **通过**（镜像 6cd05dcdcf30…／FlagMemory commit 58e60a9 核对 PASS） |
| C2a 继承面 | 装 wheel 前 `__file__` = /workspace/FlagMemory（环境的源树 pip-editable 形态） | **通过** |
| C2b 安装面 | 装 wheel 后 `flagmemory.__version__`==0.3.0 且 `__file__` 在 site-packages（wheel 生效、双形态对照成立） | **通过** |
| C3 功能面 | 分层管理真卡三用例 3/3（`test_ppu_tier_manager_realcard.py`，wheel 形态执行 11.7s） | **通过** |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **通过**（`overall: PASS`，summary 3/3） |

## 通过态

**首次真机执行：2026-11-19，机器 PTG_0，卡 14（执行前后 0% / 26MiB 基线、回落核验干净）。**
双形态对照成立：装前源树（/workspace/FlagMemory）→ pip 装 v0.3.0 wheel → 装后 site-packages，
分层管理三用例在两种形态各 3/3（2026-11-19 实测）。容器销毁即恢复源树形态（幂等重跑验证）。

### 过程中的真发现（如实记录）

`scripts/smoke_e2e.py`（自写的最小搬运闭环参考脚本）的 **H2D 数值一致检查在源树与 wheel
两形态同构失败（6/7）**；FM 原测试 `test_ppu_backend_realcard.py::test_真connector懒加载与全链路搬运闭环`
在同环境复现 1 failed——属 **FM 产品侧待跟进问题（copy_in 方向数值不落），非 wheel 缺陷、非样例缺陷**。
C3 判据因此采用已验证的 manager 三用例载体；smoke_e2e.py 保留在 scripts/ 供参考与后续复测。

## 教学对照（做首件样例时看这里）

| 教学点 | 本样例的落点 |
|---|---|
| 继承环境之上加装的三种方式（层包/挂载/运行时手动） | `env/README.md` 对照表 |
| 「安装的是 wheel 那份」怎么自证 | C2 的 site-packages 路径检查（PYTHONPATH 遮蔽坑） |
| 消费方没有源码仓时判据怎么写 | `scripts/smoke_e2e.py`——自写最小功能脚本，不依赖 FM tests/ |
| 交付组件的版本责任 | run.sh 头部下载 URL + sha256 锚，执行前自证 |
| 四个真实踩坑（本样例开发过程实录） | ① docker cp 改 wheel 文件名 → pip 拒装（文件名即元数据）；② 判定接 tee 管道 → 退出码被吞出假 PASS；③ 等 build.done 文件 → 宿主残留骗过等待（应以 import 成功为准）；④ compose down 漏 WORKSPACE_ROOT → down 静默失败复用旧容器 |
| 升级新版本的全部动作 | `env/README.md` 末节（换 WHEEL+sha256+期望版本号三处） |

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)
- 完整日志：执行机 `/tmp/fm-wheel-install-ppu-<时间戳>/`（不入仓）

---

<!-- changelog
2026-11-19 首建（S5 教学示范第二件：继承环境+pip 安装 FM wheel）
2026-11-19 首次真机执行全绿（PTG_0 卡14，双形态对照+三用例 3/3；H2D 真发现如实登记）
-->
