# 原型独立仓报备计划（草稿，v0.4.0 时启用）

> 起草：2026-11-03 ｜ 状态：**草稿**（未报备、未建仓）｜ 启用时点：**v0.4.0 发布**（四实例复验收口后）
> 依据：《组内公告-自建仓规范与双版本机制-20261023》＋ `dev/repos.yaml` 头注 + 首例 `dev/framework-adapter/STATUS.md`

## 一、为什么要独立仓（先想清楚再动）

| 现状 | 独立仓后 |
|---|---|
| 原型在协作仓 `dev/device-context/prototype/` 内，与四实例 probes/ 证据强耦合 | 功能性交付（可 pip 安装的 runtime 包）进自建仓；**协作仓只留芯片实例资产 + 看板 + STATUS 指针** |
| 其他组员要用原型 ⇒ clone 整个协作仓再 `sys.path` 指过去 | `pip install`（wheel 从 GitHub Release 装，`runtime-v0.3.0` 已具备该形态） |
| 版本锚 = 我们口头说「dev-1.0 当前」 | 版本锚 = `env.baseline.yaml` 的 `mounted_repos`（remote/commit），clone_all 只管布局不管版本 |

⚠️ **公告原文没有强制搬迁**——它要求的是「功能性开发落入 fork/自建仓 + 统一登记 + 版本锚在 env.baseline」。
我们现在「证据与原型同仓」在复验高频期是优点（一轮复跑一个 commit 全带）；**搬迁的价值在交付面**（别人 pip 装包），
所以定在 **v0.4.0（收口版）** 动，不在复验中途动。

## 二、报备四字段（照公告「报备与收录」格式）

| 字段 | 值（拟） |
|---|---|
| 仓地址 | `git@github.com:FlagRT/flagos-device-context.git`（名字待定，候选：`flagos-device-context` / `flagos-runtime-dc`） |
| 仓根目录名 | `flagos-device-context`（clone_all 收拢到仓根一级，容器挂载 `/workspace/flagos-device-context`） |
| 协作线 | `main`（kind: direction 不要求 dev-1.0；参照 framework-adapter「main 即唯一工作线」，开发仍走个人分支 + merge） |
| 可见性 | `private`（组内仓；需总组为全组开通 clone 权限） |
| 需访问成员 | 全组（runtime-team 成员） |

## 三、搬迁清单（v0.4.0 时执行）

1. **搬运**：`prototype/runtime/`（包本体）+ `pyproject.toml`（**新建**，当前是散装 `sys.path` 形态，这是 pip 化的前提）
   + `prototype/RELEASE_NOTES_*` + conformance/smoke（作为包的测试资产）。
2. **留在协作仓**：`910C|P800|MLU590|PPU/`（实例资产与证据）、`prototype/docs/` 的看板类文档（接收入口不动）。
3. **dev-1.0 留指针**：`dev/device-context/STATUS.md` 加「仓地址 + 当前版本锚」一行（公告第 5 条第 3 点）。
4. **版本锚落 env**：新环境目录的 `env.baseline.yaml` 增 `mounted_repos` 条目（remote=新仓 / commit=发版 commit）。
   ⚠️ 现有 `910C/env/{train,infer}` 不回改（ADR-0008：修复只新增 env 版本）。
5. **Release 资产平移**：`runtime-v0.3.0` tar.gz 在协作仓 Release 上；v0.4.0 起发到新仓 Release，协作仓 Release 保留旧版不删。

## 四、不搬的部分与理由（防未来误搬）

- `scripts/serve_standard.sh` / `duty_response_audit.py` 等**验收工具**：服务入口是组内标准（各方向共用），留在协作仓
  `prototype/scripts/`——若日后总组要求组件仓化，随 `framework-adapter` 先例走，不自行决定。
- 芯片 probes/ 证据：与各实例环境强绑定（路径、容器名都是实例事实），搬走即断链。

## 五、风险与对策

| 风险 | 对策 |
|---|---|
| 两处看板漂移（协作仓 STATUS vs 新仓 README） | 公告纪律「dev-1.0 只放指针」——协作仓 STATUS 记「新仓地址 + 版本锚」，事实本体只在新仓 |
| clone_all 对 private 仓无权限降级 | 报备时显式列「需访问成员=全组」；总组开通前不影响其他仓对齐（公告第 3 条） |
| pip 化后 `sys.path` 旧用法断 | `pyproject.toml` 同版保留 `runtime` 包名与导入路径不变；旧用法（sys.path 指 prototype）在 v0.4.0 过渡期继续可用，v0.5.0 起 README 只推 pip |
