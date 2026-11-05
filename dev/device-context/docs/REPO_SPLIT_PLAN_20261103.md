# 原型独立仓报备计划（草稿，v0.4.0 时启用）

> 起草：2026-11-03 ｜ 状态：**已执行（Phase 1 完成，2026-11-03）** —— 仓已建、v0.4.0 已发；`dev/repos.yaml` 收录待总组 | 启用时点：**v0.4.0 发布**（四实例复验收口后）
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


---

## 六、执行记录（2026-11-03 · Phase 1 完成）

| 步骤 | 结果 |
|---|---|
| 建仓 | **`FlagRT/flagos-device-context`**，**public**（用户裁定；等价满足公告「开放全组 clone」），`main` 单协作线 |
| 搬迁内容 | `runtime/`（31 个 `.py`/`.md`）+ **新建 `pyproject.toml`** + 顶层 `README.md` + `RELEASE_NOTES_v0.4.0.md` + `.gitignore` |
| 未搬（按 §四） | 芯片 `910C|P800|MLU590|PPU/` 资产与证据、`prototype/scripts/`（服务入口等组内标准工具）、`prototype/docs/` 看板类文档 |
| 包内清理 | **排除全部历史产物 JSON**（`runtime/proto/*.json`、`runtime/conformance/*.json` —— 已核实代码不读它们，只是归档产物）；wheel 内 0 个 json |
| pip 化验证 | clean venv **从 Release URL 直接 `pip install` 成功**；`runtime.__version__=0.1.0`、`__component_version__=0.4.0`、46 出口、4 个子包全部可导入 |
| 发布 | Release `runtime-v0.4.0`（id `432357486`）+ wheel（193,406 B）+ sdist（168,829 B）；`tag^{}` = `main` tip = `984687f` |
| 报备 | 四字段 + 版本锚 + `dev/repos.yaml` 待粘块（逐字段对齐第 85–93 行）已写入 `STATUS.md`「交付仓报备」节；**未自行改 `dev/repos.yaml`** |
| 过渡期 | 协作仓 `prototype/runtime/` 暂留同一份内容，**已核对与新仓逐字节一致（31 文件零差异）**，避免打断四实例验收 |

### Phase 2（v0.5.0 时做）

1. 实例侧取包方式切换：`DC_ROOT=<协作仓 prototype>` → 「`pip install` 或挂新仓」；
2. 切换后**删除协作仓 `prototype/runtime/` 副本**（消除两处漂移风险）；
3. README 只推 `pip install`；
4. 新环境目录的 `env.baseline.yaml` 增 `mounted_repos` 条目（remote=新仓 / commit=当次发版提交）。

⚠️ **Phase 1 → Phase 2 之间**：任何对 runtime 包的改动**必须两处同步提交**（协作仓 + 新仓），否则两边会漂。
