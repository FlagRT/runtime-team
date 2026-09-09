# 基座镜像规范（dev/images）

> 状态：生效（2026-09-09）｜ 权威环境依据：`dev/stack.lock.910c.v1.yaml`
> 目的：每个进入 `stack.lock` 的基座镜像，都在本目录有一份**可复现的构造记录**（Dockerfile + pin 清单 + provenance），并明确**官方 / 自定义边界**，避免偏离官方主线、避免"黑盒镜像当基座"。

---

## 1. 三个镜像的准确定位

| image-name | 锁定 tag | 定位 | 来源 | 谁用 |
|---|---|---|---|---|
| `ascend-operator-runtime` | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64` | **昇腾工具链 + FlagOS 组件底座**（CANN 9.0.0 + Python 3.11.15 + torch 2.10cpu + Torch-FL/FlagGems + triton-ascend）。本身不直接锁进 demo，是 `ascend-train-comm` 的父层；也是 A 线（torch_npu）主线训练基座的候选父层 | FlagRT 自定义 | 间接（作为父层）；A 线回归时 + 容器内 torch_npu |
| `ascend-train-comm` | `flagrt/ascend-operator-runtime-comm:0.1.3-…-flagcx0.13.0g55eb2ffp2-arm64` | **本月训练腿锁定基座**（段一：2 卡分布式微调）。父层之上叠 FlagCX 0.13.0 昇腾适配。`release.stage=candidate`，**torch_fl 例外线**（官方主线是 torch_npu / Route A） | FlagRT 自定义 | device-context（段一）、communication（FlagCX 验证）、performance（训练吞吐） |
| `ascend-infer-vllm` | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | **推理腿锁定基座**（段二：单卡推理服务）。vLLM 0.20.2 + torch_npu + CANN，官方自带栈 | **华为官方**，digest 锁定 | device-context（段二）、memory（显存画像）、调度、performance |

> 关系：`ascend-infer-vllm` 与 `ascend-train-comm` 互不派生。`ascend-train-comm` = `ascend-operator-runtime` + FlagCX 层。
> 历史遗留镜像（`flagos-910c-train-850`、`flagos-device-context-*`、VERSIONS.md §2 的 `pytorch-plugin-fl:manual-20260807-*` 等）**不属于本规范**，不得用于结论性验证。

---

## 2. 目录结构

```
dev/images/
  README.md
  <image-name>/v<N>/           # 一个版本一目录，多版本共存
    Dockerfile                 # 可复现构造（官方镜像可无，见 §5）
    lock.yaml                  # pin 清单 + provenance + 官方/自定义边界 + gaps
    build.sh                   # 一键重建
    assets/                    # 构建期脚本/清单（verify、patch、requirements、docker-history）
```

`<image-name>` = 稳定短名（去 registry 前缀与 tag）。

---

## 3. 版本化与平稳迭代

- 一个 `v<N>/` = 一个不可变版本，对应一个镜像 tag。当前生效版本见 §7。
- 镜像有变更 → **新建 `v<N+1>/`**，旧目录保留不动；旧 `lock.yaml` 顶部加 `superseded_by: v<N+1>`。
- `stack.lock` 与各组按自身节奏把引用从 `v<N>` 切到 `v<N+1>`；旧目录待全组确认无人依赖后才删。
- 不影响已构造产物的措辞小改，直接改当前目录并记 `lock.yaml` changelog。

（与战略文档 §8 版本化规则一致。）

---

## 4. 官方 / 自定义边界（核心）

每个 `lock.yaml` 的 `layers:` 逐项标 `origin`：

| origin | 含义 | pin 方式 | 变更纪律 |
|---|---|---|---|
| `official` | 华为昇腾 / 上游开源项目原样产物（CANN、ubuntu、vLLM-Ascend、triton-ascend 官方 wheel、MPICH 源码等） | digest 或官方版本号 | 跟随官方发布，不改写 |
| `custom` | FlagRT/FlagOS 自行组合或打补丁的部分（Torch-FL、FlagGems、FlagCX、triton 补丁、verify 脚本、运行时开关 ENV） | git commit + patch sha256 | 每次变更记 changelog，知会全组 |

**原则**：`custom` 只在官方未提供对应能力时引入；每个 `custom` 项在 `lock.yaml` 的 `mainline_fallback` 里写清"回到官方主线的路径"。当前 `ascend-train-comm` 整条 torch_fl 线是 `custom` 例外，主线是 torch_npu（Route A）——见 `stack.lock` `device_registration`。

---

## 5. 入锁门槛（挂进 stack.lock 前）

1. 本目录有 `<image-name>/v<N>/`，含 `lock.yaml`；
2. `repro_status` ≥ 🟡（配方在手，即使 registry 未发布）；
3. 官方镜像可无 `Dockerfile`，但 `lock.yaml` 须有可拉取 digest + 上游来源；
4. 🔴（黑盒无配方）不得作为**验收基座**，且须有第二位置 `docker save` 备份记录。

`repro_status`：🟢 已发布可拉取带 digest ｜ 🟡 配方在手待发布 ｜ 🔴 黑盒

---

## 6. 治理与协作机制

- **决策与发布 = 总组**：统一基座的确定、调整、发布，由**总组长收拢各子方向在 `dev/<方向>/STATUS.md` 暴露的问题与意见后统一裁定**（对齐战略文档 §5 总组职责、§8.2 STATUS 入口）。子方向不自行确定或替换基座。
- **构建与登记 = 总组指定的执行方**：当前镜像发布状态登记职责在 performance（战略文档 §5）；实际构建方以总组指派为准。执行方负责推 `harbor.baai.ac.cn`、登记 digest、补齐 `lock.yaml:gaps`。
- **子方向如何用**：
  1. 只消费不自建——`docker pull <digest>` 或加载执行方提供的 `docker save` 包；
  2. 容器名按 `stack.lock` 规范（`flagos-proto-<train|infer>-910c`）；
  3. 方向自身依赖（如 `transformers`）在容器内补装，版本回写 `stack.lock` 对应字段，**不改基座**；
  4. 结论性验证只在锁定镜像内做；
  5. 问题、装包需求、升版本诉求 → 写进本方向 `STATUS.md`，由总组收拢裁定。
- **变更合入**：镜像变更要合入，须同时——① 新 tag + `v<N+1>/` 目录齐全 ② `lock.yaml` changelog 写明官方/自定义改了哪项 ③ 至少一个下游方向确认可用 ④ 总组知会全组 ⑤ `stack.lock` 同步更新。
- **兜底**：registry 或构建机丢失时，本目录 `Dockerfile` + `assets/` + pinned 源码 commit 是重建路径。
- **节奏**：每个阶段交界，总组复核一次各 `custom` 组件 commit 是否跟进上游、`official` 部分是否有新版本。

---

## 7. 当前生效版本

| image-name | 生效版本 | repro_status |
|---|---|---|
| `ascend-operator-runtime` | v1 | 🟡 |
| `ascend-train-comm` | v1 | 🟡 |
| `ascend-infer-vllm` | v1 | 🟢 |

> 🟡 = 配方在手，**尚未重建/验证/发布**；`stack.lock` 锁的仍是 npu1-27 上的原预构建镜像。
> 可复现化（🟡 → 🟢）任务清单与负责人见 **`dev/images/TODO.md`（负责：总组）**。
