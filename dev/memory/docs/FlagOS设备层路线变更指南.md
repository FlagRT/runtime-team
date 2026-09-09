# FlagOS 设备层路线变更指南（决策记录）

> ## ⛔ 现行结论：生产交付统一走 FlagOS 官方栈（厂商设备插件 + FlagGems + FlagCX + vllm-plugin-FL）
> torch_fl 设备层路线**已冻结**：不承担交付责任、不作前置依赖、memory 子方向不再投入。相关历史资产见 [archive/](archive/README.md)。
> 本文档为 2026-08-22 决策的**理由与背景留档**，非当前进展看板。

- 日期：2026-08-22（决策）；2026-09-03（torch_fl 路线冻结、资产归档）
- 适用芯片：昇腾、昆仑芯 P800、平头哥真武（及后续接入的其他芯片）
- 变更性质：生产交付主线切换为官方栈；torch_fl 设备层由候选主线降为冻结路线
- 依据：Torch-FL 设备接入层技术调研、双路线对比评估、昆仑芯 P800 官方栈实测、官方镜像 MoE 复测、纯 MoE 隔离测试

---

## 一、两个方案是什么

FlagOS 栈自下而上分四层：**设备层 → 算子层（FlagGems）→ 通信层（FlagCX）→ 推理插件（vllm-plugin-FL）**。

两个方案**只在设备层不同，上面三层完全相同**。

| | 路线 A | 路线 B |
|---|---|---|
| 设备层 | 各芯片厂商插件 | **torch_fl**（= PyTorch-Plugin-FL = Torch-FL，同一组件，仓库改过名） |
| 算子层 | FlagGems | FlagGems（需 fork 分支） |
| 通信层 | FlagCX | FlagCX（需 fork 补丁） |
| 推理插件 | vllm-plugin-FL | vllm-plugin-FL（靠 shim 冒充厂商设备模块兜底） |
| 官方定位 | FlagOS 官方发布配置 | 生态插件，v0.1.0 |

**术语澄清**：所谓"在 PyTorch-Plugin-FL 级别做算子联调"，实际含义是"用 torch_fl 作设备底座跑联调"——torch_fl 是设备层，不是算子层。算子在各芯片的原生算子库与 FlagGems 里。

**归属判定**：用 torch_fl 做设备接入 = 路线 B；在厂商 torch 插件上跑 FlagGems = 路线 A。模型组目前的做法属于 A。

---

## 二、各芯片现状（全部适用本变更）

| 芯片 | 路线 A | 路线 B（torch_fl） |
|---|---|---|
| **昇腾** | 官方发布配置（torch_npu）；训练链路完整 | ACCELERATOR=ascend，标注 Beta，原生 aclnn；本地已跑通 Qwen3-4B 与 16 卡 DDP；≥2k prefill 未完成；无训练栈 |
| **昆仑芯 P800** | 厂商 CUDA 兼容 torch；已实测端到端跑通 dense 推理 | 2.9 分支有 ACCELERATOR=kunlun（CUDA boxing），**无 CI，FlagGems 路径被文档明确关闭** |
| **平头哥真武** | 厂商 PPU SDK；FlagGems 有完整 thead 后端；vllm-plugin-FL 有 vendor 实现 | CUDA boxing，标注 Experimental，CI 管线 PR 未合入 |
| **清微** | FlagGems 有后端 | Runtime only，**无 eager 算子**，不可作通用设备层 |

**结论**：B 在三家目标芯片上，只有昇腾具备可用性验证；昆仑芯与真武均无 CI、无端到端验证。**"跨芯统一设备层"目前不是 B 的现实优势，而是其最薄弱处。**

---

## 三、关键变更点

### 1. 主线切换

**生产交付全部走路线 A，覆盖所有芯片。** 推理与训练均适用。

训练是硬约束：Megatron-LM-FL / TransformerEngine-FL / verl-FL 的多芯适配全部构建在厂商 torch 插件上，B 无等价实现，自建为人月级。

### 2. torch_fl 路线资产保留归档，不再投入（2026-09-03 更新）

已投入的验证资产（昇腾 Qwen3-4B TP=1/2/4、16 卡 DDP、FlagCX 补丁、若干上游 PR）作为历史资产保留于 [archive/](archive/README.md)。该路线不承担交付责任、不作为任何团队的前置依赖，memory 子方向不再沿此方向开发。上游 PR 是否继续跟进由相关人员按需处理，与本子方向交付脱钩。

### 3. 对外口径必须纠正

需撤回的表述：**"必须基于 torch-plugin 做设备接入"**、**"torch_fl 是运行时层其他模块的基础"**。

代码层面不成立：上层 6 仓（vllm / sglang / Megatron / TE / verl / FlagScale）对 torch_fl **零引用**，各芯片设备层现状均为厂商插件。上层出现的 "flagos" 指 **FlagGems 算子实现优先级**（`VLLM_FL_PREFER=flagos`），与 torch_fl 设备同名不同物——对方的判断很可能源于此。

反例：FlagCX 的 torch 插件是独立 distributed backend（按 `FLAGCX_ADAPTOR` 选芯片），不经 torch_fl。

准确表述：torch_fl 是**选定 B 路线后的内部依赖**（PyTorch 单 PrivateUse1 槽位机制决定，不可拆分），不是全栈的共同基础。

### 4. 与模型组收口

模型组在 FlagGems 层的联调即路线 A，与新主线一致，继续推进。此前"运行时组走 B、模型组走 A"的不一致就此消除。

### 5. 测试环境隔离

**路线 A 的所有测试改在官方发布镜像内进行**，dev 容器仅用于 B 的开发。

已有教训：dev 容器的 triton 版本偏差导致 FlagGems GEMM 编译崩溃，一度被误判为组件缺陷。版本偏差会污染结论。

### 6. 已识别的跨路线阻塞（最高优先级）

以下缺口位于 **A 与 B 共用的上层组件**，换设备层无法解决：

| 缺口 | 位置 | 影响 |
|---|---|---|
| `xpudnn::causal_conv1d_update ret=1` | 厂商算子库 | 混合注意力架构模型（Qwen3.5/3.6 一代）在昆仑芯不可用 |
| `flag_gems._kunlunxin.topk_softmax` 签名比插件调用少一参 | FlagGems / 插件版本配对 | 纯 MoE 在该镜像组合下不可用 |
| `USE_FLAGGEMS=0` 无法绕过 MoE 路径的 flag_gems 调用 | vllm-plugin-FL 猴子补丁 | 官方承诺的回退开关在 MoE 路径失效 |

**须尽快确认目标模型清单中是否含混合注意力架构**。若含，属跨路线阻塞，需与厂商对接，优先级高于路线选择本身。

### 7. 待办要点（不含排期）

- 用新线镜像（gems 5.0.0 / vllm 0.20.2）复测纯 MoE。插件按 5 参签名编写，本次失败很可能只是与 gems 4.2.1rc0 的过期配对；可先静态检查 5.x 的 `_kunlunxin/fused/topk_softmax.py` 是否已补 `renormalize`
- 若仍失败，临时包装该函数吃掉多余参数，仅为打通到 `fused_experts`，取得 MoE 内核首份数据。**标注为探针，不进交付路径**
- 向智源提交上述三项 issue（附 file:line），建议待复测结论出来后一并提交
- 补 A/B 同条件性能对比（同卡同模型同 vllm 版本），当前为空白
- 推进 B 线补丁上游合入，降低长期 rebase 负担

---

## 四、B 线晋升门槛

B 若要重新评估升为主线，**须同时满足**：

| 条件 | 当前状态 |
|---|---|
| 私有算子路径可达，不再依赖 shim 兜底 | 未达成 |
| 端到端性能不低于 A 的约定阈值 | 无数据 |
| 长序列 prefill 可用（≥2k token 正常完成） | 未达成 |
| 主要补丁上游合入，fork 数显著下降 | 9 项待处理 |
| 上游 CI 具备端到端模型测试（非仅算子级） | 未达成 |
| 目标芯片（至少昇腾 + 一家）有 CI 与端到端验证 | 仅昇腾有部分验证 |

---

## 五、对外口径

> 运行时组的交付分两条线：
>
> **生产交付走 FlagOS 官方栈**——各芯片使用厂商设备插件，其上为 FlagGems、FlagCX、vllm-plugin-FL。该组合有官方镜像、官方安装流程与上层组件 CI 支撑，昇腾与昆仑芯均已实测跑通 dense 推理，训练链路也只在这条线上存在。
>
> **torch_fl 作为自研设备层的验证与上游贡献线并行推进**，不占用交付关键路径。它在选定路线内确实承载显存池、进程组等运行时能力，但目前不是 FlagOS 各层的共同基础——上层组件对其零引用，各芯片设备层现状均为厂商插件。是否升为主线，按晋升门槛评估。

---

## 六、遗留风险

1. **混合注意力架构模型在昆仑芯上 A/B 均不可用**，缺口在厂商 SDK。目标模型若含此类架构，是当前最高优先级阻塞。
2. **FlagOS 版本矩阵管理松动**：triton 版本在 dev 容器、官方镜像、backends.yaml 三处声明各不相同；官方安装 skills 的 vendor-mappings 无 kunlunxin 条目。选 A 不等于免维护。
3. **A 线自身并非无风险**：vllm-plugin-FL 有 22 个 P0/P1 open issue，verl-FL 近 90 天仅 1 个 commit。选 A 换来的是约 50 倍的求助面，不是稳定性保证。
4. **B 线的战略价值未评估**：本指南基于技术与工程成本作出判断。若组织层面存在"拥有设备层"或"在 FlagOS 上游获得话语权"的诉求，应单独立项讨论并明确资源，而非以技术必要性论证。
