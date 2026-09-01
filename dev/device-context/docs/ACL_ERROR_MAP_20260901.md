# ACL 错误码映射表建设记录（D10 错误码翻译）

> 状态：✅ 阶段成果（2026-09-01）｜ 作者：Kistich ｜ 路线：A 线（torch_npu）
> 关联：`INFERENCE_P3_SERVE_STATE_ERROR_20260901.md`（A9 验证）、`conformance/errors.py`
> 工具：`benchmarks/inference/gen_acl_error_map.py`、`audit_error_map_coverage.py`

---

## 0. 结论先行

| 阶段 | 映射表覆盖 | 分级不一致 | 高置信误判 |
|---|---|---|---|
| 起点 | 1/159 = **0.8%** | 27.3%（36/132） | 24 个 |
| ① 高置信差异录入后 | 20/132 = 15.2% | 6.8%（9/132） | **0** |
| ② 多域扩展 + 人工裁决后 | 49/159 = 30.8% | 9.4%（15/159） | **0** |
| ③ **高置信固化后（当前）** | **103/159 = 64.8%** | 11.3%（18/159）* | **0**（另有 3 条为有意裁决，见 §4.1） |

\* 其中 3 个"不一致"是人工有意覆盖规则（硬件单元 exception 等级对齐），非误判。

回归：conformance **13/13** + 推理 **6/6** 全 PASS，F1 用例仍正确判为 `L2_PARAM(code=161002)`。

**关于覆盖率上限**：剩余 56 条为规则无关键词可依的 `default` 条目，保持兜底 L3（与历史行为一致，不引入风险）。
64.8% 是"有依据可声明"的合理上限，继续堆覆盖率需要用无依据的猜测填充，反而可能引入错误。

---

## 1. 关键认知：不需要"攒错误示例"

D10 的瓶颈一度被认为是"没有昇腾错误样本"。实际上 **CANN 头文件就是权威错误码全集**，且**分散在多个域**：

| 域 | 文件 | 条数 | 注释形态 |
|---|---|---|---|
| rt（运行时） | `acl/error_codes/rt_error_codes.h` | 132 | 带 `//` 英文语义 |
| **aclnn（算子）** | `aclnn/opdev/op_errno.h` | 26 | **无注释**，需从宏名推导 |
| op_log | `aclnn/opdev/op_log.h` | 1 | 无注释（其余为日志级别常量，非错误码，已过滤） |
| ge（图引擎） | `acl/error_codes/ge_error_codes.h` | 0 | 仅含头保护/可见性宏，无错误码 |

样例：

```c
#define ACL_ERROR_RT_STREAM_NO_CB_REG  107015  // callback not register to stream
#define ACLNN_ERR_PARAM_NULLPTR        161001   （无注释，宏名即语义）
```

**教训：先找权威数据源，再谈覆盖率。** 早期只扫 `error_codes/` 一个目录，漏掉整个 aclnn 算子域。

---

## 2. 三层验证策略（不依赖逐个真实触发）

| 层 | 手段 | 说明 |
|---|---|---|
| **L1 覆盖审计** | 头文件 → 映射表覆盖率与分级差异 | `gen_acl_error_map.py` + `audit_error_map_coverage.py` |
| **L2 翻译链路验证** | 构造错误消息 → 跑 `translate_error` → 校验分级 | 可批量自动化，**无需真实触发** |
| **L3 真实触发验证** | 抽样，A/B 单变量对照 | 如 107015（见 P3 执行记录） |

L2 层让 159 个错误码**全部可验证**，而不只是触发过的那 1 个。

---

## 3. 规则的两处子串误匹配陷阱（已修，勿回退）

| 错误码 | 误判 | 根因 |
|---|---|---|
| `507038 DIE_MODE_CHANGE_ERROR` | 误判 L4_FATAL | `"can not c**hang**e"` 含子串 `hang` |
| `507042-045 *_TRAP_*_OVERFLOW` | 误判 L2_PARAM | 含 `overflow`，实为**硬件执行期陷阱** |

**修法**：① 改用词边界匹配 `re.search(r"\b" + kw + r"\b", text)`；② `trap/exception/abort` 规则前置到 L2 之前。

> 这两处若未发现，会污染共享资产 `errors.py`。**教训：工具产出的数据写入前必须抽查命中依据。**

---

## 4. 人工裁决原则：按责任方归属定级

规则判 low 置信的 27 条，按下述原则裁决：

| 责任方 | 归属 | 处置语义 |
|---|---|---|
| 调用方用错（用法/契约违反） | **L2_PARAM** | 重试与重放均无意义，上抛 |
| 环境资源可恢复（等待释放） | **L1_RESOURCE** | 可重试 |
| 执行期失败（算子内部 / 硬件 trap） | **L3_EXECUTION** | 留一次重放机会 |
| 硬件致命（AI Core 异常） | **L4_FATAL** | 由 R2 探针评估决定是否真需重建 |

代表性裁决：

- **`ACLNN_ERR_INNER_*`（15 条）统一 L3**：责任在算子包/部署配置而非调用方参数，重试无效、重建无据。
  顺带修正 `561001 INFER_SHAPE_ERROR` —— 它曾被 `_MESSAGE_HINTS` 的 `"shape"` 关键词误判为 L2。
- **`507015 AICORE_EXCEPTION` → L4**：硬件级异常；归 L4 是安全的，因为 R2 会先探针，健康则不重建（不会盲目重建）。

### 4.1 有意覆盖规则的三条（审计会报为"差异"，非误判）

固化 high 置信条目时发现一处**等级不一致**：我裁决 `507015 AICORE_EXCEPTION` 为 L4，
但规则把同类的 `AICPU_EXCEPTION` / `VECTOR_CORE_EXCEPTION` / `FFTS_PLUS_EXCEPTION` 判为 L3
（因为它们命中了前置的 `exception` → L3 规则）。同类硬件计算单元的 exception 不应有等级差。

统一原则：

| 形态 | 归属 | 理由 |
|---|---|---|
| `*_EXCEPTION`（`AICORE` / `AICPU` / `VECTOR_CORE` / `FFTS_PLUS`） | **L4_FATAL** | 硬件单元异常，交 R2 探针评估是否真需重建 |
| `*_TRAP_EXCEPTION` | **L3_EXECUTION** | kernel 越界访问，属软件问题，不触发设备重建 |
| `*_TIMEOUT` | **L3_EXECUTION** | 超时，重放有机会成功 |

据此把 `507018` / `507035` / `507049` 由规则的 L3 上提为 L4。
**审计工具会持续把这 3 条报为"高置信差异"——这是预期结果，勿回退。**

---

## 5. F5 分级可观测：让"未覆盖"可见

映射表覆盖率有限（30.8%），未覆盖的码会静默兜底为 `L3_EXECUTION`。若上层（尤其 D11 恢复决策）把兜底 L3 当定论，可能对致命错误跳过恢复。

因此给 `FlagosError` 增加两个字段：

| 字段 | 取值 | 含义 |
|---|---|---|
| `mapped` | `True` / `False` | 是否命中厂商错误码映射表（确定分级 vs 保守兜底） |
| `graded_by` | `code_map` / `message_hint` / `default` | 分级来源，可信度递减 |

```python
fe = translate_error(exc, location="device:0")
if not fe.is_grade_confident:
    ...  # 保守策略：不因"看起来是 L3"就跳过恢复评估
```

实测效果：

| 输入 | 分类 | mapped | graded_by |
|---|---|---|---|
| `161002` | L2_PARAM | True | code_map |
| `107015` | L2_PARAM | True | code_map |
| `107999` | L3_EXECUTION | **False** | default |
| `device lost`（无码） | L4_FATAL | **False** | message_hint |

**设计取向：可观测性优先于覆盖率**——宁可知道自己不知道，也不要用兜底值冒充确定结论。

---

## 6. 剩余缺口（当前状态）

| # | 缺口 | 规模 | 处置 |
|---|---|---|---|
| 1 | 分级不一致（多为 low 置信建议，规则本身存疑） | 15 个 | 待逐个裁决或迭代规则；其中 3 个是有意覆盖规则（见 §4.1），非缺陷 |
| 2 | `default` 无关键词可依 | 56 个 | **暂缓，且不建议强推**：兜底 L3 与历史行为一致，不引入风险；
| | | | 强行给无依据的码定级反而会用猜测覆盖保守默认值 |
| 3 | high 置信固化 | — | ✅ **已完成**（54 条写入，覆盖率 30.8% → 64.8%） |

**64.8% 是"有依据可声明"的合理上限。** 剩下 56 条之所以不动，不是做不到，而是做了不划算——
未覆盖的错误码有 F5 的 `mapped=False` 兜底标注，上层已知其不确定性，风险可控。

---

## 7. 复现方法

```bash
docker exec -it flagos-infer-910c bash
cd /mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/inference

# 提取 CANN 错误码全集（多域，含无注释域）+ 规则建议分级
python3 gen_acl_error_map.py --out acl_error_map_candidate.json

# 审计当前 errors.py 的覆盖率与分级差异
python3 audit_error_map_coverage.py --candidate acl_error_map_candidate.json \
    --out acl_error_map_audit.json
```

**维护纪律**：CANN 升级后错误码会变，应重跑上述两步并复核差异。
