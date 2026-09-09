# 跨芯片一致性测试套件（昇腾基线）

> 对应收尾计划第 3 点。以昇腾 910C 为**第一个基线**，建立跨芯片一致性测试套件雏形：
> 同一套"行为契约用例"运行于全部已接入插件，**行为差异即缺陷**（允许性能差异）。
> 全程 **torch_fl（flagos 设备）**，不 import torch_npu；所有操作在对应芯片容器内执行。

## 目录

| 文件 | 说明 |
|---|---|
| `runner.py` | 运行框架：收集 `case_*` 用例 → 逐个执行 → JSON 结果 → 汇总判定 |
| `cases.py` | 首批行为契约用例（S1/S2/E1/E2/T1/T2/F1 代表用例 + 契约覆盖清单） |

## 运行（昇腾基线）

```bash
# 容器内、tf-venv-integration 激活
cd conformance
python runner.py --chip ascend --out conformance_ascend_result.json
# 预期输出：
#   [PASS] s1_stream_order: ...
#   [FAIL] xxx: 行为差异即缺陷
#   === 汇总: N/M 通过 ===
#   CONFORMANCE_PASS / CONFORMANCE_PARTIAL
```

## 结果格式

`conformance_ascend_result.json`：

```json
{
  "chip": "ascend",
  "env": {"torch_fl": "...", "devices": 2},
  "cases": {
    "s1_stream_order": {"ok": true, "detail": "..."},
    "f1_error_translation": {"ok": true, "detail": "..."}
  },
  "summary": {"passed": 7, "failed": 0, "total": 7}
}
```

## 为下一款芯片接入（关键路径）

1. 在目标芯片环境（如寒武纪 CNRT 插件）拉起 torch_fl/flagos，确认 `flagos.device_count()` 可用
2. 运行同一套用例：`python runner.py --chip cambricon --out conformance_cambricon_result.json`
3. **比对**：两份 JSON 中同一用例的 `ok` 字段
   - 昇腾 ok=true、新芯片 ok=false → 新芯片插件行为缺陷，定位映射表并修正
   - 允许的性能差异不在此判定（性能另录 profiling）
4. 新用例扩展：在 `cases.py` 新增 `def case_<name>(ctx)`（返回 `(ok, detail)`），runner 自动收集
5. 结果与接入报告一并提交 PR 到 runtime-team

## 契约覆盖与扩充方向

首批用例覆盖行为契约的代表项（见 `cases.py` 的 `CONTRACT_COVERAGE`）：

| 契约 | 用例 | 状态 |
|---|---|---|
| S1 顺序保证 | case_s1_stream_order | 首批 |
| S2 显式依赖 | case_s2_explicit_dependency | 首批 |
| E1 事件记录/等待 | case_e1_event_record_wait | 首批（接口缺口时退化为近似） |
| E2 先 wait 后 record | case_e2_wait_before_record | 首批（接口缺口时标注） |
| T1 锁页前置 | case_t1_pinned_async_copy | 首批 |
| T2 在途保护 | case_t2_inflight_protection | 首批（外部行为） |
| F1 三维翻译输入 | case_f1_error_translation | 首批 |
| S3/S4、E3/E4、T3、F2-F4、R1-R5 | 待扩充 | 后续批次按契约逐条补 |

## 诚实标注（接口缺口）

torch_fl 当前为框架层（PrivateUse1 机制），部分统一接口尚未暴露：

- **统一事件句柄**（E1/E2）——未暴露时用例退化为同步近似并标注
- **统一错误对象**（F1 的位置投影）——当前记录框架层异常原文作为三维翻译输入证据
- **在途登记表/状态机事件**（T2/R 系列）——运行时层职责，框架层不可观测

这些"接口缺口"本身是有价值的验证结论：**跨芯片一致性要求统一接口完备**，缺口清单应
进入 torch_fl 后续接口补全的排期，也是设备执行上下文职责"从设计到落地"的差距清单。
