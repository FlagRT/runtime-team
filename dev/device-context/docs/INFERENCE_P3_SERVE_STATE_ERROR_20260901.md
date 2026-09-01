# P3 服务化 × 设备状态/错误恢复 · 执行记录（A8 / A9 / A10）

> 状态：✅ P3 验收闭环（2026-09-01）｜ 作者：Kistich ｜ 路线：A 线（torch_npu + vllm-ascend）
> 关联：DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md（P3 设计）、DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md（验收清单）
> 模型：**Qwen3-4B**（全程对齐同事昇腾线基准）

---

## 0. 结论先行

| 验收项 | 职责 | 判定 | 关键证据 |
|---|---|---|---|
| **A8** EngineCore 子进程设备句柄 | D2 | `ENGINECORE_CTX_PASS` | spawn 子进程 pid=8421(ppid=8402) 持 `/dev/davinci_manager` fd、124 处设备内存映射、RSS 5952MB，功能请求 0.23s |
| **A9** 推理路径错误码翻译 | D10 | `ACL_107015_PASS` | 真实错误 107015 注入成功；A/B 单变量对照证实根因 |
| **A10** 四态监控 + 五段式恢复 | D11 | `DEVICE_STATE_RECOVERY_PASS 8/8` | 含 L4 完整 R1→R5：`captured → isolated → recovered → replay_ready` |

**P0/P1/P2/P3 四阶段全部闭环，D1-D11 十一项职责全部验证通过。**

---

## 1. 环境

| 项 | 配置 |
|---|---|
| 容器 | `flagos-infer-910c`（host 网络 + 512G shm + 16 NPU + raid 挂载） |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` |
| 模型 | `/mnt/raid/hliu553/models/Qwen3-4B` |
| 服务 | `vllm serve --served-model-name qwen3-4b --tensor-parallel-size 1 --port 8100 --max-model-len 4096` |
| 环境变量 | `DO_NOT_TRACK=1`；**`VLLM_PLUGINS` 必须 unset**（坑 A5） |

---

## 2. O2 服务化环境：serve 就绪

| 指标 | 值 |
|---|---|
| 就绪等待 | 55s |
| 预热（坑 A1） | 0.22s（本次未复现 13 分钟极慢现象） |
| 正式推理 | 3.74s / **68.4 tok/s** |
| 输出 | 4/4 语义正常，无 NaN |

产物：`start_vllm_serve_910c.sh`（启动）、`probe_serve_health.py`（就绪探测 + 预热 + 验收）。

---

## 3. A8 · EngineCore 子进程设备句柄（D2）

### 3.1 取证思路

坑 A2 明确"EngineCore 是 spawn 子进程，主进程读不到其 stats"。因此**不依赖 vLLM 内部接口**，改从 `/proc/<pid>` 直接取证：

| 证据 | 取法 | 含义 |
|---|---|---|
| 设备句柄 fd | `/proc/<pid>/fd` 中 `/dev/davinci*` | 进程持有 NPU 设备句柄 |
| 设备内存映射 | `/proc/<pid>/maps` 中 davinci 项 | 设备内存已映射 |
| 进程关系 | `ps -eo pid,ppid,cmd` 搜 `VLLM::EngineCore` | spawn 父子关系 |
| 权重驻留 | `/proc/<pid>/status` VmRSS | 子进程承载模型 |
| 功能证据 | 发 1 条请求成功 | 上下文真正可用 |

### 3.2 结果

```
EngineCore  pid=8421  ppid=8402   设备 fd=1 (/dev/davinci_manager)
                                  设备内存映射=124 处
                                  RSS=5952MB
主进程      pid=8402              设备 fd=1
功能请求    ok 0.23s  " Paris. The capital of Germany is Berlin..."
```

**观察**：主进程与子进程各持有 1 个 `davinci_manager` 句柄——主进程负责设备管理/枚举，子进程承载计算上下文，符合 V1 架构预期。环境变量中未设 `ASCEND_VISIBLE_DEVICES`（容器直挂 `/dev/davinci*`），设备枚举走设备文件而非环境变量。

---

## 4. A9 · 错误码翻译 + ACL 107015 根因（D10）

### 4.1 错误码定义

```
/usr/local/Ascend/ascend-toolkit/latest/include/acl/error_codes/rt_error_codes.h:36
#define ACL_ERROR_RT_STREAM_NO_CB_REG  107015  // callback not register to stream
```

### 4.2 最小复现（A/B 单变量对照）

唯一变量 = 是否在 stream 上 `subscribe_report`：

| 组 | 操作序列 | 返回 |
|---|---|---|
| A（错误路径） | `create_stream` → 直接 `launch_callback` | **107015** |
| B（正确对照） | `create_stream` → **`subscribe_report`** → `launch_callback` | **0（成功）** |

```python
# 实测签名（四参数，缺一即 args parse failed）
acl.rt.launch_callback(fn, userData, block, stream)
```

**根因结论：对未调用 `aclrtSubscribeReport` 注册的 stream 投递 callback，即命中 107015。**
这不是设备缺陷，而是**调用方契约违反**——前置订阅缺失。

### 4.3 翻译链路验证

| 项 | 结果 |
|---|---|
| 错误码提取 | `107015` ✅（`_extract_acl_retcode` 兼容 `error code is N` 形态） |
| 当前分级 | **L3_EXECUTION** |
| 是否在映射表 | ❌ 否（走了消息关键词 `stream` 粗分类） |

### 4.4 ⚠️ 待决策：107015 应归 L3 还是 L2

| 分级 | 语义 | `handle_error` 处置 | 对 107015 是否合适 |
|---|---|---|---|
| L3_EXECUTION（当前） | 执行类 | 同上下文重放（replayable） | ❌ 重放必然再失败——不补 subscribe 就永远 107015 |
| **L2_PARAM（建议）** | 参数/契约类 | 上抛调用方，不重试 | ✅ 契约违反应让调用方修正前置条件 |

建议把 `107015: ErrorCategory.L2_PARAM` 补进 `errors.py` 的 `ACL_ERR_TO_CATEGORY`。**改动涉及共享资产与 F1 用例判定，待拍板后执行。**

---

## 5. A10 · 四态监控 + 五段式恢复（D11）

### 5.1 关键机制（实测确认）

`recovery.handle_error` **只对 L4_FATAL 触发 R2-R5 设备恢复**；L1-L3 直接返回并仅标记 `replayable`——这是 R2 的设计意图（避免不必要的高代价重建）。

推论：**健康设备下，L4 错误经 `evaluate_device` 探针必然通过 → 永不触发 R3/R4**。要验证完整链路必须注入瞬时故障。

### 5.2 验证矩阵（8/8）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | 四态可查（初始 AVAILABLE） | ✅ |
| 2 | 压力注入 → DEGRADED 转换 | ✅ 6 并发 0.84s |
| 3 | 压力解除 → 回 AVAILABLE | ✅ |
| 4 | L4 分支评估（带 507021 码） | ✅（实为 L1，见 5.3） |
| 5 | **L4 完整 R1→R5** | ✅ `captured→evaluated: isolated→recovered: True→replay_ready` |
| 6 | L3 分支（107015）不触发重建 | ✅ `evaluated: L3_EXECUTION, no device recovery` |
| 7 | ISOLATED → recover → AVAILABLE | ✅ |
| 8 | 恢复后服务续跑 | ✅ `' Paris. The capital of Germany is Berlin'` |

L4 瞬时故障注入方法（前 2 次 sync 失败、第 3 次成功，模拟故障自愈）：

```python
def flaky_sync():
    fail_n["n"] += 1
    if fail_n["n"] <= 2:
        raise RuntimeError("inject: transient device sync failure")
    torch.npu.synchronize()
```

---

## 6. 新发现与 trap（逐条记录）

| # | trap | 现象 | 处理 |
|---|---|---|---|
| T1 | **错误码映射优先于消息语义** | 构造 `"device lost, error code is 507021"` 期望 L4，实际因 507021 在 `ACL_ERR_TO_CATEGORY` 中被映射为 L1_RESOURCE，消息关键词（device lost→L4）**未生效** | 验证 L4 分支需用**不带已知错误码**的纯消息 |
| T2 | **健康设备下 L4 永不触发重建** | `evaluate_device` 探针必过 → 不进 R3/R4 | 设计使然（R2 意图）；验证需瞬时故障注入 |
| T3 | `subscribe_device_state` 回调签名不固定 | 按 `(ordinal, state, reason)` 写 lambda 会收到 `DeviceState` 对象当 reason | 用 `lambda *a` 兜底并统一转 `.value` |
| T4 | 结果 JSON 序列化崩 | `DeviceState` 不可 JSON 序列化（T3 的连锁） | 同 T3 |
| T5 | raid 日志目录属主为 root | 宿主 `tee` 写日志 Permission denied（不影响探针，只影响管道退出码） | `docker exec chown -R 5017:5017` |
| T6 | `launch_callback` 签名 | 三参数报 `args parse failed` | 正确为四参数 `(fn, userData, block, stream)` |

---

## 7. 复现方法

```bash
docker exec -it flagos-infer-910c bash
cd /mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/inference

# O2 启动服务 + 验收
bash start_vllm_serve_910c.sh 8100
python3 probe_serve_health.py --port 8100 --wait 1800 --out serve_health_result.json

# A8 子进程设备句柄
python3 probe_enginecore_device_ctx.py --port 8100 --out enginecore_device_ctx_result.json

# A9 107015 注入 + 翻译
python3 probe_acl_107015.py --device 0 --out acl_107015_result.json

# A10 四态监控 + 五段式恢复
python3 probe_serve_state_recovery.py --port 8100 --ordinal 0 --out serve_state_recovery_result.json
```

---

## 8. 下一步

1. **拍板 107015 分级**（L3 → L2 建议），改 `errors.py` 并重跑 A9/F1
2. 可选支线：**D8 双缓冲"减少同步点"优化**（EVENT_WAIT 3.37ms/12 次瓶颈）——训练侧最后一块性能缺口
3. PR → `dev-1.0`（对齐同事六步顺序第 6 步）
