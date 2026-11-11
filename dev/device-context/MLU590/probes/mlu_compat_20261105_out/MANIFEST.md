# MLU590 兼容性核验 · 原始证据清单（2026-11-05）

> 机器：`Mlu-1` = `tza-0a06-ai01-em9`（验收机）· `Mlu-2` = `tza-0a06-ai02-em9`
> 时间：**15:55:14 – 16:07:07 CST** ｜ 选卡：`MLU_VISIBLE_DEVICES=2`（采集前后 `cnmon` 8 卡全 `0 MiB`、`No running processes found`）
> 容器：本方向自有 `dc-mlu590-hliu553`（`NetworkMode=host`；锁定镜像 `flagos-runtime-cambricon-neuware4.4.3:2.2.0`，ID `sha256:e887e0ca4333…`）
> 结论报告：`MLU590/docs/MLU590_COMPAT_RECHECK_20261105.md`；对照表同步至 **v0.11**。
> 纪律：按时间戳分文件、**只追加不覆盖**；失败尝试**保留作反证，不得删**；未动他人容器、未改共享资产与默认配置。

## 1 文件清单

| 文件 | 用途 | 是否结论依据 |
|---|---|---|
| `l2_vendor_160008.out` / `l2_gems_160008.out` | L2 两进程：注册层 dump + 数值基线；gems 侧坐实 `ENABLE_OK` 与 `aten::mm/add/silu` dispatch 变化、`aten::matmul` 未变 | ✅ |
| `op_vendor_160334.out` / `op_gems_160334.out` | **9 算子独立判据**（每算子 try/except，同 seed 同输入） | ✅ |
| `fwd_vendor_160334.out` / `fwd_gems_160334.out` | **Qwen3-Embedding-0.6B 真实前向**（snapshot `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`） | ✅ |
| `dispatch_map_160707.out` | 9 算子 enable 前后 `PrivateUse1` 注册归属对照（8 变 / 1 不变） | ✅ |
| `l2_*_155928.out` | 第一次尝试：**缺 `PYTHONPATH=/opt/triton`** ⇒ `ModuleNotFoundError: No module named 'triton'` | ❌ 反证（保留） |
| `l2_gems_160008.out` | 第二次尝试：补齐 PYTHONPATH 后 `enable()` 生效，但算子执行报 `task_type` KeyError | ✅（含失败栈） |
| `op_*_160233.out` / `fwd_*_160233.out` | 第三次尝试：`python3 -` 未传模式参数 ⇒ `IndexError: list index out of range`（脚本自身问题，与芯片无关） | ❌ 反证（保留） |
| `dispatch_map_160612.out` | step5 首版（占位打印未修） | ❌ 保留对照 |
| `l1_probe.py` / `vendor_path.py` / `gems_path.py` / `l2_vendor.py` / `l2_gems.py` / `op_parity.py` / `forward_parity.py` / `dispatch_map.py` | 各阶段探针源码（最后一次执行版） | — |
| `scripts/mlu_compat_step{0..5}.sh` | 6 份可复跑采集脚本，均 `bash -n` 通过 | — |

## 2 脚本 md5 与期望读数

| 脚本 | md5 | 关键期望读数 |
|---|---|---|
| `mlu_compat_step0.sh` | `7aabab71d8c636dfe5d91b9374acbc41` | `CNMON v6.5.52 / Driver v6.5.52`（Mlu-1）；8 卡 `0 MiB` |
| `mlu_compat_step1.sh` | `bfdb5a702b890855ba4d81a71835ece6` | 容器 `Ubuntu 22.04.5 LTS`、`torch 2.7.1+cpu`、`Neuware Version 4.4.3`、镜像 ID `sha256:e887e0ca4333…`；Mlu-2 `v6.2.29` |
| `mlu_compat_step2.sh` | `8d165db027fd1f556137786eb40c57b6` | vendor 路径 `RegisterMLU.cpp:43487` + `flag_gems_in_sys_modules: False`；不挂卡 `RuntimeError: No MLU devices are available` |
| `mlu_compat_step3.sh` | `55f4c6fb21b16ae70f297685991d57e0` | `ENABLE_OK`；`REG_CHANGED aten::mm True` / `aten::matmul False` |
| `mlu_compat_step4.sh` | `d5a16cfedf50af4477b98c485010f12c` | 9 行 `ROW:`（gems 侧 4 行 `ok=false`）；`FORWARD vendor ok=true` / `FORWARD gems ok=false` |
| `mlu_compat_step5.sh` | `15b1b6615b2fa0328dca316b1931df60` | 9 行 `DISPATCH_ROW`，8 行 `changed=True`、1 行 `changed=False` |

复跑命令（在能连到 `Mlu-1` 的机器上）：

```
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step0.sh
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step1.sh
ssh Mlu-2 'bash -s' < scripts/mlu_compat_step1.sh
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step2.sh
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step3.sh
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step4.sh
ssh Mlu-1 'bash -s' < scripts/mlu_compat_step5.sh
```

⚠️ 复跑会按新的 `TS` 生成新文件，不会覆盖本清单里的证据。

## 3 判据口径（第三方自查用）

1. **唯一变量**：vendor 与 gems 是两个**独立进程**；同 seed、同输入、同 `MLU_VISIBLE_DEVICES=2`、同 `PYTHONPATH=/opt/triton`，差别只有是否 `import flag_gems; flag_gems.enable()`。
2. **能设 ≠ 有效果**：必须看 `torch._C._dispatch_dump_table(op)` 的 `PrivateUse1` 行是否由 `RegisterMLU.cpp:43487` 变为 `/dev/null:63`，不能只看 `enable()` 没抛异常。
3. **数值对照覆盖面**：只取输出前 8 个元素 + 全张量 sum + NaN/Inf 计数，**不是全张量逐元素比对**。
4. **`gemm64` 不得过度解读**：dispatch 已变、数值与厂商逐字一致、未触发 triton 报错，但**没有单独证明执行的是 FlagGems 的 GEMM 实现** ⇒ 报告标「实现归属待证」，不作为「可替换」证据。
5. **失败根因只登记现象与栈**：`triton.testing.do_bench → autotuner.kernel_call → jit.run` 抛 `KeyError: task_type` / `BLOCK_ROW_SIZE`；不归因到厂商/网络层，不升格为厂商缺陷。

## 4 边界

- 本轮**未重跑 11-03 的 9 段判据**（范围裁定，不等于证明无回归）；11-03 归档**未记录当次 `cnmon` 读数**，不反推其驱动版本。
- `dev/images/cambricon-mlu-runtime/v1/demo/env.baseline.yaml` 仍保留 `flag_gems` 的原始表述：**`demo/` 按 ADR-0008 永久冻结，不改**；现行口径以 `lock.yaml` / `image.baseline.yaml` 为准。
- 驱动 `6.2.29` 是 09-22/10-08/10-14 归档的实测值、也是 `Mlu-2` 现值 ⇒ 历史文档**不改写**；只在现行验收机口径登记 `6.5.52`。
- 选档依据「6.2.x ⇒ 只能 `neuware4.4.3` 档」对 `Mlu-1` 条件已变，但本方向未在 4.7.2 档验证 ⇒ 不换档、不改锁定镜像，待总组/镜像方向重评。
