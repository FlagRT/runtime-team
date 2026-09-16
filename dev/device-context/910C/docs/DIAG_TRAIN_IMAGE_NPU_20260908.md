# 训练腿锁定镜像 NPU 初始化失败 · 排查记录

> 时间：2026-09-08 ｜ 容器：`flagos-proto-train-910c`（镜像 `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`）
> 对照：`flagos-proto-infer-910c`（镜像 `vllm-ascend:v0.20.2rc1-a3`，**同机同驱动挂载，NPU 正常识别 16 卡**）

---

## 1. 现象

```bash
docker exec flagos-proto-train-910c python3 -c "import torch,torch_npu; print(torch.npu.device_count())"
# → 0
# 伴随错误：
#   DrvMngGetConsoleLogLevel failed. (ret=4)
#   path string is NULL
#   UserWarning: Can't get ascend_hal device count
```

而推理腿容器同样命令输出 `16`，可正常分配显存与计算。

## 2. 已排除的因素（逐项取证，非推测）

| # | 排查项 | 结论 |
|---|---|---|
| 1 | 镜像自带 torch 为 `2.10.0+cpu`、**无 torch_npu** | ✅ 已补装（pip `torch_npu==2.10.0.post2` → 无效；改 `2.10.0` → 仍无效） |
| 2 | CANN 环境变量未加载 | ❌ 非此因：`source /usr/local/Ascend/cann-9.0.0/set_env.sh` 后仍失败 |
| 3 | LD_LIBRARY_PATH 顺序/内容 | ❌ 非此因：换成推理容器的完整 LD_LIBRARY_PATH（893 字符）仍失败 |
| 4 | `SOC_VERSION` 缺失（推理容器有 `ascend910_9391`，训练容器无） | ❌ 非此因：单独/组合设置均无效 |
| 5 | `PYTHONPATH`（CANN python 包）与 `GEMS_VENDOR` 干扰 | ❌ 非此因：`env -u PYTHONPATH -u GEMS_VENDOR` 后仍失败 |
| 6 | `/atb` 日志目录不存在（报 `mki_log mkdir /atb`） | ❌ 非此因：创建并赋权 `/atb` 后仍失败 |
| 7 | driver 挂载缺失或不一致 | ❌ 非此因：两容器 `/usr/local/Ascend/driver/lib64/driver/libascend_hal.so`（1.8MB，同版本）、`/dev/davinci*`、`/dev/davinci_manager` 完全一致 |
| 8 | torch_npu wheel 与镜像不匹配 | ❌ 非此因：**把推理容器里可用的 torch_npu 整体复制**到训练容器后仍失败 |
| 9 | ATB（nnal）差异 | ❌ 非此因：两容器均有 `/usr/local/Ascend/nnal` 与 `libop_plugin_atb.so`；移开该文件会直接报资源缺失（说明它是必需的，但不是失败原因） |
| 10 | `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（训练容器独有） | ⚠️ **相关但非根因**：置 1 后 torch 去加载 `flagcx` 后端扩展并抛 `Failed to load the backend extension: flagcx`（镜像设了 `FLAGCX_TORCH_BACKEND=flagos`）；去掉该变量后不再崩溃，但 device_count 仍为 0 |

## 3. 当前结论

**训练腿锁定镜像内 NPU 无法初始化**，已排除 wheel、环境变量、驱动挂载、ATB、日志目录等常见因素。错误信息集中在驱动管理层：

```
DrvMngGetConsoleLogLevel failed (ret=4)  →  ascend_hal 拿不到设备数
```

同一台机器、同一驱动挂载、同一 CANN 9.0 的推理腿镜像完全正常，因此**差异指向该训练镜像内部**（用户态驱动组件版本、镜像自带 torch 构建与昇腾扩展的匹配关系，或镜像打包时的组件缺失）。

**这不是我们设备上下文/原型代码的问题** —— 同一份 `runtime/` 原型代码在推理腿锁定镜像上 37/37、13/13、6/6、推理腿自验证 10/10 全过。

## 4. 建议处理

1. 由总组将本记录转给负责镜像的方向（performance 方向负责两个锁定镜像的发布与维护），确认：
   - 镜像为何未内置 torch_npu；
   - `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + `FLAGCX_TORCH_BACKEND=flagos` 的预期用法（是否要求走 flagcx 后端、flagcx 扩展为何加载失败）；
   - 是否需要提供内置 torch_npu 的训练镜像版本。
2. 在镜像修复前，2 卡微调可用**旧训练容器**（`flagos-910c-train-850`，torch_npu 可用）预跑，结果注明"非锁定镜像，不作为验收依据"。

## 5. 附：环境事实（供复现）

| 项 | 训练腿 | 推理腿 |
|---|---|---|
| 镜像 | flagrt/ascend-operator-runtime-comm:0.1.3（CANN 9.0 / torch 2.10.0+cpu / flagcx 0.13.0 / torch_fl 0.1.0） | vllm-ascend:v0.20.2rc1-a3 |
| python | /usr/local/python3.11.15/bin/python3 | 同 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0（我们补装） | 2.10.0+cpu / 2.10.0（镜像自带） |
| device_count | **0** | **16** |
| 驱动 | 25.5.0（ascendhal 7.35.23），两容器挂载一致 | 同 |
