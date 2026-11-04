# 910C「谁在实现算子」专项核对证据（2026-11-04）

> 问题：910C 我方路径是否真的全走**厂商算子库**？「不用 FlagGems」是声明还是事实？
> 结论：**是事实**（三层独立证据），并**更正**脚本里一句错误因果（`unset VLLM_PLUGINS` ≠ 禁用插件）。
> 正文见 `docs/CHIP_COMPAT_MATRIX_20261104.md` **§3.1e**。

## 三层证据

| 层 | 做法 | 结果 |
|---|---|---|
| **注册层** | `torch._C._dispatch_dump_table(op)` 看 `PrivateUse1` 实现来源 | 我方路径下 `mm/add/silu/matmul` **全部** `torch_npu/csrc/aten/RegisterNPU.cpp:33542`；手动 `enable()` 后 `mm`/`add` 变 `flag_gems/__init__.py:62`，但 **`matmul` 仍厂商** |
| **运行期** | 跑真实前向看 `'flag_gems' in sys.modules` | **False**（连模块都没加载） |
| **服务端** | `VLLM_PLUGINS` 三态 + 真实服务日志 | unset/`fl` ⇒ `vllm_fl`+`flag_gems` **被 import 但 dispatch 不变**；`""` ⇒ 都不加载。真实日志为 `Platform plugin ascend is activated` + `SERVE_STANDARD_PASS`，`flag_gems|flaggems|vllm_fl|flagos` **零命中** |

## 关键更正（声明 vs 实现）

vLLM 源码 `vllm/plugins/__init__.py`（L32/L56）：`allowed_plugins is None or plugin.name in allowed_plugins`
⇒ **`VLLM_PLUGINS` 未设 = 加载全部插件**；**空串 = 一个都不加载**。
⇒ 脚本原注释「unset = 禁用 vllm-plugin-FL」是**错误因果**，已更正（代码行为未改）；机制实为
① `fl` 平台插件在 ascend 上不激活（无 ascend 后端）；② 即便 general 插件被 import 也不调 `enable()`。

## 文件

| 路径 | 说明 |
|---|---|
| `probe_dispatch.py` · `out/p1_dispatch.log` | 注册层：enable 前后的 `PrivateUse1` 实现对照 |
| `probe_vllm_plugins.py` · `out/p2_plugins_{unset,empty,fl}.log` | `VLLM_PLUGINS` 三态下插件/platform 解析 |
| `probe_plugin_effect.py` · `out/p3_plugin_effect.log` | **关键**：插件加载后 dispatch 是否改变（判据 `dispatch_changed` 全 false） |
| 真实服务日志 | `910C/probes/r10_serve_910c_npu_20261008_out/tp2_eager1/vllm_serve.log`（`Platform plugin ascend is activated`） |

⚠️ 未做的事：改默认值为 `VLLM_PLUGINS=""`。属行为变更，需一次服务回归（`SERVE_STANDARD_PASS` + 消费方 10/10）才可改。
