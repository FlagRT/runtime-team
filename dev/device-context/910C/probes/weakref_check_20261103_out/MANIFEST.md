# 910C 退出期 weakref UAF 复核 · 读数清单（2026-11-03）

目的：核对「昇腾 torch_npu 在 torch 2.10 同组合踩过同坑并已修」（Ascend/pytorch **PR #41596**）是否成立、
修法是什么、以及**本机是否也复现** —— 为 PPU 同缺陷（`PPU_VENDOR_WEAKREF_ISSUE_20261026.md`）提供对照臂。

环境：机器 `npu1-27`（910C）· 容器 `dc-probe-priv`（镜像 `flagrt/ascend-operator-runtime-comm:2.0.0-…-torch2.10`）·
**只挂 `davinci6/7`**（`ASCEND_RT_VISIBLE_DEVICES=0` → 物理 die6）· Python 3.11.15 ·
`torch=2.10.0+cpu` ｜ `torch_npu=2.10.0`（系统）/ **`2.10.0.post7`**（`/workspace/pkgs_npu_post7`，A/B 臂）

> ⚠️ **退出码一律是「进程 rc」**，不是脚本返回值 —— 崩溃发生在 `main()` 之后、解释器最终化阶段；
> 日志末行恒为 `reached end, normal exit ...`，此时 `rc=139`。

## A. 判别矩阵（系统 torch_npu 2.10.0，各 2 轮，**不提前 gc**）

| 变体 | 对象 | 挂 weakref | 保留强引用 | rc（#1 / #2） |
|---|---|---|---|---|
| **`wr_stream`** | `torch.npu.Stream` | 是 | 否 | **139 / 139** |
| `no_wr_drop` | Stream | 否 | 否 | 0 / 0 |
| `wr_hold` | Stream | 是 | **是** | 0 / 0 |
| `wr_tensor` | Tensor | 是 | 否 | 0 / 0 |

⇒ **三必要条件与 PPU 侧完全同构**：① 对象是 Stream ② 挂 weakref ③ 被回收。缺一即 0。

## B. ⭐ A/B：只换 torch_npu 版本（torch 2.10.0 **不变**）

| 臂 | torch_npu | 用例 | rc | `callback_fired`（Ascend 官方回归判据） |
|---|---|---|---|---|
| **A** | `2.10.0` | `ver_check.py` | **139** | **False**（called=0） |
| **B** | **`2.10.0.post7`** | `ver_check.py` | **0** | **True**（called=1） |

同轮 Event 判据：A 臂 `event_callback_fired=False` / B 臂 **`True`** ⇒ 与 PR 描述一致（`Event.cpp` + `Stream.cpp` 两处都改）。
`wr_stream` 变体重复：A 臂 139 / B 臂 **0**；`wr_tensor`、`wr_hold` 两臂均 0（无变宽效应）。

**判读**：PR #41596 的修复**确实进入 2.10.0.post7**，且**只在 torch_npu 侧**（torch 版本未动）⇒
对 PPU 的含义：这是**后端插件侧**要补的两行，不需要换 torch 主版本。

## C. 判据有效性对照（防「判据自己撒谎」）

| 对照 | 结果 | 说明 |
|---|---|---|
| `wr_cb_ctrl.py`：普通 Python 对象挂 weakref + `gc.collect()` | **回调触发 True** | 回调机制本身正常 ⇒ 上面的 False 不是脚本问题 |
| `ref()() is None` | **两臂都是 None** | ⚠️ 印证 Ascend 测试注释：**该写法在缺陷存在时也返回 None，不能用作判据**；只有「回调是否触发」能区分 |

## D. 文件

| 文件 | 内容 |
|---|---|
| `wr_check.py` | 判别矩阵脚本（`wr_stream`/`no_wr_drop`/`wr_hold`/`wr_tensor`） |
| `wr_callback.py` / `wr_cb_ctrl.py` | 回调判据 / 普通对象空白对照 |
| `ver_check.py` | 版本自证 + Stream & Event 双回调判据（A/B 用） |
| `env_probe.py` | 环境探针（torch / torch_npu 版本、device_count、is_available） |
| `{wr_*,no_wr_drop,wr_hold,wr_tensor}_r{1,2}.{log,rc}` | 判别矩阵 8 次运行 |
| `A_cb_base` / `A_str_base` / `B_cb_post7` / `B_str_post7` / `B_tensor_post7` / `B_hold_post7` / `ver_base` / `ver_post7` `.log/.rc` | A/B 各臂 |

## E. ⚠️ 顺带记下的环境事实（本轮踩到，影响后续所有 910C 上机）

宿主机 **10-24 21:41 重建过设备节点**（`/dev/davinci*` 时间戳）后，**非 `--privileged` 容器拿不到设备**：

| 容器 | privileged | 设备挂载 | `acl.init` / `device_count` |
|---|---|---|---|
| 旧 `dc-rerun-1027-910c`（10-27 曾正常） | false | davinci6/7 | **rc=500000 / 0**（`torch.npu.device_count()==0`） |
| 同镜像 + privileged + **只挂 davinci6/7** | **true** | davinci6/7 | **rc=0 / 16** ✅ |
| 对照：他人容器 `flaggems-cann9.0.0` | true | 全 16 卡 | rc=0 / 16 |

- 报错形态极易误判为「卡坏了/被占了」：`Failed to obtain the number of devices` /
  `chipType=0 does not support get device msg feature` / `set default device id failed, ret:507899`。
- **不是「必须挂全 16 卡」**：只挂自己那两张 + `--privileged` 即可 ⇒ 带卡名额口径（只挂自己 die）不变。
- 处置：重建容器加 `--privileged`（已起 `dc-probe-priv`）；旧 `dc-rerun-1027-910c` 已按此替换。
