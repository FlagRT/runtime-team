# 910C 框架格修复证据：`torch_npu 2.10.0` → `2.10.0.post7`（2026-11-04）

> 背景：镜像内 `torch_npu 2.10.0` 缺 Ascend PR #41596（子类 `tp_dealloc` 未链式清理父类）⇒ weakref 悬空 ⇒ 退出期 SIGSEGV。
> 本轮用 **env 层 overlay** 验证「升级到 `2.10.0.post7` 能修、且不动读数」。正文见 `docs/CHIP_COMPAT_MATRIX_20261104.md` **§3.1c**。

## 读数

| 臂 | `torch_npu` | weakref 回调 | 退出码 | 推理腿 | 区分度 |
|---|---|---|---|---|---|
| A 基线（镜像自带） | `2.10.0` | **False**（`called=0`） | **139** | `INFER_LEG_PASS` 14/14 | **0.6380** |
| B overlay（`PYTHONPATH`） | **`2.10.0.post7`** | **True**（`called=1`） | **0** | `INFER_LEG_PASS` 14/14 | **0.6380** |

空白对照（普通 Python 对象 weakref 回调）：两臂均 `True` ⇒ 排除探针自身问题。
⚠️ 吞吐 78.94 → 66.16 句/s 为**单次**读数且同机他人占卡 ⇒ **不作结论**。

## 文件

| 路径 | 说明 |
|---|---|
| `run_wr1104.sh` | weakref 两臂执行脚本（含空白对照 + 退出码实采） |
| `run_fw_ops.sh` / `run_fw_ops_A.sh` | 推理腿 B 臂（post7）/ A 臂（基线） |
| `weakref_out/{A_base,B_post7}_env.log` | 两臂环境自证（`torch_npu` 版本） |
| `weakref_out/{A_base,B_post7}_cb.log` | **判据本体**：`callback_fired` |
| `weakref_out/{A_base,B_post7}_ctrl.log` | 空白对照 |
| `weakref_out/A_rc.txt` / `B_rc.txt` | 进程退出码（**139 / 0**） |
| `weakref_out/*_verdict.txt` | 逐臂摘要 |

## 介质与形态（重要）

- post7 包树位置：`/home/hliu553/dc-rerun-1027/pkgs_npu_post7/`（**139 MB**，含 `_C.cpython-311-aarch64-linux-gnu.so`；本机无外网亦可装）。
- ⚠️ **形态＝挂载覆盖（env 层）**：镜像内仍是 `2.10.0`。按「挂载覆盖 vs 镜像固化不可折算」，本结论**不等于镜像已修**。
- ⛔ 是否把 post7 纳入基座属**总组裁定**（我方只消费 `stack.lock`），本方向只提供「已有官方修复版本 + 升级零读数变化」两项事实。
