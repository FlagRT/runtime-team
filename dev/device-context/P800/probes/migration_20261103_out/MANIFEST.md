# P800 服务器迁移证据（2026-11-03）

> 背景：老机 `43.180.254.67:26008` 于 **2026-11-07 到期**，迁至新机 `81.70.153.29:26008`（`P800-new1`）。
> 正文见 `../../docs/KUNLUN_P800_MIGRATION_20261103.md`。

| 文件 | 说明 |
|---|---|
| `container_spec_old.json` | 老容器 `hliu553-device-context-p800` 的完整 `docker inspect`（镜像 id / 8 卡设备 / `/dev/xpuctrl` + `/dev/fuse` / shm 64 GiB / 两个挂载）—— 新机重建容器的依据 |
| `new1_infer_leg_result.json` | **新机推理腿结果（归位前，数据在家目录）**：`INFER_LEG_PASS 13/13` · dim 1024 · 区分度 `0.6392` · 52.59 句/s · p50 56.67 ms |
| `new2_infer_leg_result_data2.json` | **归位后复验（数据在 `/data2/hliu553`，容器挂载已改）**：`INFER_LEG_PASS 13/13` · 区分度 `0.6392` · 52.58 句/s · p50 55.92 ms ⇒ 归位未改变读数 |
| `xpu_mem.py` | 逐卡显存解析脚本（`xpu-smi` 输出解析，避开 awk/perl 引号地狱） |

## 关键事实（复核用）

- 模型权重 sha256（新机取自 hf-mirror 后核验）：`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`，与老机/910C/MLU590 逐字节相同。
- 代码与证据：两侧同口径文件数 **1286 = 1286**（排除 `__pycache__`）；归位到 `/data2/hliu553` 后
  **1337 = 1337 文件 / 1,221,715,797 = 1,221,715,797 字节**（rsync 后核对再删中转副本）。
- `/data2/hliu553` 由**容器内 root** 创建并 `chown 1017:1018`（新机 docker 组可用，**无需 sudo 口令**；uid/gid 实测，非默认 1000）。
- ⚠️ **新机卡 1、卡 2 在分配显存时挂死**（卡 0、卡 3 正常）；`xpu-smi` 显示 0 MiB 不代表可用。
- ⚠️ 新机建容器**必须挂 `/dev/xpuctrl`**，否则 `torch.cuda.is_available()==False`（看似"没有卡"）。
