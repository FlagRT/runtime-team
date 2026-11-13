# 壁仞 BR_0 · 卡 0 / 卡 2 异常复核报告（供上游提交 · 2026-11-11）

> 机器：`wz-server008`（Biren 测试机，8 × Biren166M）｜ 环境：Ubuntu 22.04.5 LTS / 内核 5.15.0-119-generic / biren 驱动 **1.12.0**（SUPA 1.12）
> 复核时间：2026-11-11 10:23–10:30 CST ｜ 复核人：hliu553（device-context 方向）
> 证据：`BR0/probes/biren_card02_20261111_out/card02_timeline_20261111.log`（journalctl 内核日志导出，239 行，md5 `1245114e…`）＋ 本报告引用的原文
> 结论一句话：**问题属实。卡 2 与卡 0 在 11-10 下午约 3 小时内各自发生「smmu 上下文故障 → SPC 核挂起 → 驱动恢复失败 → 热复位」，复位后两卡至今（11-11 10:25）无法正常注册/读数（brsmi 全 N/A），且卡 2 在 11-09 重启时已有前兆。当前可用 6/8 张。**

---

## 1 现象（当前态，2026-11-11 10:25 brsmi 原文）

| 卡 | Bus-Id | 温度/功耗/利用率 | 显存 | 判定 |
|---|---|---|---|---|
| **0** | 00000000:03:00.0 | **全 N/A** | **128 MiB 残留占用**（无对应进程） | ❌ 不可用 |
| 1 | 00000000:05:00.0 | 38°C / 118W / 0% | 448 MiB（他人进程 bge-large-en） | ✅ |
| **2** | 00000000:63:00.0 | **全 N/A** | 0 MiB | ❌ 不可用 |
| 3–7 | 65/83/85/A3/A5 | 正常 | 正常 | ✅ |

健康卡与故障卡的日志量对比（本开机周期内，journalctl 统计）：

```
bus 03:00.0（卡0）: 总行数 311，其中 ERR/fail/hang/reset 相关 115 行
bus 63:00.0（卡2）: 总行数 506，其中 ERR/fail/hang/reset 相关 110 行
bus 05/65/83/85（健康卡）: 各 63 行，仅 1 行（正常的 SIGKILL 清理记录）
bus A3/A5（卡6/7）: 0 行
```

## 2 卡 2（0000:63:00.0）完整时间线

**前兆（11-09 22:09，开机初始化）**：机器 11-09 22:08 重启，所有卡初始化时卡 2 走过一次 `br_hot_reset:[2] BDF=63:00.0` 后 `device Biren166M init success`——开机即需要一次复位才能初始化，属**前兆**（健康卡无需此步）。

**故障 1（11-10 14:19–14:24）**——首次完整故障链：

```
14:19:47  [ERR ] smmu [4] context [66] receive fault [4]
          [ERR ] BDF 63:00.0 Unhandled context fault(Write): fsr=0x402, iova=0x55727a350000, fsynr=0x740030, cbfrsynra=0x6042, cb=66
14:22:58  [ERR ] PQM hang recovery fail （×4）
14:22:59  [ERR ] found spc hang, the hang spc mask is 0x1 （×3）
14:23:02  hot reset has been triggered （×4）
14:23:59  recovery fail, will do hot reset lately
14:24:01  no process in device, will do hot reset soon.
14:24:01  Biren hot reset start → 14:24:07 Biren hot reset end
```

**故障 2（11-10 15:11–16:10）**——复位后 ~47 分钟再次同型故障：

```
15:11:26  [ERR ] smmu [4] context [64] receive fault [4]（iova=0x5615b5a37000）
15:11:49  [ERR ] PQM hang recovery fail
…（中间持续 queue busy / 恢复尝试）
16:10:10  [ERR ] smmu [4] context [67] receive fault [4]（iova=0x55882bcd0000）
16:10:18  detect queue busy, trying to recover
16:10:29  [ERR ] PQM hang recovery fail
16:10:31  hot reset has been triggered（伴随一个进程收 SIGTERM）
```

**故障后（11-10 17:01 起）**——复位后无法注册，持续至今：

```
17:01:29  [ERR ] can not register, detect exceptions in device
17:01:31  [ERR ] can not register, detect exceptions in device
（此后该卡再无成功初始化记录；brsmi 读数 N/A）
```

**故障特征**：三次 smmu 故障的 iova（0x55727a350000 / 0x5615b5a37000 / 0x55882bcd0000）各不相同，context 编号（66/64/67）也不同——**不是单点地址问题，是反复的 DMA 写上下文故障**；每次都触发 `PQM hang recovery fail` + `spc hang mask`（0x1 / 0x10001），驱动级恢复失败，热复位后 **47 分钟内复发**，最终 `can not register` 卡死。

## 3 卡 0（0000:03:00.0）完整时间线

卡 0 开机初始化正常（11-09 22:09 `device Biren166M init success`，无前兆复位）。11-10 白天只有正常的 SIGKILL/SIGTERM 清理记录，**直到 17:11 单次爆发**：

```
17:11:42  [ERR ] detect CORE hang and trying to reset (hcqd mask=0x11)
17:11:42  MCU0 hang @HCQD4；MCU1/2/3 unknown OP
17:11:42  GSE eng is busy!（寄存器 dump：0x00105c00 busy）
17:11:42  Die1 SPC0 寄存器 dump：SPC_BUSY_SIG0:0xffff0000 / SIG1:0x181d0000，多组 swc pc 停在同一地址
17:11:42  debug bus path: /var/log/biren_debug_bus_0000:03:00.0.log   ← 厂商调试转储已生成
17:11:44  hot reset has been triggered
17:11:55  [ERR ] PQM hang recovery fail（×2）
17:11:55  [ERR ] found spc hang, the hang spc mask is 0x10001（×2）
17:11:57  hot reset has been triggered（×2，hcqd_id 0 与 4 两队列）
```

**故障后**：与卡 2 不同，卡 0 没有打 `can not register`，但 brsmi 从此读数全 N/A + **128 MiB 显存残留**（brsmi Processes 表中无任何对应进程）——设备未完成重注册、残留状态未清。`/dev/biren/card_0` 节点 mtime 停在 11-09 22:09（其他卡同刻），说明字符设备未重建。

**注意**：驱动在 17:11:42 自动生成了 `/var/log/biren_debug_bus_0000:03:00.0.log` 寄存器转储——**上游分析时应一并取走该文件**（需 root，本次未读取）。

## 4 两卡故障的共同特征与差异

| 维度 | 卡 2 | 卡 0 |
|---|---|---|
| 首次故障时间 | 11-10 14:19 | 11-10 17:11 |
| 触发信号 | `smmu context fault (Write)` ×3（不同 iova/context） | `detect CORE hang (hcqd mask=0x11)` + `MCU0 hang @HCQD4` |
| 挂起诊断 | `spc hang mask 0x1` | `spc hang mask 0x10001` |
| 恢复结果 | `PQM hang recovery fail` → hot reset → **47 min 复发** → 最终 `can not register` | `PQM hang recovery fail` → hot reset ×2 → 读数 N/A + 128 MiB 残留 |
| 开机前兆 | 有（11-09 开机即需 br_hot_reset 才初始化） | 无（初始化正常） |
| 复发次数 | 2 次（14:19 / 15:11）+ 持续 register 失败 | 1 次 |

共同点：**PQM（队列管理器）挂起恢复全部失败，均走到热复位，复位均未能恢复正常工作态**。差异：卡 2 是 smmu DMA 写故障驱动型、反复复发；卡 0 是计算核（CORE/SPC）挂起型、单次爆发。

## 5 时间上与用户负载的相关性（供上游定位参考）

- 卡 2 首障（14:19）前 1 秒，同一总线记录 `thread pid 1412797 receive SIGTERM`（14:18:06）；故障窗口内多个进程收到 SIGKILL/SIGTERM（12:38–14:23）。
- 卡 2 二障（15:11）与三障（16:10）之间、卡 0 故障（17:11）当时，机器上有多个 benchmark 容器在跑（xbench 系列，11-10 11:00/15:00 启动的两批）。
- **我方（hliu553）当时无任何进程在机上**——本方向 11-11 09:53 才首次登录。故障与我们的操作无关。
- 以上仅为时间相关性，**不作归因**；负载内容是其他团队的 benchmark 任务（我们不掌握其细节）。

## 6 建议处置（对上游）

1. **取走驱动自动转储**：`/var/log/biren_debug_bus_0000:03:00.0.log`（卡 0 故障时自动生成；检查同目录是否有卡 2 的同型文件）。
2. **优先查卡 2**：有开机前兆 + 两次复发 + register 卡死，疑似硬件/链路级问题；卡 0 单次爆发，可在卡 2 结论后复查。
3. 尝试 `brsmi reset`（工具含 reset/drain 子命令）或整机重启，观察两卡能否恢复注册；若重启后仍 `can not register`，基本可定硬件工单。
4. 本机当前**可用卡 1/3/4/5/6/7（6 张）**，接入工作不受阻塞。

## 7 证据清单

| 证据 | 位置 | 说明 |
|---|---|---|
| 内核日志完整时间线 | `BR0/probes/biren_card02_20261111_out/card02_timeline_20261111.log` | 239 行（14:19–11-11 10:30，两卡全部 ERR/fault/hang/reset/PQM/smmu/MCU 行），md5 `1245114e…` |
| 复核脚本 | `BR0/probes/biren_card02_20261111_out/biren_card02_deep.sh` | 10 段只读采集（brsmi/dmesg/journalctl/lspci），`bash -n` 通过 |
| 普查报告（首轮） | `BR0/docs/BIREN_BR0_RESOURCE_SURVEY_20261111.md` | 11-11 09:53 首轮发现 + 环境快照 |

> 纪律说明：本报告只登记现象、时间线与命令原文，不作硬件/驱动归因结论（归因属上游）；时间相关性段落已明确标注"不作归因"。
