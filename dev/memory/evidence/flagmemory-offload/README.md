# evidence/flagmemory-offload —— KV 卸载实测原始证据

显存方向 KV→Host 卸载线的**实测原始数据本体**（逐腿 JSON、服务端与压测链日志、DMA marker 账本、
采样探针与起服务脚本）。结果报告（PLAN / RESULT / 通俗版）不在本目录，在
[`../../docs/goals/`](../../docs/goals/) 对应批次目录下；本目录只放报告所引用的原始数据。

- **证据与组件代码分仓**：卸载栈与显存池的组件代码在独立仓 `git@github.com:FlagRT/FlagMemory.git`
  （依据 ADR-0014 §3.1 / §3.5 第 3 条）。该仓 `reports/INDEX.md` 是指向本目录的索引。
- **机器指代纪律**：报告与索引一律用「P800 节点 / 910C 节点 / PPU 节点」指代，地址与账号不入仓（铁律 7）。
  原始日志为引擎原样输出，见 §3「引用与复核」。

## 1. 批次一览

| 批次目录 | 测期 | 芯片 / 栈 | 内容 | 对应报告 |
|---|---|---|---|---|
| [`p800_results_1025/`](p800_results_1025/) | 2026-10-25 | P800（昆仑芯）+ 910C（昇腾）双腿 | `proto/` = P800 逐腿 JSON + 服务端/链日志 + DMA marker 账本；`proto-910c/` = 910C 对照腿同构数据 + 腿本脚本 + 画像执行器 | [`proto-p800-202610/`](../../docs/goals/proto-p800-202610/)：`PLAN_protocol_run_20261025.md`、`RESULT_规范验证跑_{P800,910C,通俗版}_20261025.md` |
| [`p800_results_1027/`](p800_results_1027/) | 2026-10-27 / 28 | P800（昆仑芯），真模型 OneRec-8B | `proto_lock/` = 64 GiB 锁定池档（含预注册锚 `ANCHOR_lock64_20261027.md`、OFF/ON 两趟与 `ON_pass2` 复跑、`OFF_pass1_contaminated` 为按预注册判据作废的那趟）；`proto_nat/` = 自然池档 OFF/ON 两趟 + 引擎日志；`kit/` = 起服务脚本与显存采样探针 | [`proto-p800-202610/`](../../docs/goals/proto-p800-202610/)：`PLAN_规范验证跑_真模型_20261027.md`、`RESULT_规范验证跑_真模型{,_通俗版}_20261027.md` |

**2026-10-23 / 24（P800 诊断与四步闭环、真模型 A/B）与 2026-10-26（PPU 开关冒烟）两批次没有原始数据条目** ——
这两批的交付物只有结果报告，分别在
[`proto-p800-202610/`](../../docs/goals/proto-p800-202610/) 与
[`proto-ppu-202610/`](../../docs/goals/proto-ppu-202610/)。

## 2. 目录与文件命名约定

| 形态 | 含义 |
|---|---|
| `OFF/` · `ON/` | 卸载开关的两个对照臂 |
| `B-c<并发>-r<轮次>-{ON,OFF}.json` | 批量（throughput）腿的逐跑结果，`c` = 并发数、`r` = 同档复跑轮次 |
| `L-c<并发>-{ON,OFF}.json` | 长上下文 / load 侧腿的逐跑结果 |
| `serve-*.log` · `*_engine.log` | 引擎（vLLM）服务端日志 |
| `chain*.log` | 压测链客户端日志 |
| `*marker_{diag,register,swap,transfers}.txt` | DMA marker 账本：诊断 / 预注册 / 换入换出 / 传输逐批计数 |
| `protocol_runner*.py` · `probe_meminfo.py` | 画像规范执行器与显存采样探针 |
| `serve_*.sh` · `chain_*.sh` · `switch_engine.sh` · `probe_rss.sh` | 起服务 / 跑链 / 切栈 / RSS 采样的腿本 |
| `*_pass<n>` · `*_contaminated` | 同档复跑与按预注册判据作废的趟次（作废趟保留，供复核判据执行是否一致） |

## 3. 引用与复核

- **从报告引用证据**：报告与其原始数据不同目录，引用写相对路径
  `../../evidence/flagmemory-offload/<批次目录>/<子目录>/<文件>`。
- **从 FlagMemory 仓引用证据**：引用本仓路径
  `dev/memory/evidence/flagmemory-offload/<批次目录>/…`，并带本仓提交哈希才可核对
  （证据是追加式的，只认提交，不认分支名）。
- **日志里的 `<节点地址>` 占位符**：`serve-*.log` / `*_engine.log` 第一段有引擎启动时打印的
  `distributed_init_method=tcp://<节点地址>:<端口>` 一行。本仓是**公开仓**，节点地址按铁律 7
  不入仓，故该段已脱敏为占位符，端口与行内其余字节未动，共 7 处。
  该行不参与任何判据（判据量取自逐腿 JSON 字段、DMA marker 账本与引擎日志的池结算行），
  脱敏前后 marker 行数、池 token 数、DMA 账本 total 与全等率素材计数全部相同。
  口径与前后对照见 FlagMemory 仓 `docs/refactor/03-证据与代码分家.md` 第五节与 7.7。
- **新增证据的前置动作**：本仓公开，**证据入仓前必须先过泄漏核查**
  （IPv4 地址 / 主机名与用户家目录 / 账号形态 / 口令与密钥形态四类），核完才允许 `git add`；
  命中先脱敏再入仓。依据同上文件 5.4。
- **完整性校验**：每个批次目录下的文件清单与聚合校验和见
  FlagMemory 仓 `docs/refactor/03-证据与代码分家.md` 第七节（分「脱敏前基线」与「脱敏后（现行）」两组）。
