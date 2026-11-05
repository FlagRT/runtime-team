# device-context · 当前状态

更新：**2026-11-05（第三十轮 · MLU590 上机逐格核验 / 兼容性对照表 v0.11）**：**✅18 / 🟡6 / ⚪0 / ⛔0**（24 格不变）。MLU590 操作系统 🟡→✅；唯一 ⚪ 格（FlagGems 替代库）⚪→🟡，**结论是「当前不可替换」而非「测出可用」**；驱动格发现两机分叉（Mlu-1 运行态 6.5.52 / Mlu-2 6.2.29 / dpkg 两台 6.2.29-7）。第二十八轮为「算子库」层拆两行（格数 20→24），第二十九轮为 PPU 归档复核（未上机）。

> 🆕 **2026-11-05（第三十轮 · MLU590 真机核验，15:55–16:07）**：两机 SSH 恢复后按三级采集逐格复核（`Mlu-1`/`Mlu-2`；只取卡 2，前后 8 卡全空闲）。① 容器 OS 实测 `Ubuntu 22.04.5 LTS` ⇒ 操作系统格 🟡→✅；② **FlagGems 结论性验证完成**：`enable()` 生效且 8/9 算子 dispatch 被接管，但 4/9 算子与真实前向因 triton-mlu autotuner 参数失败 ⇒ **当前不可替换**，⚪→🟡；③ 驱动格登记 `Mlu-1 6.5.52` / `Mlu-2 6.2.29` / `dpkg 6.2.29-7` 三值并列，历史值不改写；④ 注册层证据坐实我方路径全走厂商算子库（`RegisterMLU.cpp:43487`、FlagGems 0 命中）。详见 `MLU590/docs/MLU590_COMPAT_RECHECK_20261105.md`。
>
> **2026-11-05（第二十九轮 · PPU 归档复核，未上机）**：三入口实测不可达 ⇒ 本轮零新真机读数，**六格状态判定一格未改**（✅17 / 🟡6 / ⚪1）。更正两处表述：宿主内核 `5.10.134-17.2.al8.x86_64` 与容器 OS `Ubuntu 24.04.2 LTS` 有 10-10 preflight 原文（原记「未采集」不准）；宿主发行版 `Alibaba Cloud Linux 3` 仍缺宿主侧 `os-release` 原文 ⇒ P0 收窄。详见 `PPU/docs/PPU_COMPAT_RECHECK_20261105.md`。

## 近期轮次速览（**本表只给一行结论 + 报告与行号**）

> 完整轮次日志：`prototype/docs/BOARD_ACTION_LOG_202608_202610.md`（**60 项**）；
> 历史状态头部链（8 版）：`prototype/docs/BOARD_ARCHIVE_STATUS_HEAD_CHAIN.md`。
> ⚠️ 本方向纪律：**同一事实只在一处复写**（台账第 36 条）—— 看板只留结论 + 指针，明细一律在报告里。

| 轮次 | 结论（一行） | 详见（报告 + 行号） |
|---|---|---|
| **MLU590 上机逐格核验 / 对照表 v0.11 · 11-05（第三十轮）** | ① **两台 SSH 恢复**，15:55–16:07 完成 L0/L1/L2 三级采集（只取卡 2，前后 `cnmon` 8 卡全空闲、未动他人容器）。② **操作系统格 🟡→✅**：宿主 `Ubuntu 20.04.6 LTS` / `5.15.0-139-generic`，容器 `/etc/os-release` 实测 `Ubuntu 22.04.5 LTS`（原为包名推断）。③ ⭐ **唯一 ⚪ 格结案**：`flag_gems 5.4.0rc2.post3` 带卡后 `enable()` 生效、**8/9 算子 dispatch 由 `RegisterMLU.cpp:43487` → `/dev/null:63`**（`aten::matmul` 未覆盖），但 **4/9 算子执行失败**（`KeyError: task_type` ×3 / `BLOCK_ROW_SIZE` ×1，栈在 triton autotuner）且 **Qwen3-Embedding-0.6B 真实前向失败** ⇒ **当前不可替换**，判 🟡（**不是测出可用**）。④ **驱动格内容更正**：`Mlu-1` 运行态 `6.5.52`（cnmon / `/sys/module` / `modinfo` 三处一致，10-28 更新）、`Mlu-2` `6.2.29`、两台 `dpkg` 仍 `6.2.29-7` ⇒ 两机分叉 + 包元数据与运行态不同源；09-22~10-14 归档的 `6.2.29` 与 `Mlu-2` 现值仍成立，**历史值不改写**。⑤ 容器版本（torch 2.7.1+cpu / torch_mlu 1.29.2 / NeuWare 4.4.3 / 镜像 ID `sha256:e887e0ca…`）与基线逐字一致；L2 `device_count=1`、9 算子厂商路径与真实前向全 OK ⇒ 现行驱动与锁定镜像可协同。⑥ ⚠️ 三条边界：`11-03` 归档**未记录当次 cnmon 读数**（不反推）、`11-03` 九段判据本轮**未重跑**（范围裁定）、「驱动 6.2.x ⇒ 只能 4.4.3 档」依据对 Mlu-1 已过期但**不换档**（待总组/镜像方向重评）。计数 **✅17→18 / 🟡6→6 / ⚪1→0**（🟡 两处变化抵消） | `MLU590/docs/MLU590_COMPAT_RECHECK_20261105.md` **§0–§7** · `docs/CHIP_COMPAT_MATRIX_20261104.md` **§3.3 / §5.3 N-10 / §6.2 / §6.4 / §8** · `MLU590/probes/mlu_compat_20261105_out/MANIFEST.md` · `dev/images/cambricon-mlu-runtime/v1/{lock.yaml,image.baseline.yaml}` |
| **PPU 归档复核（未上机）· 11-05** | ① **三入口仍不可达**（`:22`/`:2222` `Connection refused`、旧 `:36444` 超时、路由走 `en0` 未进 VPN；15:44–15:46 三次探测）⇒ **零新真机读数，六格状态判定一格未改**（✅17/🟡6/⚪1）。② 更正「宿主 OS/内核未采集」：宿主内核 `5.10.134-17.2.al8.x86_64`、容器 OS `Ubuntu 24.04.2 LTS` 均有 10-10 `preflight_env_ppu_20261010.log` **L7–L13** 原文，且自证该 preflight 跑在**容器内**（⛔ Ubuntu 不得写成宿主 OS）；宿主发行版 `Alibaba Cloud Linux 3` 仍缺宿主侧 `/etc/os-release` 命令原文 ⇒ 🟡 维持、P0 收窄。③ 「具体算子库名未采集」→ 登记路径 `/usr/local/PPU_SDK`（CUDA wrapper cu130）/ `/opt/accl-p`（libpccl）已给出，**运行时 dispatch 归属待上机三层证据**；`PPU/probes/` 启用 FlagGems = **0 命中**，`serve_standard.sh` ppu 分支不需要任何插件。④ ⚠️ `flagtree 0.6.1+ppu3.6` 旧值（10-10 从镜像名误读，10-14 已更正）**显式作废**，v2 以 `0.7.0+ppu.gite223f42c` 为准 | `PPU/docs/PPU_COMPAT_RECHECK_20261105.md` **§1–§4** · `docs/CHIP_COMPAT_MATRIX_20261104.md` **§3.4 / §6.2 / §6.4 / §8** · `PPU/probes/collect_ppu_compat_20261105.sh`（`bash -n` 通过） |
| **四芯片兼容性对照表 v0.9（「算子库」拆两行：实际使用 / 替代）· 11-04** | 本月新交付项：**四芯片 × 五层**（v0.9 起「算子库」拆两行 ⇒ **6 行 × 4 = 24 格**）。v0.1 ✅8 → v0.6 ✅12 → v0.8 ✅13 → **v0.9 ✅17 / 🟡6 / ⚪1 / ⛔0**。① ⭐ **910C 框架格转 ✅（env 层）**：`torch_npu 2.10.0`（缺 Ascend PR #41596）⇒ 退出期 weakref UAF；用 **env 层 overlay** 换 `2.10.0.post7`（介质在本机 **139 MB 包树，无外网可装**）后 **weakref 回调 False→True、退出码 139→0、推理腿两臂均 14/14、区分度两臂同为 0.6380**（升级不动读数）。⚠️ **形态＝挂载覆盖（环境层），镜像内仍是 2.10.0** ⇒ 按「挂载覆盖 vs 镜像固化不可折算」，对外声明必须带形态；是否能固化属**总组基座裁定**。② ⭐ **910C 算子库格补齐逐算子对照**：新探针在 vendor 路径记录**我方前向实际执行的 27 个 aten 算子**，再启用 FlagGems **用同一输入重放**比对 ⇒ **304 条重放 / 184 PASS / 0 数值不一致**，覆盖 18 个算子；同时查出两条**确证缺口**：**启用后真实前向编译期失败**（`ModuleNotFoundError: No module named 'shmem'`，归因 flagtree/triton 后端栈依赖）+ **`triu` 编译失败**（最小脚本可独立复现）。⭐ 并**如实剔除探针自己造成的假缺陷**：`silu`/`lift_fresh` 的重放失败经最小脚本复验为正常，不计缺陷。该格仍 🟡 但性质由「缺证据」→「证据已硬、缺口已定位」。⑤ ⭐ **v0.9 拆行（因原文混装两件事、易误读）**：**「算子库（实际使用·厂商）」**＝验收真跑的路径 （910C: CANN 算子 + `npu_fusion_attention_v3` + **HCCL**｜P800: XTDK/XRE + `xtorch_ops` + **flagcx**｜MLU590: NeuWare + `torch_mlu_ops` + **cncl**｜PPU: SDK 2.1.0 + **nccl**）⇒ **四家全 ✅**；**「算子库（替代·FlagGems 系）」**＝随镜像自带但默认未启用（910C 不可用 / P800 K-5 / MLU590 ⚪ / PPU v2 已修）⇒ 判它＝判「能不能替换掉厂商」。⭐ 顺带更正事实错误：**`flagcx` 原列在 910C 格里，但 910C 通信走 HCCL、不用 flagcx**（flagcx 是 P800 后端）。④ ⭐ **v0.8 专项核对「算式子实现归属」**（三层证据）：注册层 `_dispatch_dump_table` ⇒ `mm/add/silu/matmul` 的 `PrivateUse1` **全为 `torch_npu/csrc/aten/RegisterNPU.cpp:33542`**；运行期 `'flag_gems' in sys.modules` = **False**；真实服务日志 `Platform plugin ascend is activated` + `SERVE_STANDARD_PASS` 且 FlagGems 字样**零命中** ⇒ **910C 我方路径全走厂商算子库**。顺带两事实：**`aten::matmul` 不在 flag_gems 覆盖内**（enable 后仍回落厂商）；⚠️ **更正「unset VLLM_PLUGINS = 禁用插件」的错误因果**（vLLM 语义：未设=加载全部，空串才不加载；脚本注释已改，**行为未改**）。③ ⚠️ 吞吐单跑 78.94→66.16 句/s **不作结论**（单次 + 同机他人负载） | `docs/CHIP_COMPAT_MATRIX_20261104.md` **§3.1c**（框架格修复 A/B）· **§3.1d**（逐算子对照）· §2 主表 · §6.2 待办 | `910C/probes/fw_post7_20261104/MANIFEST.md` · `910C/probes/op_parity_20261104/MANIFEST.md` · `910C/probes/op_path_20261104/MANIFEST.md`（含 `op_parity.tsv` 逐条 / `op_min.log` 最小复验 / `gems_forward.log`）|
| **P800 服务器迁移至新机并归位 · 11-03** | ① 老机 `43.180.254.67:26008` **11-07 到期** ⇒ 数据与模型迁至新机 **`81.70.153.29:26008`（`P800-new1`）**；备用 `152.136.33.169:26008`（`P800-new2`，未放数据）。三台**均 26008 端口、沿用原密钥免密**。② **归位到 `/data2/hliu553`（与老机同名同路径）**：1337 = 1337 文件 / 1,221,715,797 = 1,221,715,797 字节；容器挂载已改为 `-v /data2/hliu553:/workspace`。⭐ **`/data1|/data2` 顶层 root:755** ⇒ 先用**容器内 root** 完成等价 `mkdir/chown`（`docker run --rm -v /data2:/data2`；uid/gid **实测 1017:1018**），随后确认 **sudo 口令与用户名相同**（权限 `(ALL:ALL) ALL`）⇒ 再用 sudo 正式执行一遍（幂等，`rc=0`，属主/属组不符各 **0** 个）。③ **归位后复验**：离线 **43/0/1**、推理腿 **`INFER_LEG_PASS` 13/13**·**52.58 句/s**·区分度 **0.6392**（与老机归档逐字一致）⇒ 迁移不改读数。④ **镜像未搬**（38 GB 镜像新机没有，但链路实测 ≈1.2 MB/s 需 9 h+；新机已预置 `flagrt/kunlun-operator-runtime` 且内含同名 conda env）；模型改新机自建普通目录（老机那份是**别人**的只读共享目录），自 `hf-mirror` 取，权重 sha256 `0437e45c…` 与老机/910C/MLU590 逐字节相同。⑤ ⚠️ 新机两个坑：**建容器必须挂 `/dev/xpuctrl`**（缺则 `cuda.is_available()=False`，像没卡）；**卡 1/卡 2 分配显存挂死**（0/3 正常，`xpu-smi` 仍显示 0 MiB ⇒ 显示空闲 ≠ 能用），**待报厂商**（我方只记现象与判据，不作归因） | `P800/docs/KUNLUN_P800_MIGRATION_20261103.md` **§1–§6** · `P800/probes/migration_20261103_out/MANIFEST.md` （`container_spec_old.json` · `new1/new2` 推理腿结果 · `xpu_mem.py`）|
| **原型 v0.4.0 发布 + 交付仓建立（自建仓报备）· 11-03** | ① **组件 v0.4.0 已发布**（四实例收口版）：新建自建仓 **`FlagRT/flagos-device-context`（public，`main` 单协作线）**，Release `runtime-v0.4.0` 含 `flagos_device_context-0.4.0-py3-none-any.whl`（**可 pip install**）+ `.tar.gz`；发版提交 `984687f`。② **交付形态首次 pip 化**：新增 `pyproject.toml`（分包名 `flagos-device-context` / 导入名仍是 `runtime`），核心包**只用标准库**（`import runtime` 不需要 torch），`[torch]` 为可选 extra；clean venv 实测**从 Release URL 直接装成功**（46 出口、子包全可导入）。③ **双版本轴显式化**：`__version__`=0.1.0（接口轴）/ 新增 `__component_version__`=0.4.0（发布轴），**有意不对齐**。④ **报备四字段 + 版本锚**已按公告写入本文件「交付仓报备」节；`dev/repos.yaml` 待总组收录（已给逐字段对齐的可粘块，本方向不自行改）。⑤ 过渡期：协作仓 `prototype/runtime/` 暂留同一份内容（**已核对新仓逐字节一致，31 文件零差异**）以不打断四实例验收，**v0.5.0 起只推 pip**；⚠️ 在此之前改 runtime 包须两处同步提交 | `docs/REPO_SPLIT_PLAN_20261103.md` **§1–§6** · 本文件 **「交付仓（自建仓）报备与版本锚」节** · https://github.com/FlagRT/flagos-device-context/releases/tag/runtime-v0.4.0 |
| **910C gmu 实测背书 + 三机模型副本对账 · 11-03** | ① **gmu A/B 完成**（宿主 `npu-smi` 外部采样，die4，容器 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`）：`gmu=0.9` ⇒ 运行期峰值 **59,811 MB = 91.3%**（余量仅 5.7 GB）；`gmu=0.4`（新默认）⇒ **28,452 MB = 43.4%**；两臂吞吐 **41.72 / 40.65 句/s**（−2.6%，p50 72.1/72.8 ms 同档）、区分度均 **0.4123**、消费方腿均 **10/10**；⭐ **同卡共存判据**（服务占 die 时跑设备上下文 smoke）两臂均 **52 通过 / 0 失败** ⇒ `serve_standard.sh` 的 ascend 分支从「待实测背书」升级为**已背书**，四支（ascend/kunlun/ppu/cambricon）默认值各自都有本方向实测依据。② ⭐⭐ **三机模型副本 sha256 对账**：`model.safetensors` **三机逐字节相同**（`0437e45c9456…`，1,191,586,416 B），config/tokenizer/vocab/merges/modules 亦一致 ⇒ **区分度 0.6380 vs 0.6391 的 0.0011 差不能归因于模型副本**，只能落在后端数值/算子实现；本轮**只关闭「副本不同」分支，不对具体算子下结论**。差异仅 910C 侧多 4 个非权重文件（ModelScope 整包形态）+ 1 个留证坏件 `*.CORRUPT-20261027`。③ ⚠️ 环境事实（手册坑 25）：**`--privileged` 下挂载白名单不生效** ⇒ 容器内 `ASCEND_RT_VISIBLE_DEVICES=n` 指**物理 die n**（实证：挂 davinci6 的容器设 0 分配 2 GiB ⇒ 物理 die0 涨 2,113 MB）；且 **`npu-smi` HBM 读数有 10~20 s 滞后** ⇒ 峰值须稳定 ≥25 s 后采、释放判据等 ≥22 s | `910C/docs/ASCEND_910C_GMU_AND_MODEL_SHA_20261103.md` **§1–§5** · `910C/probes/gmu_ab_20261103_out/MANIFEST.md` · `910C/probes/model_sha_20261103_out/MANIFEST.md`（含 `weight_sha.txt` · `compare.txt`）· `prototype/scripts/serve_standard.sh` **L338–L381** · `prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md` **坑 25** |
| **MLU590 共享层改动后全套复跑 · 11-03** | ① **单入口 9 段 `overall: PASS`（逐段 rc=0）**：离线 **103/0/0** · 对称性 7/0 · smoke **46/0** · conformance 13 与推理 6 均 PASS · 契约不变式 PASS（rc=0）· 图捕获 **4/4** · 配额 **3/3** · 职责审计 **OK 63 / FAIL 0 / SKIP 15（共 78）** —— 与 10-27 归档**逐项一致**。② **`DC_MODEL` 两腿口径 `LEGPARITY_PASS` 9/0**：推理 13/13·区分度 **0.6391**、训练 50 步 6/6·loss **15.4498→11.1479**（两种传法各一遍）、`|Δloss|=0`。③ ⭐ **weakref 四家矩阵补齐**：MLU590（torch 2.7.1）**未中招** 🟢（Stream/Event 回调均触发、30 次创建回收 rc=0），且 **weakref 可创建（30/30）** ⇒ 此前「2.7 不支持 weakref」的推测**已证伪**，缺陷本质是「父类支持 but 子类 `tp_dealloc` 未链式清理」。④ **服务化 gmu 两轮实测**（宿主 cnmon 外部采样）：0.9→**89.3%**(87,762 MiB) / 0.4→**39.9%**(39,210 MiB)；⭐ 且**预热后计时**吞吐 4.66→4.62 句/s（−0.9%，p50 645/648 ms 同档）⇒ **MLU 上 gmu 无吞吐代价**；区分度 **0.4091** 与 10-14 归档逐字同（0.4118 是 PPU 的值，两实例不可混用）。⑤ 由此把 `serve_standard.sh` 的 `cambricon` 分支从「未实测平移 + 告警」升级为**实测 0.4**，端到端 `SERVE_STANDARD_PASS` | `MLU590/docs/CAMBRICON_MLU_RERUN_20261103.md` **§1–§7** · `MLU590/probes/acceptance_20261103_out/MANIFEST.md` · `MLU590/probes/legparity_20261103_out/MANIFEST.md` · `MLU590/probes/weakref_check_20261103_out/MANIFEST.md` · `MLU590/probes/gmu_ab_20261103_out/MANIFEST.md` · `prototype/scripts/serve_standard.sh` **L338–L377** · `prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md` **坑 24** |
| **PPU v2 分版全量复跑 + 四实例口径复验补齐 · 11-03** | ① **v2 镜像全量 13 段 `overall: PASS`（逐段 rc=0）**，与 10-26（v1）基线**零回归**：训练 50 步 6/6·loss 15.4498→**11.1530**、推理 13/13·区分度 **0.6391**、服务化 10/10·区分度 **0.4118**（三处关键读数与归档**逐字相同**）、审计 OK 73/FAIL 0/SKIP 5、多流 8/8、图捕获 4/4、配额 3/3。② **`DC_MODEL` 两腿口径 `LEGPARITY_PASS` 9/0**（补齐 10-27 遗留）⇒ **四实例口径复验 4/4**。③ **gmu 四臂单变量**（宿主 ppu-smi 外部采样）：gmu 0.9⇒**90.0%** / 0.4⇒**40.0%**，两臂均 PASS ⇒「KV 池是纯余量」在 PPU 成立；⭐ 但 **`max_num_seqs=128` 是对 memory 建议 #1 的部分不采纳**——四臂吞吐 115.51(0.9/256) · 102.13(0.9/128) · 105.79(0.4/128) · 111.21(0.4/256)，区分度全 0.4118，即**它不省显存（显存由 gmu 决定）却掉吞吐 5.1~13.1%** ⇒ embed 形态改为**不传**。④ **核出一处假 FAIL 并修**：`run_t3/t456/t5` 只给宿主视角变量拼了 runid、容器视角没拼 ⇒ 服务臂实际 PASS 却判「未通过」，且 10-14 起每轮证据互相覆盖。⑤ ⚠️ 环境事实：共享盘 `/bmcp_lvm_fs` **100% 满（可用 0）**，本轮对照改用容器 `/tmp` 落盘并在证据里标注 | `PPU/docs/PPU_V2_ACCEPTANCE_20261103.md` **§1–§8** · `PPU/probes/v2_acceptance_20261103_out/MANIFEST.md` · `PPU/probes/legparity_20261103_out/MANIFEST.md` · `PPU/probes/gmu_ab_20261103_out/MANIFEST.md` · `prototype/scripts/serve_standard.sh` **L338–L375** · `prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md` **坑 22/23** |
| **B 组上机：口径复验补齐 910C + gmu A/B 落地 + P800 多流复跑 · 11-03** | ① **910C 两腿 `DC_MODEL` 口径复验 `LEGPARITY_PASS` 9/0**（补齐 10-27 遗留）：推理 14/14（区分度 **0.6380** 与归档逐字同）、**训练腿直接吃 HF 缓存根 6/6、loss 15.4498→11.1479 与归档逐字同**（10-27 时该用法还报 `Unrecognized model`）、两腿解析同一 snapshot、诱饵未命中、`\|Δloss\|=0` ⇒ 四实例口径复验 **3/4**（余 PPU）。② **gmu A/B（P800，同卡单变量，宿主 xpu-smi 外部采样）**：`gmu=0.9` ⇒ 峰值 **91,950 MB / 93.5%**（余量仅 6.3 GB，共卡环境必挤）；`gmu=0.25+seqs128` ⇒ **28,054 MB / 25.1%**，**两臂判据均为 `SERVE_STANDARD_PASS`** ⇒ 印证 memory 建议「embedding 的 KV 池是纯余量」，且**跨实例同向**（910C 91%→43% · PPU 88% · P800 93.5%→25%）；据此把 `serve_standard.sh` 默认改成**分芯片**（ascend 0.4 / kunlun 0.25 / 其余 0.4 未实测告警）。③ **P800 多流基线复跑**（共享层改动后首次）：语义 **8/8** · 图捕获 **4/4** · 配额 **3/3**，与 09-20 归档逐项一致 ⇒ 零回归 | `910C/probes/legparity_20261103_out/MANIFEST.md` · `P800/probes/gmu_ab_20261103_out/MANIFEST.md`（A/B 两节 + 采样曲线）· `P800/probes/multistream_20261103_out/` · `prototype/scripts/serve_standard.sh` **L331–L356** |
| **PPU weakref 缺陷：运行时等价补丁验证有效 · 11-03** | PPU（v2 镜像，torch 2.10.0）缺陷态复核**与 910C 逐条同构**（`Stream+weakref+被回收` **139/139**、Stream/Event 回调均不触发）；厂商**无**含修复版本 ⇒ 照 PR #41596 语义做**运行时等价补丁**（C 扩展在 import 期给 `_CudaStreamBase`/`_CudaEventBase` 换 `tp_dealloc` 包装：先 `PyObject_ClearWeakRefs` 再回跳原函数）⇒ **P1 回调 False→True · P2 rc 139→0（2/2）· P3 功能形态 PASS（`elapsed_ms=56.8` 带内、张量正确）· P4 我方验收零变化**（补丁臂 vs 对照臂同为离线 118/0/1 · smoke 46/0 · 契约不变式 4/4 `rc=0`）。⭐ **四家矩阵（同判据）**：PPU 🔴 / 910C(torch_npu 2.10.0) 🔴 / P800(torch 2.9.0) 🟢 未中招 / MLU590 ⚪ 网络不可达未验证 ⇒ **生态共性缺陷，非某一家实现错误**。⚠️ 补丁定位＝**止血**（不进镜像、不进 runtime 包、不改仓库代码；只读页会被放开可写且不还原），正式修复仍待厂商 | `PPU/probes/weakref_check_20261103_out/README.md` **§0–§5**（含 `wrfix.c` 源码与全部日志/rc）· `910C/docs/ASCEND_WEAKREF_EXIT_UAF_20261103.md` **§5**（四家矩阵） |
| **910C weakref 缺陷复核（A/B 实证）· 11-03** | 核对「昇腾 torch_npu 在 torch 2.10 同组合踩过同坑并已修」⇒ **成立且本机可复现**：`torch 2.10.0 + torch_npu 2.10.0` 下 `Stream+weakref+被回收` **rc=139/139**（其余三变体全 0，与 PPU 侧三必要条件逐条同构）；**单变量 A/B**（torch 不变、只换 torch_npu 到 **`2.10.0.post7`**）⇒ **rc 139→0**，Stream/Event 的 **weakref 回调从「不触发」变「触发」**（Ascend **PR #41596** 修法＝`tp_dealloc` 里补 `PyObject_ClearWeakRefs`）。⭐ 判据修正：**`ref()() is None` 两臂都返回 None，不能当判据**，唯一可靠信号是**回调是否触发**。⚠️ 环境事实：宿主 10-24 重建设备节点后**非 `--privileged` 容器拿不到设备**（`acl.init rc=500000` / `device_count=0`，报错形态酷似"卡坏了"）；**只挂自己那两张 die + `--privileged` 即正常** ⇒ 带卡名额口径不变 | `910C/docs/ASCEND_WEAKREF_EXIT_UAF_20261103.md` **§1–§6** · `910C/probes/weakref_check_20261103_out/MANIFEST.md`（A–E 节）|
| **memory 建议落地 + 看板补账 · 11-03** | ① `serve_standard.sh` **embed 形态默认 `gmu=0.4` + `max_num_seqs=128`**（显式值优先；采纳 memory 建议 #1/#4，依据＝910C 实测「embedding KV 池为纯余量，0.4 吞吐与 0.9 无差」；我方归档互证：910C 55.15/61.27 GiB≈90% · PPU 84.95/96≈88% · P800 传 0.25→23%）；`enforce_eager` **部分采纳**（默认仍 1：TP=2 就绪 50s vs 95s、P800 去 eager p50 反升，均实测）⇒ **待 910C A/B 实测背书**（0.4 vs 0.9 + npu-smi 运行期峰值 + 同卡共存判据）。② memory 四条建议**认领表态**：#1 部分采纳 / #3 采纳（补运行期 HBM 采样）/ #4 采纳参数化 / #2 转调度方向。③ **原型独立仓报备草稿**（`FlagRT/flagos-device-context`，v0.4.0 时按公告四步走）。④ 看板补账：速览表补齐 10-16 后六轮（此前停在 10-16/10-08） | `prototype/scripts/serve_standard.sh` **L338–L375**（参数落地+依据注释）· `docs/REPO_SPLIT_PLAN_20261103.md` · memory 侧 `dev/memory/STATUS.md` **建议表 #1–#4** |
| **PPU v2 生效（总组裁定）· 10-29** | 总组 **ADR-0011** 把 `thead-ppu-runtime v2` 切为现行生效版本（我方 10-27 按组内流程入档发布的镜像）；「服务化 triton 固化」链路收口：拉起镜像即可服务化、容器重建不丢。**我方待办**：在 v2 上跑一轮轻量验收（离线+smoke+服务化）并按 issue 分版三态表态 | `dev/stack.lock.thead-ppu.v1.yaml`（总组改写）· `dev/images/thead-ppu-runtime/v2/README.md` |
| **两腿 DC_MODEL 口径统一 · 10-27** | 共享层缺陷：同一份 `DC_MODEL`（HF 缓存根）**只有推理腿会解析**、训练腿直接 `Unrecognized model`（MLU 复跑实测暴露）⇒ 抽成 `runtime/proto/model_path.py` 两腿共用（优先听 `refs/main`）+ **AST 静态同源守卫**（任一条腿自建 `resolve_model` 即 FAIL，离线 23/0/0）+ 四实例复验驱动（`run_legparity_dcmodel.sh`，判据 P1–P4 含**诱饵 snapshot** 对照）。复验：**P800 / MLU590 `LEGPARITY_PASS` 9/0**（两腿解析同一 snapshot、Δloss=0，loss/区分度与归档逐字同）；⚠️ **910C / PPU 未复验**（当日网络不可达；910C 后续上机还遇容器 NPU init 失败待排）。判据自身两处修正也已记录（torchrun 双 rank 交错骗过 `.*` 抽取；宿主/容器路径需双向回读断言） | `prototype/docs/LEG_MODEL_PATH_PARITY_20261027.md` **§1–§7** · `P800|MLU590/probes/legparity_20261027_out/`（含 MANIFEST） |
| **三实例共享层回归收尾 · 10-26/27** | 共享层四处改动（`create_event(enable_timing)` · `_stream_ctx` 强持有 · C13 三态 · 模型路径探测）三实例复跑**全部零回归**：**P800**（10-26）离线 114/0/1 · 审计 68/0/10 · 推理 13/13（0.6392）；**910C**（10-27）离线 92/0/2 · 审计 **73/0/5** · 推理 14/14（0.6380）· 训练 loss **15.4498→11.1479 与归档逐字同**；**MLU590**（10-27）离线 103/0/0 · 审计 **63/0/15** · 训练同 loss 逐字同。⭐ **C13 假通过翻案**：四家「事件计时不可用」中 **3 家其实是我们没测到**（统一 API 缺「建可计时事件」入口 + 方向反 + 无量纲门），补入口后 910C=10.47ms / MLU=11.38ms 实测正常 ⇒ **仅 P800 是真平台缺口**（裸 torch 恒 0.0）。⭐ **910C 迁移快照模型权重损坏取证**：两腿同 loss=nan，查权重 `model.embed_tokens.weight` 含 410 NaN+1 inf（同文件双读 sha256 一致 ⇒ 内容坏；独立副本差 657,943 字节）⇒ 换副本恢复、坏件留名 `.CORRUPT-20261027`；顺带更正 10-21 迁移记录「逐字节一致」措辞为「字节数相等+抽样」 | `910C/docs/ASCEND_910C_RERUN_20261027.md` · `MLU590/docs/CAMBRICON_MLU_RERUN_20261027.md` · `P800/probes/rerun_20261026_out/MANIFEST.md` · `910C/docs/STORAGE_MIGRATION_20261021.md` **§8** |
| **PPU 全量重跑 + v2 镜像发布 · 10-26/27** | **13 段单入口全量重跑零回归**（`run_acceptance_ppu.sh`：离线 118/0/1 · conformance 13/13+6/6 · 训练 6/6 loss 逐字同 · 推理 13/13 区分度逐字同 0.6391 · 服务化 10/10 · 错误闭环 5/0/0 · 多流 8/8 · 图捕获 4/4 · 配额 3/3 · 审计 73/0/5）；**退出期段错误第五载体收口**（`_stream_ctx` 强持有 ⇒ 契约不变式段 rc=139→0，五载体全收口）；**C13 两处假通过修正**。**v2 镜像按组内流程发布**：triton/flagtree wheel 固化（全新容器零手工 `TRITON_PPU_SMOKE_PASS` 误差 0）· 离线包 17.15 GiB 三方校验一致 · 入档+索引+候选登记；**10-29 总组裁定生效（ADR-0011）**。⚠️ 平台侧 weakref 缺陷转厂商材料已备（`PPU_VENDOR_WEAKREF_ISSUE_20261026.md`，纯 torch 最小复现 4/4 + 三必要条件判别），**发出待定** | `PPU/docs/PPU_RERUN_20261026.md` **§0–§3** · `PPU/docs/PPU_EXIT_SEGV_CARRIER5_20261026.md` · `dev/images/thead-ppu-runtime/v2/README.md` |
| **基座机制适配 · 10-16** | 按总组新发版规范把本方向「叠上去」：① 建**运行环境目录** `910C/env/{train,infer}/`（替换本方向原先已退役的 `-f ../compose.base.yml` 写法），训练腿以 **FlagCX `pip-editable` 挂载覆盖**镜像固化版本；② FlagCX 修复分支收口并推送（`b576fa6a`，含**两处本方向自己**的可编译性缺陷修正：CUDA 头缺宏守卫 / DAG 链接项无条件）；③ 910C 训练腿**实质验证** A/B 对照（同一容器同一命令、只换加载哪份 FlagCX）：挂载覆盖臂 **`overall: PASS`**（八组 + 原生 DDP 四项全 PASS）/镜像固化臂 **`FAIL`**（图案同 issue）。④ **机制路径当日补齐并跑通**（用户级装 compose **v2.29.7**（官方 sha256 校验）+ pyyaml 6.0.2，**未用 sudo**；npu1-11 无凭据可取）：`baseline.py export --require-clean` 双腿 **rc=0**（两份 `env.baseline.yaml` 入仓）· `up.sh <env>` **rc=0** 起容器（compose `Cmd` 逐字=声明的 pip-editable）· **同一单入口验收在机制起的容器里仍 `PASS`** · `down` rc=0 无残留。⑤ ⭐ **`up.sh --strict` 真机验收通过（14:34，die 5+0）**：`image.id` 跨机不同（镜像基线记制造机 npu1-11 `e13a15d0…`，本机 `7028028bb62c…`；本组镜像 `RepoDigests` 为空 ⇒ **从未 push**）⇒ `export` 会把它抄回制造机 id、`check` 恒 FAIL（已实测复现）；合规拿到同一制品的三条路**实测全不通**（离线包实体只在无凭据的 npu1-11 · registry 代理 `127.0.0.1:7890` 失效 · 重建只到功能等价）。⇒ 处置（当日收尾定稿）：**仓内文件记发布 id**（共享分支口径 —— 记本机 id 会让别人 `check` 恒 FAIL），本机要出结论时用幂等工具 `910C/env/pin_local_image_id.sh` **临时**对齐（单字段 / 本机取值 / 回读校验 / 跑完 `git checkout --` 复原），**属对 ENV-SPEC「不手工编辑」的已登记偏离**（机制侧建议见文档 §7.2 第 10 项）。结果：**`up.sh <train> --strict` rc=0** · 容器 `Cmd` 逐字=声明的 pip-editable · **容器内两项验收 `overall: PASS` / rc=0** · `down` rc=0 无残留；推理腿官方镜像按 digest 拉取、id 天然跨机一致 ⇒ **未做任何对齐** | `docs/BASELINE_LAYERS_AND_ADOPTION_20261016.md` **§1–§7** · `910C/env/README.md` **§2.1/§6** · `910C/probes/baseline_strict_20261016_out/SUMMARY.md` · `910C/probes/dc_env_20261016_out/SUMMARY.md` · `910C/probes/baseline_mechanism_20261016_out/SUMMARY.md` |
| **第 4 家上机 + 同口径自审 · 10-10** | **平头哥 PPU**：形态 **路径 B** / 后端名 **`nccl`** 实测钉死；离线 **112/0/1** · 对称 **7/0** · smoke **46/0** · conformance **13/13 + 6/6** · 步 0 模型完整性 PASS · **训练腿 50 步 6/6**（5167 tok/s）· **推理腿 13/13**（135.77 句/s）· **同日同口径自审**补齐 6 个标准步骤（多流 **8/8** · 图捕获 **4/4** · 配额 **3/3** · B/C **7/7** · 环境普查 · 职责审计）。⚠️ **未做：服务化 / 错误闭环 / 非空转**；✅ **职责审计 72/0/6 —— `L1` 根因已定位并修复**（根因＝跨类型地址复用；修复＝类型复核（机制无关）+ 强持有纵深防御；弱引用下亦 3/3 OK）；⚠️ 两条腿**退出期段错误未收口**；同轮修掉 2 处**跨实例共性缺陷**（`gloo` 静默回落 · torchrun 挂死 300 s） | `PPU/docs/PPU_ONBOARDING_PARITY_AUDIT_20261010.md` **§0–§5** · `PPU/docs/PPU_MODEL_AND_TWO_LEGS_20261010.md` **§0–§6** |
| **第 4 家调研 · 10-09** | **平头哥 PPU 接入前调研完成并逐条复核**（8 条 · 含 1 条降级）⇒ 该芯片在生态里是「**组件层三件齐备、运行时层与基座层镜像为 0**」；形态判定**间接证据指向「复用 CUDA 命名空间」**，⚠️ 直接判定**待上机**；⛔ 唯一前置阻塞 = **机器访问**（平台侧 SSH 端口当前无服务） | `PPU/README.md` **L10–L24**（§0 状态）· `PPU/docs/THEAD_PPU_FLAGOS_ADAPTATION_RESEARCH_20261009.md` **L12–L25**（结论）+ **L238–L271**（复核记录） |
| **第十六轮 · 10-08** | 回答「910C / P800 的『流优先级不适用』**属不属于本层职责、能不能修**」⇒ 两家**都不属、都不可修**；⭐ **顺带核出并修掉我方一处口径分叉**（契约 §1.9 表 P800 行自第九轮起漏改，与代码/看板/台账矛盾） | `prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` **§1.9** · `prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md` **第 36 条** |
| **发布 · 10-08** | 发 **`runtime-v0.3.0`**（接口面补齐 §1.6–§1.10 + 8 处行为修正；统一面出口 **28 → 46**，移除 0）· 附件 `runtime-prototype-v0.3.0.tar.gz` | `prototype/RELEASE_NOTES_v0.3.0.md` **L1–L12**（本版一句话） |
| **第十三轮续 · 10-08** | **D2 调度效果对照实验**三家现役实例全覆盖；MLU590 **分场景结论**，910C / P800 如实 `NOT_APPLICABLE`（**原因不同**） | `prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54 / L123–L150** |
| **第十三轮 · 10-08** | **「不适用」自证审计**（六问）⇒ 两台结论**成立**；910C 根因**升级为「Python 绑定缺失」** | `prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md` **L16–L112** |
| **第十二轮 · 10-08** | **回读口径修复**：更正「三家都行使不了 L5」；MLU590 回读原为**空转判据** ⇒ 改走**厂商 C API**；修 **id 复用越权销毁** | `prototype/docs/STREAM_PRIORITY_READBACK_FIX_20261008.md` **L10–L143** |
| **第十一轮 m1 · 10-08** | **MLU590 20 项全绿** + 台账 **D1 收尾**；同轮暴露并修 **5 处缺陷** | `MLU590/docs/CAMBRICON_MLU_M1_RERUN_20261008.md` **L10–L46 / L110–L199** |
| **第十轮 r10 · 10-08** | **910C 20 项全绿**（台账 **D4 收尾**）；过程遇**共享模型被存储层静默损坏**（非本层） | `910C/docs/ASCEND_910C_R10_RERUN_20261008.md` **L10–L49 / L89–L177** |
| **E1 · 10-08** | **职责审计口径与契约齐平（39 → 78 项）**；三实例 **73/0/5 · 67/0/11 · 62/0/16** | `prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md` **L42–L92** |
| **T1 · 10-13** | **跨后端基类改动（10-10）后复跑 78 项**：910C **73/0/5** · P800 **68/0/10**（均 `DUTY_RESPONSE_PASS`、**FAIL=0**，与 10-08 基线**逐项状态零差异**）· 契约不变式各 **4/4**；⚠️ **MLU590 待做**（该机本轮不可达） | `prototype/docs/OPEN_ITEMS_AND_PARITY_AUDIT_20261010.md` **§3 A1** · `910C/probes/t1_duty_ascend_20261013.json` · `P800/probes/t1_duty_kunlun_20261013.json` |
| **第九轮 · 09-30** | **流优先级统一 API 落地**（三把钥匙 + 四道约束）；P800 改为**真声明** | `prototype/docs/STREAM_PRIORITY_API_20260930.md` **L11–L106 / L265–L348** |
| **第七–八轮 · 09-29** | **A2 多卡 `real` 压测** · **补公开入口** · **B3 修订建议逐条复核（零僵尸）** | `prototype/docs/PUBLIC_ENTRYPOINTS_RECOVERY_20260929.md` · `910C/docs/ASCEND_910C_A2_RECOVER_STRESS_20260929.md` |
| **第五–六轮 · 09-29** | **上下文语义专项 + C 项补足为只读观测** · **两条腿 + 服务化四形态复跑** | `P800/docs/KUNLUN_CONTEXT_SEMANTICS_20260929.md` · `910C/docs/ASCEND_910C_LEGS_SERVE_RERUN_20260929.md` **L45–L142** |
| **第三家收官 · 09-28** | MLU590 **12 项判定全部通过** ⇒ **三实例 12 项判定全绿、可发布** | `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md` **L11–L27 / L65–L123** |

## 更新机制（本方向约定）

| 项 | 约定 |
|---|---|
| **更新节奏** | **每周三**评估并更新本文件 |
| **评估范围** | 本方向职责：**设备上下文（设备抽象与执行上下文）+ 多流 Stream**，以及相关的错误码翻译与状态恢复 |
| **无新增内容时** | **不动任何基座文件**；一切以总组定稿（`dev/stack.lock.910c.v1–v4.yaml`，dev-1.0）为准 |
| **有新增内容时** | ① 直接写入本文件的「基座与约束」或「阻塞与需要协调的事项」（带实测依据）；② 由总组收拢裁定后并入其定稿文件。⚠️ **2026-10-16 起本方向不再保留 `dev/stack.lock.910c.yaml` 工作草稿**（并入 `dev-1.0` 的纪律是「草稿不 PR」，草稿已从个人分支清理；本文件历史条目中对该草稿的引用均为当时记录） |
| **触发更新的典型情形** | 实测到新的环境约束（如新的并发/权限/镜像限制）、新的能力边界（如某后端新增或不支持某能力）、新的坑或缺陷、对基座的新诉求 |

> 依据：总组在 `stack.lock.910c.v2.yaml` 中约定「统一基座的确定/调整/发布由总组裁定，
> 依据各子方向 STATUS.md 暴露的问题与意见；各方向只消费不自建」。

## 交付仓（自建仓）报备与版本锚（2026-11-03 起）

> 依据《组内公告-自建仓规范与双版本机制-20261023》第 4/5 条：自建仓**在方向 STATUS 报备**，总组收录 `dev/repos.yaml`；
> **交付物正文在独立仓，dev-1.0 只放「仓地址 + 当前版本锚」指针**。

### 报备四字段（请总组收录）

| 字段 | 值 |
|---|---|
| **仓地址** | `git@github.com:FlagRT/flagos-device-context.git`（https: `https://github.com/FlagRT/flagos-device-context`） |
| **仓根目录名** | `flagos-device-context`（clone_all 收拢到**仓根一级**；容器内可见为 `/workspace/flagos-device-context`） |
| **协作线** | **`main`**（kind: `direction` —— 不要求 dev-1.0；main 即唯一工作线，开发仍走个人分支 + merge） |
| **可见性** | **public**（等价满足公告第 3 条「开放全组 clone」；无需逐个开通成员） |
| **需访问成员** | 全组（FlagRT/runtime-team） |

### 待总组收录的 `dev/repos.yaml` 条目（原样可粘）

```yaml
  - url: git@github.com:FlagRT/flagos-device-context.git
    dir: flagos-device-context
    branch: main
    kind: direction
    owner: device-context（Kistich）
    visibility: public
    purge_primaries: []
    note: main 即唯一工作线（同 framework-adapter 例外）；public 即满足「开放全组 clone」
    registered: "2026-11-03"
```

> ✅ 字段已按 `dev/repos.yaml` 第 85–93 行（`framework-adapter` 同类条目）**逐字段对齐**：`url/dir/branch/kind/owner/visibility/purge_primaries/note/registered`。
> 本方向**不自行修改** `dev/repos.yaml`（收录属总组动作），此处只给原样可粘的行。

### 当前版本锚

| 项 | 值 |
|---|---|
| 组件版本（发布轴） | **v0.4.0**（`runtime.__component_version__`） |
| Release | https://github.com/FlagRT/flagos-device-context/releases/tag/runtime-v0.4.0 |
| 发版提交 | `984687f`（`main` tip；tag `runtime-v0.4.0` 仅作提交指针，**发布状态只看 Release**） |
| 接口版本（兼容轴） | **v0.1 · 原型期**（`runtime.__version__`）—— 与组件版本**有意不对齐** |
| 附件 | `flagos_device_context-0.4.0-py3-none-any.whl`（可 `pip install`）· `flagos_device_context-0.4.0.tar.gz` |

### 过渡期约定（v0.4.0 → v0.5.0）

- 协作仓 `prototype/runtime/` **暂时保留同一份包内容**（本轮已核对：与新仓 `.py`/`.md` **逐字节一致，31 文件零差异**），
  目的是**不打断四实例验收**（各芯片 `probes/` 里的脚本仍按 `DC_ROOT=<prototype>` 取包）。
- **v0.5.0 起**：README 只推 `pip install`；届时把实例侧的取包方式切到「装包 / 挂新仓」后，
  再删协作仓里的副本（避免两处漂移）。⚠️ **在此之前改动 runtime 包必须两处同步提交**。
- 上一版 Release（`runtime-v0.3.0`）留在协作仓 `FlagRT/runtime-team`，**保留不删**。

## 当前阶段

对应《运行时层原型验证-战略目标-910C》§5「设备抽象与执行上下文」及运行时层模块拆分表子方向 1。
本月目标：交付统一运行时原型，并基于该原型完成训练腿 / 推理腿的设备侧验收。

- **统一 API 与 Backend 插件机制已完成**，并在**四个芯片实例**上取得实证（第 3 家 09-28 收官；**第 4 家 10-10 上机并取得两条腿判据**；
  10-08 三实例在**同一版本**上跑完**扩口径 78 项职责审计**）—— 逐实例结论见主看板 §2 与各芯片看板 §3。
- **阶段 5 交付**：《新芯片接入手册》·《组内服务启动标准》（唯一入口 `prototype/scripts/serve_standard.sh`，
  跨芯片只改 `DC_BACKEND`，**各方向不再自建启动脚本**）·《验证复核清单》· 原型 release **`runtime-v0.3.0`**（2026-10-08；上一版 `runtime-v0.2.0`）。
- **多流 Stream 16 项验收基线**三家逐项比对完成（**唯一差异 = S-12 流优先级**：MLU590 `(0,-3)` 可设 ·
  P800 **单档** `(0,0)` · 910C **不声明**设置）⇒ `prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54**。
- **未完成项**：组件下游反馈收集 · 全组联合 demo 合稿。（逐项未收尾清单见 `prototype/docs/OPEN_ITEMS_AUDIT_20260929.md` **§0/§2**）

## 已有量化结果（实跑）

> ⚠️ **本节不再复写数字** —— 此前维护的"跨实例总表"与本方向「同一事实多处复写必漂移」的纪律冲突
> （缺陷台账第 36 条：契约表曾因此落后 8 天无人发现）。**数字的唯一入口 = 芯片看板 + 验收报告**。

| 实例 | 量化结果入口 | 一句话现状 |
|---|---|---|
| **910C**（第 1 家） | `910C/README.md` §3 · `910C/docs/ASCEND_910C_R10_RERUN_20261008.md` **L10–L49** | 两条腿全闭环：训练腿 **6/6**（4196.6 tok/s）· 推理腿 **14/14** · 服务化 **4 形态全 PASS** |
| **P800**（第 2 家） | `P800/README.md` §0 · `P800/docs/PROGRESS_REPORT_20260914.md` | 阶段 0–5 完成：训练腿 **6/6**（3533.5 tok/s）· 推理腿 **13/13** · 服务化 **10/10** |
| **MLU590**（第 3 家） | `MLU590/README.md` §0 · `MLU590/docs/CAMBRICON_MLU_M1_RERUN_20261008.md` **L10–L46** | 接入完成 + m1 补齐轮 **20 项全绿**：训练腿 **6/6**（3015.3 tok/s）· 推理腿 **13/13** |
| **PPU**（第 4 家） | `PPU/README.md` §0 · `PPU/docs/PPU_ONBOARDING_PARITY_AUDIT_20261010.md` **§2–§5** | 接入自检 + 两条腿**同口径**（训练腿 **50 步** 6/6 · 5167 tok/s · 推理腿 13/13）；✅ 服务化 10/10 · 错误闭环 5/0/0 · 多流 16 项 · 证据索引已补（10-14 · 10-26 复跑零回归）；✅ 职责审计 **73/0/5**（10-26 C13 二次修正） |
| **三实例同口径矩阵** | `prototype/README.md` §3 · `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md` | **12 项判定全部通过**；职责审计 **73/0/5 · 67/0/11 · 62/0/16** |
| **组件级** | `prototype/RELEASE_NOTES_v0.3.0.md` | Release **`runtime-v0.3.0`**（接口面补齐 + 行为修正；出口 28 → 46） |

## 基座与约束（本方向实测）

- **设备注册路线**：遵循 v1 的 Route A 原则。**2026-09-22 口径统一**：全组统一基座为
  Qwen3-0.6B **训推** + 原型接入的**厂商 torch 插件路线** ⇒ 910C **两条腿均为 `npu`（torch_npu）**，
  训练腿由路线 B（torch_fl）切到 `npu`，**v1 登记的「权宜例外」已取消**。
  实测（2 卡/50 步）：`TRAIN_LEG_PASS 6/6`、loss **15.4498→11.1479**（torch_fl 线同为 50 步 **15.4497→11.1515**，曲线几乎重合；
  差 0.03% 属集合通信归约顺序差异带来的浮点非确定性，非逻辑差异）、
  **3954–4402 tok/s**（torch_fl 线 **2212.9** ⇒ **+79~99%**）；证据 `910C/probes/train_npu_20260922.log`、
  `910C/probes/unified_verify_20260922.log`。已写入 `dev/stack.lock.910c.yaml` 的 `per_leg.train`。
- **⚠️ 镜像本身未切换**（`lock.train.image` 未变）：本次只改**用哪个解释器/后端跑训练腿** ——
  用容器内带 torch_npu 的 `venv-infer-a`（该解释器**无 torch_fl**，**物理隔离**，
  故不触发镜像的"禁止共存"校验）；原路线 B 的 `AUTOLOAD=0 + 先导入插件` 写法已随路线 B 归档。
- **⚠️ 切换时踩到并已修的新缺陷（审计台账第 15 条）**：`hccl` 集合通信后端名**要厂商扩展被 import 后
  才在 c10d 注册**，而统一 API 后端是懒加载 ⇒ `init_process_group("hccl")` 报
  `AssertionError: Unknown backend type hccl`。修法：**初始化进程组之前先经后端触碰一次设备**。
- **带卡容器名额**（v1 规则置顶写「上限 3」，但 **09-22 / 09-28 实测到不了 3**）：名额用尽 `acl.init()` 返回 500000、设备不可见；**同一时刻只留 1 个带卡容器**（详见 `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md` §4.3）。
- **同一个 FL 插件在两家芯片上可用性相反**（第三家选型须逐个确认，不能类推）：
  `vllm-plugin-FL` 在**昆仑芯必需**（社区 vLLM 的 `vllm/platforms/` 无 kunlun，靠它提供 FL platform
  —— 不加载则 vLLM 报 `Failed to infer device type`）；在**昇腾必须禁用**
  （设 `VLLM_PLUGINS=fl` 后 `current_platform.device_type` 变空 → `RuntimeError: Device string must not be empty`，
  该插件自述 currently CUDA only）。
- **官方 `-base` 镜像开箱不含 `triton`**：`vllm_fl → flag_gems → triton` 断链，推理服务化报
  `Failed to infer device type`（**看着像设备问题，其实是包问题**）。须按官方手册 1.2 节
  `python3.10 -m pip install flagtree===0.7.0rc3+xpu3.6`（实测源 HTTP 200、wheel 3.3 GB、约 2.5 分钟）。
  ⇒ 若以该镜像入锁，**配方须显式包含此步骤**，否则第三家接入者会卡在同一位置。
- **P800 流优先级不可用（上游）**：裸调 `Stream.priority_range()` 触发 PyTorch 自身
  `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188`（XPytorch 上报非法优先级区间）。
  本层已**主动拦截、绝不透传**（避免进程级 abort），`kunlun` 后端**不声明** `stream_priority` 能力（如实）。
- **镜像的"服务入口"与"用卡工具"口径不一致**（09-20 启动标准补跑时实测，两条均为镜像特征、非缺陷）：
  ① **P800 镜像的 `vllm` 不在默认 `PATH`**——它在 conda 环境 `python310_torch29_cuda` 内，不激活就
  `nohup: failed to run command 'vllm': No such file or directory`（启动即退出，极易误判为"服务起不来"）；
  ② **910C 镜像不含 `npu-smi`**（`npu-smi: command not found`，它是宿主工具），容器内查卡需退回 torch 侧
  （`torch.npu.mem_get_info`）。
  ⇒ 本方向已在统一启动脚本内消化（自动激活环境 / 快照降级）；**建议总组在基座层明确"镜像须自带可用服务入口"**，
  否则第三家接入者会各自踩一遍。
- 本轮仓库整理与验证**未新增或升级公共依赖**；容器内按需补装 `transformers`（v1 已登记）。
- **⭐ 基座层新发现（2026-09-22 追加）：FlagOS 官方镜像体系**（`flagos-ai/build-infra` 构建，
  registry 前缀 `flagos-base` / `flagos-runtime` / `flagos-dev` / `flagos-app`，当前版本 **2.2.0**）。
  **覆盖 14 个后端、含寒武纪**（FlagTree 手册覆盖不到的一家）；`configs.yaml` 是唯一 source of truth，
  文档站由它自动生成。**对我们的意义**：
  ① **910C 有官方对应物**：`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0`
   （digest `sha256:1048d622c928e86dd004ddb58b8b88602d91dc3c15458fda0262ca091e3ffb35`，5.4 GiB，
   官方标**宿主驱动前置 26.0.rc1**）—— py3.11 / torch 2.10.0+cpu / torch-npu 2.10.0 /
   **flagtree 0.7.0rc2+ascend3.5** / triton 3.5.0(+triton_ascend 3.2.1) / flag_gems 5.4.0-rc2.post3 /
   CANN 9.0.0（含 **910C Ops**），与我们锁定栈（CANN 9.0 / py3.11 / torch 2.10 / triton 3.5 / ascend3.5）
   **逐项一致**，只是来源不同（我们 = 组内 `flagrt` 镜像 + 华为 `vllm-ascend`）。
   ✅ **已在工作草稿 `dev/stack.lock.910c.yaml` 的 `candidates.train.official` 登记**（与既有组内自建候选
   `internal_v2` 并列，**仅登记、不生效、不切换**），并出专项对照
   [`910C/docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md`](910C/docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md)。
   ⚠️ **两处缺口须注意**：① 该镜像**设备后端为 `npu`（torch_npu，即 Route A）**，
   与**我们现已统一的口径一致**（2026-09-22 起训练腿已走 torch_npu，路线 B 的权宜例外**已取消**）；
   ⇒ 本候选与现网口径的差异**只剩镜像血统**，不再涉及设备后端路线；
   ② 但**官方 runtime 镜像不含 FlagCX**（`flagcx-ascend` 镜像线最后 push 2026-02-02，陈旧不可用）
   ⇒ **训练腿缺口未解、不可直接切换**，与「官方推荐镜像不含 FlagCX」是同一个已登记诉求。
   **是否切换由总组裁定，本方向不自行切换**；
   ② **每个 `base|runtime/<backend>.md` 都写明「Host driver」前置**，这是选镜像的**第一判据**（见下条）；
  ③ 建议将本体系列为《镜像选择与确定指南》的**最高优先级来源**（各方向已在按该口径对齐）。
- **⭐ 寒武纪档位受宿主驱动硬约束（2026-09-22 实测 + 官方口径）**：寒武纪在 FlagOS 官方体系里有**两档**，
  **不是按 torch 版本选，而是按宿主驱动选**：

  | 档位 | py | torch / torch-mlu / triton | 官方标注宿主驱动前置 |
  |---|---|---|---|
  | `flagos-runtime-cambricon-neuware4.4.3:2.2.0` | 3.10 | 2.7.1+cpu / 1.29.2 / 3.2.0+mlu1.7.2 | **6.2.15** |
  | `flagos-runtime-cambricon-neuware4.7.2:2.2.0` | 3.12 | 2.11.0+cpu / 1.33.1 / 3.4.0+mlu2.1.1 | **6.5.48** |

  两台测试机实测驱动 **v6.2.29** ⇒ 落 **6.2.x 线**，**只能用 4.4.3 档**；
  4.7.2 档需宿主驱动升至 **6.5.48**（会动宿主、影响他人）。
  **⇒ 请总组/管理员裁定走哪条**：A 直接用 4.4.3（同驱动线，风险最低）；B 升驱动到 6.5.48 后用 4.7.2。
  ⚠️ 6.2.29 能否跑标 6.2.15 的 4.4.3 **未实测**（同 minor 属推断），须起容器实测定论。
- **⭐ P800 存在两条血统，不要方向侧自行切换（2026-09-22）**：我们现用 FlagTree 线的
  `flagtree-xpu3.6-…:202608-base`（= FlagTree 手册给 P800 的唯一镜像，与类脑指向同一条 xpu3.6 线，
  **镜像本身无需调整**）；FlagOS 官方线则是 `flagos-runtime-kunlunxin-xre5.37.1:2.2.0`
  （flagtree 0.7.0rc2+xpu3.6，但**底层 SDK 换代到 XRE 5.37.1**，前置要求宿主驱动 **5.37.1**，
  而我们实测宿主为 **5.0.21.47**）。**⇒ 建议总组明确 P800 走哪条**，方向侧不自行切换。
- **⭐ 对昆仑芯 KL3 缺陷的官方印证（2026-09-22）**：FlagOS 官方 `configs.yaml` 的昆仑芯 vLLM 应用层原文——
  `# XPU_EVENT_KL3_ENABLE deliberately NOT set: it is the P1 fake-hang trigger (device timeout) on this XRE stack`
  —— 即**官方明确不设该变量**并称其为「P1 假挂死触发器」。这与本方向独立定位的结论一致，
  **可作为上报时「上游已承认该触发器」的旁证**。
- **⚠️ 一处待核对差异（不臆断）**：类脑 P800 记录 `Triton 3.5.0`，我们 P800 基线实测 `triton 3.6.0`
  （两者 `flagtree` 均为 `0.6.1+xpu3.6`）。未查明原因，只影响算子编译路径，**不影响本方向结论**。

## 下一步（拟在下次周会前推进）

**第 4 家（平头哥 PPU）上机完成，按手册验收清单继续**（2026-10-10）—— 已完成：接入三步 + conformance 13+6 +
**步 0 模型完整性** + **训练腿 6/6** + **推理腿 13/13**。剩余按优先级：
⭐ **2026-10-10 四家复合核验（第二次同口径自审）**：以手册 §7 + `VERIFICATION_MANIFEST` §1（11 条）为口径、
**按产物文件**（不认文档声明）逐项比对四家 ⇒ 已当场补两项、另核出 3 项与三家不同口径、并回填 3 处过期状态。
全文 `prototype/docs/OPEN_ITEMS_AND_PARITY_AUDIT_20261010.md`。

① ✅ **职责审计 `L1` FAIL 已收口**：根因＝**跨类型地址复用**；修复＝**类型复核**（机制无关）+ 强持有纵深防御；弱引用实现下亦 3/3 OK。
⚠️ 原「根因＝持有器强度」的**机制解释已被否证**（弱引用下对象已回收 ⇒ `holder()` 返 `None` ⇒ 同样拦下），
**机制至今未解释** ⇒ 代码注释与报告口径已统一为如实版。
⭐ **跨后端基类改动后复跑（T1）**：**910C ✅ 73/0/5** · **P800 ✅ 68/0/10**（均与 10-08 基线逐项状态零差异、契约不变式 4/4）⇒ **剩余：MLU590 待做**。
② ✅ **契约不变式 I1–I4 真机 `4/4`**（本轮补跑；⚠️ 三家早已实跑、**看板矩阵漏列** ⇒ 根因已修）；
③ ✅ **A2 多卡多进程压测如实跳过留档**（未声明 `recovery_real` ⇒ `SKIP_UNSUPPORTED`，不计失败）；
④ ✅ **推理腿服务化（10-14）**：`serve_standard.sh` 已补 `ppu` 分支 ⇒ `SERVE_STANDARD_PASS` + `SERVE_LEG_PASS 10/10`
（⚠️ 前置：镜像**不含 `triton`**，需按共享目录补 FlagTree triton/flagtree；容器 pip 不可用）；
⑤ ✅ **错误闭环四类注入（10-14）**：`ERROR_RECOVERY_LOOP_PASS` **5/0/0**（第 4 类为诚实降级负向测试）；
⑥ ✅ **多流 16 项基线逐项比对报告（10-14）**：`PPU/docs/THEAD_PPU_STREAM_BASELINE_16_20261014.md`（16 项全部有读数）；
⑦ ✅ **D2 调度效果实证（10-14）**：`STREAM_PRIORITY_SCHED_EFFECT_PASS` · **`effect_observed=False`**（如实记录）；
⑧ ✅ **`EVIDENCE_INDEX_PPU.md`（10-14）**（含 `rc=139` 覆盖表与 T10 命名偏离登记）；
⑨ ✅ **退出期段错误（T11）本层侧已收口（10-14）**：原生采集现场 = `Py_Exit → Py_FinalizeEx → _PyWeakref_ClearRef`（`si_code=1`、`si_addr=0xab`）⇒ 根因＝**平台侧弱引用书签失效**（**不含本层**的纯 torch 最小复现同样 `rc=139`）；本层修法＝不依赖弱引用语义（无上下文不登记 · `_released_streams` 强持有 · 登记表加上限）⇒ 三载体真机 `rc=0`、判据不变；⚠️ **平台侧缺陷未修、转厂商**；⚠️ 该修法落**共享层** ⇒ **910C/P800/MLU590 需再各复跑一次 78 项审计**。
⑩ ✅ **第五载体（契约不变式 I3）已收口（10-26）**：原生现场 `_PyWeakref_ClearRef` ← `Py_FinalizeEx`（`si_addr=(nil)`，与 T11 的 `0xab` **不同签名**）；二分到**只有 `i3_invalidated_governed`** 单独复现；暴露面＝`_stream_ctx`（本层**最后一处**弱引用）⇒ 改强持有后 `rc=139 → rc=0`，判据逐字不变，**13 段全量 13/13 段 `rc=0`**；⚠️ 复现纪律：**不得在退出前 `gc.collect()`**（会掩盖缺陷）；📤 厂商材料已备 `PPU/docs/PPU_VENDOR_WEAKREF_ISSUE_20261026.md`（含不含本层的最小复现 4/4）。
⑪ ✅ **PPU 服务化前置 triton 已固化进镜像（10-27，镜像侧）**：官方基座不含 triton ⇒ 服务化必挂、此前靠手工 `docker cp`、容器重建即丢 ⇒ 新建 **`dev/images/thead-ppu-runtime/v2/`**（派生层：一个 flagtree wheel `pip install` 同时提供 triton 3.6.0 + flagtree，wheel sha256 `b57daff7…`）。验证均在**全新容器、零手工干预**下：真编 kernel `TRITON_PPU_SMOKE_PASS`、服务化 `SERVE_STANDARD_PASS (ready=1 smoke=1)`、现役容器切 v2 后 `s03 PASS` + `s06 CONTRACT_INVARIANTS_PASS 4/4`（`rc=0`）。基线为 ADR-0007 **登记式**，`check` 实测 **PASS**（`baseline.py` 暂不支持 wheel 固化形态 ⇒ 已提总组 TODO #23/#24）。共享机根盘余量**未恶化**（构建前后均 ~3G，旧容器可写层已回收）。**已按组内流程完成发布登记（10-27）**：入档门槛满足（`v2/lock.yaml` 的 `repro_status` 为 🟢 functional-repro，配 `image.baseline.yaml` + `env/v1`）；离线归档已产出并三重核验（17.15 GiB · sha256 `6ab0dfb9…` · `pigz -t` rc=0 · `docker load` 回读 image id 与发布物逐字相同）；镜像索引与「当前生效版本」表已同步（`dev/images/image_list.md` 新增「平头哥 PPU」节）；**生效切换未自行改**——已在 `dev/stack.lock.thead-ppu.v1.yaml` 的 `candidates:` 登记为候选，并显式提请总组裁定两点：① 是否切为生效版本；② 本文件 `rules` 有「各方向只消费不自建」一条，本派生层属需例外授权的情形（替代路径 = TODO #23 把 triton 上提官方基座线）。

⑩ ✅ **T1-re（共享层再改后的破坏面复跑）三家全部闭环（10-14）**：910C `DUTY_RESPONSE_PASS` **73 / 0 / 5**（`L1`–`L4` OK、`L5` SKIP，与 10-13 一致）· P800 **68 / 0 / 10** · MLU590 **62 / 0 / 16**；
三家契约不变式各 **4/4** · 离线自检 **92 / 114 / 103**（0 失败），且**与各自基线逐项状态零差异**（唯一 `detail` 差异 = 厂商句柄地址 或 B4 显存读数 `157→156 MB`）
⇒ **共享层两次改动（10-10 类型复核 + 10-14 T11 修法）在三个实例上均无回归**。⚠️ 910C 选卡按名额规则：**只挂 1 个未被任何容器映射的 die**（本轮 `davinci5`），
映射全 16 张会因"设备被他人共用"直接 `dcmi ... device is used`（详见 `910C/docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md` §6）。
⑪ ✅ **T10 证据命名约定（10-14 · A 方案）**：目录级命名 **`<T号>_<用途>_<YYYYMMDD>[_out]`** 写入
`prototype/docs/VERIFICATION_MANIFEST_20260920.md` **§5.1** 作为**补充约定**（目录 = 按任务检索；文件 = 仍按 §5 模板；**旧证据不追溯改名**）。

⑫ ✅ **T7 MLU590 服务化四形态补全（10-14）**：在 **vLLM 应用镜像容器** `dc-mlu590-vllm-hliu553` 上跑
`TP∈{1,2} × EAGER∈{1,0}` **四形态，全部 `SERVE_STANDARD_PASS (ready=1 smoke=1)`**（就绪 175/530/345/440 s · 维度 1024 · 范数 1.000001；
形态生效的正证据 = vLLM 自身 config dump 的 `enforce_eager` 与 `tensor_parallel_size`/`world_size` 逐形态吻合）⇒ 与 **910C（09-29 四形态）对称**；
本层消费方主腿 **`SERVE_LEG_PASS 10/10`**（区分度 0.4091，与 10-08 逐字相同）；对照腿 `TP=2/EAGER=0` 首跑撞客户端窗口 ⇒ 补跑（`SERVE_REQ_TIMEOUT=600`，服务侧一字未改）**亦 `SERVE_LEG_PASS 10/10`** ⇒ 定性为**客户端观测窗口不足**（max 87467 ms）。⚠️ **两处如实登记**：① 消费方客户端观测窗口原硬编码 **120 s**
（`TP=1/EAGER=1` 实测首请求 85 s、`TP=2/EAGER=0` 第 2 请求超时）⇒ 新增 `SERVE_REQ_TIMEOUT`（**默认值不变**）；
② 停机漏杀大写进程名 **`VLLM::Worker_TP*`** ⇒ 卡 0/1 各占 89 GiB 未释放，按 `VLLM::` 前缀清后回落 **94.6/94.8 GiB**（脚本已修）。
⇒ `MLU590/docs/CAMBRICON_MLU590_SERVE_FOUR_FORMS_20261014.md` · `MLU590/probes/t7_serve_20261014_out/`。

⭐ **10-14 新增**：**平台侧 `weakref` 悬垂 ⇒ 本层误拦**（T6 首跑全线 `RuntimeError`）已定位并**机制无关修复**
（新建登记即清除同 id 旧「已释放」记录 + 类型复核；交替对照 + 非空转判据）⇒ `PPU/docs/PPU_T3_T6_AND_WEAKREF_20261014.md`。
⭐ **10-14 再新增**：**退出期段错误根因定位与收口（T11）** —— 容器内**无 gdb**（宿主有、但无免密 sudo 且容器进程属 root ⇒ 不能 attach），
改用**自装原生 SIGSEGV 采集器**（容器自带 gcc → `LD_PRELOAD` + `dladdr` 取「模块 + 偏移 + 符号」）拿到现场；
再加 `sitecustomize` 注入做**单变量对照**（`note_off` / `strong`）⇒ 三载体全绿；离线判据新增**台账第 39 条**（去守卫即 FAIL，已反向实测）
⇒ `PPU/docs/PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md`。
⚠️ 该改动落**共享层** ⇒ **需三家复跑**（同 T1 方法）。

⚠️ **口径**：PPU 原「未对齐 6 项」**已全部完成**（10-14）；
「全口径验收」的**剩余门**是 **⑨ 退出期段错误**（依赖外部）与**共享层改动的三家复跑**。详见 `PPU/README.md` §0/§4。

⭐ **下周执行单**（含三台真机复跑 T1 的逐台命令与判据、PPU 未对齐 6 项、窗口与串行约束、闭环要求）：
`prototype/docs/TODO_NEXT_WEEK_20261012.md`（**2026-10-12 09:30 提醒指向它**）。

**第 3 家（寒武纪 MLU590）已收口（2026-09-28）** —— 12 项判定全部通过，与前两实例同口径、同判据、同证据规范。
后续不阻塞项：① **多卡 TP / 关闭 `--enforce-eager`** 的服务化形态未跑（本次 `TP=1` + eager）；
② **CNCL 的 RDMA 路径未验证**（当前走 `MLU_LINK` 片间互联，训练数据仅代表单机 2 卡）；
③ 档位升级（→ `neuware4.7.2` / 驱动 6.5.48）仍为**上报预案**，非当前诉求。
④ ✅ **已补（09-28）**：910C 按新脚本复跑服务化 —— `SERVE_STANDARD_PASS`（就绪 35 s、
   维度 1024、范数 1.000000、冒烟耗时 0 s），与 09-22 逐项一致；
   首轮因宿主名额被他人占满失败，释放后通过（口径更正见本节「带卡容器名额」条）。
⑤ ✅ **已补（09-28 网络恢复后）**：**MLU590 真机职责响应审计**（39 项）—— `DUTY_RESPONSE_PASS`
   **36/0/3**（3 项 SKIP 均为如实不具备）；离线自检 41/0/0 · 对称性 5/0；
   三处补做真机验证生效 ⇒ **三实例全部真机 PASS**（910C 39/0/0 · P800 36/0/3 · MLU590 36/0/3）。

**P800（第二实例）收敛 —— 本周剩余 + 下周**

1. ✅ **阶段 3 推理腿（2026-09-20 完成）**：单卡前向 `INFER_LEG_PASS 13/13` ——
   维度 **1024**（对齐 910C）、语义区分度 **0.6392**、**53.12 句/s**、p50 **56.17 ms**、
   真实参数异常 → **L2_PARAM/raise 且业务继续**。证据：`P800/probes/F_infer_leg_*`。
   - ✅ **vLLM 服务化形态亦已完成**（同日）：`SERVE_LEG_PASS 10/10` —— 区分度 0.4102、30.70 句/s、
     p50 96.4 ms、超长输入 → L2_PARAM/raise + 业务继续。证据：`P800/probes/F2_*`。
     ⚠️ 两条接入手册级环境要点：① `PYTHONPATH=/env/FlagGems/src` 是**硬前置**（site-packages 的
     `flag_gems` 安装不完整，缺 `runtime.backend.device`）；② 算子路径取 vendor
     （`VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0`）以避开 FlagGems 路径与其 KL3 依赖。
2. ✅ **阶段 4 错误闭环（2026-09-20 完成）**：两设置对照（唯一变量 `XPU_EVENT_KL3_ENABLE`），
   两组均 **闭环 5 / 跳过 0 / 失败 0** 且**逐字节一致** ⇒ **关闭该变量不损失错误诊断能力**；
   并说明该厂商缺陷**不影响单进程设备上下文路径**。证据：`P800/probes/error_recovery_loop_kunlun_KL3{off,on}.json`。
2b. ✅ **官方 `-base` 镜像等价性验证（2026-09-20 完成）**：全套结论在官方推荐镜像上复现
   （conformance 13+6 逐用例一致、smoke 42/0、训练腿 loss 逐位相同、推理腿 `detail` 14/14 逐字相同、
   服务化 10/10、**KL3 挂死一致重现**）。⚠️ 期间查明 `-base` 开箱缺 `triton`，补齐命令已实测走通。
   证据与完整对照见 `P800/docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md`。
3. ✅ **阶段 5 收敛（2026-09-20 完成，三件套）**：
   ① **《新芯片接入手册》** —— 4 条厂商栈判别路径 / 13 个抽象方法清单 / 8 步接入流程 /
   **可勾选验收清单 13 项** / 9 条跨芯片坑 / 厂商问题上报模板；
   ② **接口约定修订建议 6 条**（P800 是现行约定的首次非昇腾检验）：`device_type`/`vendor` 分离、
   `device_state` 入契约、`.native` 逃生舱约束、**错误对象跨模块类归一**（已修在框架层）、
   有界同步降级契约（前瞻性）、`known_issues()` 入契约；
   ③ **原型 release `runtime-v0.2.0`**（第二实例接入版）—— 第二实例接入 + 4 个框架修复 + 脚本后端无关化，
   **未改任何已有接口签名**（接口版本仍为 v0.1 原型期）。
2c. ✅ **多流 Stream 16 项验收基线逐项比对（2026-09-20 完成）**：P800 侧 **14 项通过 / 1 项如实标注不支持
   （S-12 流优先级，上游缺陷）/ 1 项不适用**；探针 8 项 `STREAM_SEMANTICS_PASS 8/8`（与 910C 逐项一致）；
   **S-7 图捕获首次实测 `GRAPH_CAPTURE_PASS 5/5`**（据此为 `kunlun` 补上 `graph_capture` 能力声明）；
   S-16 补测 2000 流无限制。探针已后端无关化并上提为**跨后端共用资产**
   （`prototype/probes/probe_stream_semantics_full.py`）。
   ⚠️ 过程中有一处**自我纠错**：S-7 首测判为"不支持"实为探针用法错误（捕获区内做了同步，违反 CUDA Graph 契约），
   修正后 5/5 通过——已撤回错误判断。本批内容按约定**随 10 月提交**。
4. **厂商缺陷上报**：待确认渠道后提交（主提交昆仑芯 XPytorch/XRE，抄送 FlagCX，
   知会 FlagGems/FlagTree）。

> 本日另发现并修复**第 4 个"只有非昇腾实例才暴露"的框架缺陷**（错误对象跨模块类不相等 → `disposition` KeyError），
> 已修在框架层（对既有后端无回归），详见 `P800/docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` §3。

**910C（第一实例）遗留**

5. **组件 v0.1.0 下发与反馈**：等下游接入，按周迭代 `v0.1.x`。
6. ~~组织架构合入 dev-1.0~~ → ✅ **已完成**（**PR #19**，2026-09-17 合入）：
   **P800 第二实例接入 + 按芯片重组目录（一份原型 + 两个芯片实例）**。
   合入时按约定排除了本方向基座工作草稿（`dev/stack.lock.910c.yaml`），草稿仍留特性分支；
   并借本次同步对齐了基座引用 v1 → **v2**。
7. **通信接口约定**：与分布式方向确认 flagcx 路线（`prototype/docs/DESIGN_DIST_COMM_20260908.md`）。

## 阻塞与需要协调的事项

> 已解除的历史阻塞（910C 带卡名额 · 寒武纪网络与权限 · 镜像渠道）原文见
> `prototype/docs/BOARD_ARCHIVE_STATUS_SECTIONS.md`。**当前无阻塞本方向验收的事项**；下列为对外诉求。

| # | 事项 | 需要谁 | 状态 |
|---|---|---|---|
| 1 | **上游诉求跟进（台账 D3）**：torch_npu 补 `ExternalStream`（或让 `Stream(stream_ptr=…)` 真正生效 —— 现状**静默忽略**） | torch_npu / CANN 上游 | ⏸ **待定**（无渠道；**不影响本方向验收**） |
| 2 | **上报渠道待确认（台账 C4）**：昆仑芯 KL3 事件挂死 + 物理卡 1 故障 —— 直连厂商支持 vs 经总组转达 | 总组裁定 | ⏸ **待定**（缺陷已定性、已对外提交正式报告） |
| 3 | **镜像血统二选一**：P800 走 FlagTree 线（现用）还是 FlagOS 官方 XRE 线（需宿主驱动 5.37.1，本机 5.0.21.47） | 总组裁定 | ⏳ 已登记（`prototype/docs/IMAGE_LINEAGE_ALIGNMENT_20260922.md`） |
| 4 | **910C 宿主带卡名额** | 同机使用者/管理员 | ✅ **口径已澄清**：独占单位 = **已 init 容器的挂载设备集** ⇒ 只挂自己要用的空闲卡即可并存，**无需停他人容器**（`910C/docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`） |
| 5 | **memory 方向下游建议（`dev/memory/STATUS.md` 建议 #1–#4）· 本方向认领表态（2026-11-03）** | memory 方向（可据此关单） | ✅ **已认领**：**#1 部分采纳**（embed 默认 `gmu=0.4`+`max_num_seqs=128` 已落 `serve_standard.sh`，显式值优先；`enforce_eager` 不采纳其默认——我方实测 TP=2 就绪 50s vs 95s、P800 去 eager p50 反升，保留默认 1 可 `EAGER=0` 切）；**#3 采纳**（后续服务化证据补「宿主 npu-smi 运行期峰值采样 + `/metrics kv_cache_usage_perc`」双指标，复用 memory 的 `hbm-sampler_910c.py`）；**#4 采纳**（参数化：embed 共享机 0.4 / generate 独占 0.9、短生成砍 `MAX_MODEL_LEN`，均落脚本注释与手册）；**#2 属调度方向**（非本方向接收面，转总组分派）。⚠️ 落地默认**待 910C A/B 实测背书**（0.4 vs 0.9 + 运行期峰值 + 同卡共存判据，排期中） |
| 6 | **PPU 平台侧 weakref 退出期段错误（纯 torch 最小复现 4/4）转厂商/群内求证** | 总组转达厂商 / 群内交叉验证 | 📤 **材料已备**（`PPU/docs/PPU_VENDOR_WEAKREF_ISSUE_20261026.md`，含三必要条件判别矩阵 + 12 份原始日志）⇒ 先发 **runtime 群求证**（他方向是否复现 / 是否我方理解偏差），再定是否正式提厂商 |

## 基座草稿本周状态（每周三填写）

**2026-09-16（周三）**

> **基座版本对齐说明**：总组已于 **2026-09-12 发布 `stack.lock.910c.v2.yaml`**，
> 本方向本次同步时一并对齐（此前一直引用 v1）。**v2 相对 v1 仅新增 `candidates:` 段**
> （登记训练腿候选新血统镜像 `dev/images/*/v2/`），**`lock:` / `rules:` / `per_leg:` 与 v1 逐字节一致**
> —— 即未发生任何镜像切换，本方向既有结论不受影响。

- 本周是否有需提总组的基座更新：**无 910C 侧新增**（以 `stack.lock.910c.v2.yaml` 为准）
- 工作草稿 `dev/stack.lock.910c.yaml` 本周是否改动：**未改动**
- 本周新增内容**全部属昆仑芯 P800 侧**（非 910C），故不写入 910c 草稿。**建议总组裁定是否在 v1 之外
  单列「新芯片（昆仑芯）机器使用约束」** —— 目前累计 3 条：
  ① 需 `docker` 组权限 + 自有可写数据目录（`/data2/hliu553`）；
  ② **设备 API 为 `torch.cuda` 而非 `torch.xpu`**（`USE_XPU=OFF` 实测，跨方向通用）；
  ③ **共享机用卡前必须 `xpu-smi` 挑「连续且空闲」卡并记录用卡**（他租户负载会污染结论，本轮已实际踩到）。
- **另需总组知悉**：P800 上存在**厂商运行时缺陷**（KL3 事件同步概率性挂死 ≈89%），
  已定位到函数级并如实标注，我方以「本方向验证运行不设该变量」规避、不阻塞其余交付；
  **上报渠道待确认**（直连昆仑芯支持 or 经总组转达）。

**2026-09-22（周二，提前填写 —— 本次为镜像血统核查带出的基座变更）**

- 本周是否有需提总组的基座更新：**有 2 项（均为登记性质，不改变现网）**
  1. **910C 新增「FlagOS 官方对应物」候选血统**：`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0`
     （digest `sha256:1048d622…`，5.4 GiB，官方标宿主驱动前置 **26.0.rc1**）。
     与我们锁定栈**逐项一致**；**设备后端为 `npu`（Route A）** —— 与**我们现已统一的口径一致**
     （训练腿的路线 B 权宜例外已于 2026-09-22 取消），故本候选与现网差异只剩镜像血统；
     ⚠️ **但官方 runtime 不含 FlagCX**（`flagcx-ascend` 镜像线最后 push 2026-02-02，陈旧）
     ⇒ **训练腿缺口未解、不可直接切换**。是否切换请总组裁定。
  2. **建议把「宿主驱动前置（Host driver）」吸收进 `dev/images/<name>/v<N>/lock.yaml`**：
     FlagOS 官方每份 `base|runtime/<backend>.md` 都明示该行，这正是我们入档材料目前缺的字段
     —— 本次若早有此字段，寒武纪档位就不会先按 torch 版本锚错（现在已按驱动修正）。
- 工作草稿 `dev/stack.lock.910c.yaml` 本周是否改动：**有改动**
  （新增 `candidates:` 段：`internal_v2` 组内自建候选 + `official` 官方对应物候选；
  **`lock:` / `rules:` / `per_leg:` 段未动** ⇒ 未发生任何镜像切换，既有结论不受影响）。
  按约定草稿**留在特性分支、不直接 PR**，由总组收拢裁定后并入 v1/v2。
- **另需总组知悉（本日新增）**：P800 的 KL3 缺陷拿到了**上游官方印证** ——
  FlagOS 官方 `configs.yaml` 明确不设 `XPU_EVENT_KL3_ENABLE` 并称之为
  「**P1 fake-hang trigger (device timeout)**」⇒ 与我方独立定位一致，
  且暴露出 **FlagGems `backends.yaml`（设 1）与 FlagOS 官方 `configs.yaml`（明确不设）的口径冲突**，
  上报时可直接指向该冲突（详见 §阻塞与需要协调的事项 · KL3 条）。
- **寒武纪（第 3 家）侧不需要总组动作**：镜像已从 FlagOS 官方 Harbor 取得并定档 `neuware4.4.3`
  （驱动同 6.2.x 线）；仅剩 `docker` 组权限一项硬阻塞待管理员开通。
