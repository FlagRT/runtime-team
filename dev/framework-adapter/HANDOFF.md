# framework-adapter 复现与运行交接

更新：2026-11-11。先看[当前状态](STATUS.md)；本页只列操作路径、环境身份与恢复注意事项。

本期dev4发布：功能提交78358ab，版本锁见package.lock.json，材料见[阶段入口](docs/阶段进展与验证入口-20261111.md)。本地发布构建在`Project/outputs/framework-package/20261111-dev4-release-02`及`-03`，核验在`20261111-dev4-publication-audit-02`；初次audit-01因扫描到10-14历史非标准JSON而停止，后限定本期证据后通过，旧记录保留。两次构建源码均匹配提交，16个运行时文件与Qwen实测wheel相同。下文实验轮次的“未发布”是当时状态，不覆盖本期发布记录。
逐次命令、完整SHA及旧运行记录见[运行交接详录](docs/archive/session-20261103/运行交接详录-截至20261102.md)和各日期报告。旧路径只用于定位，资源、权限、文件存在性执行前重查。

## 接管前

1. 读[协作规则](/Users/empty/PhD/Workspace/DomesticRuntime/协作/协作与交接.md)与[登记](/Users/empty/PhD/Workspace/DomesticRuntime/协作/会话登记.md)，确认旧执行者已停止。
2. 两仓分别核cwd、分支、HEAD及dirty；保留既有本地整理。dev2现已固定源码提交，按package.lock.json重建，不再依赖未提交功能代码。
3. 功能代码在独立framework-adapter仓；协调仓只维护本方向测试、环境与材料。
4. 11-03迁移已完成并取得回执，下文PID仅为历史记录，不再发送停止信号或重复迁移。10-29资源释放是历史观察，不是现在的资源保证。

## 本地路径

| 材料 | 绝对位置 |
|---|---|
| 功能代码与构建 | /Users/empty/PhD/Workspace/DomesticRuntime/framework-adapter |
| 集成case与报告 | /Users/empty/PhD/Workspace/DomesticRuntime/runtime-team/dev/framework-adapter |
| 实验原始备份 | /Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs |
| 安装包与构建记录 | /Users/empty/PhD/Project/DomesticRuntime/outputs/framework-package |
| 人可读的产物目录 | [运行与构建产物索引](/Users/empty/PhD/Project/DomesticRuntime/outputs/README.md) |
| 计划与分工 | [执行计划](/Users/empty/PhD/Workspace/DomesticRuntime/协作/执行计划.md) |

这些是本地路径，不放进对外交付包。正式报告引用仓内相对链接或固定Git提交。

## 11-02旧P800到期与迁移待办

总组通知旧机43.180.254.67:26008于2026-11-07清理；新机81.70.153.29:26008、152.136.33.169:26008。账号cgu135，认证材料按需读取，不抄入报告。早前核验权限不足，用户随后通知sudo开通并要求重试；以下保留早前检查背景，下午实况以“权限恢复与后台迁移”为准。

### 11-03迁移前核验

本节为当日权限开通前检查，拒绝结果已被下午复核替代；不是当前权限结论。

| 主机 | 登录及权限 | 存储 |
|---|---|---|
| 旧43.180.254.67，VM-0-2-ubuntu | SSH成功，uid1018/gid1019，仍在sudo组；本人五个目录均在 | /data2约2.3T可用；三批tensors合计6123文件、129303099624字节（120.423GiB） |
| 新81.70.153.29，VM-20-4-ubuntu | SSH成功；密码认证后的sudo -l明确拒绝，Docker socket拒绝；uid1018/gid1019 | /data2约5.1T可用，/data1约5.8T；均不可写，无/data1/cgu135或/data2/cgu135；根盘约20G余量 |
| 新152.136.33.169，VM-20-14-ubuntu | SSH成功；密码认证后的sudo -l明确拒绝，Docker socket拒绝；uid1018/gid1019 | /data2约5.2T可用，/data1约5.8T；均不可写，无/data1/cgu135或/data2/cgu135；根盘约20G余量 |

两台新机都有rsync和xpu-smi命令，但未运行算子、未核Docker镜像或占卡状态；命令存在不等于环境验收。最小解除条件是在一台新机创建本人可写的/data2/cgu135；数据复制本身不要求Docker。后续容器实验另需sudo或Docker权限。

检查命令：hostname、id、df -h、ls -ld、command -v rsync/xpu-smi、sudo -l、docker info、test -w。三批tensors用find统计文件数与字节；这不是SHA校验或数据内容完整性验收。旧机du显示约14G/14G/94G；若干本人实验的tmp/xpujitc-temp子目录由容器创建、普通账号不可读，后续迁移明确排除可再生编译临时目录或用旧机既有权限处理，不删除、不把du警告说成数据丢失。

旧机共享OneRec/Qwen权重路径存在，但未核完整权重哈希。两个新主机首连记录了SSH主机key，本轮未改服务器权限。三条控制socket均以ssh -O exit退出，返回0；无后台迁移。本地磁盘约521GiB可用，但按用户决定不启动大文件下载。

| 本人旧机资产 | 位置 | 已有本地备份 |
|---|---|---|
| OneRec原生对照约94GiB | /data2/cgu135/framework-onerec-xpu-20261029/onerec-native-01/tensors | 只有轻量结果，未全量备份 |
| Qwen原生对照约14GiB | /data2/cgu135/framework-qwen-xpu-20261029/qwen-native-01/tensors | 只有轻量结果，未全量备份 |
| 模型审核Qwen对照约14GiB | /data2/cgu135/framework-workload-xpu-20261029/workload-qwen-native-01/tensors | 只有轻量结果，未全量备份 |
| 源码、wheel、审核与日志 | /data2/cgu135/下framework-xpu-20261029、framework-qwen-xpu-20261029、framework-extensions-xpu-20261029、framework-workload-xpu-20261029、framework-onerec-xpu-20261029 | 五批轻量归档在本地；DeepFM原生张量也在本地 |

OneRec共享权重在/data1/chenyuanxin/models/OneRec-8B，Qwen共享缓存在/data1/dinghaisong/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B。先确认总组迁移安排，不移动/删除他人材料；本方向对照张量与可重新下载的模型权重不是一回事。

执行顺序：重核源文件与目标容量/归属 → 选择本人持久目录 → 复制必要资产 → 比较逐文件SHA与文件数 → 记录新路径 → 保留旧副本至确认。建议11月6日前完成。下午已获用户继续授权并启动复制，未搬动公共模型。

镜像优先用总组新机基座。P800历史docker save曾生成无效小包，export/import会涉及新镜像身份，不直接复制Docker根目录或把停机容器当备份。详见[公共归档说明](https://github.com/FlagRT/runtime-team/blob/71197e73d13021556de04ff04c42c89b3bfc8f73/dev/images/kunlun-operator-runtime/v2/ARCHIVE.md)。

### 11-03权限恢复与后台迁移

完成回执：17:06:47复制结束，17:08:08逐文件SHA和计数通过，runner.result为exit_code=0。20:13复核旧机898744/898771均已不在进程表，并从152机直接取回[verification.txt](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261103-p800-migration/verification.txt)，与源manifest SHA一致。以下启动/PID/停止说明是本次复现记录，不表示任务仍运行。

15:22—15:31复核：81与152均通过密码认证的sudo -l，允许ALL。152的sudo docker info通过，Docker 29.1.3、根目录/data2/docker，现有v2镜像ID cac5ddae5f62（tag：2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64）；没有启动容器，尚未核env/v1补丁或新模型资格。81本轮只确认sudo，不把152的Docker结果算给81。

- 新建本人目录：152机/data2/cgu135，cgu135所有，0750；/data2约5.1T余量。目的根目录为/data2/cgu135/migrations/from-43.180.254.67-20261103，初始不存在，未覆盖旧材料。
- 源：旧机/data2/cgu135下上表五个framework-*-20261029目录。排除可再生tmp/编译临时目录；9395常规文件、129387717740字节，未发现其余不可读常规文件或符号链接。源SHA清单于15:27完成。
- 旧机审计目录：/data2/cgu135/migration-audit-20261103；source.sha256的SHA256为27caf143978c199008a2ef14bb7d08b91bc2ee7a936e98cd5836482d4cf92992。该清单已保存本地；它证明源身份，不证明目标已复制。
- 执行脚本：[migrate_and_verify.sh](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261103-p800-migration/migrate_and_verify.sh)，旧机同名脚本在审计目录。rsync -a --no-owner --no-group --exclude=tmp/ --partial --stats --bwlimit=102400 --timeout=120，不使用--delete。经新机公网26008传输；私网22单次尝试不通。
- 原前台rsync已由本人SIGINT结束（退出20），为切后台的预期暂停；已传文件保留。15:31启动nohup + setsid后台续传：旧机脚本PID/进程组898744，rsync子进程898771（PID随任务退出失效，操作前核身份）。不占卡，不删除源文件。
- 输出：旧机审计目录runner.log、rsync-resume.log、runner.result。runner.result尚未出现表示仍运行或异常失联，必须同时核进程；exit_code=0且runner.log出现VERIFIED才可认定脚本完成。目标migration-audit/verification.txt还应包含9395文件、129387717740字节和上述manifest SHA；脚本逐文件sha256sum --quiet -c及计数均通过才写该回执。
- 远端传输控制socket在旧机/tmp/cgu135-migration-target-20261103.sock；任务结束前不要关闭。后台不依赖本机SSH连接；不会自动通知或更新STATUS。检查完成后取回回执，再更新STATUS和登记。
- 必须停止时，先核旧机ps -p 898744 -o user,pid,pgid,args与pgrep -P 898744，确认仍是上述本人脚本，再对该专用进程组发送TERM（kill -TERM -- -898744）；保留partial以便续传。不要对全部rsync、ssh或其他用户进程发信号。若需重试，先核失败原因及既有runner.result，使用新运行回执名，不覆盖失败记录。

本地轻量审计目录：[20261103-p800-migration](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261103-p800-migration/)，含source.sha256、runner.result、runner.log、verification.txt和执行脚本。五个源目录名在152目的根下保持不变，旧副本保留。数据迁移不等于环境复建或模型验收；共享权重和Docker根目录未迁移。完成后不需继续依赖传输socket，不凭历史PID操作进程。

## 各平台的复现入口

### 11-11 Qwen embedding资格及模型三路

- [方法与结果](cases/embedding-config/README.md#qwen-embedding独立资格及模型三路)、[核验摘要](docs/evidence-qwen-embedding-20261111/summary.json)、[实测demo](cases/embedding-config/demo/README.md)。本地根`/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261111-qwen-embedding`；PPU本人根`/bmcp_lvm_fs/cgu135/framework-qwen-embedding-20261111`。权重只读挂载`/bmcp_lvm_fs/hliu553/models/Qwen3-Embedding-0.6B`，完整权重SHA `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`、config SHA `b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9`。
- 复用PPU v2 image `1ba7ead…`、驱动`1.5.1-1d747a`、公共runtime0.4.0、dev4 wheel `1edf33728e67225c6f2a14d4264917fe7a7f82f36b9e911e23f940ad47699a49`及FlagGems wheel `4b0ff053…`；安装内容身份见摘要，未重构建或改功能包。执行前工作盘868GiB可用，结束852GiB；每次重新检查card1空闲后映射cuda0，旧记录不是预约。
- `prepared01`为本地移植稿；`input01`（64文件、tar SHA `aac32d48f942af86c0af472715f06031544271b7bb0c154c0b5d0d1f73438e26`）只跑资格。`input02`（66文件、SHA `95afb021e837619d5a18086a928303373f9e846193a49ceca57a77de299995ed`）新增人工审核的算子资格，跑原生、重复、两候选及关闭/缺审核。`input03`（72文件、SHA `676f26dc6cebc07c244fad37bf30cb5b13aac124c8ca173716f441cfa0fa4b55`）只增加模型审核材料，跑正式开启；三份执行代码一致，不覆盖旧快照。
- 执行：`sudo -n /bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python -B <root>/qwen_batch.py --root <root> --input <对应input> --model <只读权重目录> --card <重核空卡> --phase <qualification/native/candidates/controls/enabled>`。产物分别`qemb-qual-01`、`qemb-native-01`、`qemb-repeat-01`、`qemb-candidate-01/02`、`qemb-off-01`、`qemb-unknown-01`、`qemb-on-01`。重跑要新ID，勿覆盖历史目录或直接重启历史compose。
- `operator-review/`为12组独立测试后的限定资格；`model-review/`为两候选完整对照通过后的模型记录，`model-decision.json`包含原生/重复/两候选的SHA。`native.json`、`repeat.json`、`candidate-01/02.json`为便利副本。两候选result的SHA同为`948666c8…`是本次逐值一致的结果，不是复用同一执行；两容器独立开始/停止时间见原件。
- `collection01/`含106份轻量原件的`evidence.tar.gz`及逐文件SHA；归档SHA `9b82e8e7dcd358b7383e8cdc362d2188f417cfc93bb87e092c711e5e2637acc3`。原生1863份张量共14,302,794,504字节保留远端`qemb-native-01/tensors`，本地清单为`collection01/qemb-native-01-tensor-sha.json`；没有全量下载。8容器停止回执为归档中的`cleanup.host.log`，每轮实际baseline/demo、strict-start和wheel安装记录同档保留。
- 离线核验：`PYTHONPATH=<功能仓>/src <CPU python> -B cases/embedding-config/verify_qwen.py --collection <本地根>/collection01 --input <本地根>/input03 --output <新摘要路径>`；5项接线/审核控制用`QWEN_EMBEDDING_INPUT=<input03> PYTHONPATH=<功能仓>/src <CPU python> -B cases/embedding-config/test_qwen.py`。本轮无硬件失败轨迹；各阶段审核及全量比较通过，未改容差。
- 已停止8个本人容器、清除sudo票据并退出SSH89883；当前接管以会话登记为准。两仓未提交推送，发布锁不变。

### 11-11 embedding数值定位

- [方法及结果](cases/embedding-config/README.md#11月11日同输入数值定位)、[离线核验摘要](docs/evidence-embedding-config-20261111/diagnostic-summary.json)。本地根`/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261111-embedding-diagnosis`；PPU本人根`/bmcp_lvm_fs/cgu135/framework-embedding-diagnosis-20261111`。同一v2镜像`1ba7ead…`、驱动`1.5.1-1d747a`、dev4 wheel `1edf3372…`和FlagGems wheel `4b0ff053…`；没有重构建功能包。运行前本人权限、镜像与容量重核，工作盘868GiB可用。
- `input01`（365文件）为三路径诊断，tar SHA `7a640a4190fb599293b9f3f5f4799898b2b449e811278fc5ba9f08e4b05c37f0`；派生自上一节统一配置`input03`并加294个原生参考张量。输出`embedding-diagnosis-01`包含native三步参考、6组采样、前4组60次对照。`captured.pt`也在本地根，可供直接复现。
- `direct-input01`（367文件）tar SHA `c131179f8a2ac322a40ddc946bde1d442a105b87b11f2e155531f5b8080343f7`；输出`embedding-direct-01`在读`triton`发行包元数据时失败，尚未运行目标算子。`direct-input02`仅修正版本读取为`triton.__version__`，tar SHA `f78888c2f74e1d8265c26327f6371319621568e67e3bf4ad8ebc2d3765c7d2a5`；输出`embedding-direct-02`完成直接目标/原生、重复/唯一索引对照。旧失败输入和结果均保留。
- 完整启动格式：`sudo -n /bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python -B <root>/<input>/run.py --root <root> --run-id <新唯一ID> --card <重核空卡> --stage qualification`。此处runner的历史stage名只作兼容入口，实际命令是`diagnose.py`或`direct_replay.py`，不能将名称当算子资格通过。本轮映射card1→cuda0，三次连续空闲检查；每轮独立baseline/demo、限时及finally停止。
- `<run>-collection/`保存停止回执、resource-final、files.json、receipt.json和evidence.tar.gz。3批368/12/20份，共400份原件；归档SHA分别`4b19b4e4bcbd8bdd51fb79d61664c21a2e10cbb839139aba784bac2fdc2519a4`、`2be804ad71643ae2b96c66604307786920c4334c95d8f62267d70d4cfd4e76cf`、`a5e6a28d8f4f3cbfdcbcdf336881fd663223b77bc4f23ff622d26ca0e5d068c8`。本地`result01.json`是首轮结果便利副本，正式证据以归档+manifest为准；不重复另建“最新”报告。
- 离线验证：`<CPU python> cases/embedding-config/verify_diagnostic.py --root <本地诊断根> --reference <本地20261111-embedding-config根> --out <新摘要路径>`，逐文件SHA/大小、294原生张量及所有三路径配对/直接重复差异已核验。首次本地case控制未设PYTHONPATH而缺包，按`PYTHONPATH=<功能仓>/src`运行4项通过；不是包或设备故障。
- 本轮3个容器均Exited/Pid0/OOMfalse，SSH已退出；当前可接管状态只看会话登记。当轮Qwen准备稿未上卡，同日后续实测见上方Qwen小节。尚未推送。

### 11-11 embedding统一配置回归

- [case与结果](cases/embedding-config/README.md)、[核验摘要](docs/evidence-embedding-config-20261111/summary.json)。本地根`/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261111-embedding-config`，远端PPU本人根`/bmcp_lvm_fs/cgu135/framework-embedding-config-20261111`。本轮card1→cuda0，每次连续3次空卡检查；镜像1ba7ead、驱动1.5.1-1d747a与历史一致，运行前存储869GiB可用。实际demo/baseline在每个输出中。
- 构建根`outputs/framework-package/20261111-embedding-config-dev4-03`：178项控制通过，wheel SHA `1edf33728e67225c6f2a14d4264917fe7a7f82f36b9e911e23f940ad47699a49`。-01是混合作用域CPU测试误传展开梯度触发旧一元契约失败；-02修正测试输入后通过；-03补齐说明重建通过，-02/-03运行时代码一致。包仍本地dev4，发布锁不变。
- `input01`为prepare误识别wheel文件名产生的未完成副本，无上机；`input02`为-02 wheel输入，仅本地；`input03`为-03 wheel实测输入，69文件，tar SHA `db26b74bbb4b9f31272b13a2243826c7990661f280e4567efc0874a9e3ddb959`。新模型身份绑定整个安装包内容`package-sha256:9fafa4e10004d5a5332ee6ac07347992c64be993169d9494c7bc013a5e228d4b`，没有沿用旧模型审核。
- 执行格式：`sudo -n /bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python -B <root>/input03/run.py --root <root> --run-id <唯一ID> --card <重核空卡> --stage model --mode <模式>`；非native加`--reference <root>/unified-native-01`。五轮为`unified-{native,repeat,candidate,off,unknown}-01`。batch01.py在candidate失败后停止，off/unknown经核失败为数值问题、容器已停后分别执行。没有重复失败候选。
- 本地`review-failed/`为`review_model.py --decision reject`生成的模型审核，`failure-gate.json`为`check_failure.py`用该实测身份检查普通/严格启动的CPU结果。源码复现见case；不能将此失败审核改成通过以启动on。
- `collection01`含63份轻量原件，evidence.tar.gz SHA `48cca6f2a646db2e0b508d6a50428f893786754f011914cd69bd1b23915a0bc9`，本地`evidence/`逐文件核验。`final-audit/native-reference.tar.gz` SHA `738dcc7d71116b4558e60ed0e2890ea33b36b9c25dcd2310f294403d2d46afec`，294份参考张量逐大小/SHA核验，解包在final-audit/tensors。`verified-summary.json`由case/verify.py产生；停止回执见final-audit/containers.json。
- `qwen-input01`及case/prepare_qwen.py仅为本地准备，没有传到服务器或跑资格；待DeepFM差异定位后重新审查再用。PPU初次SSH超时且路由en0，用户重开VPN后utun4恢复；不能把客户端显示连接等同于目标路由可用。当前连接和可接管状态看会话登记。

### 11-10 PPU DeepFM embedding扩展

- [报告](docs/PPU-DeepFM-embedding接入验证-20261110.md)、[case](cases/embedding/README.md)。本地根`/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261110-embedding`；远端本人根`/bmcp_lvm_fs/cgu135/framework-embedding-20261110`，物理card1→cuda0。旧卡号不是预约；当前任务/SSH状态只看会话登记。
- 功能包`src/flagrt_adapter/embedding.py`为唯一新增功能实现；已核算子组FlagOP固定2591d081中PPU原生委托，本轮实际使用固定FlagGems公开前反向函数。未改内核。新多输入资格记录不能与旧一元记录混用。
- 构建`outputs/framework-package/20261110-embedding-dev4-01`为最初连续梯度版；`...-02`增加展开梯度值不变连续化，166项CPU控制通过，wheel SHA `228763f46b2e0d10ee91b14ee4f92a9b2e8b378bf68245efde1ec5722353d12a`。embedding.py/bridge.py与实测快照一致，README/PACKAGE随后补说明。首次本地CPU控制发现Parameter被过严拒绝，修正后构建；未形成硬件失败。
- embedding-input01（60文件）用于独立资格，沿用旧adapter wheel但直接调用固定provider；input02（69文件）用于routes-01；input03（69文件）用于最终routes-02及模型native/repeat/candidate/off/unknown；input04（73文件）只增加模型审核证据，用于on。input03/04源码与wheel相同，不能用on批准其他模型/精度/算子组合。
- input01/02/03/04 tar SHA依次为`5f46478b03e6060be3b75135620e8374c832b3472d39d63a7399fd853faa32db`、`10c98358b73092e7dc983ea4e5fca20694bd129a868a3d9f820dd22de0e46946`、`ffd69782c09b553ffbeb749fec28d5fcd8733bc8c57ff1467ade17f8e8bd7155`、`2b30dff6dd999a0a0aec58c14874609957f3be8ddadd6e8f31556102c59129ee`。
- runner格式：`sudo -n /bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python -B <input>/run.py --root <root> --run-id <新ID> --card <重新核空卡> --stage model --mode <模式> --reference <root>/embedding-native-01`；native不传reference，qualification用input01，routes用input03。batch01.py是串行模型调度，不是后台服务。启动routes-01前曾因sudo票据过期被拒，重新认证后首次实际执行，未重复创建同名输出。
- 九轮输出名为embedding-qual-01、embedding-routes-01/02、embedding-native/repeat/candidate/off/unknown/on-01。每个输出保存实际demo、strict-start、输入SHA及停止回执。case/demo为on的原样归档；重跑需通过runner新建输出，不能直接启动历史容器。
- collection01含113文件归档及SHA清单，tar SHA `d259a82e67a5116a62596ea92d325ec3450a767cb23e1fa69fb2c78f6b85e11c`；本地evidence解包后verify.py全核通过。native-reference.tar.gz包含全部294份参考张量，SHA `c87b3d9879636d316a17125fda3b638945707c28c3cb018f24208aa29c6502d0`，逐项大小/SHA核验通过。cleanup.json/resource-final.log是独立收尾回执，9个本人容器均停止。

### 11-10 三模型清单与ReLU扩展

- 报告：[清单与扩展验证](docs/三模型算子清单与ReLU接入验证-20261110.md)；[case](cases/operator-inventory/README.md)。功能实现仅在独立包bridge.py/config.py；协调仓增加采集/对照/实验准备和结果核验脚本。
- 本地根：`/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261110-operator-inventory`。远端PPU：`/bmcp_lvm_fs/cgu135/framework-inventory-20261110`。当前会话/资源状态只看会话登记，以下目录不是继续占卡的授权。
- `input01`是三模型原生采集输入（67文件）；tar SHA `ec922dd957366e15cbb69eeab13ca506015525243c305787614724b1f9abcac7`。结果分别`inventory-deepfm-01`、`inventory-qwen-01`、`inventory-onerec-01`。使用冻结公共runtime 0.4.0。Qwen只读权重`/bmcp_lvm_fs/hliu553/models/Qwen3-Embedding-0.6B`；OneRec只读权重`/bmcp_lvm_fs/chenyuanxin/models/OneRec-8B`；DeepFM源码为input/model。模型/源码SHA均在result.json。
- 固定FlagOP源码`flagop/`来自2591d081188d88c2ef75c2798f263186ec3837e2，tar SHA `baf197e1073b7c3d3ce29ff196b3e49f0db64aded4de02cb6e13b5ad0a913923`。`comparison02`是有效对照输出；`comparison01`仅为本地解析脚本字段名错误产生的空目录。完整原始结果为根下deepfm.json/qwen.json/onerec.json。
- ReLU安装包构建：`/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-package/20261110-relu-dev4-01`，wheel SHA `4b9d03b839b628cf631fd804498c4130777a50060f27e920b3ea733b9ade1182`，152项CPU控制；实测bridge/config与该快照逐文件一致。后续仅补README/PACKAGE说明，构建时的source快照保留。
- `relu-input01`仅本地prepare缺少wheel路径的未完成目录，无manifest、从未上机。`relu-input02`用于`relu-qual-01`；`relu-input03`加入已审单算子记录，用于`relu-routes-01`及首轮native/repeat/candidate。candidate-01因路由检查器误判停止，证据保留。`relu-input04`修正检查器并加入relu_routes.py，原生参考重新建为`relu-native-02`，随后repeat-02/candidate-02/off-01/unknown-01；不能把首轮结果套到新探针身份。
- `relu-input05`加入限定模型审核，仅运行`relu-on-01`，继续引用native-02。input03/04/05 tar SHA分别`b99927d24f5358043eddbe14f8485e789581eab35dec00f7a8170394e0e8610a`、`974cff830bd926cd5a238c84e752f7b6d5a79de5694df0e051d403a8a9e8f78e`、`29a1321f23c8affa3eba79536df2b939a2f822808eb232f55de94c5393e187b7`，本地/远端传输核验一致。仓内review-20261110对应input05；其他平台/形状/目标算子组合不能复用。
- PPU控制解释器`/bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python`，仅用于宿主机baseline/YAML。命令格式：`sudo -n <control-python> <root>/<input>/run.py --root <root> --run-id <唯一ID> --card <重新核空闲> --stage <inventory|qualification|routes|model> ...`。inventory另带kind/model；model带mode，非native带reference。scope及实际参数见case README。
- 每个输出含输入SHA、设备采样、严格基线回执、原始探针结果、实际demo和容器结束状态。本地/远端`collection01`含归档和manifest，本地`evidence/`为182文件解包，`verified-summary.json`为verify.py核验结果。归档SHA `855143e55cb41020cb03c2a1e341e995eeda5dfe9b5835e1a0205add83ec048e`。
- `native-reference.tar.gz`同时备份native-01/02两份原生参考，SHA `f6c8b5e50b140bb430f2a9c3d411d74c4395706ed3194b52ad73e31b47fa89d1`，588份张量逐文件SHA/size与collection01清单一致。`cleanup.json`、`resource-final.log`为归档后的补充清理回执：14个本人容器Exited/Pid0/OOMfalse、卡1最终26MiB/0%/无进程。SSH27432已退出、sudo票据清除，无后台任务。

### 11-10 PPU Qwen/OneRec模型验证

先读[报告](docs/PPU两模型接入与安全准入验证-20261110.md)和[cases/ppu-models](cases/ppu-models/README.md)。当前结果只看STATUS；活跃任务及连接看会话登记。沿用同日PPU dev4 wheel，未改功能包或发布锁。

- 本地根：Project/outputs/framework-runs/20261110-ppu-models；远端根：192.168.5.112:/bmcp_lvm_fs/cgu135/framework-ppu-models-20261110。Qwen资格/原生用qwen-input02，重复/候选/关闭/缺审核用qwen-input03；OneRec资格/原生用onerec-input01，其余用onerec-input02。inputs.json声明输入SHA，port.json记录移植来源；不要覆盖冻结目录。
- 只读共享权重：Qwen为/bmcp_lvm_fs/hliu553/models/Qwen3-Embedding-0.6B；OneRec为/bmcp_lvm_fs/chenyuanxin/models/OneRec-8B。OneRec经普通账户的公共模型目录元数据查询找到，三片SHA已核；没有进入他人容器或写共享模型。此前下载尝试已停止，部分文件保留、未用于测试。
- 运行：sudo -n /bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python <冻结输入>/run.py --root <远端根> --run-id <新ID> --card <重新核空闲的卡号> --model <上述只读路径> --stage model --mode <模式>。非native再传--reference <远端根>/<模型>-native-01；资格用--stage qualification。实际运行卡1→cuda:0，每轮连续3次空闲检查，不是卡预约。
- Qwen资格qwen-qual-02，其余qwen-{native,repeat,candidate,off,unknown}-01；OneRec资格onerec-qual-01，原生onerec-native-01、重复onerec-repeat-02、候选onerec-candidate-01、关闭onerec-off-01、缺审核onerec-unknown-01。每轮summary保存实际命令、输入身份及容器停止状态。
- 两模型候选均在更新参数前梯度超差。qwen-review-failed/onerec-review-failed为人工核验后的失败审核和CPU决策证据，不签发passed模型审核。正常on需要schema3模型审核；candidate仅用于独立资格实验，不能改名当作已批准开启。
- 收尾已用case/collect.py核停止状态、输入SHA并打包轻量证据：collection01归档SHA为0a2304b49bd79f9c4d6a843d4e6968eb9014b2bbc633ee774e0628c7fb5f4783；本地evidence/含159份原件。verify_collection.py逐文件及结果核验通过，输出verification.json。两份原生张量SHA清单在collection01，原生大张量约108GiB留服务器，不默认下载。12个实际测试容器最终Exited/Pid0/OOMfalse，cleanup-1110.log为复核，连接最终状态见会话登记。
- 早期qwen-input01含Mac隐藏元数据，被wheel数量检查拦截；重新打包用COPYFILE_DISABLE=1 tar --no-xattrs。onerec-repeat-01.host.log是sudo票据过期、未启动容器；不得覆盖失败记录。
- 网络曾缺VPN路由，也曾路由存在但SSH短时超时；恢复后先查旧任务。不能仅凭SSH超时认定用户断开VPN。本轮未改VPN或路由，旧utun地址不可硬编码。

### 11-10 PPU DeepFM训练与安全准入

- 读[报告](docs/PPU-DeepFM训练与安全准入验证-20261110.md)、[case](cases/ppu-deepfm/README.md)。原生/重复/关闭/缺审核通过，候选第1步参数超差；不得签发passed模型审核或复用P800审核。
- 本地：Project/outputs/framework-runs/20261110-ppu-deepfm；model-input01/02/03/04为不同冻结输入，evidence为完整原始取回结果（含294份原生张量及失败偏置），verify_results.py校验并生成summary.json。当前case与input04模型执行代码一致，新增failed审核是本地收尾材料，未下发到该轮服务器输入。
- 远端：192.168.5.112:/bmcp_lvm_fs/cgu135/framework-ppu-deepfm-20261110。沿用上午本人control-venv/python（在framework-ppu-20261110下）；镜像及两wheel不变。实际命令模板：sudo -n <control-venv/bin/python> <本轮root>/model-input03/run.py --root <本轮root> --run-id <新ID> --card <重核空卡> --stage model --mode native。其他模式用--reference <本轮root>/dfm-native-1110-02；off/unknown使用model-input04。成功资格是model-input02、stage qualification、dfm-qual-1110-04。不要覆盖旧输出/输入。
- 原生dfm-native-1110-02、重复dfm-repeat-1110-01、直接候选dfm-candidate-1110-01、关闭dfm-off-1110-01、缺审核dfm-unknown-1110-01；后5轮物理1→逻辑0，算子资格物理15。每轮summary含实际命令、输入SHA及停止状态。模型原生loss三步为1.0768573284、0.8497074246、1.1208320856，仅供识别当次固定合成配置。
- 归档deepfm-evidence-1110.tar.gz两端SHA a37fa476746c8e3e7917b7a80dff8e5093da37c1a5c3697600d1e599833dfb9c；381文件已计算SHA。7个本轮本人容器均Exited/Pid0/OOMfalse。无后台训练；交互SSH及control socket收尾状态见会话登记。
- 网络：本轮默认SSH报EADDRNOTAVAIL，路由及utun4地址仍在；显式-b 10.32.63.18后成功。此为当时VPN地址，后续必须重查，不能硬编码或据一次失败宣称VPN断开。未重启VPN/改路由。
- 资源检查已改为连续3次无进程/0%/≤64MiB，原因和旧32MiB拒绝记录见报告。旧卡号不是预约。新snapshot校验完整输入SHA后再启动，baseline/demo保留原执行路径，不直接重启历史容器。

### 11-10 PPU算子资格与接入验证

- 先读[本轮报告](docs/PPU算子接入与安全回退验证-20261110.md)及[case](cases/ppu-unary/README.md)。本地`/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261110-ppu-adapter`含公开源归档、构建wheel/日志、input01/02/03/04及evidence原始轻量记录；适配包构建在`outputs/framework-package/20261110-ppu-dev4-01`，146项通过。所有本轮修改未推，旧dev3锁不动。
- 远端仍为本人`/bmcp_lvm_fs/cgu135/framework-ppu-20261110`；`provider-build`是无卡断网离线构建，源码归档SHA c0914474…，wheel SHA 4b0ff053…；不依赖他人容器或源码。收尾发现适配input02/04与同日原生输入重名，已将适配输入移为`adapter-input01/02/04`，原生input02/04从本地保留快照恢复；新旧结果和原日志未改。后续复现用adapter-*路径，历史命令中的input路径仅用于审计当次执行。以后输入必须用case前缀且远端拒绝已有目标。
- 成功资格命令：`sudo -n /bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python /bmcp_lvm_fs/cgu135/framework-ppu-20261110/input02/run.py --root /bmcp_lvm_fs/cgu135/framework-ppu-20261110 --run-id qual-1110-02 --card 7 --stage qualification`。18项PASS；input01因探针查询不存在的triton发行元数据失败，修正后运行input02，两轮证据均保留。
- VPN曾中断，用户重启客户端后恢复；12:53核到资格轮已于10:53停止。input03传输失败、未执行；最终三路使用input04（含review及cleanup异常落盘增强），包SHA `40ce6dde0041159d1f3500b234e9d32b3e16f99ad4c7da9beb243a2942ec9005`，两端一致。
- 三路命令为同一control-venv/python运行`input04/run.py --root /bmcp_lvm_fs/cgu135/framework-ppu-20261110 --run-id routes-1110-01 --card 7 --stage routes`，通过sudo nohup启动，断开SSH不影响；18组PASS，实际12:55启动、12:57停止。启动PID3744932是已结束的历史标识，不能再凭它停止进程。本人所有cgu135-ppu-*容器已Exited，无活跃任务；sudo票据清除，SSH退出。下一轮仍须重新核资源。
- `adapter-evidence-1110.tar.gz`的SHA `8d177586f7039d66bc4c3cd4008252d116f2009576d61342e0c4e117df977325`已核两端一致，本地evidence/含三轮summary/result/probe/install/demo，排除可再建packages/cache。仓内仅放结构化摘要、审核及实际demo。原件保留；不把SSH中断归为设备或算子失败。

目录恢复包`native-input-restore.tar.gz`两端SHA为`b0bac23a31f80a8f126d3f77486226dfcb7e73f77e0647720642d8bb03da94c0`；原生input04实际demo由同日`baseline-demos.tar.gz`恢复。收尾逐文件SHA复核：原生12文件、adapter-input01的52文件、adapter-input02的52文件、adapter-input04的55文件全部无差异。只读校验连接已退出，无新任务。适配器代码/原始输出字节未变；已结束容器的历史挂载路径不得直接docker start，复跑使用runner生成新demo和唯一输出。

### 11-10 PPU原生最小验证

- 机器192.168.5.112，账号cgu135；镜像`sha256:1ba7ead15378160ba9aa1dafcfb77686de6b16732590ffab062bab19c60c32b8`，driver1.5.1-1d747a；Docker root `/bmcp_lvm_fs/docker`。检查时/data已满，根盘约13GiB，/bmcp_lvm_fs约1.1TiB可用，不使用/data存新结果。
- 本人新建`/bmcp_lvm_fs/cgu135/framework-ppu-20261110`，独立control-venv使用宿主Python3.11及PyYAML6.0.2，仅用于公共baseline/up脚本；未修改系统Python或共享镜像。公共工具和镜像基线来源于本地发布快照framework-release-20261103/runtime-status，SHA保存在每轮summary。
- 成功命令：`sudo -n /bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python /bmcp_lvm_fs/cgu135/framework-ppu-20261110/input04/run.py`。input04/demo含实际compose及自动生成env.baseline，先export再up.sh --strict。映射控制节点和物理ppu7，CUDA_VISIBLE_DEVICES=0，实查device_count=1；shape32×64，原生SiLU/sigmoid各FP32/FP16/BF16前反向，以相同量化输入的CPU FP32为参考。阈值在probe.py/result.json中，未放宽重试。
- 结果results04为PASS、6组；torch2.10.0、triton3.6.0、flagtree0.7.0+ppu.gite223f42c；FlagGems未安装。没有安装本方向wheel或做接入/回退实验。此轮只验证原生可用性。
- 失败保留：input/results首轮在FP32 SiLU反向后读取非叶子.grad报错，是新探针的共享输入requires_grad处理错误；input02修正为detach/clone，但results02和results03在创建容器前因显存34MiB超过本轮保守32MiB门限被拦截（不是硬件测试失败，也未证明有他人任务）。最终input04选择重新核空闲的7号卡，通过。
- 本地输入及四轮结果：[20261110-ppu-smoke](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261110-ppu-smoke/)。baseline-demos.tar.gz保留首轮及成功轮实际env.baseline/compose。远端原件保留；results目录不覆盖。本人`cgu135-ppu-smoke-20261110`及`…-04`均Exited/Pid=0，另外两轮没建容器；结束时ppu-smi没有本轮进程，SSH退出、sudo票据清除。尚未提交推送。

### 11-10 P800当前可用性复测

本轮已结束，152与81各跑一次inspect，均失败后停止，没有进入资格和模型三路。两机结果路径均为`/data2/cgu135/framework-check-20261110/inspect-card7-01`；本人容器`cgu135-xpu-inspect-card7-01`均Exited、Pid=0、非OOM，SSH退出并清除sudo票据。未改驱动、未重启、未推送。

命令（每台执行一次）：`sudo -n python3 /data2/cgu135/framework-regression-20261104/input/assets/run.py --public-root /data2/cgu135/framework-regression-20261104/input/public --wheel /data2/cgu135/framework-regression-20261104/input/flagrt_framework_adapter-0.3.0.dev2-py3-none-any.whl --output /data2/cgu135/framework-check-20261110/inspect-card7-01 --device 7 --stage inspect`。首次152因输出父目录不存在，在创建容器前报FileNotFoundError；创建本人父目录后执行上述唯一一次上卡检查。后续不要复用已存在的输出目录。

本地证据：[11-10目录](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261110-p800-check/)。152子目录保留summary/probe/inputs/mapping/resource/baseline/demo等；递归复制时root私有临时目录读取被拒，未放宽权限，临时目录未完整备份。81子目录保留原始JSON/log。完整原件仍在服务器。152运行时间09:59:20—10:04:41；81为10:05:06—10:10:27（北京时间）。两次probe均超时退出137，不将容器stop的137解释成OOM。

152驱动实时查询中，10:04:55的`KL_XID_KERNEL_EXCEPTION`指向PCI0000:bb:00、pid3790310，匹配本轮probe；该行记录于工具回执，未另存完整dmesg。81启动前查询恢复正常，但根盘仅约2.5GiB，Docker与输出在余量充足的/data2；未下载镜像。下一轮须重新确认资源、磁盘与底层健康，不能把本轮空卡记录当预约。

### 11-04新152机三路回归

本轮已完成并停止，不是后台待续任务。详细判据见[新机报告](docs/P800新机三路回归-20261104.md)。只验v2/GEMM关闭的DeepFM FP32 batch32三步，没验GEMM opt-in。

- 服务器目录：152.136.33.169:/data2/cgu135/framework-regression-20261104，已归本人cgu135所有；input/为冻结公共快照、case、review和wheel，结果目录为inspect-01、sigmoid-qual-01、deepfm-{native,native_repeat,off,on,unknown}-01。
- 本地完整入口：[20261104-p800-regression](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261104-p800-regression/)，含prepare_inputs.py、host_check.py、run_regression.py、collect_evidence.py、verify_local_evidence.py。prepare_inputs为当前轮冻结脚本，运行会拒绝已存在的input；不要覆盖本轮。
- 命令：在本人新建目录用python3 input/assets/run.py，传--public-root该轮input/public、--wheel该轮dev2 wheel、--device选定SMI号；先--stage inspect，再--stage qualification --operator sigmoid，再--stage deepfm --mode native --model该快照dev/model-defs/deepfm/v1 --batch-size 32 --steps 3。取得有效原生后，以同一reference运行native_repeat/off/on/unknown，三路配置在input/review。完整命令保存在evidence/regression-sequence.json及逐轮summary.json。
- run_regression.py先核新机原生、资格和既有审核环境/workload完全匹配，再执行四轮；不生成passed。移植到另一轮必须修改路径和设备并重核，不能把本轮SMI4当预约。
- 资源：SMI4→PCI0000:84:00.0→/dev/xpu4→逻辑0；7个cgu135-xpu-对应结果目录容器均停止、Pid=0，卡4结束0MiB/0%。没有删除容器。交互shell已退出，sudo票据清除，本轮专用/tmp/cgu135-p800-20261104.sock已ssh -O exit关闭。
- 备份：evidence.tar.gz（SHA 1da6cfd445f8f71762f2a0e695d622428264031f1b2293d5d4afa1f1f6bdd10c）已解到本地evidence/，103文件逐个SHA通过；native-reference.tar.gz（SHA a541ca5d5207e0f1414e4704bb3127ffa2efe693053392b057c38d54b88306b7）约17MiB含本轮完整原生张量。两份均保留服务器原件，不与10-29张量混用。
- 本轮34项case CPU控制日志在cpu-controls.log；输入115文件SHA在input-manifest.json；当前runner相对发布版只改PUBLIC提交标注和source描述。功能包commit6ee9129未改；没有提交、推送或OneDrive写入。

| 场景 | 入口 | 不能沿用的假设 |
|---|---|---|
| 910C旧官方成功环境 | [ascend-official-unary](cases/ascend-official-unary/README.md) | 不等于训练v3；11/27空闲和镜像存在性重查 |
| 910C训练v3环境检查 | [ascend-train-v3](cases/ascend-train-v3/README.md) | up.sh --strict会创建/启动容器，不是只读；已有目标编译失败，不盲目重复 |
| P800单算子及三模型 | [kunlun-unary](cases/kunlun-unary/README.md) | 10-29三模型证据及11-04新机DeepFM三路；均未启用gemm_compat，不覆盖补丁开启配置 |
| 双rank训练配置 | [配置说明](demo/training-configs/README.md)、[10-28门控报告](docs/双rank步骤门控验证-20261028.md) | 仅固定入口；P800 schema3身份采集不自动覆盖此CLI |
| 通用探针映射 | [tools说明](tools/README.md) | 以实际日期报告判定硬件成绩，不因有命令就算已测 |
| 09-22历史推理 | [旧实验复现说明](docs/handoff/实验复现说明.md) | qwen_scoped_adapter是旧原型，不是现行包全功能入口 |

P800选卡必须按SMI号 → PCI地址 → /dev/xpuN → 容器逻辑设备映射，不能按编号相同推断。上轮OneRec为SMI3→PCI0000:2e:00.0→/dev/xpu1→逻辑0，只是当次身份，不是后续预约。每次用新输出目录，不覆盖失败轨迹。

## 安装包与证据身份

11-03发布复现入口：独立仓6ee91294e7ab028f2e804225482f58d24c8ccd5a；协调仓依赖锁及[发布审查](docs/dev2发布审查-20261103.md)在本人分支f550a87。固定源码构建位于Project/outputs/framework-package/20261103-dev2-publish，独立重建位于20261103-dev2-rebuild，均含local.baseline.json、源码快照、wheel、安装副本及CPU日志。审查中间目录20261103-dev2-review不作为发布制品。原始构建清单含本机绝对路径，不上传公共仓。

复核命令为报告中的audit.py；执行参数指向上述两个构建目录、独立仓、P800 case及下述历史wheel。两次构建的100项包CPU控制、34项case控制、3条证据SHA与10个运行时文件一致性已通过，不能将重复重建相加为200种用例或新硬件验收。精确新wheel SHA在package.lock.json及机器摘要，不替换下述历史SHA。

10-29模型审核与OneRec实测使用0.3.0.dev2：
- wheel目录：Project/outputs/framework-package/20261029-workload-01/wheels；SHA de5da93ab72d00456993f74baf98a2b9aeaf1cae0a21844d2423ba4462f83008。
- 独立重建：20261029-workload-rebuild；SHA f6632f78178592238e294630e185ea05cd9f88e785254649e464549a2673d293。10个运行时文件与实测包一致，不能把两个wheel哈希混为一件制品。
- 该版包CPU控制94项通过；OneRec case最终34项通过，均为历史测试，不记作11-03新测试。
- [P800结构化证据](docs/evidence-p800-20261029/)含命令、审核及文件SHA；OneRec source-04和review-02为收尾输入。失败完整梯度并未全存，保留了差异样本，不能补称全量已备份。

重建按独立仓[PACKAGE.md](/Users/empty/PhD/Workspace/DomesticRuntime/framework-adapter/PACKAGE.md)进行，输出使用不存在的绝对目录。当前曾用Python位于/opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python，使用前检查；不自动安装或替换厂商torch。

## 11-04公共训练入口模型审核

dev3固定源码40864f9，package.lock已对齐；发布重建与实测wheel的11个运行时文件一致，见docs/dev3发布审查-20261104.md。公共源码固定719de8c，入口/路径解析助手哈希见[报告](docs/公共训练入口模型审核接入-20261104.md)，两仓未改他人方向。

- 本地原始材料：Project/outputs/framework-runs/20261104-public-entry。prepare.py/input是首轮冻结输入；qualification-input是追加独立资格探针后的第二份case快照，不覆盖初版。run_27.sh、run_11.sh、run_qualification_11.sh、run_on_11.sh记录实际命令；collect_qualification.py、collect_final.py用于原始取证/核验，不生成模型批准。
- 构建：Project/outputs/framework-package/20261104-public-entry-dev3-01（109项，实测wheel191887c6…）及20261104-public-entry-dev3-rebuild（110项）；11个运行时Python文件一致。重建补CLI降级配置拒绝控制与说明，不把不同wheel归档SHA混为一个。
- 11远端：/home/cgu135/framework-public-entry-20261104。三路为public-off-11-1104-03、public-unqualified-11-1104-01、public-on-11-1104-01。三个资格run为qual-native/repeat/candidate-11-1104-01；实际挂载2/3，逻辑0/1，结束已出现他人任务，需重新协调。
- 正式入口仍为已安装wheel的`torchrun --nproc_per_node=2 --module flagrt_adapter.train_cli`；参数MAX_STEPS=3/BATCH=1/SEQ=32/LR=1e-5。正常开启使用review-public-20261104/on.json与精确审核，资格实验独立脚本不能当生产开关。
- 11工具：sudo启动本人run脚本；tools/docker直接调用系统Docker并传私有DOCKER_CONFIG（本进程有root身份，不另套sudo丢失Compose变量）。GIT_CONFIG_GLOBAL只指向本任务git-safe.conf，信任本人固定Gems仓，不修改宿主全局Git配置。工具只含命令/路径，不含密码。
- 大张量：11的qual-native-11-1104-01/train/tensors约27GB，两个rank共3726个loss/梯度/参数张量；仍保留在远端，未全量下载。重复/候选只读此参考，不再写.pt。宿主余约25GB，不启动另一份全量native保存；后续如需清理先做备份/校验并取得删除授权。
- 27：同名任务目录下public-off-1104-01保留aclInit507899/Resource_Busy失败及基线，未进入训练。本人容器已停止。11的off-01为sudo问题，off-02为Compose环境传递问题，均保留，不能写成算法失败。
- 轻量证据：qualification-evidence.tar.gz SHA02533a9b5aba95e533c40bb8da2be22f286eba0fca07a64e338df1326af984ff；final-11/27-evidence及evidence-manifest.json含最终取证。仓内[摘要](docs/evidence-public-entry-20261104/summary.json)记录归档与文件SHA；失败传输的空tar不是证据。
- 资源：11七个本轮容器、27一个本轮容器均Running=false/Pid0/OOMKilled=false；不进入或停止其他人的容器。会话结束清除本人sudo票据并关闭两个SSH控制连接。是否已退出以会话登记为准。

## 11-04发布与公共入口安全补证

- 发布：功能仓40864f9，发布构建/独立重建在Project/outputs/framework-package/20261104-dev3-publish和20261104-dev3-rebuild；锁见package.lock.json。材料695cc32及说明补正476d274，公共STATUS ea25e9f。本轮未改功能包运行时代码。
- 安全测试本地入口：Project/outputs/framework-runs/20261104-public-safety，prepare.py冻结输入，pack.py生成无macOS元数据的tar，run.sh对应远端run-portable.sh，collect.py取证并逐字段验证身份差异。初版input.tar.gz与已中止首轮保留，后续使用input-portable.tar.gz/input-clean。
- 11机目录：/home/cgu135/framework-public-safety-20261104。输入input-clean共248文件SHA已核；场景lr-native-02、lr-strict-02、missing-strict-02、revision-strict-02；物理14/15、官方镜像。每次新输出目录；不得把此次空卡视为预留。
- 安全检查使用正式发布wheel1df42207…，模型及审核沿用上节；LR变更仅optimizer.lr；版本变化只在运行时副本增加未导入的adapter_safety_revision.py。运行时/模型原件未改，无新通过审核。
- evidence.tar.gz SHA 7b599d5036f30d32755ac4078a2a77c0ba06d0ff5fa3f15417575f4acdc090b2，原始轻量证据在本地evidence/；仓内docs/evidence-public-safety-20261104有68份SHA绑定文件。案例及报告本地未提交/推送。
- 本轮5个本人容器（含主动中止的lr-native-01）已Running=false/Pid0/OOMfalse；sudo票据清除、交互SSH98335退出、/tmp/cgu135-dev3-safety-1104.sock控制连接关闭。无后台任务，不停止/删除他人容器。
- 之前约27GB原生张量仍在上一节目录，本轮不新增大张量，不删除旧资料。

## 11-04训练回调候选复现

- 功能仓本地dev4：callback.py、training.py共享生命周期、18项新控制、CPU demo和接口文档；基准main/40864f9，AGENTS既有本地修改保留。未提交/推送；package.lock仍dev3。
- 收尾构建在Project/outputs/framework-package/20261104-callback-dev4-02，命令为`python tools/build_local.py --output <该绝对目录>`，退出0，128项安装包控制和三路CPU示例通过。local.baseline.json含45份快照SHA、完整命令/环境、wheel SHA；运行时及快照文件与当前工作树核验一致。-01是说明补全前构建，保留原件。
- 无卡示例在功能仓demo/training_callback_cpu.py。安装包调用参数看功能仓docs/公共训练回调接口.md；协调仓[本轮报告](docs/公共训练回调接口对接-20261104.md)及evidence-training-callback-20261104保存精简证据。
- 本轮无上机/后台进程，不改公共训练脚本；旧安全材料14项case控制与68份证据SHA再次通过。后续联调先按执行计划确认三项接口问题，再重新核服务器资源。

## 11-04 PPU连通性检查

用户要求先检查PPU环境。本次仅只读核查已有公共快照及网络：`nc -vz -G 5 192.168.5.112 22`超时；`nc -vz -G 5 ssh-yichang.puhui.chengfengerlai.com 36444`返回Connection refused。两命令都已结束；未进入SSH认证、未占卡、未创建容器或修改远端。不能据此判断机器坏了或账号无权限，需用户确认PPU专用VPN已连及当前有效入口。

公共快照ea25e9f中推荐thead-ppu-runtime/v2，镜像1ba7ead15378，运行配置在v2/env/v1/docker-compose.yml与env.baseline.yaml；设备方向11-03已有全量验收记录。均是参考记录，登录后重新检查。9月旧PPU实测及旧容器路径只作历史，不直接复用环境资格。

## 11-06 cann-shmem修复验证前置检查

已完成，结论与原始判据见[11-06报告](docs/cann-shmem修复验证-20261106.md)。用户确认原认证方式后11机SSH及sudo Docker成功。早期BatchMode失败不表示密码改变；旧会话84888已结束。

- 远端：11机`/home/cgu135/framework-shmem-20261106`；公共工具固定0da9977086d07a867bf915b207f08e70f541683c。`public/`为工具/镜像快照，`input/`为冻结脚本；结果`results/`，启动基线`image.baseline.yaml`与`demo/`。
- 本地：[20261106-cann-shmem](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261106-cann-shmem/)。prepare.py冻结47份输入；collect.py校验归档、初始脚本及补充脚本身份，核验60份结果/输入/基线文件。原始日志留本地evidence/results及远端，仓内只存摘要和截选。
- 固定镜像e13a15d0…、物理14→逻辑0。补包仅cann-shmem==1.6.0，wheel SHA `308a1c5a3dfe039d77d1159e27f3b1c16f38f4990a19c1fc68c7c052dee2b3f5`。1.7.0校验失败证据原样保留，无跳过校验。基线核对不检查pip叠加层，须同时核runtime-overlay.json及前后包清单。
- 命令入口：`sudo env DOCKER_CONFIG=/home/cgu135/framework-v3-20261016/config PYTHONPATH=/home/cgu135/framework-v3-20261016/tools PATH=/home/cgu135/framework-public-entry-20261104/tools:/usr/local/bin:/usr/bin:/bin python3 /home/cgu135/framework-shmem-20261106/input/run_v2.py --root /home/cgu135/framework-shmem-20261106 --stage <阶段> --version 1.6.0`。这些tools是本轮核验的既有本人工具，不是新机通用路径。
- start/before及首次默认plan用input/run.py初版；显式1.6.0的plan/install/after/stop用run_v2.py。补充index_check.py独立核官方哈希；provenance_v2.py记录真实包变化，初版provenance.py的发行名查询错误保留。当前仓内run.py/provenance.py分别对应远端v2字节。
- 最小用例前后各冷缓存；直接SiLU FP32三形状前反向通过。没有运行适配启用/拒绝或模型回归：dev3的原镜像已知失败拦截保持，后续先定义补包身份，不能复制旧审核放行。
- 归档evidence.tar.gz已取回，SHA `b82421ca041239c2df8ccb46985d1057bed88ac2da376e16bbd047486dd67d92`。容器`cgu135-cann-shmem-1106`已Stopped/Pid0/OOMfalse，保留未删除；sudo票据清除，SSH19406及控制socket `/tmp/cgu135-shmem-1106-11.sock`均关闭，无后台测试。结束时单卡SMI查询语法报错，释放以Docker停止状态为据，未补称SMI空卡复核成功。

后续操作必须另建运行目录、重查镜像/占卡。不要覆盖本轮日志或把11-06空卡当预约；本轮未改共享镜像、功能包或package.lock，未提交推送。

## 11-06补包环境审核与模型三路回归

已完成；[报告](docs/训练v3补包审核与Qwen回归-20261106.md)、[摘要](docs/evidence-overlay-regression-20261106/summary.json)。本地dev4增量是补包测量及精确环境匹配；已有回调修改保留，package.lock不变，无Git发布。

- 远端：npu1-11 `/mnt/raid/cgu135/framework-overlay-20261106`，本人新目录。物理14/15→逻辑0/1，训练v3精确镜像e13a15d0…；只读模型沿用`/home/cgu135/framework-v3-20261016/model/model-verified-20261014`。未动共享镜像/他人容器。
- 本地入口：[20261106-overlay-regression](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261106-overlay-regression/)。prepare.py冻结90个输入，review_operator.py/review_model.py打包经核验的限定审核；collect.py取证，summarize.py校验结果。input/是实测冻结脚本，evidence/保留完整轻量日志、reviews、实际baseline/demo。
- 公共runtime固定719de8c，基线工具0da9977。当前case的regression.py/qualify_overlay.py与冻结输入对应。命令为`sudo env DOCKER_CONFIG=/home/cgu135/framework-v3-20261016/config PYTHONPATH=/home/cgu135/framework-v3-20261016/tools PATH=/home/cgu135/framework-public-entry-20261104/tools:/usr/local/bin:/usr/bin:/bin python3 <root>/input/regression.py --root <root> --stage <阶段>`；init另传--devices和--model。工具目录只适用于已核验现场。
- 阶段：init→operator→人工算子审核→controls→native→repeat→candidate→人工模型审核→off→on→unqualified→stop。模型配置FP32 batch1 seq32 LR1e-5，两rank各3步。资格使用独立探针，正常三路使用已安装train_cli；缺条件场景移除模型审核、保留算子审核。
- 原生张量3726个、28,603,256,418字节保留远端`results/native/train/tensors`；本地仅取SHA清单`evidence/native-tensors.manifest.json`，没有下载这些大张量。旧11-04约27GB参考仍在原目录，未删除。
- 归档`evidence.tar.gz` SHA `8a84e23caf14d46be34c82463af04246156f5f261ff579aa63a013e40e98f311`；339个轻量证据逐文件SHA通过，90个原输入SHA一致。首次controls缺审核文件失败保留`controls-missing-review-01`，后续仅修正本人reviews目录归属重传，成功结果独立保存。
- 实测构建Project/outputs/framework-package/20261106-overlay-dev4-01，wheel `3ef0c14cad166071218d71f9201659e900f4f96ee08c34a4f8e30368a88bf46a`；文档补全重建-02，wheel `bca92d1982089b686963136b8b6eb847785354f231d31c94ba07402fa06be49a`。两次各142项CPU测试、19命令通过；13运行时文件与当前源码一致，-02的48快照SHA一致。
- 补包cann-shmem 1.6.0及其wheel SHA见[实际叠加身份](cases/cann-shmem-validation/demo/20261106-overlay/runtime-overlay.json)。正常启动器核实际安装内容，不仅信任镜像ID或版本字符串。已测审核在本轮reviews；不要跨环境复制后直接放行。
- 容器`cgu135-overlay-framework-overlay-20261106`已Running=false/Pid0/OOMfalse；143来自主动stop，全部测试阶段退出0。结束SMI显示NPU7两芯片无运行进程；sudo票据清除，SSH6531与`/tmp/cgu135-overlay-1106-11.sock`已关闭。归档进程19301完成，无后台测试。

## 退出检查

记录命令、退出码、结果/源码/环境身份，区分CPU控制、硬件通过、预期阻断和未运行。SSH退出与容器停止分别确认；使用中的主机、容器、PID、停止方式留在会话登记，凭证不写。
