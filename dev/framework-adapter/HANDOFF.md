# framework-adapter 复现与运行交接

更新：2026-11-04。先看[当前状态](STATUS.md)；本页只列操作路径、环境身份与恢复注意事项。
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

## 退出检查

记录命令、退出码、结果/源码/环境身份，区分CPU控制、硬件通过、预期阻断和未运行。SSH退出与容器停止分别确认；使用中的主机、容器、PID、停止方式留在会话登记，凭证不写。
