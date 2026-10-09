# 交接文档:v3 round 4 收尾后续任务(npu1-11,2026-10-09)

> 供新会话接续执行。前置阅读:本文全文 + `docs/v3-round4-真机验证结果与移交指导-20261009.md`(背景与结论的完整版)。按任务编号顺序执行,任务间依赖已标注。

## 0. 当前状态快照(2026-10-09 会话结束时)

| 项 | 状态 |
|---|---|
| 仓 | `~/runtime-team`,分支 xliu969/dev,HEAD `8bdf1e8`,18 个文件已改未提交(14 改 4 增,含本文件,见任务 1) |
| 镜像(本机 docker) | 基座 `flagrt/ascend-operator-runtime:2.0.0-…-arm64`(62bcaae3f839);完整镜像 `…-comm:…-flagcx0.13.0g4e0e0cb-arm64`(e13a15d0d02b,基座超集) |
| 离线归档 | `/mnt/raid/user_cache/xliu969/v3-round4-archive/`:两条 tar.gz(各 10.5G)+ SHA256SUMS.txt |
| 真机验证 | 已完成:推理 9/9 PASS(需运行时三配置);训练手动 DDP loss 2.8911→2.5528;原生 DDP 仍阻 FlagCX c10d 层。详见移交指导文档 |
| 容器 | 本任务容器已清理,机器当前无本任务占用 |

两条外部缺陷(修复责任不在本组,任务 5/6 下发):shmem 缺失(Triton tle 顶层无条件 import,任何含 for 循环 kernel 编译失败);FlagCX c10d 异步返回(卡 DDP)。需求文本已写好:决策备忘录附录 B(已重写)、B-2(新增)。

## 1. 任务清单

### 任务 1:审查并提交仓内 17 个文件改动

内容:全部改动已就位,按 git diff 逐文件审查后 commit(可拆 2-3 个逻辑提交:验证结果回填 / 归档与 provenance / 移交文档)。红线:不 push 到 origin,commit 后停下等用户确认——或用户明确说"直接推"再推。

改动清单(核对用):
- 修改:两个 v3 REBUILD.md(各加「ROUND 4 真机验证」节)、两个 v3 lock.yaml(repro_result/known_issues/image_id/built_at)、PROCESS.md、TODO.md、image_list.md(3 处 v3 行)、HANDOFF-v3-round4-npu11.md(第 7 节)、决策备忘录(附录 A 回填 + 附录 B 重写 + B-2 新增)、docker-compose.flaggems-cos-off.yml(黑名单 cos→cos,sin + 注释重写)、两个 v3 ARCHIVE.md(骨架→实际归档清单)、pins.routeA.yaml(digest 换本机 e13a15d0d02b)
- 新增:`assets/shmem_stub.py`(shmem 占位模块,已验证)、两份 pipfreeze provenance(`pipfreeze-*-round4-npu11.txt`)、`docs/v3-round4-真机验证结果与移交指导-20261009.md`、本文档(HANDOFF-v3-round4-后续任务交接-20261009.md,即本文件,共 18 个)

### 任务 2:更新任务分发主文档(在任务 1 的 commit 之前完成)

对 `docs/v3-round4-真机验证结果与移交指导-20261009.md` 做三处修订:

a. **第二节改为单镜像发布口径**:内部保留分层构建(基座+通信层,薄层重建与缓存复用的工程价值),对外发布物只有完整镜像(train-comm tag,基座超集,推理训练皆可);基座镜像照常构建并归档,但不作为发布物。发布 README 首句注明"完整镜像=推理+训练超集"。注意:不改 tag 名("comm"字样保留,改名牵连 pins/compose 不值得)。

b. **第七节 B 项补交付标准**:FlagCX c10d 修复到货后需补一个原生 DDP 验证脚本(现有 train_qwen_1_5b_npu_syncpatch.py 是手动写法版,不能验证 DDP 封装本身)。

c. 通读一遍,按既定写作口径收紧措辞(该文档由上一个会话产出,个别表述仍偏口语,如"踩坑/治不了/炸"一类,替换为规范表述;结构与事实不动)。

### 任务 3:发布 V3 镜像(用户确认后执行)

1. 目标 registry:harbor.baai.ac.cn(机器已登录,基座即从此拉取;如推送需项目权限,先 `docker login` 状态核查,无权限则停下问用户要权限或改推内部其他 registry)。
2. 只推完整镜像:`docker tag flagrt/ascend-operator-runtime-comm:2.0.0-…-arm64 harbor.baai.ac.cn/<项目名>/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64` 后 push(项目名问用户或查仓内其他镜像推送先例)。
3. 推送后在 image_list.md 补 registry 地址行,并把 raid tar.gz 的存在性重新核验一遍(发布物与归档物一致性)。
4. 不推基座镜像(单镜像发布口径,见任务 2a)。

### 任务 4:固化 case 目录标准并补齐 v3 目录

1. 新建 `dev/images/CASE-TEMPLATE.md`:目录构成(README 三段式/docker-compose*.yml/pins.*.yaml/assets 脚本三分类:复现、验证、收数)、四条硬规则(单入口命令;外部依赖必须 pins 声明;必须带自判定 PASS/FAIL 的验证脚本;日志不进仓)。模板从现有 v3 目录抽象,不发明新概念。
2. 对照模板补 v3 两个目录的缺口:大概率缺 README.md(三段式)——operator-runtime/v3 与 ascend-train-comm/v3 各补一份;已有文件不动。
3. 该标准同时写入 `dev/images/README.md` 一小节(目录标准指向 CASE-TEMPLATE.md)。

### 任务 5:通过 issue/case 包下发两个修复任务

前提:任务 1 已 commit(对方要能拿到仓)。两个 case 包均已具备自判定脚本,按下发对象打包:

a. **算子编译方向(shmem 根治,优先级高)**:case 目录 `dev/images/ascend-operator-runtime/v3/`(复现:容器内裸跑 `python3 dev/images/ascend-operator-runtime/v3/assets/v3_step5_validate.py` 即在模型加载处复现 CompilationError)。需求正文:决策备忘录附录 B(已按结构性严重度重写,两种修法:tle 改惰性加载=根治;aclshmem 官方化入镜像)。完整报错栈引用 ~/tmp-reproV3-r4/infer-leg/control-no-blacklist.log 与 round-2.log——注意这两个日志在本机,若通过仓内 issue 下发需先把关键日志段落拷入 case 目录或 lock.yaml 引用的稳定位置(建议:在 v3 目录新增 `assets/evidence/` 存放两个日志的截选,REBUILD.md 同步改引用)。
b. **FlagCX 通信方向(c10d 同步语义,卡 DDP)**:case 目录 `dev/images/ascend-train-comm/v3/`(复现:容器内 `torchrun --nproc_per_node=2 assets/flagcx_sync_test.py`,四组正反结果自判定)。需求正文:决策备忘录附录 B-2。日志同理处理(probe_agit.log 关键段入 evidence/)。
c. 下发渠道按用户指定(组内 issue 系统/邮件/群),正文=一句问题定义+case 目录路径+一行复现命令+需求文档路径,不贴长段环境描述。
d. 同步现状(非任务、只通知):上游 PR 已具备提交条件(附录 A+复验回填),给用户提醒自行提交或授权执行。

### 任务 6:收尾核查(全部任务后)

1. `git status` 干净(或只剩有意保留的未跟踪文件)、commit 在本地分支;
2. raid 归档 SHA256SUMS.txt 与实际文件一致;
3. image_list.md / lock.yaml / pins 三处的镜像 id 与 digest 一致(e13a15d0d02b 全串);
4. 移交指导文档、CASE-TEMPLATE、两份 README 相互引用的路径全部有效;
5. 向用户汇报:各任务结果 + 未决事项(需用户拍板的:push 时机、registry 项目名、issue 渠道)。

## 2. 执行红线(继承自本任务线)

- 仓内改动不 push origin,除非用户明确指示;
- 不改 `dev/stack.lock.910c.*.yaml`(v3 仍是候选,非生效);
- 共享机器礼貌:起任何带卡容器前 `docker ps` 核对并发(上限 3);本机已确认存在昇腾驱动的容器级独占约束——同机已有带卡容器时第二个必 EBUSY,多任务验证须共用容器 + `ASCEND_RT_VISIBLE_DEVICES` 分区;
- 镜像内容不做任何构建后修改(stub 只运行时注入,不进镜像);
- 推理运行时三配置(`VLLM_FL_FLAGOS_BLACKLIST=cos,sin` + PYTHONPATH 注入 stub + `VLLM_ENABLE_V1_MULTIPROCESSING=0`)是当前解堵前提,改动前必须有新的对照实验。

## 3. 关键文件与路径索引

| 内容 | 路径 |
|---|---|
| 任务分发主文档(需按任务 2 修订) | docs/v3-round4-真机验证结果与移交指导-20261009.md |
| 决策备忘录(附录 A=PR 草稿,B=shmem 需求,B-2=FlagCX 需求) | docs/v3-决策备忘录与shmem跨组需求-20261008.md |
| 推理验证证据(本机) | ~/tmp-reproV3-r4/infer-leg/(round-1..5.log、control-*.log、shmem_stub.py) |
| 训练验证证据(本机) | ~/tmp-reproV3-r4/train-leg/outputs/(flagcx_sync_test.log、train_lr1e5.log、probe_agit.log) |
| 镜像归档 | /mnt/raid/user_cache/xliu969/v3-round4-archive/(tar.gz×2+SHA256SUMS.txt) |
| pins(已改 digest,commit 后生效) | dev/images/ascend-operator-runtime/v3/pins.routeA.yaml |
| 构建上下文(可复用,勿删) | ~/tmp-reproV3-r4/ 与 ~/tmp-reproV3-comm-r4/ |
| 本任务线上一份交接文档(历史) | dev/images/HANDOFF-v3-round4-npu11.md |

## 附录:写作与交付口径(执行所有产出文档时遵守)

以下口径适用于本交接清单产出的全部文档(任务分发主文档、CASE-TEMPLATE、README、issue 正文、commit message)。产出文档正文只包含内容本身,口径本身不写入产出文档。

### A1 文风

- 专业、直接、克制。不使用口语化表述(如"踩坑""治不了""炸了""东西"一类),对应使用"问题定位""无法通过更换版本解决""失败""组件"。
- 结构固定为:背景与核心问题 → 结论/现状 → 证据(路径) → 交给谁/下一步。先给结论,过程细节只在对方需要复现时给。
- 事实导向:每个判断附证据路径;未验证的推断标注"待验证"或不写。禁止夸大严重度或弱化已知缺陷。
- 中文为主;命令、路径、字段名、镜像 tag 保持原文,不翻译不改写。
- 不在文档中记录写作过程、讨论过程、格式要求本身;文档之间用相对路径互引,不复制粘贴内容造成双份维护。

### A2 case 复现描述

- 复现步骤写成可直接执行的命令序列,单入口(一条命令或一个脚本),不写"先手动做若干准备再运行"。
- 前置条件显式列出(镜像 tag、卡数、所需目录),每项给获取方式。
- 验证脚本必须自判定结果(PASS/FAIL 退出码与输出),执行者无需人工判读;预期输出写明关键判据(如"all_gather+sync 两 rank 均 PASS")。
- 报错证据引用稳定路径:入仓的 evidence/ 目录优先;仅存于本机的日志注明机器名与绝对路径。

### A3 版本与发布口径

- pins 类文件:commit 为唯一机器校验目标;branch 仅作人工注记(strict 条目附 branch 注释但不校验;loose 条目维护 verified_commit 作为上次验证锚点,漂移提示不阻断)。
- 镜像发布:对外仅发布统一完整镜像(train-comm tag,基座超集);基座镜像内部构建并归档,不作为发布物;对外 README 首句说明镜像能力范围。

### A4 修改他人/历史文档时

- 保留原文事实与结论,只更新状态与口径;删除过时内容时,若为历史记录(如 round 1-3 的 STOP CONDITION),改为标注"已被后续轮次取代"而非删除。
- 文档间冲突时以 lock.yaml/REBUILD.md 为技术口径权威,摘要文档跟随修正。
