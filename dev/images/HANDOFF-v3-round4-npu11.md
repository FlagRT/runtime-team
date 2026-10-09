# 迁移交接文件（npu1-27 → npu1-11，2026-10-08）

> 供在 npu1-11 上新开的 Hermes session 使用。这份文件 + 仓内文档链 =
> 完整任务上下文。按「新机器首跑 prompt」一段启动即可。

## 1. 任务是什么

FlagOS runtime-team v3 候选血统的 **round 4 收尾**：两项已拍板决策的落地物
已构建完成（npu1-27），剩**真机双腿验证**未执行——迁移到本机（npu1-11）执行。

背景链（如需深读，按序）：
1. `dev/images/ascend-operator-runtime/v3/REBUILD.md`「ROUND 4」章节
2. `dev/images/PROCESS.md`「round 4」
3. `docs/v3-决策备忘录与shmem跨组需求-20261008.md`（决策依据 + 上游 PR 草稿 + shmem 跨组需求）
4. `docs/运行时基座建设-黄金准则与方案存档.v1.md`（总组设计总纲，理解"为什么"用）

## 2. 决策内容（已拍板，勿重新讨论）

- **决策①**：FlagCX sync 修复以**声明式覆盖层补丁**方式吸收（不切私有 fork）——
  上游 vllm-plugin-FL pin `8b059122e` + `assets/patches/vllm-plugin-FL/0001-*.patch`
  （sha256 `b97d0d8b…e23b5f928`）。上游合入后删除补丁层。upstream-first PR 草稿
  已写好（决策备忘录附录 A），待提交。
- **决策②**：shmem 缺口不装来路不明的包、不写 stub；用插件官方开关
  `VLLM_FL_FLAGOS_BLACKLIST=cos` 把 torch.cos 拉回 torch_npu 原生（opt-in compose
  层 `docker-compose.flaggems-cos-off.yml`）。根因（FlagTree tle/dsa/ascend 扩展
  依赖声明缺口）已写成跨组需求（决策备忘录附录 B），待发算子编译组。

## 3. 已完成（npu1-27 侧，全部有据可查）

- 镜像构建：operator-runtime v3 `be30a952c2eb`、train-comm v3 `7028028bb62c`
  （含补丁，三重证据：static 自检 / `vllm_fl 0.0.0+g8b059122e.d20261008` 脏标记 /
  communicator.py SYNC-FIX ×2）。npu1-27 侧证明见两 REBUILD.md「ROUND 4」。
- 文档链回填完毕并已入库：dev-1.0@31c741d 含 e8d62b9（round 4 决策落地 13 文件）
  与 1fdfaf8（round 4 构建事实回填 6 文件）。
- pins.routeA.yaml digest 已更新（`sha256:7028028bb62c…af7913`）。

## 4. 本机（npu1-11）已就位的资源

| 资源 | 位置 | 状态（2026-10-08 深夜） |
|---|---|---|
| 仓 | `~/runtime-team` | 🔄 首跑步骤 0 会切到 xliu969/dev（含 1fdfaf8），见下 |
| compose 插件 | `~/.docker/cli-plugins/docker-compose` v2.29.7 | ✅ |
| 基座镜像 | `harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-…:202608-torch2.10.0-vllm0.20.2`（aa697a359613，19.2GB） | ✅ 已拉 |
| 模型 | `~/models/Qwen3-Embedding-0.6B`（1.2G，含 model.safetensors） | ✅ |
| 模型 | `~/models/Qwen2.5-1.5B`（2.9G） | ✅ |
| v3 镜像 | 待本机重建（见首跑步骤 1） | ⬜ |

注意：**npu1-27 的 docker save 已损坏**（numpy libopenblas 层校验和不匹配，
save 任何派生镜像必失败；运行不受影响）。所以镜像不能搬、只能本机重建——
基座已拉好，重建走缓存会很快。

## 5. 新机器首跑 prompt（直接粘贴给新 Hermes session）

```
我在续接一个跨机器迁移过来的任务（v3 血统 round 4 真机验证），本机 npu1-11。
先读 ~/runtime-team/dev/images/HANDOFF-v3-round4-npu11.md 全文，再动手。

红线：不改本机系统配置；仓内不 git commit/push（改动留给用户审查）；
不碰 dev/stack.lock.910c.*.yaml；共享机器礼貌（本机当前零占用，起容器
前仍 docker ps 确认一下）。

按顺序执行：

0. 状态核对与分支切换：
   cd ~/runtime-team
   git status（若有未提交改动，多半是迁移中间态：git checkout -- . &&
   git clean -fd dev docs 清掉，正式内容已全部入库）；删掉本地 tmp-round4
   分支（迁移中间态）：git branch -D tmp-round4
   git fetch origin
   git checkout -B xliu969/dev origin/xliu969/dev
   git log --oneline -2   # 应看到 1fdfaf8（round-4 构建事实回填）及其之前
   du -sh ~/models/*      # 两个模型 ≈1.2G + 2.9G
   docker images | grep flagtree-ascend3.5   # 基座 19.2GB 在位

1. 重建 v3 两条镜像（不占卡；基座已在本地，走缓存很快）：
   参考 dev/images/ascend-operator-runtime/v3/build.sh 的上下文组装法——
   git archive 导出本仓 FlagGems@f7ae8e6b 与 PyTorch-Plugin-FL@162582d6
   （子库 pin commit，见 lock.yaml；本机子库若无对象则从 origin fetch）。
   注意：本机仓刚从 dev-1.0 切到 xliu969/dev，工作分支已就绪，直接做。
   构建预期产物 image id 与 npu1-27 不必一致（环境差异），但
   verify_runtime.py --static 必须过、镜像内 vllm_fl 版本串必须带 .d 日期
   脏标记、communicator.py 含 SYNC-FIX——三者齐 = 补丁层正确进镜像。
   然后级联重建 train-comm（其 build.sh，父镜像 tag 指向新 operator-runtime）。

2. 推理腿验证（1 卡即可）：
   dev/lib/up.sh dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml \
                 dev/images/ascend-operator-runtime/v3/docker-compose.flaggems-cos-off.yml
   （up.sh 会先跑 pins 校验——注意 pins.routeA.yaml 的 digest 是 npu1-27 的
   镜像 id 7028028bb62c…，本机重建后 id 会不同：这是预期内的，把 digest
   改成本机 docker image inspect 的实际值即可，属环境适配不是违规。）
   容器内跑 SMOKE_MODEL_PATH 指向本机模型（compose 挂了
   ${WORKSPACE_ROOT}:/workspace，把模型软链到 ~/runtime-team/models/ 下
   或用绝对路径自行挂载，自行判断），执行
   python3 dev/images/ascend-operator-runtime/v3/assets/v3_step5_validate.py
   预期：coexistence PASS + 真实 .encode() 输出（round 3 卡死的 shmem 缺口
   应被 BLACKLIST=cos 绕开）。对照组：不叠 cos-off 层复现原 STOP CONDITION
   （证明开关就是解堵点）——对照组失败才是"证明修好"的完整证据。

3. 训练腿验证（2 卡）：
   ascend-train-comm/v3 + 2 卡 torchrun 跑 assets/train_qwen_1_5b_npu.py
   （模型 Qwen2.5-1.5B）。目标：真实 loss 下降曲线/吞吐数字——补丁的直接
   受益验证（round 3 证明 DDP 参数校验必须落在 c10d/communicator 层修复）。
   同时跑 flagcx_sync_test.py 作为快速前置自检（应全 PASS）。

4. 结果回填：两个 lock.yaml changelog、两个 REBUILD.md、PROCESS.md round 4
   真机验证小节、TODO.md；镜像 docker save 到 /mnt/raid（本机可写区域，
   先 mkdir -p 用户目录）做离线归档（先例 v2/ARCHIVE.md）；repro_status
   双腿全 PASS 才升 🟢。

5. 完成后向用户汇报，并提醒三项移交件：上游 PR 提交（决策备忘录附录 A）、
   跨组需求发出（附录 B）、npu1-27 的 docker 存储损坏需报修（层校验和，
   save 不可用）。
```

## 6. 容易踩的坑（前机实测）

- **无卡容器 import 验证必失败**（torch_npu autoload 要宿主驱动库）：用文件级
  grep 验证镜像内容，别用 inspect.getsource。
- **/mnt/raid/models 权限**：raid 根 root-only，模型放 `~/models`（已做）。
- **v3_step5_validate.py 的模型路径**：默认 `/mnt/raid/hliu553/models/…`（npu1-27
  路径），本机用 `SMOKE_MODEL_PATH` 环境变量覆盖。
- compose `${WORKSPACE_ROOT}` 由 up.sh 设置，绕开 up.sh 需手动 export。

## 7. 验证结果（2026-10-09，npu1-11，round 4 收尾完成）

双腿验证执行完毕，全部产物入库待审查（未 commit）。一屏摘要：

- **镜像重建（本机）**：operator-runtime `62bcaae3f839` / train-comm
  `e13a15d0d02b`（与 npu1-27 同源，三重证据齐：static 自检 / vllm_fl
  `.d20261008` 脏标记 / SYNC-FIX ×2）。归档
  `/mnt/raid/user_cache/xliu969/v3-round4-archive/`（tar.gz×2 + SHA256SUMS.txt）。
- **推理腿 PASS 9/9**：真实 `.encode()` 输出（dim=1024，norm≈1.0，
  identical=false，无 NaN/全零，显存 16.8GB）。生效配置：
  `VLLM_FL_FLAGOS_BLACKLIST=cos,sin` + shmem loud-fail stub（PYTHONPATH 注入，
  已入仓 `assets/shmem_stub.py`）+ `VLLM_ENABLE_V1_MULTIPROCESSING=0`。
  对照组（空黑名单）如期复现 freqs.cos() shmem 崩溃——证据链闭合。
- **关键机制发现（改变定级）**：shmem 缺口是结构性的（tle 顶层无条件
  import + 任何 for 循环触发 + vllm 核心 kernel 也命中），黑名单只能管
  FlagGems 算子；跨组需求需按此重写（原附录 B 低估）。
- **训练腿**：flagcx_sync_test 证实 sync-fix 假设（all_gather nosync FAIL /
  sync PASS）；手动 DDP 端到端 loss 2.8911→2.5528（24 步，LR 1e-5）；
  原生 DDP 仍阻塞 flagcx c10d 层（all_gather_into_tensor 异步返回，
  vllm_fl 侧补丁覆盖不到——跨组需求第二条）。
- **repro_status 双 🟡 维持**：升 🟢 条件 = 算子编译组根治 shmem（推理）+
  FlagCX c10d sync 修复（训练）。
- **移交件（待用户执行）**：① 上游 PR 提交（决策备忘录附录 A 草稿就绪）；
  ② 跨组需求升级重发（shmem 结构性 + c10d sync 两条，证据链在
  ~/tmp-reproV3-r4/）；③ npu1-27 docker 存储损坏报修。
- 环境注意：Ascend 驱动容器级 davinci 独占（第二个带卡容器 EBUSY），
  多腿验证须单容器 ASCEND_RT_VISIBLE_DEVICES 分区。
