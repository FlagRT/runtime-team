# 重建 ascend-train-comm v3

> **阶段声明（PHASE 2 ROUND 2 已执行，2026-09-22）**：round 1 的父镜像是
> `ascend-operator-runtime/v3` 的 `-DIAGNOSTIC-novllm` 诊断变体（该轮设计 tag
> 因 vllm-ascend 层构建失败未产出实体）。**round 2：父镜像换成了真实构建成功的
> 正式 v3 tag**（`ascend-operator-runtime/v3` 把 vLLM 插件从 vllm-ascend 换成
> vllm-plugin-FL 后构建通过，image id `9ad551058f2f`，详见该目录 REBUILD.md）。
> 本层（FlagCX）自身的构建方式、commit、逻辑**完全未变**——round 2 只是在新
> 父镜像上重新构建了一遍（父镜像变了，必须重建；FlagCX 层内容本身与 vLLM 插件
> 选型无关）。**FlagCX 的 `dist.broadcast`/`dist.all_gather` STOP CONDITION 不在
> 本轮任务范围内，未受影响、未被重新调查，结论保持不变**（见下方「PHASE 2
> STOP CONDITION：FlagCX broadcast/all_gather 返回错误结果」，仍然是真实的、
> 未解决的独立阻塞项）。本轮新增的是：round 1 因 vllm-ascend 缺失而只能"部分
> 验证"的训练/推理同镜像共存检查，round 2 在这个新镜像（同时含 FlagCX 与
> vllm-plugin-FL）上做了完整检查，结果见下方「ROUND 2 共存检查」。

在 `ascend-operator-runtime:v3` 之上叠加 FlagCX，训练腿通信从"flagos 适配"切到
"HCCL 适配"。

## 方式（phase 2 执行）

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`。父层
`flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`
需先建（见 `../../ascend-operator-runtime/v3/REBUILD.md`）。需要构建期网络
（`docker build --network=host`）——与 v2 相同，FlagCX 的两个 git submodule
（`third-party/json`、`third-party/googletest`）构建期直接 clone。

## 构建上下文需备齐

| 项 | 来源 |
|---|---|
| `Dockerfile` | = 本目录 `Dockerfile.repro`（build.sh 会拷） |
| `src/FlagCX/` | `git archive` 自 `FlagCX @ 4e0e0cbcbf721169ca82348080f8353aebfe2c31`（与 v2 相同 commit，未升级） |
| `assets/verify_flagcx_runtime.py` / `verify_flagcx_p2p.py` | 本目录 `assets/`（v3 版本，见文件头注释——Route A / HCCL-adaptor 路径） |

## 与 v2 的区别

### FlagCX 本体：不变

commit `4e0e0cbcbf721169ca82348080f8353aebfe2c31`、`USE_ASCEND=1 pip install .
--no-build-isolation` 构建命令、submodule 拉取方式，逐字节未变。v2 已经解决了
"逐字节可复现"（公开 commit，无 owner 私有 patch）——v3 不重复解决这个问题，
只改消费侧。

### 消费侧：从 flagos 适配切到 HCCL 适配（任务书第 2 条核心改动）

**背景**：`train_qwen_1_5b_npu.py`（
`dev/device-context/910C/distributed_training/scripts/train_qwen_1_5b_npu.py`）
已经用 `import torch_npu` + `dist.init_process_group(backend="flagcx")`
（完全不涉及 torch_fl）跑通了 2481 步双卡 DDP 训练闭环（loss 1.9501,
4157 tok/s，见
`dev/device-context/910C/distributed_training/docs/flagcx_ascend_aline_validation_20260824.md`）。
这条路径被称为"HCCL adaptor"（相对于 v2 走 Torch-FL 的 "flagos" ProcessGroup
包装 FlagCX 那条路径而言）。

**ENV 改动**：

| ENV | v2 | v3 |
|---|---|---|
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0`（压制 torch_npu autoload） | 未设置（官方默认，autoload 生效） |
| `FLAGCX_TORCH_BACKEND` | `flagos` | 未设置（见下方「开放问题」） |
| `HCCL_WHITELIST_DISABLE` | `1` | `1`（不变） |

**verify 脚本改动**：`assets/verify_flagcx_runtime.py` / `verify_flagcx_p2p.py`
的 `--device` 路径从

```python
import torch_fl
device = f"flagos:{local_rank}"
dist.init_process_group("flagos", ...)
```

改为

```python
import torch_npu
device = f"npu:{local_rank}"
dist.init_process_group(backend="flagcx", ...)
```

直接对齐 `train_qwen_1_5b_npu.py`。

## 开放问题（任务书要求：无法从阅读确定的地方必须明说，不能悄悄猜）

### 1. FlagCX 是否存在独立于 `USE_ASCEND=1` 的"adaptor 选择"编译期 flag

本阶段只能从阅读 `ascend-operator-runtime/v2`、`ascend-train-comm/v2` 两份
`Dockerfile.repro`、`train_qwen_1_5b_npu.py`、以及
`dev/device-context/910C/distributed_training/` 下的既有验证文档得出判断：
"flagos 适配" vs "HCCL 适配"的差别在**消费侧**（谁先抢注 PrivateUse1、传给
`init_process_group` 的 backend 字符串），FlagCX 本身的构建命令
（`USE_ASCEND=1 pip install . --no-build-isolation`）两条路径完全相同——两个
Dockerfile 都只有这一个 `USE_ASCEND` 选项，没有发现第二个 adaptor 相关的构建期
环境变量或 setup.py 选项。

**但本仓库不 vendor FlagCX 源码**（构建期才 `git archive`/`clone`），本阶段
无法直接 grep 源码确认是否存在被文档遗漏的编译期开关。phase 2 建议在真机
`docker build` 拉到 FlagCX 源码后，进容器/构建中间层跑：

```bash
grep -rniE "adaptor|hccl|flagos|torch_npu" /opt/flagrt/src/FlagCX/setup.py \
  /opt/flagrt/src/FlagCX/CMakeLists.txt 2>/dev/null
```

确认是否存在遗漏的构建期开关；若发现有，需回来修正本文件与 `lock.yaml` 的
`flagcx_adaptor_terminology` 段落。

### 2. `FLAGCX_TORCH_BACKEND` 这个 ENV 是否被 FlagCX 包本身实际读取

v2 的 Dockerfile 设置了 `FLAGCX_TORCH_BACKEND=flagos`，但本组现有代码库里
（`dev/communication/`、`dev/device-context/`）没有找到任何 Python/C++ 代码
读取这个变量的证据——`verify_flagcx_p2p.py`（v2 版本）只是**断言**它等于
`"flagos"`，不代表 FlagCX 内部真的消费它。不确定它是"FlagCX 包自己读的开关"
还是"本组自定的文档标注惯例"。v3 索性不再设置它（缺失不该造成任何行为差异，
如果确实无人读取的话）；如果 phase 2 发现 FlagCX 内部真的读取这个变量做分支
（比如影响 backend 名字的自动发现逻辑），需要补回来并设成一个反映 HCCL 路径
的值。

### 3. `train_qwen_1_5b_npu.py` 不显式 `import flagcx` 但 `proto_train_leg.py`（昆仑芯路径）显式 import

见 `lock.yaml` 的 `flagcx_adaptor_terminology` 段落最后一节。两份代码对"是否
需要显式 `import flagcx` 才能让 `backend="flagcx"` / `"cuda:flagcx"` 被
`dist.init_process_group` 识别"做法不一致，本阶段无法判断是芯片差异
（910C vs 昆仑芯 P800）、安装方式差异（本组 `pip install .` vs
`setup_flagcx_plugin.sh` 的 `.pth` 注册法）、还是防御性写法差异。v3 的两份
verify 脚本保留了显式 `import flagcx`（无害），**这不是当作已验证结论**，
phase 2 应实测确认去掉这行是否仍然成功（如果成功，说明确实是自动发现；如果
失败，说明这行是必需的，需要在正式训练脚本里也补上）。

### 4. 退出期 SIGABRT 已知问题是否在 v3 默认路径下复现

见 `lock.yaml:known_issues`。v2 记录的退出期 `free(): invalid pointer`
SIGABRT 被隔离认定为"FlagCX 与 Torch-FL 清理顺序冲突"；v3 默认路径完全不
import torch_fl，理论上可能不复现，`train_qwen_1_5b_npu.py` 的既有验证记录
（不同分支：`kistich/ascend-dev1.0`）也显示"进程干净退出"，是支持性的间接
证据——但不能直接套用到本文件锁定的 FlagRT/FlagCX 公开 commit
`4e0e0cbcbf7...` 上。**必须在 phase 2 实际构建出的 v3 镜像上重新验证。**

## 静态自检（PHASE 2 ROUND 2 实测，2026-09-22）

| 判据 | 结果 |
|---|---|
| `docker build --network=host`（父镜像为正式 v3 tag，非诊断变体，见上方阶段声明） | **通过**，FlagCX 0.13.0 源码编译干净（`USE_ASCEND=1 pip install --no-build-isolation .`），两个公开 submodule（nlohmann/json、googletest）clone 正常，image id `43f3e2f70b4c` |
| FlagCX `.so` 存在 + 链接检查（`verify_flagcx_runtime.py --static`） | **通过**：`libascendcl.so`/`libhccl.so` 齐全，未链 `torch_npu`——与 v2/round 1 一致，FlagCX 本体未变，实测确认预期 |

镜像：`flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`
（无 `-DIAGNOSTIC` 后缀——父镜像是正式构建成功的 operator-runtime v3，本层内容
逐字节与 round 1 相同，只是父镜像换了）。pip freeze 落盘
`assets/provenance/pipfreeze-comm-v3.txt`（round 2 已替换 round 1 的诊断镜像
快照，文件头说明来源）。`pip freeze` 里同时可见 `flagcx @
file:///opt/flagrt/src/FlagCX` 与 `vllm-plugin-fl @ file:///root/vllm-plugin-FL`
——这是本镜像用于 Step 5 共存检查的直接证据：同一个镜像里训练侧（FlagCX）与
推理侧（vllm-plugin-FL）组件都真实存在。

## 真机动态验证

round 1（容器 `v3-validate-train-910c`，2 卡 `--device=/dev/davinci0,1`，
2026-09-22；验证完毕已 `docker rm -f`）——**FlagCX 层本身未变，round 2 不重复
这几条**：

| 判据 | 结果 |
|---|---|
| `verify_flagcx_runtime.py --device`（torch_npu + backend="flagcx"，真机 2 卡） | **通过**：两个 rank 均 `device_all_reduce_ok: true`，`device_backend_str: "flagcx"`，`torch_npu_default_check: "passed (PrivateUse1='npu' with no special import order)"`。退出期复现 v2 已知的 `free(): invalid pointer` SIGABRT（见下方，结果已在 teardown 前打印，不影响判定） |
| `verify_flagcx_p2p.py`（同上路径） | **通过**：两个 rank 全部 4 组 payload（1/4/257/65536 elements）sha256 校验一致，`status: "passed"`，`inner_backend: "name=flagcx,_get_backend_name=flagcx"` 确认走的是 FlagCX 自己注册的原生 c10d 后端而非 Torch-FL 的 flagos 包装层。退出期同样复现 SIGABRT（同上，不影响判定） |
| `dev/communication/probes/communication_correctness.py` | **未执行**——训练脚本本身（`train_qwen_1_5b_npu.py`）在 DDP 构造阶段就已失败（见下方 STOP CONDITION），已确认根因是 `flagcx` backend 的 `broadcast`/`all_gather`（而非 `all_reduce`/P2P）返回错误结果；既然已经用更直接的诊断脚本精确定位到问题，判断再跑一遍这个更大的探针脚本不会产出新信息，故未重复执行，优先把时间留给记录与报告 |
| 退出期 SIGABRT 是否复现（开放问题 4） | **复现**——v3 默认路径（不 import torch_fl）下 `verify_flagcx_runtime.py --device` 与 `verify_flagcx_p2p.py` 均在成功完成通信后于进程退出阶段报 `free(): invalid pointer` SIGABRT。这推翻了 lock.yaml 里"因为不装 torch_fl 所以可能不复现"的推测——根因显然不是"FlagCX 与 Torch-FL 的清理顺序冲突"（v3 默认路径完全不碰 torch_fl），必须是 FlagCX 自身的退出期清理逻辑问题，或 FlagCX 与 torch_npu（而非 torch_fl）的清理顺序冲突。两个 verify 脚本已经按设计在 teardown 前打印结果，所以这不影响上面两行"通过"的判定，但需要更新 known_issues 的根因假设 |
| 与 `train_qwen_1_5b_npu.py` 的 loss/吞吐对比 | **无法给出**——训练脚本本身跑不起来，见下方 STOP CONDITION |
| **训练脚本 `train_qwen_1_5b_npu.py`（2 卡 DDP，MAX_STEPS=400）** | **失败，STOP CONDITION**——见下方详述 |

以上均为 round 1 结果，FlagCX 层本身未变，round 2 不重复。

## ROUND 2 共存检查（任务书 Step 5.2：训练侧 FlagCX + 推理侧 vllm-plugin-FL 同镜像共存）——【PENDING】

与 `ascend-operator-runtime/v3/REBUILD.md`「ROUND 2 真机验证结果」同一个阻塞
原因：本轮会话观测期间机器上其他工程师的带卡容器持续保持在 5 个（超过
`dev/stack.lock.910c.v2.yaml` 规定的并发上限 3，且并非本任务所起），未能降到
留出名额的水平，按任务书"respect the shared-machine rules"的要求本轮未强行
起带卡容器。**共存检查本身未执行，明确记为 PENDING**，不是"跳过"或"假设会
通过"。

静态证据（已具备，不依赖真机）：`assets/provenance/pipfreeze-comm-v3.txt`
确认本镜像（`flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-
cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`，image id
`43f3e2f70b4c`）同时装有 `flagcx @ file:///opt/flagrt/src/FlagCX` 与
`vllm-plugin-fl @ file:///root/vllm-plugin-FL`——两个组件的 Python 包本身
不冲突（pip 依赖解析、安装过程均无报错）。但这只是"装在同一个环境里不打架"的
静态证据，**不能替代"同进程内先后 import torch_npu / flagcx / vllm_fl 后
PrivateUse1 后端与 triton driver 是否保持一致、有没有 v2 记录过的 "2 active
drivers" 类问题"这个动态问题**，这必须在真机容器里实际跑一遍才能确认，任务书
也明确要求"记录实际跑了什么、观察到了什么"而非"没看到报错"。

**待办**：机器带卡容器并发数降到 ≤2 后，在本镜像内单容器（1 卡即可）依次
`import torch_npu`（确认 PrivateUse1="npu"）→ `import flagcx`（确认可用，
不需要真跑 DDP，那是另一个独立 STOP CONDITION 范围）→ `import vllm` +
`import vllm_fl` + 触发 `vllm.platform_plugins` 的 `fl` 注册 → 确认
`torch._C._get_privateuse1_backend_name()` 全程保持 `"npu"`、triton 活跃
driver 无冲突，并记录真实观察到的输出（不是"无报错"这种弱证据）。这一步可以
与 `ascend-operator-runtime/v3` 的 vLLM 推理烟雾测试合并在同一个容器里做（本
镜像本身就同时含 FlagCX 与 vllm-plugin-FL，是任务书 Step 5 两项检查唯一都能满足
的镜像），减少额外起容器的次数。

## PHASE 2 STOP CONDITION：FlagCX `broadcast`/`all_gather` 返回错误结果（2026-09-22，round 1，未受本轮 vLLM 插件 pivot 影响，仍然成立）

**现象**：直接运行 `train_qwen_1_5b_npu.py`（`torchrun --nproc_per_node=2`，
`MAX_STEPS=400`，权重从 `/workspace/models/Qwen2.5-1.5B` 加载，2 张真机卡）在
模型加载完成、进入 `DDP(model, device_ids=[local_rank])` 构造时立即失败：

```
RuntimeError: DDP expects same model across all ranks, but Rank 1 has 338
params, while rank 0 has inconsistent 0 params.
```

两个 rank 都报同样的错（对方 0 params）——说明 PyTorch 内部
`_verify_param_shape_across_processes` 用来核对各 rank 参数形状一致性的通信
调用，在 `flagcx` backend 上没有正确传递数据。

**根因定位（追加诊断脚本，同一 2 卡真机容器）**：单独测试了 DDP 构造会用到
的几种 collective，结果：

| collective | 结果 |
|---|---|
| `dist.all_reduce`（SUM） | **正确**（`verify_flagcx_runtime.py --device` 已验证：结果与预期总和一致） |
| P2P `dist.send` / `dist.recv` | **正确**（`verify_flagcx_p2p.py` 已验证：4 组不同大小 payload 的 sha256 全部匹配） |
| `dist.broadcast(src=0)` | **错误**——实测 rank 0 发送 `[1,1,1,1]`，rank 1 广播后仍是自己原来的值 `[2,2,2,2]`，即**广播是静默 no-op，rank 1 完全没收到 rank 0 的数据** |
| `dist.all_gather`（tensor list 形式） | **错误**——两个 rank 的返回结果均为 `[[0,0,0,0],[0,0,0,0]]`，与输入值（rank0=1，rank1=2）完全无关，恒返回全零 |
| `dist.all_gather_object` | **崩溃**——`EOFError('Ran out of input')`（下游依赖 pickle 反序列化，底层 all_gather 已损坏导致的连锁失败） |

诊断脚本：`/tmp` 临时文件（未入库，属于本次调试产物，非可复现资产的一部分；
复现方法见下方「复现步骤」，任何人可在同样的 2 卡容器内重新跑出同样结果）。

**这不是训练脚本的问题**，`train_qwen_1_5b_npu.py` 本身逐字节未改动、也是这条
"HCCL adaptor" 路径唯一已知的真机成功案例（历史记录：2481 步、loss 1.95、
4245-5428 tok/s）——问题出在本血统实际构建出的 FlagCX 0.13.0 @
`4e0e0cbcbf721169ca82348080f8353aebfe2c31`（`FlagRT/FlagCX` 组织仓公开主干）
的 `broadcast`/`all_gather` 实现上，与历史验证记录所用的环境（不同分支
`kistich/ascend-dev1.0`，见 lock.yaml 已有记录）不是同一份构建，无法排除是
两者之间某个未知差异导致的问题（编译选项、HCCL 版本、还是 FlagCX 该 commit
本身在这两个 op 上确有 bug——本阶段无法判断，需要 FlagCX 维护者介入）。

**按任务书要求处理方式**：未尝试给 FlagCX 打私有补丁、未尝试绕过 DDP 的形状
校验（例如 monkeypatch 掉 `_verify_param_shape_across_processes` 强行跳过）
——即使能绕过这一步，`broadcast` 本身损坏意味着各 rank 的初始模型权重根本
不会同步，继续训练只会静默产出垃圾结果（并非"训练能跑但吞吐差"，而是
"看起来在跑，实际每个 rank 各自训练不同的模型副本，梯度/loss 数值全部不可信"）
——这正是任务书 STOP CONDITION 1（"产生错误的数值结果"）明确要求停下不要
自行强行修的场景。如实记录，未产出任何真实 loss 曲线或 tok/s 数据。

**复现步骤**（供项目负责人/FlagCX 维护者复核）：
```bash
# 容器：flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64
# （round 2：正式非诊断镜像，vLLM 插件层已换成 vllm-plugin-FL 并构建成功；本
# STOP CONDITION 与 vLLM 插件选型无关，FlagCX 层本身逐字节未变，预期同样复现
# ——round 2 未重新执行本条复现步骤去重新确认，因为这不在本轮任务范围内）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_WHITELIST_DISABLE=1
torchrun --nproc_per_node=2 --master_port=<port> train_qwen_1_5b_npu.py
# 或最小复现：dist.init_process_group(backend="flagcx") 后单独调用
# dist.broadcast(t, src=0) / dist.all_gather(out_list, t)，对比 rank0/rank1 结果
```

## ROUND 3（追加，项目负责人指派）："缺 synchronize()" 假说检验——【PENDING，机器并发占满，未能在本轮会话内执行】

**背景**：项目负责人复核了本机另一个 `FlagRT/vllm-plugin-FL` 组织私有 fork
（`/home/xliu969/runtime-team/vllm-plugin-FL/`，与 round 2 用的纯公开
`flagos-ai/vllm-plugin-FL` 上游是两回事）里已合并的 commit `5d545c9`
（`fix(ascend): int64 mask promote + flagcx sync for all_gather/all_reduce`，
PR #1，merge commit `f34b4e7`）。该 commit 在 `vllm_fl/distributed/
communicator.py` 里的说明是："flagcx backend returns async, must sync
before reading result"——即 flagcx 的集合通信在这个 backend 下是异步返回的，
读取结果前必须显式 `torch.npu.synchronize()`，否则读到的是尚未完成的旧数据。
这提出了一个值得直接验证的假说：**上面「PHASE 2 STOP CONDITION」记录的
`broadcast`/`all_gather` 错误结果，会不会只是当时的诊断脚本忘了同步，而不是
FlagCX 本身真的坏了？**

**验证方式（已设计、脚本已就绪，任务书要求"直接、廉价地测，不需要改镜像"）**：
`assets/provenance/` 之外新增（尚未入库，先放本次会话 scratchpad，验证通过后
再决定是否正式入库为 `assets/`）一个最小复现脚本
`flagcx_sync_test.py`——对 `dist.broadcast(src=0)` 与 `dist.all_gather`
（tensor-list 形式）各自跑两遍：先按原诊断方式（不加 sync，重新确认原
STOP CONDITION 是否依然复现），再在集合通信调用之后、读取结果张量之前插入
`torch.npu.synchronize()`（commit `5d545c9` 的确切修法），对比两种模式的
结果，在同一个 `torchrun --nproc_per_node=2` 进程里跑在现有
`ascend-train-comm:v3` 镜像（`43f3e2f70b4c`，round 2 已建，已有 FlagCX，无需
改镜像/改 Dockerfile）里，两个 rank 均打印明确 PASS/FAIL。如果假说成立
（sync 后两个集合通信都变正确），还准备了一个可选追加测试
`train_qwen_1_5b_npu_syncpatch.py`：在不改动 `train_qwen_1_5b_npu.py` 本身
的前提下，猴子补丁 `torch.distributed.broadcast`/`all_gather`/
`all_gather_object`，在真实调用之后立即插入
`torch.npu.synchronize()`，再跑一遍真实训练脚本的 DDP 构造，看是否不再报
"Rank 1 has 338 params, while rank 0 has inconsistent 0 params"——但已知的
一个真实风险提前记录在案：PyTorch `DistributedDataParallel.__init__` 内部的
`_verify_param_shape_across_processes` 实际调用的是 C++ 层原语
`torch._C._distributed_c10d._verify_params_across_processes(...)`，不是
Python 层的 `dist.broadcast`/`dist.all_gather`——如果 DDP 走的就是这条路径，
上面这个模块级猴子补丁根本拦不住它，即便"缺 sync"假说被下面的直接测试证实，
这个补丁版训练脚本仍可能照样失败，这本身也是一条有效结论（说明"缺 sync"的
真正修复点必须落在 FlagCX 自己的 c10d ProcessGroup 实现里，而不能在调用方
用 Python 猴子补丁绕过），不代表假说被推翻。

**未能执行的原因（与本文件其余"PENDING"项同一个阻塞源）**：项目负责人下达
本任务时机器带卡容器并发数为 5（超过 `stack.lock` 规定的上限 3），本会话
随后又追加监控/尝试了 5 轮、每轮约 10 分钟的高频（3 秒间隔）窗口捕捉（叠加
本文件更早的 Step 5 验证已经等待的约 2.5 小时），期间涉事的 `sgl-c256-*`
sweep 容器名多次变化后最终稳定在 `sgl-c256-tp4-20260922-prefill-{0..3}`，
`flaggems-cann9.0.0` 已连续运行超过 15 小时——机器带卡容器数持续保持在 5，
从未降到给本任务留出 1 个名额所需的 ≤2。按"respect shared-machine rules"
要求，本轮同样未强行起带卡容器抢占。**明确记为 PENDING，不是"跳过"，也不是
"假设 sync 能修好"——原 STOP CONDITION 的结论（broadcast 静默 no-op、
all_gather 恒返回全零）在有真机证据推翻它之前，继续按"未解决、真实"处理。**

**下一步**：机器带卡容器并发数降到 ≤2 时，运行
`torchrun --nproc_per_node=2 --master_port=<port> flagcx_sync_test.py`
（脚本已就绪，见本条目所述路径），根据其 exit code/PASS-FAIL 结果决定是否
继续跑 `train_qwen_1_5b_npu_syncpatch.py` 追加测试；跑完后把两个脚本迁入
本目录 `assets/`（若验证证实有长期价值）并回填本节的真实结果，同时更新
`lock.yaml` 与 `image_list.md` 的 broadcast/all_gather STOP CONDITION 状态
（维持、澄清为"测试遗漏 sync 导致的误报"、或其它）。
