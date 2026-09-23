# 重建 ascend-operator-runtime v3

> **阶段声明（PHASE 2 ROUND 2 已执行，2026-09-22）**：round 1（本文档下方
> 「PHASE 2 STOP CONDITION 历史记录（round 1，已被 round 2 取代）」一节）在
> vllm-ascend 安装层构建失败，未产出设计 tag 镜像实体，只构建出一个跳过该层的
> `-DIAGNOSTIC-novllm` 诊断变体。**round 2 把 vLLM 推理插件从 vllm-ascend
> 换成 vllm-plugin-FL（BAAI·FlagTree 官方手册自己为 ascend3.5 线背书的推理插件
> 路径），设计 tag 构建成功（`docker build --network=host` 全部 27 步通过，
> image id `9ad551058f2f`）**——`flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-
> routeA-cann9.0-py311-torch2.10-arm64` 现在是一个真实产出的镜像实体，不再是
> "未构建成功的设计稿"，也不需要 `-DIAGNOSTIC-novllm` 变体了（该变体镜像仍保留
> 在本机 docker images 里，未删除，但已被本次构建的正式 tag 取代，不建议再用）。
> 下面「静态自检」「真机动态验证」表格已更新为 round 2 的真实结果；round 1 的
> vllm-ascend 尝试与失败原因作为历史记录保留在「PHASE 2 STOP CONDITION 历史
> 记录」一节，不再是当前阻塞项。

Route A 默认血统 + 训练/推理统一基座，在 `ascend-operator-runtime:v2` 之上迭代。

## 方式（phase 2 执行）

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`。需要构建期网络
（`docker build --network=host`）——与 v2 相同的 FlagTree/triton 依赖拉取，外加
vLLM 推理插件仓库 clone（round 2：`https://github.com/flagos-ai/vllm-plugin-FL.git`，
BAAI·FlagTree 官方公开仓库；round 1 曾用
`https://github.com/vllm-project/vllm-ascend.git`，已废弃，见下方历史记录）。

## 构建上下文需备齐

| 项 | 来源 |
|---|---|
| `Dockerfile` | = 本目录 `Dockerfile.repro`（build.sh 会拷） |
| `src/FlagGems/` | `git archive` 自 `FlagGems @ f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4`（与 v2 相同 commit，round 2 未改动，见下方「FlagGems 版本兼容性」） |
| `src/Torch-FL/` | `git archive` 自 `Torch-FL @ 162582d678e40f133924a4d3b5b6df1cb8154dc7`（与 v2 相同 commit，未升级） |
| `assets/patch_triton_ascend_flagtree.py` / `verify_runtime.py` / `FlagGems-DSA-__init__.py` | 本目录 `assets/`（补丁脚本与 DSA 标记文件与 v2 逐字节相同；`verify_runtime.py` 已按 v3 语义重写，round 2 把 vllm_ascend 检查项改成 vllm_fl，见文件头注释） |
| llvm 工具链 + triton 编译依赖 / FlagTree 源码 | 与 v2 相同，构建期从 BAAI·FlagTree 官方公开资源现场拉取，未入库 |
| **vllm-plugin-FL 源码（round 2 新增，取代 round 1 的 vllm-ascend）** | 构建期从 `https://github.com/flagos-ai/vllm-plugin-FL.git`（BAAI·FlagTree 官方公开仓库）`git fetch --depth 1` 到 `VLLM_PLUGIN_FL_REF`（`release/0.2` 分支 tip，`git ls-remote` 实测取得，见下方「vllm-ascend → vllm-plugin-FL pivot」），未入库 |
| 基座镜像 | 与 v2 相同：`harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2` |

## 与 v2 的区别

### 1. Route A 成为默认生效设备后端（任务书第 1 条）

v2 的实测结论已经证明：torch_fl 与 torch_npu 之间只有"进程内 import 顺序"约束，
不是"不能共装"的约束（见 `ascend-operator-runtime/v2/REBUILD.md`「torch_npu
共存问题」）。v2 本身并没有在 operator-runtime 这一层强制压制 torch_npu
autoload（那是下游 `ascend-train-comm/v2` 层的 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
+ `FLAGCX_TORCH_BACKEND=flagos` ENV 做的），但 v2 的动态自检脚本
(`verify_runtime.py`) 要求"先 import torch_fl 才算通过"，隐含把 flagos 当"标准
路径"。v3 的改动：

- `Dockerfile.repro` 不添加任何压制 autoload 的 ENV（与 v2 本层相同，仅是显式
  标注、不再有歧义空间）。
- `assets/verify_runtime.py` 反转默认检查顺序：什么都不做特殊处理，纯
  `import torch`，就必须看到 `torch._C._get_privateuse1_backend_name() == "npu"`。
- 新增 `--check-torch-fl-guard`：验证"torch_npu 已抢注后再 import torch_fl"
  必须清晰 `RuntimeError`（fail-loud，不会静默变成 flagos 生效）——这是 v2
  `torch_npu_coexistence_check` 的镜像反转版本。

真正"移除 flagos 强制"的开关搬迁发生在 `ascend-train-comm/v3`（它是唯一实际
设置过那些 ENV 的层），见该目录 REBUILD.md。

### 2. FlagGems / FlagTree / Torch-FL：保留但明确"可插拔，非默认激活"（任务书第 4/5 条）

triton_v3.5.x 编译链、FlagGems、Torch-FL 的安装步骤与 v2 **逐 commit 相同**，
未做任何删减——任务书要求"保留可插拔能力，供后续可选启用"，不是要求移除。
变化只是文档/标签层面的显式化：
- `LABEL io.flagrt.torch_fl.activation` / `io.flagrt.flag_gems.activation`
  明确写"installed but not active by default"。
- `verify_runtime.py` 的默认路径不再 import 它们（只做 `find_spec` 存在性
  检查），需要 `--check-torch-fl-guard` 才会尝试 import（且预期失败）。

triton-ascend 后端补丁 (`assets/patch_triton_ascend_flagtree.py`) 逐字节未改
——它对纯 torch_npu 环境本来就是 no-op（新增的 `"torch_fl"` category 只有
`hasattr(torch, "flagos")` 才会被 `get_backend_func()` 选中）。

### 3. vLLM 推理插件层（任务书第 3 条：训练 + 推理统一血统）——round 2 PIVOTED

父镜像已经内置裸 vLLM 0.20.2（`vllm-project/vllm` 上游 editable checkout，
commit `bc150f50299199599673614f80d12a196f377655`）+ `torch_npu 2.10.0`，但**没有
任何 vLLM Platform 插件**——vLLM 0.20.x 的硬件分发走 out-of-tree Platform 插件
机制，裸 vLLM 不会自己发现 NPU 设备。

**这不是"训练腿镜像顺带装了点东西"——这正是"一条血统、训练推理都能跑"的核心
动作**：同一个 `ascend-operator-runtime:v3` 镜像，下游 `ascend-train-comm/v3`
叠加 FlagCX 后可以跑 DDP 训练；同一个镜像本身（不需要叠加 FlagCX）也具备
`vllm.LLM(...)` / `vllm serve` 走 NPU 推理的前提条件。

## vllm-ascend → vllm-plugin-FL pivot（本节取代原「开放问题」两小节）

### round 1：vllm-ascend——【STOP CONDITION，已被 round 2 取代，历史记录保留】

round 1 选用华为昇腾官方 `vllm-ascend`（`vllm-project/vllm-ascend`）。精确 pin
commit 核实无误（`367b8e62da799870a7476ce34f5f7658589a8aad`，与
`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` 内 `pip show vllm-ascend` 输出一致，
`git ls-remote ... v0.20.2rc1` 交叉确认该 tag 精确指向同一 commit）——**这一步
本身没有问题**。但 `docker build --network=host` 在安装层
（`pip install --no-build-isolation -v .`）失败：

```
ERROR: Could not find a version that satisfies the requirement
triton-ascend==3.2.1 (from vllm-ascend) (from versions: 3.2.0rc2, 3.2.0rc3,
3.2.0rc4, 3.2.0)
ERROR: No matching distribution found for triton-ascend==3.2.1
```

**根因（已核实）**：base 镜像的 `pip config list` 显示
`global.index-url='https://pypi.org/simple'`（纯公开 PyPI，无内部镜像源）。
vllm-ascend@367b8e62d 的依赖声明硬性要求 `triton-ascend==3.2.1`，但公开 PyPI
索引对这个包从未发布过 3.2.1（只有 `3.2.0rc2`/`rc3`/`rc4`/正式版 `3.2.0`）。
`dev/images/image_list.md` 版本线表格早就记录了这个事实的另一面："triton_ascend
3.2.1 ... 的 wheel 从 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` 拷出"——也就是说
这个精确版本号从未经由标准 PyPI 索引分发，v1 血统能装上它，靠的是从官方镜像里
提取 wheel 再本地 `pip install <wheel>`，不是一条能被 `pip install .` 自动解析
出来的依赖。这不是版本号猜错、也不是降级能绕过的问题（会破坏 vllm-ascend 自身
兼容性声明）。round 1 按任务书要求未自行绕过，如实记录为 STOP CONDITION，另建
了一个跳过该层的诊断变体
`flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64-DIAGNOSTIC-novllm`
（image id `4bab61434602`）来验证其余不依赖 vllm-ascend 的部分。**这个 STOP
CONDITION 本身从未被"解决"**——round 2 不是修好了它，而是换了一条完全不同的
路径，绕开了这个具体依赖墙。round 1 的候选解法列表（提取 wheel / 联系维护者 /
vendor triton-ascend 3.2.0）均未执行，如果日后仍想用 vllm-ascend，这些仍是
待评估的候选方向，不属于本次 round 2 的处理范围。

### round 2：改用 vllm-plugin-FL——【PHASE 2 ROUND 2，2026-09-22，构建成功】

重新核查 BAAI·FlagTree 官方 wiki（真实 clone
`git clone https://github.com/flagos-ai/FlagTree.wiki.git` 读取，非猜测）发现：
该 wiki 的 ascend3.5 线自己的"Run Qwen vLLM benchmark"一节走的根本不是
vllm-ascend，而是 FlagGems + `vllm-plugin-FL`
（`https://github.com/flagos-ai/vllm-plugin-FL`，"A vLLM plugin built on the
FlagOS unified multi-chip backend"）。

**核实过程（均为本轮真实 fetch，非复用上一轮缓存的记忆）**：
- README（`release/0.2` 分支）版本兼容表：`release/0.2` ↔ vLLM v0.20.2（与
  本血统精确匹配）；`main` ↔ vLLM v0.24.0（不匹配，不用）。
- `pyproject.toml`：`[project] dependencies` 只有 `pyyaml`；
  `[project.optional-dependencies]` 的 `test` extra 才引用
  `vllm[audio]==0.20.2`（仅测试用，非核心运行依赖）——核心依赖不含
  `triton-ascend`，不会撞上 round 1 那堵公开 PyPI 墙。
- entry points（`pyproject.toml` 实测确认）：
  `[project.entry-points."vllm.platform_plugins"]  fl = "vllm_fl:register"`，
  `[project.entry-points."vllm.general_plugins"]    fl = "vllm_fl:register_model"`。
- `setup.py`：`SUPPORTED_VENDORS = ("cuda",)`——昇腾/NPU 不在其中，意味着不设
  `VLLM_VENDOR` 时装的是"跳过原生 C++ 扩展的纯 Python 插件"，README 也明确写
  "If VLLM_VENDOR is not set, vllm-plugin-FL is installed as a Python-only
  plugin and the native extension is skipped."——即本层的安装命令不该设
  `VLLM_VENDOR`。
- `vllm_fl/utils.py` 的 `VENDOR_DEVICE_MAP["ascend"] = {"device_type": "npu",
  "device_name": "npu"}`——确认 ascend vendor 分发是按 `device_type == "npu"`
  识别的，与本血统 Route A 默认的 torch_npu（PrivateUse1='npu'）路径直接对应，
  不需要 torch_fl/flagos。

**精确 pin**：`git ls-remote https://github.com/flagos-ai/vllm-plugin-FL.git
release/0.2` 实测取得 `8b059122e32b9ac47b9820a9c7b1bb95077481a4`
（2026-09-22，round 2 本轮现场取得，未复用任何历史值）。已回填
`Dockerfile.repro` 的 `VLLM_PLUGIN_FL_REF` 与 `lock.yaml`。

**构建结果**：`docker build --network=host`，设计 tag，**全部 27 步成功**，
`vllm-plugin-FL` 安装层输出 `Successfully installed vllm-plugin-fl-0.0.0+g8b059122e`，
无 triton-ascend 或任何其它依赖解析错误。镜像 id `9ad551058f2f`。`verify_runtime.py
--static`（构建期最后一步）通过，`vllm_fl_installed: true`。见下方「静态自检」
完整 JSON。

**FlagRT 私有 fork 的发现与决策**：本机 `/home/xliu969/runtime-team/vllm-plugin-FL/`
存在一个 `git@github.com:FlagRT/vllm-plugin-FL.git` 组织 fork 的本地 checkout，
其 `main`/`dev-1.0` 分支与 `flagos-ai/vllm-plugin-FL` 的 `release/0.2` 已经
分叉（`git merge-base --is-ancestor` 确认互不为祖先关系，不是单纯"上游 + 几个
patch"的关系）。该 fork 上有一个已合并的 PR（commit `5d545c9`，作者
`seanl <vlev02@qq.com>`）修复了两个真实的正确性 bug：
1. `vllm_fl/dispatch/backends/vendor/ascend/impl/vocab_parallel_embedding.py`：
   PrivateUse1 后端上 `bool * int64` 类型提升错误（返回 bool 而非 int64），
   导致 `vocab_mask * (input_ - valid_offset)` 把非零 token id 全部坍缩成 1；
2. `vllm_fl/distributed/communicator.py`：`flagcx` backend 下 `all_reduce`/
   `all_gather` 异步返回，读取结果前必须 `torch.npu.synchronize()`，否则读到
   未完成的数据。

   该 commit 的验证记录写的是"Verified: Qwen3-4B single-card inference via
   torch_fl + vllm-plugin-FL"——即验证环境是 torch_fl（flagos）路径，不是本
   血统默认使用的 torch_npu（Route A / "npu"）路径。**本层按任务书"纯公开
   血统"的要求，故意不使用这个 fork、不 cherry-pick 这个 patch**，只用
   `flagos-ai` 纯公开上游——这正是任务书 STOP CONDITION 4（"需要 owner 私有
   commit/patch 才能工作"）要检验的边界。这两个 bug 是否会在真机 torch_npu
   （而非 torch_fl）路径上实际触发，是 Step 5 真机推理烟雾测试要用真实输出
   文本去验证的问题，不能靠读代码/读 commit message 预先断言——结果见下方
   「真机动态验证」。**如果真机验证发现输出确实因为这个已知 bug 而错误，这属于
   STOP CONDITION 2（产生错误推理结果），要如实停下报告，不能私下 cherry-pick
   这个未上游化的 fork patch 来"修好"它。**

### FlagGems 版本兼容性——按任务书要求先用已 pin 的 commit 试

vllm-plugin-FL 官方 README 建议 FlagGems `git checkout
3b2b55c8eda5de44ba3476d26566ecf134db0662`，与本文件下方一直在用、且与 v1/v2
一脉相承的 FlagGems pin（`f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4`）不是同一个
commit。按任务书"先用已 pin 的 commit 试，不要瞎猜切换，只有实测证明真的缺东西
才考虑换"的要求，round 2 **未改动**已有的 FlagGems pin，直接拿它构建：
**构建通过**（`pip install --no-build-isolation --no-deps` 干净安装，`vllm_fl`
的 `register()`/`register_model()` entry point 加载不直接依赖某个具体 FlagGems
commit——它们是惰性的算子分发注册，只有真正调用到某个模型层的具体算子时才会
触达 FlagGems 内部实现）。真机烟雾测试（Step 5）进一步验证了"用现有 pin 跑一次
真实 `generate()`"是否会因为缺某个 FlagGems 算子而报错——结果见下方「真机动态
验证」。**结论（Step 5 之前）**：静态构建/装配层面，现有 FlagGems pin 完全够用，
不存在"装不上"或"import 时报错"的问题；是否存在"某些模型/精度路径下需要更新
FlagGems commit 才能拿到正确算子实现"这一更深层的问题，仍需以真机推理的真实
输出文本判断，不能仅凭构建通过下结论。

## torch_npu 默认生效——设计依据（非新实测，复用 v2 结论）

v2 已经实测确认（见 `ascend-operator-runtime/v2/REBUILD.md`「torch_npu 共存
问题」）：torch_fl 与 torch_npu 的冲突是**进程内 import 顺序**约束，不是
**能否共装**约束：

- 先 `torch_fl` 后 `torch_npu`：安全，无异常，PrivateUse1 仍是 `"flagos"`。
- 先 `torch_npu`（或 autoload 生效）后 `torch_fl`：`torch_fl` 立即抛出清晰
  `RuntimeError`，fail-loud，不静默损坏。

v3 没有做新的实测——只是把这条已验证的不变量，从"v2 默认拿 flagos、torch_npu
共存但从未被用于计算"，翻转成"v3 默认拿 npu（Route A）、torch_fl 需要显式反向
操作才能激活"。`assets/verify_runtime.py --check-torch-fl-guard` 验证的正是
这条不变量的镜像方向（之前测的是"torch_fl 后 torch_npu 安全"，现在测"torch_npu
后 torch_fl 报错"）——这两条本就是同一份 v2 实测记录里写出的对称结论，不是新
猜测。**但 --check-torch-fl-guard 本身、以及 vllm_ascend 的真实可用性，仍然需要
在 phase 2 真机容器里实际跑一遍**才能从"设计推论"升级为"实测结论"。

## 静态自检（PHASE 2 ROUND 2 实测，2026-09-22）

| 判据 | 结果 |
|---|---|
| `docker build --network=host`（设计 tag，含 vllm-plugin-FL 层） | **成功**——全部 27 步通过，image id `9ad551058f2f`，tag `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`（无 `-DIAGNOSTIC` 后缀，正式产出）|
| `triton.__version__` | **`3.5.1`**（实测，与 v2 相同 commit） |
| `flagtree` 版本 | **`0.6.0+ascend.git15ec1a6c`**（实测） |
| `torch_fl` 版本 | **`0.1.0`**（实测） |
| `flag_gems` 版本 | **`0.0.0+f7ae8e6b934a`**（实测，round 2 未改动此 pin，见上方「FlagGems 版本兼容性」） |
| `torch_npu` 版本 | **`2.10.0`**（实测，`torch_npu_installed: true`） |
| `vllm_fl`（vllm-plugin-FL）版本 | **已安装**，`vllm_fl_installed: true`，dist 版本 `0.0.0+g8b059122e`（setuptools-scm 从 commit `8b059122e` 生成，与 pin 一致） |
| `vllm` 版本（父镜像自带裸 vLLM） | `0.20.2+empty`（实测，未变） |
| `verify_runtime.py --static` | **通过**（真实设计 tag 镜像，exit 0，完整 JSON 见下） |

```json
{
  "architecture": "aarch64",
  "flag_gems": "0.0.0+f7ae8e6b934a",
  "flagtree": "0.6.0+ascend.git15ec1a6c",
  "mpirun_first_line": "HYDRA build details:",
  "python": "3.11.15",
  "torch": "2.10.0+cpu",
  "torch_fl": "0.1.0",
  "torch_npu": "2.10.0",
  "torch_npu_installed": true,
  "triton": "3.5.1",
  "vllm": "0.20.2+empty",
  "vllm_fl": "0.0.0+g8b059122e",
  "vllm_fl_installed": true
}
```

pip freeze 快照：`assets/provenance/pipfreeze-operator-runtime-v3.txt`（round 2
已替换 round 1 的诊断镜像快照——现在是真实设计 tag 镜像 `9ad551058f2f` 的
`pip freeze`，非诊断变体，caveat 详见文件内头部注释）。

## 真机动态验证

以下几行是 round 1（diagnostic 镜像，2026-09-22）实测，round 2 的设计 tag 构建
成功后不需要重新证明——它们与 vllm 插件选型无关，逻辑不变：

| 判据 | 结果 |
|---|---|
| `verify_runtime.py`（动态，无特殊 import，默认路径） | **通过**：`"route_a_default_check": "passed (torch_npu claimed PrivateUse1 with no special import order)"`，`torch._C._get_privateuse1_backend_name() == "npu"` 确认，**未经任何特殊 import 顺序**（Route A 默认生效，实测证实） |
| `verify_runtime.py --check-torch-fl-guard` | **通过**（但发现并修复了 assets/verify_runtime.py 自身一处脚本 bug——见下方「PHASE 2 脚本修复」）：`"torch_fl_guard_check": "passed (RuntimeError as expected ...)"` |
| `torch_fl.flagos.device_count()`（opt-in 路径手动验证） | 未重测——v3 默认路径本就不 import torch_fl，v2 已实测 `torch_fl.flagos.device_count()==2`（见 v2/REBUILD.md），判定不必重复 |

下面两行是 round 1 因 vllm-ascend 从未装上而**完全无法执行**的部分，round 2
的设计 tag 已经真实装上了 vllm-plugin-FL，是本轮任务的核心待验证项，结果见
本节最后的「ROUND 2 真机验证结果」：

| 判据 | round 1 状态 | round 2 状态 |
|---|---|---|
| `vllm.LLM(...)` 走 NPU 推理，产出真实、非乱码的输出 | 无法执行（插件未装上） | 见下方「ROUND 2 真机验证结果」 |
| 训练/推理同一血统内组件共存（torch_npu + flagcx + vllm_fl 同进程） | 部分验证（无 vllm 插件） | 见下方「ROUND 2 真机验证结果」，以及 `ascend-train-comm/v3/REBUILD.md` |

FlagCX 训练腿（HCCL adaptor）本身的 STOP CONDITION（`dist.broadcast`/
`dist.all_gather` 返回错误结果）与本轮 vllm 插件 pivot 无关、不在本轮范围内，
详见 `dev/images/ascend-train-comm/v3/REBUILD.md`，结论不变。

### ROUND 2 真机验证结果——【PENDING，机器并发占满，未能在本轮会话内执行】

本轮构建（Step 1-4）已在真机上用 `docker build`（不需要 `--device`）完整跑通，
但 Step 5（vLLM 推理烟雾测试 + 训练/推理共存检查）需要挂 `--device` 的带卡
容器。按 `dev/stack.lock.910c.v2.yaml` 的"最高优先级"规则（机器级并发上限 3
个带卡容器），起容器前 `docker ps` 检查——**本轮会话观测期间，机器上其他
工程师的带卡容器持续保持在 5 个**（`sgl-c256-*-prefill-{0,1,2,3}` 4 个
+ `flaggems-cann9.0.0` 1 个，16 张卡对应的 `/dev/davinci0-15` 全部已被多个
容器同时挂载），**已经超过并发上限 3，且不是本次任务起的**。持续监控约
1 小时（多轮 `docker ps` 轮询，容器名多次变化——`sgl-c256-ep4-*` →
`sgl-c256-dpa4-*` → `sgl-c256-dpa4m68-*` → `sgl-c256-tp2dp2-*`——确认是其他
工程师正在跑的一组活跃、持续变化的 sweep/benchmark，不是可以等一等就会退出的
偶发占用），并发数量从未降到 ≤2（留出 1 个名额给本任务、保持总数 ≤3 所需的
条件）。

按任务书"respect the shared-machine rules"的明确要求（不得让并发带卡容器数
超过 3，不得抢占已经在跑的其它容器的设备），**本轮未强行起带卡容器**——本
可以挂载与其他容器重叠的 `/dev/davinci0,1` 之类的设备文件（Docker 本身不阻止
把同一个设备文件挂进两个不同容器），但这样做既违反并发上限规则，也有实际风险
（读写同一物理 NPU 的驱动状态可能触发 `stack.lock` 里记录的 `acl.init()`
500000 报错，或者更糟——干扰其他工程师正在跑的 benchmark 的正确性）。

**结果**：Step 5（vLLM 推理烟雾测试的真实输出文本、训练/推理组件共存检查）
**未能在本轮会话内执行，明确记为 PENDING，不是"跳过"也不是"假设会通过"**。
Step 1-4（pin 精确 commit、Dockerfile 改造、真机 `docker build` 成功、依赖
`ascend-operator-runtime:v3` 的 `ascend-train-comm:v3` 重建成功）均已完成且
有真实构建日志为证，这部分结论是扎实的；但"vllm-plugin-FL 在本血统 torch_npu
（Route A）路径上跑一次真实 `generate()` 是否产出正确、非乱码输出"这个本轮
任务的核心目的——尤其是上面「FlagRT 私有 fork 的发现与决策」一节提到的两个
已知 PrivateUse1 类型提升 bug 是否会在 torch_npu 路径上实际触发——**仍然
未知，不能假设"构建通过 = 推理正确"**。`repro_status` 因此维持 🟡
partial-repro，不升级为 🟢，见 `lock.yaml`。

**下一步**：待机器带卡容器并发数降到 ≤2（`docker ps` 确认），用
`v3-validate-` 前缀容器名，按任务书 Step 5 的两项检查（1 卡够用：vLLM 单卡
`generate()` 烟雾测试；共存检查可以同一容器顺带做）执行，跑完立即
`docker rm -f`。预计所需真机独占时间很短（单次 `generate()` + 几个 import
检查，不需要长跑训练），一旦有 1 个名额空出即可执行，不需要等到 3 个名额全空。

### PHASE 2 脚本修复：`assets/verify_runtime.py` 的 `torch_fl._C` 探测 bug

实机运行发现：`find_spec("torch_fl._C")`（presence-only 检查，设计意图是"不
import，只查是否存在"）在 v3 默认路径下并不是无副作用的——Python 为了定位子
模块 `torch_fl._C` 的 spec，必须先 import 父包 `torch_fl`（读取其
`__path__`），这就执行了 `torch_fl/__init__.py` 里的
`_check_privateuse1_unclaimed()`，在 PrivateUse1 已被 `torch_npu` 抢注时抛出
`RuntimeError`。**原脚本没有捕获这个异常**，导致哪怕不带任何 flag 的
`verify_runtime.py`（plain 动态路径）在 v3 环境下也会直接崩溃退出（exit 1），
从未打印出预期的 JSON 结果——包括 `route_a_default_check` 这个本该已经算通过的
判据。已修复（改为 try/except 捕获该 RuntimeError 并记录为证据，而不是让脚本
崩溃）；修复后两条命令均 exit 0，结果见上表。`assets/verify_runtime.py` 已更新，
修复已在诊断镜像的运行容器内用 `docker cp` 验证生效（真机实测，非仅静态审查）。

## 容器 / 卡资源使用记录（PHASE 2，任务书并发上限 3 的合规证据）

- 训练腿验证全程只用了 1 个带卡容器（`v3-validate-train-910c`，2 卡
  `--device=/dev/davinci0,1`），`docker ps` 在启动前/启动后均确认过并发数；
  验证跑完后立即 `docker rm -f`，确认清理后并发数回到本次任务发起前的基线。
- 推理腿（Step 4）因 STOP CONDITION 未能起容器，未消耗额外卡资源。
- 全程未同时运行超过 1 个本任务自己起的带卡容器，远低于 3 的上限；`docker ps`
  显示机器上同时还有其他工程师独立运行的带卡容器（如 `flaggems-cann9.0.0`），
  与本次任务用到的容器名（`v3-validate-` 前缀）无重名/无干扰。

## PHASE 2 STOP CONDITION 汇总（本文件相关部分）

1. **vllm-ascend 构建失败**（见上方「vllm-ascend 在本血统下的可构建性」一节）
   ——`triton-ascend==3.2.1` 不在公开 PyPI 索引，公开 `pip install .` 无法解析。
   设计 tag 镜像未产出实体。**待项目负责人裁定是否接受"从官方镜像提取 wheel"
   这类新增层**。
2. **vLLM 推理烟雾测试（任务书 Step 4）因上条被完全阻塞**，未能执行——不是
   本文件自行判定跳过，是设计 tag 镜像不存在导致无法起容器。
3. **训练/推理共存检查（任务书 Step 5）只完成了部分**（不含 vllm-ascend 的
   import-order 检查通过，但真正关注的 vllm-ascend 自身 triton 依赖与
   FlagTree triton 之间的潜在冲突完全未测，因为 vllm-ascend 根本没装上）。

训练腿（FlagCX/DDP）本身发现的独立 STOP CONDITION 见
`dev/images/ascend-train-comm/v3/REBUILD.md`（`dist.broadcast`/`dist.all_gather`
在 `flagcx` backend 下返回错误结果）。
