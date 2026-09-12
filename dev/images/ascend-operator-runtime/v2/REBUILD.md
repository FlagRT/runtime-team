# 重建 ascend-operator-runtime v2

新血统首次构建(非既有镜像重建),对应 `dev/images/TODO.md` 的"下一期 N1"项:
按 BAAI·FlagTree 官方 ascend3.5 手册的基座镜像 + 编译路径,产出可从零复现、
公开可锚定的训练腿祖先镜像。

## 方式

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`。需要构建期网络
(`docker build --network=host`)——从 FlagTree 官方公开上游仓库拉取
`triton_v3.5.x` 分支的 pinned commit,以及 BAAI·FlagTree 官方公开分发桶
(`ks3-cn-beijing.ksyuncs.com`)的 llvm 工具链 + triton 编译依赖包。两者都是
BAAI·FlagTree 官方公开资源,不是 FlagRT 私有资产,构建期只读、不写。

## 构建上下文需备齐

| 项 | 来源 |
|---|---|
| `Dockerfile` | = 本目录 `Dockerfile.repro`(build.sh 会拷) |
| `src/FlagGems/` | `git archive` 自 `FlagGems @ f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4`(与 v1 相同 commit) |
| `src/Torch-FL/` | `git archive` 自 `PyTorch-Plugin-FL(Torch-FL) @ 162582d678e40f133924a4d3b5b6df1cb8154dc7`(比 v1 pin 的 `af50297` 新,组内主干当前 tip) |
| `assets/patch_triton_ascend_flagtree.py` / `verify_runtime.py` / `FlagGems-DSA-__init__.py` | 本目录 `assets/` |
| llvm 工具链 + triton 编译依赖 | `Dockerfile.repro` 构建期从 BAAI·FlagTree 官方公开分发桶 curl(见 Dockerfile 内 URL),未入库(共 ~2.4GB) |
| FlagTree 源码(triton_v3.5.x) | `Dockerfile.repro` 构建期从 `https://github.com/flagos-ai/FlagTree.git`(上游)`git fetch --depth 1` 到 pinned commit,未入库 |
| 基座镜像 | `harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2`(本机已存在;BAAI·FlagTree 官方公开 harbor + 官方离线包镜像) |

## 与 v1 的区别(见 lock.yaml 逐层 diff)

1. **基座**:BAAI·FlagTree 官方预构建镜像,取代 BAAI 内部手工构建的
   `pytorch-plugin-fl:manual-20260807-ascend-dev`。这是本血统相对 v1 的核心
   可锚定性改进——基座本身现在也公开可拉、公开可核验版本号,不再是"仅 BAAI
   内部账号可拉的手搭产物"。
2. **triton**:现场编译(FlagTree `triton_v3.5.x` 分支,pinned commit
   `15ec1a6cbc8d51f597f46459a500e96f3812c58f`),取代"pip 装上游 `triton==3.5.0`
   wheel + 华为昇腾官方 `triton-ascend==3.2.1` wheel(从 vllm-ascend 镜像拷出)"
   的组合。**实测版本是 `triton.__version__ == 3.5.1`**,不是 FlagTree 版本线
   表格暗示的 3.5.0——按实测记录,未采信手册假设。
3. **triton-ascend 后端补丁脚本重写**:FlagTree 的 triton 3.5.1 已经把 ascend
   后端重构成 `triton/backends/ascend/backend_register.py` 的
   `backend_strategy_registry` 插件式结构(`"torch_npu"` / `"mindspore"` 两个已
   注册 category),不再是 v1 面对的那种扁平、硬编码 `torch_npu` 调用的
   `driver.py`/`utils.py`。**实测直接对这份新代码跑 v1 的
   `patch_triton_ascend_3_2_1.py`(以及 Torch-FL 仓库自带的
   `scripts/patch_triton_ascend.py`):13 条字符串替换规则里 12 条
   "pattern not found",且唯一命中的那条(注入 stream helper)不足以让
   `torch.npu.*` 调用消失——它们已经搬进 `backend_register.py`,旧脚本够不着。**
   `assets/patch_triton_ascend_flagtree.py` 是为此新增的独立脚本(见该文件顶部
   注释的详细对比),做法是新增一个 `"torch_fl"` category(镜像每个
   `"torch_npu"` 策略函数,`torch.npu.*` → `torch.flagos.*`,真正
   torch_npu-C++-API 相关的 `allocate_sync_block_lock` 换成 CANN 原生
   `rtMalloc`),并把后端选择逻辑从"先 try import torch_npu"改成
   "`hasattr(torch, 'flagos')` 优先"——对不装 torch_fl 的环境完全不生效,纯增量,
   不破坏华为官方 torch_npu 主线的行为。
4. **MPI**:改用基座自带系统 `mpich`(apt,4.0-3),不再从源码编译 MPICH
   4.1.3。因系统 mpich 头文件在 multiarch 路径,与 CANN `hccl_test` 自带
   Makefile 的 `${MPI_HOME}/include` 假设不兼容,遂放弃编译该二进制自检——
   通信正确性改由更强证据覆盖(见 `ascend-train-comm/v2` 的真机
   `torch.distributed` all_reduce/all_gather/p2p)。
5. **torch_npu 不再被排斥**:见下节。

## torch_npu 共存问题——实测结论(任务书要求的关键判断)

背景:新基座自带 `torch_npu==2.10.0`,而 v1 的构建脚本(`verify_flagcx_runtime.py`
等)在构建期断言 torch_npu **不存在**。任务书要求实测确认:这是真实的运行时
互斥,还是仅仅是"import 顺序"要求。

**实测方法**(容器内,构建阶段,无 NPU 设备挂载——PrivateUse1 注册逻辑是纯
Python/C++ dispatcher 层面的机制,与是否有真实设备无关,故不挂设备也可测):

```python
# 场景 A:torch_fl 先
import torch_fl                    # 声明 PrivateUse1 = "flagos"
import torch
import torch_npu                   # 之后再 import torch_npu
# 结果:import 成功,不抛异常
# torch._C._get_privateuse1_backend_name() 之后仍是 "flagos"
```

```python
# 场景 B:torch_npu 先(或未设 TORCH_DEVICE_BACKEND_AUTOLOAD=0)
import torch                       # autoload 让 torch_npu 抢注 PrivateUse1 = "npu"
import torch_npu
import torch_fl                    # 之后再 import torch_fl
# 结果:RuntimeError: "PrivateUse1 is already claimed by the 'npu' backend,
#        so torch_fl cannot register 'flagos'. ..."
```

**结论:这是进程内 import 顺序约束,不是"两个包不能共装"的约束。**
- 先 `torch_fl` 后 `torch_npu`:安全,`torch_npu` 静默不覆盖已占用的
  PrivateUse1(不崩溃、不损坏状态)。
- 先 `torch_npu`(或 autoload 生效)后 `torch_fl`:`torch_fl` 立即抛出清晰的
  `RuntimeError`,fail-loud,不会静默损坏。
- 这与 `dev/stack.lock.910c.v1.yaml` 的 `per_leg.train.note`("必须
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 且先 import torch_fl 再 import torch")完全
  一致并进一步给出了实测证据。

**处理方式**:v2 **不卸载**基座自带的 `torch_npu`(供其他用途共用同一基座,
比如推理腿),也**不在构建期断言其不存在**。`assets/verify_runtime.py` /
`ascend-train-comm/v2/assets/verify_flagcx_runtime.py` 把 v1 的硬性断言替换成
"torch_npu coexistence" 检查:先 import torch_fl 确认拿到 `"flagos"`,再 import
torch_npu,确认它没有抢注、没有异常。这是把"包不能共装"的过度保守断言改成了
验证"真正不变量(进程内 import 顺序)"的检查。

## 静态自检(2026-09-12 实测)

| 判据 | 结果 |
|---|---|
| `docker build --network=host` | 通过,~20 分钟(llvm 下载 ~3min + triton 编译 ~13min + Torch-FL/FlagGems ~2min) |
| `triton.__version__` | `3.5.1` |
| `flagtree` 包版本 | `0.6.0+ascend.git15ec1a6c` |
| `torch_fl` / `flag_gems` 版本 | `0.1.0` / `0.0.0+f7ae8e6b934a` |
| `patch_triton_ascend_flagtree.py` 四处替换 | 全部精确命中(0 条 "pattern not found") |
| `verify_runtime.py --static` | 通过 |

基线文件:见 `assets/provenance/`(pip freeze 快照等)。

## 真机动态验证(2026-09-12,2 卡,容器 `flagos-cand-train-910c-v2`)

| 判据 | 结果 |
|---|---|
| `torch_fl.flagos.device_count()` | `2`(真实设备,非 0) |
| `verify_runtime.py`(完整动态,含 import torch_fl/flag_gems/triton) | 通过,含 `torch_npu_coexistence_check: passed` |
| Torch-FL 原生 ACLNN 算子(`torch.abs` on `flagos:0`) | 结果与 CPU 一致 |
| FlagGems/triton 算子(`with flag_gems.use_gems(): torch.abs(...)`) | 触发真实 triton kernel 编译执行,结果与 CPU 一致 |
| `x @ x` matmul | 输出 finite,形状正确 |

**真机测试中额外发现并修复的问题**(见
`assets/patch_triton_ascend_flagtree.py` 第 4 条补丁 docstring)：FlagTree 顶层
triton 驱动自动发现(`triton/backends/__init__.py` +
`triton/backends/nvidia/driver.py`)会把 Torch-FL 的生态兼容 shim
(`torch.cuda.is_available = lambda: flagos.device_count() > 0`,用于让下游库
把 flagos 当"有加速器")误判成"CUDA 也是 active",与真正的 NPUDriver 冲突,
报 `RuntimeError: 2 active drivers`。修复:给 `CudaDriver.is_active()` 加一条
`torch.version.cuda is not None`(真正编译了 CUDA,而不只是生态兼容信号),
只影响 nvidia 驱动的自检逻辑,不改动 ascend 驱动。已验证修复后 triton 正常
选中 NPUDriver、FlagGems/triton kernel 在真机正确执行(见上表)。

FlagCX 层的 device 级验证(2 卡 all_reduce/all_gather/p2p)见
`dev/images/ascend-train-comm/v2/REBUILD.md`。
