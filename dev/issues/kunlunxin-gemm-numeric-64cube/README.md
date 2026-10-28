# 昆仑芯 GEMM 数值正确性：64×64×64 fp32 非确定性乱值（+ 大形状绝对容差口径）

## 问题与现状

- **一句话**：昆仑芯 P800 上，FlagGems GEMM 类算子的**编译问题已修**（compat 包，
  见 [`../kunlunxin-gemm-triton-compile/`](../kunlunxin-gemm-triton-compile/README.md)
  的 env/v1 表态），但**数值正确性存在独立缺口**：64×64×64 fp32 的 `torch.mm` 在
  FlagGems 路径下为**非确定性乱值** —— 同输入多次运行相对误差 0.29~1.02、无复现
  规律；同输入厂商原生 `torch.mm` 误差 1.3e-5。**任何精度口径都不可接受。**
- **归属方向**：算子编译方向（flagtree / triton xpu 后端；修复交付方 = 编译组
  `Xproject-CHKU-SZ/FlagTree` @`edca42a`）；大形状绝对容差**口径裁定线**归
  performance（本 case 只承载数据与复现）。
- **状态（2026-10-28）**：未修复；反馈已同步编译组（含 8 行复现法与排查建议：
  split-K / atomic_add 或 64 整倍 tile 边界路径），被动等进展（2026-11 问询窗确认）。
- **为什么单独立 case**：与编译失败（K-1）不同根因、不同判据 —— 编译维度判据
  （12/12 编译 + pytest 18/99）不覆盖该形状（64³ 不在 FlagGems pytest 的 MNK 矩阵
  内），编译验收通过后此问题依旧命中。发布物已知限制编号 **K-5**。
- **实测定量**（锁定镜像 xpu4，2026-10-27，容差 = 对 FP32 CPU 参考的最大绝对误差：
  fp16 1e-2 / bf16 5e-2 / fp32 1e-3；完整逐组数据
  P800 `/data2/xliu969/sdnn-compat-acceptance-20261027/evidence/numerics-card4.log`）：

| 组合 | max_abs_diff | 容差 | 超差倍数 | 相对误差 |
|---|---|---|---|---|
| 64×64×64 fp32 | 3.37e+1（三次重测 rel 1.02/0.29/0.72，**非确定性**） | 1e-3 | ~3.4万× | 0.29~1.02 |
| 64×64×64 bf16 | 2.27e-1 | 5e-2 | 4.5× | 6.0e-3 |
| 15×160×1024 fp16 | 3.10e-2 | 1e-2 | 3.1× | 2.8e-4 |
| 15×160×1024 bf16 | 4.89e-1 | 5e-2 | 9.8× | 4.2e-3 |
| 495×5333×71 fp16 | 1.56e-2 | 1e-2 | 1.6× | 3.4e-4 |
| 495×5333×71 bf16 | 2.50e-1 | 5e-2 | 5.0× | 5.4e-3 |

- **两部分的定性不同**：64³ fp32（非确定性）与 64³ bf16 是**硬伤**（超差且无规律
  或倍数过大）；大形状五组超绝对容差但相对误差仅 2.8e-4~6e-3，属累加/舍入**口径
  之争**，待裁定（2026-11 问询窗或转 performance），本 case 数据在案。
- **vendor 对照**：同输入厂商原生 `torch.mm` 误差 1.3e-5 —— 缺陷在 FlagGems/
  triton 编译路径，不在设备或参考实现。
- **影响**：已验可用面 = pytest mm 18 / 扩展 99 组矩阵 + minirepro 12 组编译
  （该矩阵内数值全过）。矩阵外的形状（尤其 64 整倍边界）**先自测再用于关键路径**；
  64³ fp32/bf16 禁用于关键路径，命中时以厂商原生实现兜底。

### 版本集合（分版表态）

坐标 = **镜像血统 + 运行环境修订**（第 0 版 = 冻结的 `demo/`，第 1 版起 = `env/vN`）。
**数值缺陷在 compiler/kernel 本体，与运行环境版本无关**——启用 GEMM 修复与否只决定
「能不能编译」，不改变乱值本身。

| 坐标 | 状态 | 依据 |
|---|---|---|
| `kunlun-operator-runtime v2` · 第 0 版（demo） | **不适用**（GEMM 编译即失败，走不到数值） | 见编译 case：12 组全编译失败 |
| `kunlun-operator-runtime v2` · env/v1（显式启用后） | **在该版仍命中** | 2026-10-28 本机卡 0（发布形态）：三连跑 rel = 0.97/1.0、0.79/1.0、0.98/1.0（见 `assets/evidence/env1-three-runs-20261028.txt`）；另有 pytest 矩阵外 7 组超差（minirepro 数值段） |
| `kunlun-operator-runtime v1` | **未在该版测过** | 编译失败同样先行，无数值可言 |

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | 昆仑芯发布物 `kunlun-operator-runtime v2`（image id `sha256:cac5ddae…`，[`dev/stack.lock.kunlunxin.v1.yaml`](../../stack.lock.kunlunxin.v1.yaml) `lock.runtime`） | [`kunlun-operator-runtime/v2/REBUILD.md`](../../images/kunlun-operator-runtime/v2/REBUILD.md) 重建或 ARCHIVE.md 离线包；`up.sh --strict` 机器核对 |
| 运行环境 | [`kunlun-operator-runtime/v2/env/v1/`](../../images/kunlun-operator-runtime/v2/env/v1/README.md)（现行；含 GEMM compat，**须显式启用**） | 随仓；启用方式见其 README |
| 卡 | 1 × P800 XPU | `xpu-smi` 挑空闲卡（避开有残留占用的卡——编译期新上下文遇残留可能 CPU 满转挂死） |
| 模型与数据 | 不需要（最小复现不加载模型、不联网） | — |

### 单入口命令（自判定，三态）

```bash
FLAGRT_MODELS_DIR=<任一实体目录> dev/lib/up.sh dev/images/kunlun-operator-runtime/v2/env/v1 --strict
docker exec flagrt-kunlun-v2-env-v1-p800 \
  env CUDA_VISIBLE_DEVICES=<xpu-smi 挑中的空闲卡号> \
  /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
  /workspace/dev/issues/kunlunxin-gemm-numeric-64cube/assets/verify_numeric_64cube.py
docker compose -f dev/images/kunlun-operator-runtime/v2/env/v1/docker-compose.yml down
```

（脚本头部带 `import gemm_compat` 自动启用兜底，未手工 source enable 也可直接跑；
固定输入种子 20261028，三次 trial 同输入 —— 输出不同即非确定性的直接证据。）

**预期结果（缺陷复现时）**：`overall: FAIL / 1`，三次 trial `gems rel` 在 0.1~1.0
量级且彼此不同，`vendor native max_abs` ~1e-5。

### 编译组侧最小复现（8 行，自建镜像内）

```python
import torch, flag_gems
a = torch.randn((64,64), dtype=torch.float32, device=flag_gems.device)
b = torch.randn((64,64), dtype=torch.float32, device=flag_gems.device)
ref = torch.mm(a.float().cpu(), b.float().cpu())
with flag_gems.use_gems():
    out = torch.mm(a,b)
d = (out.float().cpu()-ref).abs().max().item()
print(f"rel={d/ref.abs().max().item():.4e}  native={(torch.mm(a,b).float().cpu()-ref).abs().max().item():.4e}")
```

（多次运行观察 rel 波动；原生对照预期 ~1e-5。收录自验收反馈稿，原始探针
`assets/probe_64_cube.py`。）

## 修复验收

- **判据**：单入口命令 `overall: PASS`（退出码 0）= 三次同输入 trial 全部
  rel ≤ 1e-3 且结果一致；64³ bf16 同判据（容差 5e-2）。届时本 case 关闭、K-5
  删除，编译 case 的「编译并算对」整体判据随之达成。
- **大形状口径**（fp16 1e-2 / bf16 5e-2 绝对容差 vs 相对误差口径）不在编译组
  修复验收内，另行裁定；裁定结果回写本表。
- **排查建议**（已随反馈稿同步编译组）：split-K / atomic_add 路径的非确定性写回、
  64 整倍 tile 边界的 mask/边界处理；建议将 64×64×64 纳入回归矩阵（该形状现
  不在 FlagGems pytest MNK 矩阵内）。

<!--
changelog:
2026-10-28 —— 首版建立（自 T-017 数值线拆出，B15 / K-5 载体）：64³ fp32 非确定性乱值
  为硬伤主判据（自判定脚本 + 固定种子三 trial），64³ bf16 与大形状 fp16/bf16 数据在案
  （后者定性为口径之争、转裁定）。复现基于现行运行环境 env/v1（GEMM 编译修复已启用为
  前提——不启用则编译失败、走不到数值）。新基座三连跑数据 2026-10-28 卡 0。依据 ADR-0010。
-->
