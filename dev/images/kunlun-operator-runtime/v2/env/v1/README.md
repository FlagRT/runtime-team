# kunlun-operator-runtime v2 · 运行环境 env/v1（现行推荐）

昆仑芯 P800 发布物 `kunlun-operator-runtime v2` 的现行推荐运行环境（第 1 版）。
第 0 版 = 同级 [`../../demo/`](../../demo/)（冻结，只作该镜像版本验收时所用配置的
历史记录）。现行是哪一版以 `dev/images/image_list.md` 与现行锁定声明为准。

## 本环境是什么

锁定镜像（不变，image id 见 [`../../image.baseline.yaml`](../../image.baseline.yaml)）
+ 两件修复载体，解除已知限制 **K-1**（FlagGems GEMM 类算子 mm/bmm/addmm 编译阶段
即失败）的编译维度：

| 载体 | 内容 | 生效方式 |
|---|---|---|
| 宿主仓根 `FlagGems/`（分支 `xliu969/kunlun-gemm-compat`，基点 `73c5aff1` = 镜像固化 FlagGems） | block-pointer 补丁 04（mm）/ 05-bmm / 06（addmm composition），吸收自编译组 `Xproject-CHKU-SZ/FlagTree` @`edca42a`（见 `compat/legacy_sdnn/PROVENANCE.txt`） | 挂载 + PYTHONPATH（`enable-gemm-compat.sh`），不启用则不进入 import 路径 |
| `compat/legacy_sdnn/`（编译组包原样收录） | compiler 副本生成器 `prepare.py` + patch 01/02/03 + 诊断与验收脚本；加载胶水 `gemm_compat.py`（我方追加） | 进程内显式加载（`import gemm_compat`），安装目录从不改动 |

**显式 opt-in**：不启用时本环境行为与 demo（第 0 版）一致。要跑 GEMM 须显式启用
（两种等价方式，见下）——这是刻意设计：compat 是限定 SDK 版本的实验补丁，且存在
数值限制（见下），启用即视为已知悉。

## 用法

```bash
# 仓根执行；先 docker ps（共享机礼貌）+ xpu-smi 挑空闲卡
FLAGRT_MODELS_DIR=<模型目录> dev/lib/up.sh \
  dev/images/kunlun-operator-runtime/v2/env/v1 --strict     # 核对 image id + 宿主 FlagGems commit

# 交互调试（docker exec 进容器后）：
source /workspace/dev/images/kunlun-operator-runtime/v2/env/v1/enable-gemm-compat.sh
python -c "import gemm_compat; import flag_gems, torch; print('ok')"   # 自证启用

# 脚本内（脚本第一行，任何 torch/flag_gems import 之前）：
#   import gemm_compat
```

自证要点：`gemm_compat` 打印 `installed_compiler=7133710c…`（版本锁通过）与副本
sha256；`import flag_gems` 后 `flag_gems.__file__` 应为 `/workspace/FlagGems/src/…`
（补丁版）。up.sh --strict 已强制核对宿主 FlagGems HEAD = 补丁分支提交。

两条使用注意（机制使然）：

- **跑 FlagGems 自带测试须用补丁树的 tests**（`/workspace/FlagGems/tests/…`）。
  镜像内 `/env/FlagGems/tests` 的 conftest 会把 `/env/FlagGems/src` 插回 import
  路径最前，绕过补丁版（表现为 GEMM 测试回到未修复的失败形态）。
- **启用 shell 内的一切 Python 进程自动生效**（含测试工具起的子进程）：compat
  目录进入 PYTHONPATH 后，其 `sitecustomize.py` 在解释器启动期完成全套启用
  （解除镜像内 scikit-build editable finder 对 flag_gems 的 import 劫持 +
  compiler 副本挂载 + 补丁源树置前）。未 source 的 shell 完全不受影响。

## 数值限制（K-5 —— 启用前必读）

编译维度已验收（2026-10-27，锁定镜像 xpu4）：minirepro 12/12 编译通过、
FlagGems pytest `test_accuracy_mm` 18 passed、扩展 mm/bmm/addmm 99 passed。
**数值维度未收口**（编译组修复中，独立跟踪）：

| 形状 | 实测 | 结论 |
|---|---|---|
| 64×64×64 fp32 | 三次重测相对误差 1.02 / 0.29 / 0.72（**非确定性**；同输入 vendor 原生 torch.mm 误差 1.3e-5） | **结果不可信，禁用于任何关键路径**；命中时以 vendor 原生 mm 兜底 |
| 64×64×64 bf16 | 超绝对容差 4.5×（相对误差 6.0e-3） | 同上，慎用 |
| 大形状 fp16/bf16（如 15×160×1024、495×5333×71） | 超我组绝对容差 1.6×~9.8×，相对误差仅 2.8e-4~6e-3 | 累加/舍入口径之争（裁定中）；对绝对误差敏感的消费方暂以 vendor 兜底 |

证据：P800 `/data2/xliu969/sdnn-compat-acceptance-20261027/evidence/`；判据与规避
详见 `dev/issues/` 数值 case（K-5 / B15）。**你的形状不在已验矩阵（pytest 18/99 +
minirepro 12 组）内时，先自测再用于关键路径。**

## 适用前提（版本锁）

compat 包限定 SDK 版本，`gemm_compat` 启用时强制校验（漂移即拒绝）：
compiler.py sha256 `7133710c0153…`；配套 libtriton.so `28942c20acad…`；
flagtree `0.6.1+xpu3.6` / torch 2.9.0 / triton 3.6.0。完整清单见
`compat/legacy_sdnn/README.md`「精确适用版本」与 `PROVENANCE.txt`。

## 维护与退役

本环境为过渡形态：编译组数值修复（B15）到货并复验后出 env/v2 或直接升级
flagtree wheel（走镜像配方层，届时 compat 目录与补丁分支整体退役）。
上游 flagos-ai 正式吸收修复前，补丁分支不并 FlagGems dev-1.0。
决策依据：ADR-0010。

## 目录内容

| 文件 | 是什么 |
|---|---|
| `docker-compose.yml` | 启动与装配（demo 基础上挂载补丁版 FlagGems + 登记 mounted_repos） |
| `env.baseline.yaml` | 版本事实（baseline.py export 生成，不手工编辑） |
| `enable-gemm-compat.sh` | GEMM 修复显式启用（shell 层：环境变量 + import 路径） |
| `compat/legacy_sdnn/` | 编译组 compat 包**原样收录** + 我方加载胶水 `gemm_compat.py`（见其文件头出处说明） |

<!-- changelog:
2026-10-28 首版建立（ADR-0010）：挂载补丁版 FlagGems（xliu969/kunlun-gemm-compat @4a00b845）
+ 收录编译组 compat 包（edca42a）；K-1 编译维度解除（显式启用），数值限制 K-5 显式承载。
-->
