# kunlun-operator-runtime v2

## 用途与现状

昆仑芯 P800（XPU，x86_64）的 **flagtree + FlagGems 算子运行时底座**，可作训练与推理的推荐默认
底座：父基座已内置 vLLM 0.13.0 / transformers 4.57.1 / vllm-plugin-fl / FlagCX，本层补上基座
缺失的 flagtree（提供 `triton` 编译器与运行时），并把 FlagGems 固化为不依赖可变源码目录的
非 editable 安装。

- 镜像 tag：`flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`
- image id：见 [`image.baseline.yaml`](image.baseline.yaml) 的 `image.id`（版本事实以该文件为准，
  由 `baseline.py` 生成；本 README 不重复）
- 组件版本：同上文件的 `baked_repos` / `baked_packages` / `base_image.provides`
- `repro_status`：见 [`lock.yaml`](lock.yaml)（本层走正规 `docker build`，配方已实机重建）

**与 v1 的关系**：组件集合与 pin 完全相同，v2 只改配方形态使其落在
[`../../ENV-SPEC.md`](../../../ENV-SPEC.md) §4.2 的判定规则内、可被机器核对；三处改动与理由见
[`REBUILD.md`](REBUILD.md)「方式」。v1 保留不改并标 `superseded_by: v2`——其归档镜像实为
`docker commit` 产物、与自身配方不一致（见 [`../v1/lock.yaml`](../v1/lock.yaml)
`known_issues: v1-image-is-docker-commit-not-build`）。

**已知限制**（四条，消费前必须知晓；每条的完整证据与根因见 [`../v1/lock.yaml`](../v1/lock.yaml)
的 `known_issues`，v2 与 v1 是同一 pin 组合，这些限制原样继承）：

- GEMM（`mm`/`bmm`/`addmm`）编译期 **SIGABRT**，无可用 workaround — `gemm-compile-sigabrt`。
- 推理默认 dispatch 下 `index_select` 触发 `CUDA error: invalid device function`，**必须**设
  `VLLM_FL_PREFER=vendor` — `flagos-index-select-invalid-device-function`（demo 已内置）。
- `XPU_EVENT_KL3_ENABLE=1` 单卡单进程即**挂死**，厂商手册要求设、实测不能设 —
  `xpu-event-kl3-enable-hang`（demo 刻意不设）。
- FlagGems 在该 commit 无 `kunlunxin` extras 组，`pip install .[kunlunxin]` 静默退化并污染
  `triton` 命名空间，两层安装都**必须** `--no-deps` — `flaggems-kunlunxin-extras-missing`。

## 复现

本目录不复现缺陷，只交付可用底座；上面四条已知限制的复现方式见 [`../v1/lock.yaml`](../v1/lock.yaml)
对应 `known_issues` 条目与 [`../v1/REBUILD.md`](../v1/REBUILD.md)「验证」。

从零重建本镜像（单入口）：

```bash
dev/images/kunlun-operator-runtime/v2/build.sh <构建上下文目录>
```

前置条件见 [`REBUILD.md`](REBUILD.md)「构建前置」（基座镜像、flagtree 索引可达、宿主
`FlagGems/` 含已推送的 pin commit、`/var/lib/docker` ~15GB 余量）。

## 验证

前置条件：

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | 本目录 tag，image id 与 `image.baseline.yaml` 一致 | `build.sh` 重建，或 [`ARCHIVE.md`](ARCHIVE.md) 的离线包；由 `up.sh --strict` 机器核对 |
| 卡数 | ≥ 1 张空闲 XPU（算子验证）；多卡集合通信另见 `REBUILD.md`「GPU/机器纪律」 | `docker ps` 看本机容器，`xpu-smi` 挑 0% 利用率 / 0MiB 显存的卡 |
| 代码仓 | 无（本环境 `mounted_repos: {}`，全部为固化资产） | — |
| 模型与数据 | 本验证脚本不需要模型；训练/推理 smoke 需 Qwen3-Embedding-0.6B | `FLAGRT_MODELS_DIR` 指向模型目录 |

单入口命令（仓根执行）：

```bash
docker ps && xpu-smi                              # 共享机礼貌 + 挑空闲卡
dev/lib/up.sh dev/images/kunlun-operator-runtime/v2/demo --strict
docker exec flagrt-kunlun-v2-demo-p800 \
  env CUDA_VISIBLE_DEVICES=<上一步挑中的空闲卡号> \
  /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
  /workspace/dev/images/kunlun-operator-runtime/v2/assets/verify_kunlun_runtime.py
docker compose -f dev/images/kunlun-operator-runtime/v2/demo/docker-compose.yml down   # 用完收干净
```

脚本表：

| 脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据 |
|---|---|---|---|---|
| [`assets/verify_kunlun_runtime.py`](assets/verify_kunlun_runtime.py) | 验证 | 见上单入口（`--static` 为无设备子集） | 退出码 0 且末行 `overall: PASS`；含 flagtree/flag_gems/triton 版本、`from triton._C.libtriton import xpu`、`flag_gems` 非 editable、设备可用与显存分配、FlagGems `rms_norm`/`softmax` 与 torch 参考比对 | 2026-10-11 卡 7 **`overall: PASS`，退出码 0**（11 项通过、1 项已知缺陷不计入）：[`assets/evidence/verify-v2-card7-PASS.txt`](assets/evidence/verify-v2-card7-PASS.txt) |

判据说明：

- 三态口径（[`../CASE-TEMPLATE.md`](../../CASE-TEMPLATE.md) §2 规则 3）：`overall: PASS` / 0 = 镜像可用；
  `overall: FAIL` / 1 = 有检查项不通过；`overall: ERROR` / 2 = 无法下判定。
- **GEMM 不计入判定**：脚本在子进程里探 `mm`（否则 SIGABRT 会把验证进程一起带走），结果以
  `KNOWN` 行报告。若它反而通过了，脚本会提示去复核 `gemm-compile-sigabrt` 是否已失效。
  注意本轮实测到的失败形态（`OutOfResources: uni_sram`）与 v1 记录的 SIGABRT **不同**，
  是否同一缺陷未定论，见 [`lock.yaml`](lock.yaml)
  `known_issues: gemm-failure-signature-differs-from-v1`。
- 完整运行日志留在执行机器，仓内只放 `assets/evidence/` 截选（机器名与绝对路径见各截选文件头）。

设备模型与用卡纪律（P800 与昇腾不同，勿照抄）：实测证据
[`assets/evidence/device-model-coexistence-20261011.txt`](assets/evidence/device-model-coexistence-20261011.txt)，
完整结论见
[`demo/docker-compose.yml`](demo/docker-compose.yml) 文件头与 [`REBUILD.md`](REBUILD.md)
「GPU / 机器纪律」。
