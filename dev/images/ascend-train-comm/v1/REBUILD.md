# 重建 ascend-train-comm v1

## 方式

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`。父层 `flagrt/ascend-operator-runtime:0.2.0-reproB` 需先建（见 `../../ascend-operator-runtime/v1/REBUILD.md`）。产物 `flagrt/ascend-operator-runtime-comm:0.1.3-reproB`。

## 构建上下文需备齐

| 项 | 来源 |
|---|---|
| `Dockerfile` | = 本目录 `Dockerfile.repro`（build.sh 会拷） |
| `flagcx-vendor/flagcx/` + `flagcx-vendor/flagcx-0.13.0.dist-info/` | 从原镜像 site-packages 回收的已装包（`.so` 与原镜像逐字节一致）；校验 `assets/provenance/flagcx-vendor.sha256`；实体见 `ARCHIVE.md` |
| `verify_flagcx_runtime.py` / `verify_flagcx_p2p.py` | 本目录 `assets/` |

## 与原镜像的差异

- FlagCX：原镜像从 `flagcx-0.13.0-*.whl`（commit `55eb2ff` + 2 patch，均 owner 私有）装；此处 vendor 从原镜像回收的已装包 —— `_C.so` / `libflagcx.so` / `__init__.py` sha256 逐字节一致，仅 pip 元数据来源不同。

## 验证（2026-09-10 实测）

| 判据 | 结果 |
|---|---|
| `pip freeze` vs 原镜像 | **逐行一致**（63 包） |
| flagcx `_C.so` / `libflagcx.so` / `__init__.py` sha256 | **与原镜像逐字节一致** |
| `importlib.metadata.version('flagcx')` | `0.13.0` |
| `verify_flagcx_runtime.py --static` | 运行；裸容器下 `libascend_hal.so` 缺失非致命，须带 `--device` 验证 |
| 用时 | 3s（父层复用） |

基线文件：`assets/provenance/GOLDEN-pipfreeze-comm.txt`、`GOLDEN-verify_flagcx_runtime.json`、`reproB-pipfreeze.txt`、`reproB-build-tail.txt`。
