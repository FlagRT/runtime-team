# 逐步操作：从零复现昆仑芯 GEMM 编译失败（1 卡）

面向第一次在 P800 机器上操作的人。全部命令在**公共仓根**（本机：`/data2/xliu969/code/runtime-team`，
`~/runtime-team` 是指向它的软链）执行，按序复制即可。结论口径见 [`README.md`](README.md)「修复验收」。

## 0. 共享机礼貌 + 挑一张空闲卡

```bash
docker ps
xpu-smi
```

**P800 的设备模型与昇腾相反，不要照抄昇腾话术**：

- **没有「带卡容器独占」**。实测 9 个容器并存、同一张卡被多个容器同时映射仍可正常用
  （证据：[`../../images/kunlun-operator-runtime/v2/assets/evidence/device-model-coexistence-20261011.txt`](../../images/kunlun-operator-runtime/v2/assets/evidence/device-model-coexistence-20261011.txt)）。
  所以 `docker ps` 在这里**只是礼貌**（看清机器上有谁在跑、不要动他人容器），不是阻塞条件。
- 真实约束是**显存与算力竞争**，驱动不做强制，靠自律：从 `xpu-smi` 里挑一张
  **0% 利用率 / 0MiB 显存**的卡，记下它的物理卡号。
- **绝不**停止 / 重启 / `exec` 进入任何他人容器。

设备模型四条完整结论写在
[`demo/docker-compose.yml`](../../images/kunlun-operator-runtime/v2/demo/docker-compose.yml) 文件头。

## 1. 启动运行环境

```bash
FLAGRT_MODELS_DIR=<任一实体目录> dev/lib/up.sh dev/images/kunlun-operator-runtime/v2/demo --strict
```

- `FLAGRT_MODELS_DIR` **必须给**：demo 把它挂为容器内 `/workspace/models:ro`，缺省值
  `<仓根>/models` 在本机是指向宿主个人目录的软链，容器内无法解析，不给值容器起不来。
  本缺陷的最小复现不读模型，挂载点有效即可。
- `--strict`：启动前按镜像基线核对 image id，不一致即拒绝启动。预期输出含
  `[PASS] image: flagrt/kunlun-operator-runtime:2.0.0-...`。
- `up.sh` 会自动导出 compose 需要的 `WORKSPACE_ROOT`，这一步不用自己设。

```bash
docker ps --filter name=flagrt-kunlun-v2-demo-p800   # 应看到状态 Up
```

## 2. 复现缺陷

```bash
docker exec flagrt-kunlun-v2-demo-p800 \
  env CUDA_VISIBLE_DEVICES=<第 0 步挑中的物理卡号> \
  /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
  /workspace/dev/issues/kunlunxin-gemm-triton-compile/assets/minirepro_gemm.py
echo "exit=$?"
```

两个容易踩的点：

- **解释器必须写绝对路径**。`PATH` 上的 `python3` 是 miniconda base 的 **3.13.11**，
  不是目标 env 的 **3.10.18**，用错的那个 import 不到 `flag_gems`。
- **`CUDA_VISIBLE_DEVICES` 写物理卡号**，仅因为本 demo 映射了全部 8 张卡、容器内逻辑号
  == 宿主物理号。**若你自建只授予卡的子集的环境**，设备会按授予顺序从 0 密集重编号，
  必须按重编号后的逻辑号写（只授予 `/dev/xpu7` 时它在容器内是 0 号；写 7 会得到
  `device_count() == 0`）。

预期：退出码 1、末行 `overall: FAIL`、12 组全部失败（2 组 `native-abort@make_llir`，
10 组 `OutOfResources:uni_sram@make_ttxir`）。判定为 `overall: ERROR` / 退出码 2 时说明环境
不满足（脚本会打印 preflight 读数），**不是**缺陷复现，先把环境修好再下结论。

## 3. 收尾

```bash
docker compose -f dev/images/kunlun-operator-runtime/v2/demo/docker-compose.yml down
```

绕开 `up.sh` 直接 `docker compose` 时需自带环境变量，否则会报
`invalid spec: :/workspace: empty section between colons`：

```bash
WORKSPACE_ROOT=$PWD FLAGRT_MODELS_DIR=<同第 1 步> \
  docker compose -f dev/images/kunlun-operator-runtime/v2/demo/docker-compose.yml down
```

```bash
docker ps    # 确认自己起的容器已不在；他人容器应原样在跑
```

## 4. 可选：用 FlagGems 自带的 pytest 用例复跑

镜像内 `/env/FlagGems` 是基座自带那份（`install: editable-in-image`），**带 tests 也带 `.git`**，
`git rev-parse HEAD` 即 pin commit `73c5aff1e304c293ab525ecb03e9449644496fbb`，不需另取测试文件：

```bash
docker exec flagrt-kunlun-v2-demo-p800 \
  env CUDA_VISIBLE_DEVICES=<空闲卡号> \
  /root/miniconda/envs/python310_torch29_cuda/bin/python3 -m pytest -x -s \
  /env/FlagGems/tests/test_blas_ops.py::test_accuracy_mm
```

**`-s` 不能省**：不加它（capture 生效）时进程在 pytest 写出报告前就 abort 了，被捕获的输出
随之丢弃，落盘是 **0 字节 + rc=1**，看起来像「什么都没发生」。实测同一用例 `-s` 三次各
880 字节含 LLVM 断言、`-q` 三次 0 字节。另外**不要**设 `PYTHONFAULTHANDLER=1` —— 实测它反而
会压掉 faulthandler 的 Python 栈 dump，只剩 LLVM 断言那一行。

该用例在 normal 模式下是 3 形状 × 3 dtype × `b_column_major` 两值 = 18 组，`-x` 会停在第一组
（即 `(1,1,32)` + float16，原生 abort）。要逐组看形态须每组一个独立进程 —— 逐组读数与 48 帧
完整栈见
[`../../images/kunlun-operator-runtime/v2/assets/evidence/gemm-failure-signature-recheck-20261012.txt`](../../images/kunlun-operator-runtime/v2/assets/evidence/gemm-failure-signature-recheck-20261012.txt)。
