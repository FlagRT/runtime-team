# 重建 kunlun-operator-runtime v1

## 方式

`build.sh` 驱动 `Dockerfile.repro`。产物
`flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`，
与本条目归档的镜像内容一致。

## 构建前置

| 项 | 来源 |
|---|---|
| 基座镜像 | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`（公开 harbor，本机已存在） |
| flagtree wheel | 构建期从 `resource.flagos.net/repository/flagos-pypi-hosted` 直接 pip 装（~3.3GB，见踩坑） |
| FlagGems 源码 | 基座内置 `/env/FlagGems`（已是 pin commit `73c5aff1`），构建期从该目录直接 `pip install --no-deps .` |

无需回收任何离线资产（wheelhouse/tarball）——每一层都能从公开可达源头联网安装。

## 与基座的差异

只新增两层：
1. `flagtree===0.6.1+xpu3.6`（`--no-deps` pip 安装，见下方"踩坑 1"）
2. FlagGems 从基座自带的 `/env/FlagGems`（commit `73c5aff1`）做一次干净的非 editable 重装

不改动基座已有的 torch/torchvision/torchaudio/FlagCX 等其他组件。

## 踩坑

1. **`pip install .[kunlunxin]` 在 commit `73c5aff1` 上无效且有副作用**。该
   commit 的 `pyproject.toml` 只有 `test`/`example` 两个 extras 组，没有
   `kunlunxin` 组。加 `[kunlunxin]` 不会报错中止，只打一行 WARNING 然后
   静默退化为裸装，此时 pip 会一并解析 `torch>=2.2.0` 的可选依赖，额外装
   PyPI 通用 `triton==3.5.0`，覆盖掉 flagtree 提供的 `triton/` 目录（两者
   pip 发行名不同但共享同一顶层导入命名空间，文件路径大量重叠），
   把 `triton._C.libtriton.so` 换成不含 xpu 扩展的版本，flagtree 直接不可用
   （`ImportError: cannot import name 'xpu' from 'triton._C.libtriton'`）。
   **必须 `pip install --no-deps .`**，flagtree 和 FlagGems 两层都要
   `--no-deps`（FlagGems 的其余依赖 torch/packaging/pybind11/PyYAML/sqlalchemy
   基座已全部满足，不需要额外解析）。
2. **flagtree wheel 经代理下载易 `IncompleteRead` 中断**。该 wheel
   ~3.3GB，经 `HTTP_PROXY`/`ALL_PROXY` 转发时两次复现
   `ProtocolError: Connection broken: IncompleteRead(...)`（约在
   500MB~2.7GB 处中断，非固定位置）。改为构建期不经代理直连
   `resource.flagos.net`（本机可直连，见踩坑 3）后一次性成功
   （8分5秒，21.8MB/s）。若目标机器不能直连，建议分片下载或用支持
   断点续传的下游镜像。
   **⚠️ 若在此坑上重试 `pip install --force-reinstall`**：重试期间旧文件已被
   卸载、新文件未下载完，会在两次尝试之间留下"pip show 显示已安装但
   `triton/_C/` 目录缺失"的损坏中间态——必须重试到成功为止，不能满足于
   `pip show` 不报错。
3. **本机代理策略**：`resource.flagos.net`、`pypi.tuna.tsinghua.edu.cn`、
   `github.com` 经代理与直连均可达，但大文件（>1GB）经代理不稳定；构建期
   建议对 flagtree 安装步骤单独 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY`。

## 验证（2026-09-22 实测）

### 静态

```
python3 -c "import torch, flag_gems, triton, importlib.metadata as m
print(torch.__version__, m.version('flagtree'), flag_gems.__version__, triton.__version__)"
# 2.9.0+cu129 0.6.1+xpu3.6 4.2.1.rc.0 3.6.0
```

### pip freeze 对比

与基座 `pip freeze`（`assets/provenance/GOLDEN-pipfreeze-base-official.txt`）
相比仅 2 处变化（`assets/provenance/repro-pipfreeze.txt`）：
- `flag_gems` 从 `-e git+...@73c5aff1...#egg=flag_gems` 变为
  `flag_gems @ file:///root/FlagGems`（同 commit，安装形态从 editable 变为
  非 editable 重装）
- 新增一行 `flagtree==0.6.1+xpu3.6`

其余 269 个包完全一致。按 README §5 判据（关键包版本逐一致）达 🟢。

### 真机动态验证（P800，8× XPU）

调用契约（xpu 用户手册）：
```
docker run --net=host --privileged \
  --ulimit stack=67108864 --ulimit memlock=-1 --ulimit nofile=120000 \
  --shm-size=32g --group-add video --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --device=/dev/xpuN --device=/dev/xpuctrl \
  -e CUDA_VISIBLE_DEVICES=<空闲卡号> \
  flagrt/kunlun-operator-runtime:1.0.0-... bash
```

**不要设置 `XPU_EVENT_KL3_ENABLE=1`**——单卡单进程场景下实测会挂死（见
`lock.yaml known_issues: xpu-event-kl3-enable-hang`）。设备 API 走
`torch.cuda`，不是 `torch.xpu`（xpu 用户手册约定，本机复核一致）。

验证命令（复用 FlagGems 自带测试，基座 `/env/FlagGems/tests`）：

```
cd /env/FlagGems/tests   # 或本条目安装时 pip 源目录 /root/FlagGems/tests
python3 -m pytest -q --mode quick \
  test_blas_ops.py::test_accuracy_mm \
  test_norm_ops.py::test_accuracy_rmsnorm \
  test_reduction_ops.py::test_accuracy_softmax \
  test_special_ops.py::test_apply_rotary_pos_emb
```

| 用例 | 结果 |
|---|---|
| `test_accuracy_rmsnorm` | ✅ PASSED（1 用例，2.82s） |
| `test_accuracy_softmax` | ✅ PASSED（2 用例） |
| `test_apply_rotary_pos_emb` | ✅ PASSED（768 用例） |
| `test_accuracy_mm` / `bmm` / `addmm` | ❌ **SIGABRT**（编译期崩溃，见下） |

rmsnorm/softmax/rotary 合计 **771 passed, 0 failed, 487s**（详见
`assets/provenance/verify-rmsnorm-softmax-rotary-771passed.txt`）。

mm/bmm/addmm 崩溃栈顶一致：
```
Fatal Python error: Aborted
  File ".../triton/backends/xpu/compiler.py", line 486 in make_llir
  ...
  File "/root/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/mm.py", line 168 in mm
  File ".../test_blas_ops.py", line 287 in test_accuracy_mm
```
完整 traceback 见 `assets/provenance/verify-test_accuracy_mm-SIGABRT-traceback.txt`。
这是本 pin 组合下的确定性崩溃（非偶发），与 memory 子方向此前在同 triton
3.6.0 线上的独立实测记录一致，详见 `lock.yaml known_issues: gemm-compile-sigabrt`。

### 训练 / 推理端到端 smoke

**训练**（2 卡 DDP，卡 5,7）：复用 device-context 已验证的后端无关训练腿脚本
`dev/device-context/prototype/runtime/proto/proto_train_leg.py`
（`DC_BACKEND=kunlun`、`FLAGCX_ADAPTOR=klx`、`dist_backend=cpu:gloo,cuda:flagcx`），
模型 Qwen3-Embedding-0.6B，`MAX_STEPS=20`，不设 `XPU_EVENT_KL3_ENABLE`。
`TRAIN_LEG_PASS 6/6`，loss 15.4488→11.2841（20 步），1713.1 tok/s（两卡合计）。
详见 `lock.yaml repro_result.e2e_smoke_training`。

**推理**（单卡，卡 7）：`vllm serve` + `--runner pooling --convert embed`，
模型同上，**须设 `VLLM_FL_PREFER=vendor`**（默认 dispatch 下 `index_select`
经 FlagGems 路径触发 `CUDA error: invalid device function`，见 `lock.yaml
known_issues: flagos-index-select-invalid-device-function`）。设置后服务正常
启动，`/v1/embeddings` 请求成功返回 1024 维向量。详见 `lock.yaml
repro_result.e2e_smoke_inference`。

## GPU/机器纪律

共享机上验证前须用 `xpu-smi` 确认目标卡空闲（0% 利用率、0MiB 显存占用）再用
`CUDA_VISIBLE_DEVICES` 指定；不得 stop/rm/exec 进入任何非本人创建的容器。

**训练场景（多卡集合通信）额外要求**：优先选**同一 NUMA 组内连续且空闲**的卡
（本机 NUMA0=卡0-3、NUMA1=卡4-7）。**卡 1 历史上反复被共享机其他租户占用**，
即便临跑前 `xpu-smi` 显示瞬时空闲，训练场景也应优先避开——用卡 1,2 复现过
集合通信初始化挂死（`assets/provenance/train-smoke-flagcx-initcomm-hang.txt`），
换到同 NUMA 组的卡 5,7 后训练即正常通过。单卡场景（pytest、推理）不受此限制，
本条目单卡验证用的是卡 4/7。
