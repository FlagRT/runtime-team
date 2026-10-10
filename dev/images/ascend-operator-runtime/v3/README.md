# ascend-operator-runtime v3

## 用途与现状

- 本目录构建基座镜像
  `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`
  (npu1-11 image id `62bcaae3f839`):PyTorch 2.10 + torch_npu(Route A 默认)+
  FlagTree Triton 3.5.1 + FlagGems + vLLM 0.20.2 + vllm-plugin-FL。基座为内部构建层,
  照常构建并归档,**不作为发布物**。
- 对外发布物为其超集完整镜像(推理 + 训练),见 [`../../ascend-train-comm/v3/`](../../ascend-train-comm/v3/README.md)。
  本目录的运行层 `docker-compose.routeA.yml` 与 `pins.routeA.yaml` 声明的即为完整镜像,
  推理验证在完整镜像上执行。
- `repro_status`:🟡 partial-repro。1 卡推理验证 9/9 PASS,前提为三项运行时配置
  (见「复现」);升 🟢 条件:shmem 缺失由算子编译方向根治。
- 已知限制:

| 问题 | 归属 | 详情 |
|---|---|---|
| Triton tle 子模块无条件 `import shmem`,该模块不在镜像内;任何含 for 循环的 Triton kernel 编译失败(含 vLLM 核心 kernel) | 算子编译方向 | `lock.yaml` known_issues `flag-gems-triton-ascend-missing-shmem-module`;需求文本:[决策备忘录附录 B](../../../../docs/v3-决策备忘录与shmem跨组需求-20261008.md) |
| `assets/v3_step5_validate.py` 与 vLLM 0.20.2 有三处接口不兼容(pooling_task 必传 / PoolingOutput 无 `.embedding` / 多进程引擎下父进程读显存为 0) | 本组(待修) | `REBUILD.md`「ROUND 4 真机验证——推理腿」第 4 条;round 4 的 9/9 结果由运行时修补后的脚本得出 |

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` | 按 `REBUILD.md` + `build.sh` 构建(先构建本目录基座,再构建 `../../ascend-train-comm/v3/`);或从 `ARCHIVE.md` 所列离线包 `docker load` |
| 卡 | 1 × 910C | 启动前 `docker ps` 确认本机无其他带卡容器(昇腾驱动对带卡容器独占,第二个带卡容器启动报 EBUSY) |
| 宿主代码仓 | `pins.routeA.yaml` 声明的 commit | `dev/lib/up.sh` 启动时自动校验,不一致则拒绝启动 |
| 模型 | Qwen3-Embedding-0.6B | 放置于 `<仓根>/models/Qwen3-Embedding-0.6B`(实体目录,容器内路径 `/workspace/models/Qwen3-Embedding-0.6B`);npu1-11 现有副本:`/home/xliu969/models/Qwen3-Embedding-0.6B` |

### 启动容器(仓根执行)

```bash
docker ps
dev/lib/up.sh \
  dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml \
  dev/images/ascend-operator-runtime/v3/docker-compose.flaggems-cos-off.yml
```

### 复现 shmem 缺失(不加规避配置)

```bash
docker exec -w /workspace \
  -e VLLM_FL_FLAGOS_BLACKLIST= -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e SMOKE_MODEL_PATH=/workspace/models/Qwen3-Embedding-0.6B \
  flagos-v3-routeA-dev-910c \
  python3 dev/images/ascend-operator-runtime/v3/assets/v3_step5_validate.py
```

预期结果:`[STEP] vllm_encode_real_output: FAIL`,报错为
`CompilationError ... ModuleNotFoundError("No module named 'shmem'")`,触发点为模型加载阶段
`freqs.cos()` → `flag_gems.ops.cos` 的 grid-stride-loop kernel;退出码 1。

### 带规避配置运行(当前可用形态)

三项运行时配置:`VLLM_FL_FLAGOS_BLACKLIST=cos,sin`(由 `docker-compose.flaggems-cos-off.yml`
设置)、shmem 占位模块(`assets/shmem_stub.py`,以模块名 `shmem` 置于 `PYTHONPATH`)、
`VLLM_ENABLE_V1_MULTIPROCESSING=0`(可选,进程内引擎,便于读取显存)。

```bash
docker exec flagos-v3-routeA-dev-910c \
  cp /workspace/dev/images/ascend-operator-runtime/v3/assets/shmem_stub.py /tmp/shmem.py
docker exec -w /workspace \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e SMOKE_MODEL_PATH=/workspace/models/Qwen3-Embedding-0.6B \
  flagos-v3-routeA-dev-910c \
  bash -c 'PYTHONPATH=/tmp:$PYTHONPATH python3 dev/images/ascend-operator-runtime/v3/assets/v3_step5_validate.py'
```

`PYTHONPATH` 须追加而非覆盖:镜像内既有 `PYTHONPATH` 含 CANN 路径。占位模块的全部函数
被调用即报错,单卡推理不调用,不改变计算结果(对照证据见 `REBUILD.md`「ROUND 4 真机验证
——推理腿」第 2 条)。

## 验证

| 脚本 | 类别 | 命令(容器内,`/workspace` 下) | 通过判据 | 最近一次结果 |
|---|---|---|---|---|
| `assets/verify_runtime.py` | 验证 | `python3 /opt/flagrt/verify_runtime.py --static`(`build.sh` 构建后自动执行) | 退出码 0 | round 4 通过(`lock.yaml` `verify_runtime_static`) |
| `assets/verify_runtime.py` | 验证 | `python3 /opt/flagrt/verify_runtime.py`(需带卡) | 退出码 0,`route_a_default_check: passed` | round 1 通过,结论沿用至今(`lock.yaml` `verify_runtime_dynamic_route_a_default`) |
| `assets/v3_step5_validate.py` | 验证 | 见「带规避配置运行」 | 退出码 0;9 个 `[STEP]` 全部 OK;`embedding_dim=1024`、两段文本向量范数≈1.0、`identical=false`、无 NaN/全零、NPU 显存 > 0 | round 4 9/9 PASS(npu1-11:`~/tmp-reproV3-r4/infer-leg/round-5.log`,用户 xliu969) |
| `assets/provenance/pipfreeze-*.txt` | 收数 | `build.sh` 构建后生成 | — | round 4 npu1-11:`pipfreeze-operator-runtime-v3-round4-npu11.txt` |

构建输入类文件(不属于上述三类):`assets/patch_triton_ascend_flagtree.py`、
`assets/FlagGems-DSA-__init__.py`、`assets/patches/vllm-plugin-FL/`(见其 README)、
`assets/shmem_stub.py`(运行时注入,不进镜像)。

技术细节与完整排查过程:`REBUILD.md`、`lock.yaml`;round 4 结论摘要:
[`docs/v3-round4-真机验证结果与移交指导-20261009.md`](../../../../docs/v3-round4-真机验证结果与移交指导-20261009.md)。
