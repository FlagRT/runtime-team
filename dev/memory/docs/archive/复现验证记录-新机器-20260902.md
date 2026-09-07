# 执行记录:新机器复现验证(环境重建 + 分配器层画像)

> 日期:2026-09-02 ｜ 执行人:xliu969 ｜ 环境:新机器(openEuler 22.03 SP4 aarch64 / 16× Ascend910C 64GiB / CANN 9.0.0 / docker)
> 目的:验证 memory 子方向已有成果在新机器上可稳定复现 ｜ 结论:🟢 **分配器层 5/5 全绿,与 2026-08-17 画像语义一致**;🔴 推理闭环/V1 画像受 HBM 占用阻塞(见 §4)

## 1. 结论摘要

1. **torch_fl 环境在新机器重建成功**(harbor 镜像 + py312 venv + ACCELERATOR=ascend 源码编译),import/device_count=16/分配器接口全部就位。
2. **分配器画像(显存池第一层)复现通过**:缓存复用、同尺寸池命中、大块切分、OOM 重试 5 项全绿,行为与 [allocator-画像报告-20260817](allocator-画像报告-20260817.md) 一致。碎片冗余绝对值因块尺寸缩放(见 §3)不可直接对比,语义判定不受影响。
3. **vLLM 推理链路组装完成(2026-09-02 追加)**:venv312 内 torch_fl + triton_ascend 3.2.2 + vllm 0.20.2 + flag_gems + vllm-plugin-FL + flagcx 0.13.0 全通;import 链、PlatformFL(device_type=npu / dist_backend=flagcx)、shim、子线程分配、scatter/split fallback 全部验证;flagcx 双卡 allreduce 数值正确。
4. **推理闭环(Qwen3-4B 单卡/TP)本次未复现**:16 卡 HBM 均被 `sglang-pd-decode2`(同事负载)占用 ~57/64GiB,仅余 ~8.5GiB/卡;Qwen3-4B 推理需 ~32GiB。模型已从旧机 rsync 到位(14 文件 7.6G,safetensors 398 权重可加载)。待卡空闲后按 §5 继续。

## 2. 新机器环境重建(与旧机配方的差异)

| 项 | 旧机(文档配方) | 新机(本次实测) |
|---|---|---|
| 开发镜像 | 本地构建 `flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet` | 无本地镜像 → **harbor 匿名拉取** `harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev`(11.2GB)后重打同名 tag(内容同为 CANN 9.0.0 + Python 3.12 + 工具链) |
| 容器启动 | docker compose v2(-f 合并) | **无 compose 插件** → 等价 `docker run`(参数照抄 compose.base.yml + memory/docker-compose.yml;entrypoint 需追加 `bash` 保活) |
| Python/venv | py3.11.15(拷自 vllm-ascend v0.20.2rc1-a3)+ venv311 | **py3.12.13**(镜像自带)+ `/root/vllm-venv312`;分配器层不需要 triton_ascend,vllm 链路续接时用 nightly 镜像的 triton_ascend 3.2.2(cp312)替代文档的 3.2.1(cp311) |
| torch_fl 编译 | `ACCELERATOR=ascend pip install -e .`(文档配方) | 当前 **main 分支需加 `FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=0`**:ascend 分支未默认关闭 FlagGems(CMakeLists.txt:30 默认 ON),与 [installation.md](../../PyTorch-Plugin-FL/docs/vendors/ascend/installation.md) 的"默认纯 ACLNN"描述存在漂移——已反馈点 |
| 模型 | /workspace/models/Qwen3-4B(旧机宿主盘) | **未下载**(需 ModelScope,7.6GB) |

## 3. 验证结果(探针:dev/memory/probes/probe_allocator_profile.py)

> 探针原脚本未入库(报告引用时"待入库"),本会话按报告测试矩阵重建。HBM 空闲 8464MiB → 块基准自动缩放至 464MiB(自适应,语义判定与块大小无关)。

| # | 场景 | 期望 | 实测 | 判定 |
|---|---|---|---|---|
| 1 | 缓存复用:10 次同尺寸 alloc/free | 1 次 device malloc | 1 | ✅ |
| 2 | 交错释放:A/B/C(2G)→del B→D(2G) | D 0 次新 malloc | 0 | ✅ |
| 3 | 尺寸抖动:20 次随机,半保留半释放 | 碎片可控 | 0.31GiB / 13.9%(缩放样本;报告原值 0.54GiB/4.9%) | ✅(<15%) |
| 4 | 大块切分:8G→free→8×512MiB | 0 次新 malloc | 0 | ✅ |
| 5 | OOM 重试 | num_alloc_retries=0 | 0 | ✅ |

接口核对:FLAGOS_USE_CACHING_ALLOCATOR 默认开启语义未变(caching_device_allocator.cc:39-48);empty_cache/memory_stats/memory_allocated/memory_reserved/reset_peak_memory_stats 全部可用(torch_fl/flagos/__init__.py:212-269)。

## 4. 阻塞项(非环境问题,资源占用)

- `sglang-pd-decode2` 容器(同事负载)在全部 16 卡持有 ~53.9GB/卡,各卡 HBM 余量仅 ~8.5GiB。
- Qwen3-4B 单卡推理需 ~32GiB(加载 31.89GiB),TP 需要更多 → **推理闭环与 V1 画像复现须等该容器释放资源**。
- 新机器无 docker compose 插件、无本地 flagos-dev 镜像、无模型权重 —— 均已找到替代路径或已解决,不构成环境阻断。

## 4b. vLLM 链路组装要点(2026-09-02,与旧机配方的差异)

| 组件 | 做法 |
|---|---|
| triton_ascend | nightly vllm-ascend 镜像内 **3.2.2(cp312)**,docker cp 拷贝(doc 配方是 3.2.1/cp311);依赖 numpy==1.26.4/attrs/decorator/psutil/pybind11 需按 METADATA 装齐 |
| vllm 0.20.2 | `--no-deps` 装 wheel 后按 Requires-Dist 补依赖(避开 torch==2.11.0 pin 与官方 triton 覆盖);llguidance 需 <1.4.0、lark==1.2.2、compressed-tensors==0.15.0.1,numpy 会被顶回 2.x 需再压回 1.26.4 |
| flagcx 0.13.0 | 上游 main 的 ascend 分支**构建依赖 torch_npu 且缺 4108B rootinfo 修复** → 用 FlagRT fork 分支 `xliu969/flagcx-ascend-fix`(decouple torch_npu,直接编 CANN)叠加 5f7ad78 的 rootinfo 修复 + 本机 stream slot 修复(见 §4c);构建需 `FLAGCX_ADAPTOR=ascend` + 子模块 init + safe.directory |
| flagos_boot.py | 重建(dev/memory/probes/flagos_boot.py,已拷入 venv):npu/cann→flagos 别名(mode+device 包装)、Thread.start 继承 mode 栈、torch.npu 补 empty_cache/mem_get_info/max_memory_allocated/stream、torch_npu 补 _inductor/NPUGraph/_C/version/__file__、scatter_ 系列与 split_with_sizes CPU fallback、torch.accelerator 重定向、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`(防 flagcx 后端自动加载毒化进程) |
| torch_fl 源码补丁 | `csrc/runtime/allocator/backends/ascend_memory.h` get_device_index 失败回退 0(文档挂点9,main 未含;工作区未提交,重编后已生效) |
| 构建期 torch_npu stub | 上游 main 需要 torch_npu 头/库;fix 分支解耦后不再需要。venv 内 torch_npu 包保留(空 stub libtorch_npu.so 已被 fix 分支构建忽略;真实库不可用——静态初始化与 torch_fl 的 PrivateUse1 fallback 冲突,实测 abort) |

## 4c. flagcx 双卡 allreduce 验证(2026-09-02)

- 脚本:dev/memory/probes/flagcx_smoke.py + hccl_direct.py(纯 ctypes HCCL 对照)
- 结果:2 进程各绑 1 卡(ASCEND_RT_VISIBLE_DEVICES 单卡,TP 语义),torch.distributed flagcx 后端 init + allreduce `[1]+[2]→[3]` 数值正确
- 三个坑(均已解):
  1. **rootinfo 溢出**:HcclRootInfo 4108B vs flagcxUniqueId 256B,上游 main 的 hccl adaptor 会把 4108B 写进 256B 缓冲 → HCCL_E_PARA;用 5f7ad78 修复(thread_local + bootstrapCollBroadcast 全量分发)
  2. **双进程同卡**:smoke 初版两进程都设 `ASCEND_RT_VISIBLE_DEVICES=0,1` 且用逻辑设备 0 → 都绑 phy0;须每进程单卡可见
  3. **析构崩溃**:getStreamByIndex 把 `&acl_stream`(成员地址)当 flagcxStream 存,析构 streamDestroy `free()` 对象内部指针 → abort;改堆上 slot + ascend 析构只释放 slot 不销毁宿主共享流
- 遗留:flagcx 后端 collective **异步返回**,调用方需 `torch.npu.synchronize()`(与阶段4 根因2 一致,冒烟脚本已含)

## 5. 续接命令(卡空闲后)

```bash
# 容器已就绪: flagos-fl-dev-910c(host 网络,16 卡挂载,venv312 在 /root/vllm-venv312)
docker exec -it flagos-fl-dev-910c bash

# 1) 模型(容器内,host 网络)
/root/vllm-venv312/bin/pip install -q modelscope
modelscope download --model Qwen/Qwen3-4B --local_dir /workspace/models/Qwen3-4B

# 2) vllm 链路(继续 venv312 组装,替代文档 venv311 配方):
#    torch 2.10.0+cpu 已装; 补 vllm==0.20.2(--no-deps)+依赖、triton_ascend 3.2.2
#    (docker cp 自 quay.io/ascend/vllm-ascend:nightly-main-a3,cp312)、
#    flag_gems/vllm-plugin-FL editable、flagos_boot.py 引导(挂点清单见阶段4执行记录 §2)
# 3) 单卡闭环 → TP=2/4 数值复现
```

## 6. 产物

- 容器 `flagos-fl-dev-910c`(运行中)+ venv `/root/vllm-venv312`(torch 2.10.0+cpu + torch_fl 0.1.0 编译版 + triton_ascend 3.2.2 + vllm 0.20.2 + flag_gems/vllm-plugin-FL editable + flagcx 0.13.0)
- 模型 `/home/xliu969/runtime-team/models/Qwen3-4B`(14 文件 7.6G,容器内 `/workspace/models/Qwen3-4B`;models/ 已 gitignore)
- 探针与脚本(dev/memory/probes/,未提交):probe_allocator_profile.py(分配器画像)、flagos_boot.py(venv 引导,已拷入 site-packages + flagos_torchfl.pth)、flagcx_smoke.py(双卡 allreduce 冒烟)、hccl_direct.py(纯 ctypes HCCL 对照)、c10_npu_shim.cpp(已被 fix 分支取代,留档)
- FlagCX 本地分支 `local-ascend-fix` + 提交 `153cdfd`(rootinfo/stream slot 修复,未 push)
- torch_fl 工作区补丁:ascend_memory.h get_device_index fallback(未提交)
- torch_fl 构建日志:容器内 /tmp/torchfl_build*.log、flagcx:/tmp/flagcx_build.log
