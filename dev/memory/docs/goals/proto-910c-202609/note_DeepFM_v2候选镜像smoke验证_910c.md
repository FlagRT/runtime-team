# DeepFM · 910C 单卡 smoke 验证（训练腿 v2 候选镜像）

> 日期:2026-09-21 ｜ 执行人:xliu969(agent 代跑) ｜ 机器:npu1-27,16× Ascend910C(仅占用 davinci0 一张卡,单进程)
> 镜像:`flagrt/ascend-operator-runtime-comm:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`
> （image id `17152831f09d`，训练腿 **v2 候选镜像**，血统见 `dev/images/ascend-train-comm/v2/lock.yaml`；
> **不是** `dev/stack.lock.910c.v2.yaml` → `lock.train.image` 当前锁定的正式训练镜像——那个是 v1
> `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`）
> 被测对象:`dev/model-defs/deepfm/v1/`（`build.py:smoke_test()`）

## 结论速览

🟢 **DeepFM 参考模型在 910C 真机（v2 候选镜像，单卡 davinci0）上 build + 前向 + 反向全部跑通，`deepfm/v1/` 目录零代码改动**。

- `device=npu`，421,452 参数（与 CPU 验证一致的默认 schema `field_dims=[1000,2000,500,300,50]`）
- 3 步前向反向全部 PASS，loss 有限、无 NaN/Inf
- 只有一条无害 `UserWarning`（`torch_npu` 内部 tensor 格式提示，非错误，不影响结果）
- **不需要**改 `model.py`/`build.py`/`config.py` 任何一行——原生 `import torch_npu` 外部 runner + `pick_device()` 的 `hasattr(torch, "npu")` 探测机制按设计直接生效

**本记录的定位（重要,避免误读)**：
这只是验证"DeepFM 能在 v2 候选镜像上跑通"这一件事的一条数据点，**不改动**
`dev/stack.lock.910c.v2.yaml` 的 `lock.train`，也**不代表** v2 候选镜像已被
总组转正为官方训练锁定镜像——v2 是否取代 v1 成为正式 `lock.train.image`
是总组决策（`dev/images/ascend-train-comm/v2/lock.yaml` 明确 `superseded_by: null`，
两条血统并存），本记录只是为该决策提供输入证据。

## 复现命令

容器按共享机并发规则（同时最多 3 个挂 NPU 设备的容器，见
`dev/stack.lock.910c.v2.yaml` 规则与 `Runtime工作指导-镜像容器与跨组协作-20260814.md` §2.2）
启动，仅挂 1 张卡（davinci0），跑完立即清理：

```bash
docker run -d --name flagos-devfm-v2cand-910c \
  --ipc=host --shm-size=8g --ulimit memlock=-1 --cap-add=SYS_PTRACE \
  --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /home/xliu969/runtime-team:/workspace/runtime-team \
  flagrt/ascend-operator-runtime-comm:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 \
  sleep infinity

# 版本/可用性检查
docker exec flagos-devfm-v2cand-910c bash -lc '
python3 -c "
import torch, torch_npu
print(\"torch:\", torch.__version__)
print(\"torch_npu:\", torch_npu.__version__)
print(\"is_available:\", torch.npu.is_available())
print(\"device_count:\", torch.npu.device_count())
"'

# smoke test —— DeepFM 目录本身零厂商依赖（design 见 deepfm/v1/README.md），
# NPU 探测靠 build.py:pick_device() 的 hasattr(torch, "npu") 探测，
# 所以由外部 runner 先 import torch_npu 再调用 smoke_test()，
# glue 代码就是下面这一行内联 -c，不落盘进 deepfm/v1/ 目录：
docker exec flagos-devfm-v2cand-910c bash -lc '
cd /workspace/runtime-team
python3 -c "import torch_npu, sys; sys.path.insert(0,\"dev/model-defs/deepfm/v1\"); import build; build.smoke_test()"
'

# 清理，释放并发名额
docker stop flagos-devfm-v2cand-910c && docker rm flagos-devfm-v2cand-910c
```

## 证据 / output

版本与可用性检查：

```
=== python ===
Python 3.11.15
=== pip list (torch related) ===
flagcx                                   0.13.0
numpy                                    1.26.4
torch                                    2.10.0+cpu
torch_fl                                 0.1.0
torch_npu                                2.10.0
torchaudio                               2.10.0+cpu
torchvision                              0.25.0+cpu
=== torch_npu availability ===
torch: 2.10.0+cpu
torch_npu: 2.10.0
is_available: True
device_count: 1
```

（`device_count: 1` 符合预期——容器只挂了 `davinci0` 一张卡，未过量占用共享机资源。）

smoke test 完整输出（一次成功，无需重试、无需改代码）：

```
/usr/local/python3.11.15/lib/python3.11/site-packages/torch/autograd/__init__.py:230: UserWarning: Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format. (Triggered internally at ../torch_npu/csrc/aten/common/TensorFactories.cpp:340.)
  torch.ones_like(out, memory_format=torch.preserve_format)
[smoke_test] device=npu field_dims=[1000, 2000, 500, 300, 50] embed_dim=16 mlp_dims=(400, 400, 400) params=421452
[smoke_test] step=0 loss=1.2047
[smoke_test] step=1 loss=0.8844
[smoke_test] step=2 loss=1.1957
[smoke_test] PASS
```

## 容器与并发核对

- 启动前 `docker ps`：`x-benchmark`、`flaggems-cann9.0.0` 两个队友容器已在跑（均挂满 16 卡，但 AICore/HBM 占用为 0，未在实际使用），总数 2。
- 本次新增 `flagos-devfm-v2cand-910c`，仅挂 `davinci0` 一张卡，总运行容器数 3，未超过并发上限。
- 未触碰、未重启任何队友容器。
- 跑完立即 `docker stop && docker rm`，`docker ps` 复核后恢复为仅剩 `x-benchmark`、`flaggems-cann9.0.0` 两个原有容器。

## 未改动的文件（明确记录，避免误读）

- 未修改 `dev/model-defs/deepfm/v1/` 下任何文件（`model.py` / `build.py` / `config.py` 均零改动，`import torch_npu` 完全在外部 runner 一行内联命令里完成，符合目录零厂商依赖设计）。
- 未修改 `dev/stack.lock.910c.v2.yaml`、`dev/images/ascend-train-comm/v2/lock.yaml` 或任何 `dev/images/*/lock.yaml`。
