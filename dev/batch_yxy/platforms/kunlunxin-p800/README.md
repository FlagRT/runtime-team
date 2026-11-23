# 昆仑芯 P800 平台（机器：P800-A）

> 本目录 = P800 平台的全部开发信息。**进度与待办看 [`progress.md`](progress.md)**，
> 缺陷排查与报告看 [`investigation/`](investigation/)，测试数据看 [`data/`](data/)，
> 原始过程资产（脚本+日志）看 [`raw/`](raw/)。

## 环境事实（2026-11-23 固化）

| 项 | 值 |
| --- | --- |
| 机器 | P800-A：内网 10.20.20.4（VM-20-4-ubuntu），SSH `xianyiyuan@81.70.153.29 -p 26008` |
| 硬件 | 昆仑芯 P800 OAM ×8（96GB/卡），宿主驱动 5.0.21.47 |
| **配额卡** | **宿主 2、3、7 号**（空闲时可用；启动前 `xpu-smi` 确认 0%/0MiB） |
| 推理镜像 | `harbor.baai.ac.cn/flagrelease-public/kunlunxin001-gems5.0.0-...-vllm0.20.2-cp310-pt29-x64-v5.0.21.43:tele4`（= USAGE_MATRIX `kunlun-infer-vllm/v1`，digest `e180a0dd…`，本机 docker 已有） |
| 推理栈 | vllm 0.20.2+empty / torch 2.9.0+cu129 / **vllm-plugin-FL dev-1.0（`039fb46`，含 pooling 修复）**；设备 API 走 torch.cuda 符号重写 |
| 备用栈 | `chenyunlong/vllm-kunlun-runtime:0.11.0`（vllm 0.11 + vllm-kunlun 0.11）——历史实验用，缺陷结论见 investigation/，**新开发不用** |
| 模型 | Qwen3-Embedding-0.6B：`/data1/flagperf-kzhang519/models/Qwen3-Embedding-0.6B-modelscope/`（sha256 已校验）；**需剔除 st 配置的符号链接视图**（见 BOOTSTRAP） |

## 容器配方（推理腿，对齐 USAGE_MATRIX）

```bash
# 插件修复版：git clone -b dev-1.0 git@github.com:FlagRT/vllm-plugin-FL.git
docker run -d --name <name> --network none \
  --device /dev/xpu2 --device /dev/xpuctrl \
  -v ~/fgw:/work \
  -v ~/fgw/vllm-plugin-FL:/workspace/vllm-plugin-FL \
  -v /data1/flagperf-kzhang519/models/Qwen3-Embedding-0.6B-modelscope:/data1/flagperf-kzhang519/models/Qwen3-Embedding-0.6B-modelscope:ro \
  <tele4 镜像> sleep infinity
# 容器内运行前必须 shell export（vllm spawn 子进程不继承 os.environ）：
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
```

镜像内打包的 vllm-plugin-FL（0.2.0+g38e7dbc）**早于 pooling 修复**，必须用
dev-1.0（`039fb46`）挂载覆盖 `/workspace/vllm-plugin-FL`（editable 安装，挂载即生效），
否则 embed/pooling 必崩（见 `../../dev/issues/vllm-plugin-fl-pooling-dispatch/`）。

## 已知平台事故（影响排班）

1. **2026-11-15 驱动全局卡死**（新上下文全挂、旧进程正常），管理员修复
2. **2026-11-18 复发**：触发源为 bge-reranker-base（XLMRoberta）在
   vllm-kunlun 0.11 栈加载时 glibc 堆损坏崩溃 → 毒化驱动。**bge 系模型
   在本机一律先小规模试探，勿直接全量加载**
