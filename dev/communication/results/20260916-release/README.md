# 本轮实机结果与复现入口

[summary.json](summary.json)仅含允许发布的实验数值、失败计数、分级结果和动态库指纹；
原始日志、模型路径、服务器标识和容器信息不发布。每条记录包含原始文件 SHA-256，
由所有者的本地证据目录核对。主机测试不并入设备结果。

## 结果索引

| source 前缀 | 含义 |
| --- | --- |
| baseline | 原库旧探针，含 1 次 P2P 数值失败 |
| two_patches | 仅两项历史补丁，数值失败且退出 SIGABRT |
| owned_stream | 加借用流包装修复，不再发生该退出崩溃，数值仍失败 |
| lifetime_attempt / matrix_lifetime_attempt | 未交付为最终方案的中间试验，仍有 AllGather 失败 |
| final | 四补丁组合，100 轮 1600/1600 和整体验收 PASS |
| matrix_original / matrix_final | 两个库分别通过三档尺寸矩阵，每个库 960/960 |
| cross_original / cross_final | 两个库分别通过跨流消费，每个库 240/240 |
| training_original / training_final | 本轮原库与修复库训练，各 Rank 的曲线及计数 |
| fault_recovery | 实际成员退出及新进程组恢复，包含故障与恢复两个 run_id |
| event.log / acl_release.log | 真实 Event 成功调用计数、授权释放后的 ACL 预检 |

矩阵按 operation/dtype/elements 聚合，total 是轮数、passed 是成功轮数，保留最大误差及非有限值标记。
1600/960/240 均是逐 Rank 操作校验数，不把重复循环冒充独立用例。
matrix/training 的 phase 只证明脚本到达销毁完成；外层启动器退出 0 已在本轮执行时另行核验，
不能仅据 phase 取代进程验收。旧探针及故障重建的 acceptance.json 摘要另含退出码与超时状态。

## 复现约束

先按 stack.lock v2 核对锁定镜像及设备空闲、命名空间归属；资源释放须获得授权。
仅在锁定训练容器执行，下列路径自行替换为测试副本，输出使用新目录，不覆盖既有结果。
应用四份补丁后独立构建核心与 Torch 插件，通过 PYTHONPATH/LD_LIBRARY_PATH 加载测试库，
矩阵会验证真实内层后端及库哈希，不应更改原安装。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 run_acceptance.py \
  --iterations 100 --timeout 180 --run-root /path/to/new-results \
  -- torchrun --standalone --nproc_per_node=2 communication_correctness.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 timeout -k 5 180 \
  torchrun --standalone --nproc_per_node=2 communication_matrix.py \
  --iterations 20 --sizes 8 1024 65536 --out-dir /path/to/new-matrix

# 跨流复用同一探针，加 --cross-stream 并使用另一个新输出目录。
# 训练使用仓内设备方向原型快照；50 步、BATCH=4、SEQ=128、LR=1e-5。
MAX_STEPS=50 HF_HUB_OFFLINE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  timeout -k 10 600 torchrun --standalone --nproc_per_node=2 training_regression.py \
  --runtime-root /path/to/reference/prototype --model /path/to/cached-model \
  --out-dir /path/to/new-training

# 仅针对本次测试 Rank 注入退出，完成后新建全组，不重置共享设备。
python3 fault_recovery.py --run-root /path/to/new-fault-run \
  --runtime-root /path/to/reference/prototype
```

以上 Python 探针应放在同一目录。Event C++ 用例需真实 CANN include/lib 路径以及生产
event_flagcx.hpp、flagcx.h；编译宏 USE_ASCEND_ADAPTOR=1、链接 ascendcl/dl，
并传入四个链接包装参数：`--wrap=aclrtCreateEvent`、`--wrap=aclrtDestroyEvent`、
`--wrap=aclrtRecordEvent`、`--wrap=aclrtStreamWaitEvent`。

脱敏摘要由 [export_results.py](../../probes/export_results.py) 对本地快照生成，发布前人工审查：

```bash
python3 export_results.py --input /path/to/private-evidence --output /path/to/summary.json
```

完整技术解释及限制见[实机验证记录](../../docs/SERVER_VALIDATION_20260916.md)。
