# 训练 v3：cann-shmem 修复验证

2026-11-06，顾宬方向；本地记录，尚未提交推送。

## 结论

在 11 机的训练 v3 隔离容器中，安装 `cann-shmem==1.6.0` 后，原先因缺少 shmem 失败的最小循环算子编译、执行和数值检查通过。FlagGems SiLU 的 3 组 FP32 形状、前向和反向共 6 项检查也通过，最大绝对差 `2.3841858e-7`。

总组原命令不固定版本，当前会选择 `1.7.0`，但官方索引的 SHA256 与实际下载内容不一致，pip 拒绝安装。本次没有跳过校验；改测同源、哈希校验通过的 `1.6.0`。

可以反馈群里：

> 11 机训练 v3 已验证：固定安装 cann-shmem==1.6.0 后，原 shmem-missing 最小复现通过，SiLU 前反向 6 项检查通过。直接安装最新版会选到 1.7.0，目前源上该 wheel 的下载哈希与索引不一致，pip 拒绝安装，请编译组确认推荐版本并检查发布源。

## 环境与步骤

| 项目 | 实际记录 |
|---|---|
| 主机与设备 | npu1-11，物理 14 → 容器逻辑 0，单卡；启动前无该卡任务 |
| 镜像 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` |
| 镜像 ID | `sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15` |
| 关键版本 | CANN 9.0；torch 2.10.0+cpu；torch-npu 2.10.0；FlagTree 0.6.0+ascend.git15ec1a6c / Triton 模块 3.5.1；FlagGems f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4 |
| 公共脚本快照 | runtime-team `0da9977086d07a867bf915b207f08e70f541683c` |
| 前置检查 | 镜像基线 check、运行环境 export、`up.sh --strict` 全部退出 0 |
| 包变化 | pip 包清单仅增加 cann-shmem 1.6.0，其他包版本不变 |
| 资源收尾 | 本人容器已停止，Running=false / Pid=0 / OOMKilled=false；sudo 票据、SSH 和控制连接已清理 |

原始最小用例直接取公共 `dev/issues/shmem-missing/assets/minirepro_shmem.py`，未改内容。前后运行分别创建全新 Triton 缓存；输入输出分离；8192 个 FP32 元素，核对输出等于输入乘 2。

安装先 dry-run 审核依赖，确认只新增一个包；下载后校验官方 SHA256，再离线 `--no-deps` 安装这个 wheel。未修改共享镜像、驱动、FlagGems/FlagTree 源码或他人容器。

| 检查 | 结果 |
|---|---|
| 安装前最小用例 | exit 1，`ModuleNotFoundError: No module named 'shmem'`，问题复现 |
| 默认 1.7.0 安装解析 | exit 1，wheel 哈希不匹配；未安装 |
| 独立 HTTP 重查 | 1.7.0 仍不匹配；1.6.0 与索引匹配 |
| 1.6.0 解析、下载、安装 | 均 exit 0，来源及 SHA 已记录 |
| 安装后同一最小用例 | exit 0，`MINIREPRO_OK=True`、`overall: PASS` |
| SiLU 独立资格探针 | exit 0；`(128,)`、`(2,17)`、`(4,1024)` 各前向/反向通过 |

SiLU 使用 `flag_gems.silu` 和 `flag_gems.silu_backward` 直接调用，CPU PyTorch 为对照；固定随机种子 123，`atol=1e-5, rtol=1e-4`，未放宽阈值。没有生成模型或生产接入审核记录。

## 包来源异常

索引：`https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/`

| wheel | 索引 SHA256 | 实际 SHA256 |
|---|---|---|
| cann_shmem-1.7.0-cp311-cp311-linux_aarch64.whl | `b96de29cad1f0ee5fde75a4131523fedc7c3701cef8d70154e740ca2fba20e32` | `af8376c14688e80933d424c8118f358fb06f3b7247bb1445cbdbe02cd44d4191` |
| cann_shmem-1.6.0-cp311-cp311-linux_aarch64.whl | `308a1c5a3dfe039d77d1159e27f3b1c16f38f4990a19c1fc68c7c052dee2b3f5` | 与左侧一致 |

1.7.0 的 HTTP 响应头 `X-Checksum-Sha256` 也为实际下载的 `af8376…`。这是同一源的索引与文件校验信息不一致；原因待发布方确认，不据此认定包被篡改，也不以响应头替代 pip 索引校验。

## 后续接入

这次先完成依赖修复和单算子资格验证。适配包 dev3 对原训练 v3 镜像有 `known_compile_failure` 拦截，镜像 ID 在容器补包后没有变化，因此不能直接删除拦截或沿用旧审核。下一步把补包版本/哈希纳入运行环境身份，再做适配开关与模型三路回归；本次没有改功能仓的准入策略或跑模型。

容器没挂载宿主 `npu-smi` 命令，shmem 导入时的拓扑检测报告 UNKNOWN/0 卡及 RDMA 未配置；实际 NPU 计算已经通过。本次不验证 shmem 的多卡通信功能，不因检测警告去改宿主网络。

取证脚本初版误以为存在独立 `triton` 发行包，元数据查询失败；修正为同时记录 FlagTree 发行版本与 Triton 模块版本后取证成功，两个脚本均保留。停止容器时 sleep 主进程最终退出 137，OOMKilled=false；这是测试完成后的 stop 清理，不是测试进程退出码。

## 证据

- [机器摘要、包身份及 60 份文件 SHA](evidence-cann-shmem-20261106/summary.json)
- [关键输出截选](../cases/cann-shmem-validation/assets/evidence/20261106.md)
- [复现入口与脚本](../cases/cann-shmem-validation/README.md)
- [本轮 baseline/demo 快照](../cases/cann-shmem-validation/demo/20261106/README.md)
- 11 机完整目录：`/home/cgu135/framework-shmem-20261106`；日志在 `results/`，输入在 `input/`，保留安装前失败证据。
- 归档 `evidence.tar.gz` SHA256：`b82421ca041239c2df8ccb46985d1057bed88ac2da376e16bbd047486dd67d92`，已取回本地并核验。
