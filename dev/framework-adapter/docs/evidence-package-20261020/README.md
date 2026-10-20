# 0.2.0交付与拆仓验证摘要

本页是原始结果的精简摘录，不是原始日志替代品。运行于2026-10-20；代码来源和限制见[报告](../训练入口配置接入验证-20261020.md)。

| 实际验证 | 结果 |
|---|---|
| 原源码三组：off/on/unqualified | 每组2rank×3步；每rank上游6/6；PASS |
| 安装wheel三组，不挂/code | 每组2rank×3步；每rank上游6/6；PASS |
| 开启组真实GEMS计数 | 每rank SiLU FORWARD=84、BACKWARD=84 |
| 关闭/无资格组 | 每rank84次native，GEMS命中0，分别experimental_disabled/capability_unverified |
| 各rank上游四位小数loss | [17.0600, 16.3875, 14.4921]；不等于逐张量对照 |
| wheel安装位置 | /results/installed/flagrt_adapter；0.2.0 |
| 拆仓重建 | 41项安装包CPU控制＋示例/命令入口PASS |
| runtime-team跨仓安装测试 | 41项控制＋17项case控制PASS；不注入src |
| 新旧wheel包内容比对 | 7个flagrt_adapter文件逐字节一致；不把新归档哈希冒充旧NPU实测哈希 |

精确wheel哈希和源提交见[依赖锁定](../../package.lock.json)。环境：旧官方镜像ID `sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f`，FlagGems `f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4`，device-context `2926d1cd716c7cac55ec268e5c533940bca94294`；模型权重与脚本hash见报告。

训练v3缺shmem未修复，不属于此成功结果。首轮源码0/1初始化失败保留在完整备份，后改0/5成功；安装包三组无新增失败。容器已停止，无OOM。

原始材料由负责人保留在Project/outputs/framework-runs/20261020-training-config及20261020-wheel；包含基线、安装/导入日志、每rank JSON、实际挂载、源码哈希和完整测试日志。未将模型、凭证或编译缓存入仓。
