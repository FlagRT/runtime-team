# 修复整理与测试摘要

FlagCX 公共源码基线为 `4e0e0cbcbf721169ca82348080f8353aebfe2c31`，最终本地提交 `133dfba`。
四份[补丁](../patches/flagcx/)依次应用；未推送 FlagCX 子仓，未覆盖锁定容器的原安装库。

| 补丁 | 来源与作用 | 验证边界 |
| --- | --- | --- |
| 0001 / 8c836df | 从 Kistich 改动中整理 P2，将等待移出 host callback | 编译及同构回归完成；不等于异构 proxy 死锁路径专项已验收 |
| 0002 / 8b15547 | 整理 Event 析构释放，补充空初始化与禁止复制 | 生产头文件真实 CANN 创建/记录/跨流等待/释放各 1000 次成功 |
| 0003 / af5640d | 本轮定位借用流误销毁；改为 streamCopy/streamFree 管理包装对象 | 同一探针从退出 SIGABRT 变为正常完成销毁；数值问题当时仍存在 |
| 0004 / 133dfba | 本轮补齐 Host wait 与 AllGather 复制前后同步 | 原失败探针 1600/1600；矩阵 960/960；跨流 240/240；训练回归通过 |

P6/P7/P9/O3/O4 等原分支上的其余改动仍有 callback/runner/proxy 依赖，不整批移植。
0004 的 AllGather 同步是限定 CANN 路径的保守兼容方案，不应写成无开销或已支持异步重叠。
`wait(timeout)` 仍未实现后端内的有界超时，外层监督器负责总超时。

`python3 -m unittest discover -s dev/communication/tests -v`：25 项主机测试通过
（14 项进程/结果验收、7 项归属解析、4 项脱敏导出）。
这些数量不并入 NPU 通信测试。主机 Event 替身测试仍保留，不能与新增实机测试混为一谈。

历史 320/320、当前 1600/1600 均为逐 Rank 操作校验数，不是独立场景数或准确 API 调用数，
因为双向 P2P 的一个校验含 send 和 recv。本周实测、失败迭代及库指纹见
[服务器记录](SERVER_VALIDATION_20260916.md)和[脱敏数值摘要](../results/20260916-release/summary.json)。
