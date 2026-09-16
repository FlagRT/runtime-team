# 修复整理与测试摘要

基线为 FlagCX `4e0e0cbcbf721169ca82348080f8353aebfe2c31`；来源为 Kistich 的
`ascend-flagcx-adapt` 修复分支。本方向整理两项最小改动并补充测试，保留原作者署名。

- P2：将 host callback 内的等待移到 groupLaunch 调用线程，避免回调阻塞 proxy 所需驱动操作。
- Event：析构时释放 CANN Event，初始化空句柄并禁止复制，防止资源泄漏及重复所有权。
- P6/P7/P9/O3/O4 仍含 callback/runner/proxy 等依赖链，未整批移植，待继续审查。

核心库与 Torch 插件已经在锁定服务器环境编译成功。Event 主机测试直接编译生产头文件，
ACL 测试替身完成 1000 次创建/释放及失败创建检查；它不代表真实设备 Event 运行验证。

执行 `python3 -m unittest discover -s dev/communication/tests -v`，14/14 通过，覆盖正常退出、
数值 PASS 后异常退出/SIGABRT、超时、缺失 Rank、旧结果、计数错配等。

历史探针的 320/320 是逐 Rank 操作校验数，P2P 每项含 send/recv；不是 320 个独立场景。
新 schema 明确 counting_unit/run_id/phase，整体验收还检查进程组销毁与进程退出。

[补丁交付包](../patches/flagcx/)可在精确基线上应用；[服务器摘要](SERVER_VALIDATION_20260916.md)
记录真实运行阻塞。原始服务器日志未随本次发布，完整本地证据保持保留。
