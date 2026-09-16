# FlagCX 修复交付包

目标仓库：`FlagRT/FlagCX`；基线：`4e0e0cbcbf721169ca82348080f8353aebfe2c31`。
两份补丁由 `git format-patch` 从本地提交导出，保留提交说明及 Kistich 共同署名。
本方向整理历史修复，未重新认领原始问题定位。

1. `0001-*` 对应 `8c836df`：将 host callback 中的等待移至 groupLaunch 主线程。
2. `0002-*` 对应 `8b15547`：CANN Event 析构释放、空初始化、禁止复制，附主机测试替身。

在独立 FlagCX 工作分支、干净基线上按顺序使用 `git am <0001文件> <0002文件>`。
不要在镜像源码版本不明的目录盲目应用；锁定镜像的私有 `55eb2ff` 不是本补丁基线。

服务器上核心库及 Torch 插件均编译通过；Event 主机模拟测试通过。真实设备回归因
ACL/驱动占用阻塞，补丁尚未达到正式合入标准，也未证明解决候选镜像的退出 SIGABRT。
详见 [服务器证据](../../docs/SERVER_VALIDATION_20260916.md)。

本次仅将补丁包提交到 runtime-team 的用户指定分支，未推送 FlagCX 子仓或共享 dev。
