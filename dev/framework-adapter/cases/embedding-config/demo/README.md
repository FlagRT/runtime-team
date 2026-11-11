# embedding统一配置的实测demo

`qwen-on-20261111/`为PPU Qwen embedding审核后开启的实际启动配置、导出环境基线及strict核对回执，原样归档；镜像基线见上一级`image.baseline.yaml`。

历史绝对路径和物理卡号只用于复现定位，不是空卡预约。不要直接重启该compose；按[HANDOFF](../../../HANDOFF.md#11-11-qwen-embedding资格及模型三路)重新核资源，再用冻结runner生成新run-id和demo。

镜像内没有本方向适配包和本轮FlagGems。每次容器启动后，runner从只读输入中的固定wheel离线安装到本轮`/results/packages`，不修改共享镜像；两wheel的SHA、安装日志、实测包内容身份与模型审核见输入manifest和结果。当前demo用于说明实际执行过程，不是新的基座镜像发布物。
