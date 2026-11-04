# 2026-11-04公共训练入口实际配置

本地未推送。三个子目录为11机关闭、缺模型审核、正式开启的实际docker-compose.yml和env.baseline.yaml，保持生成字节；共享父目录image.baseline.yaml是官方镜像锁。完整结果见[报告](../../../../docs/公共训练入口模型审核接入-20261104.md)。

这是执行记录，不是可以任意机器直接启动的便携模板。宿主路径、物理2/3设备和旧官方镜像只对应本次实验，资源需重核，禁止重复容器名/结果目录。运行时源码为719de8c，实测dev3 wheel SHA191887c6…，不是已发布dev2或训练v3镜像。

复现使用[run.py](../../assets/run.py)的dc-train阶段，传入两个已协调设备、只读模型/runtime/Gems、候选wheel、`--training-config-file`和新的输出目录。配置目录含schema 3及独立审核；正式开启配置见[on.json](../../review-public-20261104/on.json)，原生关闭及缺审核配置的operators需一致。不修改镜像、原模型或公共训练源码；必须通过公共baseline及up.sh --strict。

环境脚本及完整命令在[HANDOFF](../../../../HANDOFF.md#11-04公共训练入口模型审核)。资格实验另用`--training-qualification native|repeat|candidate`，只有native保存对照张量，其他两种只读参考；这是独立探针，不是正式CLI的跳过审核选项，不自动生成passed。
