# ascend-operator-runtime v3 —— 离线归档

> 机器无关清单。实体不在仓库;每台机器的实际存放路径记在同目录
> `BACKUP.local`(不追踪)。

## 压缩包(`docker load < <file>` 还原;gzip)

> round 4 验证机为 npu1-11(npu1-27 因 docker 存储损坏无法导出,见
> REBUILD.md「ROUND 4」环境事故)。下表为 npu1-11 上的实体。

| 文件 | 大小 | sha256(压缩包) | 内容 |
|---|---|---|---|
| `operator-runtime-v3-round4-npu11-62bcaae3f839.tar.gz` | 10.5G | `062b44b4d840df0598e040ad33f6d97609c9fcfd4a67d076d320602075a49c8d` | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`(npu1-11 本机重建,image id 62bcaae3f839,含覆盖层补丁) |

镜像本身的 image id(加载后 `docker images` 可见):
`sha256:62bcaae3f8399b9444628aa24085e0da6871871eb5299ca29e3b2712dc91e106`。
npu1-27 侧曾构建同源镜像 be30a952c2eb(跨机 image id 不同属预期,内容
同源:同配方+同版本清单+同补丁,三重验证证据齐,见 lock.yaml)。

父层(BAAI·FlagTree 官方 ascend3.5 预构建基座镜像)与 v2 相同,**未**单独
离线归档,理由同 v2/ARCHIVE.md(公开 harbor + 官方离线包镜像双通道可得)。

## 重建依赖资产

与 v2 相同:本血统**不需要**离线回收资产——所有层(含 vllm-plugin-FL 及
其覆盖层补丁)都是构建期从公开 commit / 公开分发桶 / 本仓 assets/ 现场
拉取应用重建(见 `REBUILD.md`),没有需要回收的已装包。round 1 曾尝试的
vllm-ascend 路径已废弃,不涉及。

## 用途

构建机或 docker 镜像丢失时的兜底:`docker load` 直接还原;或按
`REBUILD.md` + `build.sh` 从零重建(推荐,公开可复现)。round 4 真机验证
结果与运行时配置要求(推理需黑名单+shmem stub)见
`REBUILD.md`「ROUND 4 真机验证——推理腿」与人话总结
`docs/v3-round4-真机验证结果与移交指导-20261009.md`。
