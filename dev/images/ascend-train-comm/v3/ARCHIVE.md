# ascend-train-comm v3 —— 离线归档

> 机器无关清单。实体不在仓库;每台机器的实际存放路径记在同目录
> `BACKUP.local`(不追踪)。

## 压缩包(`docker load < <file>` 还原;gzip)

> round 4 验证机为 npu1-11。下表为 npu1-11 上的实体。

| 文件 | 大小 | sha256(压缩包) | 内容 |
|---|---|---|---|
| `train-comm-v3-round4-npu11-e13a15d0d02b.tar.gz` | 10.5G | `c1890db494de27de9671c06fe0129f012bbdcac43cfce4c53dd065f739703c22` | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`(npu1-11 本机级联重建,image id e13a15d0d02b,经父镜像继承覆盖层补丁) |

镜像本身的 image id(加载后 `docker images` 可见):
`sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15`。
npu1-27 侧曾构建同源镜像 7028028bb62c(跨机 image id 不同属预期,内容同源)。

父层的离线包见 `../../ascend-operator-runtime/v3/ARCHIVE.md`。

## 重建依赖资产

与 v2 相同:本血统**不需要**离线回收资产——FlagCX 层从 `FlagRT/FlagCX` 组织仓
公开主干 commit `4e0e0cbcbf721169ca82348080f8353aebfe2c31`(与 v2 相同,未升级)
构建期直接源码编译,没有需要回收的已装包、没有私有 patch。

## 用途

构建机或 docker 镜像丢失时的兜底:`docker load` 直接还原;或按
`REBUILD.md` + `build.sh` 从零重建(推荐,公开可复现)。round 4 真机验证
结果(含 DDP 封装仍阻塞 c10d 层的现状)见 `REBUILD.md`「ROUND 4 真机
验证——训练腿」与人话总结
`docs/v3-round4-真机验证结果与移交指导-20261009.md`。
