# 训练通信回归：长序列重复验证与 FlagCX 消费侧用例

_通信方向本周交付：OneRec 长序列重复验收、DeepFM 新原型复验、FlagCX 独立回归。_

---

## 📋 结论与对齐

OneRec-8B 序列长度从16扩至64，64/128 MiB各三轮、每轮50步，[六轮全部通过](../results/20261028-onerec/matrix-summary.json)。399个参数张量的最终SHA256在两Rank、两档和全部独立运行间逐项一致。当前配置建议128 MiB。DeepFM新版原型[六轮复验通过](../results/20261028-deepfm/summary.json)，新增P800消费侧适配后主机回归72/72。

执行前核对公共提交`c3f7fb9671aca514a917a3c926a857f8e8e2c8ab`。设备方向近期更新可计时Event、共享模型路径解析及多实例复验；本方向消费的`use/device_count/set_device/current/synchronize(timeout_ms=30000)`契约继续通过检查，不重复申报设备conformance或训推腿成果。

提交前再次对齐公共`4d2f710`：新增内容为内存方向P800卸载闭环、真实模型A/B及PPU移植报告；与本方向训练通信范围分开。消费的设备原型、镜像及公共基线工具相对执行时版本没有变化，合入时保留其最新材料。

公共推荐训练环境已采用[env/v1挂载覆盖](../../images/ascend-train-comm/v3/env/v1/docker-compose.yml)，见[发版说明](../../../docs/基座发版说明-20261014.md)。上周“等重建镜像再验收”的计划据此更正。镜像仍为v3 `e13a15d0d02b...`，有效FlagCX为设备方向修复`62fe6153852658f17e729a06be4da1c766201ef7`。[独立运行目录](../env/910c-repeat/)复用窄设备配置、私有IPC、4核限额；`up.sh --strict`核对镜像及干净、已推送的FlagCX后启动。公共配方与锁定声明未修改。

## 📊 OneRec重复实测

完整检查点`OpenOneRec/OneRec-8B`，revision `29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff`，三个权重分片重算SHA256，与[原始记录](ONEREC_TRAINING_20261014.md)逐字相同。83.91亿参数全量训练、BF16、无动量SGD、lr=0.01、每端batch=1、全局batch=2、梯度检查点、eager attention、固定种子合成token。

| 指标 | 64 MiB | 128 MiB |
| --- | ---: | ---: |
| 独立启动×步数 | 3×50 | 3×50 |
| 序列长度 | 64 | 64 |
| 启动器退出0且组正常关闭 | 3/3 | 3/3 |
| 每端50步梯度AllReduce | 12,550 | 6,300 |
| 每端50步输入张量字节数 | 839,101,337,600 | 839,101,337,600 |
| 非首步耗时：三轮均值±样本标准差 | 0.750±0.015 s | 0.693±0.022 s |
| 结束时最大used_mb范围（MiB） | 36,746–36,747 | 36,747–36,748 |
| 结束占用换算 | 约35.9 GiB | 约35.9 GiB |
| 最终参数哈希 | 两端、两档、三轮全部一致 | 两端、两档、三轮全部一致 |

每轮独立启动，bucket顺序交替为64→128、128→64、64→128。统计单位是每次启动中两个Rank的非首步平均值的较大者，每轮取后49步，再对三轮取均值与样本标准差；计时包含有限值检查。128 MiB相对64 MiB调用减少49.8%，观察到的平均step耗时低约7.5%，建议作为当前配置起点；未外推更长序列、其他模型或业务端到端收益。

显存是结束快照，不是峰值。上周周报将约36,733 MiB写作36.7 GiB，正确值为35.9 GiB：原始MiB不变，不是本周新增降幅。输入字节数也不是实际链路流量。

新增`run_onerec_matrix.py`与严格汇总器，拒绝遗漏/重复轮次、非零启动器、错误序列/bucket、不完整训练及跨轮参数漂移，新增六类主机检查。[完整结果目录](../results/20261028-onerec/)保留每个Rank的399组哈希与训练轨迹。

## 🔧 FlagCX仓交付与DeepFM

库层回归位于[FlagCX/test/regression/training_consumer](https://github.com/FlagRT/FlagCX/tree/dev-1.0/test/regression/training_consumer)，从自有分支`lianzhongyou/multicard-communication`合入其`dev-1.0`。复用设备方向已有修复，新增持久缓冲复用、当前流与组生命周期覆盖：3种dtype×3档长度×默认/非默认流，交替同步调用及`Work.wait()`。最终入口三次独立启动均通过；每次每Rank3,033次检查=AllReduce 2,700+列表AllGather 300+有限值30+参数更新3，累计9次正常组销毁/重建、90步原生DDP。Linux CPU/Gloo对照每Rank1,533次通过。源码、入口、加载二进制SHA256和证据归FlagCX，模型组织和原型消费归runtime-team。

旧镜像FlagCX的同入口短负对照返回FAIL：结果读回错误、原生DDP启动异常及退出SIGABRT。初次负对照因一端来不及保存记录判ERROR，已补齐提前落盘。另一次未协调重建在第三轮bootstrap失败，驱动按[PyTorch重建要求](https://docs.pytorch.org/docs/stable/distributed.html#reinitialization)加入组外TCPStore协调及每轮独立命名空间；修正后上述三次独立启动通过。失败证据全部保留，不将测试驱动的问题冒记为新的C++修复。

DeepFM为421,452参数、FP32、全局batch64、每端32、预热5步再训练50步。逐张量/合并各三轮6/6通过：每端17/17梯度参考通过，梯度最大误差`4.19e-9`、更新误差`3.73e-9`、最终参数差0、两模式loss最大差`9.54e-7`。六次launcher退出0且通信组CLOSED。

## ⚠️ 边界、资源与未完成项

本周11号机驱动26.1.1，使用经容器映射核查和真实初始化的逻辑设备10/11。性能比较仅限本周两档bucket，不与上周不同驱动或HCCL数据直接推导收益。正常销毁/重建不等于故障恢复；模型均为合成输入功能验收。

P800继续推进后，sudo与Docker权限已实际确认，数据盘约3.1 TiB可用。卡1有当天设备执行超时及他人活进程，本轮不选用、不reset，但不据此认定物理损坏。利用物理卡5的空闲窗口完成[DeepFM单卡消费侧准备](P800_DEEPFM_PREPARATION_20261028.md)：逐张量/合并各50步、17/17梯度参考通过、最大梯度及更新误差3.73e-9、两模式loss轨迹和最终参数哈希一致，两个启动器退出0、组CLOSED。首轮无网络容器的BKCL初始化失败已保留，修正为Docker bridge后通过。健康连续双卡仍待窗口，单Rank不计多卡验收；本月交付仍2/4=50%，整体按原表7/37=18.9%，重复轮次不折算为新增功能项。

首批DeepFM命令的`prototype_revision`录入错误，留在服务器`evidence/deepfm-current/`，不计正式验收；正确完整版本号的`deepfm-verified/`六轮重新执行通过，对应仓内结果，原记录未改写。

## 📍 复现与下一步

容器内加载CANN环境，输出目录必须全新：

```bash
python3 dev/communication/training/run_onerec_matrix.py \
  --runtime-root /workspace/dev/device-context/prototype \
  --model-root /workspace/models/OneRec-8B --seq-len 64 --steps 50 --repeats 3 \
  --output /workspace/evidence/new-onerec-matrix
python3 dev/communication/training/summarize_onerec_matrix.py /workspace/evidence/new-onerec-matrix
```

完整日志留在`npu1-11:/home/lianzhongyou/lianzhongyou_data/runtime-tests/20261028-communication/evidence/`，并归档Mac本周证据目录；仓内只收结构化结果。下周推进DeepFM原生DDP模型入口与同步聚合对照、一类通信失败后的整组退出/重建，并在健康双卡窗口完成P800最小迁移及下游消费确认。
