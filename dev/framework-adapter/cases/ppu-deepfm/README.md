# PPU 公共 DeepFM 训练回归

目的：复用公共 DeepFM v1（模型源码不改）和 device-context 的 PPU backend，检验 sigmoid 接入、条件拒绝及训练结果。使用随机初始化、固定合成分类数据，FP32、batch32、Adam、3步，保留公共模型的 BatchNorm 和 Dropout。

## 执行顺序

1. 准备：`assets/prepare.py` 冻结公共 runtime/model/lib、两份精确wheel及本方向探针。输出目录必须不存在；远端也先拒绝重名目录。`assets/test_case.py` 对冻结输入做CPU控制。
2. `run.py --stage qualification`：sigmoid shape32、FP32，独立前反向、CPU及PPU原生对照，3次重复。该阶段不签发审核。
3. `run.py --stage model --mode native`：保存3步预测、loss、梯度、模型状态和优化器状态。
4. `--mode native_repeat --reference <native目录>`：相同种子和输入，逐值核原生重复性；不通过即停止目标资格。
5. `--mode candidate`：独立数值资格，用 FlagGems 的临时注册直接调用目标 sigmoid 前反向，和原生参考比较。这不是正式接入开关，不产生模型审核。
6. 人工检查算子资格、原生重复及候选证据；只有数值、真实调用、身份、恢复均满足判据后，创建明确的PPU算子记录和模型审核。
7. 新输入快照携审核，分别 `--mode off/on/unknown`：关闭、开启、缺模型审核。开启必须每步真实命中目标前反向；另外两路目标0调用且与原生逐值一致。退出恢复必须成功。

每次传 `--root <本人新根目录> --run-id <唯一ID> --card <刚核空闲的物理卡>`。运行器只映射单卡，断网、8GiB主存限制、480秒探针上限；公共baseline export后必须`up.sh --strict`通过，最终停止本人容器。`--reference`只读挂载，原生张量不覆盖。

资源门控：连续三次（间隔1秒）检查利用率0%、无该卡进程（包括0MiB进程）、显存≤64MiB。11-10观察到无进程卡26～59MiB底噪，原32MiB门控拒绝记录保留；这不是数值容差调整，也不是卡的预约。

## 判据和材料

- 原生重复、关闭、缺审核：atol=rtol=0。
- 候选/开启：沿用现有模型判据atol=1e-5、rtol=1e-4；不为通过修改容差。
- 阶段输出：result.json、summary.json、probe.log，以及实际demo/基线；native另含tensors。
- 训练脚本衍生自既有P800 DeepFM case；公共模型仍逐文件核固定SHA，环境明确为ppu/cuda，不沿用P800审核。model_probe/model_admission对照助手在prepare时按原字节冻结。
- 原生不具可重复性或候选超差时，保留证据并停止审核，不把执行后切换原生作为恢复机制。

代码和材料当前仅本地；实际成绩以STATUS及对应日期报告为准。
