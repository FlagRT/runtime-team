# PPU Qwen / OneRec 模型验证

2026-11-10：本case用于PPU实机模型回归。当前结果与运行阶段统一看[STATUS](../../STATUS.md)，本页只维护执行方式及判据。

复用kunlun-unary的完整预训练模型、合成输入、优化器及判据，只切换PPU环境和运行资源。`prepare.py`逐条检查源文本再生成冻结脚本；源文件SHA记录在port.json，全部输入SHA记录在inputs.json。源码变化导致匹配不唯一时立即拒绝生成，不能静默继续。功能包沿用本地dev4候选，不改发布锁。

| 配置 | Qwen3-Embedding-0.6B | OneRec-8B |
|---|---|---|
| 输入/精度 | batch1 seq32 FP32 | batch1 seq16 BF16 |
| 目标 | SiLU，shape1×32×3072 | SiLU，shape1×16×12288 |
| 训练 | 3步AdamW，lr1e-5 | 3步SGD，lr0.01，非重入梯度检查点 |
| 模型比较 | atol1e-5、rtol1e-4 | atol1e-3、rtol1e-2 |
| 原生重复/关闭/缺审核 | 逐值一致 | 逐值一致 |

顺序：权重SHA与空卡核验→对应形状单算子资格→原生及重复→独立候选→人工核对结果→关闭/开启/缺模型审核。候选失败则保留原生并记录失败，不签发passed模型审核、不放宽判据。

原生/候选是模型数值资格实验；普通on必须有schema3模型审核。candidate只允许单算子审核和原生oracle，不等于模型已批准。

运行每轮独立目录、单卡、1800秒上限，64GiB主机内存限额，磁盘至少150GiB；复用公共baseline export与up.sh --strict。结束停止本人容器。共享权重只读挂载，禁止写/data满盘或根盘。

本地冻结输入：Project/outputs/framework-runs/20261110-ppu-models。Qwen实际使用input02（资格/原生）及input03（单算子审核/模型对照）；OneRec使用input01（资格/原生）及input02（单算子审核/模型对照）。运行入口是冻结目录run.py，参数为`--root <本人新实验根> --run-id <新ID> --card <实查空卡> --model <只读权重目录> --stage qualification|model`；模型再给`--mode`，非native须给`--reference`。首次输入无review，只允许资格/原生/重复；审核后生成新输入，不修改已执行快照。

Qwen共享模型：`/bmcp_lvm_fs/hliu553/models/Qwen3-Embedding-0.6B`；OneRec共享模型：`/bmcp_lvm_fs/chenyuanxin/models/OneRec-8B`。均只读挂载，每轮核对固定权重SHA。路径不是永久可用性保证；不得修改或搬动共享文件。早前下载尝试已停止，下载辅助脚本归入本地产物目录download-tools，不属于接入组件。

Mac打包须使用`COPYFILE_DISABLE=1 tar --no-xattrs`，避免AppleDouble隐藏文件进入wheel目录；旧input01的失败证据保留。每轮输出中的demo由公共工具生成并经`up.sh --strict`检查，保留真实绝对路径用于审计，复跑必须生成新的demo。

CPU控制：用含torch的Python运行`test_port.py -v`，设置PPU_MODEL_INPUT为冻结目录、PYTHONDONTWRITEBYTECODE=1。两模型各8项是相同控制集的两次执行，不算16类新增功能。
