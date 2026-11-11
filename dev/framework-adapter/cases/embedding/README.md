# DeepFM embedding 接入验证

先对照固定FlagGems的独立前反向，再运行适配开关和模型。仅修改接入层，不修改算子实现。当前结论见本方向STATUS；原始输出位置见HANDOFF。

## 固定调用契约

- 入口：`torch.nn.Embedding`使用的`torch.nn.functional.embedding`。
- DeepFM两组权重：FP32 `[3850,16]`、`[3850,1]`；索引为同卡int64 `[32,5]`，非负且小于3850；输入连续。
- 暂限`padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False`。资格必须同时匹配权重和索引，不复用一元算子的shape记录。
- FlagGems调用：`embedding(weight, indices, -1, False, False)`及`embedding_backward(grad, indices, num_weights, -1, False, False)`；使用库函数调用方式，不修改全局embedding注册。
- 一阶稠密梯度；展开/转置视图先作值不变的`contiguous()`，日志记录复制次数。高阶梯度、非有限值和目标执行异常停止。前向不支持的选项保留合法原生；严格目标模式拒绝。
- 目标算子的反向采用原子加法，不承诺逐位确定。资格与模型沿用`atol=1e-5, rtol=1e-4`；原生重复/关闭/缺模型审核要求逐值一致，不放宽历史阈值。

固定FlagOP `2591d081`中的PPU后端使用原生index_select和embedding_backward，且register_a1修改模块全局后端并返回Library。本轮没有使用这个注册入口，也没有将其原生委托计为独立FlagGems内核命中。

## 材料和执行顺序

1. `prepare.py`：由既有PPU DeepFM冻结输入生成新资格目录；`probe.py`覆盖两形状×四索引分布×三重复，共24组，独立核CPU、PPU原生及目标前反向。
2. 人工读结果后显式运行`review.py --decision approve`；只生成上述两个签名的算子记录，测试程序不会自行批准。
3. 构建独立包wheel；`prepare_routes.py`生成新输入，保持公共模型原文不动。`routes.py`核关闭/开启/缺算子资格三路及退出后原生。
4. `batch.py`串行运行新原生参考、原生重复、独立模型候选、关闭、缺模型审核；每组3步。候选只有算子资格，结果再用于人工模型审核。
5. 人工核对原生、重复、候选三份结果后，用`review_model.py --decision approve`生成模型审核并另建输入跑开启；失败则`--decision reject`记录失败审核，不继续同组合盲目重试。

runner复用公共基线导出与`up.sh --strict`，每轮核输入SHA、镜像ID及连续3次空卡采样；单卡、禁网、独立pip target安装、限时执行，finally停止本人容器。共享基座/算子源码不变。

## 包接口

功能代码在独立仓`flagrt_adapter.embedding`：`EmbeddingSignature`、`EmbeddingQualification`、`EmbeddingPolicy`、`load_embedding_records`和`EmbeddingAdapter`。
本case用`embedding-case-v1`配置和既有WorkloadReview绑定模型身份；尚未并入一元算子公共CLI的schema1/2/3。不得把embedding塞进旧`shapes/dtypes`配置当作多输入已审核。

每步比较预测、loss、全部参数梯度、model state和Adam state。模型审核身份含公共源码、初始化/数据/Dropout种子、优化器、执行参数、适配器及探针源码SHA。同一模型换算子或改接入代码需重新审核。

离线核验：先复用`../ppu-models/collect.py`收集所有已停止输出及SHA，再用本目录`verify.py --root <解包目录> --manifest <evidence-sha.json> --output <新文件>`核对原件、调用、步数和容器停止状态。`test_controls.py`只验证本地检查逻辑。
