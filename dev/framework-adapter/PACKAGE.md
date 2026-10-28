# 独立包依赖

功能代码唯一来源：[FlagRT/framework-adapter](https://github.com/FlagRT/framework-adapter)。本目录不再构建或偷偷注入src，只从已安装包导入。

## 使用

1. 检出新仓的锁定提交/版本（见package.lock.json）。
2. 在新仓运行`python tools/build_local.py --output <全新绝对目录>`，得到wheel及构建清单。
3. 在已有厂商环境安装：`python -m pip install --no-deps --no-index <wheel>`。不要覆盖厂商torch。
4. 官方baseline case各阶段均传`--wheel <wheel绝对路径>`；runner独立安装至结果目录并记录哈希。训练模式不挂源码，其他模式只挂测试脚本。

本地CPU集成测试可使用隔离安装目录：设置PYTHONPATH为该目录，FLAGRT_ADAPTER_INSTALLED_ONLY=1和FLAGRT_ADAPTER_INSTALL_ROOT为同一路径，再运行probes下测试。

probes里的silu_policy/silu_training_adapter/admission_records只是兼容import；旧pytorch_eager_adapter、qwen_scoped_adapter属于历史实验材料，不是新包的平行实现。

package.lock.json现锁定0.3.0.dev0源码候选[85a9896](https://github.com/FlagRT/framework-adapter/tree/85a9896537a7d88f758d36ae20465e19470985e2)，包含tanh、启动准入及单机双rank门控参考原型。历史v0.2.0标签不变；10-23模型结果来自较早wheel，最新候选硬件范围见[10-28报告](docs/双rank步骤门控验证-20261028.md)及[证据摘要](docs/evidence-step-guard-20261028/summary.json)。不同候选wheel不混称重新实测。

训练v3的run_startup.py现已验证候选包的关闭/允许原生/严格拒绝三路，目标编译仍未通过。历史完整目标回归入口保持独立；不得将原生回退通过写成shmem修复。
