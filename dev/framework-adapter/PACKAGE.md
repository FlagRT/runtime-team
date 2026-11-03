# 独立包依赖

功能代码唯一来源：[FlagRT/framework-adapter](https://github.com/FlagRT/framework-adapter)。本目录不再构建或偷偷注入src，只从已安装包导入。

## 使用

1. 检出新仓的锁定提交/版本（见package.lock.json）。
2. 在新仓运行`python tools/build_local.py --output <全新绝对目录>`，得到wheel及构建清单。
3. 在已有厂商环境安装：`python -m pip install --no-deps --no-index <wheel>`。不要覆盖厂商torch。
4. 官方baseline case各阶段均传`--wheel <wheel绝对路径>`；runner独立安装至结果目录并记录哈希。训练模式不挂源码，其他模式只挂测试脚本。

本地CPU集成测试可使用隔离安装目录：设置PYTHONPATH为该目录，FLAGRT_ADAPTER_INSTALLED_ONLY=1和FLAGRT_ADAPTER_INSTALL_ROOT为同一路径，再运行probes下测试。

probes里的silu_policy/silu_training_adapter/admission_records只是兼容import；旧pytorch_eager_adapter、qwen_scoped_adapter属于历史实验材料，不是新包的平行实现。

package.lock.json现锁定0.3.0.dev2源码候选[6ee9129](https://github.com/FlagRT/framework-adapter/tree/6ee91294e7ab028f2e804225482f58d24c8ccd5a)，包含厂商/设备接口区分、schema 3模型审核、启动准入及保留的单机双rank门控。历史dev0/v0.2.0不变；dev2的10个运行时文件与10-29限定P800实测wheel一致，重新构建不等于新硬件验收，见[发布审查](docs/dev2发布审查-20261103.md)。

兼容注意：旧七字段昇腾JSON由load_records兼容读取，但Python Environment的torch_npu关键字/属性改为backend_version；直接使用旧关键字的调用方需迁移。P800需显式指定kunlunxin/cuda，不能借用ascend/npu资格。公共双rank CLI尚未采集真实模型身份，不能拿case中的schema 3通过结论替代公共入口验收。

训练v3的run_startup.py现已验证候选包的关闭/允许原生/严格拒绝三路，目标编译仍未通过。历史完整目标回归入口保持独立；不得将原生回退通过写成shmem修复。
