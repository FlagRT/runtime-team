# 独立包依赖

功能代码唯一来源：[FlagRT/framework-adapter](https://github.com/FlagRT/framework-adapter)。本目录不再构建或偷偷注入src，只从已安装包导入。

## 使用

1. 检出新仓的锁定提交/版本（见package.lock.json）。
2. 在新仓运行`python tools/build_local.py --output <全新绝对目录>`，得到wheel及构建清单。
3. 在已有厂商环境安装：`python -m pip install --no-deps --no-index <wheel>`。不要覆盖厂商torch。
4. 官方baseline case各阶段均传`--wheel <wheel绝对路径>`；runner独立安装至结果目录并记录哈希。训练模式不挂源码，其他模式只挂测试脚本。

本地CPU集成测试可使用隔离安装目录：设置PYTHONPATH为该目录，FLAGRT_ADAPTER_INSTALLED_ONLY=1和FLAGRT_ADAPTER_INSTALL_ROOT为同一路径，再运行probes下测试。

probes里的silu_policy/silu_training_adapter/admission_records只是兼容import；旧pytorch_eager_adapter、qwen_scoped_adapter属于历史实验材料，不是新包的平行实现。

训练v3 case保持历史复现定位，目前未完成独立包迁移的硬件验收；如运行必须事先安装锁定包，并记录安装身份。不得将官方环境成功当作训练v3通过。
