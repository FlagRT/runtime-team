# 验证启动配置

`run_validation.py`将配置映射到既有官方镜像case，不实现模型或算子，也不生成准入资格。功能实现仍由独立`FlagRT/framework-adapter`安装包提供。

复制[配置示例](../demo/validation-config.example.json)，将所有`/ABS`改为当前宿主机的绝对路径。先确认空闲物理卡及目录空间，再执行：

```bash
python tools/run_validation.py --config /absolute/run.json --dry-run
python tools/run_validation.py --config /absolute/run.json
```

`--dry-run`只检查配置及打印命令，不检查远端文件、镜像或设备；真正运行仍由case执行baseline、镜像身份、输入快照、安装包、测试结果及容器停止检查。输出目录必须新建，配置副本保存在结果目录的`validation-config.json`。

| 模型配置 | 模式 | 步数与输入 | 含义 |
|---|---|---|---|
| onerec-8b | native/off/unqualified/on | 1至3步，batch=1，seq=16或32 | 原生建基线、关闭、缺资格、启用；只有on加载所给资格记录 |
| qwen-embedding-0.6b | compare | 1至3步，batch=1，seq=32 | 复用既有原生与目标成对探针；不是OneRec四路独立worker |

当前只接受SiLU。其他算子、模型、输入组合需先补对应探针和资格验证，不会根据名字自动批准。OneRec非native模式必须指向相同步数、输入形状、权重及环境的已完成原生目录；native模式不能提供reference。完整8B每步梯度和更新参数快照约32GB，三步每个序列长度约96GB，执行前核对存储。

Qwen公共双rank训练仍使用原`dc-train`入口和`demo/training-configs`，未并入此配置；不能把成对探针当公共入口验收。DeepFM目标更新参数对照失败，继续原生，不能用统一入口绕过资格。RMSNorm、自定义融合、SDPA模型注册及跨芯片迁移不在此入口支持范围。

本入口的配置单测：

```bash
python -m unittest discover -s tools -p test_run_validation.py -v
```

真实硬件成绩及已知失败统一见[10月23日报告](../docs/环境准入与算子扩展验证-20261023.md)，代码支持的参数不等于全部组合已验收。
