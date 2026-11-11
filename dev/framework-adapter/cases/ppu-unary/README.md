# PPU SiLU / sigmoid 接入与安全回退

## 用途与现状

使用总组PPU v2镜像，在单张PPU-ZW810E上独立验证已有FlagGems算子，再验证适配器的关闭、开启、缺审核三路。原生使用厂商PyTorch；设备操作复用device-context的PPU后端。功能包只新增PPU环境身份支持，不改算子实现。

镜像ID：`sha256:1ba7ead15378160ba9aa1dafcfb77686de6b16732590ffab062bab19c60c32b8`。版本事实见[image.baseline.yaml](image.baseline.yaml)，实际运行环境由[run.py](assets/run.py)生成demo并执行公共`up.sh --strict`。

FlagGems来源为公开`flagos-ai/FlagGems`提交`73c5aff1e304c293ab525ecb03e9449644496fbb`。在上述镜像的无卡、断网构建容器中运行`pip wheel --no-deps --no-build-isolation --no-index`，默认关闭C++扩展；没有改源码。运行时显式`GEMS_VENDOR=nvidia`选取CUDA兼容通用实现；适配器身份仍为`ppu/cuda`，不借用NVIDIA或昆仑芯审核。

安装的两个wheel由run.py以`--no-deps --no-index --target /results/packages`放入本人结果目录；不修改公共镜像。安装字节和环境共同参与审核。当前仅为独立单算子资格与开关实验，模型入口后续单独对接。

2026-11-10：18组独立资格、18组三路均通过，安装包146项CPU控制通过。详见[验证报告](../../docs/PPU算子接入与安全回退验证-20261110.md)。

## 复现

| 前置条件 | 来源及要求 |
|---|---|
| 镜像与设备 | 上述精确v2镜像，一张实时空闲PPU；通过`ppu-smi`核无进程、0%利用率，本case保守门限32MiB |
| 公共工具/设备接口 | 公共快照`ea25e9fc1ac38d6a76411b69c0927245e80b4ee2`的`dev/lib`及`dev/device-context/prototype/runtime`；只读冻结副本及逐文件SHA |
| 适配包 | 本地dev4候选wheel，SHA `90c6bf14a1cee050e507586faecd8f61ee91969f4018260794421969aaa9cffa` |
| FlagGems wheel | `flag_gems-4.2.1rc0-cp312-cp312-linux_x86_64.whl`，SHA `4b0ff0534346170950bda66dc74c48d9473ae00420dfb981c6aa1777ce21512f` |
| 控制解释器 | 宿主独立Python3.11 + PyYAML6.0.2，目录`<root>/control-venv`；宿主控制不加载torch |
| 输入 | shape32×64；[-5,5]等距输入、[-1,1]等距输出梯度；FP32/FP16/BF16，无模型权重 |
| 审核 | qualification输出须人工核对后另建review；routes只加载，不自动批准 |

先用`assets/prepare.py --public-root <固定公共checkout> --adapter-wheel <wheel> --gems-wheel <wheel> --output <新输入目录>`冻结输入（路由验证增加`--review <审核目录>`）。远端输入名使用`ppu-adapter-input-<日期>-<编号>`；解包前必须确认目标目录不存在，禁止把tar解压覆盖其他case的input02/04等通用目录。核对SHA后再运行；每轮新建输出，不覆盖旧轮。

单入口：

```sh
sudo <root>/control-venv/bin/python <input>/run.py --root <root> --run-id <唯一编号> --card <重新核对的物理卡号> --stage qualification
# 人工核对资格并保存审核后，使用含review的新输入：
sudo <root>/control-venv/bin/python <input-with-review>/run.py --root <root> --run-id <新唯一编号> --card <重新核对的物理卡号> --stage routes
```

runner映射一张卡，容器里固定逻辑0；最多480秒执行探针，异常也停止本人容器。输出含镜像检查、demo/环境基线、安装记录、探针结果、输入SHA与停止状态。共享镜像不下载、不更新；不碰他人容器。

## 验证

| 脚本 | 类别 | 判据 |
|---|---|---|
| prepare.py | 输入冻结 | 拒绝已有目标目录；复制固定依赖与SHA清单 |
| probe.py --stage qualification | 验证 | 2算子×3精度×3重复；前向/反向同时对照PPU原生和量化输入的CPU FP32参考；真实GEMS入口命中、dispatcher退出恢复 |
| probe.py --stage routes | 验证 | 每个已审核组合跑off/on/unqualified；on前反向各命中1次；其余目标0调用且原因正确；退出后再次原生前反向无目标命中 |
| run.py | 复现/验证 | strict环境检查、准确镜像和设备映射、有限执行、结果PASS、本人容器停止/Pid=0 |

FP32 `atol=1e-5, rtol=1e-4`；FP16 `2e-3/2e-3`；BF16 `2e-2/2e-2`，运行前固定，不因失败放宽。数值不通过就保留失败，不颁发该组合审核。执行异常停止，不在异常后盲目换原生。

完整日志保存在PPU本人目录`/bmcp_lvm_fs/cgu135/framework-ppu-20261110`；固定轮次、结果和本地备份见[HANDOFF](../../HANDOFF.md)。实际demo见[运行环境记录](demo/README.md)。
