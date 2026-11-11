# 2026-11-06 运行环境快照

`image.baseline.yaml` 来自公共提交 0da9977086d07a867bf915b207f08e70f541683c；`demo/env.baseline.yaml` 由该版 baseline.py export 生成，随后 up.sh --strict 成功启动。生成文件保持原字节。

这是本次验收记录，里面的宿主路径和容器名是历史实际值，不是新的资源预约。重跑在本人新目录准备输入并重新生成运行基线，避免覆盖本轮结果。

- 物理卡 14 → 逻辑 0；容器 cgu135-cann-shmem-1106；12GiB 内存上限、4 CPU。
- 镜像保持 e13a15d0…；容器内新增包按 [runtime-overlay.json](runtime-overlay.json) 识别。基础 env.baseline 只核镜像与挂载，不自动校验后来 pip 装入的包。
- 修复叠加层用独立下载 SHA、安装报告、前后包清单核对；禁止只凭镜像 ID 认定环境相同。
- 容器已经停止，保留供查证；完整结果见[报告](../../../../docs/cann-shmem修复验证-20261106.md)。
