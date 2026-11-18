# env 引用方式

本样例**不自建运行环境目录，直接引用现行环境目录**
[`dev/images/kunlun-infer-vllm/v1/env/v1`](../../../../images/kunlun-infer-vllm/v1/env/v1/)
（docker-compose.yml + env.baseline.yaml，由 baseline.py export 生成）。

选择理由（ENV-SPEC §3.2 允许使用方自建或直接引用，本样例取最简且合规的直接引用）：

- 该目录即本样例的验收对象环境（tele4 镜像 + 单卡授予/xpuctrl + FlagMemory 只读挂载
  PYTHONPATH 生效 + network none），无需任何裁剪或叠加；
- 不复制装配文件＝不产生第二份会漂移的真相；版本事实唯一来源 = 该目录
  `env.baseline.yaml`，由 `dev/lib/up.sh --strict` 机器核对（镜像 id、FlagMemory 挂载 commit）。

样例对环境的事实依赖（如需换环境，改 run.sh 的 `ENV_DIR` 并重跑）：

| 依赖 | 锚点 |
|---|---|
| 镜像 | `kunlunxin001-…-vllm0.20.2-…:tele4`（xVLLM 发布线），id/digest 见 env.baseline.yaml `image.id` |
| FlagMemory | `mounted_repos.FlagMemory.commit`（v0.3.0 发版提交 58e60a9；被测组件本体，PYTHONPATH 生效） |
| pytest | 镜像 conda env `python310_torch29_cuda` 自带（判据运行零 pip 依赖） |
