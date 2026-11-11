# 数值 case 逐步操作手记（kunlunxin-gemm-numeric-64cube）

从零到判定的完整操作序列（README 只留单入口，过程与踩坑记这里）。

## 2026-10-28 首次取证（env/v1 发布形态）

1. 仓根执行 `FLAGRT_MODELS_DIR=/data2/xliu969 dev/lib/up.sh dev/images/kunlun-operator-runtime/v2/env/v1 --strict`
   —— 两项核对 PASS（image id `cac5ddae…` + 挂载 FlagGems commit `4a00b845`），容器 `flagrt-kunlun-v2-env-v1-p800`。
2. 装置自证：`bash …/env/v1/verify-env.sh` —— 4 项 PASS（enable 环境变量 / compiler
   版本锁 `7133710c` / diagnostic 副本挂载 / 补丁版 FlagGems 生效）。
3. 选卡：全 8 卡有占用，取卡 0（3GB 残留 / 0% util；卡 1 有坏卡嫌疑避开，
   卡 2/3 满载，卡 5/6/7 大进程占用）。当次窗口卡 0 可用，全程无挂死。
4. 三连跑 64³：`source …/env/v1/enable-gemm-compat.sh && for i in 1 2 3; do
   CUDA_VISIBLE_DEVICES=0 python /tmp/probe_64_cube.py; done` —— 六读数 rel 0.79~1.00
   （evidence/env1-three-runs-20261028.txt）。
5. 收尾：`WORKSPACE_ROOT=<仓根> FLAGRT_MODELS_DIR=… docker compose -f …/env/v1/docker-compose.yml down`；
   `docker ps` 确认无残留。

## 踩坑与注意

- **pytest 必须用补丁树的 tests**（`/workspace/FlagGems/tests/…`）：镜像内
  `/env/FlagGems/tests` 的 conftest 会把 `/env/FlagGems/src` 插回 import 路径最前，
  表现为回到未修复形态（ttxir 族失败）。误用时读数无效。
- **多子进程工具**（minirepro 类）不继承 monkey-patch：env/v1 的 sitecustomize 已在
  启动期做全套启用，source 过 enable 的 shell 内直接跑即可；未 source 时单脚本头部
  `import gemm_compat` 兜底（verify_numeric_64cube.py 已内置）。
- 选卡纪律：`xpu-smi` 看残留占用，避开有残留上下文的卡（编译期新上下文遇残留可能
  CPU 满转挂死，SIGTERM 无效须 SIGKILL）；探针后查 `xpu-smi` Processes 清孤儿。
- compose down 需要 `WORKSPACE_ROOT` 显式给值（变量无缺省，直接 down 报
  invalid spec）。
