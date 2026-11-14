# evidence

run.sh 执行时产出 `summary.json`（机器可判：用例名→pass/fail、计数、镜像 id、FM commit 锚、
容器内实装版本）到本目录；完整日志留在执行机器 `/tmp/connector-kunlunxin-<时间戳>/`，不进仓。

当前状态：2026-11-15 P800-A 首次真机执行，`summary.json` result: FAIL（2/5；环境与导入面全绿，阻塞点=机器级 D2H 拷贝路径问题，诊断见 ../REPORT.md）。
