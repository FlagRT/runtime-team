# evidence

run.sh 执行时产出 `summary.json`（机器可判：用例名→pass/fail、计数、镜像 id、commit 锚、
层包实装版本）到本目录；完整日志留在执行机器 `/tmp/<case>-<时间戳>/`，不进仓。

当前状态：已真机执行（2026-11-15，PTG_0 / PPU0）——**FAIL 0/3**（3 用例 setup ERROR：
FM offload/ppu import 链需 vllm 0.24 符号，本环境 vllm 0.19 线不满足；详见
`summary.json` 与 ../REPORT.md「通过态」）。
