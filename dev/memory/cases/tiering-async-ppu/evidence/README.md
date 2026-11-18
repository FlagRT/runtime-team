# evidence

run.sh 执行时产出 `summary.json`（机器可判：用例名→pass/fail、计数、镜像 id、commit 锚、
层包实装版本）到本目录；完整日志留在执行机器 `/tmp/<case>-<时间戳>/`，不进仓。

当前状态：已真机执行（2026-11-15 16:36，PTG_0 / PPU0，宿主 = flagrelease vllm0.24
定档镜像，见 `summary.json` 与 ../REPORT.md）——**PASS 3/3**（T-8 负例 / 异步轨全链 /
destroy 排空）。
