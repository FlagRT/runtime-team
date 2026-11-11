# 本地收尾核验（2026-09-16）

- `bash dev/framework-adapter/probes/run_checks.sh local`：Syntax PASS，10 Python / 4 shell；只检查语法，不计为功能或设备测试。
- `bash dev/framework-adapter/probes/run_checks.sh metadata-tests`：12 tests，OK。覆盖 padding、容差选择、显式设备、已有结果保护、NPU 懒加载顺序及初始化失败不得自动转 CPU。
- `git diff --check`：通过。
- 最终远端结果归档已下载并核对：CPU JSON 为 pass，2 组模型输入、8 组算子用例，最大绝对误差均为 0；NPU JSON 为 fail，stage=device_initialization、visible_devices=0，未运行模型用例。
- 本地 qwen_embedding_baseline.py 与远端最终 qwen_embedding_baseline-v3.py 的 SHA256 一致：`8efae2768dc9fa99d07704e01919c855f49f7ce8a643858996404be06a536aef`。
- `final-state-v3.txt` 确认本人容器为 exited；SSH 会话已退出。没有遗留本轮后台计算任务。
- 本轮新增代码、文档和证据未创建提交、未推送；个人分支只有合入公共基线的本地 merge。旧有 .DS_Store 未跟踪文件保持不动。

这些核验不等于 NPU 模型通过、vLLM 服务通过或三级安全回退完成。
