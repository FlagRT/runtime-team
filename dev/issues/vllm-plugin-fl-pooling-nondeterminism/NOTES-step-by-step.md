# 从零到判定：逐步操作

> 正常使用请直接跑 [`run.sh`](run.sh)（`XPU_CARD=<n> MODEL_DIR=/path bash run.sh`）。
> 本文件是给要手动排查的人看的。

前置步骤与 [`vllm-plugin-fl-pooling-dispatch/NOTES-step-by-step.md`](../vllm-plugin-fl-pooling-dispatch/NOTES-step-by-step.md)
的 1-5 步完全一致（起容器、export 网络变量、确认模型目录、打 dispatch 补丁）。

6. 补丁打完、确认 `EMBED_OK` 之后，跑本 case 的 `record_pooling_cos.py`（默认 + `--disable-prefix-caching`
   两次），记下两次的 `COS = ...` 输出。
7. 这是收数脚本，不是判定脚本，跑完直接看数值即可，不用等退出码之外的其他信号。
8. 收尾同上一个 case：不用恢复容器内文件改动，不碰其他用户容器。
