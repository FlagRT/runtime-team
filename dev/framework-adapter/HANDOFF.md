# framework-adapter · 开发续接

更新：2026-09-23。先读[STATUS](STATUS.md)，只在需要重现时读日期报告；本文件不再重复测试成绩。

## 工作区与协作

- 本人分支：`cgu135/framework-adapter`；09-23用户授权将09-22原型与证据按团队流程提交、合并推送。用户`.DS_Store`文件保留、不提交。
- 09-23发布前fetch公共分支到`6af89c4`，含`8cc9dbe`设备原型去torch_fl更新；发布同步不改变下表服务器0916运行副本，也不替代新基座回归。具体提交及合并位置查看Git历史，执行前重新fetch核对。
- 09-17历史成果已通过merge/push进入公共`dev-1.0@e9fd2ff`；PR #16关闭，不走原PR流程。后续按[根README](../../README.md)标准流程，在个人分支开发、同步解决冲突后合并；是否提交/推送以当轮授权为准，禁止强推共享分支。
- runtime-team保留原型、探针、部署和证据；正式子库代码独立维护。旧`vllm-plugin-FL`个人分支`cgu135/safe-op-fallback@635ff6d`不会随本仓推送，也不自动装入当前镜像。

## 最近使用的环境（运行前重新核查）

| 项目 | 记录 |
|---|---|
| 当前允许主机 | 27 / 10.120.72.11，账号cgu135；不要沿用旧五机列表授权 |
| 实测主机 | npu1-27 / 10.120.72.27，驱动26.1.1；11的Docker权限尚不足 |
| 本人容器 | flagos-proto-infer-910c，owner=cgu135，非privileged，单davinci0→npu:0 |
| 最后状态 | 09-22 18:17:35 stopped，SSH退出；[记录](docs/evidence-model-20260922/rms-final-state-20260922.txt) |
| 持久工作目录 | 宿主/home/cgu135/framework-adapter-910c/acceptance-20260916 → /work |
| 模型只读挂载 | /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B → /model |
| 初始化runtime | /work/dev/device-context/prototype，仍为0916保存副本，并非整个最新公共代码 |
| 依赖 | /work/FlagGems-f7ae8e6-20260922/src；/work/deps-20260922；仅目录覆盖，未改公共镜像 |

完整镜像digest、包版本及已知差异只维护在[实验报告§2](docs/910C模型接入与安全回退-20260922.md)与[证据](docs/evidence-model-20260922/README.md)。未引入FlagTree构建物。凭证不写记录，需要时读取用户连接材料。

## 接着运行

1. 重新检查资源、分配规则、容器归属、设备映射和Git状态；不能因上次空闲直接启动。
2. 在本人已准备好的容器内使用报告§5的环境变量；模型与回退测试看§5，RMSNorm诊断看§8。本地统一入口及代码分工见[README](README.md)。
3. 使用新结果文件名和有界超时；目标执行错误不吞掉、不自动重试/重置设备。不得停止他人任务。
4. 完成后下载原始结果及源码哈希、更新同主题日期报告，再更新STATUS。本人容器按实际使用收尾停止，记录状态；不擅自上传OneDrive或Git。

服务器文件名映射（不要覆盖旧失败快照）：

| 仓库probes/文件 | 容器/work文件 |
|---|---|
| qwen_embedding_baseline.py | qwen_embedding_baseline-v3.py |
| qwen_gems_validation.py | qwen_gems_validation-v3.py |
| qwen_scoped_adapter.py / test_qwen_scoped_adapter.py | 同名 |
| qwen_rms_shadow.py / qwen_rms_ablation.py | 同名 |

rms-shadow的completed只表示统计收集成功；rms-ablation要另看within_tolerance。模型最终向量通过不能抵消隐藏状态失败；不扩大默认准入，不放宽阈值掩盖问题。

## 历史与维护

下轮vLLM服务对接优先复用统一启动脚本与规范。09-22审查的是[个人分支脚本固定版本](https://github.com/FlagRT/runtime-team/blob/bdefeeea61f9d0f951f026e72990f672f0da5257/dev/device-context/prototype/scripts/serve_standard.sh)和[规范](https://github.com/FlagRT/runtime-team/blob/bdefeeea61f9d0f951f026e72990f672f0da5257/dev/device-context/prototype/docs/SERVICE_STARTUP_STANDARD_20260920.md)；09-23公共6af89c4已有prototype同步，后续以公共最新版本重新核对。已审查版本支持SERVE_FORM=embed；需明确MODEL/SERVED_NAME/DEV及日志目录。其清理按进程名范围匹配，不得直接在共享服务环境使用；本方向尚未运行。FAIL可能仍退出0，核验必须看ready/smoke/verdict。厂商服务已跑通不等于我们的FlagGems接入已完成。

旧环境、旧阻塞和此前步骤保存在[整理前快照](docs/历史入口快照-20260922.md)及[日期报告索引](README.md)，不在此追加流水账。新结论写STATUS，技术细节写报告，环境发生变化才改本文件。
