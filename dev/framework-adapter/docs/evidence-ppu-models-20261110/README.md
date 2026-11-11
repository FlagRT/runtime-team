# PPU两模型证据索引

人读结论见[报告](../PPU两模型接入与安全准入验证-20261110.md)。

- summary.json：13轮目录的原始归档摘要、输入SHA及容器状态；candidate失败保留ERROR/fail，不改成通过。
- verification.json：本地归档与结果核验。它的PASS表示证据核验通过，不表示候选数值通过。
- evidence-sha.json、archive.sha256：159份远端轻量证据及归档身份；完整文件在本地outputs/framework-runs/20261110-ppu-models/evidence。
- qwen-review-failed/、onerec-review-failed/：单算子资格、模型失败原件及failed审核；off/unknown为模型硬件使用的同内容配置，denied/strict-denied只做了本地CPU决策验证。
- 两份failed-review-control.json：各2项CPU拒绝路径；两份controls-final.log：同一8项case控制在两个冻结输入上的执行。
- cleanup-1110.log：12个本人测试容器最终状态。大张量留服务器，其逐文件SHA清单在本地collection01，不在此重复存放。

全量原始输入、wheel、日志、tar与中断证据均保留原路径；本目录只放便于审阅的材料。发布状态看STATUS。
