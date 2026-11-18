# 两库完整RAG显存水位测试

入口为 `scripts/checks/run_dual_memory_watermarks.sh`。复用已经运行的8088、8089和8090，不重启服务、不改模型配置或85%/75%/5GiB规则。每条请求显式选择Milvus或Elasticsearch，执行向量化、检索、精排和OneRec vLLM生成。

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
# 只读预检，不发送查询
bash scripts/checks/run_dual_memory_watermarks.sh
# 依次测试Milvus和Elasticsearch，各自覆盖同时、错峰、持续到达
bash scripts/checks/run_dual_memory_watermarks.sh --run
```

单独执行某库或某模式：

```bash
bash scripts/checks/run_dual_memory_watermarks.sh --run --backends milvus
bash scripts/checks/run_dual_memory_watermarks.sh --run --backends elasticsearch --modes burst
```

## 负载与边界

- 只用真实RAG请求制造负载，不额外分配占位显存。输入来自 `experiment_outputs/inputs/request_pressure/queries.jsonl` 的NFCorpus衍生长查询；不是原始50条短查询。构造后的实际输入另存 `inputs.json`。
- 默认按目标查询token数/请求精排batch逐级增加：1024/8、1024/16、2048/16、2112/32、2304/32、3072/32。token数按embedding tokenizer构造，实际精排拼接长度、准入调整后的batch以HTTP记录为准。
- 同时到达：每档4个客户端同时释放发送；错峰：每档4条，间隔0.2秒；持续：每档3秒，间隔0.2秒，最多16条。客户端最多16个在途任务，输出上限256 tokens。通过 `--count`、`--spacing`、`--continuous-seconds`、`--max-load-requests`、`--client-limit` 调整，实际参数保存在manifest中。
- 前一档完成且预算释放后才增加下一档。任一die达到85%即停止加压，继续用小型完整RAG请求检测拒绝和恢复；每个探测请求只发送一次，不重试503。
- 每个库/模式默认观察窗口180秒，单请求超时180秒，另留180秒等待在途工作释放。窗口不是杀死服务器请求的超时。
- 显存低于5GiB、遥测过期、估算器故障或异常HTTP时停止发送；保留服务和已接收请求，不主动制造OOM。交互终端Ctrl-C停止提交并等待清理，不能保证立即返回。

## 判定与输出

两颗die分别验收，不能互相替代：die7向量化/精排，die6生成。每颗die均须具有：

1. 实测占用达到85%。
2. 服务端该die准入快照证明高水位导致HTTP503，并带Retry-After。
3. 随后回落到75%～85%仍有该die压力503。
4. 再降至不高于75%后完整生成请求成功。
5. 高水位前已接收的请求在压力期间继续完成生成。

两库三模式、两颗die全部满足才报告总结果 `passed`。未达到阈值或证据缺失为 `incomplete`（退出码2）；异常或安全检查失败为 `failed`（退出码1）。预算不足的503单独统计，不算高水位拒绝。

OneRec vLLM当前预分配约55%的设备显存，增加请求可能只复用现有KV池，**不保证die6能达到85%**。这种情况如实记录 `threshold_not_reached`，不能据此声称双die验收通过。测试保留各轮自然缓存状态；缓存未降至75%时后续轮次标记未执行，不靠重启制造恢复。阶段预算也可能让大请求在实测85%前被拒绝。

输出位置：

`/home/jliu171/jliu171_data/hetero_data/rag/experiments/checks/dual_memory_watermarks/<UTC时间>/`

容器内对应 `experiment_outputs/checks/dual_memory_watermarks/<UTC时间>/`。

- `manifest.json`：命令、配置、初始健康状态、软件及模型信息、输入SHA。
- `source/`、`inputs.json`：执行代码快照与实际输入。
- `<数据库>/<模式>/events.jsonl`：实际显存采样、请求内容、HTTP/响应头、准入快照、结果及停止原因。
- `<数据库>/<模式>/summary.json`：每颗die的独立证据及结果。
- `summary.json`：整个验收矩阵。

请求到达模式、实测显存占用分别记录；本脚本不测NPU计算利用率，不把请求数或显存比例当作计算利用率。当前状态见[status.md](../status.md)。
