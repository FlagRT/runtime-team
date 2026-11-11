# 固定查询输入

`nfcorpus-50.jsonl` 是2026-09-23已保存的同一组50条NFCorpus查询，按query ID
排序，供后续检索比较与多请求检查复用。此文件逐字节复制原baseline的queries.jsonl。

原选择过程：将test query ID按字典序排列，用seed=42选50条，再按ID排列。
不得将200次检索测量解释为200条不同查询。

用同目录 `nfcorpus-50.sha256` 校验输入：

```bash
cd data/queries
sha256sum -c nfcorpus-50.sha256
```

新增OneRec完整工作流也使用这份原始文件。宿主机从`dev/rag_ljy`执行：

```bash
bash scripts/checks/run_generation_50.sh --backend elasticsearch --run
```

默认串行执行全部50条，每条只发送一次并开启生成；脚本不会切换当前API的数据库。
