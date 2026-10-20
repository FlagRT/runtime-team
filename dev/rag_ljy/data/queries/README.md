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
