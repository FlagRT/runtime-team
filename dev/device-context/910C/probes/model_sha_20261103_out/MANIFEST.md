# 三机模型副本 sha256 对账（2026-11-03）—— 定「区分度 0.0011 差」的来源

> 问题：910C 推理腿区分度 **0.6380**，P800 / MLU590 均为 **0.6391**（差 0.0011）。
> 候选解释：① 模型副本不同 ② 后端数值差异。
> 判据：对三机的 `Qwen3-Embedding-0.6B` **逐文件 sha256**（HF 缓存的 `snapshots/*` 是软链 ⇒ 脚本**跟随链接取目标内容**）。

## 结论：权重完全一致 ⇒ 排除「模型副本」假设

```
910C weight_sha256= 0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd  size=1191586416
P800 weight_sha256= 0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd  size=1191586416
MLU  weight_sha256= 0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd  size=1191586416
```

`model.safetensors` **三机逐字节相同**；`config.json` / `tokenizer*.json` / `vocab.json` / `merges.txt` /
`modules.json` / `1_Pooling/config.json` / `README.md` 也都一致（见 `compare.txt`）。

⇒ 0.0011 的区分度差**不能归因于模型副本**，应归因于**后端数值/算子实现**（三方架构与 torch 版本不同：
910C torch 2.10.0 + torch_npu / P800 torch 2.9.0 / MLU590 torch 2.7.1 + torch_mlu）。
本次**不对具体算子下结论**（未做逐算子对照），只关闭「副本不同」这一分支。

## 唯一差异 = 910C 多出 4 个非权重文件 + 1 个我们留下的坏件

| 文件 | 910C | P800 | MLU | 说明 |
|---|---|---|---|---|
| `.msc` / `.mv` | 有 | 无 | 无 | ModelScope 侧的元数据文件；不参与权重加载 |
| `configuration.json` / `generation_config.json` | 有 | 无 | 无 | 同上，910C 用的是**整包下载**形态；HF 缓存形态不含 |
| `model.safetensors.CORRUPT-20261027` | 有（sha `dba4a6dd…`） | 无 | 无 | **10-27 发现的静默损坏件，按纪律改名留证、未删除**；当前加载的 `model.safetensors` 是干净副本 |

## 文件

| 路径 | 说明 |
|---|---|
| `sha_910c.json` · `sha_p800.json` · `sha_mlu.json` | 三机逐文件 sha256 / size / 是否软链 / 链接目标 |
| `compare.txt` | 三机逐文件并排对照（含差异标记） |
| `weight_sha.txt` | 三机权重指纹一行一条（便于对外引用） |
