# 证据索引：壁仞 BR_0 conformance 13+6 与多流 16 项基线（2026-11-11）

> 采集时刻：2026-11-11 14:49–14:56 CST ｜ 载体：`wz-server008` 容器 `dc-biren-hliu553`
> 镜像：`birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx`
> 镜像 ID：`sha256:081634a335d48760cb10fb240bf3655530b0e2748a46d836e097e6006953281a`（本地无 digest）
> 设备：`/dev/biren/card_5`（PCI `85:00.0`）+ `/dev/biren/card_7`（PCI `a5:00.0`），容器内 `device_count=2`

## 目录

| 目录 / 文件 | 内容 | 判定串 |
|---|---|---|
| `biren_conf_20261111_144929/conf13.log` | conformance 13 例逐例输出（真机） | `CONFORMANCE_PASS` 13/13 |
| `biren_conf_20261111_144929/conf13.json` | 同上，机读结果（`verdict/passed/total/env`） | 同上 |
| `biren_conf_20261111_144929/conf6.log` | 推理 6 例逐例输出 | `CONFORMANCE_PASS` 6/6 |
| `biren_conf_20261111_144929/conf6.json` | 同上，机读结果 | 同上 |
| `biren_stream2_20261111_145225/stream_biren2.log` | 多流探针（8 子项，修正 S12 文案后重跑） | `STREAM_SEMANTICS_PASS` 8/8 |
| `biren_stream2_20261111_145225/stream_semantics_full_result_biren.json` | 同上，机读逐项 detail | 同上 |
| `biren_stream2_20261111_145225/i6w.log` | i6 连续三次调用（冷 / 预热后 ×2），重叠率观测 | 3 次 `ok=True` |
| `biren_facts_20261111_145533/facts_s12_i6_20261111_145533.log` | S12 流优先级三问（构造/回读/范围）+ i6 随工作量缩放 | 见报告 §5.1 / §5.2 |
| `biren_facts_20261111_145533/pp2.py`、`ov_scale.py` | 上条日志对应的探针源码（容器内原样副本） | — |

## 复跑脚本（`scripts/`，宿主侧执行，均过 `bash -n`）

| 脚本 | 作用 | md5 |
|---|---|---|
| `biren_conf_run.sh` | 同步代码 → conformance 13 例 → 推理 6 例 → 取回 JSON/日志 | `71006450d081ff40ee1cc6fa3755f55e` |
| `biren_stream_run.sh` | 多流探针首跑（S12 文案修正前，保留作反证） | `bb0b162028af1d3ef374348184c8a865` |
| `biren_stream_rerun.sh` | 多流探针重跑 + i6 连续三次调用 | `6fde087f872b73ffa63d49c8fd86801e` |
| `biren_facts_run.sh` | S12 优先级三问 + i6 缩放，落归档日志 | `d6beb15084ebb9c25135629d5db6ad50` |
| `biren_priority_probe.py` | 优先级：命名空间/类入口、构造、回读 | `8e151add14d8c8037fe5c7d546885299` |
| `biren_priority_range_probe.py` | 优先级：`priority_range()` + 档内逐值设置-回读 | `83030299176730affc7bd738ffdd0027` |
| `biren_ov_scale.py` | i6 口径随 n/batches 缩放（预热后计时） | `c3639c364c7d96c39d79c4084bcb1fe5` |

## 证据文件 md5

```
ee7e409695f185b7fa83fd1c0056d6cb  biren_conf_20261111_144929/conf13.json
212ebd9d300e47eef48beb6a4f0b69dd  biren_conf_20261111_144929/conf13.log
6ea5c6c2bab77de942a5f959c7ea5a39  biren_conf_20261111_144929/conf6.json
89071877b73af286be9345980650550b  biren_conf_20261111_144929/conf6.log
833de6e764e643791156dfbb6e54e5a3  biren_facts_20261111_145533/facts_s12_i6_20261111_145533.log
922031ef117235eb89c8332c4d4febda  biren_stream2_20261111_145225/i6w.log
780d0bdfcf7e3584dbe8427c8a5cff0e  biren_stream2_20261111_145225/stream_biren2.log
074b5053c19ae9344a37ecfc62eb9143  biren_stream2_20261111_145225/stream_semantics_full_result_biren.json
```

## 运行前置（缺一即假失败）

```bash
# 容器内任何 python 调用前都必须 source 厂商环境脚本（否则 import torch_br 报 libsupti.so.1 缺失）
. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set
cd /work/dc/prototype
python3 runtime/conformance/runner.py --backend biren --out <out>/conf13.json
python3 runtime/conformance/runner.py --backend biren --cases infer_cases --out <out>/conf6.json
DC_BACKEND=biren DC_TAG=_biren DC_OUT_DIR=<out> python3 probes/probe_stream_semantics_full.py --rounds 5
```

> 期望读数：`CONFORMANCE_PASS`（13/13 与 6/6 各一次）、`STREAM_SEMANTICS_PASS 8/8`、两轮 `rc=0`。
> 若见 `CONFORMANCE_ABORT`，先查是否漏了 `source brsw_set_env.sh`（不是芯片问题）。
