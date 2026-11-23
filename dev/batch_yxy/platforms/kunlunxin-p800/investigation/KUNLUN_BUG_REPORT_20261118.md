# vllm-kunlun 0.11.0（昆仑芯 P800）推理输出错误缺陷报告

> 报告日期：2026-11-18 ｜ 状态：已定位触发条件与规避方案，待厂商根修
> 完整排查过程见同目录 [INVESTIGATION_20261117.md](INVESTIGATION_20261117.md)；
> 本文为自包含交付版：环境 + 可直接运行的复现脚本 + 实测输出。

## 1. 摘要

两个**相互独立**的缺陷导致推理输出错误，**均波及 embedding（pooling）与
文本生成两条路径**：

| # | 缺陷 | 触发条件 | 错误性质 | 规避 |
|---|------|----------|----------|------|
| 1 | prefix-caching 命中路径 | `enable_prefix_caching=True`（vllm 0.11 **默认开**）且长文本（多 KV 块）命中缓存 | 命中重算最后 token 时位置/偏移扭曲，输出半错（cos≈0.55）或生成从第 1 个 token 分叉 | `enable_prefix_caching=False` |
| 2 | 跨 prefill 调度步 | 单次调用总 token > `max_num_batched_tokens`（默认 16384） | 后续步序列块表指向无效内存，从垃圾数据计算（与一切正确输出近正交） | 单次调用总 token ≤16384 |

**已排除**：结果-请求映射错位（4×4 交叉余弦 + 顺序交换实验）、批间串扰
（全批交叉比对无 ≈1.0 匹配）、fp16 批数值噪声（0~1/32 重合，噪声应晚分叉
高重合）、uint8 位置回绕、单调长度阈值。
**定位**：昆仑芯算子层（kunlun_ops / vllm-kunlun 注意力后端）的**寻址/
元数据分支**——主路径算术逐位正确，仅上述两条特殊分支出错。

## 2. 环境

### 2.1 硬件与驱动

| 项 | 值 |
|---|---|
| 服务器 | 昆仑芯 P800 OAM ×8（单卡测试，容器绑定 /dev/xpu2） |
| 驱动 / XPU-RT | 5.0.21.47 / 5.0.21（`xpu-smi` 报告） |
| 宿主 OS | Ubuntu（内核 6.8），Docker + containerd |

### 2.2 容器与软件栈

```bash
docker run -d --name repro \
  --device /dev/xpu2 --device /dev/xpuctrl \
  -e XPU_VISIBLE_DEVICES=0 \
  -v <工作目录>:/work \
  chenyunlong/vllm-kunlun-runtime:0.11.0-xpytorch20260226 sleep infinity
```

| 组件 | 版本 |
|---|---|
| 镜像 | `chenyunlong/vllm-kunlun-runtime:0.11.0-xpytorch20260226` |
| vLLM | 0.11.0 |
| vllm-kunlun | 0.11.0 |
| kunlun_ops | 0.1.58+ee39020a |
| torch | 2.5.1+cu118 |
| Python | 3.10（`/root/miniconda/envs/python310_torch25_cuda/bin/python`） |

**运行配方**（所有复现脚本均以此环境变量运行）：

```bash
XPU_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
VLLM_PLUGINS=kunlun \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
/root/miniconda/envs/python310_torch25_cuda/bin/python <脚本>
```

### 2.3 环境周边的另外两个已知问题（非本报告主缺陷，但影响复现）

1. **EngineCore 子进程起不来**：`KunlunPlatform` 未实现 `set_device`，且
   spawn 子进程内插件激活时序缺陷使 `current_platform` 落到基类 →
   `NotImplementedError`。规避即上述 `VLLM_ENABLE_V1_MULTIPROCESSING=0`
   （进程内引擎）；或在容器内为 `vllm_kunlun/platforms/kunlun.py` 的
   `KunlunPlatform` 补一个 `set_device = torch.cuda.set_device` 方法。
2. **本地模型路径误判**：模型目录带 sentence-transformers 根配置
   （`config_sentence_transformers.json`/`modules.json`）时，
   `vllm/transformers_utils/config.py` 的 `get_hf_file_to_dict` 对本地
   路径调 `hf_hub_download` 且异常捕获列表缺 `ValueError` → 直接崩溃。
   规避：使用剔除 st 配置的目录（权重本体不变）。

### 2.4 模型

| 用途 | 模型 | 说明 |
|---|---|---|
| 主复现 | Qwen3-Embedding-0.6B（fp16，last-token pooling） | 本机 ModelScope 副本，sha256 校验一致 |
| 生成路径验证 | 同一权重（Qwen3ForCausalLM，tie embeddings） | 检查点键名重映射为标准布局（`embed_tokens.*` → `model.embed_tokens.*`），映射脚本见 §5.3 |

## 3. 缺陷一：prefix-caching 命中输出错误

### 3.1 最小复现（embedding，可直接运行）

```python
"""bug1_repro.py — 同一 300tok 文本连发两次（缓存开，vllm 0.11 默认）。"""
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM

MODEL = "<本地 Qwen3-Embedding-0.6B 路径（剔除 st 配置）>"
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, task="embed", max_model_len=4096,
          dtype="float16", enforce_eager=True)  # 不传 enable_prefix_caching → 默认开

words = ["甲"] * 308
text = " ".join(words)
while len(tok.encode(text)) > 300:            # 构造恰好 ~300 token
    words = words[:-1]; text = " ".join(words)

def E(t):
    return np.asarray(llm.embed([t], use_tqdm=False)[0].outputs.embedding)

v1, v2 = E(text), E(text)                      # 第一次冷缓存；第二次全量命中
cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
print(f"cos(v1, v2) = {cos:+.4f}")             # 期望 1.0；实测 +0.546
```

**实测输出（2026-11-17，P800 xpu2）**：`cos(v1, v2) = +0.5461`。
**CPU transformers 真值判定**：`cos(CPU真值, 第一次) = +1.0000`（冷缓存
正确）；`cos(CPU真值, 第二次) = +0.5462`（命中路径错误）。

### 3.2 触发边界

- 短文本（约 ≤255 tok / 单~数个 KV 块）实测不受影响
- 同批组成矩阵：损坏值跨所有批组成（2/3/4 条、任意顺序、缓存冷热）确定
  复现（同一文本恒为同一错误向量）→ 确定性错误，非随机

## 4. 缺陷二：跨 prefill 调度步输出错误

### 4.1 可靠复现（embedding，缓存关，新鲜会话即触发）

```python
"""bug2_repro.py — 64 条混合长度一次调用（总~55k tok ≈ 4 个 prefill 步），
对照各自单条调用。缓存关闭以隔离缺陷二。"""
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM

MODEL = "<本地 Qwen3-Embedding-0.6B 路径>"
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, task="embed", max_model_len=4096,
          dtype="float16", enforce_eager=True, enable_prefix_caching=False)

def make(target, seed):                        # 构造 ≈target token 的重复词文本
    words = [seed] * (target + 8); text = " ".join(words)
    while len(tok.encode(text)) > target:
        words = words[:-1]; text = " ".join(words)
    return text

lengths = [64, 96, 128, 160, 200, 256, 300, 384, 448, 512,
           640, 768, 900, 1024, 1200, 1536, 1800, 2048] * 2
lengths += [64] * 14 + [2048] * 14             # 64 条，两端加密
texts = [make(L, f"w{i}") for i, L in enumerate(lengths)]

refs = [np.asarray(llm.embed([t], use_tqdm=False)[0].outputs.embedding) for t in texts]
V = [np.asarray(o.outputs.embedding) for o in llm.embed(texts, use_tqdm=False)]

bad = sum(1 for v, r in zip(V, refs)
          if float(v @ r / (np.linalg.norm(v) * np.linalg.norm(r))) < 0.999)
print(f"损坏 {bad}/{len(texts)}")              # 期望 0；实测 35/64（两次独立会话 35/64、52/64）
```

**实测输出**：`损坏 35/64`（另一会话/更小步预算下 52/64）。

### 4.2 触发边界（边界矩阵，缓存关）

| 调用形状 | 总 token / 步数 | 损坏 |
|---|---|---|
| 64×64tok | 4k / 1 步 | **0/64** |
| 8×2048tok | 16k / 1 步 | **0/8** |
| 16×2048tok | 32k / 2 步 | 8/16（恰为第 2 步全部） |
| 32×2048tok | 64k / 4 步 | 24/32 |
| `max_num_batched_tokens=4096` 下 64 混长 | ~55k / 13 步 | 52/64（步越多坏越多） |

- 与批内条数无关（单步内 64 条全对）、与长度混合无关（同长度跨步同样坏）
- **会话布局依赖**：同形状调用在无批量调用历史的会话中可全部正确（0/16、
  0/32 复测通过）——触发取决于块分配器空闲布局

### 4.3 错误性质（铁证）

16×2048 两步调用：第 0~7 条（步 1）全对，**第 8~15 条（步 2）8/8 全坏**。
把每个损坏向量与批内全部 64 条参照交叉比对：**无任何非对角 ≈1.0**
（交叉 cos 仅 0.01~0.12）→ 排除"读到其他序列 KV"的串扰；损坏向量与
一切正确嵌入近正交 → **从无效/未初始化/已释放 KV 块计算**。

## 5. 生成路径验证（两缺陷均波及 generate）

### 5.1 缺陷一 × 生成

同一 400tok 提示，`SamplingParams(temperature=0, max_tokens=32)` 贪心
解码两次（缓存开）：

```
第一次: [100043, 5185, 1418, 3818, 276, ...]
第二次: [100043, 99436, 13072, 104650, ...]   ← 第 1 个实质 token 起分叉
```

### 5.2 缺陷二 × 生成

缓存关，16×2048 提示批量调用（2 步 prefill + 多步 decode）vs 逐条单独
调用：**16/16 输出不一致，全部从第 0~1 个 token 完全分叉（32 token 仅
0~1 个相同）**——排除 fp16 批数值噪声；解码阶段的跨步 KV 读取使步 1
序列也全坏（embedding 场景步 1 完好，因其无解码阶段）。

### 5.3 生成实验用检查点重映射脚本

```python
"""remap_ckpt.py — embedding 检查点键名重映射为标准 Qwen3ForCausalLM 布局。"""
import json, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

SRC, DST = Path("<embedding 模型目录>"), Path("/root/qwen3-gen")
DST.mkdir(parents=True, exist_ok=True)
sd = load_file(str(SRC / "model.safetensors"))
save_file({"model." + k if k.startswith(("layers.",)) or k in
           ("embed_tokens.weight", "norm.weight") else k: v for k, v in sd.items()},
          str(DST / "model.safetensors"), metadata={"format": "pt"})
cfg = json.loads((SRC / "config.json").read_text())
cfg["architectures"] = ["Qwen3ForCausalLM"]; cfg["tie_word_embeddings"] = True
(DST / "config.json").write_text(json.dumps(cfg, indent=2))
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "generation_config.json"):
    shutil.copy(SRC / f, DST / f)
```

（embedding 结论均在原始检查点上取得；重映射仅为让 generate 路径能加载，
非缺陷来源。）

### 5.4 尚未完成验证

`classify()` / `score()`（pooling 家族另两个入口）与 `vllm serve` 在线
服务模式未完成实测（score 入口用 bge-reranker-base 的测试中断于引擎
初始化）；按机制推断同样受影响，待补测。

## 6. 定位建议（给厂商）

排查 vllm-kunlun 的 prefill/attention 路径在以下两分支的**块表寻址与
偏移处理**（含适配层传入的元数据，非算术运算——主路径算术逐位正确）：

1. **prefix-caching 命中重算**分支（`num_cached_tokens` > 0 时最后 token
   隐状态的重算：位置编码/块表偏移）
2. **跨调度步**分支（同一次请求批次内第 2+ 个 prefill 步的序列块表
   续用/分配）

严重程度提示：vllm 0.11 默认开启 prefix caching，且在线服务持续多步
调度——按 §5 实测，**默认配置下的重复提示与多步批量场景，生成内容与
嵌入输出均不可信**。

## 7. 规避方案（已实测有效）

| 措施 | 覆盖 |
|---|---|
| `enable_prefix_caching=False` | 缺陷一 |
| 单次调用总 token ≤ 16384（外层组批预算钳制，如 FlagScheduler `max_total_tokens`） | 缺陷二 |
| 两者叠加的安全包络实测：64 条任意混长批**逐位正确**（cos=1.0） | — |

## 8. 证据索引

| 证据 | 位置（FlagScheduler 仓） |
|---|---|
| 结论性数据 JSON（含边界矩阵/铁证/生成路径/历史翻案） | `probes/results/xpu_pooling_defects_20261117.json` |
| 首轮探针 + 结果 | `probes/qwen3_embed_xpu_batch_composition_probe.py`、`probes/results/xpu_batch_composition_20261110.json` |
| 完整排查过程（含证伪假说） | `reports/kunlunxin/INVESTIGATION_20261117.md` |
| 平台数据与基准 | `reports/kunlunxin/README.md`、`reports/kunlunxin/data/` |
| 相关提交 | b68cd22 → 412b7fb → dd8aeb9 → 833cbc5（+本文） |
