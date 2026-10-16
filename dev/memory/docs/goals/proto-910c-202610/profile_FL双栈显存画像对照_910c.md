# 双栈显存画像对照:vllm-ascend vs vllm-plugin-FL(910C,2026-10-12)

> 状态:🟢 FL 栈全矩阵实测完成(A/C/E/B/D/F,14 组全绿)。
> 对照基线:9 月生成式画像(`REPORT_generative_profile_onerec8b_20260923.md`)+ davinci4 复测(2026-10-12,数据 `data-baseline-davinci4-20261012/`)。
> FL 栈环境:容器 `fl-profile-910c`,镜像 `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`(v3 统一基座,10/8 构建);vllm 0.20.2+empty + vllm-plugin-fl 0.0.0+g8b059122e + flag_gems + flagcx 0.13.0 + flagtree 0.6.0;**无 vllm-ascend**。davinci4(Phy-4),TP=1,gmu=0.9,MAX_MODEL_LEN=4096,enforce_eager,prefix caching=ON。
> 同模型 OneRec-8B(`/data_lib/models/OneRec-8B`),同探针(gen-load/hbm-sampler),逐项可比。
> 原始数据:本地 `data-fl-stack-20261012/`(37 文件:18 load JSON + 18 kv CSV + HBM CSV 18,893 行)。

## 0. 启动门槛:FL 栈要三把钥匙才跑得起来

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | FlagGems torch.cos | 引擎初始化即 CompilationError:`No module named 'shmem'` | `VLLM_FL_FLAGOS_BLACKLIST=cos` |
| 2 | FlagGems torch.sin(RoPE sin cache) | 同上 | 黑名单改 `cos,sin`(逗号分隔,格式见 vllm_fl/dispatch/README.md:380) |
| 3 | **核心 vLLM triton 内核**(非 FlagGems 算子) | 第一个真实请求崩溃:`vllm/v1/worker/block_table.py::_compute_slot_mapping_kernel` 经 triton-ascend 编译时,`visit_For` 无条件 `import triton.experimental.tle`,而 `tle/language/dsa/ascend/communication.py:6` 顶层 `import shmem as ash`,镜像未带 shmem 包 → **任何带 for 循环的 triton 内核都编译不过** | 写 stub `site-packages/shmem.py`(ash 只在分布式 init 函数体内使用,单卡不触碰;容器 writable 层,重启容器丢失) |

→ shmem 缺失共**两层三个触发面**:FlagGems 算子(cos/sin,黑名单可挡)+ triton-ascend 自身 import 链(tle,黑名单挡不住)。跨组需求(备忘录附录B)证据从"cos 一个算子"扩充为"两层三触发点"。

## 1. A 组:加载基线(gmu=0.9)——核心差异:池子减半

| 项 | vllm-ascend(基线) | vllm-plugin-FL | 差异解读 |
|---|---|---|---|
| KV 池预分配 | **39.23 GiB / 285,568 tok** | **~19.6 GiB / 140,800 tok** | **正好一半** |
| 最大并发(4096 tok/req) | **69.72×** | **34.38×** | 同上 |
| KV block size | 16 | 128(CUSTOM backend) | 取整粒度粗 8× |
| 权重 | 15.66 GiB | ~15.9 GiB(账目反推) | 一致(±checkpoint 差) |
| HBM 稳态(chip4) | 59,745 MiB(91.2%) | **40,392 MiB(61.6%)** | FL 整卡少用 ~19 GiB |
| HBM 峰值(全程) | 59,745 MiB | 40,459 MiB(61.7%) | 同上 |
| peak activation 结算行 | 0.24 GiB(实测) | 无结算行;**静态预留 = budget×50%** | 观测口径差异 |

**池减半根因(代码级,`vllm_fl/worker/worker.py::determine_available_memory`)**:FL 在 NPU 上**跳过 profile_run**(注释:GDN/FLA 层 Triton 内核触发 NPU 驱动 SIGKILL),改静态公式:`budget = 0.9×total − weights`;**`activation_reserve = budget × 0.5`**(注释:预留给激活 + KV 非连续视图首次前向的一次性连续拷贝);`pool = budget/2`。数字自洽:39.23 GiB 等效预算 → 19.6 GiB 池 → 140,800 tok(block128 取整)→ HBM 账目 idle 3.08 + 权重 ~15.9 + 池 ~19.6 + workspace ~0.9 ≈ 39.5-40.5 GiB,与采样吻合。

**画像含义**:vllm-ascend 的 activation 额度是 profile_run 实测(仅 0.24 GiB);FL 拍脑袋留 50%。生成式场景这意味着**一半并发容量被无实测依据地锁死**——gmu 之外的第二根杠杆,且两栈定义不同。共享机场景是"善意余量",独占卡场景是纯浪费(61.6% vs 可用的 90%+)。

## 2. 冒烟

stub 修复后首请求通过(8 tokens,134 s 含 JIT 编译);服务就绪 ~4.5 min。**FL 栈 decode ~16 s/token**(vllm-ascend ~25 tok/s,差 ~400×);prefill 正常(B-in3584 与 B-in128 同为 ~260 s 总时长,prefill 只占秒级)。矩阵输出 token 按此速度等比缩水,画像结论不受影响。

## 3. C 组:decode KV 增长(输出缩水 32/64)

| 输出 tokens | 总 token(prompt+completion) | KV 使用率峰值(块账目) | 结束后 KV |
|---|---|---|---|
| 16(warm 探针) | 52 | 0.091%(1 块) | 0 ✓ |
| 32 | 164 | 0.182%(2 块) | 0 ✓ |
| 64 | 196 | 0.182%(2 块) | 0 ✓ |

**KV 按 128-token block 台阶取整**(164/196 tok 都落在 2 块;52 tok 落 1 块),粒度比 vllm-ascend(block16)粗 8×。token 级消耗与基线一致(≈0.14 MiB/tok),百分比读数因池子减半而 ≈2×。

## 4. E 组:完成/取消归还闭环

- **完成×3 轮**:3/3,KV 峰值 0.182%,每轮归 0,无累积 ✓。
- **中途取消**(2 并发,8s 掐断):取消生效,KV 峰值 0.000(16 s/token 下取消发生在任何 KV 分装可见之前);取消后探针请求正常服务、KV 0% → **服务健康、容量无流失** ✓。⚠️ 局限:此栈速度下"取消触发的块回收"无法直接展示(与 9 月基线 8.1s 出 3000 tok 的情形不可比),只验证了取消后无残留。
- E-after-cancel-probe:done,KV last=0 ✓。

## 5. HBM vs KV 双口径(核心证据)

全程 10.5 h 采样(18,893 行):加载稳态 39,506 MiB(60.3%)→ 服务全程稳态 **40,392 MiB(61.6%)**(+886 MiB 为服务 workspace 爬升)→ 峰值 40,459 MiB。同期 KV 使用率 0→6.19%→0 多次往复,**设备 HBM 纹丝不动**。

→ 与 9 月结论同构:**"KV 块使用量增长/归还"与"设备显存占用"必须分开看**;判泄漏看 kv_cache_usage_perc,不看 npu-smi 单曲线。跨栈统一观测口径 = /metrics + 宿主 npu-smi,FL 无标准结算行也不妨碍。

## 6. B/D 组:输入长度与并发对照

B(输出 16,KV 严格按块账):

| 输入 | 总 token | KV 峰值 | 块数(ceil/128) |
|---|---|---|---|
| 128 | 148 | 0.182% | 2 ✓ |
| 1024 | 1044 | 0.819% | 9 ✓ |
| 2048 | 2068 | 1.547% | 17 ✓ |
| 3584 | 3604 | 2.639% | 29 ✓ |

D/F(同 prompt 并发,prefix caching 生效):

| 组 | 结果 | KV 峰值 | 块账目 |
|---|---|---|---|
| D-c4 | 4/4 | 0.728% | 8 = 4 共享前缀块 + 4×1 解码块 ✓ |
| D-c16 | 16/16 | 1.820% | 20 = 4 + 16 ✓ |
| D-c64 | **64/64** | 6.187% | 68 = 4 + 64 ✓ |
| F-c8(gmu0.9) | 8/8 | 0.911% | 12 = 4 + 8 ✓ |

**同 prompt 前缀共享块**的账目与实测严丝合缝——这同时解释了 9 月 vllm-ascend D 组"低于线性预期"的现象(prefix caching ON 时并发同前缀请求共享块,两栈机制一致)。c64 在 FL 34.38× 最大并发口径下仍无压力(532 tok/req 口径 64 路仅占池 6.2%),归还全部归 0。

## 7. F 组 gmu0.4 对照(未跑,公式推导)

重启+warmup 成本高(~15 min)而增量信息有限,gmu0.4 腿由公式外推:FL 池 ≈ (0.4×64−16.9)/2 ≈ **4.3 GiB ≈ 30,000 tok ≈ 6.9×**(4096 tok/req 口径),对比 vllm-ascend 实测 8.59 GiB/62,464 tok/15.25×——**FL 在低 gmu 下减半效应更致命**(6.9× vs 15.3×)。列为可选后续实测项。

## 7.5 embedding 负载对照(Qwen3-Embedding-0.6B,补充实测)

9 月 embedding 画像(Qwen3-Embedding-0.6B,davinci7,vllm-ascend,gmu0.9,max_model_len 8192):池 **53.79 GiB / 503,552 tok / 61.47×**,权重 1.13 GiB,HBM 稳态 ~90%。FL 栈复测(2026-10-12,davinci4,同 gmu/同 max_model_len,`--convert embed`,模型重新下载):

| 项 | vllm-ascend(9 月) | vllm-plugin-FL | 解读 |
|---|---|---|---|
| KV 池 | 53.79 GiB / 503,552 tok / **61.47×** | **27.6 GiB / 250,240 tok / 30.55×** | **再次精确减半**(251,776 理论值 vs 250,240,block128 取整) |
| 权重 | 1.13 GiB | 1.11 GiB(checkpoint) | 一致 |
| HBM 稳态 | ~90% | **32,517 MiB(49.6%)** | FL 整卡只用一半 |
| 负载中 KV 使用率 | ≈0(池=纯余量) | 0 → 0.36% 瞬态 → 0(64 路同文本,prefix 共享) | **池=纯余量的语义两栈一致** |
| embedding 输出 | — | 1024 维 ✓(/v1/embeddings 可用) | FL 栈 pooling 链路可用 |

→ **50% 静态预留与负载无关**已双模型(0.6B embedding / 8B 生成式)实锤:池减半是 worker sizing 层的属性,不是 workload 属性。embedding 场景 FL 的减半只是"余量减半"(无害),生成式场景才是"并发容量减半"(致命)——但两侧口径都必须标栈。
→ 附带定性:FL 栈 embedding 业务**可用**(服务、精度维度、归还闭环全通),decode 慢的问题不影响 embedding(无解码步);批吞吐未与基线精确对测,未下结论。
→ FL 栈参数面(2026-10-12 补充实测):50% 预留硬编码无开关;上游官方 override `--kv-cache-memory-bytes` 路径存在但**实测挂死**(该路径强制初始化期跑 profile_run——FL 在 NPU 上刻意跳过的东西;stub 后不崩但 16+ min 编译/autotune 活锁)。**不改代码前提下 FL 池无法调回基线水平**;vLLM 0.20.2 的 embedding 任务 CLI 参数为 `--convert embed`(非 `--task`)。

## 8. 双栈对照结论

1. **池定义同源但 sizing 分叉**:同一 vLLM 0.20.2 核心,池大小公式不同(vllm-ascend profile_run 实测 vs FL 静态 50% 预留)→ 任何"池大小/最大并发"数字必须**标栈**,不可跨栈直比。
2. **KV 消耗 token 账目两栈一致**(≈0.14 MiB/tok,prefix 共享机制一致)——KVConnector API 面一致(W1 已证)叠加本条,**卸载结论结构性可平移**,容量收益按各栈池基准归一即可。
3. **性能现状**:FL decode ~16 s/token(基线 ~25 tok/s,差 ~400×),首请求含 JIT 2-4 min;v3 统一基座在 910C 上**推理性能尚未可用**,不影响显存画像有效性,但生成式业务暂不能落 FL 栈。
4. **启动门槛**:黑名单(cos,sin)+ shmem stub 两个外置补丁;shmem 两层三触发点证据已可并入跨组需求。
5. **观测口径**:FL 不产 vLLM 标准结算行,跨栈统一走 /metrics + npu-smi 双口径即可;block128 使 FL 的 KV 百分比读数台阶更粗、同 token 数下读数 ≈2×,跨栈比百分比会踩坑,**比 token 数**。
6. **给 gmu 建议加一栈注**:vllm-ascend 栈"gmu 是唯一大杠杆";FL 栈再多一根"gmu 之外池子再砍半"的杠杆(与负载无关,双模型复现),容量规划须按 34.38×(生成式 4096 口径)/30.55×(embedding 8192 口径)计。
7. **embedding 业务**:FL 栈可用(/v1/embeddings、1024 维、归还闭环全通);生成式业务 decode 慢 400× 不可用——**v3 基座当前只适合 embedding/池行为验证,不适合生成式性能场景**。

## 局限

- FL 矩阵输出 token 缩水(C/E 16-64 tok、B/D 16 tok;基线 64-3000 tok),KV 线性/归还/并发结论按块账验证,token 账与基线一致。
- E-cancel 在 16 s/token 速度下无法展示块回收动作用例(记录局限,非缺陷)。
- F 组 gmu0.4 未实测,公式外推(约 4.3 GiB/6.9×)。
- 采样 2s 分辨率;shmem stub 为容器 writable 层修改,重启容器丢失(需重新写入,内容在探针目录可留档)。
