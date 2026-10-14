# KV 卸载到 Host:W2 最小闭环 + W3 开关对照(vllm-ascend 锁定栈,910C,2026-10-14)

> 状态:🟢 vllm-ascend 侧卸载四步闭环全部实测完成(卸→存→召回→结果正确)+ 开关对照第一版数字。
> **范围声明:本报告全部卸载数据产自 vllm-ascend 锁定栈(v0.20.2rc1-a3 + vLLM 0.20.2 核心)。vllm-plugin-FL 栈未测卸载**:decode ~16 s/token(基线 ~25 tok/s,差 ~400×),压测不可行;两栈共享 vLLM 0.20.2 核心 KVConnector API(W1 已证一致),结论结构性可平移,FL 侧实测排 v3 验证窗口。
> 环境:容器 `mem-profile-910c`(锁定镜像 quay.io/ascend/vllm-ascend:v0.20.2rc1-a3),davinci4,TP=1,gmu=0.9(小池召回轮 gmu=0.3),MAX_MODEL_LEN=4096,enforce-eager 执行模式,prefix caching=ON,OneRec-8B,与 9 月基线/10-12 davinci4 复测同口径。
> 卸载配置:上游 SimpleCPUOffloadConnector 机制(kv_transfer_config,kv_role=kv_both,prefix caching 必开)+ 自研外挂适配件(kv_connector_module_path 加载,**vllm-ascend 本体零改动**):NPU DMA 拷贝后端(torch.npu 低优先级流 + 逐块 copy_ non_blocking + Event 上报)+ KV 注册形状适配。Host 池 8 GiB 档(cpu_bytes_to_use=8 GiB → 455 块),随 cpu_bytes_to_use 线性放大。
> 原始数据:本地 `w2_results_1014/`(connon 13 腿 JSON+kv CSV、压测/召回各轮 JSON、HBM CSV、compare.py 对照脚本;baseline-1012 备份副本同目录)。

## 0. 块账目(全报告单位)与适配件过的三个死点

| 项 | 值 |
|---|---|
| KV 页(单层) | 524,288 B(K+V 各 262,144 B) |
| 逻辑块(36 层全层口径) | **18 MiB/块** |
| GPU 池(gmu=0.9) | 2,231 块 ≈ **39.2 GiB** / 285,568 tok / 69.72×(与基线一致,开 connector 不改池) |
| Host 池(8 GiB 档) | 455 块 ≈ 8.0 GiB |
| 真 L2 门槛 | Host 池 > GPU 池,即 **>2,231 块 ≈ >39.2 GiB**(64 GiB 档 = 1.63×GPU 池;服务器 2 TB RAM 充裕) |

| # | 死点 | 现象 | 修法 |
|---|---|---|---|
| 1 | 类型假设 | 上游 register 期望 Tensor/list,vllm-ascend 实传 `dict[str,(k,v) 元组]`(model_runner_v1.py:3486 传值、:4003 存元组),断言炸 | normalize_kv_caches 拆 `.0/.1` 双条目(K/V 都卸载,只取 v[0] 会漏 K 半边) |
| 2 | 搬运后端 | 上游 copy backend 纯 CUDA(cudaStream + cuMemcpyBatchAsync),NPU 无对应 | NpuDmaCopyBackend:torch.npu 低优先级流 + 逐块 copy_ non_blocking + Event 上报;pin_memory 失败降级 pageable |
| 3 | 对齐垫块 | vllm-ascend KV 分配带对齐垫块(实测 2,239 块 vs config 2,231),上游 view 假设 storage=config块数×页 → 炸 | 从 KVCacheConfig 组规格取页大小,反推 actual_blocks(36 层×2,231 块×512KiB=39.2 GiB 与基线池吻合) |

## 1. W2 四步闭环(能不能跑):全部实测 ✓

**① 卸(注册)**:register 36 项 → 拆 K/V 去重 72 tensor → Host 侧 455 块 pin 成功(pin_modes=['pinned'])→ 注册 COMPLETED;GPU 基线池未动(HBM 91.24% 与基线一致)。

**② 存(store)**:DMA 后端逐批次埋点(/tmp/w2_marker_transfers.txt,每 DMA 批次一行)。13 腿对照窗口 **34 store 事件/174 块**逐块下发;压测大池轮单轮 **873 次 store**。存只在 eager 存储模式下对已确认算完的 128-token 满块下发,数据恒先于淘汰落 Host;21-token 冒烟凑不满一块、零搬运属预期。

**③ 召回(load)**:见 §2。GPU 小池(gmu=0.3,139 块 < Host 455 块)加压下 **15 批 CPU→GPU DMA 真实搬运**(单批 n_blocks 4~35)。

**④ 结果正确**:13 腿 status/输出 token 与基线全同(A 腿 prompt 21→36 差异除外,wall 相等);D-c64 KV 峰值 5.919% 与基线一字不差(池核算一致);召回轮 12/12 请求正确完成。

## 2. 召回实测(加压变体):大池代数死结 → 小池一枪打通

压测脚本 `pressure-fill_910c.py`(每请求种子化唯一 prompt,消除 gen-load 同 prompt 前缀去重导致的池压力不足;长输出撑池)。六轮迭代定位:

- **大池(2,231 块)深超订**:47 次抢占、两轮合计 **462 次 CPU 命中查询**(r4/r5 各 231,ext=384/640 token,async=True)——**全部蒸发,零 DMA**。
- **根因=代数死结**:GPU 池满时原生 LRU 淘汰的永远是最老块 [0, churn−G];eager 存储模式下 Host LRU 永远留最新块 [churn−K, churn]。两区间相交 **当且仅当 Host 容量 K > GPU 容量 G**。8 GiB 档 K=455 < G=2,231 → CPU 目录里持有的后缀与 GPU 被淘汰的前缀永不相交。
- **抢占恢复窗口同时被调度饥饿关死**:深超订下请求排不上队,等排到时 GPU 前缀已缩过 Host 持有的后缀,连续匹配 ext→0 → 整段重算。
- **小池(gmu=0.3,GPU 139 块 < Host 455 块)反转代数** → 15 批 load DMA 真实发生,12/12 召回请求正确完成;小池同时让抢占恢复窗口复活(请求几块就能排上,恢复前窗口不关)。

## 3. W3 开关对照(值不值):第一版数字

| 项 | 结果 |
|---|---|
| 池等效扩容 | **+20.4%**(GPU 2,231 块 + Host 455 块 = 2,686 块;随 cpu_bytes_to_use 线性放大) |
| C 腿(decode)代价 | **+8.4% wall**(吞吐 −7~9%) |
| D 腿(并发 decode) | **+7.3% wall** |
| B 腿(prefill) | ±1.7% ≈ 0 |
| A/E 腿 | 0% |
| 正确性 | 全腿 status/token 与基线一致 ✓ |

**代价偏高解释**:当前后端为正确性优先的逐块 copy_(72 tensor × 逐块小拷贝),上游参照实现是单次 cuMemcpyBatchAsync 聚合调用;优化路径明确 = staging 聚合拷贝,预期显著收窄。另 eager 存储模式会白存无人读的块,7-8% 代价或含此虚耗(lazy_offload=true 一档 A/B 待做,见 §6)。

## 4. 池定义结构发现:Host 层 ≤ GPU 池只是"写缓冲",> GPU 池才是真 L2

**Host 层容量 ≤ GPU 池时,召回路径结构性受限(§2 代数死结),此时它只是"写缓冲",+20.4% 只是账面数;Host 池 > GPU 池(910C 上 >2,231 块 ≈ >39.2 GiB,64 GiB 档 = 1.63×)才是真两级缓存。** 服务器 2 TB RAM,门槛完全可行。

口径提醒:"等效扩容"的本质 = 淘汰块以 PCIe 速度取回替代重算,池没变大、"装得下的不同内容"变多了;收益只对前缀复用型负载成立。本结论应写入显存池定义文档的分层存储节。

## 5. 架构归属(评审答辩用)

- **调度决策全在 vLLM 0.20.2 核心自带**,两栈共享,我方一行未动:GPU 池满淘汰 = 原生前缀缓存 LRU;"何时存" = SimpleCPUOffloadScheduler eager 存储模式(块确认算完即存);"何时取回" = 新请求 GPU 前缀未命中 → 按块哈希查 CPU 目录 → 命中生成 load 计划;CPU 侧 455 块自有 LRU 目录。
- **我方代码 = 昇腾的手**:NpuDmaCopyBackend(存/取两方向同一后端)+ KV 注册形状适配。形状适配是 vllm-ascend 专属,拷贝后端栈无关 → **FL 栈可平移、若基座切换不返工**。
- 定性:不是"给 vllm-ascend 补新版本功能"——卸载机制在锁定版 0.20.2 核心里本来就有,缺的是昇腾适配;外挂适配件把拼图拼上,vllm-ascend 本体零改动。

## 6. 遗留与局限

- **lazy_offload A/B 未做**:lazy 存储模式(临淘汰才存)可能直接降低代价数字,单档实验,排下个占卡窗口。
- **大 Host 池(>39.2 GiB,如 64 GiB 档)召回率未复测**:§4"真 L2"推论未经大池实测命中数验证。
- 代价未优化:staging 聚合拷贝是明确优化路径,当前 7-8% 是第一版(未优化)数字。
- E-cancel 泄漏观测为间接证据(after-cancel-probe 正常 + 后续腿无退化;pinned 内存不经 RSS,RSS 探针在此场景无效)。
- 原始 transfers/diag 埋点文件因容器 stop 清 /tmp 未拉回(证据在各轮 JSON + 会话记录;小池召回 demo ~5 分钟可重新生成)。
- 适配件内三层诊断埋点(transfers/get_finished/cpu_lookup/meta)为实验观测用,**PR 提交前拆除**。
- **FL 栈卸载未测**(见范围声明):decode ~400× 慢使压测不可行,非机制性障碍;实测排 v3 验证窗口。
- 采样 1 s 分辨率;num_preemptions 指标受 Prometheus _created 序列污染,以服务内 live 读数为准。
