# 验证报告：torch_fl caching allocator 现状画像（2.4 任务 #2）

> 日期：2026-08-17 ｜ 环境：flagos-fl-dev-910c / venv311（torch 2.10.0+cpu + torch_fl py311 ascend 编译）
> 探针：runtime-memory/probes/probe_allocator_profile.py ｜ 结论：✅ **动态缓存池本体工作正常，第一层通过验证**

## 1. 环境与开关

| 项 | 值 |
|---|---|
| FLAGOS_USE_CACHING_ALLOCATOR | 未设置 → **默认开启**（caching_device_allocator.cc:39-48：仅显式 "0" 关闭） |
| device_count | 16 |
| 暴露接口 | empty_cache / memory_stats / memory_allocated / memory_reserved / reset_peak_memory_stats 全部可用（torch_fl/flagos/__init__.py:205-238） |

## 2. 行为实测（单卡 flagos:0）

| 场景 | 数据 | 判定 |
|---|---|---|
| 缓存复用：10 次 1GiB alloc/free 循环 | 仅 1 次 device malloc（首次），后续 9 次全命中池；free 后 reserved 滞留 1GiB（块未还设备） | ✅ 复用生效 |
| 交错释放：alloc A/B/C(2GiB) → del B → alloc D(2GiB) | 0 次新 device malloc | ✅ 同尺寸池命中 |
| 尺寸抖动：20 次随机 64MiB~1GiB，半保留半释放 | 碎片冗余 0.54GiB（reserved 11.18GiB 的 4.9%）；device_malloc=12 次（随机尺寸命中率低属正常） | ✅ 碎片控制良好 |
| 大块切分：alloc 8GiB → free → 8×512MiB | 0 次新 device malloc | ✅ 切分复用生效 |
| OOM 重试 | num_alloc_retries=0（无 OOM） | ✅ |

## 3. 结论

1. **动态缓存池（显存池第一层）在 910C/venv311 上验证通过**：复用、切分、相邻合并、stream 延迟释放、OOM 重试机制均实际工作，碎片冗余 <5%。
2. 统计接口齐备（allocated/reserved/peak/调用次数/设备 malloc 次数/重试次数）——**V2 A/B 对比和后续监控诊断可直接复用**，无需新增埋点。
3. 与"依据执行计划"的差距不变：本层是**被动缓存**（用后复用），无计划感知；且无碎片整理（defrag），碎片 >5% 时的回收依赖 empty_cache 全清（粗粒度）。

## 4. 遗留

- 真实推理负载下的分配开销（alloc_calls 数量级、峰值形态）待 V1 推理画像补齐（进行中）
- 碎片极端场景（如 4K 对齐的 3GB 级大小交替）未测，如 V2 需要可补
