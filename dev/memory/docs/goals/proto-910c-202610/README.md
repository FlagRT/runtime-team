# ✅ proto-910c-202610 —— FL 双栈画像 + KV 卸载闭环（910C，2026-10）

> 目标：v3/FL 统一基座一致性画像 + KV→Host 卸载在锁定栈的闭环验证与开关对照（战略文档 §5 memory「分层缓存/Host 溢出」）。
> 状态：🟢 vllm-ascend 侧四步闭环 + 开关对照完成（2026-10-12/14）；FL 栈实测排 v3 验证窗口。

| 文件 | 内容 |
|---|---|
| `profile_FL双栈显存画像对照_910c.md` | vllm-plugin-FL 14 组全矩阵对照 vllm-ascend 基线：池减半根因 / decode ~400× 慢 / shmem 两层三触发点；卸载结论可平移，容量数字须标栈 |
| `profile_KV卸载Host对照_910c.md` | KV→Host 卸载 W2 四步闭环（召回 15 批 DMA 实测）+ W3 开关对照第一版（+20.4% 等效扩容 / decode 7-8%）；真两级缓存门槛 = Host 池 > GPU 池（>39.2 GiB） |

跨方向速览与最新结论见根 [STATUS.md](../../../STATUS.md)。
