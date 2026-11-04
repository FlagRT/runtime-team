# MLU590 weakref 退出期缺陷判据（2026-11-03）—— 四家矩阵第 4 家

> 判据脚本与 910C / PPU / P800 **完全同源**，仅命名空间不同（`torch.mlu`）。
> 结论：**MLU590 未中招** 🟢

| 项 | 读数 |
|---|---|
| `torch.__version__` | 2.7.1+cpu（torch_mlu 1.29.2） |
| Stream/Event weakref 可创建 | 是（30/30 weakrefable） |
| Stream 回调 / Event 回调 | **True / True**（缺陷态应为 False） |
| `Stream + weakref + 被回收` × 30 的退出码 | **0** |

⭐ 补充事实：MLU 侧 weakref **可创建且清理正确** ⇒ 不存在"torch <2.8 一律不支持弱引用"的说法；
缺陷只出现在「父类已支持 weakref、子类 `tp_dealloc` 未链式清理」的组合上（PPU / 910C torch_npu 2.10.0）。

| 路径 | 说明 |
|---|---|
| `xchip_cb.py` / `xchip_cb.log` | 回调是否触发（`--` 参数 `mlu`） |
| `xchip_rc.py` / `xchip_rc.log` | 30 次创建+回收后的进程退出码 |
