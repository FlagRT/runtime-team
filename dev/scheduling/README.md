# scheduling · 任务调度方向落点

- **定位**：五组件「任务调度引擎」的仓内落点（T-020）。本期由 rag_ljy（刘继阳）承接、联合 batch_yxy（袁先乙）共建。
- **指针（正文不进本仓，铁律：不复制正文）**：
  - rag_ljy 线：检索—排序—生成的阶段队列 + 数量准入 + 背压 → `dev/rag_ljy/`
  - batch_yxy 线：FlagScheduler（独立方向仓，地址待报备；报备-收录机制见 `dev/README.md` 与 `dev/ENV-SPEC.md` §2.4）
  - memory → 调度：并发/排队建议的 handoff → `dev/memory/`（STATUS 的 handoff 段）
- **更新机制**：随交付进展由方向负责人更新本目录 `STATUS.md`；本目录现为骨架（2026-10-24 总组建立），待填充。

<!-- changelog
2026-10-24  新建（T-020）：建立「调度」方向仓内落点。此前调度能力散在 rag_ljy（dev-1.0）与
            batch_yxy（FlagScheduler，待报备）两处，仓内无目录（B4）。本目录只放指针与状态，
            不搬正文。
-->
