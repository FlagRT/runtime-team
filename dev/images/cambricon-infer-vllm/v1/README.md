# cambricon-infer-vllm v1 —— 寒武纪 MLU590 推理腿发布物（登记式基线）

- **是什么**：FlagOS 官方寒武纪推理应用镜像（flagos-app 线：vllm 0.20.2+flagos + vllm-plugin-FL 0.2.2 + neuware4.4.3 栈），入锁为寒武纪推理腿发布物。承担推理服务化；训练与设备上下文仍用 cambricon-mlu-runtime v1 基座（两目录各管一腿，同 910C 两腿分栈思路）。
- **版本事实**：image.baseline.yaml（manual-registration）+ lock.yaml（digest 锚、验收记录）。
- **怎么用**：demo/ 起容器（up.sh / compose down），服务起停走 dev/device-context/prototype/scripts/serve_standard.sh（DC_BACKEND=cambricon，embed 形态；READY_BUDGET 建议 >=300s）。模型目录须显式给 MLU_MODELS_DIR。
- **纪律**：vllm-plugin-FL 在寒武纪线为必需组件（勿套用 910C 的 unset VLLM_PLUGINS）；18 设备节点 + cnmon 挂载 + 共享机选卡纪律见 cambricon-mlu-runtime/v1。

<!-- changelog
2026-10-24  随目录首登（T-041 入锁）：候选 2.2.0-0.2.2 通过 serve_standard 口径验收（HWJ_1 SERVE_STANDARD_PASS）后转正；解除 stack.lock.cambricon.v1 已知限制 C-1。
-->
