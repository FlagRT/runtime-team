# kunlun-infer-vllm v1 —— 昆仑芯 P800 推理腿运行环境（登记式基线）

- **是什么**：上游 xVLLM 发布线镜像（harbor.baai.ac.cn/flagrelease-public：vllm 0.20.2+empty + torch 2.9.0+cu129 + xmlir 1.0.0.1 + flag_gems 5.0.0 + flagcx 0.13.0），入锁为昆仑芯推理腿运行环境。承担推理服务化与 KV connector（simple_cpu_offload_connector）宿主栈；训练与设备上下文仍用 kunlun-operator-runtime v2 基座（两腿并存，同 910C/寒武纪两腿分栈思路，ADR-0026）。
- **为什么**：训练腿镜像的 vllm 0.13.0 缺 simple_cpu_offload_connector，FlagMemory xpu KV connector import 链断（环境性）；本镜像 vllm 0.20.2 系自带且 2026-11-09 真卡三件验证通过。
- **版本事实**：image.baseline.yaml（manual-registration）+ lock.yaml（digest 锚、验证记录）。
- **纪律**：设备注册/选卡同 kunlun-operator-runtime v2（xpu-smi 挑空闲卡 + CUDA_VISIBLE_DEVICES 自律 + devices 授卡与 /dev/xpuctrl）；P800-A 上 fresh docker run/exec 已知 ENOEXEC（单机病），取证只读进行。vllm-plugin-FL 插件自动加载在 import flag_gems 阶段报错（不阻断），connector 路径不受影响。

<!-- changelog
2026-11-09  随目录首登（ADR-0026 定档）：昆仑芯 KV connector 运行环境 = 本镜像（选项 A，机器已就位 + 真卡三件验证）；训练腿 operator-runtime v2 不变。
-->
