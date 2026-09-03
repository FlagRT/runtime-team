# 框架接入与算子调用适配（framework-adapter）

> **状态：进行中** ｜ 负责人：顾宬

## 目标

复用框架已有调用入口，在 FlagGems、厂商原生实现和参考实现之间进行可验证的算子选择，并限制不安全的运行时重试。

## 当前结论

- 主线先验证 PyTorch/vLLM，不同时铺开 TensorFlow 和 ONNX Runtime。
- 算子实现由 FlagGems 或厂商库提供，本方向不重复开发内核。
- 功能代码入口是 `vllm-plugin-FL/vllm_fl/dispatch`；本目录只保存环境配置、探针和联调记录。
- 已完成执行前输入兼容性检查和运行时回退安全开关的实现；相关 60 项单元测试在 910C 服务器的 FlagGems 镜像中通过。

## 任务看板

| 任务 | 状态 | 出口标准 |
|---|---|---|
| 现有注册与调度机制梳理 | 完成 | 明确 FlagGems、vendor、reference 三类实现入口 |
| 安全回退最小改造 | 进行中 | 不兼容输入执行前跳过；有副作用实现失败后不重试 |
| Ascend 代表算子联调 | 待进行 | 至少完成 `silu_and_mul`、`rms_norm`、`rotary_embedding` 的路径和数值验证 |
| 兼容性矩阵 | 待进行 | 记录框架、芯片、版本、dtype、shape、命中实现和测试证据 |

## 启动环境

```bash
cd dev/framework-adapter
cp .env.example .env
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker exec -it flagos-framework-adapter-dev-910c bash
```

容器内检查注册结果：

```bash
python /workspace/dev/framework-adapter/probes/dispatch_registry_probe.py
```

## 当前环境记录

- 910C 宿主机可以 SSH 登录，16 个 NPU 芯片健康且查询时无计算进程。
- 2026-09-03 使用临时容器执行最小 NPU 张量计算，`aclInit` 返回 `507899 Resource_Busy`；机器上同时存在 4 个挂载全部 NPU 的常驻容器。真实算子验证仍需等待 DrvMng 槽位释放或换用备用机器。
- 详细记录见 `docs/安全回退最小验证-20260903.md`。
