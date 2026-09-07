# 框架接入与算子调用适配（framework-adapter）

> **状态：进行中** ｜ 负责人：顾宬

## 目标

复用框架已有调用入口，在 FlagGems、厂商原生实现和参考实现之间进行可验证的算子选择，并限制不安全的运行时重试。

## 当前结论

- 主线先验证 PyTorch/vLLM，不同时铺开 TensorFlow 和 ONNX Runtime。
- 算子实现由 FlagGems 或厂商库提供，本方向不重复开发内核。
- vLLM 功能代码入口是 `vllm-plugin-FL/vllm_fl/dispatch`；本目录保存环境配置、接入原型探针和联调记录。
- 已补齐执行前输入检查、`CachedOp` 热路径保护，以及有副作用实现失败后禁止重试的规则。
- 2026-09-07：27 机单卡环境通过 275 项 dispatch 单测、74 项真实 NPU 测试，微型随机 Llama 完成 vLLM 解码并与 CPU 的 4 token 对照一致。详见 [当日联调记录](docs/910C单卡联调-20260907.md)。
- PyTorch 独立接入：`PreferGems` 作用域内支持 F.silu/F.rms_norm 及对应 nn 层，68 项测试与独立示例通过；不导入 vLLM。目前有保守 NPU 同步开销，异步互操作仍待排查，不是全局默认注册或性能交付。见 [说明与限制](docs/PyTorch独立接入-20260907.md)。

## 任务看板

| 任务 | 状态 | 出口标准 |
|---|---|---|
| 现有注册与调度机制梳理 | 完成 | 明确 FlagGems、vendor、reference 三类实现入口 |
| 安全回退最小改造 | 进行中 | 不兼容输入执行前跳过；有副作用实现失败后不重试 |
| Ascend 代表算子联调 | 进行中 | SiLU、RMSNorm、Rotary 已有真实单卡数值与调用语义测试，继续扩大模型覆盖 |
| 兼容性矩阵 | 初版 | 已记录当前镜像与小输入矩阵；不代表多芯片、多框架支持完成 |
| 普通 PyTorch eager 接入 | 同步原型 | 作用域内优先选择，执行前回到原生；继续定位异步互操作问题 |

## 启动环境

```bash
# 在 27 机，已有容器直接复用，不要重复创建：
docker start flagos-cgu135-dev-910c
docker exec -it flagos-cgu135-dev-910c bash
```

`docker-compose.910c.yml` 保存独立单卡配置，不要叠加挂载全部设备的 base 配置。
27 机没有可用的 `docker compose` 子命令；当前容器通过 `docker run` 创建。首次创建命令见联调记录，已有容器只需 start。
源码位于宿主机 `/home/cgu135/framework-adapter-910c`，映射到容器 `/workspace`。
`vllm-plugin-FL` 必须另行上传到此目录；它是独立仓库，不随 runtime-team 自动拉取。

容器内首次安装与检查注册结果（不升级镜像内依赖）：

```bash
pip install -e /workspace/vllm-plugin-FL --no-deps --no-build-isolation
python /workspace/dev/framework-adapter/probes/dispatch_registry_probe.py
```

## 当前环境记录

- 27 机容器只映射 `davinci0`，限制 8 CPU / 32 GiB 内存 / 4 GiB shm；不独占其他卡，也不修改其他人的容器。
- 2026-09-03 的 `aclInit 507899` 是历史阻塞；9 月 7 日清理容器后重新实测已可运行。先前日志不足以单独证明容器数量就是唯一根因。
- 宿主机执行 `bash /home/cgu135/framework-adapter-910c/dev/framework-adapter/probes/run_910c_checks.sh` 可重跑单测及真实算子测试，日志按时间留存。
- 用完后 `docker stop flagos-cgu135-dev-910c`，下次再 start；停止不删除源码或容器内安装。

## 安全边界

当前为代表算子的适配原型，并非所有算子的三级安全回退已经完成。
`runtime_fallback_safe` 为兼容旧注册仍默认 True；未审计实现不能据此认定设备错误可恢复。
建议开发验证采用 strict 策略：输入不兼容可以执行前选择其他实现，但执行异常立即抛出。
`resolve()` 返回原始函数，不具备 `call()` / `CachedOp` 的执行前选择保障。
完整模型、性能、自动求导及多芯片仍需独立验收。
