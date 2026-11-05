# flagmemory-pool · 显存池真卡实测证据（昇腾 910C）

> 本目录 = 显存池组件（FlagMemory）在昇腾 910C 单芯片受控配置下的实测证据**脱敏副本**；原始证据私有保留，不随本目录公开。环境与版本事实见 [env.baseline.yaml](env.baseline.yaml)（字段依 [ENV-SPEC](../../../ENV-SPEC.md) §3.2 mounted_repos 口径）；逐文件「私有原件 / 公开副本」sha256 对照与脱敏分类计数见 [checksums.json](checksums.json)。
> 入仓前泄漏核查（IP / 主机名与家目录 / 账号形态 / 口令与密钥形态四类）：本批命中仅自家家目录与账号名，已按占位符脱敏（`/home/<user>`、`<user>`）；无 IP、无主机名、无密钥形态命中。

## 证据集

| 集 | 日期 | 池代码（FlagMemory commit） | 结论 |
|---|---|---|---|
| `ascend-pool-lifecycle-3de6d5a/` | 2026-11-04 | `3de6d5a69d8a35ed12a88bd974e746a5fb0c81d5` | 单芯片受控配置下：设备映射、4096 字节健康读写、三生命周期用例 3 通过 0 跳过 |
| `onerec-input-consume-4967c6aa/` | 2026-11-04 | `4967c6aa07df419eae54ea0e66f0863b8ac14c17` | OneRec-8B 单输入：普通输入与池输入各一次前向，末 token logits 最大绝对差 0、argmax 一致；复用 / 归还 / 销毁完成 |

## 已验范围声明（不冒充）

仅覆盖**单芯片受控路径与单模型输入缓冲**：不表示权重 / KV / 激活接管、多卡多 rank、生产业务负载或性能收益经过验证。主机测试不充抵芯片实测。验证叙事与配置细节见组件仓 `docs/validation/2026-11-04-ascend-pool-onerec.md`（FlagMemory dev-1.0）。

## 排除项

源码快照（`source.tar` 与 `*.py` 运行脚本）不入本目录——源码事实以组件仓提交为准；模型权重不入公开仓。
