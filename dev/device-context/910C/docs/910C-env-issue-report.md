# 910C 环境问题记录：容器内 aclInit 500000（已解决，根因 = DrvMng 容器上限）

> 记录人：Kistich ｜ 2026-08-19 发现，2026-08-20 定位并解决 ｜ 状态：✅ 已解决
> 仓库位置：runtime-team/dev/device-context/README.md「重要发现」

## 一、问题现象

在基于 `flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet` 镜像的容器内，
调用 torch_fl 设备接口时报：

```
[flagos-ascend] aclInit failed: 500000
RuntimeError: CachingDeviceAllocator: invalid device index
device_count: 0
```

- 错误码 `500000` = `ACL_ERROR_INTERNAL_ERROR`（内部错误，非参数错误）
- 手动 `ctypes` 直调 `libascendcl.so` 的 `aclInit(None)` 同样返回 500000
- **非 torch_fl 代码 bug**（绕过 torch_fl 直调 ACL 也失败）

## 二、复现步骤

```bash
# 容器内（任一挂载 davinci 设备的容器）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "
import ctypes
lib = ctypes.CDLL('/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so')
lib.aclInit.argtypes = [ctypes.c_char_p]
lib.aclInit.restype = ctypes.c_int
print(lib.aclInit(None))   # 槽位满时 500000，槽位有空时 0
"
```

## 三、根因（2026-08-20 证实，与同事排查结论一致）

**DrvMng（驱动侧管理进程）对同时挂载 davinci 设备的容器客户端数量有上限（实测 ≈3）**。
槽位占满后，任何新容器/新进程调用 aclInit 都报 500000——**与容器配置、torch_fl、CANN 版本均无关**。

## 四、决定性验证

| 状态 | 挂设备容器数 | aclInit 结果 |
|------|:---:|:---:|
| 初始（5 个挂设备容器）| 5 | ❌ 500000（所有容器）|
| 停 2 个闲置容器后 | 3（=上限）| ✅ **ret=0，device_count=16** |
| torch_fl 全链路 | — | ✅ is_available=True，flagos 设备真实矩阵乘 OK |

## 五、为什么之前误判为"版本不匹配"（排查教训）

1. **手动 ctypes 直调也失败**：只排除了 torch_fl bug，但失败点本来就是 aclInit（ACL 层）——**恰好印证 DrvMng 拒客而非版本校验**
2. **"memory 容器同样失败"不能排除机器共享状态**：所有容器共用宿主同一个 DrvMng，槽位满后新 aclInit 全失败正是上限的预测结果
3. **"宿主直调成功"不能证明版本兼容**：宿主走的是 8.5.0 老组合，且宿主进程不走容器客户端的 DrvMng 注册路径
4. **官方兼容矩阵**是"官方支持的最低驱动版本"，不是"能否运行"；真正的版本不兼容会报明确版本校验错误，不是 500000 这种通用运行时错误；且 9.0.0+25.5.0 之前真实推理成功过

## 六、协作提醒（写进子方向看板）

- 多卡测试前检查挂设备容器数（`docker ps` 数 --device davinci 的容器）
- DrvMng 上限 ≈3：超过需先停闲置容器（停他人容器前在群里打招呼）
- **无需升级宿主驱动**（25.5.0 → 25.5.1/2 为主机级变更，按上述证据大概率白干，仅需进入官方支持矩阵时才考虑）
