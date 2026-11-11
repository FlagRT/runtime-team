"""910C D2 轮环境前置核查（容器内执行）。

目的：在跑 D2 之前，用**当次实测**确认
  ① ACL 能初始化（拿得到卡）—— 名额纪律要求「用卡当次重探」，不靠上一次的结论；
  ② 设备数可见；
  ③ 真能算（一次最小 matmul）；
  ④ 流配额（B2 v2 的既有事实，用于确认调的是同一套栈）；
  ⑤ D2 探针所需的统一面出口是否齐（create_stream / create_event / stream_priority_readback）。
"""
import os
import sys

# ⚠️ 必须在 import runtime 之前设：否则核查脚本自己会在仓库副本里留 `__pycache__`
#   （2026-10-08 首跑即踩到 —— 主编排脚本设了，这个独立核查脚本漏了，且产物是 **root 建的**、
#    宿主侧删不掉）。见运行手册 §11.2。
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

print("container ASCEND_RT_VISIBLE_DEVICES =", os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"))

try:
    import acl
except BaseException as e:                                              # noqa: BLE001
    print("FAIL import acl:", type(e).__name__, e)
    sys.exit(1)

rc = acl.init()
print("acl.init rc =", rc)
cnt = acl.rt.get_device_count()
print("device_count =", cnt)

import torch                                                        # noqa: E402
import torch_npu                                                    # noqa: E402

print("torch", torch.__version__, "| torch_npu", torch_npu.__version__)
print("torch sees", torch.npu.device_count(), "npu devices")
try:
    print("device name =", torch.npu.get_device_name(0))
except BaseException as e:                                          # noqa: BLE001
    print("device name 不可读:", e)

x = torch.ones(64, 64, device="npu:0")
y = x @ x
print("compute ok, sum =", float(y.sum().item()))

try:
    print("stream_available_num =", acl.rt.get_stream_available_num())
except BaseException as e:                                          # noqa: BLE001
    print("quota 不可读:", e)

# 统一面出口检查（D2 探针依赖这些）
sys.path.insert(0, os.environ.get("DC_ROOT", "/mnt/raid/hliu553/dc_d2_20261008/prototype"))
os.environ.setdefault("DC_BACKEND", "ascend")
try:
    import runtime
    need = ["create_stream", "create_event", "stream_priority_readback",
            "stream_priority_range", "release_stream"]
    missing = [n for n in need if not callable(getattr(runtime, n, None))]
    print("runtime 出口缺失 =", missing if missing else "（无）")
    bk = runtime.use("ascend")          # ⚠️ 能力查询在后端对象上（模块级没有 supports）
    print("backend supports:", {k: bk.supports(k) for k in
                                ("stream_priority", "stream_priority_control",
                                 "stream_priority_readback")})
    print("stream_priority_range() =", runtime.stream_priority_range())
    st = runtime.create_stream()
    print("默认流 priority 回读 =", runtime.stream_priority_readback(st))
    print("release_stream(默认流) =", runtime.release_stream(st))
    print("ENV_CHECK_PASS")
except BaseException as e:                                          # noqa: BLE001
    print("ENV_CHECK_FAIL:", type(e).__name__, e)
    sys.exit(2)
