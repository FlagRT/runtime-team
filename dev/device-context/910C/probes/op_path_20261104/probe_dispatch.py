# 判定「我方路径究竟由谁实现算子」——注册层证据（不需要设备）
import sys, re
import torch, torch_npu  # torch_npu 注册 PrivateUse1 的 aclnn 实现

def dump(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as exc:
        return "dump_failed:%r" % (exc,)
    return t

OPS = ["aten::mm", "aten::matmul", "aten::add.Tensor", "aten::mul.Tensor", "aten::silu", "aten::triu"]
print("=== 阶段1：仅 import torch + torch_npu（我方路径的真实状态）===")
print("flag_gems in sys.modules:", "flag_gems" in sys.modules)
for op in OPS[:3]:
    t = dump(op)
    has_py = "python" in t.lower()
    # 提取 PrivateUse1 行
    lines = [l.strip() for l in t.splitlines() if "PrivateUse1" in l]
    print("  %-18s PrivateUse1 行: %s" % (op, lines[:2] if lines else "(无)"))

print()
print("=== 阶段2：flag_gems.enable() 后（对照，看它新增了什么）===")
import flag_gems
flag_gems.enable()
print("flag_gems in sys.modules:", "flag_gems" in sys.modules)
for op in OPS[:3]:
    t = dump(op)
    lines = [l.strip() for l in t.splitlines() if "PrivateUse1" in l]
    print("  %-18s PrivateUse1 行: %s" % (op, lines[:2] if lines else "(无)"))
