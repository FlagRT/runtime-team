# 关键判定：vllm_fl 插件加载后，「算子到底由谁实现」有没有变
import os, sys, json
os.environ.pop("VLLM_PLUGINS", None)          # 复现我方 910C 分支的做法
import torch, torch_npu

def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as exc:
        return ["dump_failed:%r" % (exc,)]
    return [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:1]

OPS = ["aten::mm", "aten::add.Tensor", "aten::silu", "aten::matmul"]
before = {op: priv(op) for op in OPS}
res = {"before_plugin": before, "flag_gems_before": "flag_gems" in sys.modules}

from vllm.plugins import load_general_plugins
load_general_plugins()
after = {op: priv(op) for op in OPS}
res["after_plugin"] = after
res["vllm_fl_loaded"] = "vllm_fl" in sys.modules
res["flag_gems_loaded"] = "flag_gems" in sys.modules
# 插件是否调用了 enable()（enable 会把 dispatch 换成 flag_gems 的实现）
res["dispatch_changed"] = {op: (before[op] != after[op]) for op in OPS}

print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
