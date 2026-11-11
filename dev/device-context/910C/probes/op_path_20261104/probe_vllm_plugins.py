# 判定 vLLM 启动时是否加载 vllm-plugin-fl（进而 import flag_gems / 接管算子）
# 我方 910C 分支的做法是 `unset VLLM_PLUGINS`；这里对比三种取值
import os, sys, json
mode = sys.argv[1]
if mode == "unset":
    os.environ.pop("VLLM_PLUGINS", None)
elif mode == "empty":
    os.environ["VLLM_PLUGINS"] = ""
else:
    os.environ["VLLM_PLUGINS"] = mode

res = {"mode": mode, "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS", "<unset>")}
try:
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    res["load_general_plugins"] = "ok"
except Exception as exc:
    res["load_general_plugins"] = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:90])
try:
    from vllm.platforms import resolve_current_platform_cls_qualname
    res["platform"] = resolve_current_platform_cls_qualname()
except Exception as exc:
    res["platform"] = "ERR %s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:70])
res["vllm_fl_loaded"] = "vllm_fl" in sys.modules
res["flag_gems_loaded"] = "flag_gems" in sys.modules
print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
