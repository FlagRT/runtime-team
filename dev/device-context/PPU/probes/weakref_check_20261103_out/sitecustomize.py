# P4 零回归：用**注入式单变量对照**把补丁挂到我们自己的验收脚本上（不改仓库任何代码）
# 铁律（本方向已登记）：bytes 字面量只用 ASCII / 入口须真实文件 / 每轮打印加载自证
import os
if os.environ.get("DCFIX") == "1":
    try:
        import wrfix  # noqa: F401  import 即打补丁
        print("[sitecustomize] wrfix LOADED (self-proof)", flush=True)
    except Exception as exc:  # 加载失败必须显形，不得静默
        print("[sitecustomize] wrfix LOAD FAILED: %r" % (exc,), flush=True)
else:
    print("[sitecustomize] active but DCFIX!=1 -> no patch (control arm)", flush=True)
