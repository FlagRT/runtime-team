"""i3 载体 rc=139 的**注入对照**（T11 §4 同法）。

三种注入，各独立进程、各 2 轮：
  control  —— 不改行为（只跑一遍，预期 139）
  note_off —— `note_stream_created` 变 no-op（⇒ 不建弱引用；预期 0）
  strong   —— `_id_holder` 改强持有（⇒ 不建弱引用，但登记仍生效；预期 0 且 I3 判据仍 PASS）
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import runtime.conformance.runner as R  # noqa: E402

MODE = os.environ.get("INJ", "control")

if MODE == "note_off":
    from runtime.backends.base import RuntimeBackend
    RuntimeBackend.note_stream_created = lambda self, s: None
elif MODE == "strong":
    from runtime.backends.base import RuntimeBackend
    RuntimeBackend._id_holder = staticmethod(lambda obj: (lambda _o=obj: _o))

sys.argv = ["runner", "--backend", "ppu", "--cases",
            os.environ.get("CASES", "ci_i3"),
            "--out", f"/tmp/inj_{MODE}.json"]
print(f"[inject] mode={MODE}", flush=True)
rc = R.main()
print(f"[inject] main returned {rc}", flush=True)
sys.exit(rc)
