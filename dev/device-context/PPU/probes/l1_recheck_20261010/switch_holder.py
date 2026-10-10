import pathlib
import sys

MODE = sys.argv[1]
P = pathlib.Path(
    "/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype/runtime/backends/base.py")
t = P.read_text(encoding="utf-8")
STRONG = ('self._reg("_owned_streams")[id(native_stream)] = '
          '(int(handle), lambda: native_stream)')
WEAK = ('self._reg("_owned_streams")[id(native_stream)] = '
        '(int(handle), self._id_holder(native_stream))')
old, new = (STRONG, WEAK) if MODE == "weak" else (WEAK, STRONG)
if t.count(old) != 1:
    if t.count(new) == 1:
        print(f"当前已是 {MODE} 模式，无需切换")
        raise SystemExit(0)
    raise SystemExit(f"锚点异常：命中 {t.count(old)} 次（{old[:50]}...）")
P.write_text(t.replace(old, new), encoding="utf-8")
print(f"已切换为 {MODE}")
