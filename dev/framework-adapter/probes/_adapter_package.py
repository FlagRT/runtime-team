"""Integration tests require the independent installed package; never add sources."""
from pathlib import Path
import os
try:
    import flagrt_adapter
except ImportError as exc:
    raise ImportError("Install the pinned FlagRT/framework-adapter wheel first; see PACKAGE.md") from exc
if os.environ.get("FLAGRT_ADAPTER_INSTALLED_ONLY") == "1":
    expected = Path(os.environ["FLAGRT_ADAPTER_INSTALL_ROOT"]).resolve()
    if expected not in Path(flagrt_adapter.__file__).resolve().parents:
        raise RuntimeError("Tests must import the installed wheel, not the source tree")
