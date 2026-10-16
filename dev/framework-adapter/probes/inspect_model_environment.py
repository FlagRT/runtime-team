"""Read-only package/source inventory. Does not initialize a device or model."""
import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def main():
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "vllm", "vllm-ascend",
                 "flag-gems", "triton", "triton-ascend"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "machine": platform.machine(), "packages": packages,
              "source_roots": {}}
    for name in ("transformers", "vllm", "vllm_ascend"):
        spec = importlib.util.find_spec(name)
        result["source_roots"][name] = list(spec.submodule_search_locations or []) if spec else []
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/model")
    result["model_config"] = json.loads((model / "config.json").read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
