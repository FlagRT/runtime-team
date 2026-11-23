"""定位 ModelConfig 构造中 repo id 校验的真实调用方。"""
import traceback

import huggingface_hub.utils._validators as validators

_orig = validators.validate_repo_id


def hooked(repo_id):
    if str(repo_id).startswith("/"):
        traceback.print_stack()
        print(">>> validate_repo_id 收到本地路径:", repo_id, flush=True)
    return _orig(repo_id)


validators.validate_repo_id = hooked

from vllm.engine.arg_utils import EngineArgs

try:
    EngineArgs(model="/models/qwen3-embedding-0.6b", task="embed").create_engine_config()
    print("OK")
except Exception as e:
    print("最终异常:", type(e).__name__, str(e)[:120])
