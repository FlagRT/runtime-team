"""Stub for the missing `shmem` (aclshmem python bindings) module.

【入仓说明，2026-10-09】本文件是 v3 round 4 推理腿真机验证（npu1-11）实证的
运行时解堵件：单卡推理下 stub 不会被真调用（所有函数 loud-fail raise），
仅让 triton 编译期 `import shmem` 不炸。验证日志：
~/tmp-reproV3-r4/infer-leg/round-5.log（9/9 PASS）与
control-no-blacklist.log（对照组：无 stub/黑名单如期炸）。
用法：容器内 PYTHONPATH 指向本文件所在目录（docker cp 或 -v 挂载）。
归属：根治方是算子编译组（tle 惰性 import 或 aclshmem python 绑定官方入
镜像）——跨组需求结构性严重度版本；根治后本文件与
docker-compose.flaggems-cos-off.yml 一并删除。详见同目录 REBUILD.md
「ROUND 4 真机验证——推理腿」。

The flagtree triton toolchain (triton/experimental/tle/backends.py ->
_build_ascend_impl -> language/dsa/ascend/communication.py:6) does
`import shmem as ash` at module import time. The real aclshmem python
module is NOT present in this image, so ANY triton kernel containing a
`for` loop (grid-stride-loop) fails to compile with
ModuleNotFoundError: No module named 'shmem' -- including vLLM core
kernels such as vllm/v1/worker/block_table.py::_compute_slot_mapping_kernel,
which no VLLM_FL_FLAGOS_BLACKLIST entry can reach (blacklist only affects
torch ops taken over by FlagGems).

Single-NPU / non-distributed kernels never call any aclshmem_* function
at runtime; compilation only needs the import to succeed. Every stubbed
callable raises loudly if actually invoked, so this cannot silently
alter distributed semantics.
"""

_STUB_MSG = (
    "shmem stub was CALLED: real aclshmem python bindings are not installed "
    "in this image; distributed tle features are unavailable"
)


def _forbidden(*args, **kwargs):
    raise RuntimeError(_STUB_MSG)


def set_conf_store_tls(*args, **kwargs):
    _forbidden(*args, **kwargs)


def aclshmem_init(*args, **kwargs):
    _forbidden(*args, **kwargs)


def aclshmem_create_tensor(*args, **kwargs):
    _forbidden(*args, **kwargs)


def aclshmem_free_tensor(*args, **kwargs):
    _forbidden(*args, **kwargs)


def aclshmem_finalize(*args, **kwargs):
    _forbidden(*args, **kwargs)


class InitAttr:
    def __init__(self, *args, **kwargs):
        _forbidden(*args, **kwargs)


class OpEngineType:
    MTE = 0
    AICORE = 1
