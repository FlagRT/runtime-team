"""昆仑芯 GEMM 兼容修复 · 进程内启用模块（kunlun-operator-runtime v2 · env/v1）。

用法：在你的 Python 脚本**第一行**（任何 torch / flag_gems import 之前）：

    import gemm_compat

（需要 PYTHONPATH 已含本目录 —— 交互 shell 先 source ../enable-gemm-compat.sh，
或脚本内先 sys.path.append 本目录再 import。）

做四件事（与 2026-10-27 验收形态一致）：
  1. 设置必需环境变量（KLX_USE_AUTOTUNE=0 / XMLIR_MATMUL_FAST_MODE=1 /
     XPU_SDNN_PRESERVE_TTXIR_ERRORS=1 / FLAGTREE_BACKEND=xpu）；
  2. 把补丁版 FlagGems 源树插到 sys.path 最前（import flag_gems 取到补丁版；
     未启用时容器内 import 的仍是镜像内 73c5aff1 原版）；
  3. 校验安装的 triton xpu compiler.py 与验收基线逐字节同源 —— SHA 漂移即抛错拒绝
     启用（compat 包限定 SDK 版本，版本漂移后禁止盲目套用）；
  4. 用同目录编译组 prepare.py 生成 diagnostic 副本，进程内把副本的
     make_ttxir / make_llir 挂到活的后端类上。安装目录从不改动。

⚠️ 数值限制：64×64×64 fp32 非确定性乱值（结果不可信）；64³ bf16 与大形状 fp16/bf16
绝对误差超我组容差口径。已验可用面 = minirepro 12 组编译 + pytest mm 18 / mm·bmm·addmm 99。
详见 env/v1 README「数值限制」与 dev/issues/ 数值 case（K-5 / B15）。

出处：吸收自 Xproject-CHKU-SZ/FlagTree 分支 fix/xpu-sdnn-legacy-lowering-20261019
@edca42a7bd9c6bebfb0c6951beb64b9b84e36ccf（编译组）；本文件是 env/v1 的加载胶水，
加载逻辑与对方 run_pytest.py 的 load_profile 等价。本目录其余文件（prepare.py /
kernel.py / 0*.patch / reproduce.py / run_*.sh / test_*.py / README.md / PROVENANCE.txt）
为编译组包原样收录，见 PROVENANCE.txt。
"""
import hashlib
import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_HOME = Path(__file__).resolve().parent          # …/env/v1/compat/legacy_sdnn
FLAGGEMS_PATCHED_SRC = "/workspace/FlagGems/src"
_EXPECTED_COMPILER_SHA256 = (
    "7133710c0153c52dfe17822eb03cfd026a19c7c861b293dd3c9e818b6a64abef"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def enable() -> None:
    # 1. 环境变量
    os.environ["FLAGTREE_BACKEND"] = "xpu"
    os.environ["KLX_USE_AUTOTUNE"] = "0"
    os.environ["XMLIR_MATMUL_FAST_MODE"] = "1"
    os.environ["XPU_SDNN_PRESERVE_TTXIR_ERRORS"] = "1"

    # 2. 补丁版 FlagGems 源树置于 import 路径最前
    if FLAGGEMS_PATCHED_SRC not in sys.path:
        if Path(FLAGGEMS_PATCHED_SRC).is_dir():
            sys.path.insert(0, FLAGGEMS_PATCHED_SRC)
        else:
            raise RuntimeError(
                f"补丁版 FlagGems 源树不存在：{FLAGGEMS_PATCHED_SRC}。"
                "请以 env/v1 起容器（仓根挂载 /workspace）并确认宿主 FlagGems "
                "检出 xliu969/kunlun-gemm-compat。"
            )

    # 2.5 撤销镜像内 scikit-build editable finder 对 flag_gems 的劫持。
    #     _flag_gems_editable.ScikitBuildRedirectingFinder 在 sys.meta_path 中排在
    #     PathFinder 之前，把 flag_gems.* 固定解析到 /env/FlagGems/src（安装时的
    #     模块名→绝对路径映射），完全绕过 sys.path / PYTHONPATH。移除后由
    #     PathFinder 按 sys.path 解析，补丁源树（首位）生效。若 flag_gems 的
    #     csrc C 扩展在运行路径上需要镜像内构建产物，import 将显式报错，
    #     不会静默取到错版本。
    _hijackers = [f for f in sys.meta_path
                  if type(f).__name__ == "ScikitBuildRedirectingFinder"]
    if _hijackers:
        sys.meta_path[:] = [f for f in sys.meta_path if f not in _hijackers]
        print(f"[gemm_compat] removed meta_path hijacker(s): "
              f"{[type(f).__module__ + '.' + type(f).__name__ for f in _hijackers]}",
              file=sys.stderr)

    # 防御：若 flag_gems 已被（旧版）加载，清掉缓存再继续（本模块要求在
    # torch/flag_gems 之前 import，正常不触发）
    for _m in [m for m in list(sys.modules)
               if m == "flag_gems" or m.startswith("flag_gems.")]:
        del sys.modules[_m]

    # 3. 安装 compiler 版本锁（漂移即拒绝）
    import triton  # noqa: F401
    backend = importlib.import_module("triton.backends.xpu.compiler")
    installed = Path(backend.__file__)
    original = installed.read_bytes()
    digest = _sha256(original)
    if digest != _EXPECTED_COMPILER_SHA256:
        raise RuntimeError(
            f"安装的 compiler.py SHA256={digest} ≠ 验收基线 {_EXPECTED_COMPILER_SHA256}。"
            "flagtree/镜像版本已漂移，本 compat 包不适用（禁止盲目启用）。"
        )

    # 4. 生成 diagnostic 副本并进程内挂载（安装目录不改动）
    if str(_HOME) not in sys.path:
        sys.path.insert(0, str(_HOME))
    from prepare import prepare

    patched = prepare(original, "diagnostic")
    out = Path(tempfile.gettempdir()) / "xpu-compiler-copy-diagnostic.py"
    out.write_bytes(patched)
    spec = importlib.util.spec_from_file_location("isolated_sdnn_compiler", out)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    live_backend = backend.XPUBackend
    for name in ("make_ttxir", "make_llir"):
        setattr(live_backend, name, staticmethod(getattr(module.XPUBackend, name)))

    # 自证：安装文件未被改动；生效的是副本实现
    if installed.read_bytes() != original:
        raise RuntimeError("安装的 compiler.py 被改动（不应发生）")
    print(
        f"[gemm_compat] enabled: installed_compiler={digest[:12]}… "
        f"copy_sha256={_sha256(patched)} flaggems_src={FLAGGEMS_PATCHED_SRC}",
        file=sys.stderr,
    )


enable()
