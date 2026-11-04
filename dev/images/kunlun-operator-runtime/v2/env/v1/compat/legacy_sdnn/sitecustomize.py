# env/v1 GEMM compat · 启用范围内进程的启动期全套启用（我方追加，非编译组包内容）
#
# 生效条件：本文件被 Python 的 site 机制加载 = 本目录已进入 PYTHONPATH =
# 该 shell source 过 enable-gemm-compat.sh（显式启用）。未启用的 shell 完全不受影响。
#
# 为什么在启动期做（而不是只靠脚本首行 import gemm_compat）：
#   1. 镜像内 FlagGems 是 scikit-build editable 安装，其 meta_path finder
#      （_flag_gems_editable.ScikitBuildRedirectingFinder）排在 PathFinder 之前，
#      把 flag_gems.* 固定解析到 /env/FlagGems/src，绕过 sys.path / PYTHONPATH；
#   2. 多进程测试工具（如 minirepro 每形状一个子进程）不继承父进程的
#      monkey-patch —— 在解释器启动期启用，启用 shell 的一切孙进程自动生效。
# gemm_compat.py 仍可直接 import（脚本首行用法不变，双保险）。
import sys

# 1. 解除 flag_gems import 劫持
sys.meta_path[:] = [
    f for f in sys.meta_path
    if type(f).__name__ != "ScikitBuildRedirectingFinder"
]

# 2. 全套启用（compiler 版本锁校验 + diagnostic 副本进程内挂载 + 补丁源树置前）。
#    失败（如版本漂移、非 env/v1 容器误把本目录加入 PYTHONPATH）只告警不炸进程，
#    随后 import flag_gems 的行为会显式暴露问题，不静默。
try:
    import gemm_compat  # noqa: F401  （模块级完成 enable()）
except Exception as _e:  # noqa: BLE001
    print(f"[gemm-compat/sitecustomize] 警告：启用未完成：{_e}", file=sys.stderr)
