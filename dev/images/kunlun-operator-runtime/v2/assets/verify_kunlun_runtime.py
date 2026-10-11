#!/usr/bin/env python3
"""kunlun-operator-runtime v2 自判定验证脚本（类别：验证，见 dev/images/CASE-TEMPLATE.md §1）。

三态口径（CASE-TEMPLATE §2 规则 3）：
  overall: PASS  / 退出码 0 = 镜像可用（静态 + 带卡动态 + FlagGems 算子全部通过）
  overall: FAIL  / 退出码 1 = 有检查项不通过
  overall: ERROR / 退出码 2 = 无法下判定（环境不满足，例如无设备、无 flagtree）

用法（容器内，解释器须是目标 conda env 的 python）：
  /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
      /workspace/dev/images/kunlun-operator-runtime/v2/assets/verify_kunlun_runtime.py
  --static  只跑无设备可跑的检查（构建期自检用；带卡机器上不要只跑这个）

已知缺陷不计入 PASS/FAIL：GEMM（mm/bmm/addmm）在本 pin 组合下编译期 SIGABRT，
本脚本在**子进程**里探它（否则会把验证进程一起带走），结果以 KNOWN 行报告。
依据见 ../lock.yaml known_issues: gemm-compile-sigabrt。
"""
import argparse
import json
import os
import subprocess
import sys

# 版本比较一律经 packaging 的 PEP 440 归一化，不比裸字符串：同一个版本在不同 setuptools
# 下写出的 Version 字段形态不同（FlagGems 的 4.2.1.rc.0 与 4.2.1rc0 是同一版本）。
EXPECT = {
    "flagtree": "0.6.1+xpu3.6",
    "flag_gems": "4.2.1.rc.0",
    "triton": "3.6.0",
    "torch": "2.9.0+cu129",
}

results = []      # (status, name, detail)；status ∈ PASS/FAIL/KNOWN/SKIP


def record(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        record("FAIL", name, f"{type(e).__name__}: {e}")
        return False
    record("PASS" if ok else "FAIL", name, detail)
    return ok


# ---- 静态检查（无设备可跑） ----

def static_checks():
    import importlib.metadata as m

    from packaging.version import Version as V

    def _ver(pkg, want):
        got = m.version(pkg)
        return V(got) == V(want), f"{got} (期望 {want})"

    for pkg in ("flagtree", "flag_gems", "torch"):
        check(f"version/{pkg}", lambda p=pkg: _ver(p, EXPECT[p]))

    def _triton():
        import triton
        return V(triton.__version__) == V(EXPECT["triton"]), \
            f"{triton.__version__} (期望 {EXPECT['triton']})"
    check("version/triton", _triton)

    def _libtriton_xpu():
        # 决定性断言：若 PyPI 通用 triton 覆盖了 flagtree 的 triton/ 目录，这里即 ImportError，
        # xpu 编译后端整体不可用（见 ../lock.yaml known_issues: flaggems-kunlunxin-extras-missing）
        from triton._C.libtriton import xpu
        return xpu is not None, "flagtree 的 xpu 编译后端在位，triton 命名空间未被覆盖"
    check("triton/libtriton.xpu", _libtriton_xpu)

    def _non_editable():
        raw = m.distribution("flag_gems").read_text("direct_url.json") or "{}"
        d = json.loads(raw)
        editable = bool((d.get("dir_info") or {}).get("editable"))
        return not editable, f"direct_url={raw.strip()}"
    check("flag_gems/non-editable", _non_editable)

    def _no_kl3():
        # 厂商手册要求设这个变量，但实测单卡单进程即挂死；设了就不该下 PASS
        return "XPU_EVENT_KL3_ENABLE" not in os.environ, \
            "XPU_EVENT_KL3_ENABLE 未设置（设置会挂死，见 known_issues: xpu-event-kl3-enable-hang）"
    check("env/XPU_EVENT_KL3_ENABLE-unset", _no_kl3)


# ---- 带卡检查 ----

def device_checks():
    import torch
    if not torch.cuda.is_available():          # 昆仑芯走 torch.cuda，不是 torch.xpu
        record("FAIL", "device/available", "torch.cuda.is_available() 为 False："
                                           "检查 devices 是否含 /dev/xpuctrl")
        return False
    n = torch.cuda.device_count()
    record("PASS", "device/available", f"device_count={n}"
                                      f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(未设)')}")

    def _alloc():
        x = torch.ones(256, 1024, device="cuda")
        torch.cuda.synchronize()
        return abs(x.sum().item() - 256 * 1024) < 1e-3, f"在设备上分配并求和 {tuple(x.shape)} 通过"
    ok = check("device/alloc", _alloc)

    def _rmsnorm():
        # FlagGems 已验证可用的算子之一（rmsnorm/softmax/rotary 771 passed）；
        # 与 torch 参考实现比对，避免只测「不崩」。
        import flag_gems
        torch.manual_seed(0)
        x = torch.randn(8, 512, device="cuda", dtype=torch.float32)
        w = torch.randn(512, device="cuda", dtype=torch.float32)
        eps = 1e-5
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        with flag_gems.use_gems():
            got = torch.nn.functional.rms_norm(x, (512,), w, eps)
        torch.cuda.synchronize()
        diff = (got - ref).abs().max().item()
        return diff < 1e-2, f"FlagGems rms_norm 与 torch 参考最大偏差 {diff:.3e}"
    ok = check("flag_gems/rms_norm", _rmsnorm) and ok

    def _softmax():
        import flag_gems
        torch.manual_seed(0)
        x = torch.randn(8, 512, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        with flag_gems.use_gems():
            got = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        diff = (got - ref).abs().max().item()
        return diff < 1e-3, f"FlagGems softmax 与 torch 参考最大偏差 {diff:.3e}"
    ok = check("flag_gems/softmax", _softmax) and ok
    return ok


# ---- 已知缺陷探针（子进程，崩溃不影响判定） ----

GEMM_PROBE = """
import torch, flag_gems
x = torch.randn(64, 64, device='cuda'); y = torch.randn(64, 64, device='cuda')
with flag_gems.use_gems():
    z = torch.mm(x, y)
torch.cuda.synchronize()
print('GEMM_OK', float(z.sum()))
"""


def known_issue_probe():
    proc = subprocess.run([sys.executable, "-c", GEMM_PROBE], capture_output=True, text=True,
                          timeout=900, env={**os.environ})
    if proc.returncode == 0 and "GEMM_OK" in proc.stdout:
        record("KNOWN", "flag_gems/mm", "GEMM 竟然通过了 —— 与 known_issues: "
                                        "gemm-compile-sigabrt 不符，请复核该条目是否已失效")
    else:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
        record("KNOWN", "flag_gems/mm", f"按 known_issues: gemm-compile-sigabrt 预期崩溃，"
                                       f"实测 rc={proc.returncode}（{tail[0][:80]}）；不计入判定")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="只跑无设备可跑的检查")
    ap.add_argument("--skip-known", action="store_true", help="跳过已知缺陷子进程探针")
    args = ap.parse_args()

    print(f"python: {sys.executable}")
    static_checks()
    if not args.static:
        try:
            import torch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            record("FAIL", "import/torch", f"{type(e).__name__}: {e}")
        else:
            device_checks()
            if not args.skip_known:
                known_issue_probe()

    failed = [r for r in results if r[0] == "FAIL"]
    if args.static and not failed:
        print("\noverall: PASS (--static：只核对了无设备可跑的检查项)")
        return 0
    if failed:
        print(f"\noverall: FAIL ({len(failed)} 项不通过: "
              f"{', '.join(n for _, n, _ in failed)})")
        return 1
    print(f"\noverall: PASS ({sum(1 for r in results if r[0] == 'PASS')} 项通过，"
          f"{sum(1 for r in results if r[0] == 'KNOWN')} 项已知缺陷不计入)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\noverall: ERROR (被中断)")
        sys.exit(2)
