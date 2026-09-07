#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inject_error_translation.py — D10 集成：把错误码翻译挂到 vLLM serve 的真实错误路径

【做什么】不修改 vLLM 源码，用 python -c 包装器在 cli.main() 之前替换
          vllm.entrypoints.openai.api_server 模块级的兜底异常处理器 `exception_handler`，
          使 serve 的所有未捕获异常（EngineGenerateError / EngineDeadError / 其他 Exception）
          都先经过 conformance/errors.py 的 translate_error 三维翻译，再走 vLLM 原逻辑。

【时序保证】cli.main() 内部 build_app() 执行 `app.exception_handler(Exception)(exception_handler)`
           时，取到的是**已被我们替换**的版本 → 无需碰 app 对象。

【验证点】serve 运行中发起一个触发 vLLM 异常的错误请求，日志应出现
         `[device-context][L2_PARAM] mapped=... graded_by=... :: root_cause`

【用法】改造后的 serve 启动（见 start_vllm_serve_910c.sh）：
    python3 -c "import inject_error_translation; inject_error_translation.serve()" serve <model> ...

【效果】A9 从"独立模块验证"升级为"接入真实推理错误路径" —— 联调时可见、可用。
"""
import os
import sys


def _conformance_dir() -> str:
    """定位 conformance 目录（含 errors.py / device_state.py / recovery.py）"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "ascend_regression", "conformance"),
        os.path.join(here, "..", "ascend_regression", "conformance"),
        "/mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/ascend_regression/conformance",
    ):
        if os.path.exists(os.path.join(cand, "errors.py")):
            return cand
    return here


def _translate(exc, location):
    """延迟 import：避免 serve 启动时加载 torch_npu 等重型依赖的顺序问题"""
    from errors import translate_error  # noqa: E402
    return translate_error(exc, location=location)


def patch():
    """替换 api_server 模块级各异常处理器（全部经 translate_error 再走原逻辑）"""
    import vllm.entrypoints.openai.api_server as api

    sys.path.insert(0, _conformance_dir())

    # 必须 async def：FastAPI/Starlette 异常中间件会 await handler（同步函数会抛
    # TypeError: 'coroutine' object is not callable —— 2026-09-02 集成实测发现）
    # 注意：外层 wrap 必须是**普通函数**（async def 会让 wrap(...) 返回 coroutine
    # 而非函数 → 注册后运行时抛 "the first argument must be callable"，已实测）
    def wrap(name, orig):
        async def _wrapped(request, exc):
            try:
                fe = _translate(exc, location=f"http:{request.url.path}")
                print(
                    f"[device-context][{name}][{fe.category.name}] "
                    f"mapped={fe.mapped} graded_by={fe.graded_by} "
                    f"code={fe.error_code} :: {fe.root_cause[:140]}",
                    flush=True,
                )
            except Exception as e:  # 翻译失败不阻断原逻辑
                print(f"[device-context][WARN] {name} 翻译失败: {e}", flush=True)
            return await orig(request, exc)
        return _wrapped

    # 覆盖 vLLM 全部推理错误路径（server_utils 提供，api_server 模块级引用）
    for name in (
        "exception_handler",          # 兜底 Exception
        "validation_exception_handler",  # 请求校验（超长 prompt 等）
        "engine_error_handler",       # 引擎生成错误
        "generation_error_handler",   # 生成层错误
    ):
        orig = getattr(api, name)
        setattr(api, name, wrap(name, orig))

    print("[device-context] 错误码翻译已挂接 vLLM 错误路径"
          "（exception/validation/engine/generation），D10 集成", flush=True)


def serve():
    """patch 后进入 vLLM serve 入口（argv 透传）"""
    patch()
    # 实际入口（vllm 0.20.2）：vllm.entrypoints.cli.main:main，而非 vllm.entrypoints.openai.cli
    from vllm.entrypoints.cli.main import main

    main()


if __name__ == "__main__":
    serve()
