#!/usr/bin/env python3
"""Print the registered implementation order for representative operators."""

from vllm_fl.dispatch import get_default_manager


REPRESENTATIVE_OPS = (
    "silu_and_mul",
    "rms_norm",
    "rotary_embedding",
    "topk_softmax",
)


def main() -> int:
    manager = get_default_manager()
    failed = False

    for op_name in REPRESENTATIVE_OPS:
        try:
            candidates = manager.resolve_candidates(op_name)
        except RuntimeError as exc:
            failed = True
            print(f"{op_name}: ERROR: {exc}")
            continue

        routes = ", ".join(
            f"{impl.impl_id}(runtime_fallback_safe={impl.runtime_fallback_safe})"
            for impl in candidates
        )
        print(f"{op_name}: {routes}")

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
