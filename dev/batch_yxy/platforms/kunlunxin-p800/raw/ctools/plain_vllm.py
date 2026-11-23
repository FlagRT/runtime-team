"""裸 vLLM 冒烟：不含任何 FlagScheduler 代码，验证故障是否与仓库无关。"""
import time


def main() -> None:
    from vllm import LLM

    t0 = time.time()
    llm = LLM(  # noqa: F841
        model="/work/models/qwen3-embed-0.6b-xpu",
        max_model_len=4096,
        dtype="float16",
        enforce_eager=True,
        task="embed",
    )
    print(f"[裸vllm] 加载完成 {time.time() - t0:.1f}s — 不涉及 FlagScheduler", flush=True)


if __name__ == "__main__":
    main()
