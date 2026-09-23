from dynamic_batching import BatchPolicy, BatchRequest


class ManualClock:
    def __init__(self, start_ns: int = 1_000_000_000):
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += int(ms * 1e6)


def make_policy(**overrides) -> BatchPolicy:
    defaults = dict(
        model_id="qwen3-embedding-0.6b",
        policy_version="v1",
        bucket_upper_bounds=(128,),
        max_batch_size=8,
        max_total_tokens=1024,
        max_wait_ms=50.0,
    )
    defaults.update(overrides)
    return BatchPolicy(**defaults)


def make_request(rid: str, tokens: int, arrival: int = 0, **overrides) -> BatchRequest:
    defaults = dict(
        request_id=rid,
        model_id="qwen3-embedding-0.6b",
        model_version="v1.0",
        input_token_count=tokens,
        payload_ref=f"payload-{rid}",
        arrival_mono_ns=arrival,
    )
    defaults.update(overrides)
    return BatchRequest(**defaults)
