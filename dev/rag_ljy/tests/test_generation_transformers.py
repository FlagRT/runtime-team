"""CPU library-interface check; this is not a OneRec/NPU model validation."""
import pytest


def test_real_qwen3_generation_uses_full_prefill_then_cached_single_token_decode():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    config = transformers.Qwen3Config(vocab_size=128, hidden_size=64,
        intermediate_size=128, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=127, pad_token_id=0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    calls = []
    def observe(module, args, kwargs):
        calls.append((kwargs['input_ids'].shape[-1], kwargs.get('logits_to_keep')))
    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        ids = torch.tensor([[1,2,3,4]])
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                min_new_tokens=2, max_new_tokens=2, do_sample=False, num_beams=1,
                use_cache=True, cache_implementation='dynamic', logits_to_keep=1,
                return_dict_in_generate=False, output_scores=False, output_logits=False,
                output_hidden_states=False, output_attentions=False, pad_token_id=0)
        assert torch.equal(output[:,:4], ids)
        assert output.shape == (1,6)
        assert calls == [(4,1),(1,1)]
    finally:
        handle.remove()
