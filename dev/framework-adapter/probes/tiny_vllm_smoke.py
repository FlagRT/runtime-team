"""Offline random-weight Llama engine smoke; not model quality validation."""
import json
from pathlib import Path


def main():
    import os
    import yaml
    # Preserve vendor routes/blacklists, but never retry an executed failure.
    base_config = Path("/workspace/vllm-plugin-FL/vllm_fl/dispatch/config/ascend.yaml")
    config_data = yaml.safe_load(base_config.read_text())
    config_data["strict"] = True
    strict_config = Path("/workspace/tiny-smoke-strict.yaml")
    strict_config.write_text(yaml.safe_dump(config_data))
    os.environ["VLLM_FL_CONFIG"] = str(strict_config)
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path = Path("/workspace/tiny-random-llama")
    path.mkdir(exist_ok=True)
    torch.manual_seed(135)
    config = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = LlamaForCausalLM(config).half()
    model.save_pretrained(path)
    model.eval()
    prompt = [1, 4, 5, 6]
    reference = []
    with torch.inference_mode():
        for _ in range(4):
            logits = model(torch.tensor([prompt + reference])).logits[0, -1]
            reference.append(int(logits.argmax()))
    del model
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i for i in range(4, 128)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                           bos_token="[BOS]", eos_token="[EOS]",
                           pad_token="[PAD]").save_pretrained(path)

    from vllm import LLM, SamplingParams
    llm = LLM(model=str(path), tokenizer=str(path), dtype="float16",
              tensor_parallel_size=1, max_model_len=64, enforce_eager=True,
              gpu_memory_utilization=0.05, max_num_seqs=1,
              enable_prefix_caching=False)
    outputs = llm.generate([{"prompt_token_ids": prompt}],
                           SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
    tokens = list(outputs[0].outputs[0].token_ids)
    assert len(tokens) == 4 and all(0 <= t < 128 for t in tokens)
    print("TINY_RANDOM_MODEL_SMOKE " + json.dumps({"generated_token_ids": tokens,
          "cpu_reference_token_ids": reference, "matches_cpu": tokens == reference,
          "scope": "random weights, single NPU, prefill and decode; no quality claim"}))
    assert tokens == reference, "Engine runs, but greedy tokens differ from CPU reference"


if __name__ == "__main__":
    main()
