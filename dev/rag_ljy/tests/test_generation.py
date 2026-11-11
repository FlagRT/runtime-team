from types import SimpleNamespace

import pytest

from rag_engine.generation import GenerationLimits, GenerationMemoryEstimator, PromptBuilder
from rag_engine.memory_estimate import ModelShape
from test_memory_estimate import estimator, req


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(c) for c in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['return_dict'] is False
        return self.encode('|'.join(m['content'] for m in messages) + '|assistant:')


def generation_estimator():
    return GenerationMemoryEstimator(
        ModelShape(4096, 12288, 32, 8, 128, 36, 176384, 2, True), GenerationLimits())


def test_generation_budget_grows_with_prompt_output_and_dtype():
    e = generation_estimator()
    short = e.estimate(128, 16)
    long = e.estimate(4096, 16)
    longer_output = e.estimate(4096, 1024)
    assert short['estimated_mib'] < long['estimated_mib'] < longer_output['estimated_mib']
    assert long['components']['kv_cache_mib'] == 2 * 2 * 8 * 128 * 36 * 2 * (4096+16) / 1024**2
    assert short['components']['output_mib'] == long['components']['output_mib']
    fp32 = GenerationMemoryEstimator(
        ModelShape(4096, 12288, 32, 8, 128, 36, 176384, 4, True), GenerationLimits())
    assert fp32.estimate(4096, 16)['estimated_mib'] > long['estimated_mib']


@pytest.mark.parametrize('input_tokens,output_tokens', [(0,1),(4097,1),(1,0),(1,1025),(True,1)])
def test_invalid_generation_sizes_fail_closed(input_tokens, output_tokens):
    with pytest.raises(ValueError):
        generation_estimator().estimate(input_tokens, output_tokens)


def test_context_keeps_question_and_reports_exact_truncation():
    b = PromptBuilder(CharacterTokenizer(), GenerationLimits(450, 64))
    hits = [{'_source': {'text': 'A'*500, 'document_id': 'first', 'title': 'Evidence'}},
            {'_source': {'text': 'B'*100, 'document_id': 'second'}}]
    prompt = b.build('QUESTION MUST SURVIVE', hits)
    assert len(prompt['input_ids']) <= 450
    assert 'QUESTION MUST SURVIVE' in b.tokenizer.decode(prompt['input_ids'])
    assert prompt['context_truncated']
    assert len(prompt['sources']) == 1
    assert prompt['sources'][0]['document_id'] == 'first'
    assert prompt['sources'][0]['truncated']
    assert prompt['sources'][0]['citation_id'] == 1


def test_small_context_keeps_all_sources_and_overlong_question_is_rejected():
    b = PromptBuilder(CharacterTokenizer(), GenerationLimits(800, 64))
    prompt = b.build('question', [{'text': 'first'}, {'text': 'second'}])
    assert not prompt['context_truncated']
    assert [s['text'] for s in prompt['sources']] == ['first', 'second']
    with pytest.raises(ValueError, match='Question'):
        b.build('Q'*800, [])


def test_request_plan_never_drops_generation_when_reducing_rerank_batch():
    e = estimator()
    request = req(100, batch=32)
    retrieval_only = e.estimate(request)['estimated_incremental_mib']
    e.generation_estimator = generation_estimator()
    request.generate, request.max_new_tokens = True, 512
    with_generation = e.estimate(request)
    generation = with_generation['generation']['estimated_mib']
    assert with_generation['estimated_incremental_mib'] == retrieval_only + generation
    plan = e.plan(request, generation + 1)
    assert plan['planned_reranker_batch_size'] == 1
    assert plan['generation']['estimated_mib'] == generation
    assert plan['estimated_incremental_mib'] > generation + 1
    request.generate = False
    assert e.estimate(request)['estimated_incremental_mib'] == retrieval_only


def test_generation_without_estimator_is_not_given_a_fixed_fallback():
    request = req()
    request.generate, request.max_new_tokens = True, 128
    with pytest.raises(ValueError, match='unavailable'):
        estimator().estimate(request)


def test_bound_execution_fault_and_cache_propagate_to_whole_request_estimator():
    from rag_engine.generation import _BoundGenerationEstimate
    owner = SimpleNamespace(fault=None, cache_snapshot={})
    bound = _BoundGenerationEstimate(generation_estimator(), owner, 16)
    bound.fault = 'underestimate'
    bound.cache_snapshot = {'reclaimable_mib': 123}
    assert owner.fault == 'underestimate'
    assert owner.cache_snapshot['reclaimable_mib'] == 123
    assert bound.forward_estimate(10, 1)['max_new_tokens'] == 16


def test_checkpoint_rejects_truncated_or_misindexed_shards(tmp_path):
    import json, struct
    from rag_engine.generation import checkpoint_manifest
    header = json.dumps({'weight':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}).encode()
    shard = tmp_path/'model.safetensors'
    shard.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*3)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(
        {'metadata':{'total_size':4},'weight_map':{'weight':shard.name}}))
    with pytest.raises(ValueError, match='Incomplete safetensors payload'):
        checkpoint_manifest(tmp_path)
    shard.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(
        {'metadata':{'total_size':4},'weight_map':{'other':shard.name}}))
    with pytest.raises(ValueError, match='tensor index'):
        checkpoint_manifest(tmp_path)


def test_real_onerec_chat_template_produces_flat_integer_ids_when_checkpoint_available():
    from pathlib import Path
    path = Path('/mnt/onerec/OneRec-8B')
    if not path.is_dir():
        pytest.skip('Optional downloaded OneRec tokenizer is not mounted')
    transformers = pytest.importorskip('transformers')
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    builder = PromptBuilder(tokenizer, GenerationLimits())
    prompt = builder.build('Which planet is largest?', [{'text':'Jupiter is the largest planet.'}])
    assert len(prompt['input_ids']) > 20
    assert all(isinstance(token,int) for token in prompt['input_ids'])
    assert 'Jupiter is the largest planet.' in tokenizer.decode(prompt['input_ids'])

