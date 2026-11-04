from types import SimpleNamespace
import pytest

from rag_engine.memory_estimate import ModelShape, RequestMemoryEstimator
from rag_engine.admission import MemoryAdmissionConfig, MemoryAdmissionController
from rag_engine.forward_guard import ForwardMemoryGuard


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))
    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text)[:kwargs['max_length']]}


def estimator():
    shape = ModelShape(1024, 3072, 16, 8, 128, 28, 151669, 2, True)
    return RequestMemoryEstimator(shape, shape, Tokenizer(), Tokenizer(), 'retrieve')


def req(words=10, batch=4, candidates=30, embedding_batch=8):
    return SimpleNamespace(query='word '*words, reranker_batch_size=batch,
                           coarse_top_k=candidates, embedding_batch_size=embedding_batch)


def test_estimate_depends_on_tokens_and_actual_rerank_batch():
    e = estimator()
    short = e.estimate(req())
    long = e.estimate(req(6144))
    larger = e.estimate(req(6144, 8))
    assert short['estimated_incremental_mib'] < long['estimated_incremental_mib'] < larger['estimated_incremental_mib']
    assert short['estimated_incremental_mib'] != 2048
    assert long['rerank']['components']['attention_mib'] > short['rerank']['components']['attention_mib']


def test_single_query_embedding_does_not_multiply_by_api_batch_hint():
    e = estimator()
    assert e.estimate(req(100, embedding_batch=1)) == e.estimate(req(100, embedding_batch=8))
    assert e.estimate(req())['embedding']['batch'] == 1


def test_truncation_candidate_cap_and_known_bad_large_request():
    e = estimator()
    result = e.estimate(req(12000, 32, 20))
    assert result['embedding']['tokens'] == 8192
    assert result['rerank']['tokens'] == 8192 and result['rerank']['batch'] == 20
    # The former 4x4943-token workload cannot be assigned the old 2GiB budget.
    assert e.estimate(req(4943, 4))['estimated_incremental_mib'] > 40000


def test_resident_weights_not_counted_again_and_components_are_visible():
    e = estimator().estimate(req())
    assert e['estimated_incremental_mib'] == e['embedding']['estimated_mib'] + e['rerank']['estimated_mib']
    assert e['status'] == 'analytical_uncalibrated'
    assert e['document_token_limit'] == 1024


def test_execution_guard_shrinks_batch_before_any_allocation():
    estimates = SimpleNamespace(forward_estimate=lambda n, b: {'estimated_mib': n*b})
    stats = lambda: {'total_mb': 1000, 'used_mb': 500, 'free_mb': 500}
    reclaimed = []
    guard = ForwardMemoryGuard(estimates, stats, lambda: reclaimed.append(True), safety_mib=100)
    batch, estimate = guard.choose_batch([100]*8, 8)
    assert batch == 4 and estimate['estimated_mib'] == 400 and not reclaimed


def test_execution_guard_reclaims_once_then_fails_without_allocating():
    reclaimed = []
    e = SimpleNamespace(forward_estimate=lambda n, b: {'estimated_mib': 1000})
    g = ForwardMemoryGuard(e, lambda: {'total_mb': 1000, 'used_mb': 900, 'free_mb': 100},
                           lambda: reclaimed.append(True), safety_mib=100, timeout_s=0)
    with pytest.raises(RuntimeError, match='No safe single-item'):
        g.choose_batch([10], 1)
    assert reclaimed == [True]


def test_unknown_model_fails_closed():
    with pytest.raises(ValueError, match='Qwen3'):
        ModelShape.from_config({'model_type': 'other'})


def test_embedding_dtype_is_read_from_verified_engine_not_assumed(tmp_path):
    from rag_engine.memory_estimate import embedding_execution_profile
    proc = tmp_path/'123'
    proc.mkdir()
    command = ['python', '-m', 'vllm.entrypoints.openai.api_server', '--port', '8089',
               '--runner', 'pooling', '--no-enable-prefix-caching', '--model', str(tmp_path/'model'), '--dtype', 'float32', '--max-model-len', '8192', '--enforce-eager']
    (proc/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
    profile = embedding_execution_profile(tmp_path/'model', 'http://127.0.0.1:8089/v1', tmp_path)
    assert profile['dtype_bytes'] == 4
    command[command.index('--dtype')+1] = 'auto'
    (proc/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
    with pytest.raises(ValueError, match='dtype'):
        embedding_execution_profile(tmp_path/'model', 'http://127.0.0.1:8089/v1', tmp_path)


def test_unsupported_remote_engine_fails_closed(tmp_path):
    from rag_engine.memory_estimate import embedding_execution_profile
    with pytest.raises(ValueError, match='not verified'):
        embedding_execution_profile(tmp_path/'model', 'http://unverified:8089/v1', tmp_path)


def test_allocator_peak_exceeding_estimate_latches_execution_fault(caplog):
    import logging
    caplog.set_level(logging.INFO)
    mib = 1024**2
    estimator = SimpleNamespace(fault=None)
    allocator = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
        memory_allocated=lambda: 100*mib, memory_reserved=lambda: 200*mib,
        max_memory_allocated=lambda: 500*mib, max_memory_reserved=lambda: 600*mib)
    g = ForwardMemoryGuard(estimator, lambda: {}, lambda: None, allocator=allocator)
    with g.observe_forward({'estimated_mib':300, 'tokens':1024, 'batch':2}):
        pass
    assert estimator.fault
    with pytest.raises(RuntimeError, match='Previous forward'):
        g.choose_batch([1024], 1)
    assert 'forward_memory_observation' in caplog.text


def test_fused_accounting_requires_actual_operator_confirmation():
    e = estimator()
    e.flash_candidate = True
    e.execution_profile = {}
    e.fingerprint = {}
    dense = e.forward_estimate(6144, 4)
    assert not e.verify_flash_operators(['aten::scaled_dot_product_attention'])
    assert e.forward_estimate(6144, 4) == dense
    assert e.verify_flash_operators(['aclnnFlashAttentionScore'])
    fused = e.forward_estimate(6144, 4)
    assert fused['estimated_mib'] < dense['estimated_mib']
    assert fused['peak_live_mib'] >= fused['components']['output_mib'] + fused['components']['kv_cache_mib']
    assert e.forward_estimate(6144, 1, embedding=True)['components']['kv_cache_mib'] == 0
    assert fused['attention_accounting'] == 'verified_fused'


def test_planning_uses_actual_batch_without_discarding_candidates():
    e = estimator()
    request = req(4096, batch=32, candidates=50)
    allowed = e.estimate(req(4096, batch=4, candidates=50))['estimated_incremental_mib']
    plan = e.plan(request, allowed)
    assert plan['planned_reranker_batch_size'] == 4
    assert plan['estimated_incremental_mib'] == allowed
    assert request.reranker_batch_size == 32 and request.coarse_top_k == 50
    minimum = e.plan(request, 1)
    assert minimum['planned_reranker_batch_size'] == 1
    assert minimum['estimated_incremental_mib'] > 1  # admission must still reject


def test_fragmented_cache_is_reclaimed_before_shrinking_a_fitting_batch():
    mib=1024**2
    state={'reclaimed':False}
    estimator=SimpleNamespace(forward_estimate=lambda n,b: {'estimated_mib':250*b,'tokens':n,'batch':b,
        'components':{'output_mib':200*b},'workspace_mib':0})
    allocator=SimpleNamespace(memory_snapshot=lambda: [] if state['reclaimed'] else
        [{'total_size':200*mib,'active_size':0,'allocated_size':0,'device':0}])
    def reclaim():state['reclaimed']=True
    stats=lambda: {'total_mb':1000,'used_mb':700 if state['reclaimed'] else 900,'free_mb':300 if state['reclaimed'] else 100}
    guard=ForwardMemoryGuard(estimator,stats,reclaim,allocator=allocator,safety_mib=50)
    assert guard.choose_batch([100],1)[0]==1
    assert state['reclaimed']
