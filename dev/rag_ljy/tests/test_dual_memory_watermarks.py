"""Acceptance evidence tests; these do not exercise real NPU watermarks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def checker(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/checks/check_dual_memory_watermarks.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('dual_watermarks_checker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reading(die, used):
    return {'physical_die': die, 'source': 'device-context.runtime.memory_stats',
            'status': 'ok', 'stale': False, 'age_seconds': .01, 'sampling_process': die + 100,
            'stats': {'used_mb': used, 'total_mb': 100000, 'free_mb': 100000-used}}


def state(used):
    pressure = used > 75000
    return {'watermark_basis': 'device_measured_usage', 'reserved_mib': 0,
            'status': 'paused' if pressure else 'ready',
            'reason': 'memory_pressure' if pressure else 'memory_available',
            'config': {'enabled': True, 'estimate_requests': True, 'request_budget_mib': None,
                       'safety_margin_mib': 5120, 'high_watermark': .85, 'low_watermark': .75}}


def health(primary=20000, generation=55000):
    return {'status': 'ready', 'max_inflight': 0, 'inflight': 0,
            'retrieval_backend': 'milvus', 'retrieval_backends': ['milvus', 'elasticsearch'],
            'memory_estimation': {'enabled': True, 'fault': None},
            'generation': {'enabled': True, 'execution_profile': {'backend': 'vllm', 'max_new_tokens': 1024}},
            'device_memory': reading(7, primary), 'memory_admission': state(primary),
            'device_memory_by_stage': {'embedding_rerank': reading(7, primary), 'generation': reading(6, generation)},
            'memory_admission_by_stage': {'embedding_rerank': state(primary), 'generation': state(generation)}}


def rejection(die, used, at, reason='memory_pressure'):
    return {'status': 503, 'sent': at, 'headers': {'retry-after': '1'},
            'response': {'detail': {'reason': reason, 'decision_monotonic': at,
                                   'admission_snapshot': {'device_memory': reading(die, used),
                                                          'memory_admission': state(used)}}}}


def evidence(die):
    h = health(85000 if die == 7 else 20000, 85000 if die == 6 else 55000)
    samples = [{'time': 10, 'health': h}, {'time': 20, 'health': health()}]
    rows = [rejection(die, 85000, 10), rejection(die, 80000, 15),
            {'status': 200, 'role': 'probe', 'sent': 20, 'completed': 21,
             'complete_generation': True, 'before': health(), 'after': health()},
            {'status': 200, 'role': 'load', 'sent': 5, 'completed': 18,
             'complete_generation': True, 'response': {'scheduling': {'stages': {
                 'embedding': [{'dispatched_monotonic': 6}]}}}}]
    return samples, rows


@pytest.mark.parametrize('stage,die', [('embedding_rerank', 7), ('generation', 6)])
def test_full_chain_and_die_separation(checker, stage, die):
    samples, rows = evidence(die)
    assert checker.assess_stage(samples, rows, stage)['status'] == 'passed'
    other = 'generation' if die == 7 else 'embedding_rerank'
    assert checker.assess_stage(samples, rows, other)['status'] == 'threshold_not_reached'


@pytest.mark.parametrize('missing', ['high', 'middle', 'recovery', 'accepted_completion'])
def test_missing_evidence_never_passes(checker, missing):
    samples, rows = evidence(7)
    rows.pop({'high': 0, 'middle': 1, 'recovery': 2, 'accepted_completion': 3}[missing])
    assert checker.assess_stage(samples, rows, 'embedding_rerank')['status'] == 'incomplete'


def test_budget_rejection_not_high_pressure(checker):
    samples, rows = evidence(7)
    rows[0]['response']['detail']['reason'] = 'request_budget'
    result = checker.assess_stage(samples, rows, 'embedding_rerank')
    assert result['checks']['actual_reached_85']
    assert not result['checks']['high_memory_503']
    assert result['status'] == 'incomplete'


def test_earlier_low_success_cannot_prove_recovery(checker):
    samples, rows = evidence(7)
    rows[2]['sent'] = 1
    assert not checker.assess_stage(samples, rows, 'embedding_rerank')['checks']['low_recovery_http_200']


def test_retry_after_and_valid_snapshot_required(checker):
    row = rejection(7, 85000, 10)
    row['headers'] = {}
    assert checker.decision(row, 'embedding_rerank') is None
    row['response'] = 'gateway error'
    assert checker.decision(row, 'embedding_rerank') is None


@pytest.mark.parametrize('change', ['sampler', 'policy', 'stale', 'headroom', 'generation_budget', 'engine'])
def test_safety_or_configuration_change_stops_trial(checker, change):
    initial = health()
    checker.validate_health(initial, initial, ['milvus', 'elasticsearch'], 256)
    changed = deepcopy(initial)
    if change == 'sampler': changed['device_memory_by_stage']['generation']['sampling_process'] += 1
    if change == 'policy': changed['memory_admission_by_stage']['generation']['config']['high_watermark'] = .9
    if change == 'stale': changed['device_memory_by_stage']['generation']['age_seconds'] = 4
    if change == 'headroom': changed['device_memory_by_stage']['generation']['stats']['free_mb'] = 5000
    if change == 'generation_budget': changed['memory_admission_by_stage']['generation']['reserved_mib'] = 19000
    if change == 'engine': changed['generation']['execution_profile']['model_path'] = 'changed'
    with pytest.raises(RuntimeError):
        checker.validate_health(changed, initial, ['milvus', 'elasticsearch'], 256)


def test_default_preflight_does_not_post(checker, monkeypatch):
    calls = []
    def http(url, *args, **kwargs):
        calls.append((url, args))
        return 200, {}, health()
    monkeypatch.setattr(checker, 'http', http)
    monkeypatch.setattr(checker, 'prepare_inputs', lambda *_: {})
    monkeypatch.setattr(sys, 'argv', ['check_dual_memory_watermarks.py'])
    assert checker.main() == 0
    assert calls == [('http://127.0.0.1:8088/health', ())]
