"""Script coverage with a local fake HTTP server; no real model/NPU requests."""
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/checks/check_generation_50.py'


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec=importlib.util.spec_from_file_location('generation_50_checker',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready_health():
    return {'status':'ready','max_inflight':0,'inflight':0,'retrieval_backend':'elasticsearch',
        'generation':{'enabled':True,'execution_profile':{'max_input_tokens':4096,'max_new_tokens':1024}},
        'memory_estimation':{'enabled':True,'fault':None},
        'device_memory':{'backend':'ascend','device':'npu:0','source':'device-context.runtime.memory_stats','status':'ok','stale':False,
                         'stats':{'total_mb':62740,'used_mb':30000,'free_mb':32740}},
        'memory_admission':{'status':'ready','reason':'memory_available','reserved_mib':0,
            'config':{'enabled':True,'estimate_requests':True,'request_budget_mib':None,
                      'safety_margin_mib':5120,'high_watermark':.85,'low_watermark':.75}}}


@contextmanager
def fake_service(*, reject_first=False, broken_generation=False, fault=False, priority_evidence=False):
    state={'posts':[],'health':ready_health()}
    if fault:
        state['health']['memory_estimation']['fault']='test-only fault'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def respond(self,status,body,headers=None):
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            for key,value in (headers or {}).items():
                self.send_header(key,value)
            self.end_headers(); self.wfile.write(json.dumps(body).encode())
        def do_GET(self):
            assert self.path=='/health'
            self.respond(200,state['health'])
        def do_POST(self):
            assert self.path=='/query'
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['posts'].append(body)
            if reject_first and len(state['posts'])==1:
                self.respond(503,{'detail':{'reason':'request_budget'}},{'Retry-After':'1'})
                return
            result={'answer':'A test-only answer [1]','sources':[{'citation_id':1}], 'hits':[{'rank':1}],
                    'usage':{'input_tokens':123,'output_tokens':8,'max_new_tokens':body['max_new_tokens']},
                    'finish_reason':'eos','timings_ms':{key:1.0 for key in (
                        'embedding_ms','rerank_ms','generation_ms','total_ms','embedding_queue_ms',
                        'retrieval_queue_ms','rerank_queue_ms','generation_queue_ms')}}
            if broken_generation:
                result.pop('answer')
            headers = {'X-Request-ID':str(len(state['posts'])),'X-Memory-Estimate-MiB':'18000'}
            if 'retrieval_backend' in body:
                headers['X-Retrieval-Backend'] = body['retrieval_backend']
                result['retrieval_backend'] = body['retrieval_backend']
            if priority_evidence:
                headers['X-Request-Priority'] = body['priority']
                result['scheduling'] = {'priority': body['priority'], 'stages': {
                    stage: [{'priority': body['priority'], 'effective_priority': body['priority'],
                             'overtook_lower_priority': 0}] for stage in
                    ('embedding','retrieval','rerank','generation')}}
            self.respond(200,result,headers)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    state['url']='http://127.0.0.1:'+str(server.server_port)
    try:
        yield state
    finally:
        server.shutdown(); server.server_close(); thread.join(5)


def args(service,parallel=1):
    return SimpleNamespace(url=service['url'],backend='elasticsearch',max_new_tokens=256,
                           parallel=parallel,ready_timeout=2,request_timeout=2)


def test_frozen_input_rejects_shortened_and_modified_sets(checker,tmp_path):
    path=checker.ROOT/'data/queries/nfcorpus-50.jsonl'
    assert len(checker.frozen_queries(path))==50
    altered=tmp_path/'queries.jsonl'
    altered.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='bytes differ'):
        checker.frozen_queries(altered)
    altered.write_text('\n'.join(path.read_text().splitlines()[:49])+'\n')
    with pytest.raises(ValueError,match='Exactly the 50'):
        checker.frozen_queries(altered)


def test_default_preflight_sends_no_query(checker,monkeypatch):
    with fake_service() as service:
        monkeypatch.setattr('sys.argv',[str(SCRIPT),'--backend','elasticsearch','--url',service['url']])
        assert checker.main()==0
        assert service['posts']==[]


@pytest.mark.parametrize('parallel',[1,2])
def test_all_50_distinct_queries_are_sent_once_with_generation(checker,tmp_path,parallel):
    queries=checker.frozen_queries(checker.ROOT/'data/queries/nfcorpus-50.jsonl')
    with fake_service() as service:
        summary=checker.replay(args(service,parallel),queries,ready_health(),tmp_path)
        assert summary['status']=='passed' and summary['successful_generation']==50
        assert len(service['posts'])==50
        assert sorted(b['query'] for b in service['posts'])==sorted(q.query for q in queries)
        assert all(b['generate'] is True and b['max_new_tokens']==256 for b in service['posts'])
    rows=[json.loads(line) for line in (tmp_path/'requests.jsonl').read_text().splitlines()]
    assert len({r['query_id'] for r in rows})==50
    assert (tmp_path/'memory.jsonl').exists() and not summary['unsubmitted_query_ids']


def test_budget_rejection_is_not_retried_or_counted_as_complete(checker,tmp_path):
    queries=checker.frozen_queries(checker.ROOT/'data/queries/nfcorpus-50.jsonl')
    with fake_service(reject_first=True) as service:
        summary=checker.replay(args(service),queries,ready_health(),tmp_path)
        assert len(service['posts'])==50
    assert summary['status']=='incomplete'
    assert summary['successful_generation']==49 and summary['http_503']==1
    assert summary['failed_query_ids']==[queries[0].query_id]


def test_200_without_generation_output_cannot_pass(checker,tmp_path):
    queries=checker.frozen_queries(checker.ROOT/'data/queries/nfcorpus-50.jsonl')
    with fake_service(broken_generation=True) as service:
        summary=checker.replay(args(service),queries,ready_health(),tmp_path)
    assert summary['status']=='failed' and summary['successful_generation']==0


def test_estimator_fault_stops_submissions_and_lists_unsubmitted_queries(checker,tmp_path):
    queries=checker.frozen_queries(checker.ROOT/'data/queries/nfcorpus-50.jsonl')
    with fake_service(fault=True) as service:
        summary=checker.replay(args(service),queries,ready_health(),tmp_path)
        assert service['posts']==[]
    assert summary['status']=='failed' and len(summary['unsubmitted_query_ids'])==50


@pytest.mark.parametrize('key,value',[('max_inflight',8),('generation',{'enabled':False})])
def test_preflight_preserves_required_runtime_configuration(checker,key,value):
    health=deepcopy(ready_health()); health[key]=value
    with pytest.raises(RuntimeError):
        checker.validate_service(health,'elasticsearch',256)


def test_priority_mode_requires_updated_live_service(checker):
    with pytest.raises(RuntimeError, match='Restart'):
        checker.validate_priority_service(ready_health())
    health = ready_health()
    health['scheduling'] = {'policy': 'priority_with_aging', 'dispatch_evidence': True}
    checker.validate_priority_service(health)


def test_priority_mode_sends_fifty_original_queries_and_rejects_missing_evidence(checker,tmp_path):
    with fake_service() as service:
        options = args(service,parallel=4)
        options.priority_cycle = True
        service['health']['scheduling'] = {'policy': 'priority_with_aging', 'dispatch_evidence': True}
        queries = checker.frozen_queries(checker.ROOT/'data/queries/nfcorpus-50.jsonl')
        result = checker.replay(options,queries,service['health'],tmp_path)
        assert len(service['posts']) == 50
        assert {p['query'] for p in service['posts']} == {q.query for q in queries}
        assert {p['priority'] for p in service['posts']} == {'high','normal','low'}
        assert result['status'] == 'failed'  # Fake responses intentionally lack dispatch evidence.
        assert result['priority_validation']['overtaking'] == 'not_observed'
        assert (tmp_path/'generated_responses.txt').is_file()


def test_fifty_generated_answers_without_contention_do_not_certify_overtaking(checker,tmp_path):
    with fake_service(priority_evidence=True) as service:
        options = args(service,parallel=4)
        options.priority_cycle = True
        service['health']['scheduling'] = {'policy': 'priority_with_aging', 'dispatch_evidence': True}
        queries = checker.frozen_queries(checker.ROOT/'data/queries/nfcorpus-50.jsonl')
        result = checker.replay(options,queries,service['health'],tmp_path)
        assert result['status'] == 'passed'
        assert result['successful_generation'] == 50
        assert result['priority_validation']['dispatch_decisions'] == 200
        assert result['priority_validation']['overtaking'] == 'not_observed'
        assert result['priority_validation']['aging'] == 'not_observed'


def test_vllm_workflow_preflight_requires_both_physical_dies(checker):
    h=ready_health()
    with pytest.raises(RuntimeError,match='complete RAG'):
        checker.validate_vllm_workflow(h)
    h['generation']['execution_profile']['backend']='vllm'
    h['device_memory_by_stage']={stage:{'physical_die':die,'source':'device-context.runtime.memory_stats'}
                                for stage,die in [('embedding_rerank',7),('generation',6)]}
    checker.validate_vllm_workflow(h)
    h['device_memory_by_stage']['generation']['physical_die']=7
    with pytest.raises(RuntimeError,match='physical die 6'):
        checker.validate_vllm_workflow(h)


def test_fifty_requests_select_backend_explicitly_on_dual_service(checker,tmp_path):
    with fake_service() as service:
        service['health']['retrieval_backend']='milvus'
        service['health']['retrieval_backends']=['milvus','elasticsearch']
        options=args(service,parallel=4)
        checker.validate_service(service['health'],'elasticsearch',256)
        queries=checker.frozen_queries(checker.ROOT/'data/queries/nfcorpus-50.jsonl')
        result=checker.replay(options,queries,service['health'],tmp_path)
        assert result['status']=='passed'
        assert len(service['posts'])==50
        assert all(p['retrieval_backend']=='elasticsearch' for p in service['posts'])
