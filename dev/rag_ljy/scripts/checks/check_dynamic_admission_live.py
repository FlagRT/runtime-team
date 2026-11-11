#!/usr/bin/env python3
"""Bounded live verification of dynamic estimates using real RAG requests.

Default inputs require estimates <=8GiB; extended input/batch coverage has an
explicit estimate ceiling. Each executed request requires 8GiB predicted free
headroom. A known dangerous request must be rejected before model work.
This is NOT the 85% three-mode pressure acceptance test.
"""
import json
from pathlib import Path
import shutil
import time
import argparse
import sys
from datetime import datetime, timezone
from check_watermarks_modes import ROOT, http, conflicting_processes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true', help='include independent input/batch combinations, default estimate ceiling 16GiB')
    parser.add_argument('--max-estimate-mib', type=int, default=16384, help='extended validation estimate ceiling, maximum 32768 MiB; always also require 8GiB predicted headroom')
    args=parser.parse_args()
    if not 1 <= args.max_estimate_mib <= 32768:
        parser.error('max-estimate-mib must be between 1 and 32768')
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError('Another load test is active: '+json.dumps(conflicts))
    url='http://127.0.0.1:8088'
    def health():return http(url+'/health')[2]
    def drain():
        end=time.monotonic()+30
        while time.monotonic()<end:
            h=health()
            if h['inflight']==0 and h['memory_admission']['reserved_mib']==0:
                return h
            time.sleep(.1)
        raise RuntimeError('API did not drain')
    before=drain()
    assert before['memory_estimation']['enabled'] and before['memory_admission']['config']['request_budget_mib'] is None
    records=[json.loads(l) for l in (ROOT/'experiment_outputs/inputs/request_pressure/queries.jsonl').read_text().splitlines()]
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file('/mnt/raid/jliu171/models/Qwen/Qwen3-Embedding-0.6B/tokenizer.json')
    long=next(q['query'] for q in records if q['tier']=='6144')
    long=tokenizer.decode(tokenizer.encode(long,add_special_tokens=False).ids[:4943],skip_special_tokens=False)
    cases={
        'short':{'query':'What are the benefits of exercise?', 'reranker_batch_size':1},
        'context_1024':{'query':next(q['query'] for q in records if q['tier']=='1024'), 'reranker_batch_size':1},
        'known_large':{'query':long, 'reranker_batch_size':32, 'coarse_top_k':50},
    }
    if args.extended:
        cases.update({
            'short_batch4': {'query':'How does exercise affect blood pressure?', 'reranker_batch_size':4},
            'context_1024_holdout_batch2': {'query':[q['query'] for q in records if q['tier']=='1024'][1], 'reranker_batch_size':2},
            'context_2048_holdout_batch1': {'query':[q['query'] for q in records if q['tier']=='2048'][1], 'reranker_batch_size':1},
            'context_2048_holdout_batch2': {'query':[q['query'] for q in records if q['tier']=='2048'][2], 'reranker_batch_size':2},
            'context_2048_holdout_batch4': {'query':[q['query'] for q in records if q['tier']=='2048'][3], 'reranker_batch_size':4},
            'context_4096_holdout_batch1': {'query':[q['query'] for q in records if q['tier']=='4096'][1], 'reranker_batch_size':1},
        })
    out=ROOT/'experiment_outputs/checks/dynamic_estimator_live'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    shutil.copyfile(__file__,out/'checker.py')
    (out/'inputs.json').write_text(json.dumps(cases,indent=2))
    result={'status':'failed','command':sys.argv,'options':vars(args),'before':before,'cases':{},'scope':'Bounded dynamic admission check, not 85% pressure verification'}
    print('OUTPUT='+str(out),flush=True)
    try:
        for name,body in cases.items():
            h=drain()
            code,headers,preview=http(url+'/memory/estimate',body)
            assert code==200
            estimate=preview['memory_estimate']['estimated_incremental_mib']
            assert health()['memory_admission']['accepted']==h['memory_admission']['accepted']
            if name!='known_large':
                ceiling=args.max_estimate_mib if args.extended else 8192
                if estimate>ceiling:
                    result['cases'][name]={'estimate':preview['memory_estimate'], 'http_status':None,
                                          'status':'not_run_estimate_exceeds_verifier_limit', 'limit_mib':ceiling}
                    continue
                if h['device_memory']['stats']['free_mb']-estimate<8192:
                    result['cases'][name]={'estimate':preview['memory_estimate'], 'http_status':None,
                                          'status':'not_run_insufficient_predicted_headroom', 'health':h}
                    continue
            else:
                if estimate<=h['device_memory']['stats']['free_mb']:
                    result['cases'][name]={'estimate':preview['memory_estimate'], 'http_status':None,
                                          'status':'not_run_batch_planning_now_fits'}
                    continue
            start=time.monotonic()
            code,headers,response=http(url+'/query',body,timeout=120)
            after=drain()
            result['cases'][name]={'estimate':preview['memory_estimate'],'http_status':code,'headers':headers,
                                   'response':response,'seconds':time.monotonic()-start,'health_after':after}
            if name=='known_large':
                assert code==503 and response['detail']['reason']=='request_budget'
                assert headers.get('retry-after')=='1'
                assert after['memory_admission']['accepted']==h['memory_admission']['accepted']
            else:
                assert code==200 and float(headers['x-memory-estimate-mib'])==estimate
            assert after['memory_estimation']['fault'] is None
            print(json.dumps({'case':name,'estimated_mib':estimate,'http_status':code}),flush=True)
        assert result['cases']['short']['estimate']['estimated_incremental_mib'] < result['cases']['context_1024']['estimate']['estimated_incremental_mib'] < result['cases']['known_large']['estimate']['estimated_incremental_mib']
        skipped=[name for name,case in result['cases'].items() if case['http_status'] is None]
        result['skipped_cases']=skipped
        result['status']='incomplete' if skipped else 'passed'
    except Exception as e:
        result['error']=repr(e)
        raise
    finally:
        try:
            result['after']=health()
        except Exception as e:
            result['final_health_error']=repr(e)
            result['status']='failed'
        (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('Bounded dynamic-admission checks: '+result['status']+'; 85% acceptance remains unverified.')


if __name__=='__main__':main()
