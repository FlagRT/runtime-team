"""Case-specific multi-input config; reuse package workload review semantics."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

OPERATORS={'embedding':{'contract':'embedding-dense-no-padding-v1',
    'weight_shapes':[[3850,16],[3850,1]],'indices_shape':[32,5],
    'weight_dtype':'torch.float32','indices_dtype':'torch.int64'}}

def prepare_model(a,r,*unused):
    import flagrt_adapter
    from flagrt_adapter.embedding import EmbeddingPolicy,EmbeddingSignature,load_embedding_records
    from flagrt_adapter.records import environment_from_record
    from flagrt_adapter.workload import WorkloadPolicy,load_workload_reviews,review_reason
    from flagrt_adapter.startup import StrictPolicy,TargetBlocked
    from model_admission import workload_from_report
    package=Path(flagrt_adapter.__file__).parent
    r['adapter_bridge_sha256']=hashlib.sha256((package/'embedding.py').read_bytes()).hexdigest()
    r['runtime_sources']['adapter-base-bridge.py']=hashlib.sha256((package/'bridge.py').read_bytes()).hexdigest()
    workload=workload_from_report(r,OPERATORS,r['adapter_bridge_sha256'])
    r['workload_identity']=asdict(workload)
    actual=environment_from_record(r['environment'])
    signatures=tuple(EmbeddingSignature(tuple(s),(32,5)) for s in OPERATORS['embedding']['weight_shapes'])
    enabled=a.mode not in ('native','native_repeat','off')
    records=load_embedding_records('/input/review/records.json') if enabled else ()
    policy=EmbeddingPolicy(actual,signatures,records,enabled)
    config=None
    if a.mode in ('off','on','unknown'):
        config=json.loads(a.adapter_config.read_text())
        if config != {'schema':'embedding-case-v1','mode':a.mode,'operators':OPERATORS}:
            raise ValueError('Wrong embedding case config')
        r['adapter_config_sha256']=hashlib.sha256(a.adapter_config.read_bytes()).hexdigest()
    if a.mode in ('on','unknown'):
        reviews=load_workload_reviews('/input/review/workload-reviews.json') if a.mode=='on' else ()
        policy=WorkloadPolicy(policy,review_reason(actual,workload,OPERATORS,reviews))
    if a.mode=='candidate': r['qualification_only']=True
    checks=[dict(operator='embedding',signature=asdict(s),reason=policy.reason(actual,(s,actual.device_type))) for s in signatures]
    r['startup']=dict(checks=checks,require_target=a.mode=='on',workload=asdict(workload))
    if a.mode=='on':
        if any(c['reason'] for c in checks): raise TargetBlocked('Embedding startup rejected',r['startup'])
        policy=StrictPolicy(policy)
    return actual,{'embedding':policy}

def check_model_routes(routes,mode):
    from collections import Counter
    actual=Counter({tuple(row[:-1]):row[-1] for row in routes})
    if mode in ('native','native_repeat'): expected=Counter()
    elif mode in ('candidate','on'):
        expected=Counter({('embedding','flaggems','forward','completed'):2,
                          ('embedding','flaggems','backward','completed'):2})
    else:
        reason={'off':'experimental_disabled','unknown':'workload_unverified'}[mode]
        expected=Counter({('embedding','native','forward',reason):2})
    if actual!=expected: raise AssertionError(f'Unexpected embedding routes: {actual} vs {expected}')
