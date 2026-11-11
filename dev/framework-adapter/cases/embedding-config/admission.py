"""Unified schema-4 DeepFM entry; candidate is an isolated qualification probe."""
from dataclasses import asdict
import hashlib
from pathlib import Path

OPERATORS={'embedding':{'signatures':[
    dict(weight_shape=[3850,width],indices_shape=[32,5],
         weight_dtype='torch.float32',indices_dtype='torch.int64') for width in (16,1)]}}


def configuration(mode):
    if mode not in ('native','native_repeat','off','on','unknown'):
        raise ValueError('Production configuration cannot bypass model review')
    return dict(schema=4,enabled=mode in ('on','unknown'),operators=OPERATORS,records=None,
                embedding_records='records.json',
                workload_reviews='workload-reviews.json' if mode=='on' else None,
                require_target=mode=='on')


def prepare_model(a,r):
    import flagrt_adapter
    from flagrt_adapter.config import load_config
    from flagrt_adapter.embedding_policy import EmbeddingPolicy,embedding_signature,load_embedding_records
    from flagrt_adapter.records import environment_from_record
    from flagrt_adapter.startup import prepare_policies, package_content_identity, TargetBlocked
    from model_admission import workload_from_report
    package=Path(flagrt_adapter.__file__).parent
    r['adapter_bridge_sha256']=hashlib.sha256((package/'embedding.py').read_bytes()).hexdigest()
    identity,details=package_content_identity(flagrt_adapter.__file__)
    r['adapter_package_identity']=identity
    # Bind ALL installed adapter code (including config/policies), not only bridge.py.
    r['runtime_sources']['adapter-package']=identity
    workload=workload_from_report(r,OPERATORS,r['adapter_bridge_sha256'])
    r['workload_identity']=asdict(workload)
    actual=environment_from_record(r['environment'])
    if a.mode=='candidate':
        # Explicit standalone experiment that creates evidence, never a production
        # config or approval. No model review is issued by this execution path.
        r['qualification_only']=True
        signatures=tuple(embedding_signature(raw) for raw in OPERATORS['embedding']['signatures'])
        records=load_embedding_records('/input/review/records.json')
        policy=EmbeddingPolicy(actual,signatures,records,experimental=True)
        checks=[dict(operator='embedding',signature=asdict(sig),reason=policy.reason(actual,(sig,actual.device_type)))
                for sig in signatures]
        r['startup']=dict(checks=checks,qualification_only=True,workload=asdict(workload))
        if any(c['reason'] for c in checks): raise TargetBlocked('Candidate operator qualification absent',r['startup'])
        return actual,{'embedding':policy}
    if a.mode in ('native','native_repeat'):
        config,records=configuration(a.mode),()
    else:
        config,records=load_config(a.adapter_config)
        expected=configuration(a.mode)
        if expected['workload_reviews']:
            expected['workload_reviews']=str((a.adapter_config.parent/expected['workload_reviews']).resolve())
        if config!=expected: raise ValueError('Wrong unified DeepFM config')
        r['adapter_config_sha256']=hashlib.sha256(a.adapter_config.read_bytes()).hexdigest()
    r['configuration_schema']=config['schema']
    policies,r['startup']=prepare_policies(config,records,actual,workload=workload)
    expected_reason={'off':'experimental_disabled','unknown':'workload_unverified'}.get(a.mode)
    if expected_reason and any(c['reason']!=expected_reason for c in r['startup']['checks']):
        raise AssertionError('Unexpected startup route')
    return actual,policies


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
    if mode not in ('native','native_repeat'):
        # Unified scope also reports disabled unary entries; they stay native.
        expected.update({('relu','native','forward','operator_disabled'):3,
                         ('sigmoid','native','forward','operator_disabled'):1})
    if actual!=expected: raise AssertionError(f'Unexpected routes: {actual} vs {expected}')
