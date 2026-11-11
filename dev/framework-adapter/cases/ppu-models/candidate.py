"""Explicit experiment; model qualification remains a separate approval step."""
from dataclasses import asdict
from pathlib import Path
from model_admission import prepare_model, workload_from_report

def prepare_candidate(a, report, operator, shape):
    if a.mode != 'candidate':
        return prepare_model(a,report,operator,shape)
    if a.adapter_config or not a.records or not a.reference:
        raise ValueError('Candidate requires operator records and a native oracle only')
    import flagrt_adapter
    from flagrt_adapter import load_records
    from flagrt_adapter.records import environment_from_record
    from flagrt_adapter.startup import prepare_policies
    from model_probe import file_hash
    bridge=Path(flagrt_adapter.__file__).parent/'bridge.py'
    operators={operator:dict(shapes=[shape],dtypes=['torch.float32'])}
    report['adapter_bridge_sha256']=file_hash(bridge)
    report['workload_identity']=asdict(workload_from_report(report,operators,file_hash(bridge)))
    actual=environment_from_record(report['environment'])
    config=dict(schema=2,enabled=True,require_target=True,records=None,operators=operators)
    policies,report['startup']=prepare_policies(config,load_records(a.records),actual)
    report.update(qualification_experiment=True,model_approved=False)
    return actual,policies
