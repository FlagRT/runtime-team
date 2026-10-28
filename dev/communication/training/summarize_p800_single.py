"""Summarize P800 one-rank preparation, never a two-card acceptance result."""
import argparse
import json
from pathlib import Path
import statistics

from run_p800_selected import validate_single


def summarize(root):
    execution=json.loads((root/'selection-execution.json').read_text())
    if not (execution['mode']=='single' and execution['verdict']=='PASS'
            and len(execution['runs'])==2 and all(r['exit_code']==0 for r in execution['runs'])
            and len(execution['selected'])==1):
        raise ValueError('Require two successful one-rank launchers')
    identity=json.loads((root/'identity.json').read_text())
    expected=execution['selected'][0]['uuid'].removeprefix('GPU-')
    if not (identity['verdict']=='PASS' and len(identity['visible_uuid_order'])==1
            and identity['visible_uuid_order'][0].removeprefix('GPU-')==expected):
        raise ValueError('Visible physical device identity mismatch')
    records={}
    for mode in ('manual','fused'):
        validate_single(root,mode)
        records[mode]=json.loads((root/mode/'rank0.json').read_text())
    manual,fused=records.values()
    for field in ('model_sha256','prototype_revision','torch','seed','dtype'):
        if manual[field]!=fused[field]:raise ValueError('Mismatched model/environment')
    delta=max(abs(a-b) for a,b in zip(manual['training']['losses'],fused['training']['losses']))
    hashes_equal=manual['final_parameter_sha256']==fused['final_parameter_sha256']
    if delta>=1e-6 or not hashes_equal:raise ValueError('Mode numerical mismatch')
    return {'verdict':'PASS','scope':'P800 single-card preparation only','world_size':1,
            'two_card_acceptance':False,'launchers':2,'warmup_steps_per_launcher':5,
            'measured_steps_per_launcher':50,'global_batch':64,
            'selected':execution['selected'],'gradient_checks_per_launcher':17,
            'gradient_reference_max_abs':max(c['max_abs'] for r in records.values() for c in r['oracle']['gradient_checks']),
            'update_reference_max_abs':max(r['oracle']['update_max_abs'] for r in records.values()),
            'manual_fused_loss_max_abs':delta,'mode_final_parameter_hashes_equal':hashes_equal,
            'modes':{mode:{'loss_first':r['training']['losses'][0],
                           'loss_last':r['training']['losses'][-1],
                           'mean_step_ms':1000*statistics.mean(r['training']['step_seconds']),
                           'gradient_allreduce_calls':r['training']['gradient_communication']['gradient_allreduce_calls'],
                           'input_payload_bytes':r['training']['gradient_communication']['gradient_payload_bytes']}
                     for mode,r in records.items()},
            'caveat':'One rank cannot establish inter-card communication, rank consistency or scaling. Synthetic labels are not CTR quality. Shared-server one-run timings are diagnostic only. Input bytes are not wire traffic.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    result=summarize(args.root)
    with (args.root/'summary.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result,indent=2))
