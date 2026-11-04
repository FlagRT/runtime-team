"""Verify both ranks, real kernel logs, adapter decisions and upstream verdicts."""
import json
from pathlib import Path


def check_run(output,mode,*,measured=False):
    if mode not in ('off','on','unqualified'):
        raise ValueError('Unknown training mode')
    rows=[]
    for rank in range(2):
        root=Path(output)/'train'
        train=json.loads((root/f'train_leg_result_rank{rank}.json').read_text())
        adapter=json.loads((root/f'adapter_rank{rank}.json').read_text())
        hits=json.loads((root/f'kernel_hits_rank{rank}.json').read_text())
        assert train['rank']==rank and train['world_size']==2
        assert train['verdict']=='TRAIN_LEG_PASS' and train['passed']==train['total']==6
        assert train['perf']['steps']==3 and train['perf']['batch']==1 and train['perf']['seq']==32
        assert adapter['status']=='PASS' and not adapter['scope_failed']
        counts={(r['operator'],r['backend'],r['stage'],r['reason']):r['count'] for r in adapter['counts']}
        if mode=='on':
            assert counts=={('silu','flaggems','forward','completed'):84,('silu','flaggems','backward','completed'):84},counts
            assert hits=={'GEMS SILU FORWARD':84,'GEMS SILU BACKWARD':84},hits
        else:
            reason='experimental_disabled' if mode=='off' else ('workload_unverified' if measured else 'capability_unverified')
            assert counts=={('silu','native','forward',reason):84},counts
            assert not hits,hits
        rows.append(dict(rank=rank,loss_curve=train['loss_curve'],counts=adapter['counts'],kernel_hits=hits,
                         upstream_verdict=train['verdict']))
        if measured:
            identity=json.loads((root/f'workload_rank{rank}.json').read_text())
            assert adapter['startup']['workload_required'] is True
            assert adapter['startup']['workload']==identity['workload']
            guard=json.loads((root/f'step-control/rank{rank}/report.json').read_text())
            assert guard['status']=='PASS' and guard['steps']==3 and guard['hooks_restored']
            rows[-1]['workload']=identity['workload']
    if measured: assert rows[0]['workload']==rows[1]['workload'], 'rank identity disagreement'
    return dict(status='pass',mode=mode,ranks=rows)
