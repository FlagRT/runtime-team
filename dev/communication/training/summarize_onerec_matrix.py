"""Validate every scheduled run; statistical unit is an independent launch."""
import argparse
import json
from pathlib import Path
import statistics

from summarize_onerec import summarize


def summarize_matrix(root):
    manifest=json.loads((root/'matrix.json').read_text())
    runs=manifest['runs']
    repeats=manifest['repeats']
    if repeats<3 or len(runs)!=2*repeats:
        raise ValueError('Incomplete repeat matrix')
    expected={(r,cap) for r in range(1,repeats+1) for cap in (64,128)}
    observed={(r['repeat'],r['bucket_mib']) for r in runs}
    if observed!=expected or len({r['name'] for r in runs})!=len(runs):
        raise ValueError('Duplicate or unknown trial')
    first=None
    all_ranks=[]
    grouped={64:[],128:[]}
    for run in runs:
        path=root/run['name']
        if run['exit_code']!=0 or summarize(path,manifest['steps'])['verdict']!='PASS':
            raise ValueError(f"Failed trial: {run['name']}")
        pair=[json.loads((path/f'rank{rank}.json').read_text()) for rank in range(2)]
        for rank in pair:
            if (rank['backend']!='flagcx' or rank['comm_mode']!='bucketed'
                    or rank['seq_len']!=manifest['seq_len']
                    or rank['bucket_bytes']!=run['bucket_mib']*1024**2):
                raise ValueError('Trial metadata mismatch')
            identity={key:rank[key] for key in ('revision','dtype','optimizer','lr',
                      'global_batch','local_batch','implementation_sha256')}
            if first is None: first=identity
            if first!=identity: raise ValueError('Environment or implementation differs')
        grouped[run['bucket_mib']].append(pair)
        all_ranks.extend(pair)
    same_initial=all(r['initial_sha256']==all_ranks[0]['initial_sha256'] for r in all_ranks)
    same_final=all(r['final_sha256']==all_ranks[0]['final_sha256'] for r in all_ranks)
    metrics={}
    for cap,pairs in grouped.items():
        means=[max(statistics.mean(s['seconds_with_finite_checks'] for s in r['steps'][1:])
                   for r in pair) for pair in pairs]
        metrics[str(cap)]=dict(runs=len(pairs),run_mean_step_seconds_excluding_first=means,
            mean_seconds=statistics.mean(means),sample_stddev_seconds=statistics.stdev(means),
            end_used_mib_max=[max(r['memory']['used_mb'] for r in pair) for pair in pairs],
            gradient_calls_per_rank=[pair[0]['communication']['gradient_allreduce_calls'] for pair in pairs])
    return dict(verdict='PASS' if same_initial and same_final else 'FAIL',
                backend='flagcx',seq_len=manifest['seq_len'],steps=manifest['steps'],runs=len(runs),
                identical_initial_hashes_all_runs=same_initial,
                identical_final_hashes_all_ranks_and_buckets=same_final,metrics=metrics,
                scope='Synthetic tokens, full-parameter BF16/SGD; end memory snapshots, not peaks. '
                      'Timing includes finite checks; shared machine; no business quality claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args()
    result=summarize_matrix(a.root)
    with (a.root/'matrix-summary.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print('overall: '+result['verdict'])
    raise SystemExit(0 if result['verdict']=='PASS' else 1)
