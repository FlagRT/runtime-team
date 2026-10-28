"""Fail-closed validation of repeated manual/fused training evidence."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(root):
    manifest = json.loads((root/'execution.json').read_text())
    runs = manifest['runs']
    grouped = {m: [r for r in runs if r['mode'] == m] for m in ('manual', 'fused')}
    if len(runs) != sum(map(len, grouped.values())) or any(len(v) < 3 for v in grouped.values()):
        raise ValueError('Require >=3 trials per mode; no unknown modes')
    reference = None
    records = {}
    for run in runs:
        if run['exit_code'] != 0 or run['timed_out']: raise ValueError('Unclean launcher exit')
        key = (run['repeat'], run['mode'])
        if key in records: raise ValueError('Duplicate trial')
        pair = [json.loads((root/run['name']/f'rank{rank}.json').read_text()) for rank in range(2)]
        for rank, r in enumerate(pair):
            if not (r['rank'] == rank and r['world_size'] == 2 and r['group_state'] == 'CLOSED'
                    and r['verdict'] == 'PASS' and r['comm_mode'] == run['mode']
                    and r['global_batch'] == 64 and r['local_batch'] == 32
                    and r['measured_steps'] == 50 and r['warmup_steps'] == 5):
                raise ValueError('Unexpected rank metadata')
            identity = {k:r[k] for k in ('backend','evidence_scope','model_sha256','prototype_revision','torch','seed','dtype')}
            if reference is None: reference = identity
            if reference != identity: raise ValueError('Environment/model mismatch')
            if manifest['scope'] == 'CPU/Gloo host-only':
                if r['backend'] != 'gloo' or r['evidence_scope'] != 'CPU/gloo host-only':
                    raise ValueError('Scope mismatch')
            elif manifest['scope'] == 'Ascend integration':
                if r['backend'] not in ('hccl','flagcx') or r['evidence_scope'] != 'NPU integration':
                    raise ValueError('Scope mismatch')
            elif manifest['scope'] == 'Kunlun integration':
                if (r['backend'] != 'flagcx' or r['evidence_scope'] != 'P800 integration'
                        or not r.get('device','').startswith('cuda:')
                        or r.get('runtime_info',{}).get('name') != 'kunlun'):
                    raise ValueError('Kunlun vendor/device/backend mismatch')
            else: raise ValueError('Unknown scope')
            o, t = r['oracle'], r['training']
            if not (len(o['gradient_checks']) == 17 and all(c['pass'] for c in o['gradient_checks'])
                    and o['update_pass'] and o['rank_parameter_max_abs'] == 0
                    and t['rank_parameter_max_abs'] == 0): raise ValueError('Correctness failure')
            for field in ('losses', 'step_seconds'):
                if len(t[field]) != 50 or not all(math.isfinite(v) for v in t[field]):
                    raise ValueError('Missing or nonfinite samples')
            if min(t['step_seconds']) <= 0: raise ValueError('Invalid timing')
            if t['sample_id_ranges'] != [[s*64+rank*32,s*64+(rank+1)*32] for s in range(5,55)]:
                raise ValueError('Invalid shard coverage')
            stats = t['gradient_communication']
            if stats['gradient_allreduce_calls'] != (850 if run['mode']=='manual' else 50):
                raise ValueError('Unexpected gradient call count')
            if stats['gradient_payload_bytes'] != t['parameter_count']*4*50:
                raise ValueError('Unexpected input payload count')
        records[key] = pair
    if {r for r,m in records if m=='manual'} != {r for r,m in records if m=='fused'}:
        raise ValueError('Unpaired trials')
    deltas = [abs(a-b) for (trial,mode), pair in records.items() if mode=='manual'
              for rank in range(2) for a,b in zip(pair[rank]['training']['losses'],
              records[(trial,'fused')][rank]['training']['losses'])]
    if max(deltas) >= 1e-6: raise ValueError('Manual/fused loss mismatch')
    metrics = {}
    for mode in grouped:
        # Statistical unit = a run, not each rank or each correlated step.
        values = [max(sum(r['training']['step_seconds'])/50 for r in pair)*1000
                  for (trial,m), pair in records.items() if m==mode]
        metrics[mode] = {'run_mean_step_ms': values, 'mean_ms': statistics.mean(values),
                         'sample_stddev_ms': statistics.stdev(values)}
    return {'verdict':'PASS', 'scope':manifest['scope'], 'identity':reference,
            'runs':len(runs), 'gradient_checks_per_rank':17,
            'gradient_reference_max_abs':max(c['max_abs'] for pair in records.values() for r in pair for c in r['oracle']['gradient_checks']),
            'update_reference_max_abs':max(r['oracle']['update_max_abs'] for pair in records.values() for r in pair),
            'manual_fused_loss_max_abs':max(deltas), 'final_rank_parameter_max_abs':0,
            'gradient_calls_per_rank_per_50_steps':{'manual':850,'fused':50},
            'timing':metrics,
            'caveat':'Synthetic data; no CTR claim. Input bytes are not wire traffic. Timings include buffer broadcasts/packing; exclude data generation and post-step validation. CPU results do not establish NPU performance or FlagCX correctness.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); args=p.parse_args()
    result=summarize(args.root)
    with (args.root/'summary.json').open('x') as output: json.dump(result,output,indent=2)
    print(json.dumps(result,indent=2))
