#!/usr/bin/env python3
"""Audit three independent trials and stale-sampler validation; no NPU calls."""
import argparse, hashlib, json, shutil, statistics, subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',nargs=3,type=Path,required=True)
    parser.add_argument('--deployments',nargs=3,type=Path,required=True)
    parser.add_argument('--stale',type=Path,required=True)
    args=parser.parse_args()
    out=ROOT/'experiment_outputs/checks/memory_acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    result={'status':'passed','scope':'Three independent cold-start Milvus trials; each contains its own real-request high/middle/low cycle without restart.',
            'compute_utilization':'not measured','pressure_source':'real RAG inference only; includes allocator cache produced by those requests',
            'modes':{},'stale':str(args.stale),'limitations':['Finite input/configuration coverage; analytical estimator is not a universal upper-bound proof.',
            'The same profiles on a warmed/reclaimed allocator did not always reach 85%; earlier incomplete trials are retained.',
            'Arrival modes describe HTTP arrivals; the reranker still has one model worker.']}
    for trial,deployment in zip(args.trials,args.deployments):
        summary=json.loads((trial/'summary.json').read_text());manifest=json.loads((trial/'manifest.json').read_text())
        assert summary['status']=='passed' and len(summary['modes'])==1
        mode=next(iter(summary['modes']));assert mode not in result['modes']
        assert all(summary['modes'][mode]['checks'].values()) and summary['minimum_free_mib']>=5120
        assert manifest['initial_health']['device_memory']['sampling_process']==summary['health_after']['device_memory']['sampling_process']
        initial=json.loads((deployment/'health_after.json').read_text())
        assert initial['device_memory']['sampling_process']==manifest['initial_health']['device_memory']['sampling_process']
        data=[json.loads(l) for l in (trial/'events.jsonl').read_text().splitlines()]
        requests=[e for e in data if e['kind']=='request'];loads=sorted((e for e in requests if e.get('role')=='load'),key=lambda r:r['sent'])
        decisions=[];budget=[]
        for e in requests:
            detail=e.get('body',{}).get('detail',{}) if isinstance(e.get('body'),dict) else {}
            if not isinstance(detail,dict):continue
            h=detail.get('admission_snapshot')
            if not h:continue
            m=h['device_memory']['stats'];percent=100*max(m['used_mb'],m['total_mb']-m['free_mb'])/m['total_mb']
            if e['status']==503 and detail.get('reason')=='memory_pressure':decisions.append(percent)
            if e['status']==503 and detail.get('reason')=='request_budget' and percent<85:
                estimate=detail['memory_estimate']['estimated_incremental_mib'];reserved=h['memory_admission']['reserved_mib']
                budget.append({'actual_percent':percent,'free_mib':m['free_mb'],'reserved_mib':reserved,'new_estimate_mib':estimate,
                               'remaining_mib':m['free_mb']-reserved-estimate,'retry_after':e['headers'].get('retry-after')})
        assert any(75<v<85 for v in decisions) and any(v>=85 for v in decisions)
        assert budget and all(b['remaining_mib']<5120 and b['retry_after'] for b in budget)
        row={**summary['modes'][mode],'trial':str(trial),'deployment':str(deployment),'minimum_sampled_free_mib':summary['minimum_free_mib'],
             'total_http_requests':len(requests),'roles':dict(Counter(e.get('role','unspecified') for e in requests)),
             'load_http_counts':dict(Counter(str(e['status']) for e in loads)),
             'load_interarrival_seconds':[b['sent']-a['sent'] for a,b in zip(loads,loads[1:])],
             'middle_rejection_percents':[v for v in decisions if 75<v<85], 'budget_rejections_below_high':budget,
             'final_actual_percent':summary['health_after']['device_memory']['stats']['memory_used_percent'],
             'inflight_at_first_high':summary['modes'][mode]['first_actual_high']['health']['inflight']}
        result['modes'][mode]=row
        shutil.copytree(deployment,out/('deployment_'+mode))
    assert set(result['modes'])=={'burst','staggered','continuous'}
    stale=json.loads((args.stale/'result.json').read_text());assert stale['status']=='passed'
    shutil.copytree(args.stale,out/'stale_memory')
    result['minimum_sampled_free_mib']=min(v['minimum_sampled_free_mib'] for v in result['modes'].values())
    result['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for root in ('src','scripts') for p in (ROOT/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    git=subprocess.run(['git','-c','safe.directory='+str(ROOT.parents[1]),'rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    result['git_head']=git.stdout.strip() if git.returncode==0 else {'error':git.stderr.strip(),'version_evidence':'source snapshots and SHA256 hashes'}
    status=subprocess.run(['git','-c','safe.directory='+str(ROOT.parents[1]),'status','--porcelain','--','.'],cwd=ROOT,capture_output=True,text=True)
    result['git_worktree_status']=status.stdout.splitlines() if status.returncode==0 else status.stderr
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('OUTPUT='+str(out));print('RESULT=passed')

if __name__=='__main__':main()
