#!/usr/bin/env python3
"""Plot measured memory and software reservations separately from saved evidence."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/rag-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
output=a.output_dir or a.summary.parent
output.mkdir(parents=True,exist_ok=True)
s=json.loads(a.summary.read_text())
fig,axes=plt.subplots(3,2,figsize=(12,9),sharex='row')
for i,mode in enumerate(('burst','staggered','continuous')):
    trial=Path(s['modes'][mode]['trial'])
    if not trial.is_absolute():trial=ROOT/trial
    events=[json.loads(l) for l in (trial/'events.jsonl').read_text().splitlines()]
    samples=sorted((e for e in events if e['kind']=='health'),key=lambda e:e['t'])
    start=samples[0]['t'];t=[e['t']-start for e in samples]
    used=[100*max(e['health']['device_memory']['stats']['used_mb'],e['health']['device_memory']['stats']['total_mb']-e['health']['device_memory']['stats']['free_mb'])/e['health']['device_memory']['stats']['total_mb'] for e in samples]
    reserved=[e['health']['memory_admission']['reserved_mib']/1024 for e in samples]
    # Do not interpolate through HTTP observation gaps during recovery probes.
    tx,uy,ry=[],[],[]
    for j,x in enumerate(t):
        if j and x-t[j-1]>.05:
            tx.append(None);uy.append(None);ry.append(None)
        tx.append(x);uy.append(used[j]);ry.append(reserved[j])
    axes[i,0].plot(tx,uy,label='Measured device use',linewidth=1.3,marker='.',markersize=1.5)
    axes[i,0].axhline(85,color='red',linestyle='--',label='85% pause')
    axes[i,0].axhline(75,color='green',linestyle=':',label='75% resume')
    axes[i,0].set(ylim=(0,100),ylabel='Device memory (%)',title=mode+' — real memory')
    axes[i,1].plot(tx,ry,color='purple',linewidth=1.3,marker='.',markersize=1.5)
    axes[i,1].set(ylabel='Software reservation (GiB)',title=mode+' — accounting (not watermark)')
    for ax in axes[i]:ax.set_xlabel('Seconds since trial telemetry began');ax.grid(alpha=.2)
axes[0,0].legend(loc='lower right',fontsize=8)
fig.suptitle('Real RAG requests: independent cold-start trials, service-managed recovery\nGaps >50ms are not interpolated; NPU compute utilization was not measured',fontsize=12)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(output/'memory_timelines.svg')
fig.savefig(output/'memory_timelines.png',dpi=160)
print(output/'memory_timelines.svg')
