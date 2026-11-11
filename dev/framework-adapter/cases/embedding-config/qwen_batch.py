"""Serial Qwen stages; stop on failure, no automatic approval or retry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
for n in ('root','input','model'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--card',type=int,required=True)
p.add_argument('--phase',choices=('qualification','native','candidates','controls','enabled'),required=True)
a=p.parse_args()
phases={'qualification':[('qualification',None,'qual-01')],
        'native':[('model','native','native-01')],
        'candidates':[('model','native_repeat','repeat-01'),('model','candidate','candidate-01'),('model','candidate','candidate-02')],
        'controls':[('model','off','off-01'),('model','unknown','unknown-01')],
        'enabled':[('model','on','on-01')]}
for stage,mode,label in phases[a.phase]:
    name='qemb-'+label
    command=[sys.executable,'-B',str(a.input/'run.py'),'--root',str(a.root),
        '--run-id',name,'--card',str(a.card),'--model',str(a.model),'--stage',stage]
    if mode:
        command+=['--mode',mode]
        if mode!='native':command+=['--reference',str(a.root/'qemb-native-01')]
    print('START',name,flush=True)
    with (a.root/(name+'.host.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    summary=json.loads((a.root/name/'summary.json').read_text())
    print('END',name,summary['status'],flush=True)
    if result.returncode:raise SystemExit(result.returncode)
