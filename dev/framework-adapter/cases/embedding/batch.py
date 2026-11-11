"""Sequential bounded model checks, leave failures intact and native controls runnable."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--card',type=int,required=True)
a=p.parse_args()
for mode,label in [('native','native'),('native_repeat','repeat'),('candidate','candidate'),('off','off'),('unknown','unknown')]:
    name='embedding-'+label+'-01'
    command=[sys.executable,'-B',str(a.input/'run.py'),'--root',str(a.root),
        '--run-id',name,'--card',str(a.card),'--stage','model','--mode',mode]
    if mode!='native':command+=['--reference',str(a.root/'embedding-native-01')]
    with (a.root/(name+'.host.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    summary=json.loads((a.root/name/'summary.json').read_text())
    print(name,summary['status'],flush=True)
    if result.returncode and mode!='candidate':raise SystemExit(result.returncode)
    # A numeric candidate failure still permits off/unknown controls. Infrastructure
    # errors and uncertain cleanup do not authorize further launches.
    if result.returncode:
        report=json.loads((a.root/name/'result.json').read_text())
        if report['status']!='fail' or summary.get('stop',{}).get('Running',True):
            raise SystemExit(result.returncode)
