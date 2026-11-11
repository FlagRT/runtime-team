"""Serial DeepFM controls after a new native baseline. No approvals or retries."""
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
for mode,label in [('native_repeat','repeat'),('candidate','candidate'),('off','off'),('unknown','unknown')]:
    name='unified-'+label+'-01'
    command=[sys.executable,'-B',str(a.input/'run.py'),'--root',str(a.root),
        '--run-id',name,'--card',str(a.card),'--stage','model','--mode',mode,
        '--reference',str(a.root/'unified-native-01')]
    with (a.root/(name+'.host.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    summary=json.loads((a.root/name/'summary.json').read_text())
    print(name,summary['status'],flush=True)
    if result.returncode:raise SystemExit(result.returncode)
