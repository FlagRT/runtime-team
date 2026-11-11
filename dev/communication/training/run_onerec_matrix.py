"""Alternate bounded bucket sizes in independent two-rank training launches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',required=True)
    p.add_argument('--model-root',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seq-len',type=int,default=64)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args()
    if a.repeats<3 or not 2<=a.seq_len<=128 or not 2<=a.steps<=50 or a.timeout<1:
        p.error('Require >=3 repeats, seq 2..128, steps 2..50 and positive timeout')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(backend='flagcx',seq_len=a.seq_len,steps=a.steps,repeats=a.repeats,runs=[])
    try:
        for repeat in range(1,a.repeats+1):
            for cap in ((64,128) if repeat%2 else (128,64)):
                name=f'{repeat:02d}-bucket{cap}'
                command=[sys.executable,str(Path(__file__).with_name('run_onerec_case.py')),
                         '--runtime-root',a.runtime_root,'--model-root',a.model_root,
                         '--collective','flagcx','--seq-len',str(a.seq_len),'--steps',str(a.steps),
                         '--bucket-mib',str(cap),'--timeout',str(a.timeout),'--output',str(a.output/name)]
                with (a.output/f'{name}-driver.log').open('x') as log:
                    code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
                manifest['runs'].append(dict(name=name,repeat=repeat,bucket_mib=cap,argv=command,exit_code=code))
                (a.output/'matrix.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print(f'{name}: exit={code}',flush=True)
                if code:
                    return code
    finally:
        (a.output/'matrix.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
