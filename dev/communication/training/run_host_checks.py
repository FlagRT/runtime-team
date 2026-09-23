"""Bounded host-only DDP comparison; does not establish NPU acceptance."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    runs={}
    for mode in ['observed','native']:
        out=args.output/mode
        # Host-only test: avoid hostname/reverse-DNS dependent rendezvous.
        # Another process can race for this port; any bind failure fails the run.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        command=[sys.executable,'-m','torch.distributed.run','--rdzv-backend=static',
                 '--master-addr=127.0.0.1',f'--master-port={port}','--nproc_per_node=2',
                 str(Path(__file__).with_name('deepfm_train.py')),'--backend','cpu',
                 '--comm-mode',mode,'--model-root',str(args.model_root.resolve()),
                 '--output',str(out),'--steps','50']
        with (args.output/f'{mode}.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise RuntimeError(f'{mode}: launcher timeout')
        if code!=0:raise RuntimeError(f'{mode}: launcher exited {code}')
        records=[json.loads((out/f'rank{r}.json').read_text()) for r in range(2)]
        for rank,record in enumerate(records):
            assert record['rank']==rank and record['world_size']==2
            assert record['backend']=='gloo' and record['evidence_scope']=='CPU/gloo host-only'
            assert record['verdict']=='PASS' and record['group_state']=='CLOSED'
            assert len(record['oracle']['gradient_checks'])==17
            assert all(c['pass'] for c in record['oracle']['gradient_checks'])
            assert record['oracle']['update_pass'] and record['training']['rank_parameter_max_abs']==0
            assert len(record['training']['losses'])==50
        a,b=[r['training']['sample_id_ranges'] for r in records]
        for i,((a0,a1),(b0,b1)) in enumerate(zip(a,b)):
            assert a0==(i+5)*64 and a1==b0 and b1==a0+64
        runs[mode]=records
    differences=[]
    for rank in range(2):
        a=runs['observed'][rank];b=runs['native'][rank]
        assert a['model_sha256']==b['model_sha256']
        delta=max(abs(x-y) for x,y in zip(a['training']['losses'],b['training']['losses']))
        assert delta<1e-6
        differences.append(delta)
    summary={'evidence_scope':'CPU/gloo host-only; NPU NOT RUN','verdict':'PASS',
             'launcher_exit_codes':[0,0],'world_size':2,'model_parameters':421452,
             'unique_gradient_tensors_checked_per_rank':17,'warmup_steps':5,'measured_steps':50,
             'native_vs_observed_loss_curve_max_abs_by_rank':differences,
             'observed_gradient_communication_by_rank':[r['training']['gradient_communication'] for r in runs['observed']],
             'oracle_update_max_abs_by_rank':[r['oracle']['update_max_abs'] for r in runs['observed']],
             'final_rank_parameter_max_abs':0.0}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
