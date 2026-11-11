"""Verify only this case's containers have stopped; retain an explicit receipt."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
assert a.root.resolve()==Path('/bmcp_lvm_fs/cgu135/framework-qwen-embedding-20261111')
names=['qemb-'+n for n in ('qual-01','native-01','repeat-01','candidate-01','candidate-02','off-01','unknown-01','on-01')]
states={}
for name in names:
    state=json.loads(subprocess.check_output(['docker','inspect','cgu135-ppu-'+name,'--format','{{json .State}}'],text=True))
    assert not state['Running'] and state['Pid']==0 and not state['OOMKilled']
    states[name]=state
(a.root/'cleanup.host.log').write_text(json.dumps(states,indent=2)+'\n')
(a.root/'resource-final.host.log').write_text(subprocess.check_output(['ppu-smi'],text=True,timeout=20))
print('Verified stopped containers:',len(states))
