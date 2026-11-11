"""Read-only final checks of this case's exact owned containers; writes receipts."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
names=['embedding-qual-01','embedding-routes-01','embedding-routes-02','embedding-native-01',
       'embedding-repeat-01','embedding-candidate-01','embedding-off-01','embedding-unknown-01','embedding-on-01']
checks={}
for name in names:
    raw=json.loads(subprocess.check_output(['docker','inspect','cgu135-ppu-'+name],text=True))[0]
    state=raw['State']
    assert not state['Running'] and state['Pid']==0 and not state['OOMKilled'],name
    checks[name]=dict(name=raw['Name'],image=raw['Image'],state=state)
with (a.root/'cleanup.json').open('x') as f:json.dump(checks,f,indent=2)
with (a.root/'resource-final.log').open('x') as f:
    subprocess.run(['ppu-smi'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=20)
print('All',len(checks),'owned containers stopped')
