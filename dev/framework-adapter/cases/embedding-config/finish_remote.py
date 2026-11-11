"""Read-only checks of exact owned containers; archive this case's tensor evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();out=root/'final-audit';out.mkdir(exist_ok=False)
states={}
for label in ('native','repeat','candidate','off','unknown'):
    name='cgu135-ppu-unified-'+label+'-01'
    state=json.loads(subprocess.check_output(['docker','inspect',name,'--format','{{json .State}}'],text=True))
    assert not state['Running'] and state['Pid']==0 and not state['OOMKilled'],name
    states[name]=state
(out/'containers.json').write_text(json.dumps(states,indent=2)+'\n')
(out/'resource-final.log').write_text(subprocess.check_output(['ppu-smi'],text=True,timeout=20))
native=root/'unified-native-01/tensors'
files=list(native.glob('*.pt'));assert len(files)==294
archive=out/'native-reference.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(native,arcname='tensors')
h=hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
(out/'native-reference.sha256').write_text(h.hexdigest()+'  native-reference.tar.gz\n')
print('All five owned containers stopped; 294 reference tensors archived',h.hexdigest())
