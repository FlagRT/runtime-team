"""Collect only own stopped diagnostic outputs, excluding installed caches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
a=p.parse_args();root=a.run.resolve()
assert root.parent==Path('/bmcp_lvm_fs/cgu135/framework-embedding-diagnosis-20261111')
assert root.name in ('embedding-diagnosis-01','embedding-direct-01','embedding-direct-02')
state=json.loads(subprocess.check_output(['docker','inspect','cgu135-ppu-'+root.name,
                                         '--format','{{json .State}}'],text=True))
assert not state['Running'] and state['Pid']==0 and not state['OOMKilled']
collection=root.parent/(root.name+'-collection');collection.mkdir()
(collection/'container.json').write_text(json.dumps(state,indent=2)+'\n')
(collection/'resource-final.log').write_text(subprocess.check_output(['ppu-smi'],text=True,timeout=20))
files=[f for f in root.rglob('*') if f.is_file()
       and f.relative_to(root).parts[0] not in ('packages','triton-cache','gems-cache','__pycache__')]
manifest={str(f.relative_to(root)):dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),size=f.stat().st_size)
          for f in sorted(files)}
(collection/'files.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=collection/'evidence.tar.gz'
with tarfile.open(archive,'x:gz') as t:
    for f in sorted(files):t.add(f,arcname=str(f.relative_to(root)),recursive=False)
receipt={'files':len(files),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
         'archive_bytes':archive.stat().st_size}
(collection/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
