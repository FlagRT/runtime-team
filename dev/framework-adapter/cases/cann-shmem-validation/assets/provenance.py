"""Collect allowlisted runtime identity and package delta; no credentials."""
import importlib.metadata as m
import importlib.util
import json
from pathlib import Path

out=Path('/results')
before={x['name']:x['version'] for x in json.loads((out/'before-packages.log').read_text())}
after={x['name']:x['version'] for x in json.loads((out/'after-packages.log').read_text())}
delta={name:{'before':before.get(name),'after':after.get(name)} for name in sorted(set(before)|set(after)) if before.get(name)!=after.get(name)}
versions={}
for name in ['torch','torch-npu','triton','flagtree','flag-gems','cann-shmem']:
    try:
        versions[name]=m.version(name)
    except m.PackageNotFoundError:
        versions[name]=None  # FlagTree can provide triton without a triton distribution.
import triton
versions['triton_module']=triton.__version__
locations={name:importlib.util.find_spec(name).origin for name in ['shmem','torch','triton','flag_gems']}
report={'versions':versions,'module_locations':locations,'package_changes':delta}
(out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert delta=={'cann-shmem':{'before':None,'after':'1.6.0'}}, 'Unexpected package changes'
