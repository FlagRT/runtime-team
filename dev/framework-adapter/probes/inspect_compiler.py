"""No-card package/source audit. Never initializes ACL or launches a kernel."""
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import traceback


def main():
    result={'scope':'no-card imports and compiler source; not a compile test','distributions':{},'modules':{},'sources':[]}
    for name in ('torch','torch-npu','flag-gems','flagtree','triton','triton-ascend','shmem'):
        try: result['distributions'][name]=metadata.version(name)
        except metadata.PackageNotFoundError: result['distributions'][name]=None
    for name in ('flag_gems','triton','shmem'):
        spec=importlib.util.find_spec(name)
        result['modules'][name]={'file':spec.origin if spec else None}
    try:
        triton=importlib.import_module('triton')
        result['modules']['triton']['version']=triton.__version__
        root=Path(triton.__file__).parent
        for file in sorted(root.rglob('*.py')):
            text=file.read_text(errors='replace')
            lines=[{'line':n,'text':line.strip()} for n,line in enumerate(text.splitlines(),1)
                   if ('import shmem' in line or 'from shmem' in line or
                       'import triton.experimental.tle' in line or
                       'from triton.experimental.tle' in line)]
            if lines:
                result['sources'].append({'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'matches':lines})
        try:
            importlib.import_module('triton.experimental.tle')
            result['tle_import']='PASS'
        except Exception:
            result['tle_import']='FAIL'; result['tle_error']=traceback.format_exc()
    except Exception:
        result['triton_error']=traceback.format_exc()
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
