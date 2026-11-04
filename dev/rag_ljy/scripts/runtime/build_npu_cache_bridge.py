#!/usr/bin/env python3
"""CPU build of a GIL-releasing wrapper around the installed public NPU allocator."""
from pathlib import Path
import hashlib,json,os,platform,shutil,subprocess,sysconfig
import torch,torch_npu
from torch.utils.cpp_extension import include_paths
root=Path(__file__).resolve().parents[2]
source=root/'src/rag_engine/native/npu_cache_bridge.cpp'
metadata={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'torch':torch.__version__,'torch_npu':torch_npu.__version__,
          'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),'machine':platform.machine()}
key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:20]
out=root/'experiment_outputs/runtime_extensions'/key
out.mkdir(parents=True,exist_ok=True)
npu=Path(torch_npu.__file__).parent
torch_lib=Path(torch.__file__).parent/'lib'
ascend=Path(os.getenv('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest'))
includes=include_paths()+[str(npu/'include'),str(ascend/'include'),str(ascend/'aarch64-linux/include'),sysconfig.get_path('include')]
library=out/'libnpu_cache_bridge.so'
built=out/f'libnpu_cache_bridge.{os.getpid()}.tmp.so'
cmd=['c++','-shared','-fPIC','-O1','-g0','-std=c++17',f'-D_GLIBCXX_USE_CXX11_ABI={int(metadata["cxx11_abi"])}',str(source),'-o',str(built)]
cmd += ['-I'+p for p in includes]
cmd += ['-L'+str(p) for p in (npu/'lib',torch_lib)]
cmd += ['-Wl,-rpath,'+str(p) for p in (npu/'lib',torch_lib)]
cmd += ['-ltorch_npu','-ltorch_cpu','-lc10']
with (out/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
# Never truncate a shared object that may already be mapped by the live API.
os.replace(built,library)
metadata['command']=cmd
metadata['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
shutil.copyfile(source,out/source.name)
print('RAG_NPU_CACHE_BRIDGE='+str(library),flush=True)
