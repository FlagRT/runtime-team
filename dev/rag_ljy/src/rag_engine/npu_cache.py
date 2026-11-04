"""Use the public C++ cache API without holding Python's global interpreter lock."""
import ctypes
import hashlib
import json
import os
from pathlib import Path


def cache_reclaimer(module):
    path=os.getenv('RAG_NPU_CACHE_BRIDGE')
    if not path:
        return module.empty_cache, 'vendor_python_binding'
    import torch,torch_npu
    library=Path(path)
    metadata=json.loads((library.parent/'metadata.json').read_text())
    source=Path(__file__).parent/'native/npu_cache_bridge.cpp'
    if (metadata['torch'] != torch.__version__ or metadata['torch_npu'] != torch_npu.__version__
            or metadata['cxx11_abi'] != bool(torch._C._GLIBCXX_USE_CXX11_ABI)
            or metadata['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest()
            or metadata['library_sha256'] != hashlib.sha256(library.read_bytes()).hexdigest()):
        raise RuntimeError('NPU cache bridge does not match installed runtime/source; rebuild it')
    lib=ctypes.CDLL(str(library))  # CDLL releases the GIL; PyDLL would not.
    function=lib.rag_empty_npu_cache
    function.argtypes=[ctypes.c_char_p,ctypes.c_size_t]
    function.restype=ctypes.c_int
    def release():
        error=ctypes.create_string_buffer(4096)
        if function(error,len(error)):
            raise RuntimeError(error.value.decode(errors='replace'))
    return release, 'public_cpp_allocator_gil_released:'+metadata['source_sha256']
