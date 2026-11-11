import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from rag_engine.npu_cache import cache_reclaimer


def test_default_cache_release_keeps_vendor_implementation(monkeypatch):
    monkeypatch.delenv('RAG_NPU_CACHE_BRIDGE',raising=False)
    module=SimpleNamespace(empty_cache=lambda:None)
    function,kind=cache_reclaimer(module)
    assert function is module.empty_cache and kind=='vendor_python_binding'


def test_mismatched_native_bridge_is_never_loaded(monkeypatch,tmp_path):
    library=tmp_path/'bridge.so';library.write_bytes(b'not a native library')
    (tmp_path/'metadata.json').write_text(json.dumps({'torch':'wrong','torch_npu':'wrong','cxx11_abi':False}))
    monkeypatch.setenv('RAG_NPU_CACHE_BRIDGE',str(library))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(__version__='current',_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)))
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace(__version__='current'))
    with pytest.raises(RuntimeError,match='does not match'):
        cache_reclaimer(SimpleNamespace(empty_cache=lambda:None))
