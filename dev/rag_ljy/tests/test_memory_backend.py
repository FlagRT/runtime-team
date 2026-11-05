"""Prevent retrieval-backend mislabelling in live memory acceptance."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/checks'))
from memory_backend import require_backend

@pytest.mark.parametrize('actual,expected',[('milvus','elasticsearch'),('elasticsearch','milvus'),(None,'elasticsearch')])
def test_wrong_or_missing_backend_is_rejected(actual,expected):
    with pytest.raises(RuntimeError,match='Expected retrieval backend'):
        require_backend({'retrieval_backend':actual},expected)

@pytest.mark.parametrize('backend',['milvus','elasticsearch'])
def test_backend_matches_actual_health(backend):
    assert require_backend({'retrieval_backend':backend},backend)==backend
