from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from elasticsearch import BadRequestError
from elastic_transport import ApiResponseMeta, NodeConfig
from rag_engine.stores.elasticsearch import ElasticsearchStore


def error(reason):
    return BadRequestError('search failed',meta=ApiResponseMeta(400,'1.1',{},0,NodeConfig('http','localhost',9200)),body={'error':{'reason':reason}})


def store():
    obj=ElasticsearchStore.__new__(ElasticsearchStore)
    obj.settings=SimpleNamespace(elasticsearch_index_name='test')
    obj.client=Mock()
    return obj


def test_normal_queries_preserve_original_search():
    obj=store();obj.client.search.return_value={'hits':{'hits':[]}}
    assert obj.bm25_search('short query')==[]
    obj.client.indices.analyze.assert_not_called()
    assert obj.client.search.call_args.kwargs['query']['multi_match']['query']=='short query'


def test_clause_overflow_has_bounded_analyzed_fallback():
    obj=store();obj.client.search.side_effect=[error('too_many_nested_clauses'),{'hits':{'hits':[]}}]
    obj.client.indices.analyze.return_value={'tokens':[{'token':str(i)} for i in range(2000)]+[{'token':'0'}]}
    query='unchanged full model input'
    assert obj.bm25_search(query,size=100)==[]
    assert obj.client.search.call_count==2
    for call in obj.client.indices.analyze.call_args_list:assert call.kwargs['text']==query
    bounded=obj.client.search.call_args.kwargs
    assert bounded['size']==100
    fields=bounded['query']['dis_max']['queries']
    assert sum(len(f['bool']['should']) for f in fields)==512
    assert [f['bool']['boost'] for f in fields]==[2.0,1.0]
    assert fields[0]['bool']['should'][0]=={'term':{'title':'0'}}


def test_other_bad_requests_are_not_retried():
    obj=store();obj.client.search.side_effect=error('unknown field')
    with pytest.raises(BadRequestError):obj.bm25_search('query')
    obj.client.indices.analyze.assert_not_called()
    assert obj.client.search.call_count==1


def test_empty_analysis_matches_nothing():
    obj=store();obj.client.indices.analyze.return_value={'tokens':[]}
    assert obj._bounded_bm25_query('...')=={'match_none':{}}
