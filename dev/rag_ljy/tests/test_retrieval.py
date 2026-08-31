from rag_engine.retrieval import hybrid_search, reciprocal_rank_fusion


def hit(identifier: str, score: float) -> dict:
    return {
        "_id": identifier,
        "_score": score,
        "_source": {"text": identifier},
    }


def test_rrf_rewards_documents_returned_by_both_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": [hit("shared", 10.0), hit("lexical", 9.0)],
            "dense": [hit("semantic", 0.9), hit("shared", 0.8)],
        },
        rank_constant=60,
        limit=3,
    )
    assert fused[0]["_id"] == "shared"
    assert fused[0]["_retrieval_ranks"] == {"bm25": 1, "dense": 2}


def test_rrf_limit_is_applied() -> None:
    fused = reciprocal_rank_fusion(
        {"bm25": [hit("a", 3.0), hit("b", 2.0), hit("c", 1.0)]},
        limit=2,
    )
    assert [item["_id"] for item in fused] == ["a", "b"]


def test_hybrid_search_uses_one_store_for_sparse_and_dense() -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.calls = []

        def bm25_search(self, query: str, size: int):
            self.calls.append(("bm25", query, size))
            return [hit("sparse", 3.0)]

        def dense_search(self, query_vector, size: int, num_candidates: int):
            self.calls.append(("dense", query_vector, size, num_candidates))
            return [hit("dense", 0.9)]

    store = FakeStore()
    results = hybrid_search(
        store,
        "example query",
        [0.1, 0.2],
        retriever_top_k=10,
        coarse_top_k=2,
        num_candidates=25,
    )

    assert store.calls == [
        ("bm25", "example query", 10),
        ("dense", [0.1, 0.2], 10, 25),
    ]
    assert {result["_id"] for result in results} == {"sparse", "dense"}
