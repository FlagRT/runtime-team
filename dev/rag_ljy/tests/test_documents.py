from rag_engine.documents import chunk_documents, split_text


def test_split_text_preserves_overlap() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = split_text(text, max_chars=20, overlap_chars=5)
    assert len(chunks) > 1
    assert chunks[0]
    assert chunks[-1].endswith("theta")


def test_chunk_ids_are_stable() -> None:
    documents = [
        {
            "document_id": "doc-1",
            "title": "Example",
            "text": "A stable piece of text.",
            "source_uri": "test://doc-1",
        }
    ]
    first = chunk_documents(documents)
    second = chunk_documents(documents)
    assert first[0]["chunk_id"] == second[0]["chunk_id"]
