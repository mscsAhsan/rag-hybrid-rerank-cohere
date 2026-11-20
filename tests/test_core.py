"""Offline unit tests (no API key needed): run with  python -m pytest -q"""
from rag.chunker import chunk_text


def test_chunk_overlap_and_ids():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, "doc.txt", chunk_size=40, overlap=10)
    assert len(chunks) >= 3
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    # consecutive chunks share the overlap
    assert chunks[0]["text"].split()[-10:] == chunks[1]["text"].split()[:10]


def test_short_text_single_chunk():
    chunks = chunk_text("only a few words", "a.txt", chunk_size=50, overlap=5)
    assert len(chunks) == 1


def test_rrf_prefers_items_ranked_well_in_both_lists():
    from rag.hybrid import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    order = [i for i, _ in fused]
    assert order[0] == 1  # rank 1 + rank 2 beats rank 3 + rank 1
    assert set(order) == {1, 2, 3, 4}


def test_tokenize():
    from rag.hybrid import tokenize

    assert tokenize("BM25 is Great!") == ["bm25", "is", "great"]
