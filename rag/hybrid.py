"""Approach 2 - Hybrid RAG: BM25 + dense vectors, Reciprocal Rank Fusion, Cohere Rerank."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .cohere_utils import RERANK_MODEL, _with_retry, embed_texts


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(rankings, k: int = 60):
    """rankings: list of ranked lists of chunk indices (best first). Returns [(idx, score)] best first."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRAG:
    def __init__(self, client):
        self.client = client
        self.chunks = []
        self.matrix = None
        self.bm25 = None
        self.dim = 0

    def build_index(self, chunks):
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = chunks
        # sparse index
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        # dense index (normalised so dot product == cosine)
        mat = np.array(embed_texts(self.client, [c["text"] for c in chunks], "search_document"), dtype="float32")
        self.matrix = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        self.dim = self.matrix.shape[1]

    # ---- individual retrievers -------------------------------------------------
    def bm25_search(self, query: str, k: int):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def dense_search(self, query: str, k: int, query_vec=None):
        if query_vec is None:
            query_vec = np.array(embed_texts(self.client, [query], "search_query")[0], dtype="float32")
        query_vec = query_vec / np.linalg.norm(query_vec)
        scores = self.matrix @ query_vec
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]

    # ---- reranker --------------------------------------------------------------
    def rerank(self, query: str, candidate_ids, top_n: int):
        docs = [self.chunks[i]["text"] for i in candidate_ids]
        resp = _with_retry(
            lambda: self.client.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=min(top_n, len(docs)))
        )
        return [(candidate_ids[r.index], float(r.relevance_score)) for r in resp.results]

    # ---- full pipeline ---------------------------------------------------------
    def _pack(self, pairs):
        return [{**self.chunks[i], "score": s, "rank": r} for r, (i, s) in enumerate(pairs, 1)]

    def search(self, query: str, candidate_k: int = 10, top_n: int = 4, use_rerank: bool = True):
        bm25 = self.bm25_search(query, candidate_k)
        dense = self.dense_search(query, candidate_k)
        fused = reciprocal_rank_fusion([[i for i, _ in bm25], [i for i, _ in dense]])[:candidate_k]
        if use_rerank:
            final = self.rerank(query, [i for i, _ in fused], top_n)
        else:
            final = fused[:top_n]
        return {
            "bm25": self._pack(bm25),
            "dense": self._pack(dense),
            "fused": self._pack(fused),
            "final": self._pack(final),
        }
