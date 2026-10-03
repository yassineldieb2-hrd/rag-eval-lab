"""Retrieval modes: vector, keyword, hybrid (Reciprocal Rank Fusion)."""
from __future__ import annotations

from dataclasses import dataclass

MODES = ("vector", "keyword", "hybrid")


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1.
    Uses ranks, not raw scores, so cosine similarity and BM25 need no normalising."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


@dataclass
class Hit:
    chunk_id: str
    score: float


class Retriever:
    def __init__(self, store, embedder, candidates: int = 20, rrf_k: int = 60):
        self.store, self.embedder = store, embedder
        self.candidates, self.rrf_k = candidates, rrf_k

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5, filters=None) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = self.candidates
        vec = kw = []
        if mode in ("vector", "hybrid"):
            vec = self.store.vector_search(self.embedder.embed_query(query), n, filters)
        if mode in ("keyword", "hybrid"):
            kw = self.store.keyword_search(query, n, filters)
        if mode == "vector":
            ranked = vec
        elif mode == "keyword":
            ranked = kw
        else:
            ranked = rrf_fuse([[c for c, _ in vec], [c for c, _ in kw]], self.rrf_k)
        return [Hit(c, s) for c, s in ranked[:top_k]]
