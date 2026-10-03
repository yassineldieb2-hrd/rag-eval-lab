"""Two interchangeable stores with the same interface.

`MemoryStore`  - numpy cosine + own BM25; no services needed.
`PgVectorStore` - Postgres: pgvector HNSW (cosine) + full-text search (GIN).

Both return ranked lists of (chunk_id, score); fusion happens in retrieval.py.
Filters are exact-match on chunk metadata, e.g. {"category": "billing"}.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np

from .chunking import Chunk
from .embedders import tokenize


def _matches(chunk: Chunk, filters: dict | None) -> bool:
    return all(chunk.metadata.get(k) == v for k, v in (filters or {}).items())


class MemoryStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None
        self._tf: list[Counter] = []
        self._df: Counter = Counter()

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.chunks = list(chunks)
        self.matrix = np.asarray(embeddings, dtype=np.float32)
        self._tf = [Counter(tokenize(c.text + " " + c.heading)) for c in chunks]
        self._df = Counter(t for tf in self._tf for t in tf)

    def get(self, chunk_id: str) -> Chunk:
        return next(c for c in self.chunks if c.id == chunk_id)

    def vector_search(self, q: np.ndarray, k: int, filters=None) -> list[tuple[str, float]]:
        sims = self.matrix @ q  # vectors are unit length -> dot == cosine
        order = np.argsort(-sims)
        return [(self.chunks[i].id, float(sims[i])) for i in order
                if _matches(self.chunks[i], filters)][:k]

    def keyword_search(self, query: str, k: int, filters=None, k1=1.5, b=0.75):
        n = len(self.chunks)
        avg = sum(sum(tf.values()) for tf in self._tf) / max(n, 1)
        terms = set(tokenize(query))
        scored = []
        for chunk, tf in zip(self.chunks, self._tf):
            if not _matches(chunk, filters):
                continue
            dl, s = sum(tf.values()), 0.0
            for t in terms & tf.keys():
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avg))
            if s > 0:
                scored.append((chunk.id, s))
        return sorted(scored, key=lambda x: -x[1])[:k]


class PgVectorStore:
    def __init__(self, database_url: str, dim: int, table: str = "rag_chunks"):
        import psycopg
        from pgvector.psycopg import register_vector

        if not table.isidentifier():
            raise ValueError("table must be a plain identifier")
        self.table, self.dim = table, dim
        self.conn = psycopg.connect(database_url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def reset(self) -> None:
        t = self.table
        self.conn.execute(f"DROP TABLE IF EXISTS {t}")
        self.conn.execute(
            f"CREATE TABLE {t} (id text PRIMARY KEY, source text, heading text, content text, "
            f"metadata jsonb, embedding vector({self.dim}))")
        self.conn.execute(
            f"CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops)")
        self.conn.execute(
            f"CREATE INDEX ON {t} USING gin (to_tsvector('english', heading || ' ' || content))")

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.reset()
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} VALUES (%s,%s,%s,%s,%s,%s)",
                [(c.id, c.source, c.heading, c.text, json.dumps(c.metadata), e)
                 for c, e in zip(chunks, embeddings)])

    def get(self, chunk_id: str) -> Chunk:
        row = self.conn.execute(
            f"SELECT id, source, heading, content, metadata FROM {self.table} WHERE id=%s",
            (chunk_id,)).fetchone()
        return Chunk(row[0], row[1], row[2], row[3], row[4])

    def vector_search(self, q: np.ndarray, k: int, filters=None):
        rows = self.conn.execute(
            f"SELECT id, 1 - (embedding <=> %s) FROM {self.table} WHERE metadata @> %s::jsonb "
            f"ORDER BY embedding <=> %s LIMIT %s",
            (q, json.dumps(filters or {}), q, k)).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def keyword_search(self, query: str, k: int, filters=None):
        # plainto_tsquery ANDs every word, so a natural-language question matches
        # almost nothing. OR the terms together and let ts_rank_cd order them.
        terms = sorted(set(tokenize(query)))
        if not terms:
            return []
        tsq = " | ".join(terms)
        rows = self.conn.execute(
            f"SELECT id, ts_rank_cd(to_tsvector('english', heading || ' ' || content), q) AS r "
            f"FROM {self.table}, to_tsquery('english', %s) q "
            f"WHERE to_tsvector('english', heading || ' ' || content) @@ q "
            f"AND metadata @> %s::jsonb ORDER BY r DESC LIMIT %s",
            (tsq, json.dumps(filters or {}), k)).fetchall()
        return [(r[0], float(r[1])) for r in rows]
