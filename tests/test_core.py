import os

import numpy as np
import pytest

from rag_eval_lab.chunking import chunk_document, chunk_directory, parse_frontmatter
from rag_eval_lab.embedders import HashingEmbedder
from rag_eval_lab.evaluate import evaluate, load_eval
from rag_eval_lab.generate import valid_citations
from rag_eval_lab.retrieval import Retriever, rrf_fuse
from rag_eval_lab.stores import MemoryStore, PgVectorStore

DOC = "---\ncategory: x\n---\n# T\n\nintro text\n\n## A\n" + "word " * 50 + "\n## B\nshort"


def test_frontmatter():
    meta, body = parse_frontmatter(DOC)
    assert meta == {"category": "x"} and body.startswith("# T")


def test_chunk_headings_and_metadata():
    chunks = chunk_document("d", DOC, max_words=100, overlap=10)
    assert [c.heading for c in chunks] == ["T", "A", "B"]
    assert all(c.metadata == {"category": "x"} for c in chunks)


def test_long_section_split_with_overlap():
    chunks = chunk_document("d", DOC, max_words=20, overlap=5)
    parts = [c for c in chunks if c.heading.startswith("A")]
    assert len(parts) >= 3 and parts[1].heading == "A (part 2)"
    assert all(len(c.text.split()) <= 20 for c in chunks)


def test_bad_window_args():
    with pytest.raises(ValueError):
        chunk_document("d", DOC, max_words=10, overlap=10)


def test_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        chunk_directory(tmp_path)


def test_rrf_rewards_agreement():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60))
    assert fused["a"] > fused["b"] and fused["a"] > fused["d"]
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)


def test_hashing_embedder_deterministic_unit_norm():
    e = HashingEmbedder()
    a, b = e.embed_query("refund policy"), e.embed_query("refund policy")
    assert np.allclose(a, b) and np.linalg.norm(a) == pytest.approx(1.0)


def build_memory():
    chunks = chunk_directory("data/corpus")
    e = HashingEmbedder()
    s = MemoryStore()
    s.add(chunks, e.embed_documents([c.text for c in chunks]))
    return s, Retriever(s, e)


def test_filter_restricts_results():
    s, r = build_memory()
    hits = r.search("how do I pay", mode="hybrid", top_k=10, filters={"category": "billing"})
    assert hits and all(s.get(h.chunk_id).metadata["category"] == "billing" for h in hits)


def test_keyword_search_finds_exact_term():
    s, r = build_memory()
    top = r.search("Norway customs duties", mode="keyword", top_k=1)[0]
    assert s.get(top.chunk_id).source == "shipping"


def test_invalid_mode():
    _, r = build_memory()
    with pytest.raises(ValueError):
        r.search("x", mode="nope")


def test_eval_metrics_and_file_validation(tmp_path):
    s, r = build_memory()
    items = load_eval("data/eval/questions.jsonl")
    res = evaluate(r, s, items, "keyword")
    assert 0 <= res["mrr"] <= 1 and res["hit@1"] <= res["hit@3"] <= res["hit@5"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "q"}\n')
    with pytest.raises(ValueError):
        load_eval(bad)


def test_valid_citations():
    assert valid_citations("yes [1] and [3] and [9]", 3) == ([1, 3], [9])


DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not DB, reason="set TEST_DATABASE_URL to run pgvector tests")
def test_pgvector_matches_memory_store_ranking():
    chunks = chunk_directory("data/corpus")
    e = HashingEmbedder()
    emb = e.embed_documents([c.text for c in chunks])
    mem, pg = MemoryStore(), PgVectorStore(DB, e.dim, table="rag_chunks_test")
    mem.add(chunks, emb)
    pg.add(chunks, emb)
    q = e.embed_query("how long until my refund arrives")
    assert [i for i, _ in pg.vector_search(q, 5)] == [i for i, _ in mem.vector_search(q, 5)]
    assert pg.keyword_search("refund card days", 3)[0][0].startswith("refunds")
    f = pg.vector_search(q, 20, {"category": "orders"})
    assert f and all(pg.get(i).metadata["category"] == "orders" for i, _ in f)
