# Interview notes - rag-eval-lab

Answers describe what this repository actually does.

## 10 likely questions
1. **What does the project do?** It indexes markdown docs, retrieves with vector, keyword or hybrid search, evaluates each mode with hit@k and MRR, and can generate cited answers via a local Ollama model.
2. **Why hybrid search?** Vector search handles paraphrase; keyword search handles exact terms. On my offline lexical embedder the keyword mode scored higher than hybrid on my small set, which is a reminder that hybrid is not automatically better; it must be measured. I haven't measured it with a real embedding model.
3. **How does RRF work?** `rrf_fuse` sums `1/(60 + rank)` per ranked list. It uses ranks only, so cosine similarity and BM25 scores (different scales) need no normalisation.
4. **How do you chunk?** `chunk_document` splits on markdown headings, keeps the heading as metadata, then cuts long sections into 120-word windows with 20-word overlap (configurable). Overlap reduces the chance of splitting an answer.
5. **What did you change versus the upstream pgvector example?** Python-side RRF, OR-ed full-text query (upstream's `plainto_tsquery` ANDs terms), chunk metadata and filters, evaluation, citation validation, tests (see SOURCES.md).
6. **How does pgvector search work here?** `embedding <=> query` is cosine distance with an HNSW index (`vector_cosine_ops`); similarity is reported as `1 - distance`. Metadata filtering uses `metadata @> filter::jsonb`.
7. **How do you evaluate retrieval?** 22 hand-written questions, each labelled with the source file and section containing the answer. A hit means a retrieved chunk from that source/heading is in the top k; MRR averages 1/rank of the first hit.
8. **What are the weaknesses of that evaluation?** Tiny, fictional corpus; labels are section-level, not answer-level; the questions were written by me knowing the corpus, which can bias toward lexical overlap.
9. **How do you handle hallucinated citations?** The prompt numbers the context chunks; afterwards `valid_citations` flags any `[n]` outside the supplied range. It doesn't verify that the claim is supported.
10. **How do you test it without services?** A deterministic `HashingEmbedder` and `MemoryStore`; the pgvector test is skipped unless `TEST_DATABASE_URL` is set, and I ran it against a real local Postgres 16 + pgvector 0.8.7 and checked it ranks like the in-memory store.

## 5 areas to study further
- How transformer embedding models are trained (contrastive learning) and why prefixes matter
- BM25 math and Postgres `tsvector` configuration/stemming
- HNSW parameters (`m`, `ef_construction`, `ef_search`) and recall/latency trade-offs
- Reranking with cross-encoders and when it pays off
- Evaluating generation: faithfulness, answer relevance, LLM-as-judge pitfalls

## Design decisions
- **Same store interface, two backends:** lets tests and CI run without a database while still exercising real pgvector.
- **Fusion outside SQL:** easier to test and swap; costs two round trips instead of one.
- **Offline hashing embedder:** deterministic tests; explicitly not used for quality claims.
- **Section-level relevance labels:** cheap to write; deliberately lenient and documented as such.
- **Rebuild table on index:** simple and correct for a lab; not suitable for production updates.
