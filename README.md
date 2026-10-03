# rag-eval-lab

A small RAG lab for answering "does my retrieval actually work?" Chunk markdown docs, index them in
Postgres/pgvector (or in memory), compare **vector**, **keyword** and **hybrid (RRF)** retrieval, measure each with
hit@k and MRR, and generate answers with validated citations using a local Ollama model.

Based in part on [pgvector-python](https://github.com/pgvector/pgvector-python)'s hybrid-search and RAG examples (MIT). See [SOURCES.md](SOURCES.md).

## Why I built it
To learn the parts of RAG that tutorials skip: how chunking choices affect retrieval, why hybrid search exists, and how to
measure retrieval instead of eyeballing answers.

## Features
- Markdown-aware chunking: heading as metadata, word windows with overlap, frontmatter metadata
- Embedders: Ollama `nomic-embed-text` (with required task prefixes) or an offline deterministic hashing embedder for tests
- Stores: `pgvector` (HNSW cosine + GIN full-text) or in-memory (numpy + own BM25), same interface
- Three retrieval modes; hybrid uses Reciprocal Rank Fusion
- Metadata filtering (`--category billing`)
- Answers with `[n]` citations; citations pointing at non-existent sources are flagged
- Evaluation: hit@1/3/5 and MRR per mode, on a hand-written question set
- 13 pytest tests, including pgvector tests when a database is available

## Architecture
```
data/corpus/*.md -> chunking.py -> Chunk(id, source, heading, text, metadata)
                                      |  embedders.py (Ollama | hashing)
                                      v
                          stores.py: MemoryStore | PgVectorStore
                              |  vector_search     |  keyword_search
                              v                    v
                      retrieval.py:  vector | keyword | hybrid = rrf_fuse(both)
                              |
              +---------------+----------------+
              v                                v
        evaluate.py (hit@k, MRR)       generate.py (prompt + citation check)
```

## Tech stack
Python 3.10+, numpy, psycopg 3, pgvector, Postgres 16, Ollama (optional), pytest.

## How to run
```bash
pip install -e ".[dev]"
python -m pytest                               # no services needed (pgvector test skipped)
python -m rag_eval_lab eval                    # offline: in-memory store + hashing embedder
python -m rag_eval_lab search "how long do refunds take" --mode hybrid
```
With Postgres (`docker compose up -d`) and Ollama (`ollama pull nomic-embed-text llama3.2`):
```bash
cp .env.example .env && export $(grep -v '^#' .env | xargs)
python -m rag_eval_lab eval --store pgvector --embedder ollama
python -m rag_eval_lab ask "Can I return a gift card?" --store pgvector --embedder ollama --category billing
TEST_DATABASE_URL=$DATABASE_URL python -m pytest   # runs pgvector test too
```

## Example usage / output
Offline run (hashing embedder, which is **lexical, not semantic**: this only shows the pipeline works, it is not a model benchmark):
```
21 chunks, 22 questions, embedder=hashing, store=memory

mode       hit@1   hit@3   hit@5     mrr
vector      0.36    0.64    0.77    0.52
keyword     0.73    0.91    0.91    0.80
hybrid      0.64    0.77    0.86    0.73
```
I have not run the Ollama embedder path in my build environment (no Ollama available there), so I make no claims about its numbers. Run it yourself and compare.

## What I changed / added vs. the upstream examples
See [SOURCES.md](SOURCES.md): Python-side RRF, OR-based full-text query, chunking with metadata, filters, evaluation harness, citation validation, two interchangeable stores, tests, corpus.

## Key Technical Concepts
- **Chunking**: embeddings represent a whole text with one vector, so chunk size trades precision against context. Overlap avoids cutting an answer in half.
- **Embeddings & cosine distance**: pgvector's `<=>` is cosine distance; similarity = 1 - distance. Normalised vectors make dot product = cosine.
- **Task prefixes**: `nomic-embed-text` is trained with `search_document:` / `search_query:` prefixes.
- **Lexical vs semantic search**: keyword search wins on exact terms (names, numbers); vectors win on paraphrase. Hybrid covers both.
- **BM25 / ts_rank_cd**: term-frequency based ranking with length normalisation (BM25) and cover-density ranking (Postgres).
- **RRF**: `score = Σ 1/(k + rank)`. Uses ranks so incompatible score scales don't matter; k=60 is the common default and damps top-rank dominance.
- **HNSW**: approximate nearest-neighbour graph index, trading a little recall for speed.
- **Retrieval metrics**: hit@k = fraction of questions with a relevant chunk in top k; MRR = mean of 1/rank of the first relevant chunk.
- **Grounded generation**: instruct the model to use only the context, cite, and say "don't know"; then verify citations mechanically.

## Limitations
- Corpus is tiny (21 chunks) and fictional; 22 questions is too few for statistically meaningful comparisons.
- Relevance is judged at source + heading level, not by answer content.
- Offline hashing embedder is lexical; semantic behaviour requires Ollama.
- Citation check verifies the cited number exists, not that the claim is supported.
- No reranker, no streaming, no incremental indexing (the table is rebuilt on every index).

## Future improvements
Cross-encoder reranking, LLM-judged answer faithfulness, incremental indexing, larger real corpus, comparing chunk sizes systematically.
