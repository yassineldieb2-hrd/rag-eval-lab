# Attribution

## pgvector-python (MIT)
Repository: https://github.com/pgvector/pgvector-python - Copyright (c) 2021-2026 Andrew Kane.
The hybrid-search approach (Reciprocal Rank Fusion of a pgvector cosine search and a Postgres full-text search, k = 60) and the
local RAG flow (Ollama `nomic-embed-text`, pgvector, local generation) follow the upstream examples
`examples/hybrid_search/rrf.py` and `examples/rag/example.py`. The code in this repository is a separate implementation
with these differences: fusion is done in Python over ranked ID lists, keyword search ORs the query terms, chunking is
markdown-aware with metadata filters, and the project adds an evaluation harness, an in-memory store with BM25, citation validation, a CLI, tests and its own corpus.
The upstream MIT license text and copyright notice are kept in `LICENSE`.

## Hugging Face Cookbook, "RAG Evaluation" (Apache-2.0)
https://github.com/huggingface/cookbook - conceptual reference for evaluating retrieval separately from generation. No code or text is copied.

## External tools
pgvector (PostgreSQL extension by Andrew Kane), Ollama, and `nomic-embed-text` (Apache-2.0, Nomic AI) are used as external tools and are not redistributed.
The documents in `data/corpus/` and the questions in `data/eval/questions.jsonl` are original to this project.
