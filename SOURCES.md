# Sources and attribution

This project is a learning project. It was **inspired by and partly adapted from** the
following open-source work. Both are permissively licensed; their licenses are respected here.

## 1. pgvector-python - Andrew Kane (MIT)
Repository: https://github.com/pgvector/pgvector-python
Files studied: `examples/hybrid_search/rrf.py`, `examples/rag/example.py`

What was taken as a starting idea:
- Reciprocal Rank Fusion of a pgvector cosine search and a Postgres full-text search (k = 60).
- The basic flow of a local RAG example: embed with Ollama `nomic-embed-text`, store in pgvector, retrieve, generate with a local model.

What is different (my own code):
- Fusion is done in Python over ranked ID lists so it works with any store and can be unit-tested (upstream does it in one SQL statement).
- Keyword search ORs the query terms. Upstream's `plainto_tsquery` ANDs every word, which returns almost nothing for natural-language questions.
- Markdown-aware chunking with heading metadata, overlap, and frontmatter-based metadata filters (upstream splits on `## ` and lists "improve chunking" as a TODO).
- `nomic-embed-text` task prefixes (`search_document:` / `search_query:`).
- A retrieval evaluation harness (hit@k, MRR), a pure-Python/numpy `MemoryStore` with own BM25, citation validation, CLI, tests, and an original support-docs corpus + question set.

License of the original: MIT, Copyright (c) 2021-2026 Andrew Kane. Copyright notice kept in `LICENSE`.

## 2. Hugging Face Cookbook - "RAG Evaluation" notebook (Apache-2.0)
Repository: https://github.com/huggingface/cookbook (notebook `rag_evaluation.ipynb`, author Aymeric Roucher)

Used only as **conceptual inspiration** for evaluating retrieval separately from generation. No code or text from the notebook is copied; this repo uses a hand-labelled question set rather than LLM-generated questions.

## Other
- The corpus in `data/corpus/` (a fictional company's support articles) and `data/eval/questions.jsonl` were written by me for this project.
- pgvector (the Postgres extension) by Andrew Kane, PostgreSQL-style license; Ollama, `nomic-embed-text` (Apache-2.0, Nomic AI) are used as external tools.
