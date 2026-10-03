"""Command line interface: index / search / eval / ask."""
from __future__ import annotations

import argparse
import sys

from .chunking import chunk_directory
from .config import Config
from .embedders import HashingEmbedder, OllamaEmbedder
from .evaluate import evaluate, format_table, load_eval
from .generate import ask_ollama, build_prompt, valid_citations
from .retrieval import MODES, Retriever
from .stores import MemoryStore, PgVectorStore

DEFAULT_DOCS = "data/corpus"


def build(args, cfg: Config):
    embedder = HashingEmbedder() if args.embedder == "hashing" else OllamaEmbedder(
        cfg.ollama_url, cfg.embed_model)
    if args.store == "pgvector":
        if not cfg.database_url:
            sys.exit("DATABASE_URL is not set (see .env.example)")
        store = PgVectorStore(cfg.database_url, embedder.dim)
    else:
        store = MemoryStore()
    chunks = chunk_directory(args.docs, max_words=args.chunk_words, overlap=args.overlap)
    store.add(chunks, embedder.embed_documents([f"{c.heading}\n{c.text}" for c in chunks]))
    return Retriever(store, embedder), store, chunks


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    p.add_argument("command", choices=["search", "eval", "ask"])
    p.add_argument("query", nargs="?", help="question (search/ask)")
    p.add_argument("--docs", default=DEFAULT_DOCS)
    p.add_argument("--eval-file", default="data/eval/questions.jsonl")
    p.add_argument("--store", choices=["memory", "pgvector"], default="memory")
    p.add_argument("--embedder", choices=["hashing", "ollama"], default="hashing")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--chunk-words", type=int, default=120)
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument("--category", help="metadata filter on frontmatter 'category'")
    args = p.parse_args(argv)
    cfg = Config.from_env()
    try:
        retriever, store, chunks = build(args, cfg)
        filters = {"category": args.category} if args.category else None
        if args.command == "eval":
            items = load_eval(args.eval_file)
            print(f"{len(chunks)} chunks, {len(items)} questions, embedder={args.embedder}, "
                  f"store={args.store}\n")
            print(format_table([evaluate(retriever, store, items, m) for m in MODES]))
            return 0
        if not args.query:
            p.error("query required")
        hits = retriever.search(args.query, args.mode, args.top_k, filters)
        found = [store.get(h.chunk_id) for h in hits]
        if args.command == "search":
            for h, c in zip(hits, found):
                print(f"{h.score:.4f}  {c.citation}\n        {c.text[:110]}...")
            return 0
        answer = ask_ollama(build_prompt(args.query, found), cfg.ollama_url, cfg.llm_model)
        ok, bad = valid_citations(answer, len(found))
        print(answer, "\n\nSources:")
        for i in ok:
            print(f"  [{i}] {found[i - 1].citation}")
        if bad:
            print(f"WARNING: model cited non-existent sources {bad}")
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
