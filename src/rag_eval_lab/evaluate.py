"""Retrieval evaluation: hit-rate@k (a.k.a. recall@k with one relevant doc) and MRR.

An eval item is {"question": ..., "source": <doc stem that contains the answer>,
"heading": optional section}. A hit is correct if the retrieved chunk comes from the
expected source (and heading, if given). Document-level matching keeps labelling cheap
but is lenient; see README limitations.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_eval(path: str | Path) -> list[dict]:
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for it in items:
        if "question" not in it or "source" not in it:
            raise ValueError(f"eval item needs 'question' and 'source': {it}")
    return items


def _is_relevant(chunk, item) -> bool:
    if chunk.source != item["source"]:
        return False
    return "heading" not in item or chunk.heading.startswith(item["heading"])


def evaluate(retriever, store, items: list[dict], mode: str, ks=(1, 3, 5)) -> dict:
    hits_at = {k: 0 for k in ks}
    rr_total, misses = 0.0, []
    for item in items:
        hits = retriever.search(item["question"], mode=mode, top_k=max(ks),
                                filters=item.get("filters"))
        rank = next((i for i, h in enumerate(hits, 1)
                     if _is_relevant(store.get(h.chunk_id), item)), None)
        if rank is None:
            misses.append(item["question"])
        else:
            rr_total += 1.0 / rank
            for k in ks:
                hits_at[k] += rank <= k
    n = len(items)
    return {"mode": mode, "n": n,
            **{f"hit@{k}": hits_at[k] / n for k in ks},
            "mrr": rr_total / n, "misses": misses}


def format_table(results: list[dict]) -> str:
    cols = [k for k in results[0] if k.startswith("hit@")] + ["mrr"]
    lines = ["mode      " + "  ".join(f"{c:>6}" for c in cols)]
    for r in results:
        lines.append(f"{r['mode']:<9} " + "  ".join(f"{r[c]:6.2f}" for c in cols))
    return "\n".join(lines)
