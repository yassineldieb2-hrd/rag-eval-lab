"""Answer generation with citations via a local Ollama model.

The prompt numbers each context chunk and asks the model to cite [n]. We then check
that every citation the model emitted points at a chunk we really supplied.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


def build_prompt(question: str, chunks) -> str:
    ctx = "\n\n".join(f"[{i}] ({c.source} > {c.heading})\n{c.text}" for i, c in enumerate(chunks, 1))
    return ("Answer the question using ONLY the context below. Cite sources as [n] after each "
            "claim. If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")


def valid_citations(answer: str, n_chunks: int) -> tuple[list[int], list[int]]:
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    return [c for c in cited if 1 <= c <= n_chunks], [c for c in cited if not 1 <= c <= n_chunks]


def ask_ollama(prompt: str, url: str, model: str, timeout: float = 120.0) -> str:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["response"].strip()
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Could not reach Ollama at {url} ({e}). Try `ollama pull {model}`.") from e
