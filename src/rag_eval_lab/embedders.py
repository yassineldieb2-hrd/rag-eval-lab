"""Embedders. `OllamaEmbedder` is the real one; `HashingEmbedder` is deterministic
and offline so tests and CI never need a model server.

nomic-embed-text expects task prefixes: 'search_document: ' when indexing and
'search_query: ' when querying. Forgetting them silently hurts quality.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature-hashing of unigrams+bigrams. Lexical, NOT semantic: useful
    only to exercise the pipeline offline. Do not treat its scores as model quality."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class OllamaEmbedder:
    def __init__(self, url: str = "http://localhost:11434", model: str = "nomic-embed-text",
                 dim: int = 768, timeout: float = 60.0):
        self.url, self.model, self.dim, self.timeout = url.rstrip("/"), model, dim, timeout

    def _embed(self, inputs: list[str]) -> np.ndarray:
        req = urllib.request.Request(
            f"{self.url}/api/embed",
            data=json.dumps({"model": self.model, "input": inputs}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.url} ({e}). Is it running, and did you "
                f"`ollama pull {self.model}`?") from e
        vecs = np.array(data["embeddings"], dtype=np.float32)
        if vecs.shape[1] != self.dim:
            raise RuntimeError(f"model returned dim {vecs.shape[1]}, expected {self.dim}")
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def embed_documents(self, texts):
        return self._embed([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return self._embed([f"search_query: {text}"])[0]
