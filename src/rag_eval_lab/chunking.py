"""Markdown-aware chunking with metadata.

The heading is kept as metadata, a small amount of markdown noise is stripped, and split over-long sections into
overlapping windows of words so every chunk stays a predictable size.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.source} > {self.heading}]"


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse a minimal `key: value` frontmatter block delimited by '---'."""
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---\n", 4)
    if end == -1:
        return {}, raw
    meta = {}
    for line in raw[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[end + 5:]


def strip_markdown(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
    return re.sub(r"[ \t]+", " ", text).strip()


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    if len(words) <= size:
        return [words]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_document(source: str, raw: str, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    meta, body = parse_frontmatter(raw)
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    for line in body.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)", line)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for heading, lines in sections:
        text = strip_markdown("\n".join(lines))
        if not text:
            continue
        for i, win in enumerate(_windows(text.split(), max_words, overlap)):
            chunks.append(Chunk(
                id=f"{source}#{len(chunks)}",
                source=source,
                heading=heading if i == 0 else f"{heading} (part {i + 1})",
                text=" ".join(win),
                metadata=dict(meta),
            ))
    return chunks


def chunk_directory(path: str | Path, **kwargs) -> list[Chunk]:
    files = sorted(Path(path).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md files found in {path}")
    out: list[Chunk] = []
    for f in files:
        out.extend(chunk_document(f.stem, f.read_text(encoding="utf-8"), **kwargs))
    return out
