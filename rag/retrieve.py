from dataclasses import dataclass
from typing import Callable

from rag.ingest import embed_texts


@dataclass(frozen=True)
class Passage:
    id: str
    page: int
    text: str
    score: float


def retrieve(collection, theme: str, k: int, embed: Callable | None = None) -> list[Passage]:
    """Top-k passages for the theme; an empty theme samples the course evenly by page."""
    embed = embed or embed_texts
    if not theme.strip():
        return _spread(collection, k)
    count = collection.count()
    result = collection.query(
        query_embeddings=embed([theme]),
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        Passage(i, m["page"], d, 1.0 - dist)
        for i, d, m, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def _spread(collection, k: int) -> list[Passage]:
    got = collection.get(include=["documents", "metadatas"])
    items = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=lambda t: t[2]["page"])
    if len(items) > k:
        items = [items[i * len(items) // k] for i in range(k)]
    return [Passage(i, m["page"], d, 0.0) for i, d, m in items]
