import io
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable

import chromadb
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from rag.config import Settings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_model = None
MAX_COLLECTIONS = 20
_client = chromadb.EphemeralClient()
_names: deque[str] = deque()


class PdfRejected(Exception):
    """The uploaded PDF cannot be used; the message is shown to the visitor."""


@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    text: str


def extract_pages(data: bytes, settings: Settings) -> list[str]:
    if len(data) > settings.max_pdf_mb * 1024 * 1024:
        raise PdfRejected(f"Le PDF dépasse {settings.max_pdf_mb} Mo.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfRejected("Le PDF est protégé par un mot de passe.")
        if len(reader.pages) > settings.max_pages:
            raise PdfRejected(f"Le PDF dépasse {settings.max_pages} pages.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfRejected:
        raise
    except (PyPdfError, ValueError, OSError, KeyError) as exc:
        raise PdfRejected("Le fichier n'est pas un PDF lisible.") from exc
    if not any(p.strip() for p in pages):
        raise PdfRejected("Aucun texte trouvé : les PDF scannés ne sont pas supportés.")
    return pages


def split_pages(pages: list[str], settings: Settings) -> list[Chunk]:
    step = max(settings.chunk_size - settings.chunk_overlap, 1)
    chunks = []
    for number, text in enumerate(pages, 1):
        text = text.strip()
        for n, start in enumerate(range(0, len(text), step)):
            piece = text[start:start + settings.chunk_size]
            chunks.append(Chunk(f"p{number}-c{n}", number, piece))
            if start + settings.chunk_size >= len(text):
                break
    if len(chunks) > settings.max_chunks:
        raise PdfRejected(f"Le cours est trop long ({len(chunks)} passages, maximum {settings.max_chunks}).")
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model.encode(texts).tolist()


def build_index(chunks: list[Chunk], embed: Callable | None = None):
    embed = embed or embed_texts
    name = f"course-{uuid.uuid4().hex}"
    collection = _client.create_collection(name, metadata={"hnsw:space": "cosine"})
    _names.append(name)
    while len(_names) > MAX_COLLECTIONS:
        _delete(_names.popleft())
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed([c.text for c in chunks]),
        metadatas=[{"page": c.page} for c in chunks],
    )
    return collection


def _delete(name: str) -> None:
    try:
        _client.delete_collection(name)
    except Exception:
        pass


def drop_index(collection) -> None:
    """Free a course index as soon as the session no longer needs it."""
    if collection.name in _names:
        _names.remove(collection.name)
    _delete(collection.name)
