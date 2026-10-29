import dataclasses

import pytest

from rag.config import load_settings
from rag.ingest import Chunk, PdfRejected, build_index, extract_pages, split_pages
from tests.conftest import encrypt_pdf, make_pdf


@pytest.fixture
def settings():
    return load_settings()


def test_split_respects_size_and_overlap(settings):
    s = dataclasses.replace(settings, chunk_size=100, chunk_overlap=20)
    chunks = split_pages(["a" * 250], s)
    assert [len(c.text) for c in chunks] == [100, 100, 90]
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_chunks_keep_page_numbers(settings):
    chunks = split_pages(["alpha " * 10, "beta " * 10], settings)
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].id == "p1-c0"


def test_extract_pages_reads_text(settings):
    pages = extract_pages(make_pdf(["Hello course", "Second page"]), settings)
    assert pages[0].strip() == "Hello course"
    assert len(pages) == 2


def test_oversize_pdf_rejected(settings):
    s = dataclasses.replace(settings, max_pdf_mb=0)
    with pytest.raises(PdfRejected):
        extract_pages(make_pdf(["x"]), s)


def test_too_many_pages_rejected(settings):
    s = dataclasses.replace(settings, max_pages=2)
    with pytest.raises(PdfRejected):
        extract_pages(make_pdf(["a", "b", "c"]), s)


def test_corrupt_pdf_rejected(settings):
    with pytest.raises(PdfRejected):
        extract_pages(b"not a pdf at all", settings)


def test_encrypted_pdf_rejected(settings):
    with pytest.raises(PdfRejected):
        extract_pages(encrypt_pdf(make_pdf(["secret text"])), settings)


def test_scanned_pdf_rejected(settings):
    with pytest.raises(PdfRejected):
        extract_pages(make_pdf(["", ""]), settings)


def test_build_index_stores_page_metadata():
    fake_embed = lambda texts: [[float(len(t)), 1.0] for t in texts]
    coll = build_index([Chunk("p1-c0", 1, "hello"), Chunk("p2-c0", 2, "world!")], embed=fake_embed)
    got = coll.get(ids=["p2-c0"], include=["metadatas", "documents"])
    assert got["metadatas"][0]["page"] == 2
    assert got["documents"][0] == "world!"


def test_owner_password_only_pdf_is_readable(settings):
    data = encrypt_pdf(make_pdf(["Open text"]), user_password="")
    assert extract_pages(data, settings)[0].strip() == "Open text"


def test_too_many_chunks_rejected(settings):
    s = dataclasses.replace(settings, chunk_size=100, chunk_overlap=0, max_chunks=3)
    with pytest.raises(PdfRejected):
        split_pages(["a" * 1000], s)


def test_old_collections_are_evicted():
    from rag import ingest

    fake = lambda texts: [[1.0, 1.0] for _ in texts]
    for _ in range(ingest.MAX_COLLECTIONS + 5):
        ingest.build_index([Chunk("p1-c0", 1, "x")], embed=fake)
    assert len(ingest._client.list_collections()) <= ingest.MAX_COLLECTIONS


def test_drop_index_removes_collection():
    from rag import ingest

    coll = ingest.build_index([Chunk("p1-c0", 1, "x")], embed=lambda t: [[1.0, 1.0]] * len(t))
    ingest.drop_index(coll)
    assert coll.name not in [c if isinstance(c, str) else c.name for c in ingest._client.list_collections()]
