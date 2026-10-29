from rag.ingest import Chunk, build_index
import pytest

from rag.retrieve import ThemeNotFound, retrieve

TOPICS = {
    "cats": "cats purr and chase mice",
    "ocean": "the ocean is salty and deep",
    "code": "python functions return values",
}


def fake_embed(texts):
    keys = ["cat", "ocean", "python"]
    return [[1.0 if k in t.lower() else 0.0 for k in keys] + [0.1] for t in texts]


def make_collection():
    chunks = [Chunk(f"p{i}-c0", i, text) for i, text in enumerate(TOPICS.values(), 1)]
    return build_index(chunks, embed=fake_embed)


def test_query_is_the_theme_not_the_prompt():
    seen = []

    def spy(texts):
        seen.extend(texts)
        return fake_embed(texts)

    coll = make_collection()
    retrieve(coll, "ocean", 1, embed=spy)
    assert seen == ["ocean"]


def test_returns_page_numbers_and_best_match_first():
    passages = retrieve(make_collection(), "cats", 2, embed=fake_embed)
    assert passages[0].page == 1 and "cats" in passages[0].text


def test_irrelevant_chunks_are_dropped_even_when_k_is_large():
    passages = retrieve(make_collection(), "python", 10, embed=fake_embed)
    assert [p.page for p in passages] == [3]


def test_score_is_cosine_similarity():
    passages = retrieve(make_collection(), "ocean", 1, embed=fake_embed)
    assert passages[0].score == pytest.approx(1.0, abs=1e-3)


def test_unrelated_theme_raises_theme_not_found():
    with pytest.raises(ThemeNotFound):
        retrieve(make_collection(), "zzz", 3, embed=fake_embed)


def test_empty_theme_covers_course():
    passages = retrieve(make_collection(), "", 2, embed=fake_embed)
    assert len(passages) == 2
    assert passages[0].page != passages[1].page
