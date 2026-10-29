from rag.ingest import Chunk, build_index
from rag.retrieve import retrieve

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


def test_fewer_chunks_than_k_returns_all():
    assert len(retrieve(make_collection(), "python", 10, embed=fake_embed)) == 3


def test_empty_theme_covers_course():
    passages = retrieve(make_collection(), "", 2, embed=fake_embed)
    assert len(passages) == 2
    assert passages[0].page != passages[1].page
