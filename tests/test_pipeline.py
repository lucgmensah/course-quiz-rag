import dataclasses
import json
from types import SimpleNamespace

from rag.config import load_settings
from rag.fallback import list_samples
from rag.ingest import Chunk, build_index
from rag.limits import Limiter
from rag.pipeline import produce_quiz

GOOD = json.dumps({"questions": [{
    "question": "Q?", "choices": ["a", "b", "c", "d"], "correct_index": 1,
    "explanation": "e", "source_chunk_ids": ["p1-c0"]}]})


def fake_embed(texts):
    return [[1.0, float(len(t))] for t in texts]


class Client:
    def __init__(self, reply=GOOD, error=None):
        self.reply, self.error, self.calls = reply, error, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def setup(tmp_path, **over):
    s = dataclasses.replace(load_settings(), **over)
    lim = Limiter(str(tmp_path / "u.sqlite3"), s, today=lambda: "2026-10-29")
    coll = build_index([Chunk("p1-c0", 1, "some course text")], embed=fake_embed)
    return s, lim, coll


def call(client, s, lim, coll, client_ok=True):
    return produce_quiz(client if client_ok else None, lim, "v1", coll, list_samples()[0],
                        "text", "qcm", "debutant", 1, s, embed=fake_embed)


def test_success_records_usage_and_returns_sources(tmp_path):
    s, lim, coll = setup(tmp_path, visitor_daily_cap=1)
    result = call(Client(), s, lim, coll)
    assert result.fallback_reason is None
    assert result.sources["p1-c0"].page == 1 and result.prompt
    assert not lim.allow("v1")


def test_cap_reached_serves_fallback_without_calling_llm(tmp_path):
    s, lim, coll = setup(tmp_path, visitor_daily_cap=1)
    lim.record("v1")
    client = Client()
    result = call(client, s, lim, coll)
    assert result.fallback_reason and client.calls == 0 and result.quiz.questions


def test_quota_error_serves_fallback_and_does_not_count(tmp_path):
    err = Exception("too many requests")
    err.status_code = 429
    s, lim, coll = setup(tmp_path)
    result = call(Client(error=err), s, lim, coll)
    assert result.fallback_reason and result.quiz.questions
    assert lim.allow("v1")


def test_missing_api_key_serves_fallback(tmp_path):
    s, lim, coll = setup(tmp_path)
    assert call(None, s, lim, coll).fallback_reason
