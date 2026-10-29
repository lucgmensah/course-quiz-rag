import json
from types import SimpleNamespace

import pytest

from rag.config import load_settings
from rag.generate import GenerationFailed, QuotaExhausted, build_prompt, generate_quiz
from rag.retrieve import Passage

PASSAGES = [Passage("p1-c0", 1, "Mitochondria produce ATP.", 0.9),
            Passage("p2-c0", 2, "Ribosomes build proteins.", 0.8)]


def question(**over):
    q = {"question": "What produces ATP?", "choices": ["Mitochondria", "Ribosomes", "Nucleus", "Golgi"],
         "correct_index": 0, "explanation": "See page 1.", "source_chunk_ids": ["p1-c0"]}
    return {**q, **over}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def quiz_json(*questions):
    return json.dumps({"questions": list(questions)})


def run(client, n=1):
    return generate_quiz(client, PASSAGES, "qcm", "debutant", n, load_settings())


def test_parses_correct_index():
    quiz, _ = run(FakeClient([quiz_json(question(correct_index=2))]))
    assert quiz.questions[0].correct_index == 2


def test_strips_json_code_fence():
    quiz, _ = run(FakeClient(["```json\n" + quiz_json(question()) + "\n```"]))
    assert len(quiz.questions) == 1


def test_correct_index_out_of_range_retries_then_fails():
    client = FakeClient([quiz_json(question(correct_index=4))] * 2)
    with pytest.raises(GenerationFailed):
        run(client)
    assert len(client.calls) == 2


def test_three_choices_is_invalid():
    client = FakeClient([quiz_json(question(choices=["a", "b", "c"]))] * 2)
    with pytest.raises(GenerationFailed):
        run(client)


def test_retry_once_then_success():
    client = FakeClient(["not json", quiz_json(question())])
    quiz, _ = run(client)
    assert len(quiz.questions) == 1 and len(client.calls) == 2


def test_429_raises_quota_exhausted():
    err = Exception("rate limited")
    err.status_code = 429
    with pytest.raises(QuotaExhausted):
        run(FakeClient([err]))


def test_prompt_contains_all_passages_and_level():
    prompt = build_prompt(PASSAGES, "qcm", "avance", 3)
    assert "Mitochondria produce ATP." in prompt and "Ribosomes build proteins." in prompt
    assert "avance" in prompt and "3" in prompt and "p2-c0" in prompt
