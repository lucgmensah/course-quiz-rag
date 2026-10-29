from pathlib import Path
from streamlit.testing.v1 import AppTest

from app import score
from rag.generate import Question, Quiz


def make_quiz():
    q = lambda i: Question(question="q", choices=["a", "b", "c", "d"], correct_index=i,
                           explanation="e", source_chunk_ids=[])
    return Quiz(questions=[q(0), q(1), q(2)])


def test_score_counts_correct_answers():
    assert score(make_quiz(), {0: 0, 1: 3, 2: 2}) == 2


def test_score_ignores_unanswered():
    assert score(make_quiz(), {0: 0}) == 1


def test_app_shows_three_tabs():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py")).run(timeout=60)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Quiz", "Architecture du RAG", "À propos"]
