from rag.fallback import list_samples, load_fallback_quiz


def test_samples_are_listed():
    assert len(list_samples()) == 2


def test_fallback_quiz_validates_and_respects_n():
    for course in list_samples():
        quiz = load_fallback_quiz(course, 3)
        assert len(quiz.questions) == 3
        assert all(len(q.choices) == 4 and 0 <= q.correct_index <= 3 for q in quiz.questions)


def test_n_larger_than_available_returns_all():
    course = list_samples()[0]
    assert len(load_fallback_quiz(course, 50).questions) >= 3
