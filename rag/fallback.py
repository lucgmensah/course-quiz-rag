import json
from pathlib import Path

from rag.generate import Quiz

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def list_samples() -> list[str]:
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.pdf"))


def sample_pdf(course: str) -> bytes:
    return (SAMPLES_DIR / f"{course}.pdf").read_bytes()


def load_fallback_quiz(course: str, n: int) -> Quiz:
    data = json.loads((SAMPLES_DIR / "quizzes" / f"{course}.json").read_text(encoding="utf-8"))
    quiz = Quiz.model_validate(data)
    return Quiz(questions=quiz.questions[:n])
