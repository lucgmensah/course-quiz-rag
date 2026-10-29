from dataclasses import dataclass

from rag.config import Settings
from rag.fallback import load_fallback_quiz, sample_pdf
from rag.generate import GenerationFailed, Quiz, QuotaExhausted, generate_quiz
from rag.ingest import extract_pages, split_pages
from rag.limits import Limiter
from rag.retrieve import Passage, retrieve


@dataclass
class QuizResult:
    quiz: Quiz
    sources: dict[str, Passage]
    prompt: str | None
    fallback_reason: str | None


def _fallback(course: str, n: int, settings: Settings, reason: str) -> QuizResult:
    chunks = split_pages(extract_pages(sample_pdf(course), settings), settings)
    sources = {c.id: Passage(c.id, c.page, c.text, 0.0) for c in chunks}
    return QuizResult(load_fallback_quiz(course, n), sources, None, reason)


def produce_quiz(client, limiter: Limiter, visitor: str, get_collection, fallback_course: str,
                 theme: str, exercise_type: str, level: str, n: int, settings: Settings,
                 embed=None) -> QuizResult:
    """Retrieve, generate and record usage; serve a pre-generated quiz when generation is unavailable."""
    if client is None:
        return _fallback(fallback_course, n, settings, "Aucune clé API configurée.")
    if not limiter.allow(visitor):
        return _fallback(fallback_course, n, settings, "Limite d'utilisation atteinte pour aujourd'hui.")
    # get_collection is lazy: a capped visitor never pays for PDF extraction and embedding
    passages = retrieve(get_collection(), theme, settings.top_k, embed=embed,
                        min_similarity=settings.min_similarity)
    try:
        quiz, prompt = generate_quiz(client, passages, exercise_type, level, n, settings)
    except QuotaExhausted:
        return _fallback(fallback_course, n, settings, "Quota du modèle atteint.")
    except GenerationFailed:
        limiter.record(visitor)  # failed attempts cost LLM calls too
        raise
    limiter.record(visitor)
    return QuizResult(quiz, {p.id: p for p in passages}, prompt, None)
