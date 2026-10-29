import json

from pydantic import BaseModel, Field, ValidationError

from rag.config import Settings
from rag.retrieve import Passage


class Question(BaseModel):
    question: str
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3, strict=True)
    explanation: str
    source_chunk_ids: list[str]


class Quiz(BaseModel):
    questions: list[Question] = Field(min_length=1)


class QuotaExhausted(Exception):
    """The LLM quota or rate limit is reached."""


class GenerationFailed(Exception):
    """The LLM did not return a usable quiz."""


EXERCISE_LABELS = {
    "qcm": "des QCM à 4 choix, une seule bonne réponse",
    "analyse": "des exercices d'analyse : une question ouverte dont la réponse attendue "
               "est donnée comme choix correct parmi 4 propositions",
}


def build_prompt(passages: list[Passage], exercise_type: str, level: str, n: int) -> str:
    context = "\n\n".join(f"[{p.id}] (page {p.page})\n{p.text}" for p in passages)
    kind = EXERCISE_LABELS.get(exercise_type, EXERCISE_LABELS["qcm"])
    return (
        f"Tu es un enseignant. À partir UNIQUEMENT des extraits de cours ci-dessous, "
        f"crée {n} question(s) : {kind}. Niveau : {level}. "
        "Réponds avec du JSON seul, sans texte autour, de la forme "
        '{"questions": [{"question": str, "choices": [4 str], "correct_index": 0-3, '
        '"explanation": str, "source_chunk_ids": [ids des extraits utilisés]}]}.\n\n'
        f"Extraits :\n{context}"
    )


MAX_TOKENS = 4096


def _parse(raw: str, valid_ids: set[str], n: int) -> Quiz:
    """Extract the JSON object (tolerating prose or fences around it) and check it against the passages."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    quiz = Quiz.model_validate(json.loads(raw[start:end + 1]))
    for question in quiz.questions:
        question.source_chunk_ids = [i for i in question.source_chunk_ids if i in valid_ids]
        if not question.source_chunk_ids:
            raise ValueError("question without a known source passage")
    return Quiz(questions=quiz.questions[:n])


def generate_quiz(client, passages, exercise_type, level, n, settings: Settings) -> tuple[Quiz, str]:
    prompt = build_prompt(passages, exercise_type, level, n)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429 or "quota" in str(exc).lower():
                raise QuotaExhausted(str(exc)) from exc
            raise GenerationFailed(f"Erreur du modèle : {exc}") from exc
        try:
            return _parse(response.choices[0].message.content or "", {p.id for p in passages}, n), prompt
        except (ValueError, ValidationError):
            continue
    raise GenerationFailed("Le modèle n'a pas renvoyé un quiz valide.")
