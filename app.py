import hashlib
import os
import uuid

import streamlit as st
from openai import OpenAI

from rag.architecture import pipeline_dot, pipeline_stages
from rag.config import load_settings
from rag.fallback import list_samples, sample_pdf
from rag.generate import GenerationFailed, Quiz
from rag.ingest import PdfRejected, build_index, extract_pages, split_pages
from rag.limits import Limiter, visitor_id
from rag.pipeline import produce_quiz

SETTINGS = load_settings()
LEVELS = ["débutant", "intermédiaire", "avancé"]
EXERCISES = {"QCM": "qcm", "Exercice d'analyse": "analyse"}


def score(quiz: Quiz, answers: dict[int, int]) -> int:
    return sum(1 for i, q in enumerate(quiz.questions) if answers.get(i) == q.correct_index)


@st.cache_resource
def get_limiter() -> Limiter:
    return Limiter(os.environ.get("USAGE_DB", "usage.sqlite3"), SETTINGS)


def get_client():
    key = os.environ.get("OPENCODE_API_KEY")
    return OpenAI(base_url=SETTINGS.llm_base_url, api_key=key) if key else None


def current_visitor() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    try:
        forwarded = st.context.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0].strip() if forwarded else None
    except Exception:
        ip = None
    return visitor_id(st.session_state.session_id, ip)


def load_course(data: bytes, key: str):
    """Index the course once per session and course."""
    if st.session_state.get("course_key") != key:
        pages = extract_pages(data, SETTINGS)
        st.session_state.collection = build_index(split_pages(pages, SETTINGS))
        st.session_state.course_key = key
    return st.session_state.collection


def render_quiz_tab():
    samples = list_samples()
    source = st.radio("Cours", ["Cours d'exemple", "Mon PDF"], horizontal=True)
    if source == "Mon PDF":
        upload = st.file_uploader(
            f"PDF (max {SETTINGS.max_pdf_mb} Mo, {SETTINGS.max_pages} pages)", type="pdf")
        course, data = samples[0], (upload.getvalue() if upload else None)
    else:
        course = st.selectbox("Choisir un cours", samples)
        data = sample_pdf(course)

    col1, col2, col3 = st.columns(3)
    exercise = col1.selectbox("Type d'exercice", list(EXERCISES))
    level = col2.selectbox("Niveau", LEVELS)
    n = col3.slider("Nombre de questions", 1, 10, 5)
    theme = st.text_input("Thème (laisser vide pour tout le cours)")

    if st.button("Générer le quiz", type="primary", disabled=data is None):
        try:
            with st.spinner("Indexation du cours et génération du quiz…"):
                collection = load_course(data, hashlib.sha256(data).hexdigest())
                result = produce_quiz(get_client(), get_limiter(), current_visitor(), collection,
                                      course, theme, EXERCISES[exercise], level, n, SETTINGS)
        except PdfRejected as exc:
            st.error(str(exc))
        except GenerationFailed as exc:
            st.error(str(exc))
        else:
            st.session_state.result = result
            st.session_state.submitted = False
            for key in [k for k in st.session_state if str(k).startswith("answer_")]:
                del st.session_state[key]

    result = st.session_state.get("result")
    if result:
        render_quiz(result)


def render_quiz(result):
    if result.fallback_reason:
        st.warning(f"{result.fallback_reason} Quiz pré-généré d'un cours d'exemple affiché à la place.")
    quiz = result.quiz
    submitted = st.session_state.get("submitted", False)
    answers = {}
    for i, q in enumerate(quiz.questions):
        st.markdown(f"**{i + 1}. {q.question}**")
        choice = st.radio("Réponse", range(4), index=None, key=f"answer_{i}", label_visibility="collapsed",
                          format_func=lambda c, q=q: q.choices[c], disabled=submitted)
        if choice is not None:
            answers[i] = choice
        if submitted:
            if answers.get(i) == q.correct_index:
                st.success("Correct")
            else:
                st.error(f"Mauvaise réponse. Bonne réponse : {q.choices[q.correct_index]}")
            st.caption(q.explanation)
            with st.expander("Passages du cours à l'origine de cette question"):
                for cid in q.source_chunk_ids:
                    p = result.sources.get(cid)
                    if p:
                        st.markdown(f"*Page {p.page}* : {p.text}")
    if not submitted:
        if st.button("Corriger"):
            st.session_state.submitted = True
            st.rerun()
    else:
        st.metric("Score", f"{score(quiz, answers)} / {len(quiz.questions)}")


def render_architecture_tab():
    st.markdown("Le pipeline RAG réellement exécuté, avec les paramètres actuels de l'application.")
    st.graphviz_chart(pipeline_dot(SETTINGS), use_container_width=True)
    for i, stage in enumerate(pipeline_stages(SETTINGS), 1):
        params = ", ".join(f"{k} : {v}" for k, v in stage["params"].items())
        st.markdown(f"**{i}. {stage['name']}** : {stage['description']}" + (f" *({params})*" if params else ""))
    result = st.session_state.get("result")
    if result and result.prompt:
        st.subheader("Dernière génération")
        with st.expander("Passages récupérés"):
            for p in result.sources.values():
                st.markdown(f"*Page {p.page}* (score {p.score:.2f}) : {p.text}")
        with st.expander("Prompt envoyé au modèle"):
            st.code(result.prompt, language="text")
    elif result:
        st.caption("La dernière génération était un quiz de secours : aucun appel au modèle.")


def render_about_tab():
    st.markdown(
        "Démo de génération de quiz par **RAG** : le cours est découpé, indexé avec des embeddings, "
        "puis un LLM crée des questions à partir des passages retrouvés. Le code est sur "
        "[GitHub](https://github.com/lucgmensah/course-quiz-rag)."
    )


def main():
    st.set_page_config(page_title="Course Quiz RAG", page_icon="📚")
    st.title("Course Quiz RAG")
    quiz_tab, arch_tab, about_tab = st.tabs(["Quiz", "Architecture du RAG", "À propos"])
    with quiz_tab:
        render_quiz_tab()
    with arch_tab:
        render_architecture_tab()
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
