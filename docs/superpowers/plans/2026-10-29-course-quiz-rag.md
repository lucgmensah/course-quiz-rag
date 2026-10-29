# course-quiz-rag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit demo that turns an uploaded course PDF into an interactive, source-cited quiz through a RAG pipeline, and shows that pipeline's architecture.

**Architecture:** `rag/` holds four small modules (ingest, retrieve, generate, limits) plus config; `app.py` is a thin Streamlit layer over them. The LLM client is injected so tests use a fake. When a cap or the LLM quota is hit, the app serves pre-generated quizzes from `samples/`.

**Tech Stack:** Python 3.11, Streamlit, pypdf, sentence-transformers (`all-MiniLM-L6-v2`), chromadb, openai client (OpenCode Go endpoint), pydantic v2, sqlite3, pytest, Docker.

**Spec:** `docs/superpowers/specs/2026-10-29-course-quiz-rag-design.md`

## Global Constraints

- LLM via `openai` client; `LLM_BASE_URL` default `https://opencode.ai/zen/go/v1`, key in `OPENCODE_API_KEY`, `LLM_MODEL` env-configurable. Key never in the repo.
- Embeddings: `all-MiniLM-L6-v2`. Vector store: Chroma in memory, one collection per session.
- Defaults (all env-configurable): max PDF 5 MB, max 40 pages, chunk 800 chars / overlap 100, top-k 6, 1-10 questions, 5 generations per visitor per day, 100 per day globally.
- Quiz JSON per question: `{question, choices[4], correct_index, explanation, source_chunk_ids}`; one retry on invalid JSON.
- Uploaded PDFs held in memory only, never written to disk, never logged.
- Retrieval query is the theme, never the whole prompt.
- Scanned (text-less) PDFs are refused with a clear message.
- `requirements.txt` is minimal and UTF-8 (the current one is a UTF-16 `pip freeze`).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- Corrupt or password-protected PDF: clear error message, no traceback.
- LLM returns JSON wrapped in a ```json fence: parsed correctly.
- LLM returns `correct_index` outside 0-3 or fewer than 4 choices: counts as invalid, triggers the retry.
- Course has fewer chunks than top-k, or the theme matches nothing relevant: still returns what exists, or a clear "theme not found in this course" message.
- Visitor identity unavailable (no IP header): falls back to session id, cap still applies.

## File Structure

- `rag/config.py` — `Settings` dataclass read from env.
- `rag/ingest.py` — PDF to chunks to Chroma collection.
- `rag/retrieve.py` — theme to top-k passages.
- `rag/generate.py` — prompt, LLM call, validation, retry.
- `rag/limits.py` — SQLite caps.
- `rag/fallback.py` — loads pre-generated quizzes from `samples/`.
- `app.py` — Streamlit UI, 3 tabs.
- `tests/` — one test file per module.
- `samples/` — sample PDFs and fallback quiz JSON.
- `Dockerfile`, `README.md`, `requirements.txt`, `pytest.ini`.

---

### Task 1: Scaffold and config

**Files:**
- Create: `rag/__init__.py`, `rag/config.py`, `tests/test_config.py`, `pytest.ini`
- Modify: `requirements.txt` (rewrite as UTF-8)

**Interfaces:**
- Produces: `Settings` frozen dataclass with fields `max_pdf_mb: int`, `max_pages: int`, `chunk_size: int`, `chunk_overlap: int`, `top_k: int`, `llm_base_url: str`, `llm_model: str`, `visitor_daily_cap: int`, `global_daily_cap: int`; `load_settings() -> Settings`.

- [ ] **Step 1: Write failing tests** `test_defaults_match_spec` (asserts 5, 40, 800, 100, 6, 5, 100) and `test_env_overrides` (monkeypatch `CHUNK_SIZE=500`, asserts 500). Env names are the upper-case field names.
- [ ] **Step 2: Run** `pytest tests/test_config.py -v`; expect FAIL (module missing).
- [ ] **Step 3: Implement `load_settings`** in `rag/config.py`; rewrite `requirements.txt` with only: streamlit, pypdf, sentence-transformers, chromadb, openai, pydantic, pytest.
- [ ] **Step 4: Run** `pytest tests/test_config.py -v`; expect PASS.
- [ ] **Step 5: Commit** `feat: add config and minimal requirements`.

### Task 2: Ingestion

**Files:**
- Create: `rag/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces: `class PdfRejected(Exception)`; `Chunk` dataclass `(id: str, page: int, text: str)`; `extract_pages(data: bytes, settings) -> list[str]` (raises `PdfRejected` for oversize, too many pages, corrupt, encrypted, no extractable text); `split_pages(pages: list[str], settings) -> list[Chunk]`; `build_index(chunks: list[Chunk], embed=None) -> chromadb.Collection` (in-memory, unique collection name, `page` metadata; `embed` is an optional callable for tests).

- [ ] **Step 1: Write failing tests**: `test_split_respects_size_and_overlap`, `test_chunks_keep_page_numbers`, `test_oversize_pdf_rejected`, `test_too_many_pages_rejected`, `test_corrupt_pdf_rejected`, `test_encrypted_pdf_rejected`, `test_scanned_pdf_rejected` (use small PDFs built with pypdf in a fixture; blank pages for the scanned case).
- [ ] **Step 2: Run** `pytest tests/test_ingest.py -v`; expect FAIL.
- [ ] **Step 3: Implement** the four functions. Sliding-window split per page; stable ids `p{page}-c{n}`; default embedding via sentence-transformers loaded once (module-level cache).
- [ ] **Step 4: Run** the tests; expect PASS.
- [ ] **Step 5: Commit** `feat: add PDF ingestion and chunk index`.

### Task 3: Retrieval

**Files:**
- Create: `rag/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: Chroma collection from Task 2.
- Produces: `Passage` dataclass `(id: str, page: int, text: str, score: float)`; `retrieve(collection, theme: str, k: int) -> list[Passage]` (empty theme means whole course: return k chunks spread evenly across pages).

- [ ] **Step 1: Write failing tests** with a fake `embed` and a 3-topic mini course: `test_query_is_the_theme_not_the_prompt` (spy on the collection query text), `test_returns_page_numbers`, `test_fewer_chunks_than_k_returns_all`, `test_empty_theme_covers_course`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement `retrieve`.**
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat: add theme-based retrieval`.

### Task 4: Generation

**Files:**
- Create: `rag/generate.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `Passage`, `Settings`.
- Produces: `Question` pydantic model `(question: str, choices: list[str] (exactly 4), correct_index: int (0-3), explanation: str, source_chunk_ids: list[str])`; `Quiz` model `(questions: list[Question])`; `class QuotaExhausted(Exception)`; `class GenerationFailed(Exception)`; `build_prompt(passages, exercise_type: str, level: str, n: int) -> str`; `generate_quiz(client, passages, exercise_type, level, n, settings) -> tuple[Quiz, str]` returning the quiz and the exact prompt sent. `client` is any object with `chat.completions.create(...)`.

- [ ] **Step 1: Write failing tests** with a fake client: `test_parses_correct_index` (the regression for the old parser dropping the answer), `test_strips_json_code_fence`, `test_correct_index_out_of_range_retries_then_fails`, `test_three_choices_is_invalid`, `test_retry_once_then_success`, `test_429_raises_quota_exhausted`, `test_prompt_contains_all_passages_and_level`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement.** Ask the model for JSON only, strip fences, validate with pydantic, retry once, map HTTP 429 or quota messages to `QuotaExhausted`, other failures to `GenerationFailed`.
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat: add structured quiz generation`.

### Task 5: Limits and fallback

**Files:**
- Create: `rag/limits.py`, `rag/fallback.py`, `tests/test_limits.py`, `tests/test_fallback.py`, `samples/` (2 sample PDFs, `samples/quizzes/<course>.json`)

**Interfaces:**
- Consumes: `Settings`, `Quiz`.
- Produces: `class Limiter(db_path: str, settings)` with `allow(visitor_id: str) -> bool` and `record(visitor_id: str) -> None` (SQLite, per-day counts, thread-safe); `visitor_id(session_id: str, ip: str | None) -> str` (hash; falls back to session id when `ip` is None); `list_samples() -> list[str]`; `load_fallback_quiz(course: str, n: int) -> Quiz`.

- [ ] **Step 1: Write failing tests**: `test_visitor_cap_blocks_sixth`, `test_global_cap_blocks`, `test_counts_reset_next_day` (inject a clock), `test_missing_ip_falls_back_to_session`, `test_fallback_quiz_validates_and_respects_n`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** both modules. Pick two freely licensed courses and note the licence in `samples/README.md`. Generate the fallback quizzes once with `generate_quiz` and commit them as JSON.
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat: add usage caps and pre-generated fallback quizzes`.

### Task 6: Quiz UI

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: everything above. Produces the Streamlit entrypoint; pure helper `score(quiz: Quiz, answers: dict[int, int]) -> int` in `app.py`.

- [ ] **Step 1: Write failing tests**: `test_score_counts_correct_answers`; a `streamlit.testing.v1.AppTest` smoke test that the app starts and shows the Quiz, Architecture du RAG and A propos tabs.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement the Quiz tab**: upload or pick a sample, type / level / count / theme, generate (limiter check first; on cap or `QuotaExhausted` show a banner and use the fallback), interactive answers with per-question explanation and source passages (page + text), final score. Store retrieved passages and prompt in `st.session_state` for Task 7. Show `PdfRejected` and `GenerationFailed` messages with `st.error`.
- [ ] **Step 4: Run** the tests, then start the app with the `run` skill and play a quiz in the browser with a sample course. Expected: quiz is playable, sources shown, score shown.
- [ ] **Step 5: Commit** `feat: add Streamlit quiz tab`.

### Task 7: Architecture tab

**Files:**
- Create: `rag/architecture.py`; Modify: `app.py`
- Test: `tests/test_architecture.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces: `pipeline_stages(settings) -> list[dict]` (each `{name, description, params: dict}`) and `pipeline_dot(settings) -> str` (Graphviz source for PDF to parse to chunk to embed to Chroma, theme query to top-k to prompt to LLM to JSON to quiz).

- [ ] **Step 1: Write failing tests**: `test_stage_params_come_from_settings` (override `CHUNK_SIZE`, assert it appears in the chunk stage and the dot source), `test_dot_contains_every_stage`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** the module; render with `st.graphviz_chart` in the second tab, with a short explanation per stage. After a generation, add expanders showing the retrieved passages and the prompt sent.
- [ ] **Step 4: Run** the tests; check the tab in the browser before and after a generation.
- [ ] **Step 5: Commit** `feat: add RAG architecture tab`.

### Task 8: Container, docs, deployment

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `README.md`, `.env.example`

**Interfaces:** none.

- [ ] **Step 1: Write the Dockerfile**: python 3.11 slim, install requirements, pre-download MiniLM at build, expose 8501, run `streamlit run app.py`; SQLite path from env on a volume.
- [ ] **Step 2: Build and run locally**: `docker build -t course-quiz-rag .` then `docker run -p 8501:8501 -e OPENCODE_API_KEY=... course-quiz-rag`; expect the app on `localhost:8501`.
- [ ] **Step 3: Write `README.md`** (what it is, architecture summary, env vars from Global Constraints, local run, deploy) and `.env.example` without real values.
- [ ] **Step 4: Deploy on Dokploy** from the GitHub repo with the Dokploy tools: ask the user before creating the app, setting the domain and setting `OPENCODE_API_KEY`.
- [ ] **Step 5: Commit** `feat: add Dockerfile and README`.

---

## Self-review notes

- Spec coverage: structure (Tasks 1-5, 7), data flow (2-6), architecture tab (7), caps and fallback (5, 6), errors (2, 4, 6), tests (each task), deployment (8). Open items (model choice, sample courses, domain) are handled in Tasks 5 and 8; the model is chosen by listing `/v1/models` with the user's go-ahead before the first live call in Task 5.
- Names match across tasks: `Settings`, `Chunk`, `Passage`, `Question`, `Quiz`, `Limiter`, `QuotaExhausted`, `GenerationFailed`, `PdfRejected`.
