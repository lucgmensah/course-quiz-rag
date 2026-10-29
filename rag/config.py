import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_pdf_mb: int
    max_pages: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    llm_base_url: str
    llm_model: str
    visitor_daily_cap: int
    global_daily_cap: int
    max_chunks: int
    min_similarity: float


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        max_pdf_mb=int(env("MAX_PDF_MB", 5)),
        max_pages=int(env("MAX_PAGES", 40)),
        chunk_size=int(env("CHUNK_SIZE", 800)),
        chunk_overlap=int(env("CHUNK_OVERLAP", 100)),
        top_k=int(env("TOP_K", 6)),
        llm_base_url=env("LLM_BASE_URL", "https://opencode.ai/zen/go/v1"),
        llm_model=env("LLM_MODEL", "glm-5.2"),
        visitor_daily_cap=int(env("VISITOR_DAILY_CAP", 5)),
        global_daily_cap=int(env("GLOBAL_DAILY_CAP", 100)),
        max_chunks=int(env("MAX_CHUNKS", 500)),
        min_similarity=float(env("MIN_SIMILARITY", 0.10)),
    )
