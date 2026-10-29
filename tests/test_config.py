from rag.config import load_settings


def test_defaults_match_spec(monkeypatch):
    for name in ("MAX_PDF_MB", "MAX_PAGES", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K",
                 "VISITOR_DAILY_CAP", "GLOBAL_DAILY_CAP"):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert (s.max_pdf_mb, s.max_pages) == (5, 40)
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (800, 100, 6)
    assert (s.visitor_daily_cap, s.global_daily_cap) == (5, 100)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    assert load_settings().chunk_size == 500
