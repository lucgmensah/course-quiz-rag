import dataclasses

from rag.config import load_settings
from rag.limits import Limiter, visitor_id


def make(tmp_path, today=lambda: "2026-10-29", **over):
    s = dataclasses.replace(load_settings(), **over)
    return Limiter(str(tmp_path / "usage.sqlite3"), s, today=today)


def test_visitor_cap_blocks_sixth(tmp_path):
    lim = make(tmp_path)
    for _ in range(5):
        assert lim.allow("v1")
        lim.record("v1")
    assert not lim.allow("v1")
    assert lim.allow("v2")


def test_global_cap_blocks(tmp_path):
    lim = make(tmp_path, global_daily_cap=2)
    lim.record("a")
    lim.record("b")
    assert not lim.allow("c")


def test_counts_reset_next_day(tmp_path):
    day = ["2026-10-29"]
    lim = make(tmp_path, today=lambda: day[0], visitor_daily_cap=1)
    lim.record("v1")
    assert not lim.allow("v1")
    day[0] = "2026-10-30"
    assert lim.allow("v1")


def test_missing_ip_falls_back_to_session():
    assert visitor_id("sess-1", None) == visitor_id("sess-1", None)
    assert visitor_id("sess-1", None) != visitor_id("sess-2", None)
    assert visitor_id("sess-1", "1.2.3.4") == visitor_id("sess-2", "1.2.3.4")
