import hashlib
import sqlite3
import threading
from datetime import date
from typing import Callable

from rag.config import Settings


def visitor_id(session_id: str, ip: str | None) -> str:
    """Stable anonymous id: the IP when known, otherwise the session."""
    return hashlib.sha256((ip or f"session:{session_id}").encode()).hexdigest()[:16]


class Limiter:
    def __init__(self, db_path: str, settings: Settings, today: Callable[[], str] | None = None):
        self._path = db_path
        self._settings = settings
        self._today = today or (lambda: date.today().isoformat())
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS usage "
                "(day TEXT, visitor TEXT, n INTEGER, PRIMARY KEY (day, visitor))"
            )

    def _connect(self):
        return sqlite3.connect(self._path)

    def allow(self, visitor: str) -> bool:
        day = self._today()
        with self._lock, self._connect() as db:
            mine = db.execute("SELECT n FROM usage WHERE day=? AND visitor=?", (day, visitor)).fetchone()
            total = db.execute("SELECT COALESCE(SUM(n), 0) FROM usage WHERE day=?", (day,)).fetchone()[0]
        return (mine[0] if mine else 0) < self._settings.visitor_daily_cap and total < self._settings.global_daily_cap

    def record(self, visitor: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO usage (day, visitor, n) VALUES (?, ?, 1) "
                "ON CONFLICT(day, visitor) DO UPDATE SET n = n + 1",
                (self._today(), visitor),
            )
