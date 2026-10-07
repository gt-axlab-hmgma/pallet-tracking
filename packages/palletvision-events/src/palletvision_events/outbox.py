import json
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from palletvision_events.record import ZoneEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    row_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id  TEXT NOT NULL UNIQUE,
    device_id TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    payload   TEXT NOT NULL,
    sent_at   REAL
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox (row_id) WHERE sent_at IS NULL;
CREATE INDEX IF NOT EXISTS outbox_device_seq ON outbox (device_id, seq);
"""


class Outbox:
    """Durable local queue on the device (SQLite, WAL).

    The inference loop only pays for one small local write. A separate sync agent drains it,
    so a network outage never blocks detection and nothing is lost across a reboot.
    """

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)

    def write(self, events: Sequence[ZoneEvent]) -> None:
        if not events:
            return
        rows = [
            (str(e.event_id), e.device_id, e.seq, json.dumps(e.to_dict(), separators=(",", ":")))
            for e in events
        ]
        with self._lock:
            self._db.execute("BEGIN")
            self._db.executemany(
                "INSERT OR IGNORE INTO outbox (event_id, device_id, seq, payload) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            self._db.execute("COMMIT")

    def pending(self, limit: int = 200) -> list[tuple[int, ZoneEvent]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT row_id, payload FROM outbox WHERE sent_at IS NULL ORDER BY row_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [(row_id, ZoneEvent.from_dict(json.loads(payload))) for row_id, payload in rows]

    def pending_count(self) -> int:
        with self._lock:
            return int(
                self._db.execute("SELECT count(*) FROM outbox WHERE sent_at IS NULL").fetchone()[0]
            )

    def mark_sent(self, row_ids: Sequence[int]) -> None:
        if not row_ids:
            return
        with self._lock:
            self._db.execute("BEGIN")
            self._db.executemany(
                "UPDATE outbox SET sent_at = ? WHERE row_id = ?",
                [(time.time(), r) for r in row_ids],
            )
            self._db.execute("COMMIT")

    def next_seq(self, device_id: str) -> int:
        """First unused per-device sequence number, including events already sent."""
        with self._lock:
            row = self._db.execute(
                "SELECT max(seq) FROM outbox WHERE device_id = ?", (device_id,)
            ).fetchone()
        return 0 if row[0] is None else int(row[0]) + 1

    def purge_sent(self, older_than_s: float) -> int:
        """Delete rows that were acknowledged more than ``older_than_s`` seconds ago."""
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM outbox WHERE sent_at IS NOT NULL AND sent_at < ?",
                (time.time() - older_than_s,),
            )
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
