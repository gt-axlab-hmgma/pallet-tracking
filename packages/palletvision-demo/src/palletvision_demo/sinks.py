import contextlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from palletvision_demo.settings import Settings
from palletvision_events.db import connection_url
from palletvision_events.outbox import Outbox
from palletvision_events.record import ZoneEvent
from palletvision_events.sinks import EventSink, JsonlSink, StdoutSink
from palletvision_events.sinks_postgres import PostgresSink
from palletvision_events.zone_store import ZoneConfigStore
from palletvision_zones.config import ZoneConfig


def _with_timeout(url: str, seconds: int = 2) -> str:
    if "connect_timeout" in url:
        return url
    return f"{url}{'&' if '?' in url else '?'}connect_timeout={seconds}"


def _jsonl_next_seq(path: Path, device_id: str) -> int:
    last = -1
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("device_id") == device_id:
                last = max(last, int(row.get("seq", -1)))
    return last + 1


class SafeSink:
    """Wraps a sink so a failing destination (say Postgres is down) never breaks the demo.

    The sink is created lazily and discarded after a failure, so it reconnects on the next
    write. ``write`` returns an error string instead of raising.
    """

    def __init__(self, make: Callable[[], EventSink]) -> None:
        self._make = make
        self._sink: EventSink | None = None

    def write(self, events: Sequence[ZoneEvent]) -> str | None:
        try:
            if self._sink is None:
                self._sink = self._make()
            self._sink.write(events)
            return None
        except Exception as exc:
            self.close()
            return f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        if self._sink is not None:
            with contextlib.suppress(Exception):
                self._sink.close()
            self._sink = None


def build_sink(settings: Settings) -> tuple[SafeSink, int, list[str]]:
    """Returns the sink, the first unused seq for this device, and any startup warnings."""
    warnings: list[str] = []
    if settings.sink == "stdout":
        return SafeSink(StdoutSink), 0, warnings
    if settings.sink == "jsonl":
        start = _jsonl_next_seq(settings.jsonl_path, settings.device_id)
        return SafeSink(lambda: JsonlSink(settings.jsonl_path)), start, warnings
    if settings.sink == "outbox":
        outbox = Outbox(settings.outbox_path)
        return SafeSink(lambda: outbox), outbox.next_seq(settings.device_id), warnings
    url = _with_timeout(settings.db_url or connection_url())
    start = 0
    try:
        probe = PostgresSink(url)
        start = probe.next_seq(settings.device_id)
        probe.close()
    except Exception as exc:
        warnings.append(
            f"Postgres is not reachable yet ({type(exc).__name__}); events will be retried "
            "on each move. Start it with: docker compose up -d db && uv run pv-db migrate"
        )
    return SafeSink(lambda: PostgresSink(url)), start, warnings


class SafeConfigStore:
    """Stores zone versions in Postgres without ever breaking the demo if the database is down.

    Methods return an error string (or None) instead of raising. A version is only inserted
    once per run; the connection is dropped after a failure and retried on the next call.
    """

    def __init__(self, make: Callable[[], ZoneConfigStore]) -> None:
        self._make = make
        self._store: ZoneConfigStore | None = None
        self._stored: tuple[str, int] | None = None

    def latest_version(self, camera_id: str) -> int:
        """Highest stored version, or 0 when there is none or the database is unreachable."""
        try:
            if self._store is None:
                self._store = self._make()
            return self._store.latest_version(camera_id)
        except Exception:
            self.close()
            return 0

    def ensure(self, config: ZoneConfig) -> str | None:
        """Make sure this exact version is stored. Returns an error message, or None."""
        key = (config.camera_id, config.version)
        if self._stored == key:
            return None
        try:
            if self._store is None:
                self._store = self._make()
            self._store.save(config)
            self._stored = key
            return None
        except Exception as exc:
            self.close()
            return f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        if self._store is not None:
            with contextlib.suppress(Exception):
                self._store.close()
            self._store = None


def build_config_store(settings: Settings) -> SafeConfigStore | None:
    """Only the Postgres sink stores zone versions; the edge path will sync them later."""
    if settings.sink != "postgres":
        return None
    url = _with_timeout(settings.db_url or connection_url())
    return SafeConfigStore(lambda: ZoneConfigStore(url, saved_by="demo"))
