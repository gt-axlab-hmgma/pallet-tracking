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
