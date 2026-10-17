import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from palletvision_events.record import EventFactory, ZoneEvent
from palletvision_events.sinks_postgres import PostgresSink
from palletvision_events.storage.outbox import Outbox
from palletvision_events.sync import Syncer, backoff_delay
from palletvision_zones.engine import Transition

T0 = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)


def make_events(n: int) -> list[ZoneEvent]:
    state = {"i": 0}

    def clock() -> datetime:
        state["i"] += 1
        return T0 + timedelta(seconds=state["i"])

    f = EventFactory("nano-01", "cam01", "mock", clock=clock)
    return [f.make(Transition("enter", "z", i, "pallet")) for i in range(n)]


class Recorder:
    def __init__(self, fail_first: int = 0) -> None:
        self.batches: list[list[ZoneEvent]] = []
        self.calls = 0
        self.fail_first = fail_first
        self.closed = 0

    def write(self, events: Sequence[ZoneEvent]) -> None:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("network down")
        self.batches.append(list(events))

    def close(self) -> None:
        self.closed += 1


def test_sync_once_drains_in_batches_and_marks_sent(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "o.db")
    events = make_events(5)
    outbox.write(events)
    sink = Recorder()
    syncer = Syncer(outbox, lambda: sink, batch_size=2)
    assert syncer.sync_once() == 2
    assert syncer.drain() == 3
    assert [len(b) for b in sink.batches] == [2, 2, 1]
    assert [e for b in sink.batches for e in b] == events
    assert outbox.pending_count() == 0


def test_failure_keeps_events_pending_then_recovers(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "o.db")
    outbox.write(make_events(3))
    sink = Recorder(fail_first=2)
    syncer = Syncer(outbox, lambda: sink, batch_size=10)
    with pytest.raises(ConnectionError):
        syncer.sync_once()
    assert outbox.pending_count() == 3  # nothing lost
    with pytest.raises(ConnectionError):
        syncer.sync_once()
    assert syncer.sync_once() == 3
    assert outbox.pending_count() == 0
    assert sink.closed == 2  # the broken sink is discarded after each failure


def test_backoff_grows_and_is_capped() -> None:
    delays = [backoff_delay(n, base=1.0, cap=30.0, jitter=0.0) for n in range(8)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
    jittered = backoff_delay(3, base=1.0, cap=30.0, jitter=0.5)
    assert 8.0 * 0.5 <= jittered <= 8.0 * 1.5


def test_run_retries_until_success_and_stops_on_event(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "o.db")
    outbox.write(make_events(4))
    sink = Recorder(fail_first=3)
    stop = threading.Event()
    syncer = Syncer(outbox, lambda: sink, batch_size=10, sleep=lambda s: None)
    original = sink.write

    def stopping_write(events: Sequence[ZoneEvent]) -> None:
        original(events)
        stop.set()

    sink.write = stopping_write  # type: ignore[method-assign]
    syncer.run(interval=0.0, stop=stop)
    assert outbox.pending_count() == 0
    assert syncer.failures == 3


def test_end_to_end_outage_loses_nothing_and_duplicates_nothing(
    tmp_path: Path, db_url: str
) -> None:
    outbox = Outbox(tmp_path / "o.db")
    events = make_events(6)
    outbox.write(events[:3])
    attempts = {"n": 0}

    def factory() -> PostgresSink:
        attempts["n"] += 1
        if attempts["n"] <= 2:  # database "unreachable" for the first two tries
            raise psycopg.OperationalError("connection refused")
        return PostgresSink(db_url)

    syncer = Syncer(outbox, factory, batch_size=2)
    for _ in range(2):
        with pytest.raises(psycopg.OperationalError):
            syncer.sync_once()
    outbox.write(events[3:])  # detection keeps producing events during the outage
    assert syncer.drain() == 6
    assert syncer.drain() == 0
    with psycopg.connect(db_url) as conn:
        rows = conn.execute("SELECT seq FROM events ORDER BY seq").fetchall()
    assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5]
