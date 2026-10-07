import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from palletvision_events.outbox import Outbox
from palletvision_events.record import EventFactory, ZoneEvent
from palletvision_events.sinks import EventSink
from palletvision_zones.engine import Transition

T0 = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)


def make_events(n: int, device: str = "nano-01", start_seq: int = 0) -> list[ZoneEvent]:
    state = {"i": 0}

    def clock() -> datetime:
        state["i"] += 1
        return T0 + timedelta(seconds=state["i"])

    f = EventFactory(device, "cam01", "mock", start_seq=start_seq, clock=clock)
    return [f.make(Transition("enter", "z", i, "pallet")) for i in range(n)]


def test_pending_returns_events_in_order(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.db")
    sink: EventSink = outbox
    events = make_events(5)
    sink.write(events[:2])
    sink.write(events[2:])
    assert [e for _, e in outbox.pending(limit=100)] == events
    assert [e for _, e in outbox.pending(limit=2)] == events[:2]


def test_mark_sent_removes_from_pending(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.db")
    outbox.write(make_events(4))
    rows = outbox.pending(limit=10)
    outbox.mark_sent([rows[0][0], rows[1][0]])
    assert [e.seq for _, e in outbox.pending(limit=10)] == [2, 3]
    assert outbox.pending_count() == 2


def test_events_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "outbox.db"
    outbox = Outbox(path)
    events = make_events(3)
    outbox.write(events)
    outbox.close()
    assert [e for _, e in Outbox(path).pending(limit=10)] == events


def test_duplicate_event_ids_are_ignored(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.db")
    events = make_events(2)
    outbox.write(events)
    outbox.write(events)
    assert outbox.pending_count() == 2


def test_next_seq_continues_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "outbox.db"
    outbox = Outbox(path)
    assert outbox.next_seq("nano-01") == 0
    outbox.write(make_events(3))
    outbox.write(make_events(2, device="nano-02"))
    outbox.close()
    reopened = Outbox(path)
    assert reopened.next_seq("nano-01") == 3
    assert reopened.next_seq("nano-02") == 2


def test_next_seq_counts_sent_events_too(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.db")
    outbox.write(make_events(3))
    outbox.mark_sent([r for r, _ in outbox.pending(limit=10)])
    assert outbox.next_seq("nano-01") == 3


def test_purge_removes_only_old_sent_rows(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.db")
    outbox.write(make_events(3))
    rows = outbox.pending(limit=10)
    outbox.mark_sent([rows[0][0], rows[1][0]])
    assert outbox.purge_sent(older_than_s=3600) == 0  # sent just now
    assert outbox.purge_sent(older_than_s=-1) == 2
    assert outbox.pending_count() == 1


def test_write_latency_stays_in_the_millisecond_range(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.db")
    events = make_events(300)
    start = time.perf_counter()
    for e in events:
        outbox.write([e])
    per_write_ms = (time.perf_counter() - start) * 1000 / len(events)
    print(f"outbox: {per_write_ms:.3f} ms per single-event write")
    assert per_write_ms < 2.0
