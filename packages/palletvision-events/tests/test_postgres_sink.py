from datetime import UTC, datetime, timedelta

import psycopg

from palletvision_events.record import EventFactory, ZoneEvent
from palletvision_events.sinks import EventSink
from palletvision_events.sinks_postgres import PostgresSink
from palletvision_zones.engine import Transition

T0 = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)


def make_events(n: int, start: datetime = T0, camera: str = "cam01") -> list[ZoneEvent]:
    clock_state = {"t": start}

    def clock() -> datetime:
        clock_state["t"] += timedelta(seconds=1)
        return clock_state["t"]

    f = EventFactory("nano-01", camera, "mock", config_version=1, clock=clock)
    kinds = ["enter", "exit"]
    return [
        f.make(Transition(kinds[i % 2], "zone_1", 1, "pallet_type_a"))  # type: ignore[arg-type]
        for i in range(n)
    ]


def count(url: str, table: str = "events") -> int:
    with psycopg.connect(url) as conn:
        row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def test_writes_events(db_url: str) -> None:
    sink: EventSink = PostgresSink(db_url)
    sink.write(make_events(3))
    sink.close()
    assert count(db_url) == 3
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            "SELECT event_type, zone_id, object_type, track_id, source, config_version "
            "FROM events ORDER BY seq LIMIT 1"
        ).fetchone()
    assert row == ("enter", "zone_1", "pallet_type_a", 1, "mock", 1)


def test_resending_a_batch_does_not_duplicate(db_url: str) -> None:
    events = make_events(4)
    sink = PostgresSink(db_url)
    sink.write(events)
    sink.write(events)  # a retry after a lost ack
    sink.write(events[:2] + make_events(1, start=T0 + timedelta(hours=1)))
    sink.close()
    assert count(db_url) == 5


def test_empty_batch_is_a_no_op(db_url: str) -> None:
    sink = PostgresSink(db_url)
    sink.write([])
    sink.close()
    assert count(db_url) == 0


def test_creates_partition_for_a_new_month(db_url: str) -> None:
    far = datetime(2031, 3, 15, tzinfo=UTC)
    sink = PostgresSink(db_url)
    sink.write(make_events(2, start=far))
    sink.close()
    assert count(db_url, "events_2031_03") == 2


def test_registers_device_and_camera(db_url: str) -> None:
    sink = PostgresSink(db_url)
    sink.write(make_events(1))
    sink.write(make_events(1, start=T0 + timedelta(minutes=5)))  # second write re-touches rows
    sink.close()
    with psycopg.connect(db_url) as conn:
        device = conn.execute("SELECT device_id, last_seen_at IS NOT NULL FROM devices").fetchall()
        cameras = conn.execute("SELECT camera_id, device_id FROM cameras").fetchall()
    assert device == [("nano-01", True)]
    assert cameras == [("cam01", "nano-01")]


def test_zone_visits_view_reports_dwell(db_url: str) -> None:
    sink = PostgresSink(db_url)
    sink.write(make_events(2))  # enter at +1s, exit at +2s
    sink.write(make_events(1, start=T0 + timedelta(minutes=1)))  # second enter, still open
    sink.close()
    with psycopg.connect(db_url) as conn:
        rows = conn.execute(
            "SELECT zone_id, object_type, dwell_s, t_exit IS NULL FROM zone_visits ORDER BY t_enter"
        ).fetchall()
    assert rows[0][:2] == ("zone_1", "pallet_type_a")
    assert float(rows[0][2]) == 1.0 and rows[0][3] is False
    assert rows[1][2] is None and rows[1][3] is True  # open visit


def test_next_seq_continues_after_existing_events(db_url: str) -> None:
    sink = PostgresSink(db_url)
    assert sink.next_seq("nano-01") == 0
    sink.write(make_events(4))
    assert sink.next_seq("nano-01") == 4
    assert sink.next_seq("someone-else") == 0
    sink.close()
