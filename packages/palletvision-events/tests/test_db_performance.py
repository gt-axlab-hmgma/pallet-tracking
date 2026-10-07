"""Throughput and query-plan checks for the events table (need a running Postgres)."""

import time
from datetime import UTC, datetime, timedelta

import psycopg

from palletvision_events.record import EventFactory, ZoneEvent
from palletvision_events.sinks_postgres import PostgresSink
from palletvision_zones.engine import Transition

T0 = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=2)


def many_events(n: int, cameras: int = 1, zones: int = 1) -> list[ZoneEvent]:
    state = {"i": 0}

    def clock() -> datetime:
        state["i"] += 1
        return T0 + timedelta(milliseconds=state["i"] * 100)

    factories = [
        EventFactory(f"nano-{c % 3}", f"cam{c:02d}", "mock", clock=clock) for c in range(cameras)
    ]
    out = []
    for i in range(n):
        kind = "enter" if i % 2 == 0 else "exit"
        out.append(
            factories[i % cameras].make(
                Transition(kind, f"zone_{i % zones}", i % 50, "pallet_type_a")  # type: ignore[arg-type]
            )
        )
    return out


def test_batching_beats_row_by_row(db_url: str) -> None:
    events = many_events(2000)
    sink = PostgresSink(db_url)
    start = time.perf_counter()
    sink.write(events[:1000])
    batched = 1000 / (time.perf_counter() - start)

    start = time.perf_counter()
    for e in events[1000:1200]:
        sink.write([e])
    single = 200 / (time.perf_counter() - start)
    sink.close()
    print(f"postgres sink: {batched:,.0f} events/s batched vs {single:,.0f} events/s row by row")
    assert batched > 1000  # far above real load; catches accidental per-row behaviour
    assert batched > 2 * single


def test_dashboard_queries_use_indexes(db_url: str) -> None:
    sink = PostgresSink(db_url)
    sink.write(many_events(30000, cameras=20, zones=10))
    sink.close()
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("ANALYZE events")
        by_camera = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN SELECT * FROM events WHERE camera_id = 'cam03' "
                "AND device_ts >= %s AND device_ts < %s",
                (T0, T0 + timedelta(hours=1)),
            )
        )
        by_zone = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN SELECT * FROM events WHERE zone_id = 'zone_4' "
                "AND device_ts >= %s AND device_ts < %s",
                (T0 + timedelta(minutes=30), T0 + timedelta(hours=1)),
            )
        )
    for plan in (by_camera, by_zone):
        assert "Index" in plan, plan
        assert "Seq Scan on events_" not in plan, plan
