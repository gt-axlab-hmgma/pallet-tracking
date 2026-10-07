import io
import json
from datetime import UTC, datetime
from pathlib import Path

from palletvision_events.record import EventFactory, describe
from palletvision_events.sinks import EventSink, JsonlSink, StdoutSink
from palletvision_zones.engine import Transition

T0 = datetime(2026, 10, 20, 14, 3, 11, tzinfo=UTC)


def events(n: int = 2):  # type: ignore[no-untyped-def]
    f = EventFactory("nano-01", "cam01", "mock", clock=lambda: T0)
    kinds = ["enter", "exit"]
    return [f.make(Transition(kinds[i % 2], "zone_1", 1, "pallet_type_a")) for i in range(n)]  # type: ignore[arg-type]


def test_describe_reads_naturally() -> None:
    enter, exit_ = events()
    assert describe(enter) == "object of type pallet_type_a joined zone zone_1"
    assert describe(exit_) == "object of type pallet_type_a left zone zone_1"


def test_jsonl_sink_appends_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    sink: EventSink = JsonlSink(path)
    batch = events(2)
    sink.write(batch[:1])
    sink.write(batch[1:])
    sink.close()
    lines = path.read_text().splitlines()
    assert [json.loads(line)["event_id"] for line in lines] == [str(e.event_id) for e in batch]


def test_jsonl_sink_appends_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    for _ in range(2):
        sink = JsonlSink(path)
        sink.write(events(1))
        sink.close()
    assert len(path.read_text().splitlines()) == 2


def test_stdout_sink_prints_human_readable_lines() -> None:
    stream = io.StringIO()
    StdoutSink(stream).write(events(2))
    out = stream.getvalue().splitlines()
    assert out[0].endswith("object of type pallet_type_a joined zone zone_1")
    assert out[1].endswith("object of type pallet_type_a left zone zone_1")
