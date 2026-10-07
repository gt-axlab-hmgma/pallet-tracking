import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator

from palletvision_events.record import EventFactory, ZoneEvent
from palletvision_zones.engine import Transition

SCHEMA = Path(__file__).resolve().parents[3] / "contracts" / "events.schema.json"
T0 = datetime(2026, 10, 20, 14, 3, 11, 250000, tzinfo=UTC)


def factory(start_seq: int = 0) -> EventFactory:
    return EventFactory(
        device_id="nano-01",
        camera_id="cam01",
        source="mock",
        config_version=3,
        start_seq=start_seq,
        clock=lambda: T0,
    )


def test_make_fills_fields_from_transition() -> None:
    event = factory().make(Transition("enter", "zone_1", 7, "pallet_type_a"))
    assert event.event_type == "enter"
    assert event.zone_id == "zone_1"
    assert event.track_id == 7
    assert event.object_type == "pallet_type_a"
    assert event.device_id == "nano-01"
    assert event.camera_id == "cam01"
    assert event.config_version == 3
    assert event.device_ts == T0
    uuid.UUID(str(event.event_id))


def test_seq_increments_and_ids_are_unique() -> None:
    f = factory(start_seq=10)
    events = [f.make(Transition("exit", "z", 1, "pallet")) for _ in range(3)]
    assert [e.seq for e in events] == [10, 11, 12]
    assert len({e.event_id for e in events}) == 3


def test_to_dict_matches_contract_schema() -> None:
    event = factory().make(Transition("enter", "zone_1", 7, "pallet_type_a"))
    data = event.to_dict()
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(data)
    assert data["device_ts"] == "2026-10-20T14:03:11.250Z"


def test_dict_round_trip() -> None:
    event = factory().make(Transition("exit", "zone_2", 3, "pallet_type_b"))
    assert ZoneEvent.from_dict(event.to_dict()) == event
