import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


def validator(name: str) -> Draft202012Validator:
    schema = json.loads((CONTRACTS / name).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


ZONES = {
    "camera_id": "cam01",
    "version": 1,
    "image_size": [1920, 1080],
    "zones": [
        {
            "id": "zone_1_unloading",
            "polygon": [[112, 980], [860, 940], [910, 610], [240, 620]],
            "min_dwell_s": 5,
        }
    ],
}

EVENT = {
    "event_id": "8f14e45f-ceea-4672-9e4e-0a4c1a7b2b11",
    "device_id": "nano-01",
    "camera_id": "cam01",
    "seq": 0,
    "event_type": "enter",
    "zone_id": "zone_1_unloading",
    "object_type": "pallet_type_a",
    "track_id": 7,
    "config_version": 1,
    "source": "mock",
    "device_ts": "2026-10-20T14:03:11.250Z",
}


def test_valid_zone_config_passes() -> None:
    validator("zones.schema.json").validate(ZONES)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda z: z["zones"][0].update(polygon=[[0, 0], [1, 1]]),  # fewer than 3 vertices
        lambda z: z["zones"][0].update(polygon=[[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        lambda z: z.update(image_size=[1920]),
        lambda z: z.pop("camera_id"),
        lambda z: z["zones"][0].pop("id"),
    ],
)
def test_invalid_zone_configs_fail(mutate) -> None:  # type: ignore[no-untyped-def]
    bad = json.loads(json.dumps(ZONES))
    mutate(bad)
    with pytest.raises(ValidationError):
        validator("zones.schema.json").validate(bad)


def test_valid_event_passes() -> None:
    validator("events.schema.json").validate(EVENT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(event_type="moved"),
        lambda e: e.update(event_id="not-a-uuid"),
        lambda e: e.update(seq=-1),
        lambda e: e.update(source="camera"),
        lambda e: e.update(extra="field"),
        lambda e: e.pop("zone_id"),
    ],
)
def test_invalid_events_fail(mutate) -> None:  # type: ignore[no-untyped-def]
    bad = dict(EVENT)
    mutate(bad)
    with pytest.raises(ValidationError):
        validator("events.schema.json").validate(bad)
