import json
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from palletvision_demo.app import create_app
from palletvision_demo.settings import Settings

SCHEMA = Path(__file__).resolve().parents[3] / "contracts" / "events.schema.json"
ZONE_1 = {"id": "zone_1", "polygon": [[100, 100], [500, 120], [480, 400], [80, 380]]}  # slanted
ZONE_2 = {"id": "zone_2", "polygon": [[700, 100], [1000, 100], [1000, 400], [700, 400]]}


def box_at(x: float, y: float) -> list[float]:
    """A 60x40 box whose ground point (bottom-centre) is (x, y)."""
    return [x - 30, y - 40, x + 30, y]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        camera_id="camA",
        device_id="demo-1",
        jsonl_path=tmp_path / "events.jsonl",
    )
    c = TestClient(create_app(settings))
    assert c.put("/api/zones", json={"zones": [ZONE_1, ZONE_2]}).status_code == 200
    return c


def add(client: TestClient, x: float, y: float, kind: str = "pallet_type_a") -> dict[str, Any]:
    res = client.post("/api/objects", json={"object_type": kind, "box": box_at(x, y)})
    assert res.status_code == 200, res.text
    return res.json()  # type: ignore[no-any-return]


def move(client: TestClient, oid: int, x: float, y: float) -> dict[str, Any]:
    res = client.put(f"/api/objects/{oid}", json={"box": box_at(x, y)})
    assert res.status_code == 200, res.text
    return res.json()  # type: ignore[no-any-return]


def summary(body: dict[str, Any]) -> list[tuple[str, str]]:
    return [(e["event_type"], e["zone_id"]) for e in body["events"]]


def test_adding_an_object_outside_all_zones_logs_nothing(client: TestClient) -> None:
    body = add(client, 600, 500)
    assert body["events"] == []
    assert body["object"]["zones"] == []


def test_adding_an_object_inside_a_zone_logs_a_join(client: TestClient) -> None:
    body = add(client, 300, 250, "pallet_type_b")
    assert summary(body) == [("enter", "zone_1")]
    event = body["events"][0]
    assert event["object_type"] == "pallet_type_b"
    assert event["message"] == "object of type pallet_type_b joined zone zone_1"
    assert body["object"]["zones"] == ["zone_1"]


def test_dragging_out_logs_a_leave(client: TestClient) -> None:
    oid = add(client, 300, 250)["object"]["id"]
    body = move(client, oid, 600, 500)
    assert summary(body) == [("exit", "zone_1")]
    assert body["events"][0]["message"] == "object of type pallet_type_a left zone zone_1"


def test_dragging_between_zones_logs_leave_then_join(client: TestClient) -> None:
    oid = add(client, 300, 250)["object"]["id"]
    assert summary(move(client, oid, 850, 300)) == [("exit", "zone_1"), ("enter", "zone_2")]


def test_staying_inside_logs_nothing(client: TestClient) -> None:
    oid = add(client, 300, 250)["object"]["id"]
    assert move(client, oid, 310, 260)["events"] == []


def test_deleting_an_object_inside_a_zone_logs_a_leave(client: TestClient) -> None:
    oid = add(client, 300, 250)["object"]["id"]
    res = client.delete(f"/api/objects/{oid}")
    assert [(e["event_type"], e["zone_id"]) for e in res.json()["events"]] == [("exit", "zone_1")]
    assert client.get("/api/state").json()["objects"] == []


def test_state_lists_objects_with_their_current_zones(client: TestClient) -> None:
    a = add(client, 300, 250)["object"]["id"]
    b = add(client, 850, 300, "pallet_type_c")["object"]["id"]
    objects = {o["id"]: o for o in client.get("/api/state").json()["objects"]}
    assert objects[a]["zones"] == ["zone_1"]
    assert objects[b]["zones"] == ["zone_2"]
    assert objects[b]["object_type"] == "pallet_type_c"


def test_drawing_a_zone_around_an_object_logs_a_join(client: TestClient) -> None:
    add(client, 600, 600)
    zone_3 = {"id": "zone_3", "polygon": [[500, 500], [700, 500], [700, 700], [500, 700]]}
    res = client.put("/api/zones", json={"zones": [ZONE_1, ZONE_2, zone_3]})
    assert [(e["event_type"], e["zone_id"]) for e in res.json()["events"]] == [("enter", "zone_3")]


def test_redrawing_a_zone_away_from_an_object_logs_a_leave(client: TestClient) -> None:
    add(client, 300, 250)
    moved = {"id": "zone_1", "polygon": [[0, 600], [100, 600], [100, 700], [0, 700]]}
    res = client.put("/api/zones", json={"zones": [moved, ZONE_2]})
    assert [(e["event_type"], e["zone_id"]) for e in res.json()["events"]] == [("exit", "zone_1")]


def test_deleting_a_zone_logs_nothing(client: TestClient) -> None:
    add(client, 300, 250)
    assert client.put("/api/zones", json={"zones": [ZONE_2]}).json()["events"] == []


def test_events_are_schema_valid_ordered_and_written_to_jsonl(
    client: TestClient, tmp_path: Path
) -> None:
    oid = add(client, 300, 250)["object"]["id"]
    move(client, oid, 850, 300)
    lines = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    validator = Draft202012Validator(json.loads(SCHEMA.read_text()))
    for line in lines:
        validator.validate(line)
    assert [line["seq"] for line in lines] == [0, 1, 2]
    assert {line["device_id"] for line in lines} == {"demo-1"}
    assert {line["source"] for line in lines} == {"mock"}
    assert {line["config_version"] for line in lines} == {2}  # zones saved once, from version 1


def test_seq_continues_across_restarts(client: TestClient, tmp_path: Path) -> None:
    add(client, 300, 250)  # seq 0
    again = TestClient(
        create_app(
            Settings(
                camera_id="camA",
                device_id="demo-1",
                jsonl_path=tmp_path / "events.jsonl",
            )
        )
    )
    again.put("/api/zones", json={"zones": [ZONE_1, ZONE_2]})  # zones are per session here
    body = add(again, 300, 250)
    assert body["events"][0]["seq"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"object_type": "forklift", "box": [0, 0, 10, 10]},
        {"object_type": "pallet_type_a", "box": [0, 0, 10]},
        {"object_type": "pallet_type_a", "box": [10, 10, 5, 20]},
        {"object_type": "pallet_type_a", "box": [0, 0, 10, 10, 5]},
    ],
)
def test_invalid_objects_are_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/api/objects", json=payload).status_code == 422


def test_unknown_object_returns_404(client: TestClient) -> None:
    assert client.put("/api/objects/99", json={"box": box_at(1, 1)}).status_code == 404
    assert client.delete("/api/objects/99").status_code == 404


def test_changing_type_uses_the_new_type_in_later_events(client: TestClient) -> None:
    oid = add(client, 300, 250, "pallet_type_a")["object"]["id"]
    client.put(f"/api/objects/{oid}", json={"object_type": "pallet_type_c"})
    body = move(client, oid, 600, 500)
    assert body["events"][0]["object_type"] == "pallet_type_c"


def test_postgres_sink_writes_rows(tmp_path: Path, db_url: str) -> None:
    c = TestClient(create_app(Settings(camera_id="camA", sink="postgres", db_url=db_url)))
    c.put("/api/zones", json={"zones": [ZONE_1, ZONE_2]})
    oid = add(c, 300, 250)["object"]["id"]
    move(c, oid, 850, 300)
    with psycopg.connect(db_url) as conn:
        rows = conn.execute(
            "SELECT event_type, zone_id, object_type, source FROM events ORDER BY seq"
        ).fetchall()
    assert rows == [
        ("enter", "zone_1", "pallet_type_a", "mock"),
        ("exit", "zone_1", "pallet_type_a", "mock"),
        ("enter", "zone_2", "pallet_type_a", "mock"),
    ]


def test_unreachable_database_does_not_break_dragging(tmp_path: Path) -> None:
    c = TestClient(
        create_app(
            Settings(
                camera_id="camA",
                sink="postgres",
                db_url="postgresql://u:p@127.0.0.1:1/none",
            )
        )
    )
    c.put("/api/zones", json={"zones": [ZONE_1]})
    body = add(c, 300, 250)
    assert summary(body) == [("enter", "zone_1")]  # the user still sees the event
    assert body["sink_error"] is not None
    assert "OperationalError" in body["sink_error"]
