from pathlib import Path
from typing import Any

import psycopg
from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.settings import Settings
from palletvision_events.zone_store import ZoneConfigStore
from palletvision_zones.config import Zone, ZoneConfig, save_config

ZONE_A = {"id": "zone_a", "polygon": [[100, 100], [500, 120], [480, 400], [80, 380]]}
ZONE_B = {"id": "zone_b", "polygon": [[700, 100], [1000, 100], [1000, 400], [700, 400]]}
BOX_IN_A = [250.0, 260.0, 330.0, 330.0]  # ground point (290, 330) is inside zone_a


def client(tmp_path: Path, db_url: str, sink: str = "postgres") -> TestClient:
    settings = Settings(
        camera_id="camA",
        device_id="demo-1",
        config_dir=tmp_path / "cams",
        jsonl_path=tmp_path / "e.jsonl",
        sink=sink,  # type: ignore[arg-type]
        db_url=db_url,
    )
    return TestClient(create_app(settings))


def rows(db_url: str, sql: str) -> list[tuple[Any, ...]]:
    with psycopg.connect(db_url) as conn:
        return conn.execute(sql).fetchall()


def test_saving_zones_stores_an_immutable_version(tmp_path: Path, db_url: str) -> None:
    c = client(tmp_path, db_url)
    res = c.put("/api/zones", json={"zones": [ZONE_A]})
    assert res.status_code == 200
    assert res.json()["config_error"] is None
    c.put("/api/zones", json={"zones": [ZONE_A, ZONE_B]})
    stored = rows(db_url, "SELECT version, saved_by, jsonb_array_length(zones) FROM zone_configs")
    assert sorted(stored) == [(2, "demo", 1), (3, "demo", 2)]  # the empty v1 is never stored


def test_events_join_to_the_polygons_they_were_computed_against(
    tmp_path: Path, db_url: str
) -> None:
    c = client(tmp_path, db_url)
    c.put("/api/zones", json={"zones": [ZONE_A]})  # version 2
    c.post("/api/objects", json={"object_type": "pallet_type_a", "box": BOX_IN_A})
    c.put("/api/zones", json={"zones": [ZONE_B]})  # version 3: zone_a is gone
    joined = rows(
        db_url,
        "SELECT e.event_type, e.zone_id, e.config_version, z.zones->0->>'id' "
        "FROM events e JOIN zone_configs z "
        "ON z.camera_id = e.camera_id AND z.version = e.config_version ORDER BY e.seq",
    )
    assert joined == [("enter", "zone_a", 2, "zone_a")]


def test_version_numbering_continues_from_the_database(tmp_path: Path, db_url: str) -> None:
    store = ZoneConfigStore(db_url, saved_by="earlier-session")
    store.save(ZoneConfig("camA", (1280, 720), (Zone("old", ((0, 0), (9, 0), (9, 9))),), version=5))
    store.close()
    c = client(tmp_path, db_url)  # no yaml on disk, so it starts at version 1 locally
    assert c.put("/api/zones", json={"zones": [ZONE_A]}).json()["version"] == 6


def test_existing_yaml_version_is_stored_before_its_events(tmp_path: Path, db_url: str) -> None:
    polygon = ((100.0, 100.0), (500.0, 120.0), (480.0, 400.0), (80.0, 380.0))
    cfg = ZoneConfig("camA", (1280, 720), (Zone("zone_a", polygon),), version=3)
    save_config(cfg, tmp_path / "cams" / "camA.yaml")
    c = client(tmp_path, db_url)
    res = c.post("/api/objects", json={"object_type": "pallet_type_a", "box": BOX_IN_A})
    assert res.json()["config_error"] is None
    assert rows(db_url, "SELECT version FROM zone_configs") == [(3,)]
    assert rows(db_url, "SELECT config_version FROM events") == [(3,)]


def test_unreachable_database_still_saves_zones_and_warns(tmp_path: Path) -> None:
    c = client(tmp_path, "postgresql://u:p@127.0.0.1:1/none")
    res = c.put("/api/zones", json={"zones": [ZONE_A]})
    assert res.status_code == 200
    assert res.json()["version"] == 2
    assert "OperationalError" in res.json()["config_error"]
    assert (tmp_path / "cams" / "camA.yaml").exists()


def test_other_sinks_do_not_touch_the_database(tmp_path: Path) -> None:
    c = client(tmp_path, "postgresql://u:p@127.0.0.1:1/none", sink="jsonl")
    res = c.put("/api/zones", json={"zones": [ZONE_A]})
    assert res.json()["config_error"] is None
    obj = c.post("/api/objects", json={"object_type": "pallet_type_a", "box": BOX_IN_A}).json()
    assert obj["config_error"] is None
