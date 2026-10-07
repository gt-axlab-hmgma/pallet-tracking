from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.settings import Settings

SLANTED = [[112, 580], [860, 540], [910, 310], [240, 320]]
CONCAVE = [[0, 0], [100, 0], [100, 40], [40, 40], [40, 100], [0, 100]]


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(camera_id="camA")))


def put_zones(client: TestClient, zones: list[dict[str, Any]]) -> Any:
    return client.put("/api/zones", json={"zones": zones})


def test_save_zones_persists_and_bumps_version(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    res = put_zones(
        client, [{"id": "slanted", "polygon": SLANTED}, {"id": "l", "polygon": CONCAVE}]
    )
    assert res.status_code == 200
    body = res.json()
    assert body["version"] == 2  # empty starting config is version 1
    assert [z["id"] for z in body["zones"]] == ["slanted", "l"]

    assert put_zones(client, [{"id": "slanted", "polygon": SLANTED}]).json()["version"] == 3
    assert client.get("/api/state").json()["zones"][0]["id"] == "slanted"


def test_zones_survive_a_restart_when_stored_in_postgres(tmp_path: Path, db_url: str) -> None:
    def start() -> TestClient:
        return TestClient(create_app(Settings(camera_id="camA", sink="postgres", db_url=db_url)))

    start().put("/api/zones", json={"zones": [{"id": "z", "polygon": SLANTED}]})
    again = start().get("/api/state").json()
    assert [z["id"] for z in again["zones"]] == ["z"]
    assert again["version"] == 2


def test_zones_are_never_written_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    put_zones(make_client(tmp_path), [{"id": "z", "polygon": SLANTED}])
    assert list(tmp_path.rglob("*")) == []


def test_empty_list_clears_zones(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    put_zones(client, [{"id": "z", "polygon": SLANTED}])
    res = put_zones(client, [])
    assert res.status_code == 200
    assert res.json()["zones"] == []


@pytest.mark.parametrize(
    ("zones", "needle"),
    [
        ([{"id": "a", "polygon": [[0, 0], [1, 1]]}], "at least 3"),
        ([{"id": "a", "polygon": [[0, 0], [10, 10], [10, 0], [0, 10]]}], "self-intersect"),
        (
            [{"id": "a", "polygon": SLANTED}, {"id": "a", "polygon": CONCAVE}],
            "unique",
        ),
        ([{"polygon": SLANTED}], "id"),
    ],
)
def test_invalid_zones_are_rejected_without_saving(
    tmp_path: Path,
    zones: list[dict[str, Any]],
    needle: str,
) -> None:
    client = make_client(tmp_path)
    res = put_zones(client, zones)
    assert res.status_code == 422
    assert needle in res.json()["detail"]
    assert client.get("/api/state").json()["version"] == 1
