from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.frame import encode_jpeg, load_reference_frame, placeholder_frame
from palletvision_demo.settings import Settings
from palletvision_events.zone_store import ZoneConfigStore
from palletvision_zones.config import Zone, ZoneConfig


def test_placeholder_frame_is_a_usable_image() -> None:
    img = placeholder_frame()
    assert img.shape == (720, 1280, 3)
    assert img.std() > 0  # not a flat colour


def test_loads_image_file(tmp_path: Path) -> None:
    path = tmp_path / "ref.png"
    cv2.imwrite(str(path), np.full((50, 80, 3), 200, np.uint8))
    assert load_reference_frame(str(path)).shape == (50, 80, 3)


def test_loads_first_frame_of_video(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.release()
    assert load_reference_frame(str(path)).shape == (48, 64, 3)


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_reference_frame(str(tmp_path / "nope.png"))


def test_encode_jpeg_returns_jpeg_bytes() -> None:
    assert encode_jpeg(placeholder_frame())[:3] == b"\xff\xd8\xff"


def client(tmp_path: Path, **kw: object) -> TestClient:
    return TestClient(create_app(Settings(**kw)))  # type: ignore[arg-type]


def test_frame_endpoint_serves_jpeg(tmp_path: Path) -> None:
    res = client(tmp_path).get("/api/frame")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content[:3] == b"\xff\xd8\xff"


def test_state_starts_empty(tmp_path: Path) -> None:
    state = client(tmp_path, camera_id="camA", device_id="devB").get("/api/state").json()
    assert state["camera_id"] == "camA"
    assert state["device_id"] == "devB"
    assert state["image_size"] == [1280, 720]
    assert state["zones"] == []
    assert state["objects"] == []
    assert state["object_types"] == [
        "pallet_type_a",
        "pallet_type_b",
        "pallet_type_c",
        "pallet_unknown",
    ]
    assert state["sink"] == "jsonl"


def test_state_loads_the_latest_saved_zones_from_the_database(tmp_path: Path, db_url: str) -> None:
    store = ZoneConfigStore(db_url)
    store.save(ZoneConfig("camA", (1280, 720), (Zone("old", ((0, 0), (9, 0), (9, 9))),), version=2))
    cfg = ZoneConfig("camA", (1280, 720), (Zone("z1", ((0, 0), (100, 0), (100, 100))),), version=4)
    store.save(cfg)
    store.close()
    state = (
        client(tmp_path, camera_id="camA", sink="postgres", db_url=db_url).get("/api/state").json()
    )
    assert state["version"] == 4
    assert state["zones"] == [
        {"id": "z1", "polygon": [[0, 0], [100, 0], [100, 100]], "min_dwell_s": 0}
    ]
    assert state["warnings"] == []


def test_state_warns_when_saved_zones_do_not_match_frame_size(tmp_path: Path, db_url: str) -> None:
    store = ZoneConfigStore(db_url)
    store.save(ZoneConfig("camA", (640, 480), (Zone("z1", ((0, 0), (10, 0), (10, 10))),)))
    store.close()
    state = (
        client(tmp_path, camera_id="camA", sink="postgres", db_url=db_url).get("/api/state").json()
    )
    assert any("640" in w for w in state["warnings"])


def test_state_says_zones_are_not_saved_without_postgres(tmp_path: Path) -> None:
    state = client(tmp_path).get("/api/state").json()
    assert any("memory" in w and "--sink postgres" in w for w in state["warnings"])


def test_state_warns_when_the_database_is_unreachable(tmp_path: Path) -> None:
    state = (
        client(tmp_path, sink="postgres", db_url="postgresql://u:p@127.0.0.1:1/none")
        .get("/api/state")
        .json()
    )
    assert state["zones"] == []
    assert any("not reachable" in w for w in state["warnings"])
