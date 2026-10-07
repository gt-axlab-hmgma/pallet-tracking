from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.frame import encode_jpeg, load_reference_frame, placeholder_frame
from palletvision_demo.settings import Settings
from palletvision_zones.config import Zone, ZoneConfig, save_config


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
    return TestClient(create_app(Settings(config_dir=tmp_path / "cams", **kw)))  # type: ignore[arg-type]


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


def test_state_loads_saved_zone_config(tmp_path: Path) -> None:
    cfg = ZoneConfig("camA", (1280, 720), (Zone("z1", ((0, 0), (100, 0), (100, 100))),), version=4)
    save_config(cfg, tmp_path / "cams" / "camA.yaml")
    state = client(tmp_path, camera_id="camA").get("/api/state").json()
    assert state["version"] == 4
    assert state["zones"] == [
        {"id": "z1", "polygon": [[0, 0], [100, 0], [100, 100]], "min_dwell_s": 0}
    ]
    assert state["warnings"] == []


def test_state_warns_when_saved_zones_do_not_match_frame_size(tmp_path: Path) -> None:
    cfg = ZoneConfig("camA", (640, 480), (Zone("z1", ((0, 0), (10, 0), (10, 10))),))
    save_config(cfg, tmp_path / "cams" / "camA.yaml")
    state = client(tmp_path, camera_id="camA").get("/api/state").json()
    assert any("640" in w for w in state["warnings"])
