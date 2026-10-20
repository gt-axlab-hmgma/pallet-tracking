from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.live.state import DemoState
from palletvision_demo.settings import Settings
from palletvision_infer.detections import Detections
from palletvision_zones.engine import Track

RIGHT_HALF = [{"id": "right", "polygon": [[160, 0], [320, 0], [320, 240], [160, 240]]}]


def make_video(path: Path, frames: int = 12) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, (320, 240))
    for i in range(frames):
        writer.write(np.full((240, 320, 3), i * 10, np.uint8))
    writer.release()
    return path


class Scripted:
    """Reports one pallet that moves 12 px right per call, whatever the image shows.

    It is 30 px wide, so consecutive boxes overlap enough (IoU 0.43) for the tracker to follow it.
    """

    def __init__(self, score: float = 0.9) -> None:
        self.calls = 0
        self.score = score

    def detect(self, image: np.ndarray) -> Detections:
        cx = 40 + 12 * self.calls
        self.calls += 1
        box = np.array([[cx - 15, 100, cx + 15, 150]], np.float32)
        return Detections(box, np.array([self.score], np.float32), np.array([0], np.int64))


def live_settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(
        source=str(make_video(tmp_path / "clip.mp4")),
        weights=tmp_path / "unused.onnx",
        jsonl_path=tmp_path / "events.jsonl",
        fps=1000.0,  # no waiting between frames
        loop=False,
        **kw,  # type: ignore[arg-type]
    )


def run_to_the_end(app_settings: Settings, detector: Scripted) -> tuple[TestClient, DemoState]:
    app = create_app(app_settings, detector)
    state: DemoState = app.state.demo
    state.replace_zones(RIGHT_HALF)
    client = TestClient(app)
    with client:
        app.state.runner._thread.join(timeout=20)
    return client, state


def test_a_tracked_pallet_driving_into_a_zone_logs_an_enter_event(tmp_path: Path) -> None:
    client, state = run_to_the_end(live_settings(tmp_path), Scripted())
    events = state.recent_events(10)
    assert [(e["event_type"], e["zone_id"], e["track_id"]) for e in events] == [
        ("enter", "right", 1)
    ]
    assert events[0]["source"] == "model"
    # the pallet's bottom centre reaches x = 160 on the 11th frame (cx 40 + 12 * 10)
    assert state.snapshot()["live"] is True
    assert state.snapshot()["objects"][0]["zones"] == ["right"]


def test_live_mode_is_off_without_weights(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(jsonl_path=tmp_path / "e.jsonl")))
    assert client.get("/api/state").json()["live"] is False


def test_live_frames_are_published(tmp_path: Path) -> None:
    _, state = run_to_the_end(live_settings(tmp_path), Scripted())
    seq, jpeg = state.wait_frame(0, timeout=0.1)
    assert seq == 12
    assert jpeg is not None and jpeg[:3] == b"\xff\xd8\xff"


def test_wait_frame_times_out_when_nothing_new(tmp_path: Path) -> None:
    state = DemoState(Settings(jsonl_path=tmp_path / "e.jsonl"))
    assert state.wait_frame(0, timeout=0.01) == (0, None)
    state.publish_frame(b"x")
    assert state.wait_frame(0, timeout=0.01) == (1, b"x")
    assert state.wait_frame(1, timeout=0.01) == (1, b"x")  # nothing newer: returns after waiting


def test_a_track_the_tracker_gives_up_on_exits_its_zones(tmp_path: Path) -> None:
    state = DemoState(Settings(jsonl_path=tmp_path / "e.jsonl", weights=tmp_path / "w.onnx"))
    state.replace_zones(RIGHT_HALF)
    box = (200.0, 100.0, 230.0, 150.0)
    entered = state.apply_tracks([Track(7, "pallet_unknown", box)], [])
    assert [e["event_type"] for e in entered.events] == ["enter"]
    hidden = state.apply_tracks([], [])  # not seen this frame: still inside
    assert hidden.events == [] and state.snapshot()["objects"] == []
    gone = state.apply_tracks([], [7])  # the tracker removed it
    assert [(e["event_type"], e["track_id"]) for e in gone.events] == [("exit", 7)]
    assert state.apply_tracks([], [7]).events == []  # reported once


def test_clear_tracks_exits_everything_inside(tmp_path: Path) -> None:
    state = DemoState(Settings(jsonl_path=tmp_path / "e.jsonl", weights=tmp_path / "w.onnx"))
    state.replace_zones(RIGHT_HALF)
    inside = (200.0, 100.0, 230.0, 150.0)
    state.apply_tracks([Track(1, "pallet_unknown", inside), Track(2, "pallet_unknown", inside)], [])
    cleared = state.clear_tracks()
    assert sorted(e["track_id"] for e in cleared.events) == [1, 2]
    assert state.snapshot()["objects"] == []


def test_a_failing_detector_is_reported_not_raised(tmp_path: Path) -> None:
    class Broken:
        def detect(self, image: np.ndarray) -> Detections:
            raise RuntimeError("model blew up")

    app = create_app(live_settings(tmp_path), Broken())
    with TestClient(app):
        app.state.runner._thread.join(timeout=20)
    assert any("model blew up" in w for w in app.state.demo.warnings)


def test_live_needs_a_source(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="--source"):
        create_app(Settings(weights=tmp_path / "w.onnx", jsonl_path=tmp_path / "e.jsonl"))
