from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from palletvision_demo.app import create_app
from palletvision_demo.live.runner import camera_label, parse_camera, redact, source_kind
from palletvision_demo.live.state import DemoState
from palletvision_demo.settings import Settings
from palletvision_infer.detections import Detections
from palletvision_zones.engine import Track

RIGHT_HALF = [{"id": "right", "polygon": [[160, 0], [320, 0], [320, 240], [160, 240]]}]


def make_video(path: Path, frames: int = 12, size: tuple[int, int] = (320, 240)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 10, np.uint8))
    writer.release()
    return path


class Scripted:
    """Reports one pallet that moves 12 px right per call, whatever the image shows.

    It is 30 px wide, so consecutive boxes overlap enough (IoU 0.43) for the tracker to follow it.
    """

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> Detections:
        cx = 40 + 12 * self.calls
        self.calls += 1
        box = np.array([[cx - 15, 100, cx + 15, 150]], np.float32)
        return Detections(box, np.array([0.9], np.float32), np.array([0], np.int64))


def settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(
        weights=tmp_path / "unused.onnx",
        jsonl_path=tmp_path / "events.jsonl",
        upload_dir=tmp_path / "uploads",
        fps=1000.0,  # no waiting between frames
        loop=False,
        **kw,  # type: ignore[arg-type]
    )


def finish(app: FastAPI) -> None:
    """Wait for the player to reach the end of its video."""
    thread = app.state.player._thread
    assert thread is not None
    thread.join(timeout=20)


def upload(c: TestClient, path: Path, name: str | None = None) -> Response:
    res: Response = c.put(
        "/api/source/video", params={"name": name or path.name}, content=path.read_bytes()
    )
    return res


def test_uploading_a_video_plays_it_and_logs_the_zone_crossing(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path), Scripted())
    state: DemoState = app.state.demo
    state.replace_zones(RIGHT_HALF)
    c = TestClient(app)
    res = c.put(
        "/api/source/video",
        params={"name": "clip.mp4"},
        content=make_video(tmp_path / "a.mp4").read_bytes(),
    )
    assert res.status_code == 200
    assert res.json()["source"]["kind"] == "video"
    assert res.json()["source"]["name"] == "clip.mp4"
    finish(app)
    events = state.recent_events(10)
    assert [(e["event_type"], e["zone_id"], e["track_id"]) for e in events] == [
        ("enter", "right", 1)
    ]
    assert events[0]["source"] == "model"
    snap = state.snapshot()
    assert snap["source"]["state"] == "ended"
    assert snap["objects"][0]["zones"] == ["right"]
    assert (tmp_path / "uploads" / "clip.mp4").exists()


def test_a_video_given_at_start_up_plays_without_the_ui(tmp_path: Path) -> None:
    video = make_video(tmp_path / "start.mp4")
    app = create_app(settings(tmp_path, source=str(video)), Scripted())
    app.state.demo.replace_zones(RIGHT_HALF)
    with TestClient(app):
        finish(app)
    assert [e["event_type"] for e in app.state.demo.recent_events(10)] == ["enter"]
    assert app.state.demo.snapshot()["source"]["name"] == "start.mp4"


def test_frames_are_published_and_the_reference_frame_follows_the_source(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path), Scripted())
    state: DemoState = app.state.demo
    assert state.snapshot()["image_size"] == [1280, 720]  # placeholder
    version = state.frame_version
    c = TestClient(app)
    upload(c, make_video(tmp_path / "a.mp4", size=(320, 240)))
    finish(app)
    assert state.snapshot()["image_size"] == [320, 240]
    assert state.frame_version > version
    seq, jpeg = state.wait_frame(0, timeout=0.1)
    assert seq >= 12
    assert jpeg is not None and jpeg[:3] == b"\xff\xd8\xff"
    assert c.get("/api/frame").content[:3] == b"\xff\xd8\xff"


def test_switching_sources_resets_tracks_and_warns_about_zones_on_another_size(
    tmp_path: Path,
) -> None:
    app = create_app(settings(tmp_path), Scripted())
    state: DemoState = app.state.demo
    c = TestClient(app)
    upload(c, make_video(tmp_path / "a.mp4", size=(320, 240)))
    finish(app)
    state.replace_zones(RIGHT_HALF)  # drawn on 320x240
    upload(c, make_video(tmp_path / "b.mp4", size=(160, 120)))
    finish(app)
    snap = state.snapshot()
    assert snap["image_size"] == [160, 120]
    assert snap["source"]["name"] == "b.mp4"
    assert any("320x240" in w and "160x120" in w for w in snap["warnings"])
    exits = [e for e in state.recent_events(20) if e["event_type"] == "exit"]
    assert len(exits) == 1  # the first video's pallet left when the source changed


def test_eject_stops_playing_and_exits_open_visits(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path), Scripted())
    state: DemoState = app.state.demo
    state.replace_zones(RIGHT_HALF)
    c = TestClient(app)
    upload(c, make_video(tmp_path / "a.mp4"))
    finish(app)
    res = c.delete("/api/source")
    assert res.json()["source"] is None
    assert res.json()["objects"] == []
    assert [e["event_type"] for e in state.recent_events(10)] == ["enter", "exit"]


def test_everything_needs_a_model(tmp_path: Path) -> None:
    c = TestClient(create_app(Settings(jsonl_path=tmp_path / "e.jsonl")))
    video = make_video(tmp_path / "a.mp4")
    for res in (
        upload(c, video),
        c.post("/api/source/camera", json={"device": "0"}),
        c.delete("/api/source"),
    ):
        assert res.status_code == 409
        assert "--weights" in res.json()["detail"]


@pytest.mark.parametrize("name", ["notes.txt", "clip", ".hidden.mp4", ""])
def test_a_file_that_is_not_a_video_name_is_refused(tmp_path: Path, name: str) -> None:
    c = TestClient(create_app(settings(tmp_path), Scripted()))
    res = c.put("/api/source/video", params={"name": name}, content=b"x")
    assert res.status_code == 422


def test_an_empty_upload_is_refused_and_leaves_nothing_behind(tmp_path: Path) -> None:
    c = TestClient(create_app(settings(tmp_path), Scripted()))
    res = c.put("/api/source/video", params={"name": "a.mp4"}, content=b"")
    assert res.status_code == 422
    assert list((tmp_path / "uploads").iterdir()) == []


def test_a_file_that_is_not_a_readable_video_is_refused(tmp_path: Path) -> None:
    c = TestClient(create_app(settings(tmp_path), Scripted()))
    res = c.put("/api/source/video", params={"name": "fake.mp4"}, content=b"not a video")
    assert res.status_code == 422
    assert "cannot play" in res.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []  # the bad file is not kept


def test_an_upload_cut_off_halfway_leaves_no_partial_file(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path), Scripted())
    c = TestClient(app)

    def body() -> Iterator[bytes]:
        yield b"x" * 1000
        raise ConnectionError("browser went away")

    with pytest.raises(ConnectionError):
        c.put("/api/source/video", params={"name": "a.mp4"}, content=body())
    assert list((tmp_path / "uploads").iterdir()) == []


def test_an_upload_name_cannot_escape_the_upload_folder(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path), Scripted())
    c = TestClient(app)
    upload(c, make_video(tmp_path / "a.mp4"), name="../../evil.mp4")
    finish(app)
    assert (tmp_path / "uploads" / "evil.mp4").exists()
    assert not (tmp_path.parent / "evil.mp4").exists()


@pytest.mark.parametrize("device", ["", "/etc/passwd", "file:///etc/passwd", "movie.mp4", "-1"])
def test_a_camera_must_be_an_index_or_a_stream_url(tmp_path: Path, device: str) -> None:
    c = TestClient(create_app(settings(tmp_path), Scripted()))
    res = c.post("/api/source/camera", json={"device": device})
    assert res.status_code == 422
    assert "index" in res.json()["detail"]


def test_a_camera_that_cannot_be_opened_is_reported(tmp_path: Path) -> None:
    c = TestClient(create_app(settings(tmp_path), Scripted()))
    res = c.post("/api/source/camera", json={"device": "97"})
    assert res.status_code == 422
    assert "camera 97" in res.json()["detail"]
    assert c.get("/api/state").json()["source"] is None  # nothing was replaced


def test_parse_camera_and_labels() -> None:
    assert parse_camera("0") == 0
    assert parse_camera(" 2 ") == 2
    assert parse_camera("rtsp://cam.local/stream") == "rtsp://cam.local/stream"
    assert camera_label(1) == "camera 1"
    assert source_kind("0") == "camera"
    assert source_kind("rtsp://h/x") == "camera"
    assert source_kind("clip.mp4") == "video"
    with pytest.raises(ValueError, match="index"):
        parse_camera("/tmp/clip.mp4")


def test_stream_passwords_never_reach_the_ui() -> None:
    url = "rtsp://admin:hunter2@10.0.0.5:554/live"
    assert camera_label(url) == "rtsp://10.0.0.5:554/live"
    assert "hunter2" not in redact(f"cannot open video source: {url}")


def test_a_failing_detector_shows_as_a_source_error(tmp_path: Path) -> None:
    class Broken:
        def detect(self, image: np.ndarray) -> Detections:
            raise RuntimeError("model blew up")

    app = create_app(settings(tmp_path), Broken())
    c = TestClient(app)
    upload(c, make_video(tmp_path / "a.mp4"))
    finish(app)
    source = app.state.demo.snapshot()["source"]
    assert source["state"] == "error"
    assert "model blew up" in source["message"]


def test_wait_frame_times_out_when_nothing_new(tmp_path: Path) -> None:
    state = DemoState(Settings(jsonl_path=tmp_path / "e.jsonl"))
    assert state.wait_frame(0, timeout=0.01) == (0, None)
    state.publish_frame(b"x")
    assert state.wait_frame(0, timeout=0.01) == (1, b"x")
    assert state.wait_frame(1, timeout=0.01) == (1, b"x")  # nothing newer: returns after waiting


def test_a_track_the_tracker_gives_up_on_exits_its_zones(tmp_path: Path) -> None:
    state = DemoState(Settings(jsonl_path=tmp_path / "e.jsonl"))
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
    state = DemoState(Settings(jsonl_path=tmp_path / "e.jsonl"))
    state.replace_zones(RIGHT_HALF)
    inside = (200.0, 100.0, 230.0, 150.0)
    state.apply_tracks([Track(1, "pallet_unknown", inside), Track(2, "pallet_unknown", inside)], [])
    cleared = state.clear_tracks()
    assert sorted(e["track_id"] for e in cleared.events) == [1, 2]
    assert state.snapshot()["objects"] == []
