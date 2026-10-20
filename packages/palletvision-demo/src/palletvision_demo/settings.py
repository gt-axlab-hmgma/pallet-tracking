from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SinkKind = Literal["jsonl", "postgres", "outbox", "stdout"]


@dataclass(frozen=True)
class Settings:
    camera_id: str = "cam01"
    device_id: str = "demo-laptop"
    source: str | None = (
        None  # a video file, camera index or stream URL to start on; else pick in the UI
    )
    sink: SinkKind = "jsonl"
    jsonl_path: Path = Path("runs/demo/events.jsonl")
    outbox_path: Path = Path("runs/outbox.db")
    db_url: str | None = None
    hysteresis_frames: int = 1
    # With weights, a video or camera chosen in the UI is run through the detector and tracker
    # and the tracks drive the zones.
    weights: Path | None = None  # ONNX model
    upload_dir: Path = Path("runs/demo/uploads")  # where videos uploaded from the browser go
    fps: float = 10.0  # frames per second the detector and tracker run at
    conf: float = 0.1  # detections below this score are dropped; ByteTrack uses the low ones
    lost_seconds: float = 5.0  # how long a hidden object keeps its id
    loop: bool = True  # restart the video when it ends
