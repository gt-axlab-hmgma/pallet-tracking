from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SinkKind = Literal["jsonl", "postgres", "outbox", "stdout"]


@dataclass(frozen=True)
class Settings:
    camera_id: str = "cam01"
    device_id: str = "demo-laptop"
    source: str | None = None  # image, video or camera index; None = generated placeholder
    config_dir: Path = Path("configs/cameras")
    sink: SinkKind = "jsonl"
    jsonl_path: Path = Path("runs/demo/events.jsonl")
    outbox_path: Path = Path("runs/outbox.db")
    db_url: str | None = None
    hysteresis_frames: int = 1
