import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from palletvision_zones.engine import Transition

Source = Literal["model", "mock"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_ts(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class ZoneEvent:
    """One enter/exit record. Mirrors contracts/events.schema.json."""

    event_id: uuid.UUID
    device_id: str
    camera_id: str
    seq: int
    event_type: Literal["enter", "exit"]
    zone_id: str
    object_type: str
    track_id: int
    source: Source
    device_ts: datetime
    config_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event_id": str(self.event_id),
            "device_id": self.device_id,
            "camera_id": self.camera_id,
            "seq": self.seq,
            "event_type": self.event_type,
            "zone_id": self.zone_id,
            "object_type": self.object_type,
            "track_id": self.track_id,
            "source": self.source,
            "device_ts": _format_ts(self.device_ts),
        }
        if self.config_version is not None:
            out["config_version"] = self.config_version
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneEvent":
        return cls(
            event_id=uuid.UUID(data["event_id"]),
            device_id=data["device_id"],
            camera_id=data["camera_id"],
            seq=int(data["seq"]),
            event_type=data["event_type"],
            zone_id=data["zone_id"],
            object_type=data["object_type"],
            track_id=int(data["track_id"]),
            source=data["source"],
            device_ts=datetime.fromisoformat(data["device_ts"].replace("Z", "+00:00")),
            config_version=data.get("config_version"),
        )


class EventFactory:
    """Stamps engine transitions with an id, a per-device sequence number and a timestamp."""

    def __init__(
        self,
        device_id: str,
        camera_id: str,
        source: Source,
        config_version: int | None = None,
        start_seq: int = 0,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.device_id = device_id
        self.camera_id = camera_id
        self.source = source
        self.config_version = config_version
        self._seq = start_seq
        self._clock = clock

    def make(self, transition: Transition) -> ZoneEvent:
        event = ZoneEvent(
            event_id=uuid.uuid4(),
            device_id=self.device_id,
            camera_id=self.camera_id,
            seq=self._seq,
            event_type=transition.event_type,
            zone_id=transition.zone_id,
            object_type=transition.object_type,
            track_id=transition.track_id,
            source=self.source,
            device_ts=self._clock(),
            config_version=self.config_version,
        )
        self._seq += 1
        return event
