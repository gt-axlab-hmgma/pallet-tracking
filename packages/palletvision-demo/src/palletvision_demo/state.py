import math
import threading
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from palletvision_demo.frame import encode_jpeg, load_reference_frame
from palletvision_demo.settings import Settings
from palletvision_demo.sinks import build_sink
from palletvision_demo.stream import EventBus
from palletvision_events.record import EventFactory, describe
from palletvision_zones.config import (
    ConfigError,
    ZoneConfig,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from palletvision_zones.engine import Track, Transition, ZoneEngine
from palletvision_zones.geometry import PolygonError

DEFAULT_OBJECT_TYPES = ["pallet_type_a", "pallet_type_b", "pallet_type_c", "pallet_unknown"]


class ZoneInputError(ValueError):
    """The submitted zones are invalid; the message is safe to show to the user."""


class ObjectInputError(ValueError):
    """The submitted object is invalid; the message is safe to show to the user."""


class UnknownObjectError(KeyError):
    pass


Box = tuple[float, float, float, float]


@dataclass
class MockObject:
    id: int
    object_type: str
    box: Box


@dataclass
class Result:
    """What one user action caused: the events it produced and whether storing them failed."""

    events: list[dict[str, Any]]
    sink_error: str | None = None


class DemoState:
    """Everything the demo API holds in memory for one camera."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        frame = load_reference_frame(settings.source)
        self.frame_jpeg = encode_jpeg(frame)
        self.image_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.warnings: list[str] = []
        self.object_types = list(DEFAULT_OBJECT_TYPES)
        self.config = self._load_or_empty()
        self.lock = threading.RLock()
        self.engine = ZoneEngine(self.config.zones, settings.hysteresis_frames)
        self.objects: dict[int, MockObject] = {}
        self.bus = EventBus()
        self.log: deque[dict[str, Any]] = deque(maxlen=500)
        self._next_id = 1
        self.sink, start_seq, sink_warnings = build_sink(settings)
        self.warnings.extend(sink_warnings)
        self.factory = EventFactory(
            settings.device_id,
            settings.camera_id,
            "mock",
            config_version=self.config.version,
            start_seq=start_seq,
        )

    @property
    def config_path(self) -> Path:
        return self.settings.config_dir / f"{self.settings.camera_id}.yaml"

    def _load_or_empty(self) -> ZoneConfig:
        if self.config_path.exists():
            config = load_config(self.config_path)
            if config.image_size != self.image_size:
                self.warnings.append(
                    f"saved zones were drawn on a {config.image_size[0]}x{config.image_size[1]} "
                    f"frame but this frame is {self.image_size[0]}x{self.image_size[1]}"
                )
            return config
        return ZoneConfig(self.settings.camera_id, self.image_size, ())

    def replace_zones(self, raw_zones: list[dict[str, Any]]) -> Result:
        """Validate and save a new zone list (new config version). Raises ZoneInputError."""
        with self.lock:
            candidate = {
                "camera_id": self.settings.camera_id,
                "image_size": list(self.image_size),
                "version": self.config.version + 1,
                "zones": raw_zones,
            }
            try:
                config = config_from_dict(candidate)
            except (ConfigError, PolygonError) as exc:
                raise ZoneInputError(str(exc)) from exc
            to_save = replace(config, reference_frame=self.config.reference_frame)
            save_config(to_save, self.config_path)
            self.config = config
            self.factory.config_version = config.version
            self.engine.set_zones(config.zones)
            return self._emit(self.engine.update(self._tracks()))

    def _tracks(self) -> list[Track]:
        return [Track(o.id, o.object_type, o.box) for o in self.objects.values()]

    def _emit(self, transitions: list[Transition]) -> Result:
        events = [self.factory.make(t) for t in transitions]
        sink_error = self.sink.write(events) if events else None
        dicts = [{**e.to_dict(), "message": describe(e)} for e in events]
        for d in dicts:
            self.log.append(d)
            self.bus.publish(d)
        return Result(dicts, sink_error)

    def recent_events(self, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.log)[-limit:] if limit > 0 else []

    def _check_type(self, object_type: str) -> None:
        if object_type not in self.object_types:
            raise ObjectInputError(f"object_type must be one of {', '.join(self.object_types)}")

    @staticmethod
    def _check_box(raw: list[float]) -> Box:
        if len(raw) != 4 or not all(math.isfinite(v) for v in raw):
            raise ObjectInputError("box must be four finite numbers [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(v) for v in raw)
        if x2 <= x1 or y2 <= y1:
            raise ObjectInputError("box must have x2 > x1 and y2 > y1")
        return (x1, y1, x2, y2)

    def add_object(self, object_type: str, raw_box: list[float]) -> tuple[MockObject, Result]:
        with self.lock:
            self._check_type(object_type)
            obj = MockObject(self._next_id, object_type, self._check_box(raw_box))
            self._next_id += 1
            self.objects[obj.id] = obj
            return obj, self._emit(self.engine.update(self._tracks()))

    def update_object(
        self, object_id: int, raw_box: list[float] | None, object_type: str | None
    ) -> tuple[MockObject, Result]:
        with self.lock:
            obj = self.objects.get(object_id)
            if obj is None:
                raise UnknownObjectError(object_id)
            if object_type is not None:
                self._check_type(object_type)
                obj.object_type = object_type
            if raw_box is not None:
                obj.box = self._check_box(raw_box)
            return obj, self._emit(self.engine.update(self._tracks()))

    def delete_object(self, object_id: int) -> Result:
        with self.lock:
            if self.objects.pop(object_id, None) is None:
                raise UnknownObjectError(object_id)
            return self._emit(self.engine.remove_track(object_id))

    def object_dict(self, obj: MockObject) -> dict[str, Any]:
        return {
            "id": obj.id,
            "object_type": obj.object_type,
            "box": list(obj.box),
            "zones": self.engine.zones_of(obj.id),
        }

    def snapshot(self) -> dict[str, Any]:
        cfg = config_to_dict(self.config)
        return {
            "camera_id": self.settings.camera_id,
            "device_id": self.settings.device_id,
            "image_size": list(self.image_size),
            "version": self.config.version,
            "zones": cfg["zones"],
            "objects": [self.object_dict(o) for o in self.objects.values()],
            "object_types": self.object_types,
            "sink": self.settings.sink,
            "warnings": self.warnings,
        }
