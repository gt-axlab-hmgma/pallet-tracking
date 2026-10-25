import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from palletvision_demo.live.frame import encode_jpeg, load_reference_frame
from palletvision_demo.live.stream import EventBus
from palletvision_demo.settings import Settings
from palletvision_demo.sinks import build_config_store, build_sink
from palletvision_events.record import EventFactory, describe
from palletvision_zones.config import (
    ConfigError,
    ZoneConfig,
    config_from_dict,
    config_to_dict,
)
from palletvision_zones.engine import Track, Transition, ZoneEngine
from palletvision_zones.geometry import PolygonError

# Stage 1 has one pallet class and the tracker ignores classes, so every track is the same type.
OBJECT_TYPE = "pallet_unknown"


class ZoneInputError(ValueError):
    """The submitted zones are invalid; the message is safe to show to the user."""


Box = tuple[float, float, float, float]


@dataclass
class TrackedBox:
    id: int
    box: Box


@dataclass
class Result:
    """What one action caused: the events it produced and whether storing them failed."""

    events: list[dict[str, Any]]
    sink_error: str | None = None
    config_error: str | None = None  # the zone version could not be stored in the database


class DemoState:
    """Everything the demo API holds in memory for one camera."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = threading.RLock()
        self.warnings: list[str] = []
        self.objects: dict[int, TrackedBox] = {}  # tracks seen in the latest frame
        self.bus = EventBus()
        self._known_ids: set[int] = set()  # track ids the zone engine holds state for
        self.source: dict[str, Any] | None = None  # what is playing, for the UI
        self._frame_cond = threading.Condition()
        self._live_jpeg: bytes | None = None
        self._live_seq = 0
        self.log: deque[dict[str, Any]] = deque(maxlen=500)
        self.frame_version = 0
        self.frame_jpeg = b""
        self.image_size = (0, 0)
        self.reference_image: np.ndarray | None = (
            None  # the real first frame, never the placeholder
        )
        self.set_reference(load_reference_frame(None))  # a placeholder until a source is loaded
        self.reference_image = None
        self.sink, start_seq, sink_warnings = build_sink(settings)
        self.config_store = build_config_store(settings)
        self.warnings.extend(sink_warnings)
        self.config = self._load_or_empty()
        self.engine = ZoneEngine(self.config.zones, settings.hysteresis_frames)
        self.factory = EventFactory(
            settings.device_id,
            settings.camera_id,
            "model",
            config_version=self.config.version,
            start_seq=start_seq,
        )

    def _load_or_empty(self) -> ZoneConfig:
        """Zones live in the database. Without it they exist for this session only."""
        empty = ZoneConfig(self.settings.camera_id, self.image_size, ())
        if self.config_store is None:
            self.warnings.append(
                "Zones are kept in memory and are lost when the API stops. "
                "Start with --sink postgres to save them in the database."
            )
            return empty
        config, error = self.config_store.load_latest(self.settings.camera_id)
        if error:
            self.warnings.append(
                "Postgres is not reachable, so saved zones could not be loaded and new zones "
                "will not be saved until it is."
            )
            return empty
        return config or empty

    def set_reference(self, frame: np.ndarray) -> None:
        """The frame zones are drawn on: a placeholder, or the first frame of the source.

        The raw frame is kept as well as the JPEG: zones are proposed from the raw pixels, because
        JPEG noise moves the corners that are extrapolated from a short piece of paint.
        """
        with self.lock:
            self.reference_image = np.array(frame, copy=True)
            self.frame_jpeg = encode_jpeg(frame)
            self.image_size = (int(frame.shape[1]), int(frame.shape[0]))
            self.frame_version += 1

    def set_source(self, kind: str, name: str, state: str, message: str | None = None) -> None:
        """What the UI shows as the current source: ``state`` is playing, ended or error."""
        with self.lock:
            self.source = {"kind": kind, "name": name, "state": state, "message": message}

    def clear_source(self) -> None:
        with self.lock:
            self.source = None

    def replace_zones(self, raw_zones: list[dict[str, Any]]) -> Result:
        """Validate and save a new zone list (new config version). Raises ZoneInputError."""
        with self.lock:
            # Continue numbering from the database so a version never collides with an old one.
            stored = (
                self.config_store.latest_version(self.settings.camera_id)
                if self.config_store
                else 0
            )
            candidate = {
                "camera_id": self.settings.camera_id,
                "image_size": list(self.image_size),
                "version": max(self.config.version, stored) + 1,
                "zones": raw_zones,
            }
            try:
                config = config_from_dict(candidate)
            except (ConfigError, PolygonError) as exc:
                raise ZoneInputError(str(exc)) from exc
            self.config = config
            self.factory.config_version = config.version
            self.engine.set_zones(config.zones)
            return self._emit(self.engine.update(self._tracks()), store_config=True)

    def _tracks(self) -> list[Track]:
        return [Track(o.id, OBJECT_TYPE, o.box) for o in self.objects.values()]

    def _emit(self, transitions: list[Transition], store_config: bool = False) -> Result:
        events = [self.factory.make(t) for t in transitions]
        # The zone version an event refers to must be in the database before the event is.
        config_error = None
        if self.config_store and (events or store_config):
            config_error = self.config_store.ensure(self.config)
        sink_error = self.sink.write(events) if events else None
        dicts = [{**e.to_dict(), "message": describe(e)} for e in events]
        for d in dicts:
            self.log.append(d)
            self.bus.publish(d)
        return Result(dicts, sink_error, config_error)

    def apply_tracks(self, tracks: list[Track], removed_ids: list[int]) -> Result:
        """One frame of tracker output: the tracks seen now, and the ids it gave up on.

        Only tracks seen this frame move through the zones. A hidden track keeps the zones it was
        last confirmed in, and leaves them (an exit event) only once the tracker removes it.
        """
        with self.lock:
            self.objects = {t.track_id: TrackedBox(t.track_id, t.box) for t in tracks}
            transitions = self.engine.update(tracks)
            self._known_ids.update(t.track_id for t in tracks)
            for track_id in removed_ids:
                if track_id in self._known_ids:
                    transitions += self.engine.remove_track(track_id)
                    self._known_ids.discard(track_id)
            return self._emit(transitions)

    def clear_tracks(self) -> Result:
        """Forget every track, as when the video restarts. Objects still inside a zone exit."""
        with self.lock:
            transitions: list[Transition] = []
            for track_id in sorted(self._known_ids):
                transitions += self.engine.remove_track(track_id)
            self._known_ids.clear()
            self.objects = {}
            return self._emit(transitions)

    def publish_frame(self, jpeg: bytes) -> None:
        with self._frame_cond:
            self._live_jpeg = jpeg
            self._live_seq += 1
            self._frame_cond.notify_all()

    def wait_frame(self, after_seq: int, timeout: float) -> tuple[int, bytes | None]:
        """Block until a frame newer than ``after_seq`` is published, or ``timeout`` passes."""
        with self._frame_cond:
            self._frame_cond.wait_for(lambda: self._live_seq > after_seq, timeout)
            return self._live_seq, self._live_jpeg

    def add_warning(self, text: str) -> None:
        with self.lock:
            self.warnings.append(text)

    def recent_events(self, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.log)[-limit:] if limit > 0 else []

    def object_dict(self, obj: TrackedBox) -> dict[str, Any]:
        return {"id": obj.id, "box": list(obj.box), "zones": self.engine.zones_of(obj.id)}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            cfg = config_to_dict(self.config)
            warnings = list(self.warnings)
            if self.config.zones and self.config.image_size != self.image_size:
                (zw, zh), (w, h) = self.config.image_size, self.image_size
                warnings.append(
                    f"Zones were drawn on a {zw}x{zh} frame but the current frame is {w}x{h}."
                )
            return {
                "camera_id": self.settings.camera_id,
                "device_id": self.settings.device_id,
                "image_size": list(self.image_size),
                "frame_version": self.frame_version,
                "can_play": self.settings.weights is not None,
                "source": self.source,
                "version": self.config.version,
                "zones": cfg["zones"],
                "objects": [self.object_dict(o) for o in self.objects.values()],
                "sink": self.settings.sink,
                "warnings": warnings,
            }
