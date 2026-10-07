from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from palletvision_zones.config import Zone
from palletvision_zones.geometry import Point, ground_point, point_in_polygon

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Track:
    track_id: int
    object_type: str
    box: Box  # xyxy in image pixels


@dataclass(frozen=True)
class Transition:
    event_type: Literal["enter", "exit"]
    zone_id: str
    track_id: int
    object_type: str


@dataclass
class _ZoneState:
    inside: bool = False  # confirmed membership
    streak: int = 0  # consecutive observations that disagree with `inside`


@dataclass
class _TrackState:
    object_type: str
    last_point: Point | None = None
    zones: dict[str, _ZoneState] = field(default_factory=dict)


@dataclass(frozen=True)
class _Prepared:
    zone: Zone
    bounds: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


def _prepare(zone: Zone) -> _Prepared:
    xs = [p[0] for p in zone.polygon]
    ys = [p[1] for p in zone.polygon]
    return _Prepared(zone, (min(xs), min(ys), max(xs), max(ys)))


class ZoneEngine:
    """Turns per-frame tracked boxes into enter/exit transitions.

    A track is in a zone when its ground point (bottom-centre of the box) is inside the polygon.
    Membership only flips after ``hysteresis_frames`` consecutive disagreeing observations, so
    one noisy frame cannot create events. Exits are reported before enters.
    """

    def __init__(self, zones: Sequence[Zone], hysteresis_frames: int = 1) -> None:
        if hysteresis_frames < 1:
            raise ValueError("hysteresis_frames must be at least 1")
        self.hysteresis_frames = hysteresis_frames
        self._zones: list[_Prepared] = [_prepare(z) for z in zones]
        self._tracks: dict[int, _TrackState] = {}

    def set_zones(self, zones: Sequence[Zone]) -> None:
        """Replace the zones. State is kept for zone ids that still exist; removed zones vanish."""
        self._zones = [_prepare(z) for z in zones]
        keep = {z.zone.id for z in self._zones}
        for state in self._tracks.values():
            state.zones = {zid: s for zid, s in state.zones.items() if zid in keep}
            state.last_point = None  # force re-evaluation against the new geometry

    def zones_of(self, track_id: int) -> list[str]:
        state = self._tracks.get(track_id)
        if state is None:
            return []
        return [p.zone.id for p in self._zones if state.zones.get(p.zone.id, _ZoneState()).inside]

    def update(self, tracks: Iterable[Track]) -> list[Transition]:
        exits: list[Transition] = []
        enters: list[Transition] = []
        for track in tracks:
            state = self._tracks.setdefault(track.track_id, _TrackState(track.object_type))
            state.object_type = track.object_type
            point = ground_point(track.box)
            pending = any(s.streak for s in state.zones.values())
            if point == state.last_point and not pending:
                continue  # stationary and nothing waiting to be confirmed
            state.last_point = point
            for prepared in self._zones:
                flipped = self._observe(state, prepared, point)
                if flipped is None:
                    continue
                event = Transition(
                    "enter" if flipped else "exit",
                    prepared.zone.id,
                    track.track_id,
                    track.object_type,
                )
                (enters if flipped else exits).append(event)
        return exits + enters

    def remove_track(self, track_id: int) -> list[Transition]:
        state = self._tracks.pop(track_id, None)
        if state is None:
            return []
        return [
            Transition("exit", p.zone.id, track_id, state.object_type)
            for p in self._zones
            if state.zones.get(p.zone.id, _ZoneState()).inside
        ]

    def _observe(self, state: _TrackState, prepared: _Prepared, point: Point) -> bool | None:
        """Update one zone's state. Returns the new membership if it flipped, else None."""
        zs = state.zones.setdefault(prepared.zone.id, _ZoneState())
        min_x, min_y, max_x, max_y = prepared.bounds
        x, y = point
        now_inside = min_x <= x <= max_x and min_y <= y <= max_y  # cheap reject first
        if now_inside:
            now_inside = point_in_polygon(point, prepared.zone.polygon)
        if now_inside == zs.inside:
            zs.streak = 0
            return None
        zs.streak += 1
        if zs.streak < self.hysteresis_frames:
            return None
        zs.inside, zs.streak = now_inside, 0
        return now_inside
