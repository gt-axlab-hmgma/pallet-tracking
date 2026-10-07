from palletvision_zones.config import Zone
from palletvision_zones.engine import Track, Transition, ZoneEngine

ZONE_A = Zone("a", ((0, 0), (100, 0), (100, 100), (0, 100)))
ZONE_B = Zone("b", ((200, 0), (300, 0), (300, 100), (200, 100)))
SLANTED = Zone("s", ((400, 90), (500, 60), (480, 10), (410, 20)))


def box_at(x: float, y: float, w: float = 20, h: float = 20) -> tuple[float, float, float, float]:
    """A box whose ground point (bottom-centre) is exactly (x, y)."""
    return (x - w / 2, y - h, x + w / 2, y)


def track(x: float, y: float, tid: int = 1, kind: str = "pallet_type_a") -> Track:
    return Track(tid, kind, box_at(x, y))


def t(kind: str, zone: str, tid: int = 1, obj: str = "pallet_type_a") -> Transition:
    return Transition(kind, zone, tid, obj)  # type: ignore[arg-type]


def test_enter_when_first_seen_inside() -> None:
    engine = ZoneEngine([ZONE_A])
    assert engine.update([track(50, 50)]) == [t("enter", "a")]


def test_no_event_while_staying_inside_or_outside() -> None:
    engine = ZoneEngine([ZONE_A])
    assert engine.update([track(500, 500)]) == []
    engine.update([track(50, 50)])
    assert engine.update([track(60, 60)]) == []
    assert engine.update([track(60, 60)]) == []  # stationary


def test_exit_when_leaving() -> None:
    engine = ZoneEngine([ZONE_A])
    engine.update([track(50, 50)])
    assert engine.update([track(150, 50)]) == [t("exit", "a")]


def test_move_between_zones_is_exit_then_enter() -> None:
    engine = ZoneEngine([ZONE_A, ZONE_B])
    engine.update([track(50, 50)])
    assert engine.update([track(250, 50)]) == [t("exit", "a"), t("enter", "b")]


def test_overlapping_zones_enter_both() -> None:
    inner = Zone("inner", ((20, 20), (80, 20), (80, 80), (20, 80)))
    engine = ZoneEngine([ZONE_A, inner])
    assert engine.update([track(50, 50)]) == [t("enter", "a"), t("enter", "inner")]
    assert engine.zones_of(1) == ["a", "inner"]


def test_slanted_zone() -> None:
    engine = ZoneEngine([SLANTED])
    assert engine.update([track(450, 50)]) == [t("enter", "s")]
    assert engine.update([track(490, 85)]) == [t("exit", "s")]  # inside bbox, below the slant


def test_tracks_are_independent_and_carry_their_type() -> None:
    engine = ZoneEngine([ZONE_A])
    events = engine.update([track(50, 50, 1, "pallet_type_a"), track(60, 60, 2, "pallet_type_b")])
    assert events == [t("enter", "a", 1, "pallet_type_a"), t("enter", "a", 2, "pallet_type_b")]
    assert engine.update([track(50, 50, 1), track(500, 60, 2, "pallet_type_b")]) == [
        t("exit", "a", 2, "pallet_type_b")
    ]


def test_hysteresis_needs_consecutive_frames() -> None:
    engine = ZoneEngine([ZONE_A], hysteresis_frames=3)
    assert engine.update([track(50, 50)]) == []
    assert engine.update([track(50, 50)]) == []  # stationary but still counting
    assert engine.update([track(50, 50)]) == [t("enter", "a")]
    assert engine.update([track(150, 50)]) == []
    assert engine.update([track(50, 50)]) == []  # flicker back resets the exit streak
    assert engine.update([track(150, 50)]) == []
    assert engine.update([track(150, 50)]) == []
    assert engine.update([track(150, 50)]) == [t("exit", "a")]


def test_hysteresis_flicker_never_enters() -> None:
    engine = ZoneEngine([ZONE_A], hysteresis_frames=2)
    for x in (50, 150, 50, 150, 50, 150):
        assert engine.update([track(x, 50)]) == []


def test_remove_track_exits_its_zones() -> None:
    engine = ZoneEngine([ZONE_A, ZONE_B])
    engine.update([track(50, 50)])
    assert engine.remove_track(1) == [t("exit", "a")]
    assert engine.remove_track(1) == []
    assert engine.update([track(50, 50)]) == [t("enter", "a")]  # seen as a new track


def test_set_zones_new_zone_is_entered_on_next_update() -> None:
    engine = ZoneEngine([])
    assert engine.update([track(50, 50)]) == []
    engine.set_zones([ZONE_A])
    assert engine.update([track(50, 50)]) == [t("enter", "a")]


def test_set_zones_removed_zone_drops_state_silently() -> None:
    engine = ZoneEngine([ZONE_A])
    engine.update([track(50, 50)])
    engine.set_zones([])
    assert engine.zones_of(1) == []
    assert engine.update([track(50, 50)]) == []


def test_editing_a_zone_polygon_re_evaluates_membership() -> None:
    engine = ZoneEngine([ZONE_A])
    engine.update([track(50, 50)])
    moved = Zone("a", ((200, 0), (300, 0), (300, 100), (200, 100)))
    engine.set_zones([moved])
    assert engine.update([track(50, 50)]) == [t("exit", "a")]


def test_boundary_counts_as_inside() -> None:
    engine = ZoneEngine([ZONE_A])
    assert engine.update([track(100, 50)]) == [t("enter", "a")]
