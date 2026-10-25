from pathlib import Path

import cv2
import numpy as np
import pytest

from palletvision_demo.live.autozone import Polygon, propose_zones
from palletvision_zones.geometry import point_in_polygon, validate_polygon

REPO = Path(__file__).parents[3]
CLIP = REPO / "data/infer/gemini_generated_video_a2c2e4ab.mp4"

# The paint color of the test footage (hue 27 of 179, pale), and a tan that is not paint.
PAINT = tuple(
    int(v) for v in cv2.cvtColor(np.array([[[27, 90, 190]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
)
CARDBOARD = tuple(
    int(v) for v in cv2.cvtColor(np.array([[[17, 90, 185]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
)
FLOOR = (128, 130, 126)


def floor_with_bays(scale: float = 1.0) -> np.ndarray:
    """A grey floor with two bays side by side, painted as seen from an angled camera.

    The base line runs along the front, and three lines stand on it: the left wall of the first
    bay, the wall the two bays share, and the right wall of the second.
    """
    img = np.full((round(720 * scale), round(1280 * scale), 3), FLOOR, np.uint8)
    width = max(1, round(8 * scale))

    def line(a: tuple[int, int], b: tuple[int, int]) -> None:
        cv2.line(
            img,
            (round(a[0] * scale), round(a[1] * scale)),
            (round(b[0] * scale), round(b[1] * scale)),
            PAINT,
            width,
            cv2.LINE_AA,
        )

    line((300, 600), (900, 620))  # the base line
    line((300, 600), (380, 280))  # left wall
    line((600, 610), (640, 280))  # the shared wall
    line((900, 620), (880, 280))  # right wall
    return img


def vanishing_point_of_the_walls() -> tuple[float, float]:
    """Where the left and right walls of floor_with_bays() meet, from their two line equations."""
    (x1, y1), (x2, y2) = (300, 600), (380, 280)
    (x3, y3), (x4, y4) = (900, 620), (880, 280)
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    cross1, cross2 = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    x = (cross1 * (x3 - x4) - (x1 - x2) * cross2) / denominator
    y = (cross1 * (y3 - y4) - (y1 - y2) * cross2) / denominator
    return x, y


def with_a_third_bay_whose_wall_is_hidden(stub: int = 50) -> np.ndarray:
    """A third bay to the right, of which only the foot of its right wall (a stub) can be seen."""
    img = floor_with_bays()
    cv2.line(img, (900, 620), (1180, 632), PAINT, 8)  # the base line carries on
    vx, vy = vanishing_point_of_the_walls()
    foot = np.array([1180.0, 632.0])
    toward = (np.array([vx, vy]) - foot) / np.hypot(vx - foot[0], vy - foot[1])
    top = foot + toward * stub
    cv2.line(img, tuple(int(v) for v in foot), tuple(int(v) for v in top), PAINT, 8, cv2.LINE_AA)
    return img


def corners_near(polygon: Polygon, expected: list[tuple[int, int]], tol: float) -> bool:
    return all(min(np.hypot(x - ex, y - ey) for x, y in polygon) <= tol for ex, ey in expected)


def test_two_bays_on_one_base_line_give_two_polygons_left_to_right() -> None:
    zones = propose_zones(floor_with_bays())
    assert len(zones) == 2
    assert corners_near(zones[0], [(300, 600), (600, 610)], tol=8)
    assert corners_near(zones[1], [(600, 610), (900, 620)], tol=8)
    assert min(x for x, _ in zones[0]) < min(x for x, _ in zones[1])


def test_a_bay_stops_where_both_of_its_walls_are_still_seen() -> None:
    # the walls end at y=280, but the shared wall is hidden above y=400 by something on top
    img = floor_with_bays()
    cv2.rectangle(img, (560, 270), (680, 400), FLOOR, -1)
    first, _ = propose_zones(img)
    assert min(y for _, y in first) >= 395  # cut at the lower of the two tops, never stretched up


def test_a_pillar_and_cardboard_add_no_zone() -> None:
    img = floor_with_bays()
    cv2.line(img, (900, 620), (1060, 626), PAINT, 8)  # the base line carries on
    cv2.rectangle(img, (1010, 280), (1060, 630), PAINT, -1)  # a paint colored pillar standing on it
    cv2.rectangle(img, (150, 100), (260, 220), CARDBOARD, -1)  # the wrong color
    assert len(propose_zones(img)) == 2  # not a third bay between the right wall and the pillar


def test_a_wall_hidden_in_the_middle_by_a_pallet_still_makes_the_bay() -> None:
    img = floor_with_bays()
    cv2.rectangle(img, (540, 380), (700, 470), FLOOR, -1)  # covers the middle of the shared wall
    assert len(propose_zones(img)) == 2


def test_a_wall_hidden_except_for_its_foot_is_run_up_to_the_height_of_the_neighbouring_wall() -> (
    None
):
    zones = propose_zones(with_a_third_bay_whose_wall_is_hidden())
    assert len(zones) == 3
    third = zones[2]
    assert corners_near(third, [(900, 620), (1180, 632)], tol=20)
    vx, vy = vanishing_point_of_the_walls()
    top_y = max(y for _, y in third if y < 400)
    assert abs(top_y - 280) <= 12  # as tall as the wall beside it
    far_x = next(x for x, y in third if y < 400 and x > 950)
    expected_x = 1180 + (280 - 632) * (vx - 1180) / (
        vy - 632
    )  # on the line from foot to vanishing point
    assert abs(far_x - expected_x) <= 25


def test_a_short_post_that_does_not_point_at_the_vanishing_point_is_not_a_wall() -> None:
    img = floor_with_bays()
    cv2.line(img, (900, 620), (1180, 632), PAINT, 8)
    cv2.line(img, (1180, 632), (1200, 600), PAINT, 8)  # short, leans right: the walls lean left
    assert len(propose_zones(img)) == 2


def test_the_same_frame_gives_the_same_zones() -> None:
    img = floor_with_bays()
    assert propose_zones(img) == propose_zones(img.copy())


def test_a_frame_without_paint_gives_no_zones() -> None:
    assert propose_zones(np.full((720, 1280, 3), FLOOR, np.uint8)) == []


def test_a_single_line_is_not_a_bay() -> None:
    img = np.full((720, 1280, 3), FLOOR, np.uint8)
    cv2.line(img, (300, 600), (900, 620), PAINT, 8)
    assert propose_zones(img) == []


def test_every_zone_is_a_valid_polygon_inside_the_frame() -> None:
    img = floor_with_bays()
    for zone in propose_zones(img):
        validate_polygon([(float(x), float(y)) for x, y in zone])
        assert all(0 <= x < img.shape[1] and 0 <= y < img.shape[0] for x, y in zone)


def test_a_frame_of_another_size_gives_the_same_bays_scaled() -> None:
    full = propose_zones(floor_with_bays(1.0))
    half = propose_zones(floor_with_bays(0.5))
    assert len(half) == len(full) == 2
    for big, small in zip(full, half, strict=True):
        assert corners_near(big, [(2 * x, 2 * y) for x, y in small], tol=14)


def test_an_image_that_is_not_bgr_is_refused() -> None:
    with pytest.raises(ValueError, match="BGR"):
        propose_zones(np.zeros((100, 100), np.uint8))


@pytest.mark.skipif(not CLIP.exists(), reason="the generated test clip is not present")
def test_the_first_frame_of_the_generated_clip_gives_the_bay_on_the_floor() -> None:
    cap = cv2.VideoCapture(str(CLIP))
    ok, frame = cap.read()
    cap.release()
    assert ok
    zones = propose_zones(np.asarray(frame, np.uint8))

    def inside(point: tuple[float, float]) -> bool:
        return any(
            point_in_polygon(point, [(float(x), float(y)) for x, y in zone]) for zone in zones
        )

    assert inside((680.0, 450.0))  # the empty floor of bay 16
    assert inside((1080.0, 450.0))  # bay 14, whose right wall is hidden but for its foot
    assert not inside((700.0, 40.0))  # the yellow cages on the back wall
