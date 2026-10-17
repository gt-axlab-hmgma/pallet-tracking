import pytest

from palletvision_zones.geometry import point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
# Slanted quadrilateral, like a floor area seen from an overhead angled camera.
SLANTED = [(112, 980), (860, 940), (910, 610), (240, 620)]
# L shape: the notch at the top right is outside.
CONCAVE = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]


@pytest.mark.parametrize(
    ("point", "expected"),
    [((5, 5), True), ((-1, 5), False), ((11, 5), False), ((5, 11), False), ((5, -1), False)],
)
def test_square(point: tuple[float, float], expected: bool) -> None:
    assert point_in_polygon(point, SQUARE) is expected


@pytest.mark.parametrize(
    ("point", "expected"),
    [((500, 800), True), ((120, 700), False), ((900, 700), False), ((500, 500), False)],
)
def test_slanted_quad(point: tuple[float, float], expected: bool) -> None:
    assert point_in_polygon(point, SLANTED) is expected


@pytest.mark.parametrize(
    ("point", "expected"),
    [((2, 8), True), ((8, 2), True), ((8, 8), False), ((6, 6), False), ((2, 2), True)],
)
def test_concave_notch_is_outside(point: tuple[float, float], expected: bool) -> None:
    assert point_in_polygon(point, CONCAVE) is expected


@pytest.mark.parametrize("point", [(0, 5), (10, 5), (5, 0), (5, 10), (0, 0), (10, 10)])
def test_boundary_and_vertices_count_as_inside(point: tuple[float, float]) -> None:
    assert point_in_polygon(point, SQUARE) is True


def test_vertex_order_does_not_matter() -> None:
    assert point_in_polygon((5, 5), list(reversed(SQUARE))) is True
    assert point_in_polygon((15, 5), list(reversed(SQUARE))) is False


def test_ray_through_a_vertex_is_not_double_counted() -> None:
    diamond = [(5, 0), (10, 5), (5, 10), (0, 5)]
    assert point_in_polygon((5, 5), diamond) is True
    assert point_in_polygon((11, 5), diamond) is False  # ray passes through vertex (10, 5)
    assert point_in_polygon((-1, 5), diamond) is False
