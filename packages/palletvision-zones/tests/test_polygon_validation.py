import math

import pytest

from palletvision_zones.geometry import PolygonError, validate_polygon


def test_valid_polygons_pass() -> None:
    validate_polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    validate_polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])  # concave
    validate_polygon([(112, 980), (860, 940), (910, 610), (240, 620)])  # slanted


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ([(0, 0), (1, 1)], "at least 3"),
        ([(0, 0), (1, 1), (math.nan, 2)], "finite"),
        ([(0, 0), (math.inf, 1), (2, 2)], "finite"),
        ([(0, 0), (0, 0), (5, 5), (0, 5)], "duplicate"),
        ([(0, 0), (5, 5), (10, 10)], "area"),
        ([(0, 0), (10, 10), (10, 0), (0, 10)], "self-intersect"),  # bow tie
    ],
)
def test_invalid_polygons_raise(polygon: list[tuple[float, float]], message: str) -> None:
    with pytest.raises(PolygonError, match=message):
        validate_polygon(polygon)
