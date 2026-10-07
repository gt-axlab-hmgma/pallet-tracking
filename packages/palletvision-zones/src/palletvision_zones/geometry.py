import math
from collections.abc import Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]


def _on_segment(p: Point, a: Point, b: Point) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if cross != 0:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray casting for any simple polygon, concave included. Boundary counts as inside."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if _on_segment(point, a, b):
            return True
        # Count edges that straddle the horizontal ray going right from the point.
        # The half-open comparison avoids double-counting a ray through a vertex.
        if (a[1] > y) != (b[1] > y):
            x_cross = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < x_cross:
                inside = not inside
    return inside


class PolygonError(ValueError):
    """Raised when a drawn polygon cannot be used as a zone."""


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and 0 not in (o1, o2, o3, o4):
        return True
    return (
        (o1 == 0 and _on_segment(c, a, b))
        or (o2 == 0 and _on_segment(d, a, b))
        or (o3 == 0 and _on_segment(a, c, d))
        or (o4 == 0 and _on_segment(b, c, d))
    )


def validate_polygon(polygon: Polygon) -> None:
    """Reject polygons that cannot be a zone: too few vertices, NaN, zero area, self-crossing."""
    n = len(polygon)
    if n < 3:
        raise PolygonError("polygon needs at least 3 vertices")
    if not all(math.isfinite(v) for p in polygon for v in p):
        raise PolygonError("polygon coordinates must be finite")
    if any(polygon[i] == polygon[(i + 1) % n] for i in range(n)):
        raise PolygonError("polygon has duplicate consecutive vertices")
    for i in range(n):
        for j in range(i + 1, n):
            adjacent = j == i + 1 or (i == 0 and j == n - 1)
            if adjacent:
                continue
            if _segments_intersect(
                polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]
            ):
                raise PolygonError("polygon edges self-intersect")
    area2 = sum(
        polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
        for i in range(n)
    )
    if area2 == 0:
        raise PolygonError("polygon has zero area")
