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
