"""Propose zones from the colored lines painted on the floor.

This is a helper for drawing zones, not a replacement for it: it looks at one frame, returns the
bays it can see, and the user then corrects them by hand (drag a corner, delete, draw a missing
one). Every step is plain arithmetic with fixed parameters, so the same frame always gives the
same polygons.

The idea, in the order it runs:

1. Smooth the frame a little, then keep the pixels whose color is the paint color (a hue band
   in HSV). Unless told the color, find it: try every hue and keep the one that makes the
   strongest bay structure in the next steps.
2. Drop what is too wide to be a painted strip (pillars) and what is too small (specks).
3. Fit straight segments to what is left and join the ones that lie on the same line, so a line
   that a pallet hides in the middle comes back as one line.
4. A bay is painted as a U: a base line along the front and two lines running back from it. So
   for each long, flat base line, find the lines that stand on it, and take every two neighbours
   with the base between them as one bay.

The back of a bay is usually not painted. The polygon stops where both side lines are still seen
and is never stretched past paint that was not found.

The pixel sizes below are for a 1280 pixel wide frame and are scaled to the real frame size.
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from palletvision_zones.geometry import PolygonError, validate_polygon

Frame = NDArray[np.uint8]
Polygon = list[tuple[int, int]]
Pt = tuple[float, float]

REFERENCE_WIDTH = 1280  # the frame width the pixel sizes in AutoZoneParams are written for
MIN_CLICKED_SATURATION = 25  # a clicked spot paler than this is floor, not paint


@dataclass(frozen=True)
class AutoZoneParams:
    # What the paint looks like, in OpenCV HSV (hue 0-179, a circle: red is at both ends). With
    # ``hue`` left as None the color is found from the frame itself: every hue is tried, and the
    # one that gives the strongest bay structure (long base lines with walls standing on them)
    # wins. Set it to use one known color, for example 27 for the pale yellow of the test footage.
    hue: int | None = None
    hue_half_width: int = 6
    hue_step: int = 3  # how far apart the hues tried in the search are
    search_width: int = 640  # the search runs on a frame scaled down to this width, for speed
    min_saturation: int = 45
    min_value: int = 110
    # When the color is searched for and nothing is found, these (saturation, value) floors are
    # tried in turn. A camera that sees the paint paler or darker than the usual footage, such as
    # a photo of a screen, needs them; a frame that already works never reaches them.
    fallback_floors: tuple[tuple[int, int], ...] = ((25, 90),)
    blur_sigma: float = 0.8  # smoothing before the color test, so sensor noise cannot break it
    # Pixel sizes at REFERENCE_WIDTH.
    max_stroke_half_width: float = 9.0  # fatter than this is a pillar, not a painted strip
    min_blob_area: float = 200.0
    hough_min_length: float = 25.0
    hough_max_gap: float = 30.0
    hough_votes: float = 30.0
    merge_offset: float = 14.0  # how far apart two segments may sit and still be one line
    merge_gap: float = 60.0  # how long a hidden stretch may be between two pieces of a line
    merge_angle_deg: float = 5.0
    min_base_length: float = 250.0
    min_side_length: float = 60.0
    # A wall mostly hidden behind a pallet may show only its foot, where it meets the base line.
    # That stub counts if it points where the other walls point (their vanishing point), and it
    # is run up to the height of its neighbour when it is much shorter than the neighbour.
    min_stub_length: float = 25.0
    vanishing_deg: float = 6.0  # how far a long wall may point away from the vanishing point
    stub_direction_deg: float = (
        25.0  # the same for a short wall: its direction is only good to ~15 degrees
    )
    short_wall_ratio: float = 0.6
    foot_distance: float = 45.0  # how close a side line must end to the base line to stand on it
    # Angles are measured against the image x axis, 0 to 180 degrees.
    base_max_deg: float = 20.0  # a base line is at most this far from horizontal
    side_min_deg: float = 40.0  # a side line runs between these two angles
    side_max_deg: float = 140.0
    min_bay_area: float = 6000.0
    duplicate_overlap: float = 0.8  # two bays whose boxes overlap this much are one bay


@dataclass(frozen=True)
class _Line:
    p: Pt
    q: Pt

    @property
    def length(self) -> float:
        return math.hypot(self.q[0] - self.p[0], self.q[1] - self.p[1])

    @property
    def angle(self) -> float:
        return math.degrees(math.atan2(self.q[1] - self.p[1], self.q[0] - self.p[0])) % 180.0

    @property
    def lower(self) -> Pt:
        """The end nearest the camera, which is the end lower in the image."""
        return self.p if self.p[1] >= self.q[1] else self.q

    @property
    def upper(self) -> Pt:
        return self.q if self.p[1] >= self.q[1] else self.p


@dataclass(frozen=True)
class Proposal:
    """The bays found, and the hue (OpenCV 0-179) of the paint they were found with."""

    polygons: list[Polygon]
    hue: int | None


def propose_zones(image: Frame, params: AutoZoneParams | None = None) -> list[Polygon]:
    """Bays found in a BGR frame, as integer polygons ordered left to right.

    Returns an empty list when there is no paint to find. Every polygon passes the same checks as
    a drawn zone, so it can be sent to the zones API as it is.
    """
    return find_zones(image, params).polygons


def find_zones(image: Frame, params: AutoZoneParams | None = None) -> Proposal:
    """Like ``propose_zones``, and also says which paint colour the bays were found with.

    The colour lets the caller draw the zones in one that stands out from the paint.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected a BGR image of shape (height, width, 3) and type uint8")
    params = params or AutoZoneParams()
    smoothed = _smooth(image, params)
    attempts = [params]
    if params.hue is None:  # a known color has nothing to retry
        attempts += [
            replace(params, min_saturation=s, min_value=v) for s, v in params.fallback_floors
        ]
    for attempt in attempts:
        hue = attempt.hue if attempt.hue is not None else _find_paint_hue(smoothed, attempt)
        if hue is None:
            continue
        polygons = [polygon for polygon, _ in _bays_of_color(smoothed, hue, attempt)]
        if polygons:
            polygons = _drop_duplicates(polygons, attempt.duplicate_overlap)
            polygons = sorted(
                polygons, key=lambda poly: (min(x for x, _ in poly), min(y for _, y in poly))
            )
            return Proposal(polygons, hue)
    return Proposal([], None)


def sample_paint(
    image: Frame, x: float, y: float, params: AutoZoneParams | None = None
) -> AutoZoneParams | None:
    """The settings for the paint colour under the point (x, y), which the user clicked.

    Reading the colour from the line itself needs no guess about the camera, the lighting or how
    pale the paint looks. The hue is the one under the click, and the saturation and brightness
    floors are set a fixed share below what was read, so a paler or darker stretch of the same
    paint still counts. Returns None if the point is outside the frame or not coloured (a grey
    floor), which means the user did not click on paint.
    """
    params = params or AutoZoneParams()
    height, width = image.shape[:2]
    cx, cy = round(x), round(y)
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    reach = max(2, round(4 * width / REFERENCE_WIDTH))
    patch = _smooth(image, params)[
        max(cy - reach, 0) : cy + reach + 1, max(cx - reach, 0) : cx + reach + 1
    ]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    # A click can land on the edge of a thin line, so read only the more colourful half of the
    # patch, where the paint is.
    paint = hsv[hsv[:, 1] >= np.median(hsv[:, 1])]
    saturation, value = float(np.median(paint[:, 1])), float(np.median(paint[:, 2]))
    if saturation < MIN_CLICKED_SATURATION:
        return None
    angle = np.radians(paint[:, 0] * 2.0)  # hue is a circle of 180 steps
    hue = round((math.degrees(math.atan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360) / 2)
    return replace(
        params,
        hue=hue % 180,
        min_saturation=max(15, int(saturation * 0.55)),
        min_value=max(50, int(value * 0.6)),
    )


def _smooth(image: Frame, params: AutoZoneParams) -> Frame:
    """Light blur, sized to the frame. Sensor noise moves the hue of pale pixels at random."""
    sigma = params.blur_sigma * image.shape[1] / REFERENCE_WIDTH
    if sigma <= 0:
        return image
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), dtype=np.uint8)


def _find_paint_hue(image: Frame, params: AutoZoneParams) -> int | None:
    """The hue whose pixels form the strongest bay structure, or None if none forms any.

    Cardboard, cages and pillars are colored too, but their pixels do not make long base lines
    with walls standing on them, so they score nothing.
    """
    height, width = image.shape[:2]
    if width > params.search_width:
        small = cv2.resize(
            image, (params.search_width, round(height * params.search_width / width)),
            interpolation=cv2.INTER_AREA,
        )  # fmt: skip
    else:
        small = image
    scores = {
        hue: sum(e for _, e in _bays_of_color(np.asarray(small, np.uint8), hue, params))
        for hue in range(0, 180, params.hue_step)
    }
    best = max(scores.values())
    if best <= 0:
        return None
    # Hues next to the true paint colour score alike, so take the middle of the ones that tie
    # (within 2%) instead of the first. Hue is a circle, so the middle is a circular mean.
    tied = [hue for hue, score in scores.items() if score >= 0.98 * best]
    angle = np.radians(np.array(tied, dtype=np.float64) * 2.0)
    middle = math.degrees(math.atan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360
    return round(middle / 2) % 180


def _bays_of_color(image: Frame, hue: int, params: AutoZoneParams) -> list[tuple[Polygon, float]]:
    """The bays that the paint of one hue makes, each with the length of paint that supports it."""
    height, width = image.shape[:2]
    scale = width / REFERENCE_WIDTH
    strokes = _strokes(_paint_mask(image, hue, params), params, scale)
    lines = _merge_collinear(_segments(strokes, params, scale), params, scale)
    found: list[tuple[Polygon, float]] = []
    for corners, evidence in _bays(lines, params, scale):
        polygon = _to_polygon(corners, width, height)
        if polygon is not None and _area(polygon) >= params.min_bay_area * scale**2:
            found.append((polygon, evidence))
    return found


def _paint_mask(image: Frame, hue: int, params: AutoZoneParams) -> NDArray[np.uint8]:
    hues, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    apart = np.abs(hues.astype(np.int16) - hue)
    apart = np.minimum(apart, 180 - apart)  # hue is a circle
    on = (apart <= params.hue_half_width) & (saturation >= params.min_saturation)
    on &= value >= params.min_value
    return np.asarray(on, dtype=np.uint8)


def _strokes(mask: NDArray[np.uint8], params: AutoZoneParams, scale: float) -> NDArray[np.uint8]:
    """The mask without specks and without anything too wide to be a painted strip."""
    mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.uint8)
    # The distance to the nearest background pixel is half the stroke width at the stroke's
    # centre. A pillar is far wider than a floor line, so its core is well over the limit.
    limit = params.max_stroke_half_width * scale
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thick = (distance >= limit).astype(np.uint8)
    # The rim around a core is `limit` wide, so growing the core by that much (and a margin)
    # takes the whole pillar and leaves no edge strips that would look like painted lines.
    reach = math.ceil(limit) + 2
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    mask[cv2.dilate(thick, grow) > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = [
        i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < params.min_blob_area * scale**2
    ]
    mask[np.isin(labels, small)] = 0
    return mask


def _segments(
    strokes: NDArray[np.uint8], params: AutoZoneParams, scale: float
) -> list[tuple[float, float, float, float]]:
    found = cv2.HoughLinesP(
        strokes * 255,
        1,
        np.pi / 360,
        threshold=max(1, round(params.hough_votes * scale)),
        minLineLength=params.hough_min_length * scale,
        maxLineGap=params.hough_max_gap * scale,
    )
    if found is None:
        return []
    return [(float(a), float(b), float(c), float(d)) for a, b, c, d in found.reshape(-1, 4)]


@dataclass
class _Group:
    """Segments known to lie on one line: its direction, a point on it, and its extent along it."""

    angle: float
    anchor: NDArray[np.float64]
    start: float
    end: float

    @property
    def direction(self) -> NDArray[np.float64]:
        rad = math.radians(self.angle)
        return np.array([math.cos(rad), math.sin(rad)])


def _merge_collinear(
    segments: list[tuple[float, float, float, float]], params: AutoZoneParams, scale: float
) -> list[_Line]:
    """Join segments that lie on the same line into one line each.

    The longest segment is placed first, so what it starts is the line that the shorter pieces
    join. Sorting on the coordinates after the length keeps the result the same on every run.
    """
    groups: list[_Group] = []
    for x1, y1, x2, y2 in sorted(
        segments, key=lambda s: (-math.hypot(s[2] - s[0], s[3] - s[1]), s)
    ):
        angle = _Line((x1, y1), (x2, y2)).angle
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        ends = np.array([[x1, y1], [x2, y2]])
        for group in groups:
            if abs((angle - group.angle + 90.0) % 180.0 - 90.0) > params.merge_angle_deg:
                continue
            d = group.direction
            normal = np.array([-d[1], d[0]])
            if abs(float((mid - group.anchor) @ normal)) > params.merge_offset * scale:
                continue
            along = (ends - group.anchor) @ d
            if along.max() < group.start - params.merge_gap * scale:
                continue
            if along.min() > group.end + params.merge_gap * scale:
                continue
            group.start = min(group.start, float(along.min()))
            group.end = max(group.end, float(along.max()))
            break
        else:
            d = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            along = (ends - mid) @ d
            groups.append(_Group(angle, mid, float(along.min()), float(along.max())))
    lines = []
    for g in groups:
        a = g.anchor + g.direction * g.start
        b = g.anchor + g.direction * g.end
        lines.append(_Line((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return lines


def _bays(lines: list[_Line], params: AutoZoneParams, scale: float) -> list[tuple[list[Pt], float]]:
    """Each bay's corners with the length of paint behind it (its base line and its two walls)."""
    bases = sorted(
        (
            line
            for line in lines
            if min(line.angle, 180.0 - line.angle) <= params.base_max_deg
            and line.length >= params.min_base_length * scale
        ),
        key=lambda b: (min(b.p[0], b.q[0]), b.p[1]),
    )
    steep = [
        line
        for line in lines
        if params.side_min_deg <= line.angle <= params.side_max_deg
        and line.length >= params.min_stub_length * scale
    ]
    # Only walls standing on a base line can be bay walls. The posts of the wall cages and the
    # pillars stand on nothing, so they neither make bays nor disturb the vanishing point.
    floor_walls = [w for w in steep if any(_stands_on(w, b, params, scale) for b in bases)]
    long_walls = [w for w in floor_walls if w.length >= params.min_side_length * scale]
    vanishing = _vanishing_point(long_walls, params)
    walls = [
        w
        for w in floor_walls
        if w.length >= params.min_side_length * scale
        or (vanishing is not None and _points_at(w, vanishing, params.stub_direction_deg))
    ]
    bays: list[tuple[list[Pt], float]] = []
    for base in bases:
        standing = sorted(
            (w for w in walls if _stands_on(w, base, params, scale)),
            key=lambda w: (w.lower[0], w.lower[1]),
        )
        for a, b in zip(standing, standing[1:], strict=False):
            bay = _bay_between(a, b, vanishing, params)
            if bay is not None:
                bays.append((bay, a.length + b.length + base.length))
    return bays


def _stands_on(wall: _Line, base: _Line, params: AutoZoneParams, scale: float) -> bool:
    """Whether the foot of the wall is on the base line, within the length of the base line."""
    left, right = sorted((base.p, base.q))
    along = np.array(right) - np.array(left)
    normal = np.array([-along[1], along[0]]) / math.hypot(along[0], along[1])
    foot = np.array(wall.lower)
    distance = abs(float((foot - left) @ normal))
    position = float((foot - left) @ along / (along @ along))
    return distance <= params.foot_distance * scale and -0.05 <= position <= 1.05


def _vanishing_point(walls: list[_Line], params: AutoZoneParams) -> NDArray[np.float64] | None:
    """The point the walls head for, if three or more agree on one.

    Walls of bays painted side by side are parallel on the floor, so in the image they run toward
    one point. Any two lines meet somewhere, so a point only counts when at least three walls
    point at it. Every pair is tried, and the point that most wall length agrees with wins, which
    makes one stray post or fence line unable to move it.
    """
    best: tuple[float, NDArray[np.float64]] | None = None
    for i, first in enumerate(walls):
        for second in walls[i + 1 :]:
            point = _meet(first, second)
            if point is None:
                continue
            agreeing = [w for w in walls if _points_at(w, point, params.vanishing_deg)]
            if len(agreeing) < 3:
                continue
            weight = sum(w.length for w in agreeing)
            if best is None or weight > best[0]:
                best = (weight, point)
    return None if best is None else best[1]


def _meet(a: _Line, b: _Line) -> NDArray[np.float64] | None:
    """Where the infinite lines through ``a`` and ``b`` cross, or None if they are parallel."""
    p, r = np.array(a.p), np.array(a.q) - np.array(a.p)
    q, s = np.array(b.p), np.array(b.q) - np.array(b.p)
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) < 1e-9 * a.length * b.length:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / denominator
    return np.asarray(p + t * r, dtype=np.float64)


def _points_at(line: _Line, target: NDArray[np.float64], tolerance_deg: float) -> bool:
    """Whether the line, from its foot, heads toward ``target`` within the tolerance."""
    foot = np.array(line.lower)
    wanted = target - foot
    if wanted[1] >= 0:  # the point must be above the foot, further from the camera
        return False
    seen = np.array(line.upper) - foot
    cosine = float(wanted @ seen / (np.hypot(*wanted) * np.hypot(*seen)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) <= tolerance_deg


def _height(line: _Line) -> float:
    return line.lower[1] - line.upper[1]


def _bay_between(
    a: _Line, b: _Line, vanishing: NDArray[np.float64] | None, params: AutoZoneParams
) -> list[Pt] | None:
    """The four corners of the bay between two walls that stand on one base line.

    Both walls are cut at the lower of their two tops, the highest point at which both are still
    seen. The paint above it is hidden or missing, and nothing is stretched past it.

    The exception is a wall that shows only its foot, because a pallet hides the rest. It is run
    up, toward the vanishing point, to the height where its neighbour ends: the two walls of a
    bay are as long as each other.
    """
    short, tall = (a, b) if _height(a) < _height(b) else (b, a)
    if (
        vanishing is not None
        and _height(short) < params.short_wall_ratio * _height(tall)
        and _points_at(short, vanishing, params.stub_direction_deg)
    ):
        top_y = tall.upper[1]
        foot = np.array(short.lower)
        t = (top_y - foot[1]) / (vanishing[1] - foot[1])
        if top_y >= foot[1] - 1.0 or not 0.0 < t < 1.0:
            return None
        far = foot + t * (vanishing - foot)
        tops = {id(short): (float(far[0]), float(far[1])), id(tall): tall.upper}
        return [a.lower, b.lower, tops[id(b)], tops[id(a)]]
    top_y = max(a.upper[1], b.upper[1])
    if top_y >= min(a.lower[1], b.lower[1]) - 1.0:
        return None
    return [a.lower, b.lower, _cut_at(b, top_y), _cut_at(a, top_y)]


def _cut_at(line: _Line, y: float) -> Pt:
    """The point on a side line at height ``y``, between its foot and its top."""
    (fx, fy), (tx, ty) = line.lower, line.upper
    if ty >= y:
        return (tx, ty)
    t = (y - fy) / (ty - fy)
    return (fx + t * (tx - fx), y)


def _to_polygon(corners: list[Pt], width: int, height: int) -> Polygon | None:
    polygon = [
        (min(max(round(x), 0), width - 1), min(max(round(y), 0), height - 1)) for x, y in corners
    ]
    try:
        validate_polygon(polygon)
    except PolygonError:
        return None
    return polygon


def _area(polygon: Polygon) -> float:
    n = len(polygon)
    twice = sum(
        polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
        for i in range(n)
    )
    return abs(twice) / 2.0


def _box(polygon: Polygon) -> tuple[int, int, int, int]:
    xs, ys = [x for x, _ in polygon], [y for _, y in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _drop_duplicates(polygons: list[Polygon], overlap: float) -> list[Polygon]:
    """Keep the bigger of two bays whose bounding boxes mostly coincide."""
    kept: list[Polygon] = []
    for polygon in sorted(polygons, key=_area, reverse=True):
        x1, y1, x2, y2 = _box(polygon)
        same = False
        for other in kept:
            a1, b1, a2, b2 = _box(other)
            inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
            smaller = min((x2 - x1) * (y2 - y1), (a2 - a1) * (b2 - b1))
            if smaller > 0 and inter / smaller >= overlap:
                same = True
                break
        if not same:
            kept.append(polygon)
    return kept
