"""Performance budget for the engine: it must stay negligible next to a 100 ms frame."""

import math
import time

from palletvision_zones.config import Zone
from palletvision_zones.engine import Track, ZoneEngine

N_TRACKS = 20
N_ZONES = 10
N_FRAMES = 500
# Target is 0.5 ms per frame on a laptop. The assertion is looser so shared CI runners
# do not flake while a real slowdown (an order of magnitude) still fails.
BUDGET_MS = 1.0


def octagon(cx: float, cy: float, r: float) -> tuple[tuple[float, float], ...]:
    return tuple(
        (cx + r * math.cos(2 * math.pi * k / 8), cy + r * math.sin(2 * math.pi * k / 8))
        for k in range(8)
    )


def test_update_time_per_frame_is_within_budget() -> None:
    zones = [
        Zone(f"z{i}", octagon(100 + 180 * (i % 5), 150 + 300 * (i // 5), 90))
        for i in range(N_ZONES)
    ]
    engine = ZoneEngine(zones)
    frames = []
    for f in range(N_FRAMES):
        frames.append(
            [
                Track(
                    t,
                    "pallet_type_a",
                    (t * 45 + f % 90, f % 600, t * 45 + 20 + f % 90, 20 + f % 600),
                )
                for t in range(N_TRACKS)
            ]
        )
    start = time.perf_counter()
    for tracks in frames:
        engine.update(tracks)
    per_frame_ms = (time.perf_counter() - start) * 1000 / N_FRAMES
    print(f"engine: {per_frame_ms:.3f} ms/frame ({N_TRACKS} tracks, {N_ZONES} zones)")
    assert per_frame_ms < BUDGET_MS
