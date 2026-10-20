"""Performance budget for the tracker: it must stay negligible next to a 100 ms frame.

Run ``uv run pytest packages/palletvision-track/tests/test_tracker_perf.py -s`` to see the table.
"""

import time

import numpy as np
import pytest

from palletvision_track.bytetrack import ByteTracker

N_WARMUP = 30
N_FRAMES = 200
# Measured on a laptop: about 0.6 ms at 100 objects, 2.5 ms at 300. The budgets are 3x looser so
# shared CI runners do not flake, while a real slowdown (an order of magnitude) still fails.
BUDGET_MS = {100: 2.0, 300: 8.0}


def scene(n_objects: int, n_frames: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Objects drifting on a grid with misses, low-score dips and clutter, so every branch runs."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_objects)))
    pos = np.array([(i % side, i // side) for i in range(n_objects)], dtype=np.float64) * 120 + 60
    vel = rng.normal(0, 1.5, (n_objects, 2))
    frames = []
    for _ in range(n_frames):
        pos += vel + rng.normal(0, 0.5, pos.shape)
        r = rng.random(n_objects)
        keep = r > 0.05  # 5% missed
        dip = rng.uniform(0.2, 0.55, n_objects)  # 20% partly hidden
        scores = np.where(r < 0.25, dip, rng.uniform(0.7, 0.98, n_objects))
        clutter = rng.uniform(0, side * 120, (max(1, n_objects // 20), 2))
        boxes = np.vstack(
            [
                np.column_stack([pos - [20, 40], pos + [20, 40]])[keep],
                np.column_stack([clutter, clutter + [30, 60]]),
            ]
        )
        weak = rng.uniform(0.1, 0.5, len(clutter))
        frames.append((boxes, np.concatenate([scores[keep], weak])))
    return frames


@pytest.mark.parametrize("n_objects", [10, 100, 300])
def test_update_time_per_frame_is_within_budget(n_objects: int) -> None:
    frames = scene(n_objects, N_WARMUP + N_FRAMES)
    tracker = ByteTracker()
    times = []
    for f, (boxes, scores) in enumerate(frames, start=1):
        start = time.perf_counter()
        tracker.update(f, boxes, scores)
        if f > N_WARMUP:
            times.append((time.perf_counter() - start) * 1000)
    mean, p99 = float(np.mean(times)), float(np.percentile(times, 99))
    print(f"tracker: {mean:.3f} ms/frame mean, {p99:.3f} ms p99 ({n_objects} objects)")
    if n_objects in BUDGET_MS:
        assert mean < BUDGET_MS[n_objects]
