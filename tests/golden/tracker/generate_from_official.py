"""Regenerates the *.json fixtures here by running the official ByteTrack on scripted scenes.

Not part of the test suite: it needs a checkout of https://github.com/ifzhang/ByteTrack (MIT) and
the ``lap`` package. Run from the repo root:

    uv run --with lap python tests/golden/tracker/generate_from_official.py /path/to/ByteTrack SHA

``SHA`` is the commit of the checkout; it is written into every fixture.

Only the data it produces is committed (detections in, official tracks out), never their code.
Two shims let the official files run unchanged: torch and cv2 are stubbed because the tracker
never calls them, and ``cython_bbox`` is replaced by our own IoU. The official one adds one pixel
to every width and height, which this tracker deliberately does not do (see ``bytetrack.py``).
"""

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np

from palletvision_track.boxes import pairwise_iou

HERE = Path(__file__).resolve().parent
CONFIG = {
    "high_thresh": 0.6,
    "low_thresh": 0.1,
    "match_thresh": 0.8,
    "second_thresh": 0.5,
    "unconfirmed_thresh": 0.7,
    "max_time_lost": 30,
    "fuse_score": True,
}
Frame = tuple[list[list[float]], list[float]]


def load_official(checkout: Path) -> type:
    """Import ``BYTETracker`` straight from the checkout, skipping ``yolox/__init__``."""
    np.float = float  # type: ignore[attr-defined]  # the official code predates numpy 1.24
    for name in ("torch", "torch.nn", "torch.nn.functional"):
        sys.modules[name] = types.ModuleType(name)
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")
    cy = types.ModuleType("cython_bbox")
    cy.bbox_overlaps = lambda a, b: pairwise_iou(a, b)  # type: ignore[attr-defined]
    sys.modules["cython_bbox"] = cy
    for name, sub in (("yolox", ""), ("yolox.tracker", "yolox/tracker")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(checkout / sub)]
        sys.modules[name] = pkg
    from yolox.tracker.byte_tracker import BYTETracker  # type: ignore[import-not-found]

    return BYTETracker  # type: ignore[no-any-return]


def box(cx: float, cy: float, w: float = 40.0, h: float = 80.0) -> list[float]:
    return [round(v, 2) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]


def single_mover() -> list[Frame]:
    return [([box(100 + 5 * f, 100)], [0.9]) for f in range(40)]


def crossing() -> list[Frame]:
    return [([box(100 + 10 * f, 100), box(500 - 10 * f, 130)], [0.9, 0.85]) for f in range(41)]


def occlusion_gap() -> list[Frame]:
    return [([], []) if 20 <= f < 32 else ([box(100 + 2 * f, 100)], [0.9]) for f in range(50)]


def low_score_recovery() -> list[Frame]:
    return [([box(100 + 4 * f, 100)], [0.3 if 15 <= f < 25 else 0.9]) for f in range(40)]


def false_positive_and_clutter() -> list[Frame]:
    rng = np.random.default_rng(3)
    frames: list[Frame] = []
    for f in range(40):
        boxes, scores = [box(300 + 2 * f, 200)], [0.9]
        if f == 10:
            boxes.append(box(700, 600))  # appears for exactly one frame
            scores.append(0.95)
        for _ in range(rng.integers(0, 3)):  # weak clutter that should never become a track
            boxes.append(box(*(rng.uniform(50, 950, 2))))
            scores.append(round(float(rng.uniform(0.15, 0.5)), 2))
        frames.append((boxes, scores))
    return frames


def expiry() -> list[Frame]:
    return [([box(200, 200)], [0.9]) if f < 10 or f >= 50 else ([], []) for f in range(60)]


def busy_random() -> list[Frame]:
    """Several moving objects with misses, score dips, clutter and scores sitting on thresholds."""
    rng = np.random.default_rng(7)
    n = 6
    pos = rng.uniform(100, 900, (n, 2))
    vel = rng.normal(0, 3, (n, 2))
    size = rng.uniform(30, 120, (n, 2))
    born = rng.integers(0, 40, n)
    died = rng.integers(60, 150, n)
    frames: list[Frame] = []
    for f in range(120):
        pos += vel + rng.normal(0, 0.8, pos.shape)
        boxes: list[list[float]] = []
        scores: list[float] = []
        for i in range(n):
            if not born[i] <= f < died[i]:
                continue
            r = rng.random()
            if r < 0.08:
                continue
            score = rng.uniform(0.75, 0.98) if r > 0.30 else rng.uniform(0.15, 0.58)
            if 0.2 < r < 0.22:
                score = rng.choice([0.6, 0.7])
            c = pos[i] + rng.normal(0, 1.5, 2)
            boxes.append(box(c[0], c[1], *(size[i] * rng.uniform(0.95, 1.05, 2))))
            scores.append(round(float(score), 3))
        for _ in range(rng.integers(0, 3)):
            boxes.append(box(*rng.uniform(0, 1000, 2), *rng.uniform(20, 100, 2)))
            scores.append(round(float(rng.uniform(0.05, 0.9)), 3))
        frames.append((boxes, scores))
    return frames


SCENES = {
    "single_mover": (
        "One object moving at a steady speed keeps one id.",
        single_mover,
    ),
    "crossing": (
        "Two objects walk through each other. Ids are whatever the official tracker assigns.",
        crossing,
    ),
    "occlusion_gap": (
        "An object disappears for 12 frames (shorter than max_time_lost) and keeps its id.",
        occlusion_gap,
    ),
    "low_score_recovery": (
        "An object's score drops to 0.3 for 10 frames; the second round keeps its id.",
        low_score_recovery,
    ),
    "false_positive_and_clutter": (
        "A one-frame high-score box and weak clutter boxes never produce an output track.",
        false_positive_and_clutter,
    ),
    "expiry": (
        "An object gone for 40 frames (longer than max_time_lost) comes back with a new id.",
        expiry,
    ),
    "busy_random": (
        "Six objects with misses, score dips, clutter and scores exactly on thresholds.",
        busy_random,
    ),
}


def main(checkout: Path, sha: str) -> None:
    official = load_official(checkout)
    for name, (description, build) in SCENES.items():
        from yolox.tracker.basetrack import BaseTrack  # type: ignore[import-not-found]

        BaseTrack._count = 0
        args = Namespace(track_thresh=0.6, track_buffer=30, match_thresh=0.8, mot20=False)
        tracker = official(args, frame_rate=30)
        frames = []
        for boxes, scores in build():
            det = np.column_stack([boxes, scores]) if boxes else np.zeros((0, 5))
            out = tracker.update(det.copy(), (1000, 1000), (1000, 1000))
            expect = sorted(
                (
                    {"track_id": int(t.track_id), "box": [round(float(v), 3) for v in t.tlbr]}
                    for t in out
                ),
                key=lambda e: e["track_id"],
            )
            frames.append({"boxes": boxes, "scores": scores, "expect": expect})
        case = {
            "description": description,
            "source": f"official ByteTrack (MIT) at {sha}; IoU without the +1 pixel",
            "config": CONFIG,
            "frames": frames,
        }
        (HERE / f"{name}.json").write_text(json.dumps(case) + "\n")
        print(f"{name}: {len(frames)} frames")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
