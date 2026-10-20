"""Replays tests/golden/tracker/*.json through the tracker.

The fixtures hold detections in and the official ByteTrack's tracks out (see
tests/golden/tracker/generate_from_official.py), so passing means this tracker behaves like the
official one on those scenes. They are language-neutral for the future C++ port.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from palletvision_track.bytetrack import ByteTracker, TrackerConfig

GOLDEN = Path(__file__).resolve().parent / "golden" / "tracker"
FIXTURES = sorted(GOLDEN.glob("*.json"))


def test_fixtures_exist() -> None:
    assert len(FIXTURES) >= 6


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_golden_sequence(path: Path) -> None:
    case = json.loads(path.read_text())
    tracker = ByteTracker(TrackerConfig(**case["config"]))
    for i, frame in enumerate(case["frames"]):
        out = tracker.update(i + 1, frame["boxes"], frame["scores"])
        actual = sorted(((t.track_id, t.box) for t in out), key=lambda e: e[0])
        expect = frame["expect"]
        assert [a[0] for a in actual] == [e["track_id"] for e in expect], f"{path.stem} frame {i}"
        for (_, box), e in zip(actual, expect, strict=True):
            np.testing.assert_allclose(box, e["box"], atol=0.01, err_msg=f"{path.stem} frame {i}")
