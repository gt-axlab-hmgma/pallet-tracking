"""Replays tests/golden/zones/*.json through the engine.

The fixtures are language-neutral on purpose: the future C++ runtime must produce the same
events for the same input.
"""

import json
from pathlib import Path

import pytest

from palletvision_zones.config import config_from_dict
from palletvision_zones.engine import Track, ZoneEngine

GOLDEN = Path(__file__).resolve().parent / "golden" / "zones"
FIXTURES = sorted(GOLDEN.glob("*.json"))


def test_fixtures_exist() -> None:
    assert len(FIXTURES) >= 4


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_golden_sequence(path: Path) -> None:
    case = json.loads(path.read_text())
    zones = config_from_dict(
        {"camera_id": "golden", "image_size": [1000, 1000], "zones": case["zones"]}
    ).zones
    engine = ZoneEngine(zones, hysteresis_frames=case["hysteresis_frames"])
    for i, frame in enumerate(case["frames"]):
        tracks = [Track(t["track_id"], t["object_type"], tuple(t["box"])) for t in frame["tracks"]]
        actual = [
            {
                "event_type": e.event_type,
                "zone_id": e.zone_id,
                "track_id": e.track_id,
                "object_type": e.object_type,
            }
            for e in engine.update(tracks)
        ]
        assert actual == frame["expect"], f"{path.stem} frame {i}"
