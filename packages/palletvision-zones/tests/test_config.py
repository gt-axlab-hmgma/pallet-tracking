import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from palletvision_zones.config import (
    ConfigError,
    Zone,
    ZoneConfig,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from palletvision_zones.geometry import PolygonError

SCHEMA = Path(__file__).resolve().parents[3] / "contracts" / "zones.schema.json"

CONFIG = ZoneConfig(
    camera_id="cam01",
    image_size=(1920, 1080),
    version=2,
    reference_frame="refs/cam01.png",
    zones=(
        Zone("zone_1", ((112, 980), (860, 940), (910, 610), (240, 620)), min_dwell_s=5),
        Zone("zone_2", ((930, 960), (1800, 1000), (1700, 640), (950, 600))),
    ),
)


def test_dict_matches_contract_schema() -> None:
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator(schema).validate(config_to_dict(CONFIG))


def test_round_trip_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "cam01.yaml"
    save_config(CONFIG, path)
    assert load_config(path) == CONFIG


def test_defaults_when_optional_fields_missing() -> None:
    cfg = config_from_dict(
        {
            "camera_id": "c",
            "image_size": [10, 10],
            "zones": [{"id": "a", "polygon": [[0, 0], [5, 0], [5, 5]]}],
        }
    )
    assert cfg.version == 1
    assert cfg.reference_frame is None
    assert cfg.zones[0].min_dwell_s == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        {"image_size": [10, 10], "zones": []},  # no camera_id
        {"camera_id": "c", "zones": []},  # no image_size
        {"camera_id": "c", "image_size": [10], "zones": []},
        {"camera_id": "c", "image_size": [10, 10], "zones": [{"polygon": [[0, 0]] * 3}]},
        {
            "camera_id": "c",
            "image_size": [10, 10],
            "zones": [
                {"id": "a", "polygon": [[0, 0], [5, 0], [5, 5]]},
                {"id": "a", "polygon": [[1, 1], [6, 1], [6, 6]]},  # duplicate id
            ],
        },
    ],
)
def test_malformed_configs_raise(bad: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_bad_polygon_is_reported() -> None:
    with pytest.raises(PolygonError):
        config_from_dict(
            {
                "camera_id": "c",
                "image_size": [10, 10],
                "zones": [{"id": "a", "polygon": [[0, 0], [1, 1]]}],
            }
        )
