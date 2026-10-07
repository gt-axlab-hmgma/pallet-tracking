from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from palletvision_zones.geometry import Point, validate_polygon


class ConfigError(ValueError):
    """Raised when a zone config does not match contracts/zones.schema.json."""


@dataclass(frozen=True)
class Zone:
    id: str
    polygon: tuple[Point, ...]
    min_dwell_s: float = 0.0


@dataclass(frozen=True)
class ZoneConfig:
    camera_id: str
    image_size: tuple[int, int]
    zones: tuple[Zone, ...]
    version: int = 1
    reference_frame: str | None = None


def config_from_dict(data: dict[str, Any]) -> ZoneConfig:
    for key in ("camera_id", "image_size", "zones"):
        if key not in data:
            raise ConfigError(f"missing '{key}'")
    size = data["image_size"]
    if not (isinstance(size, list | tuple) and len(size) == 2):
        raise ConfigError("image_size must be [width, height]")
    zones: list[Zone] = []
    for raw in data["zones"]:
        if "id" not in raw or "polygon" not in raw:
            raise ConfigError("every zone needs 'id' and 'polygon'")
        polygon = tuple((float(x), float(y)) for x, y in raw["polygon"])
        validate_polygon(polygon)
        zones.append(Zone(str(raw["id"]), polygon, float(raw.get("min_dwell_s", 0.0))))
    ids = [z.id for z in zones]
    if len(ids) != len(set(ids)):
        raise ConfigError("zone ids must be unique")
    return ZoneConfig(
        camera_id=str(data["camera_id"]),
        image_size=(int(size[0]), int(size[1])),
        zones=tuple(zones),
        version=int(data.get("version", 1)),
        reference_frame=data.get("reference_frame"),
    )


def config_to_dict(config: ZoneConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "camera_id": config.camera_id,
        "version": config.version,
        "image_size": list(config.image_size),
        "zones": [
            {
                "id": z.id,
                "polygon": [[x, y] for x, y in z.polygon],
                "min_dwell_s": z.min_dwell_s,
            }
            for z in config.zones
        ],
    }
    if config.reference_frame is not None:
        out["reference_frame"] = config.reference_frame
    return out


def load_config(path: Path | str) -> ZoneConfig:
    return config_from_dict(yaml.safe_load(Path(path).read_text()))


def save_config(config: ZoneConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))
