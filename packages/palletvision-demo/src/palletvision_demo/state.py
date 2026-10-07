from pathlib import Path
from typing import Any

from palletvision_demo.frame import encode_jpeg, load_reference_frame
from palletvision_demo.settings import Settings
from palletvision_zones.config import ZoneConfig, config_to_dict, load_config

DEFAULT_OBJECT_TYPES = ["pallet_type_a", "pallet_type_b", "pallet_type_c", "pallet_unknown"]


class DemoState:
    """Everything the demo API holds in memory for one camera."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        frame = load_reference_frame(settings.source)
        self.frame_jpeg = encode_jpeg(frame)
        self.image_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.warnings: list[str] = []
        self.object_types = list(DEFAULT_OBJECT_TYPES)
        self.config = self._load_or_empty()

    @property
    def config_path(self) -> Path:
        return self.settings.config_dir / f"{self.settings.camera_id}.yaml"

    def _load_or_empty(self) -> ZoneConfig:
        if self.config_path.exists():
            config = load_config(self.config_path)
            if config.image_size != self.image_size:
                self.warnings.append(
                    f"saved zones were drawn on a {config.image_size[0]}x{config.image_size[1]} "
                    f"frame but this frame is {self.image_size[0]}x{self.image_size[1]}"
                )
            return config
        return ZoneConfig(self.settings.camera_id, self.image_size, ())

    def snapshot(self) -> dict[str, Any]:
        cfg = config_to_dict(self.config)
        return {
            "camera_id": self.settings.camera_id,
            "device_id": self.settings.device_id,
            "image_size": list(self.image_size),
            "version": self.config.version,
            "zones": cfg["zones"],
            "objects": [],
            "object_types": self.object_types,
            "sink": self.settings.sink,
            "warnings": self.warnings,
        }
