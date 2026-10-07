import json
from dataclasses import replace

import psycopg

from palletvision_zones.config import ZoneConfig, config_from_dict, config_to_dict


class ZoneConfigConflict(Exception):
    """A version number already exists in the database with different polygons."""


class ZoneConfigStore:
    """Immutable, versioned zone configs in Postgres.

    Every save is a new row. A version is never edited, so an event's (camera_id,
    config_version) always points at exactly the polygons it was computed against.
    """

    def __init__(self, url: str, saved_by: str = "unknown") -> None:
        self._conn = psycopg.connect(url)
        self._saved_by = saved_by

    def latest_version(self, camera_id: str) -> int:
        row = self._conn.execute(
            "SELECT coalesce(max(version), 0) FROM zone_configs WHERE camera_id = %s", (camera_id,)
        ).fetchone()
        self._conn.rollback()
        return int(row[0]) if row else 0

    def get(self, camera_id: str, version: int) -> ZoneConfig | None:
        row = self._conn.execute(
            "SELECT image_size, zones FROM zone_configs WHERE camera_id = %s AND version = %s",
            (camera_id, version),
        ).fetchone()
        self._conn.rollback()
        if row is None:
            return None
        return config_from_dict(
            {"camera_id": camera_id, "version": version, "image_size": row[0], "zones": row[1]}
        )

    def latest(self, camera_id: str) -> ZoneConfig | None:
        """The highest stored version for a camera, or None if it has no saved zones."""
        version = self.latest_version(camera_id)
        return self.get(camera_id, version) if version else None

    def save(self, config: ZoneConfig) -> bool:
        """Insert a version. Returns False if the identical version is already stored.

        Raises ZoneConfigConflict if the version exists with different content.
        """
        data = config_to_dict(config)
        with self._conn.transaction():
            cur = self._conn.execute(
                "INSERT INTO zone_configs (camera_id, version, image_size, zones, saved_by) "
                "VALUES (%s, %s, %s, %s::jsonb, %s) ON CONFLICT (camera_id, version) DO NOTHING",
                (
                    config.camera_id,
                    config.version,
                    list(config.image_size),
                    json.dumps(data["zones"]),
                    self._saved_by,
                ),
            )
            if cur.rowcount == 1:
                return True
        existing = self.get(config.camera_id, config.version)
        # the reference frame path is not part of the stored geometry
        if existing != replace(config, reference_frame=None):
            raise ZoneConfigConflict(
                f"{config.camera_id} version {config.version} already exists with different zones"
            )
        return False

    def close(self) -> None:
        self._conn.close()
