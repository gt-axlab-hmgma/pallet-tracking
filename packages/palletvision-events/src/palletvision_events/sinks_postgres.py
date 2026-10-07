from collections.abc import Sequence

import psycopg

from palletvision_events.record import ZoneEvent

_INSERT = """
INSERT INTO events (event_id, device_id, camera_id, seq, event_type, zone_id, object_type,
                    track_id, config_version, source, device_ts)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id, device_ts) DO NOTHING
"""


class PostgresSink:
    """Idempotent batch writer. Replaying a batch never creates duplicate rows.

    One transaction per batch. Missing monthly partitions, devices and cameras are created on
    the fly so a new Nano can start sending without any setup.
    """

    def __init__(self, url: str) -> None:
        self._conn = psycopg.connect(url)
        self._partitions: set[tuple[int, int]] = set()

    def write(self, events: Sequence[ZoneEvent]) -> None:
        if not events:
            return
        with self._conn.transaction(), self._conn.cursor() as cur:
            months = {(e.device_ts.year, e.device_ts.month): e.device_ts for e in events}
            for key, ts in sorted(months.items()):
                if key not in self._partitions:
                    cur.execute("SELECT pv_ensure_events_partition(%s)", (ts,))
                    self._partitions.add(key)
            for device_id in {e.device_id for e in events}:
                cur.execute(
                    "INSERT INTO devices (device_id, last_seen_at) VALUES (%s, now()) "
                    "ON CONFLICT (device_id) DO UPDATE SET last_seen_at = now()",
                    (device_id,),
                )
            for camera_id, device_id in {(e.camera_id, e.device_id) for e in events}:
                cur.execute(
                    "INSERT INTO cameras (camera_id, device_id) VALUES (%s, %s) "
                    "ON CONFLICT (camera_id) DO NOTHING",
                    (camera_id, device_id),
                )
            cur.executemany(
                _INSERT,
                [
                    (
                        e.event_id, e.device_id, e.camera_id, e.seq, e.event_type, e.zone_id,
                        e.object_type, e.track_id, e.config_version, e.source, e.device_ts,
                    )
                    for e in events
                ],
            )  # fmt: skip

    def close(self) -> None:
        self._conn.close()
