from dataclasses import replace

import psycopg
import pytest

from palletvision_events.zone_store import ZoneConfigConflict, ZoneConfigStore
from palletvision_zones.config import Zone, ZoneConfig

V1 = ZoneConfig(
    "cam01",
    (1280, 720),
    (Zone("z1", ((112, 580), (860, 540), (910, 310), (240, 320)), min_dwell_s=5),),
    version=1,
)
V2 = replace(V1, version=2, zones=(Zone("z1", ((0, 0), (100, 0), (100, 100))),))


def test_save_then_get_round_trips(db_url: str) -> None:
    store = ZoneConfigStore(db_url, saved_by="demo")
    assert store.save(V1) is True
    assert store.get("cam01", 1) == V1
    assert store.get("cam01", 2) is None
    assert store.get("other", 1) is None
    store.close()


def test_latest_version_tracks_saves_per_camera(db_url: str) -> None:
    store = ZoneConfigStore(db_url)
    assert store.latest_version("cam01") == 0
    store.save(V1)
    store.save(V2)
    store.save(replace(V1, camera_id="cam02", version=7))
    assert store.latest_version("cam01") == 2
    assert store.latest_version("cam02") == 7
    store.close()


def test_saving_the_same_version_twice_is_a_no_op(db_url: str) -> None:
    store = ZoneConfigStore(db_url)
    assert store.save(V1) is True
    assert store.save(V1) is False
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT count(*) FROM zone_configs").fetchone() == (1,)
    store.close()


def test_a_version_is_immutable(db_url: str) -> None:
    store = ZoneConfigStore(db_url)
    store.save(V1)
    with pytest.raises(ZoneConfigConflict, match="cam01.*version 1"):
        store.save(replace(V1, zones=V2.zones))  # same version number, different polygons
    assert store.get("cam01", 1) == V1  # untouched
    store.close()


def test_records_who_saved_it(db_url: str) -> None:
    store = ZoneConfigStore(db_url, saved_by="alice")
    store.save(V1)
    store.close()
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT saved_by FROM zone_configs").fetchone() == ("alice",)


def test_reference_frame_path_is_not_part_of_the_stored_geometry(db_url: str) -> None:
    store = ZoneConfigStore(db_url)
    store.save(V1)
    assert store.save(replace(V1, reference_frame="refs/cam01.png")) is False
    store.close()
