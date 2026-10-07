import psycopg

from palletvision_events.db import applied_versions, migrate, pending_versions


def tables(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()
    return {r[0] for r in rows}


def test_migrate_creates_core_tables_and_view(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        names = tables(conn)
        assert {"devices", "cameras", "zone_configs", "events", "schema_migrations"} <= names
        views = conn.execute("SELECT viewname FROM pg_views WHERE schemaname='public'").fetchall()
        assert ("zone_visits",) in views


def test_migrate_is_idempotent(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        before = applied_versions(conn)
        assert before
        assert migrate(conn) == []  # nothing new to apply
        assert applied_versions(conn) == before
        assert pending_versions(conn) == []


def test_events_is_partitioned_by_month_and_has_current_partition(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        strategy = conn.execute(
            "SELECT partstrat FROM pg_partitioned_table WHERE partrelid = 'events'::regclass"
        ).fetchone()
        assert strategy == ("r",)
        children = conn.execute(
            "SELECT count(*) FROM pg_inherits WHERE inhparent = 'events'::regclass"
        ).fetchone()
        assert children is not None and children[0] >= 1


def test_events_has_the_planned_indexes(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        rows = conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'events'").fetchall()
    defs = " ".join(r[0] for r in rows)
    assert "(camera_id, device_ts)" in defs
    assert "(zone_id, device_ts)" in defs
    assert "(device_id, seq)" in defs
