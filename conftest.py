import uuid
from collections.abc import Iterator

import psycopg
import pytest

from palletvision_events.db import connection_url, migrate


def _admin_url() -> str:
    base, _, _ = connection_url().rpartition("/")
    return f"{base}/postgres"


@pytest.fixture(scope="session")
def _server_available() -> bool:
    try:
        with psycopg.connect(_admin_url(), connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


@pytest.fixture
def db_url(_server_available: bool) -> Iterator[str]:
    """A fresh, migrated database per test. Skips when no Postgres is running."""
    if not _server_available:
        pytest.skip("postgres not running (docker compose up -d db)")
    name = f"pv_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_admin_url(), autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    base, _, _ = connection_url().rpartition("/")
    url = f"{base}/{name}"
    try:
        with psycopg.connect(url) as conn:
            migrate(conn)
        yield url
    finally:
        with psycopg.connect(_admin_url(), autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
