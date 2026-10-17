import argparse
import os
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import psycopg

DEFAULT_URL = "postgresql://palletvision:palletvision@localhost:5432/palletvision"


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def connection_url() -> str:
    """PV_DATABASE_URL wins; otherwise build it from PV_DB_* in the environment or ./.env."""
    if url := os.environ.get("PV_DATABASE_URL"):
        return url
    env = {**_read_dotenv(Path(".env")), **os.environ}
    user = env.get("PV_DB_USER", "palletvision")
    password = env.get("PV_DB_PASSWORD", "palletvision")
    name = env.get("PV_DB_NAME", "palletvision")
    port = env.get("PV_DB_PORT", "5432")
    host = env.get("PV_DB_HOST", "localhost")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def _migration_files() -> dict[str, str]:
    root = resources.files("palletvision_events") / "migrations"
    return {
        f.name.removesuffix(".sql"): f.read_text()
        for f in sorted(root.iterdir(), key=lambda f: f.name)
        if f.name.endswith(".sql")
    }


def _ensure_table(conn: psycopg.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    conn.commit()


def applied_versions(conn: psycopg.Connection) -> list[str]:
    _ensure_table(conn)
    return [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]


def pending_versions(conn: psycopg.Connection) -> list[str]:
    done = set(applied_versions(conn))
    return [v for v in _migration_files() if v not in done]


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply every unapplied migration, each in its own transaction. Returns what was applied."""
    files = _migration_files()
    applied: list[str] = []
    for version in pending_versions(conn):
        with conn.transaction():
            conn.execute(files[version])
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        applied.append(version)
    return applied


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-db", description="Manage the local events database.")
    p.add_argument("command", choices=["migrate", "status"])
    p.add_argument("--url", default=None, help="Postgres URL (default: from PV_DB_* / .env)")
    args = p.parse_args(argv)
    url = args.url or connection_url()
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            if args.command == "migrate":
                applied = migrate(conn)
                print(f"applied: {', '.join(applied)}" if applied else "already up to date")
            else:
                print("applied:", ", ".join(applied_versions(conn)) or "none")
                print("pending:", ", ".join(pending_versions(conn)) or "none")
    except psycopg.OperationalError as exc:
        raise SystemExit(
            f"pv-db: cannot reach Postgres at {url.split('@')[-1]}\n"
            f"  start it with: docker compose up -d db\n  ({exc})"
        ) from exc
