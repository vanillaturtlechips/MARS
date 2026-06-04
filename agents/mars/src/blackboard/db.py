"""
Database connection factory and migration runner for the MARS blackboard.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import psycopg

from mars.config import DB_DSN

log = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATIONS = ["0001_initial.sql"]


def connect(dsn: str = DB_DSN, autocommit: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(dsn)
    conn.autocommit = autocommit
    return conn


def apply_migrations(conn) -> None:
    """
    Ensure the schema is up to date.  Idempotent: already-applied migrations
    are skipped.  Must be called with autocommit=False (uses a single txn).
    """
    # Bootstrap: schema_migrations may not exist yet on a brand-new DB.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'schema_migrations'
            )
            """
        )
        table_exists: bool = cur.fetchone()[0]

    if not table_exists:
        applied: set[str] = set()
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations")
            applied = {row[0] for row in cur.fetchall()}

    for migration_file in _MIGRATIONS:
        version = migration_file.replace(".sql", "")
        if version in applied:
            log.debug("Migration %s already applied — skipping", version)
            continue

        sql_path = _MIGRATIONS_DIR / migration_file
        sql = sql_path.read_text()
        log.info("Applying migration: %s", version)
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        log.info("Migration %s applied successfully", version)


def ping(dsn: str = DB_DSN) -> bool:
    """Return True if the database is reachable."""
    try:
        with connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as exc:
        log.warning("DB ping failed: %s", exc)
        return False
