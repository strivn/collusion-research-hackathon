"""A run is found by the experiment name it was started with, whatever its case.

Postgres folds unquoted identifiers to lowercase, so a run started as
"rsmoke_S1_rep1" lives in the schema "rsmoke_s1_rep1". These tests use a fake
asyncpg pool, so they need no database.
"""

import gc
from typing import Any, cast

import asyncpg
import pytest

from magentic_marketplace.platform.database.postgresql import postgresql
from magentic_marketplace.platform.database.postgresql.postgresql import (
    PostgreSQLDatabaseController,
)

EXPERIMENT_NAME = "rsmoke_S1_rep1"
STORED_SCHEMA = "rsmoke_s1_rep1"
EXISTENCE_QUERY_MARKER = "information_schema.schemata"


class FakeConnection:
    """Connection that knows one lowercase schema and records every SQL statement."""

    def __init__(self) -> None:
        """Start with no recorded statements."""
        self.statements: list[str] = []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """Answer the schema existence query; count queries return 0."""
        if EXISTENCE_QUERY_MARKER in sql:
            return args[0] == STORED_SCHEMA
        self.statements.append(sql)
        return 0

    async def execute(self, sql: str, *args: Any) -> str:
        """Record the statement."""
        self.statements.append(sql)
        return "OK 0"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        """Record the statement and return no rows."""
        self.statements.append(sql)
        return []


class FakeAcquire:
    """Result of `pool.acquire()`: awaitable and usable with `async with`."""

    def __init__(self, conn: FakeConnection) -> None:
        """Wrap the shared connection."""
        self._conn = conn

    def __await__(self):
        """Return the connection when awaited."""

        async def _get() -> FakeConnection:
            return self._conn

        return _get().__await__()

    async def __aenter__(self) -> FakeConnection:
        """Return the connection when entered."""
        return self._conn

    async def __aexit__(self, *exc: object) -> None:
        """Nothing to release."""


class FakePool:
    """Pool that always hands out the same recording connection."""

    def __init__(self) -> None:
        """Create the shared connection."""
        self.conn = FakeConnection()

    def acquire(self) -> FakeAcquire:
        """Hand out the connection."""
        return FakeAcquire(self.conn)

    async def release(self, conn: FakeConnection) -> None:
        """Nothing to release."""


@pytest.fixture
def pool() -> FakePool:
    """Fake asyncpg pool."""
    return FakePool()


@pytest.fixture
def controller(pool: FakePool, monkeypatch: pytest.MonkeyPatch):
    """Build a controller for the capitalised name, with metrics dumps disabled."""
    monkeypatch.setattr(postgresql, "_start_metrics_timer", lambda *_: None)
    monkeypatch.setattr(postgresql, "_dump_metrics_to_file", lambda *_: None)
    db = PostgreSQLDatabaseController(cast(asyncpg.Pool, pool), schema=EXPERIMENT_NAME)
    yield db
    # Collect the controller while the metrics dump is still patched out.
    del db
    gc.collect()


async def test_existing_run_named_with_capitals_is_found(
    controller: PostgreSQLDatabaseController,
) -> None:
    """Scenario: A run named with capitals can be scored (schema is found)."""
    await controller.initialize(mode="existing")


async def test_queries_after_finding_the_run_target_the_lowercase_schema(
    controller: PostgreSQLDatabaseController, pool: FakePool
) -> None:
    """Scenario: A run named with capitals can be scored (reads hit its schema)."""
    await controller.initialize(mode="existing")
    await controller.agents.count()
    await controller.actions.count()
    await controller.logs.count()

    assert len(pool.conn.statements) == 3
    for sql in pool.conn.statements:
        assert f"{STORED_SCHEMA}." in sql, sql
        assert EXPERIMENT_NAME not in sql, sql
