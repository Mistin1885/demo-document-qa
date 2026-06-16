"""Shared pytest fixtures for integration and unit tests.

Database strategy
-----------------
Integration tests use the Docker Compose postgres instance
(``paper_notebook`` DB, postgres/postgres).  Each test function runs inside a
SAVEPOINT which is rolled back after the test so that:
1. Tests are fully isolated from each other.
2. No ``DROP``/``CREATE`` round-trip per test (fast).
3. Schema is created once at session start via ``Base.metadata.create_all``.

Environment variables are injected here so tests don't need a real ``.env``
file.  Tests that need only the service layer (unit tests with a real DB)
reuse the same ``db_session`` fixture.  Pure unit tests that mock the DB can
ignore it entirely.

Phase 3.1 (Chat service + API) and Phase 3.2 (Session service + API) both
import from this file — **do not remove or rename existing fixtures**.
Phase 3.3 adds ``make_document`` and ``tmp_storage_dir`` fixtures.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text as _text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# Inject required env vars before any app module is imported
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/paper_notebook",
)

os.environ.setdefault("DATABASE_URL", TEST_DATABASE_URL)
os.environ.setdefault("APP_ENCRYPTION_KEY", "test-encryption-key-for-pytest")
os.environ.setdefault("APP_ENV", "test")

# ---------------------------------------------------------------------------
# App + ORM imports (after env vars are set)
# ---------------------------------------------------------------------------

from app.db import get_session  # noqa: E402
from app.main import app  # noqa: E402

# ---------------------------------------------------------------------------
# Engine scoped to the entire test session
# ---------------------------------------------------------------------------

_async_engine = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
_async_sessionmaker = async_sessionmaker(
    bind=_async_engine, expire_on_commit=False, class_=AsyncSession
)


@pytest.fixture(scope="session")
def event_loop():
    """Provide a single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True, loop_scope="session")
async def create_tables() -> AsyncGenerator[None, None]:
    """Verify the schema is present before the test session starts.

    The ``paper_notebook`` database schema is managed by Alembic; tests must
    not call ``Base.metadata.create_all`` or ``drop_all`` because:

    1. ``create_all(checkfirst=True)`` still tries to create indexes even when
       the tables already exist, causing ``DuplicateTableError`` from Alembic-
       managed indexes.
    2. ``drop_all`` at teardown would destroy the shared production database.

    Instead we simply check connectivity and confirm the ``chats`` table is
    accessible.  Per-test isolation is provided by SAVEPOINT rollback in
    ``db_session``.
    """
    async with _async_engine.connect() as conn:
        # Lightweight check — raises if the DB or table is missing.
        await conn.execute(_text("SELECT 1 FROM chats LIMIT 0"))
    yield
    # No teardown: schema is owned by Alembic.


@pytest_asyncio.fixture(loop_scope="session")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` whose transaction is rolled back after each test.

    Each test gets a fresh ``AsyncSession``.  SQLAlchemy's autobegin implicitly
    starts a transaction on first use.  The ``finally`` block always rolls back,
    guaranteeing a clean database state per test without schema teardown.

    We rely on autobegin (do NOT call ``session.begin()`` explicitly) so that
    asyncpg's connection state machine stays consistent.
    """
    async with _async_sessionmaker() as session:
        try:
            yield session
        finally:
            await session.rollback()


@pytest_asyncio.fixture(loop_scope="session")
async def api_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Return an ``AsyncClient`` wired to the FastAPI app with the test DB session.

    The app's ``get_session`` dependency is overridden to return the same
    ``db_session`` used by service-layer assertions, so tests can inspect DB
    state without a second connection.
    """

    async def _override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client
    app.dependency_overrides.pop(get_session, None)


# ---------------------------------------------------------------------------
# Phase 3.3 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage_dir(tmp_path: Path) -> Path:
    """Return a temporary directory to use as the blob storage root.

    Tests that exercise ``LocalBlobStorage`` should inject this path so that
    no files are written to the real ``data/storage/`` directory.
    """
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    return tmp_path  # LocalBlobStorage appends "/storage" internally


@pytest_asyncio.fixture(loop_scope="session")
async def make_document(db_session: AsyncSession, tmp_storage_dir: Path):  # type: ignore[no-untyped-def]
    """Factory fixture: create and persist a Document + ChatDocument association.

    Usage::

        doc = await make_document(chat_id, filename="paper.pdf")
    """
    from app.models.domain import DocumentRead
    from app.services import document_service
    from app.storage.local import LocalBlobStorage

    async def _factory(
        chat_id,  # type: ignore[no-untyped-def]
        *,
        filename: str = "test.pdf",
        file_bytes: bytes = b"%PDF-1.4 fake pdf content",
        mime_type: str = "application/pdf",
    ) -> DocumentRead:
        from app.services.vespa_indexer import NullVespaIndexer

        storage = LocalBlobStorage(root=tmp_storage_dir)
        indexer = NullVespaIndexer()
        return await document_service.upload_document(
            db_session,
            chat_id,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            storage=storage,
            indexer=indexer,
        )

    return _factory
