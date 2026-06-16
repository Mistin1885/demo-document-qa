"""Async database engine and session factory.

Usage
-----
FastAPI dependency injection::

    from app.db import get_session

    @router.get("/example")
    async def example(session: AsyncSession = Depends(get_session)):
        ...

The engine and sessionmaker are created lazily on first call to avoid
connecting to the database at import time.  Tests can override
``database_url`` via environment variables before importing this module
(or by calling ``get_settings.cache_clear()``).

Rules
-----
- ``expire_on_commit=False`` so ORM objects remain accessible after commit.
- Do NOT import this module from within ORM/domain model files to avoid
  circular imports.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@lru_cache(maxsize=1)
def _get_engine() -> AsyncEngine:
    """Return (or create) the cached async engine.

    The engine is created on first call so that importing ``app.db`` does
    not immediately attempt a database connection.
    """
    from app.config import get_settings  # lazy import — avoids circulars

    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def _get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return (or create) the cached session factory."""
    return async_sessionmaker(
        bind=_get_engine(),
        expire_on_commit=False,
        class_=AsyncSession,
    )


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a transactional ``AsyncSession``.

    The session is automatically closed (and rolled back on unhandled
    exceptions) when the request completes.
    """
    factory = _get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


__all__ = ["get_session"]
