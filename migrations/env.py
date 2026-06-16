"""Alembic environment configuration for Paper Notebook Agent.

- Pulls the database URL from ``app.config.get_settings().database_url``.
- If the URL contains ``+asyncpg``, it is rewritten to ``+psycopg`` so that
  Alembic (which runs synchronously) uses the sync driver.
- ``target_metadata`` is set to ``Base.metadata`` from the ORM so that
  ``--autogenerate`` picks up all table/index/FK definitions.
"""

from __future__ import annotations

import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Alembic Config object (provides access to alembic.ini values)
# ---------------------------------------------------------------------------

config = context.config

# Configure Python logging from alembic.ini if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Inject database URL from application settings
# ---------------------------------------------------------------------------


def _get_sync_url() -> str:
    """Return a synchronous SQLAlchemy URL for Alembic.

    Converts ``postgresql+asyncpg://`` → ``postgresql+psycopg://`` so that
    Alembic can run synchronously regardless of what the runtime app uses.
    Any other ``+asyncpg`` variant is also normalised.
    """
    from app.config import get_settings

    url: str = get_settings().database_url
    # Replace async driver variant with psycopg (sync)
    url = re.sub(r"\+asyncpg\b", "+psycopg", url)
    return url


config.set_main_option("sqlalchemy.url", _get_sync_url())

# ---------------------------------------------------------------------------
# Target metadata (all 10 ORM tables)
# ---------------------------------------------------------------------------

from app.models.orm import Base  # noqa: E402

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Offline migration (generates SQL without a live DB connection)
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine, though an
    Engine is acceptable here as well.  By skipping the Engine creation we
    don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration (uses a live DB connection)
# ---------------------------------------------------------------------------


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
