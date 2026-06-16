-- Ensure required PostgreSQL extensions are available.
-- Note: the `paper_notebook` database is already created by the postgres image
-- via POSTGRES_DB env var; this script only enables extensions.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
