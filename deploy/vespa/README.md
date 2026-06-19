# Vespa Deployment Notes

## Runtime vs. Application Package

The `vespaengine/vespa:8` container started by `docker compose` provides only the **runtime** (config server + content node). It does **not** include the application package (schema, rank profiles, etc.) at startup.

The application package lives in `deploy/vespa/application/` and is deployed separately in **Phase 6** via the script:

```bash
uv run python scripts/deploy_vespa.py
```

This script connects to the Vespa config server at `http://localhost:19072` for the compose stack, uploads the application package (including `schemas/document_chunk.sd`), and waits for deployment to complete. Run it after `docker compose up -d vespa` reports a healthy container.

## Quick-start

```bash
# Start Vespa (and Postgres) in the background
docker compose -f deploy/docker-compose.yml up -d postgres vespa

# Wait until Vespa config server is healthy (up to ~5 minutes on first run)
docker compose -f deploy/docker-compose.yml ps

# Deploy the application package (Phase 6+)
uv run python scripts/deploy_vespa.py --config-url http://localhost:19072
```

## Ports

| Port  | Purpose                                  |
|-------|------------------------------------------|
| 8089  | Query / feed endpoint on the host (`VESPA_ENDPOINT=http://localhost:8089`) |
| 19072 | Config server on the host (deploy / health check)   |

## Data persistence

Vespa state is stored in the named Docker volume `vespa_data`. To reset Vespa to a clean state, stop the container and remove the volume:

```bash
docker compose -f deploy/docker-compose.yml down -v
```
