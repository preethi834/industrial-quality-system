# Industrial Quality AI Engineering System

An integrated AI engineering platform for industrial quality management.

## Projects

### Project 1: Active Root-Cause Diagnosis Agent
An AI-powered system for identifying and diagnosing industrial process failures.

### Project 2: Open-World Defect Discovery
A computer vision system for detecting, discovering, and classifying unknown industrial defects.

## Technology Stack

- Python 3.11
- FastAPI
- Docker
- PostgreSQL
- Neo4j
- PyTorch
- Streamlit
- MLflow

## Development Environment

- Ubuntu on WSL2
- Docker Desktop
- Visual Studio Code
- uv for Python dependency management

## Local infrastructure (Postgres + pgvector, Neo4j)

Run everything from the repository root (WSL2 shell, Docker Desktop running).

```bash
make setup                  # uv sync --all-packages --frozen
cp .env.example .env        # then set the two passwords (Neo4j: 8+ chars) and IQS_API__KEYS
make up                     # starts the "core" profile and waits until both are healthy
make db-migrate             # alembic upgrade head
make psql                   # or: make cypher
make build && make up PROFILES=apps   # + vision API on :8001, diagnosis API on :8002
make down                   # stops containers; data volumes are kept
make test-integration       # testcontainers; run after `make down`, not alongside the stack
```

| Service | Image (pinned by digest) | Host address | Idle memory (measured) |
| --- | --- | --- | --- |
| PostgreSQL 16 + pgvector 0.8.7 | `pgvector/pgvector:0.8.7-pg16-bookworm` | `127.0.0.1:5433` | 38.6 MiB (limit 1 GiB) |
| Neo4j 5.26 LTS community | `neo4j:5.26.33-community` | `127.0.0.1:7687` (Bolt), `:7474` (browser) | 974 MiB (limit 2 GiB; heap 512m–1G, page cache 512m) |

Idle figures: `docker stats` three minutes after `make up`, empty databases, WSL2 VM with 9.7 GiB.
Unpacked images take 1.58 GB on disk (Neo4j 934 MB, pgvector 647 MB).

- **One set of credentials.** `.env` holds only `IQS_*` variables. The Python settings read them,
  and compose builds `POSTGRES_*` and `NEO4J_AUTH` from the same values.
- **Port 5433, not 5432,** because a PostgreSQL installed on Windows usually holds 5432 and WSL
  would silently reach it instead of the container.
- **Migrations** live in `infrastructure/database/` (Alembic, one history for all service schemas:
  `diagnosis`, `vision`, `qms`, `platform`). New table: define the model on
  `iqs_common.db.Base`, import its module in `migrations/env.py`, then
  `make db-revision m="..."` and review the generated file.

### Service APIs

| Service | Port | Routes today |
| --- | --- | --- |
| vision | 8001 | `GET /health`, `GET /ready` (Postgres), `GET /v1/capabilities` |
| diagnosis | 8002 | `GET /health`, `GET /ready` (Postgres + Neo4j), `GET /v1/capabilities` |

`/v1/*` needs the header `X-API-Key: <one of IQS_API__KEYS>`. OpenAPI docs at `/docs`.

```bash
curl http://127.0.0.1:8002/ready
curl -H "X-API-Key: $KEY" http://127.0.0.1:8002/v1/capabilities
```
