# Technical decisions

ADR-style records: context, decision, consequences. Newest at the bottom.
Status values: accepted, superseded (by ADR-n), rejected.

Decisions made before step 0d (two services, pgvector, fixed taxonomy, Python 3.11, uv
workspace, CPU-only torch) are justified in `TECHNICAL_BLUEPRINT.md` §0, §2 and §3 and are not
repeated here.

---

## ADR-001 — Local infrastructure: compose, pgvector image, Neo4j memory, Alembic location (step 0d)

**Status:** accepted · 2026-11-17

**Context.** The blueprint (§2.4) fixes PostgreSQL 16 + pgvector as the system of record and
Neo4j for the knowledge graph. The development machine is a WSL2 VM limited to 10 GB
(9.7 GiB visible), which must later also hold Ollama and a PyTorch process. A PostgreSQL 17
service is installed on the Windows host and listens on 5432.

**Decisions.**

1. **Images, pinned by tag and digest.** `pgvector/pgvector:0.8.7-pg16-bookworm` (pgvector 0.8.7,
   PostgreSQL 16.16) and `neo4j:5.26.33-community` (5.26 is the Neo4j 5 LTS line). The Debian
   release is part of the pgvector tag so the base OS cannot change underneath us. Integration
   tests read the image from `docker-compose.yml`, so tests and compose cannot drift apart.
2. **Neo4j memory caps.** Heap 512m initial / 1G max, page cache 512m, container `mem_limit` 2g;
   Postgres `mem_limit` 1g. Measured idle use: Neo4j 974 MiB, Postgres 38.6 MiB.
3. **Compose profiles.** Both services are in `core`; `vision`, `llm`, `observability` are added by
   the steps that need them.
4. **Single source for credentials.** Compose interpolates `POSTGRES_*` and `NEO4J_AUTH` from the
   `IQS_*` variables in the repo-root `.env` (Makefile passes `--project-directory .`). Missing
   values fail at `docker compose config` time (`${VAR:?}`); ports bind to 127.0.0.1 only.
5. **Host port 5433** for Postgres, to avoid the Windows PostgreSQL on 5432 (which WSL reaches via
   localhost, so a wrong port would silently hit the wrong server).
6. **Alembic in `infrastructure/database/`,** not inside one package: one migration history for
   all service schemas. `env.py` takes the URL from `iqs_common` settings (never from
   `alembic.ini`) and restricts autogenerate to `diagnosis`, `vision`, `qms`, `platform` with
   `include_name` / `include_object`. `alembic_version` stays in `public`, because Alembic creates
   it before the first migration creates our schemas.
7. **Baseline migration 0001:** `vector` extension, the four schemas, and the `platform.events`
   outbox (model `iqs_common.outbox.OutboxEvent`; CHECK on status ∈ {pending, processed, failed}
   and attempts ≥ 0; row id = `event_id`, so a duplicate event fails on the primary key).
   `pg_trgm` is not installed; BM25-style search uses `tsvector` (blueprint §2.4). Domain tables
   arrive with the steps that own them; the text embedding dimension is chosen in step 3.

**Consequences.**
- `alembic check` (drift between models and migrations) runs in the integration tests, together
  with a test proving the check fails when a model has no migration.
- Every module defining tables must be imported in `migrations/env.py`, or autogenerate would
  propose dropping its tables.
- Compose and testcontainers should not run at the same time on this VM (two Neo4j JVMs).

---

## ADR-002 — Service skeletons: shared HTTP plumbing, API conventions, images (step 0e)

**Status:** accepted · 2026-11-17

**Context.** Blueprint §2.3/§2.6 requires `/health`, `/ready`, API-key auth on every non-health
endpoint, CORS limited to the dashboard, JSON logs with a request ID and versioned `/v1` routes,
for two services (vision, diagnosis) with separate Docker images. `iqs-common` did not depend on
FastAPI.

**Decisions.**

1. **Shared plumbing in `iqs_common.api`, behind an optional extra `iqs-common[api]`** (FastAPI).
   Only the two HTTP services request the extra, so the simulator and MCP server stay free of it.
   `create_service_app()` builds the app; each service passes its readiness checks and feature
   flags. Services use an app factory (`uvicorn --factory`), so imports have no side effects.
2. **Endpoints.** `GET /health`: liveness, no dependencies, no auth. `GET /ready`: all checks run
   concurrently in threads with a timeout (`IQS_API__READY_TIMEOUT_S`, default 2 s); 200 or 503
   with per-check `ok` / `timeout` / `error: <ExceptionType>` (details in logs only). Vision checks
   Postgres; diagnosis checks Postgres and Neo4j. `GET /v1/capabilities`: service, version,
   `api_version`, feature flags (all false at step 0); requires the API key.
3. **Auth.** Header `X-API-Key`; keys from `IQS_API__KEYS` as a comma list (rotation: `new,old`);
   constant-time comparison against every key; no keys configured → all `/v1` requests 401
   (fail closed). Compose refuses to start the apps without `IQS_API__KEYS`.
4. **Ports.** vision 8001, diagnosis 8002, bound to 127.0.0.1. Compose profile `apps` = core + both
   services (Postgres and Neo4j carry both `core` and `apps`).
5. **Images.** `deploy/docker/<svc>.Dockerfile`: `python:3.11.18-slim-bookworm` and
   `ghcr.io/astral-sh/uv:0.13.8`, both pinned by digest; multi-stage `uv sync --frozen --no-dev
   --package iqs-<svc> --no-editable` into `/opt/venv`; dependencies layer cached separately from
   source; non-root user (uid 10001); Docker `HEALTHCHECK` on `/health`; `.dockerignore` is an
   allow-list (no `.env`, `.venv`, `.git`, data or models in the build context).

**Measured.** Compressed image size (`docker save`): vision 718 MB, diagnosis 616 MB; unpacked
venv 2.2 GB / 2.0 GB (torch 814 MB). Wheels downloaded during the first build: diagnosis 409 MiB,
vision 158 MiB more (shared uv cache mount). Idle memory with the full `apps` profile: vision
117 MiB, diagnosis 177 MiB, Postgres 37 MiB, Neo4j 1007 MiB. First build 10 min 21 s.

**Consequences.**
- Each new HTTP route goes in a router passed to `create_service_app(routers=...)` and is
  automatically behind the API key.
- torch is in both images (diagnosis needs it for sentence-transformers); if image size becomes a
  problem, CPU-only torch is already used and the next lever is a smaller embedding stack.

---

## ADR-003 — CI, pre-commit and one mypy command (step 0f)

**Status:** accepted · 2026-11-17

**Context.** Blueprint: CI runs lint, types and unit + API tests on every push; integration tests
with testcontainers on every push; a compose smoke test; pre-commit with ruff, mypy, a secret
scan and a large-file guard. mypy could not check all trees in one run ("Source file found twice
under different module names").

**Decisions.**

1. **mypy:** `files = ["packages", "tests", "infrastructure"]` and every `packages/*/src` on
   `mypy_path` (with `explicit_package_bases`). Package code gets its import name
   (`iqs_common.db`), tests and migrations keep path-based names, so `uv run mypy` checks all 38
   files with no duplicates.
2. **pre-commit hooks are all `repo: local`** and run through `uv run --frozen`, so hook versions
   are the `uv.lock` versions (identical in CI) and pre-commit downloads nothing. Hooks: large
   files (>1 MB), merge-conflict markers, case conflicts, YAML/TOML validity, end-of-file,
   trailing whitespace, private keys, detect-secrets (baseline `.secrets.baseline`, empty), `uv
   lock --check`, ruff check, ruff format, mypy.
3. **Secret scan:** detect-secrets 1.5.0 (pure Python, lockable). Known false positives are
   marked inline with `# pragma: allowlist secret (<reason>)` rather than hidden in the baseline,
   so every exception is visible in review.
4. **Workflows.** `ci.yml` (every push and PR): job `checks` = `uv sync --all-packages --frozen`,
   all pre-commit hooks, unit + API tests with branch coverage of `iqs_*`; job `integration` =
   `pytest -m integration` (testcontainers, images read from the compose file). `docker.yml`
   (push/PR to `main`, manual): build both images, `make up PROFILES=apps`,
   `scripts/smoke_test.sh`, logs on failure, `make down`. `eval.yml` arrives with the evaluation
   harness in step 3.
5. **Actions referenced by major tag** (`actions/checkout@v4`, `astral-sh/setup-uv@v6`) with uv
   pinned to 0.13.8 and `permissions: contents: read`. Pinning actions to commit SHAs is open
   (needs a GitHub lookup; see KNOWN_ISSUES).

**Consequences.** `make check` runs exactly what the first CI job runs. Integration tests are
excluded from plain `pytest` (`-m 'not integration'` in addopts) and run with
`make test-integration`. Workflows have not run on GitHub yet (nothing pushed).
