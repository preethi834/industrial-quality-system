# Technical decisions

ADR-style records: context, decision, consequences. Newest at the bottom.
Status values: accepted, superseded (by ADR-n), rejected.

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
