# Testing

How the project is tested, how to run each level, and what exists today (step 0).
Strategy source: `TECHNICAL_BLUEPRINT.md` §7.2–7.3.

## Commands

| Command | Runs | Needs |
| --- | --- | --- |
| `make test` | unit + API tests (`pytest`, integration tests excluded by default) | nothing |
| `make test-integration` | `pytest -m integration`: real Postgres + pgvector and Neo4j via testcontainers | Docker; stop the compose stack first (`make down`) |
| `make typecheck` | `mypy` over packages, tests and migrations (one command, config in `pyproject.toml`) | nothing |
| `make lint` | ruff check + ruff format --check | nothing |
| `make pre-commit` | every pre-commit hook on every file | nothing |
| `make check` | `pre-commit` + `test` (what CI's first job runs) | nothing |
| `make smoke` | `scripts/smoke_test.sh` against a running stack: `/health`, `/ready`, API key | `make build && make up PROFILES=apps` |

Coverage: `uv run pytest --cov` measures the `iqs_*` packages with branch coverage.

## Levels

| Level | Tooling | Where | Runs |
| --- | --- | --- | --- |
| Unit (domain logic, contracts) | pytest, hypothesis | `packages/<pkg>/tests/` | every push |
| Config / data validation | pytest | `tests/test_*.py` | every push |
| API | FastAPI `TestClient` | `packages/<svc>/tests/test_api.py`, `packages/common/tests/test_api.py` | every push |
| Integration | testcontainers (images read from `deploy/compose/docker-compose.yml`) | `tests/integration/` | every push (`ci.yml`, job `integration`) |
| Compose smoke | `make build`, `make up PROFILES=apps`, `scripts/smoke_test.sh` | `.github/workflows/docker.yml` | push / PR to `main`, manual |
| Agent behaviour (recorded LLM responses) | — | — | planned (step 3, `eval.yml`) |
| ML evaluation (full) | `scripts/evaluate_*` + MLflow | — | planned (manual, before releases) |
| End-to-end | compose + `tests/e2e/` | — | planned (step 7) |

## What exists today

Measured on 2026-11-17: **121 unit/API tests** and **15 integration tests**, all passing;
branch coverage of `iqs_*` from unit/API tests alone: **99 %** (437 statements, 2 missed).

| File | Tests | Covers |
| --- | --- | --- |
| `packages/common/tests/test_api.py` | 23 | `/health`, `/ready` (ok, failure, timeout, no detail leak), API key (missing, wrong, rotation, fail closed), `/v1` routers, request IDs, CORS, OpenAPI, shutdown hooks, API settings |
| `packages/common/tests/test_taxonomy.py` | 26 | taxonomy model and loader validation |
| `packages/common/tests/test_events.py` | 18 | event envelopes, payload rules, `parse_event` |
| `packages/common/tests/test_ids.py` | 10 | UUIDv7 layout/order, deterministic IDs |
| `packages/common/tests/test_settings.py` | 10 | defaults, env and `.env` precedence, DB URL |
| `packages/common/tests/test_log.py` | 7 | JSON/console output, redaction, context |
| `packages/common/tests/test_outbox.py` | 7 | `platform.events` model, CHECKs, index, DDL, `from_event` round trip |
| `packages/common/tests/test_db.py` | 4 | engine, sessions, naming convention |
| `packages/{vision,diagnosis}/tests/test_api.py` | 4 + 4 | app factories: health, readiness checks per service, capabilities + key, env |
| `tests/test_taxonomy_config.py` | 8 | the real `configs/taxonomy/causes.yaml` (v0.1.0, 10 causes) |
| `tests/integration/test_migrations.py` | 11 | upgrade/downgrade, `alembic check` drift (and a probe that it fails on drift), objects outside our schemas ignored, outbox defaults/constraints/duplicates, pgvector round trip |
| `tests/integration/test_neo4j_smoke.py` | 1 | pinned Neo4j image answers Cypher |
| `tests/integration/test_service_ready.py` | 3 | both services `/ready` against real databases; wrong Neo4j password → 503 |

## Rules

- **Isolation.** Every package's `conftest.py` removes `IQS_*` variables, runs each test in an
  empty temporary directory (so a real `.env` is never read) and restores logging. Integration
  tests point Alembic at the test container through environment variables only.
- **No network in unit tests.** Readiness failures are tested against `127.0.0.1:1` (refused
  immediately) or fake checks.
- **Integration tests are marked** `@pytest.mark.integration` (module-level `pytestmark`) and
  excluded from plain `pytest` via `addopts`.
- **Definition of done per endpoint** (blueprint §7.2): happy path, validation error, auth error;
  an integration test wherever a database is involved.
- **Numbers in docs come from runs.** The counts above are from `pytest` output on the date given;
  update them when they change.
