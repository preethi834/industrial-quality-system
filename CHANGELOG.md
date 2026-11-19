# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow the roadmap tags (`v0.1-step0`, `v0.1-step1`, ...).

## [Unreleased] — step 1 (P2 Layer 1: defect detector + demo, in progress)

### Added
- **1a** Blueprint Revision 2; GitHub Actions pinned by SHA on Node.js 24 (checkout v7.0.1,
  setup-uv v10.3.3); approval-gated asset allowlist (`configs/assets.yaml`, fetch/import
  scripts with `--approve` and SHA-256 checks); `docs/APPROVALS.md`; compose profile `vision`.
- **1b** MVTec AD adapter (never downloads), split manifests with per-image SHA-256 and
  leakage checks for all 15 categories, preprocessing, synthetic fixtures.
- **1c** CPU PatchCore (anomalib components, preallocated bank, faster greedy k-center),
  metrics, MLflow tracking, train / evaluate / report scripts, training queue.
- **1d** `POST /v1/inspections`, `GET /v1/models`, heatmaps, model registry, VLM interface
  (disabled: field absent), latency benchmark script.
- **1e** Supervised ResNet-18 baseline on a stratified re-split of the test set.
- **1f** Dashboard (home + defect inspection page), dashboard image, extended smoke test.
- **1g** EfficientAD Colab notebook, run bundles with verified import and CPU re-evaluation.

### Fixed
- Smoke test no longer echoes the API key.

## [v0.1-step0] — step 0 (foundation)

### Added
- **0a** uv workspace with six packages (`iqs_common`, `iqs_vision`, `iqs_diagnosis`, `iqs_mcp`,
  `iqs_sim`, `iqs_dashboard`), CPU-only torch index, Linux x86_64 lockfile, repo skeleton.
- **0b** `iqs_common`: typed settings (`IQS_*`, `.env`), structlog JSON logging with secret
  redaction, SQLAlchemy plumbing, UUIDv7 / deterministic IDs, taxonomy model, event envelopes.
- **0c** Cause taxonomy v0.1.0 (10 causes), config test, AI4I labelling guide.
- **0d** Docker Compose with PostgreSQL 16 + pgvector 0.8.7 and Neo4j 5.26 (pinned digests,
  localhost-only ports, single source for credentials); Alembic baseline (`vector`, four schemas,
  `platform.events` outbox); `OutboxEvent` model; Makefile; integration tests with testcontainers.
- **0e** `iqs_common.api` (extra `iqs-common[api]`): `/health`, `/ready`, API-key protected
  `/v1`, `/v1/capabilities`, request IDs, CORS. Vision (:8001) and diagnosis (:8002) app
  factories and Docker images (non-root, healthcheck); compose profile `apps`.
- **0f** GitHub Actions `ci.yml` (pre-commit hooks, unit/API tests with coverage, integration
  tests) and `docker.yml` (image build + compose smoke test); local pre-commit hooks incl.
  detect-secrets; `make check`, `make smoke`.
- **0g** Docs: ARCHITECTURE, IMPLEMENTATION_PLAN, TECHNICAL_DECISIONS (ADR-001–003), EXPERIMENTS,
  TESTING, KNOWN_ISSUES, this changelog.

### Changed
- Postgres is published on host port 5433 (5432 is taken by a Windows PostgreSQL).
- One `mypy` command checks packages, tests and migrations.
