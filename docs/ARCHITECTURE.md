# Architecture

Status: **steps 1–6** (defect detector + demo; conformal threshold; cost-aware decisions; diagnosis agent minimum version, conformal cause sets, simulator + scientist). This page describes what exists in the repository today and
marks everything else as *planned*. The full design is in `TECHNICAL_BLUEPRINT.md` §2; this
document is updated by each step as parts become real.

## System overview

One monorepo, two connected systems, one shared data layer:

- **Vision service** (Project 2, open-world defect discovery): detects defects, guarantees a
  maximum false-alarm rate, makes cost-aware accept / reject / review decisions, discovers new
  defect types. *Today: PatchCore inspection API (`POST /v1/inspections`, `GET /v1/models`) with
  heatmaps, a split-conformal threshold with a false-alarm bound per category (step 2) and
  accept / reject / review by expected cost in assumed euros that never flags a part at or below
  that threshold (step 6, `iqs_vision.decision`).*
- **Diagnosis service** (Project 1, active root-cause diagnosis agent): retrieves similar past
  cases, returns a conformal set of causes, asks the most informative question, tests hypotheses
  against process data, drafts the 8D report. *Today (simulator cases only, no LLM): symptom
  profiles, hybrid retrieval, novelty flag, conformal cause sets, scientist hypothesis tests,
  8D draft, case API; question selection and the knowledge graph are planned.*
- **Factory simulator** (SimPy): production line with injected, known root causes; ground truth
  for evaluation. *Today: line L1 with 10 faults, Parquet export with a separate ground truth (step 5a).*
- **MCP server**: mock QMS/MES exposed as MCP tools; the only way the agent changes "factory
  systems". *Today: QMS ticket tools (`create_ticket`, `update_ticket`).*
- **Dashboard** (Streamlit, :8501): talks only to the APIs, never to databases. *Today: home
  (status, capabilities) and the defect-inspection page.*

```text
                    ┌──────────────┐  REST + X-API-Key   ┌──────────────────┐
  engineer ───────► │  dashboard   │ ──────────────────► │ vision  :8001    │──┐
                    │  :8501       │ ──────────┐         └──────────────────┘  │ outbox event
                    └──────────────┘           │         ┌──────────────────┐  │ new_defect_cluster
                                               └───────► │ diagnosis :8002  │◄─┘ (planned, step 7)
                                                         └──┬────────┬──────┘
                                                  MCP (planned)│     │ Bolt
                                                  ┌────────────▼┐  ┌─▼─────────────┐
                                                  │ mcp_server  │  │ Neo4j 5.26    │
                                                  └─────────────┘  └───────────────┘
   both services ──SQL──► PostgreSQL 16 + pgvector (schemas: diagnosis, vision, qms, platform)
   diagnosis ──DuckDB (read-only)──► simulator Parquet (planned)
```

## Packages (uv workspace, Python 3.11)

| Package | Import | Role | State |
| --- | --- | --- | --- |
| `packages/common` | `iqs_common` | Contracts shared by every service: settings, logging, DB plumbing, IDs, taxonomy, events, outbox model, HTTP service plumbing (`[api]` extra) | working (tested) |
| `packages/vision` | `iqs_vision` | Vision service: `data/` (MVTec adapter, split manifests, preprocessing), `detectors/` (PatchCore, EfficientAD CPU inference, coreset), `supervised/` (ResNet-18 baseline), `evaluation/`, `explain/` (heatmaps, VLM interface), `inference/` (model registry), `bundles.py` (cloud run bundles), `api/` | working (tested) |
| `packages/diagnosis` | `iqs_diagnosis` | Diagnosis service: `symptoms/`, `retrieval/`, `scoring/`, `novelty/`, `conformal/`, `scientist/`, `agent/` (LangGraph + Postgres checkpointer), `reporting/` (8D), `persistence/`, `ingestion/` (NHTSA), `evaluation/` (the only module allowed to read ground truth), `api/` | working (tested) on simulator cases |
| `packages/mcp_server` | `iqs_mcp` | Mock QMS over MCP (ticket tools) | working (tested) |
| `packages/simulator` | `iqs_sim` | Factory simulator (SimPy), Parquet export, case generator | working (tested) |
| `packages/dashboard` | `iqs_dashboard` | Streamlit UI: vision client, home, defect inspection page | working (tested) |

The vision and agent stacks have heavy, conflicting dependency trees (PyTorch/Anomalib vs
LangGraph/LLM SDKs), so each service has its own Docker image built from the same `uv.lock`
with only its own package installed.

## `iqs_common` modules

| Module | What it provides |
| --- | --- |
| `settings` | `CommonSettings` from `IQS_*` environment variables (nested with `__`) and `.env`: environment, logging, database, Neo4j, API keys/CORS, taxonomy path |
| `log` | structlog JSON logs (console in dev), secret redaction, per-request context (`request_id`) |
| `db` | SQLAlchemy 2 engine/session factory, `Base` with a fixed naming convention, `check_database` probe |
| `ids` | UUIDv7 for records/events; deterministic UUIDv5 for simulator entities |
| `taxonomy` | Loader and validator for `configs/taxonomy/causes.yaml` (fixed cause label space) |
| `events` | Pydantic event envelopes (`new_defect_cluster`, `case_confirmed`) and `parse_event` |
| `outbox` | `OutboxEvent` → `platform.events` (transactional outbox, at-least-once delivery) |
| `api` | `create_service_app`: `/health`, `/ready`, API-key protected `/v1`, request IDs, CORS |

## HTTP conventions (both services)

| Route | Auth | Meaning |
| --- | --- | --- |
| `GET /health` | none | Liveness: the process answers. Never touches a dependency. |
| `GET /ready` | none | Readiness: every dependency check passes within `IQS_API__READY_TIMEOUT_S` (default 2 s). 200 `ready`, else 503 `not_ready` with per-check status (`ok`, `timeout`, `error: <ExceptionType>`; details only in logs). Vision checks Postgres; diagnosis checks Postgres and Neo4j. |
| `GET /v1/capabilities` | `X-API-Key` | Service name, version, `api_version`, and feature flags (all `false` at step 0). |
| `/v1/*` | `X-API-Key` | Every versioned route. Keys from `IQS_API__KEYS` (comma list, for rotation); constant-time comparison; no keys configured → every request gets 401 (fail closed). |
| `GET /docs`, `/openapi.json` | none | OpenAPI. |

Every response has an `X-Request-ID` header (taken from the request if present and ≤128 chars,
otherwise a new UUIDv7); the same ID is on every log line written for that request. CORS allows
only the dashboard origin (`IQS_API__CORS_ORIGINS`, default `http://localhost:8501`).

Services are built by an app factory (`uvicorn --factory iqs_<svc>.api.main:create_app`), so
importing a module never reads settings or opens connections, and tests construct apps with
explicit settings.

Inside each service, code will follow four layers (blueprint §2.3): API (routers, no business
logic) → application (use cases, transactions) → domain (pure Python, no I/O) → infrastructure
adapters behind interfaces.

## Data layer

| Store | Purpose | State |
| --- | --- | --- |
| PostgreSQL 16 + pgvector 0.8.7 | System of record and vector store. Schemas `diagnosis`, `vision`, `qms`, `platform`. Migrations: Alembic in `infrastructure/database/` | `0001` baseline (`vector`, schemas, `platform.events` outbox); `0002` diagnosis cases / causes / 8D reports and QMS tickets; `0003` calibration runs; `0004` hypothesis tests |
| Neo4j 5.26 community | Knowledge graph (parts, failure modes, causes, FMEA entries, cases) | in compose; no graph schema yet (step 3) |
| DuckDB over Parquet | Simulator `observed/` tables, read-only for the agent (allow-list, external access off, configuration locked; ADR-008) | working (steps 3–5) |
| MLflow 3.18 (SQLite store `mlruns/mlflow.db`, artifacts `mlruns/artifacts`) | Experiment tracking for scripts (never the API); `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` | working (step 1) |

Cross-service events use the **transactional outbox**: a producer writes its domain rows and one
`platform.events` row in the same transaction; a worker delivers pending rows at least once, and
consumers are idempotent on `event_id` (also the row's primary key).

## Vision pipeline (step 1)

```text
MVTec AD archive ──download_dataset.py (approve, SHA-256, safe extract)──► data/raw/mvtec_ad
      │ make_splits.py (seed 42)
      ▼
eval_sets/vision/mvtec_ad/<cat>.json   fit │ calib │ test (+ sup_train/val/test re-split)
      │ train_detector.py --protocol official|deploy        (one category per process)
      ▼
models/patchcore/<cat>/<protocol>/{model.pt, metrics.json}  ──► MLflow (sqlite, mlruns/)
      │ ModelRegistry (LRU 2, weights via configs/assets.yaml, read-only mount)
      ▼
POST /v1/inspections ─► score, provisional threshold (max calib score), decision, heatmap
      ▲
dashboard 2_Defect_Inspection (upload → result)

finetune/efficientad (Colab GPU) ─► run bundle ─► import_run_bundle.py (verify, CPU re-eval)
```

Weights and datasets are never downloaded at runtime: everything external is in
`configs/assets.yaml` and fetched only by the approval-gated scripts (ADR-004, ADR-005).

## Diagnosis pipeline (steps 3–5)

```text
simulate.py ─► <run>/observed/ (Parquet) │ <run>/ground_truth/ (evaluation only)
                    │ DuckDB allow-list (read-only, locked)
                    ▼
case (alarm) ─► symptom profile ─► hybrid retrieval (BM25 + feature vector) ─► votes ─► p(cause)
                                         │ novelty flag                         │
                                         ▼                                      ▼
                               conformal set (APS, LAC-empty flag)   scientist: SQL template per cause,
                                         │                            BH per case (top 3, or all 10 if novel)
                                         ▼                                      │
                                 8D draft (D0/D2/D4) ◄──────────────────────────┘
                                         │ MCP create_ticket / update_ticket
                                         ▼
                 Postgres: diagnosis.cases, case_causes, reports_8d, calibration_runs,
                           hypothesis_tests; qms.tickets; LangGraph checkpoints
```

Run directory layout and the simulator signal graph: `docs/notes/step-5a.md` §1–2.

## Deployment (local)

`deploy/compose/docker-compose.yml`, always run through the Makefile from the repo root so the
root `.env` is the single source of configuration.

| Profile | Services | Host ports (127.0.0.1 only) |
| --- | --- | --- |
| `core` | postgres, neo4j | 5433 (Postgres), 7474 / 7687 (Neo4j) |
| `apps` | core + vision, diagnosis, dashboard | + 8001 (vision), 8002 (diagnosis), 8501 (dashboard) |
| `vision` | postgres + vision + dashboard (no Neo4j; ADR-004) | 5433, 8001, 8501 |

Images: `deploy/docker/{vision,diagnosis,dashboard}.Dockerfile`, multi-stage, `uv sync --frozen --no-dev
--package iqs-<svc>`, non-root user, Docker `HEALTHCHECK` on `/health`. Credentials reach
containers only through variables listed under `environment` in the compose file.

## Decisions

See `docs/TECHNICAL_DECISIONS.md` (ADR-001 onwards).
