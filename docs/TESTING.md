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
| Agent behaviour (deterministic, no LLM yet) | pytest | `packages/diagnosis/tests/test_eval_harness.py` (10 reference cases through the full agent) | every push |
| Agent behaviour (recorded LLM responses) | — | — | planned (step 3e, `eval.yml`) |
| ML evaluation (full) | `scripts/evaluate_*` + MLflow | — | planned (manual, before releases) |
| End-to-end | compose + `tests/e2e/` | — | planned (step 7) |

## What exists today

Step 6 (2026-11-22, branch `step-6`): **58 new tests** pass (`test_decision.py` 26,
`test_decision_evaluate.py` 24, `test_api_costs.py` 7, dashboard 1), plus the existing vision
API and dashboard tests they touch (`test_api.py`, `test_api_calibrations.py`,
`tests/test_dashboard_inspection.py`). Only changed-file tests were run (`nice -n 19`,
another worktree was running heavy jobs); the full `make check` is still to run before the PR.

Latest full run (2026-11-21, merge of steps 3–5 into `main`): **421 unit/API tests pass, 2 skipped**
in `make check`, and **20 integration tests pass** in `make test-integration` (Docker, `nice -n 19`).

End of step 1 (2026-11-21): **199 unit/API tests** pass in `make check` (ruff,
mypy and hooks clean); the 15 integration tests are deselected there and were last run in step 0.
Real-data checks at the end of step 1: same-seed rerun (toothbrush, Δ = 0.0000, tolerance
±0.005) and reload check (30 models, `max_abs_diff = 0.0`), see EXPERIMENTS EXP-1-01.

Step 0 baseline, measured on 2026-11-17: **121 unit/API tests** and **15 integration tests**, all passing;
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

### Diagnosis and simulator (steps 3–5)

| Where | Covers |
| --- | --- |
| `packages/simulator/tests` | determinism (same scenario ⇒ identical SHA-256 of every file), fault isolation (each fault vs its fault-free twin changes exactly its declared columns, at its target, after its onset, and raises the exposed reject rate), schemas, no ground-truth tokens in `observed/` or in cases, DuckDB allow-list cannot read `ground_truth/`, case generator works with `ground_truth/` deleted, CLI, v2 families |
| `packages/diagnosis/tests` (symptoms, retrieval, scoring) | vocabulary rules, reference-scenario signatures (regression check, not accuracy), hybrid retrieval, vote scoring, novelty threshold |
| `packages/diagnosis/tests` (agent, API, 8D) | agent graph incl. conformal and scientist nodes, MCP ticket round trip, case API, 8D completeness, AST leak guard (only `evaluation` may read ground truth) |
| `test_conformal.py`, `test_calibration_run.py`, `test_conformal_eval.py` | index rule, 10,000-trial coverage band + KS test, (α, δ) bound, ties, hand-computed LAC/APS sets, hash split and disjointness, committed run consistent and leak-free |
| `test_scientist_*.py` | templates on synthetic data, synthetic null tests (FP ≤ α + 3 SE), BH runner, top-3 vs all-causes mode and persistence rows, evaluation arithmetic |
| `test_nhtsa_labelling.py` | flat-file reader (short and malformed rows), seeded stratified sample, sheet never overwritten, label rules, Excel `;`/cp1252 files |
| `tests/integration/test_diagnosis_persistence.py` | migrations 0002–0004 tables, SQL ticket store, case survives a restart with the Postgres checkpointer, calibration-run rows and cause sets, `hypothesis_tests` rows replaced per case |

The 1,000-run null check and the evaluations (`scripts/check_scientist_null.py`,
`evaluate_diagnosis.py`, `evaluate_conformal.py`, `evaluate_scientist.py`) are manual runs,
not part of `make test`; their results are in EXPERIMENTS EXP-3-02, EXP-4-01, EXP-5-01.

### Cost-aware decisions (step 6)

| Where | Covers |
| --- | --- |
| `packages/vision/tests/test_decision.py` | cost config (every value labelled an assumption, ranges, sha256), hand-computed cut-offs and expected costs, actions at each cut-off, review dropped when it never wins, guaranteed policy cases; property tests: raising C_review never raises the review share, raising C_escape never raises the accept share, raising C_false_reject never raises the reject share, free perfect review takes every 0 < p < 1, guaranteed never flags score ≤ q̂ and never accepts above it, cut-off rule = brute-force argmin; Platt / isotonic monotone in score, prior shift exact on known densities, Platt recovers a logistic model and stays finite on separated data |
| `packages/vision/tests/test_decision_evaluate.py` | gate 2 on every committed manifest (sup_train ∪ sup_val ∩ (calib ∪ sup_test) = ∅ by path and sha256), leak by content refused, hand-computed cost per 10,000 parts, test-set defect share has no effect, guaranteed flags exactly the fixed-q̂ set, `scripts/evaluate_costs.py` on a synthetic category (hash in report, ≥ 3 cost ratios, SVG), stale manifest refused |
| `packages/vision/tests/test_api_costs.py` | `cost_decision` present with hash when cost config + matching `decision.json` + conformal threshold exist; never `accept` above the threshold; absent (and capability off) for each missing artefact, a foreign run id or a missing cost file |
| `tests/test_dashboard_inspection.py::test_cost_decision_view` | dashboard decision headline, cost table and hash caption |

`scripts/evaluate_costs.py` itself is a manual run (EXPERIMENTS EXP-6-01).

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
