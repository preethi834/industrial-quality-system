# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow the roadmap tags (`v0.1-step0`, `v0.1-step1`, ...).

## [Unreleased] — step 2 (P2 Layers 2–3: hard benchmark + guarantee)

### Added
- **2f** VisA (cashew, pipe_fryum) and MVTec AD 2 (vial, fruit_jelly) adapters and manifests;
  split `shift` for MVTec AD 2 lighting conditions; archive SHA-256 recorded in
  `configs/assets.yaml` and `docs/APPROVALS.md`; `--dataset` for make_splits / train / evaluate;
  4 deploy models trained (EXP-2-01).
- **2a** `iqs_vision.calibration`: split-conformal threshold with the n ≥ ⌈1/α⌉ − 1 refusal,
  (α, δ) threshold from the Beta coverage law, Clopper–Pearson intervals, Beta-binomial
  prediction of the realised false-alarm rate; `scipy` as a direct dependency.
- **2b** `scripts/calibrate.py`: per-image `scores.json` + `calibration.json` per model, α by
  policy D1, leakage refusal, validation-only check for MVTec AD 2.
- **2c** Repeated-split validity check, `scripts/evaluate_vision_conformal.py`, reports
  `reports/conformal_<dataset>.{md,json}` and guarantee plots (EXP-2-02).
- **2d** `scripts/shift_experiment.py`: real MVTec AD 2 lighting conditions + six synthetic
  photometric shifts (EXP-2-03).
- **2e** `GET/POST /v1/calibrations`, conformal threshold and `threshold_note` in inspections,
  `conformal_threshold` capability, dashboard model-evaluation page.

### Changed
- Masks: any nonzero pixel is a defect (VisA region-id masks); MVTec AD results unchanged.

## [Unreleased] — steps 3–5 (diagnosis agent, conformal cause sets, simulator + scientist)

All diagnosis numbers are **simulator results, not real-world results** (KNOWN_ISSUES #18).

### Added
- **5a** Factory simulator `iqs_sim`: SimPy line L1 (4 stations, 5 machines), batches,
  shifts, sensors (temperature, ambient, spindle load, vibration, tool wear, commanded vs
  actual power); 10 injectable faults named as the taxonomy `simulator_fault`s; scenario
  YAMLs in `configs/simulator/scenarios/`; Parquet export with a separate `ground_truth/`;
  case generator; `scripts/simulate.py` (run / validate / family). ADR-008.
- **3a–3d** Diagnosis agent, minimum version (no LLM): simulator case corpus `dx_corpus_v1`
  and symptom vocabulary; hybrid retrieval (BM25 + feature vector), vote scoring and a
  novelty threshold; LangGraph agent with 8D draft, MCP ticket tools (`iqs_mcp`) and case
  API; migration 0002 (diagnosis cases, causes, 8D reports; QMS tickets); evaluation
  harness (test split: top-1 1.000, false novelty 0.061 on known causes). ADR-006.
- Harder corpus `dx_corpus_v2` (near-limit magnitudes, two-fault cases, larger dev split).
- **4a–4d** Split-conformal cause sets (LAC, APS, (α, δ) bound), frozen calibration runs
  (migration 0003), cause sets in agent, 8D and API (APS by default, LAC-empty flag);
  evaluation on v2 test (LAC 0.894, APS 0.989 coverage at α = 0.1). ADR-007.
- **5b** Scientist agent: one pre-registered SQL template per cause on the locked DuckDB
  store, BH per case, evidence figures; null check over 1,000 fault-free runs passed for
  every template; evaluation on `sci_eval_v1` (correct cause supported 0.957). Novel cases
  test all 10 causes (post-evaluation change, unmeasured); results stored in
  `diagnosis.hypothesis_tests` (migration 0004). ADR-009.
- **3f (preparation)** NHTSA recalls reader, labelling sheet and checker; NHTSA recalls and
  AI4I 2020 downloaded (approved, SHA-256 in `configs/assets.yaml`); 50-row NHTSA pilot
  labelled (AI-assisted, author-reviewed). Step 6 plan (`docs/notes/step-6.md`).

### Fixed
- The Postgres checkpointer raised `NotImplementedError` under the async agent; it now runs
  the sync saver's methods in a thread (`agent/checkpoint.py`).
- Integration tests: checkpoint tables are cleaned up between tests; the migrations test
  compares with the Alembic head instead of a fixed revision.
- `scripts/check_scientist_null.py`: a loop variable reused with another type (found by
  mypy now that `scripts/` is checked).

## [v0.1-step1] — step 1 (P2 Layer 1: defect detector + demo)

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
- **1h** Results: PatchCore on all 15 MVTec AD categories (official mean image / pixel AUROC
  0.984 / 0.982; deploy 0.981 / 0.982), supervised baseline (mean 0.849 vs PatchCore 0.977 on
  sup_test), same-seed rerun and reload checks, inspection latency p50 529 ms / p95 560 ms;
  reports `reports/patchcore_mvtec_ad_{official,deploy}.md`; grid / pill investigation (#11, #14).

### Fixed
- Smoke test no longer echoes the API key.
- `docker` workflow: free runner disk space (unused toolchains, `docker system prune`) before
  the build and prune the build cache after it; the PR run had failed with "no space left on device".

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
