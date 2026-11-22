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

---

## ADR-004 — Step 1 groundwork: blueprint Rev 2, asset gate, action pins, vision profile (step 1a)

**Status:** accepted · 2026-11-19

**Context.** The blueprint was replaced by Revision 2. Its top note says repository ADRs win where
they differ (one uv workspace stays; ADR-001/002). Rev 2 §3.9 adds enforced approval gates for
downloads; GitHub warned that `actions/checkout@v4` and `astral-sh/setup-uv@v6` run on the
deprecated Node.js 20; step 1 adds a dashboard and heavier vision inference, which makes the
"both APIs or nothing" `apps` profile wasteful on a 10 GB WSL VM (KNOWN_ISSUES #8).

**Decisions.**

1. **Asset allowlist** `configs/assets.yaml` (source, licence, size, SHA-256, pinned Hugging Face
   commit, target path, approval date) + `iqs_common.assets`. `scripts/fetch_models.py` and
   `scripts/download_dataset.py` refuse to run without `--approve` and verify the checksum before
   a file is used. Services and training code only resolve local paths and raise
   `AssetMissingError` (with the fetch command) — they never download. MVTec needs a web form, so
   its archive is downloaded by hand and imported (copy → verify → safe extract with the tar
   `data` filter). Approvals are logged in `docs/APPROVALS.md`.
2. **Backbone weights loaded from local safetensors** (`timm.create_model(pretrained=False)` +
   `load_state_dict`), so neither timm nor anomalib can reach the Hugging Face hub at runtime.
   PatchCore uses `wide_resnet50_2.tv_in1k` (torchvision V1, as in the PatchCore paper), not
   anomalib's default `racm_in1k`.
3. **GitHub Actions pinned by commit SHA**, on Node.js 24: `actions/checkout` v7.0.1
   (`3d3c42e5…`), `astral-sh/setup-uv` v10.3.3 (`68559e99…`). SHAs verified with `git ls-remote`
   against the release tags; the inputs we use (`version`, `enable-cache`,
   `cache-dependency-glob`) exist in the pinned `action.yml`. Breaking changes between the old
   and new majors affect only `pull_request_target` / `workflow_run` and `prune-cache`, which we
   do not use. Workflows are checked locally with actionlint 1.7.12.
4. **Compose profile `vision`** = Postgres + vision API + dashboard (no Neo4j, no diagnosis).
   `core` and `apps` keep their meaning; the dashboard is also in `apps`. Supersedes the "core
   and apps only" part of ADR-002.

**Consequences.** Nothing large enters the machine without an approval record. A missing weight
file fails fast with the exact fetch command. Updating an action means a new SHA lookup.

---

## ADR-005 — Step 1 detector design: splits, PatchCore on CPU, thresholds, serving (steps 1b–1g)

**Status:** accepted · 2026-11-19

**Context.** Step 1 needs a real detector on MVTec AD, compared with published numbers, served
through the API and dashboard, on a 10 GB WSL VM without a GPU. Rev 2 §2.8 requires that
calibration images are never seen by the detector; Rev 2 §3.9 forbids implicit downloads.

**Decisions.**

1. **Own MVTec adapter, no anomalib datamodule** for local work: anomalib's `MVTecAD.prepare_data`
   downloads the archive when the folder is missing. The adapter raises `DatasetNotFoundError`.
2. **One split manifest per category** (`eval_sets/vision/mvtec_ad/*.json`, SHA-256 per image,
   seed 42): `fit` / `calib` (20 % of train/good, good only) / `test` (official), plus a
   stratified `sup_train/sup_val/sup_test` (50/20/30) re-split of the test set for the
   supervised baseline. Leakage is checked by path **and content hash** on load. Generated
   manifests are excluded from detect-secrets (they hold only checksums).
3. **Two training protocols.** `official` fits on fit + calib (= the full MVTec train set) and is
   the one compared with published numbers. `deploy` fits on `fit` only and is the one served;
   its provisional threshold is the largest calibration score, labelled in every response as
   *not* a conformal guarantee (that is step 2).
4. **PatchCore = anomalib 2.7 torch components + own fit loop**: preallocated memory bank
   (anomalib concatenates a list, which doubles peak RAM), and greedy k-center with a
   squared-distance update (one mat-vec per step instead of an N×d temporary): same selections
   as anomalib's `KCenterGreedy` under the same seeds (tested), ~2.8× faster per iteration
   (104 → 37 ms at N = 214k, d = 303 on the Ryzen 7 5700U). bf16 distances were 5× faster
   but rejected: the expanded distance formula loses too much precision. Settings = anomalib
   defaults (WRN-50-2 layers 2+3, 256 px, no centre crop, 10 % coreset, k = 9).
5. **One category per process** (`scripts/run_patchcore_queue.sh`) so RAM returns to the OS
   between categories; training runs with the compose stack stopped.
6. **Serving**: `ModelRegistry` discovers `models/patchcore/<cat>/<protocol>/`, loads on demand
   (LRU of 2), weights resolved through the asset allowlist; the vision container mounts
   `./models` read-only and gets 2 GB. Heatmaps use an absolute scale (0.5 = threshold) so a
   normal part stays cool. The VLM description is an interface with provider `none`: the field
   is absent, never faked.
7. **EfficientAD trains only in the cloud** (`finetune/efficientad`, same script + config) and
   returns as a **run bundle** (zip: `bundle.json` + `model.pt`, checksums, git commit,
   manifest fingerprint). The importer rejects tampering, extra members, a different manifest
   or an unknown commit, re-evaluates on CPU and requires |Δ image AUROC| ≤ 0.005.

**Consequences.** Published-number comparisons use `official` models; the API serves `deploy`
models, so the two tables differ slightly (20 % fewer fit images). Training all 15 categories
twice takes most of a day on this CPU; coreset selection dominates (quadratic in train size).

---

## ADR-006 — Diagnosis minimum version: simulator symptom profiles, retrieval votes, novelty threshold (step 3a–3b)

**Status:** accepted · 2026-11-20 (proposed in `docs/notes/step-3.md`; numbered and adopted at the merge into `main`, 2026-11-21)

**Context.** Blueprint §5 Step 3 asks for a working agent with retrieval, novelty and
top-3 causes, measured on a test split. The real-data path (NHTSA, AI4I, LLM) needs
downloads, hand labels and an LLM; the simulator (step 5a) already gives labelled cases
with exact ground truth. Simulator case text is templated and does not say where
rejects concentrate, so text alone cannot separate the 10 causes.

**Decisions.**

1. **Minimum version without an LLM** (approved 2026-11-20). Probabilities come only
   from retrieval votes (§2.3 rule). The LLM arrives in 3e via Ollama only; no paid API
   before step 8 (user rule).
2. **Symptom extraction may compute descriptive concentration and shift statistics**
   (approved, answers 5a open question 2): effect sizes only, no p-values or verdicts;
   statistical tests with BH correction stay in the scientist agent (5b).
3. **Symptom codes are observations in lower snake case**, separate from cause codes
   (answers 5a open question 1). Rules live in `configs/diagnosis/symptoms.yaml`.
4. **Features are machine-agnostic** (max/min over machines, batches, shifts); the
   location is evidence (`where`). A tool-wear case on M1 and on M3 then look alike.
5. **Leak guards.** Only `iqs_diagnosis.evaluation` may name the ground-truth directory
   or be imported for labels (AST test); profiles and labels are separate files joined
   by `case_id`; run directories are opaque hashes; the process store is the locked
   DuckDB allow-list from 5a.
6. **Splits = separate scenario families** with their own names and seeds (train =
   case library, dev = tuning, test = reported once in 3d). Two causes
   (`MACHINE.DRIVE_FAULT`, `PERSONNEL.PART_HANDLING_ERROR`) are left out of the
   library to measure novelty; chosen before any results.
7. **Generated data outside the repo** (`~/iqs-data/sim/<corpus>`, answers 5a open
   question 3); only the manifest (family hashes, file hashes, counts, timings) and the
   fitted ranker parameters are committed.
8. **Dense embeddings and reranker deferred to 3f** (approved): for numeric simulator
   profiles a standardised feature vector is the "dense" side of hybrid retrieval.

**Consequences.**
- Simulator results say nothing about real-world accuracy; they are reported separately
  from NHTSA results (§6.3, §6.4).
- The vocabulary thresholds were read off the same reference scenarios the signature
  test uses; the test is a regression check, not evidence of accuracy.
- Changing a feature or rule changes every profile; the corpus manifest records the
  vocabulary hash, so stale profiles are detectable.

---

## ADR-007 — Conformal cause sets: split by case hash, frozen runs, LAC + APS (step 4)

**Status:** accepted · 2026-11-20 (proposed in `docs/notes/step-4.md`; numbered and adopted at the merge into `main`, 2026-11-21)

**Context.** §2.8 requires disjoint fit and calibration data, frozen calibration runs, an
error instead of a silent threshold when n is too small, and honest reporting of
marginal coverage and of new problems. The step-3 scorer had its smoothing on the grid
edge and no errors on v1, so v2 (harder) was built first.

**Decisions.**
1. Own implementation (no MAPIE; user decision); cross-check against hand-computed sets.
2. Scorer refit and calibration on two hash-defined halves of v2 dev; library = v2 train.
3. Calibration runs are JSON artifacts with scorer parameters, case IDs and sorted scores;
   the service loads the scorer from the run so served scores match calibration; runs are
   mirrored into `diagnosis.calibration_runs` (one row per variant).
4. Four variants per run (LAC, APS; marginal and (α, δ = 0.1)); the service chooses one.
5. Non-randomised APS with a fixed tie order (cause code); LAC empty set = escalate.
6. Calibration only on known-cause cases (closed label space); new problems are a stress test.
7. One calibration run per data source (`source` field); real cases will get their own run.

**Consequences.** Coverage claims are simulator-only until gate 1; ties make coverage
conservative and the Beta law inapplicable; two-fault cases are under-covered by LAC.

---

## ADR-008 — Factory simulator: exact fault isolation, observed/ground-truth split (step 5a)

**Status:** accepted · 2026-11-19 (proposed in `docs/notes/step-5a.md`; numbered and adopted at the merge into `main`, 2026-11-21)

**Context.** Blueprint §2.2/§2.4/§5 Step 5: a SimPy line with injectable faults whose ground
truth is stored where the agent cannot read it; determinism per seed (identical Parquet
hashes); each fault must change only its intended signals. The taxonomy (v0.1.0) has 10 leaf
causes, each with a planned `simulator_fault` name. The scientist agent (5b) needs data in
which every cause leaves a distinguishable, *observable* trace.

**Decisions.**

1. **One fault type per taxonomy leaf, same names.** `FaultType` values equal the
   `simulator_fault` names; a test enforces the equality, and ground truth stores the cause
   code looked up from `causes.yaml` at export time (no second copy of the mapping).
2. **Exact isolation instead of statistical checks.** Named random streams with fixed draw
   counts, and timing/routing independent of faults. A fault run and its fault-free twin line
   up row by row; the test asserts the changed columns equal the fault's declared signal set.
3. **Line layout with one parallel station** (S2: M2A ‖ M2B), so machine-level causes
   (fixture) are distinguishable from line-level ones through `station_path`.
4. **Observable references for measurement faults**: periodic master-part gauge checks in
   `station_events`; batch certificates show nominal hardness while the true hardness is
   ground truth only.
5. **Separate directories, not separate columns.** `observed/` and `ground_truth/` are
   siblings; schemas of `observed/` are fixed; no scenario name, seed or description in
   observed files. The agent's DuckDB connection should use
   `allowed_directories=[<run>/observed/]`, `enable_external_access=false`,
   `lock_configuration=true` (set in that order; verified by a test).
6. **Case generator reads only `observed/`** and reports what monitoring would see
   (alarm time, reject rates, reason mix), not where rejects concentrate.
7. **Extra observable table `machines`** (master data: nominal setpoints, validated window,
   tool life, wear limit) and an observable `run.json` (spec limits, table hashes). The
   blueprint lists five tables; the agent needs the validated window and tolerances to judge
   "wrong parameters" and "out of spec".
8. **CLI logic in the package** (`iqs_sim.cli`, typed and tested); `scripts/simulate.py`
   is a thin wrapper (scripts/ is outside the mypy file list).

**Consequences.**
- Changing a random-draw order or count changes every downstream number for that seed;
  reference results must be regenerated (the determinism test catches accidental changes
  only within one run, not across commits — `docs/notes/step-5a.md`, open question 4).
- The quality model is a modelling choice; it is documented in this note and in
  `sensors.py` and must be reported as such.
- Evaluation families: `simulate.py family` writes a manifest with `family_sha256`, which
  the 5b gate needs to commit before the first evaluation run.

---

## ADR-009 — Scientist agent: one pre-registered template per cause, BH per case, evidence only (step 5b)

**Status:** accepted · 2026-11-20 (proposed as ADR-008 in `docs/notes/step-5b.md`; numbered and adopted at the merge into `main`, 2026-11-21)

**Context.** Blueprint §2.2 (`scientist`), §2.5 P1-D and §5 Step 5: test the agent's candidate
causes against the simulator's process data with valid error control, without voiding the
step-4 calibration (ADR-007). Store: the locked DuckDB allow-list over `observed/` (ADR-008).

**Decisions.**

1. One template per taxonomy cause, fixed SQL on the locked DuckDB store, pre-registered
   one-sided test; Bonferroni over targets inside a template; BH (q = 0.05) over the
   hypotheses of one case.
2. Verdicts supported / rejected / inconclusive; "rejected" means no evidence, not absence.
3. The scientist does not change probabilities or conformal sets (calibration stays valid);
   it adds evidence and a data-supported cause.
4. Evidence figures are plotly-compatible JSON; no plotting dependency in diagnosis.
5. Evidence is stored in `diagnosis.hypothesis_tests` (migration 0004) and summarised in
   the 8D report content.
6. Cases flagged novel test every cause and pick by smallest adjusted p (post-evaluation
   change; unmeasured until a new family is evaluated).

**Consequences.** "Supported" means "the data show this pattern", not "this is the cause":
faults have physical side effects, so wrong causes are supported in 14 % of tests on
`sci_eval_v1` (`docs/notes/step-5b.md`). The null check is vacuous for four templates
(KNOWN_ISSUES #23). Decision 6 is unmeasured until a new family is evaluated.

---

## ADR-010 — Step 2: new datasets and the conformal false-alarm threshold (steps 2a–2f)

**Status:** accepted · 2026-11-21 (numbered from 010: ADR-006…009 come from step-5b, PR #3)

**Context.** Step 2 replaces the provisional threshold (largest calibration score) with a
split-conformal threshold that has a stated false-alarm guarantee (blueprint Rev 2 §2.8), and
runs the pipeline on VisA and MVTec AD 2. Owner decisions: A1/A2 categories, A3 `scipy`, D1–D4,
N1 (2026-11-20/21).

**Decisions.**

1. **Adapters, not datamodules** (as ADR-005 §1): `data/visa.py` reads `split_csv/1cls.csv`
   (rejects unsafe paths and anomalies in train); `data/mvtec_ad2.py` takes the lighting
   condition from the file name (`regular`, `overexposed`, `underexposed`, `shift_1..4`). Masks:
   any nonzero pixel is a defect (VisA masks hold region ids, MVTec 0/255). Archives are verified
   against our SHA-256 in `configs/assets.yaml` (VisA also against anomalib's) and only the
   approved categories are extracted (`--only`).
2. **Manifests gain** split `shift` (MVTec AD 2 non-regular conditions, never used for fitting
   or calibration) and optional `source` / `condition` fields, omitted from the JSON when unset,
   so the 15 MVTec AD manifests stay byte-identical.
3. **Calibration set (N1).** MVTec AD 2: `validation/good` + 20 % of `train/good` held out
   (never fitted); the validation-only threshold is reported as a check. VisA and MVTec AD:
   20 % of train-good. New datasets use the deploy protocol only (D3).
4. **Threshold rule.** q̂ = k-th smallest calibration score with k = ⌈(n+1)(1−α)⌉ (float slack
   1e-9); n < ⌈1/α⌉ − 1 is refused (`InsufficientCalibrationError`, HTTP 422). A part is flagged
   iff score > q̂. α per category by D1 (0.02 if n ≥ 49, 0.05 if n ≥ 19, else 0.10). The (α, δ)
   threshold uses the Beta(k, n+1−k) coverage law (δ = 0.1) and is reported, not served: it is
   infeasible at α = 0.02 unless n ≥ 114.
5. **Scores, not images, are the calibration artefact.** `scripts/calibrate.py` writes
   `scores.json` (per-image scores of calib / test / shift) and `calibration.json` next to the
   model. Validity checks, the shift experiment and `POST /v1/calibrations` all recompute from
   `scores.json`; a calibration whose images overlap the fit set is refused by path and hash.
6. **Validity gate** = 100 resplits of the pooled unseen good images (calib ∪ good test),
   mean realised FA ≤ α + 3 SE, plus the fixed official split reported separately. Pooling makes
   the data exchangeable by construction, so the resplit check tests the code, not the data;
   the fixed split is the data check (KNOWN_ISSUES #17).
7. **API.** `POST /v1/calibrations` activates a new threshold **in memory only** (models are
   mounted read-only); the persistent calibration is `calibration.json`. Every inspection
   response carries `threshold_note`, which states α, n and that the guarantee is marginal and
   assumes exchangeability.
8. **Guarantee plot as hand-written SVG** (`reports/figures/conformal_<dataset>.svg`): matplotlib
   is not a direct dependency, and adding it would change `uv.lock` without approval.
9. **Script name `scripts/evaluate_vision_conformal.py`** (blueprint §5 step 2 lists
   `evaluate_conformal.py`): that name is taken by the step-4 diagnosis cause-set evaluation
   (ADR-007), which reached `main` first.

**Consequences.** Thresholds are reproducible from committed manifests + `scores.json` without
re-running the detector. At n ≈ 50–100 and α = 0.02 the threshold is the largest or second
largest calibration score, so miss rates are high on hard categories (KNOWN_ISSUES #29).

## ADR-011 — Step 6: cost-aware decisions that keep the false-alarm guarantee (steps 6a–6c)

**Status:** accepted · 2026-11-22. Owner decisions 1–14 of 2026-11-22 (all defaults approved).

**Context.** Blueprint Rev 2 §2.5 P2-C: choose accept / reject / review by expected cost in
euros, without breaking step 2's "at most α of good parts flagged". `docs/notes/step-6.md`
(2026-11-20) predates step 2 and is superseded where it differs (below).

**Decisions.**

1. **Served policy = guaranteed.** A part is accepted iff score ≤ q̂ (the active step-2
   threshold); above q̂ the cost rule chooses reject (p ≥ p_rr) or review. Since flagging
   depends only on score > q̂, P(good part flagged) ≤ α holds exactly as in step 2. The
   unconstrained three-way rule is evaluated and reported as having **no** guarantee, never served.
2. **Closed-form cut-offs.** All expected costs are linear in p, so accept iff p ≤ min(p_ra, p_ar),
   reject iff p ≥ max(p_rr, p_ar); review drops out automatically when p_ra ≥ p_rr (e.g. when a
   review costs more than a false reject). Ties go to accept at the lower cut and reject at the
   upper one (equal expected cost).
3. **Probabilities.** Platt scaling (soft targets, slope ≥ 0, own SciPy code) on
   sup_train ∪ sup_val only; isotonic is a reported comparison. The calib images are **not**
   used (the step-6 note proposed adding them; that would break gate 2). Prior shift to the
   assumed production defect rate by odds rescaling.
4. **Costs as a frozen YAML** (`configs/costs/example_costs.yaml`): every value carries a note
   that must start with "assumption"; reports and API responses cite the file's sha256. The
   `vision.cost_configs` table (blueprint §2.4) waits for step 7.
5. **No score export.** Step 2's `scores.json` already holds every calib / test score; the
   evaluation joins them to the manifest sub-splits by path and sha256 and refuses a manifest
   whose fingerprint differs. No inference in step 6.
6. **Evaluation.** Cost per 10,000 parts from class-conditional action rates on sup_test and the
   assumed π (the test-set defect share does not enter); equal weight per category; stratified
   bootstrap with calibrator and q̂ fixed. Categories need ≥ 4 good and ≥ 4 defective sup_test
   images (MVTec AD 2 has 1 good). The guaranteed policy's false-alarm rate is measured on all
   official good test images (flagging does not use p, so no leak). α stays step 2's; other α
   are a sensitivity row only.
7. **Serving.** `decision.json` (Platt parameters + model run id) next to each model, written by
   `scripts/evaluate_costs.py --write-decision`; ignored if the run id differs. The inspection
   response gains `cost_decision` only when a cost config, a matching `decision.json` and a
   conformal threshold all exist; capability `cost_decisions` follows the same rule.
8. **Plots as hand-written SVG**, as in ADR-010 §8 (no `uv.lock` change).

**Consequences.** The guarantee is untouched by any cost setting. The cost rule saves money mainly
where good parts often score above q̂ (it reviews instead of scrapping); elsewhere it can cost more
than plain rejection, because a prior-shifted p below p_rr sends real defects to review
(KNOWN_ISSUES #33). Euro results remain assumptions on benchmark images.
