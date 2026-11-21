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
