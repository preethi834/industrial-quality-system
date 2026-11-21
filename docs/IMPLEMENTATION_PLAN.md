# Implementation plan

Persistent checklist and session hand-off (blueprint §7.6). A new session starts here, then runs
`make setup && make check`.

Status labels (blueprint §7.1): **planned** · **in progress** · **working (tested)** · **known issue**.

## Roadmap

| Step | Scope (blueprint §5) | Status |
| --- | --- | --- |
| 0 | Foundation: repo, settings, logging, databases, service skeletons, CI, docs | **working (tested) locally**; known issue: CI not yet run on GitHub |
| 1 | P2 L1: defect detector (PatchCore / EfficientAD + supervised baseline) + demo | **working (tested) locally**: 15/15 categories, both protocols; EfficientAD deferred (KNOWN_ISSUES #15); not pushed, no tag yet |
| 2 | P2 L2–3: MVTec AD 2 / VisA + conformal threshold | planned |
| 3 | P1 L1–2: diagnosis agent (hybrid RAG + KG, MCP, 8D) + evaluation | planned |
| 4 | P1 L3: conformal cause sets | planned |
| 5 | Factory simulator + scientist agent | planned |
| 6 | Cost-aware accept / reject / review | planned |
| 7 | Integrated quality loop (vision event → diagnosis case) | planned |
| 8 | Advanced features (8a–8g), one at a time | planned |
| 9 | Public proof of work (benchmark release, report, video) | planned |

## Step 0 — Foundation

| Sub-step | Content | Commit | Status |
| --- | --- | --- | --- |
| 0a | uv workspace (six packages), CPU-only torch index, Linux x86_64 lock, repo skeleton | `b6c666f` | working (tested) |
| 0b | `iqs_common`: settings, logging, db, ids, taxonomy model, events | `51c0ea0` | working (tested) |
| 0c | Cause taxonomy v0.1.0 (10 causes), config test, AI4I labelling guide | `516a244` | working (tested) |
| 0d | Compose (Postgres 16 + pgvector, Neo4j 5.26), Alembic baseline, outbox model | `98baecc` | working (tested) |
| 0e | Vision + diagnosis skeletons: `/health`, `/ready`, `/v1/capabilities`, API key, Docker images | `18c3700` | working (tested) |
| 0f | CI workflows, local pre-commit hooks, single mypy command | `3583846` | working locally; known issue: not run on GitHub |
| 0g | Docs: architecture, plan, decisions, experiments, testing, known issues, changelog | this commit | working |

Blueprint "done when" for step 0:

- [x] `make setup && make test` works; `make up` starts the stack on a clean checkout (+ `.env`)
- [x] both services answer `/health` (and `/ready`) in Docker — verified 2026-11-17
- [x] docs exist with real content
- [ ] green CI on `main` — needs a push (KNOWN_ISSUES #1, #2)

## Step 1 — Project 2 Layer 1

| Sub-step | Content | Commit | Status |
| --- | --- | --- | --- |
| 1a | Blueprint Rev 2, actions on Node 24 (SHA-pinned), asset allowlist + approval scripts, vision profile, KNOWN_ISSUES renumbered | `76debd5`, `a6d3876` | working (tested) |
| 1b | MVTec adapter, manifests (15 categories), preprocessing, synthetic fixtures | `ad7bb04` | working (tested) |
| 1c | CPU PatchCore, metrics, MLflow, train/evaluate/report scripts, queue | `903d949` | working (tested): 30 models trained, reload check exact |
| 1d | Inspection API, heatmaps, registry, VLM interface, latency benchmark | `3ed8bfe`, `2ddd22e` | working (tested); latency p50 529 ms / p95 560 ms (EXP-1-03) |
| 1e | Supervised ResNet-18 baseline | `32452e3` | working (tested): run on all 15 (EXP-1-02) |
| 1f | Dashboard + image + smoke test | `b5244dc`, `a5aca26` | working: Docker smoke + dashboard client with the 15 deploy models (images rebuilt 2026-11-21, smoke passed again) |
| 1g | EfficientAD notebook, run bundles, import | `34b7d3e` | code working (tested); not run, deferred (KNOWN_ISSUES #15) |
| 1h | Docs, results, tag `v0.1-step1` | `e2a181f`, this commit | docs done; tag waits for the owner |

Step 1 gates (blueprint Rev 2 §5):

- [x] (1) AUROC for all 15 categories, official and deploy (EXP-1-01)
- [x] (2) same-seed rerun within ±0.005 — toothbrush Δ = 0.0000
- [x] (3) deviations vs anomalib — 14/15 within ±0.02; grid +0.028 investigated (#11), pill +0.020 noted (#14)
- [x] (4) API 200 / 415 / 401 — tests + Docker smoke
- [x] (5) dashboard against Docker with the **deploy** models — smoke test + dashboard client in the container (screw defect → anomalous); images rebuilt 2026-11-21 and re-checked (#16 resolved)
- [x] (6) EfficientAD deferred in KNOWN_ISSUES #15 (confirmed by the owner 2026-11-21)
- [ ] universal gate: CI green on `main`, tag `v0.1-step1` — after the owner's push

## Hand-off notes

### 2026-11-21 — step 1 done locally (1h)

- **Done:** queue 28/28 rc = 0 (2026-11-20 18:38 → 2026-11-21 01:05). Supervised baseline (15
  categories, 22 min), same-seed rerun (toothbrush, Δ = 0), reload check (30 models, exact),
  reports for both protocols, Docker demo with the deploy models, latency benchmark. `make check`:
  199 tests, ruff, mypy, hooks pass. Results: EXPERIMENTS EXP-1-01…03.
- **Owner decisions 2026-11-21:** EfficientAD deferral confirmed (#15); images rebuilt with network
  access and smoke-tested (#16); branch `step-1` pushed for a PR; tag `v0.1-step1` only after CI is
  green on `main`.
- **Next:** step 2 with the decisions of 2026-11-20 (A1 VisA cashew + pipe_fryum, archive
  1,929,840,640 bytes; A2 MVTec AD 2 vial + fruit_jelly, archives in Windows Downloads, not yet
  imported; A3 `scipy` direct dependency; A4 skipped; D1 α = 0.02 only where n ≥ 49, else 0.05,
  toothbrush 0.10; D2 δ = 0.1; D3 deploy-only for new datasets; D4 cap fit at 400). Read #17
  (non-exchangeable train/test goods) before the conformal work.

### 2026-11-19 — step 1 stopped for the day (13:31)

- **Done:** 1a–1g code committed locally (not pushed), 199 unit/API tests pass, hooks + mypy
  clean. MVTec AD imported (SHA-256 verified) into `data/raw/mvtec_ad`; archive copy kept in
  `data/raw/archives/` (5.3 GB, can be deleted). Weights fetched and verified. Official PatchCore
  models: **bottle, grid** (`models/patchcore/<cat>/official/`, MLflow `mlruns/mlflow.db`).
- **Stopped:** screw official was 7 of ~38 min in; no files written; MLflow run marked KILLED.
- **Restart the queue** (compose stack stopped first; completed jobs are skipped):

  ```bash
  make down
  Q="official:screw deploy:bottle deploy:grid deploy:screw"
  for c in toothbrush metal_nut transistor zipper capsule cable tile wood leather pill carpet hazelnut; do Q="$Q official:$c"; done
  for c in toothbrush metal_nut transistor zipper capsule cable tile wood leather pill carpet hazelnut; do Q="$Q deploy:$c"; done
  nohup scripts/run_patchcore_queue.sh "$Q" > reports/logs/queue.log 2>&1 &
  tail -f reports/logs/queue.log
  ```

  Then (after the queue): `uv run python scripts/train_supervised.py --categories <all 15>`;
  same-seed rerun `uv run python scripts/train_detector.py --categories toothbrush --protocol official --models-dir models/patchcore_repro`;
  reload check `uv run python scripts/evaluate_detector.py --protocol official --categories <done>`;
  `uv run python scripts/report_patchcore.py --protocol official` (and `--protocol deploy`);
  Docker demo `make up PROFILES=vision && SMOKE_PORTS=8001 make smoke`;
  latency `uv run python scripts/benchmark_inspections.py --category bottle --requests 100`.
- **Estimate:** ~8–10 h for the remaining 28 queue jobs with nothing else on the CPU.



### 2026-11-17 — end of step 0 (0d–0g)

- **Done:** 0d–0g committed locally on branch `step-0` (not pushed). 121 unit/API tests and 15
  integration tests pass; one `mypy` run over 38 files is clean; all 13 pre-commit hooks pass;
  compose `apps` profile passes `scripts/smoke_test.sh`.
- **Broken / open:** CI never ran on GitHub; actions not SHA-pinned; one unexplained integration
  failure (1 of 6 runs). See `KNOWN_ISSUES.md`.
- **Local state:** `.env` exists (git-ignored, random passwords and API key); Docker volumes
  `iqs_pg_data`, `iqs_neo4j_data`, `iqs_neo4j_logs` hold migration `0001`; images
  `iqs-vision:dev`, `iqs-diagnosis:dev` are built; stack is stopped.
- **Next:** push `step-0`, get CI green, merge to `main`, tag `v0.1-step0`; then step 1.
