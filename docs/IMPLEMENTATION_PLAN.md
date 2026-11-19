# Implementation plan

Persistent checklist and session hand-off (blueprint §7.6). A new session starts here, then runs
`make setup && make check`.

Status labels (blueprint §7.1): **planned** · **in progress** · **working (tested)** · **known issue**.

## Roadmap

| Step | Scope (blueprint §5) | Status |
| --- | --- | --- |
| 0 | Foundation: repo, settings, logging, databases, service skeletons, CI, docs | **working (tested) locally**; known issue: CI not yet run on GitHub |
| 1 | P2 L1: defect detector (PatchCore / EfficientAD + supervised baseline) + demo | **in progress**: code 1a–1g committed and tested; training 2 of 15 categories |
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
| 1c | CPU PatchCore, metrics, MLflow, train/evaluate/report scripts, queue | `903d949` | code working (tested); training in progress |
| 1d | Inspection API, heatmaps, registry, VLM interface, latency benchmark | `3ed8bfe`, `2ddd22e` | working (tested); latency not measured yet |
| 1e | Supervised ResNet-18 baseline | `32452e3` | code working (tested); not run on MVTec |
| 1f | Dashboard + image + smoke test | `b5244dc`, `a5aca26` | working: Docker smoke passed with the official bottle model |
| 1g | EfficientAD notebook, run bundles, import | `34b7d3e` | code working (tested); not run (needs your Colab GPU + download approval) |
| 1h | Docs, results, tag `v0.1-step1` | — | open |

Step 1 gates (blueprint Rev 2 §5):

- [ ] (1) AUROC for all 15 categories — 2 done (bottle, grid)
- [ ] (2) same-seed rerun within ±0.005 — not run (follow-up job)
- [ ] (3) deviations vs anomalib — computed for 2; grid +0.028 needs a note (KNOWN_ISSUES #11)
- [x] (4) API 200 / 415 / 401 — tests + Docker smoke
- [ ] (5) dashboard against Docker with the **deploy** models — smoke passed with official bottle; deploy models not trained yet
- [ ] (6) EfficientAD bundle imported or deferred — to defer in KNOWN_ISSUES if no GPU quota

## Hand-off notes

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
